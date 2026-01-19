from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


def _read_gray_u8(path: Path) -> np.ndarray:
    """
    Read grayscale PNG to uint8 HxW.
    Prefers cv2. Fallback PIL.
    """
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        return img.astype(np.uint8)
    if Image is not None:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
    raise RuntimeError("Need opencv-python or pillow to read png.")


def _u8_to_float(img_u8: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "0_1":
        return (img_u8.astype(np.float32) / 255.0)
    if normalize == "-1_1":
        return (img_u8.astype(np.float32) / 127.5) - 1.0
    # none
    return img_u8.astype(np.float32)


def _binarize(img: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (img >= thr).astype(np.float32)


def _load_split_ids_from_txt(splits_dir: Path, split: str) -> set:
    p = splits_dir / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    ids = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


@dataclass
class TrainPackRow:
    sample_key: str
    dataset: str
    mode: str
    mask_1280_path: str
    ld_1280_aligned_path: str
    thr_random_dir: str
    thr_random_count: int
    thr_fixed_map_json: str
    split: str


class TrainPackManifestDataset(Dataset):
    """
    DLP TrainPack dataset (inverse by default):
      input:  thr image (random or fixed or ld_1280_aligned)
      target: mask_1280 (binary or grayscale)

    Returns:
      x: FloatTensor [C,H,W]
      y: FloatTensor [C,H,W]
      meta: dict (strings/ints) for logging/debug
    """
    def __init__(
        self,
        trainpack_root: Path,
        manifest_csv: Path,
        splits_dir: Path,
        split_source: str,
        split: str,
        modes: List[str],
        task_cfg: Dict,
        image_cfg: Dict,
        filters_cfg: Optional[Dict] = None,
        seed: int = 1234,
        shuffle_thr: bool = True,
    ):
        self.root = Path(trainpack_root)
        self.manifest_csv = Path(manifest_csv)
        self.splits_dir = Path(splits_dir)
        self.split_source = split_source
        self.split = split
        self.modes = set(modes)
        self.task_cfg = task_cfg
        self.image_cfg = image_cfg
        self.filters_cfg = filters_cfg or {}
        self.seed = int(seed)
        self.shuffle_thr = bool(shuffle_thr)

        # inverse settings
        inv = (task_cfg or {}).get("inverse", {})
        self.input_source = inv.get("input_source", "thr_random")  # thr_random|thr_fixed|ld_1280_aligned
        self.thr_random_policy = inv.get("thr_random_policy", "expand_all")  # expand_all|random_one
        self.thr_fixed_values = inv.get("thr_fixed_values", [])
        self.thr_stack = bool(inv.get("thr_stack", False))
        self.target_key = inv.get("target_key", "mask_1280")

        # image settings
        self.normalize = image_cfg.get("normalize", "0_1")
        self.binarize_target = bool(image_cfg.get("binarize_target", True))

        # load manifest
        rows = self._read_manifest_rows()

        # split filter
        rows = self._apply_split(rows)

        # mode filter
        rows = [r for r in rows if r.mode in self.modes]

        # optional dataset allowlist
        allow = self.filters_cfg.get("dataset_allow", [])
        if allow:
            allow = set(allow)
            rows = [r for r in rows if r.dataset in allow]

        # build index (may expand by thr_random images)
        self.items: List[Dict] = self._build_items(rows)
        if len(self.items) == 0:
            raise RuntimeError("No samples after filtering. Check split/modes/filters.")

        # per-epoch rng base
        self._rng = random.Random(self.seed)

    def _read_manifest_rows(self) -> List[TrainPackRow]:
        out: List[TrainPackRow] = []
        with open(self.manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(
                    TrainPackRow(
                        sample_key=row["sample_key"],
                        dataset=row.get("dataset", ""),
                        mode=row.get("mode", ""),
                        mask_1280_path=row.get("mask_1280_path", ""),
                        ld_1280_aligned_path=row.get("ld_1280_aligned_path", ""),
                        thr_random_dir=row.get("thr_random_dir", ""),
                        thr_random_count=int(row.get("thr_random_count", "0") or "0"),
                        thr_fixed_map_json=row.get("thr_fixed_map_json", "{}"),
                        split=row.get("split", ""),
                    )
                )
        return out

    def _apply_split(self, rows: List[TrainPackRow]) -> List[TrainPackRow]:
        if self.split_source == "txt":
            ids = _load_split_ids_from_txt(self.splits_dir, self.split)
            return [r for r in rows if r.sample_key in ids]
        # default: manifest split column
        return [r for r in rows if r.split == self.split]

    def _list_thr_random_files(self, thr_dir_rel: str, sample_key: str) -> List[Path]:
        if not thr_dir_rel:
           return []
        d = self.root / thr_dir_rel
        if not d.exists():
            return []
        # naming: {sample_key}__Rxx__thr.png (extract script)
        files = sorted(d.glob(f"{sample_key}__R*.png"))
        return files

    def _pick_thr_fixed_file(self, thr_fixed_map_json: str, sample_key: str) -> Optional[Path]:
        try:
            m = json.loads(thr_fixed_map_json or "{}")
        except Exception:
            m = {}
        if not m:
            return None
        # if user specifies values, try them in order
        if self.thr_fixed_values:
            for t in self.thr_fixed_values:
                key = str(int(t))
                if key in m and m[key]:
                    return self.root / m[key]
        # otherwise pick any one (stable order)
        k = sorted(m.keys(), key=lambda x: int(x) if str(x).isdigit() else 999999)[0]
        return self.root / m[k]

    def _build_items(self, rows: List[TrainPackRow]) -> List[Dict]:
        items: List[Dict] = []
        for r in rows:
            target = self.root / r.mask_1280_path
            if not target.exists():
                continue

            if self.input_source == "ld_1280_aligned":
                x = self.root / r.ld_1280_aligned_path
                if x.exists():
                    items.append({"row": r, "x_path": x, "y_path": target, "thr_id": ""})
                continue

            if self.input_source == "thr_fixed":
                x = self._pick_thr_fixed_file(r.thr_fixed_map_json, r.sample_key)
                if x is not None and x.exists():
                    items.append({"row": r, "x_path": x, "y_path": target, "thr_id": "fixed"})
                continue

            # default: thr_random
            thr_files = self._list_thr_random_files(r.thr_random_dir, r.sample_key)
            if not thr_files:
                continue
            if self.thr_random_policy == "random_one":
                # store the list; choose at __getitem__
                items.append({"row": r, "x_paths": thr_files, "y_path": target})
            else:
                # expand_all
                for p in thr_files:
                    items.append({"row": r, "x_path": p, "y_path": target, "thr_id": p.stem})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _get_x_path(self, item: Dict) -> Path:
        if "x_path" in item:
            return item["x_path"]
        # random_one
        paths: List[Path] = item["x_paths"]
        if self.shuffle_thr:
            # sample per call (deterministic-ish with seed + index)
            return paths[self._rng.randrange(0, len(paths))]
        return paths[0]

    def __getitem__(self, idx: int):
        item = self.items[idx]
        r: TrainPackRow = item["row"]
        x_path = self._get_x_path(item)
        y_path = item["y_path"]

        x_u8 = _read_gray_u8(x_path)
        y_u8 = _read_gray_u8(y_path)

        x = _u8_to_float(x_u8, self.normalize)
        y = _u8_to_float(y_u8, self.normalize)
        if self.binarize_target:
            y = _binarize(y, 0.5)
        # to CHW
        x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]
        y = torch.from_numpy(y).unsqueeze(0)  # [1,H,W]

        meta = {
            "sample_key": r.sample_key,
            "dataset": r.dataset,
            "mode": r.mode,
            "x_path": str(x_path),
            "y_path": str(y_path),
            "input_source": self.input_source,
        }
        return x, y, meta


def create_data_loaders(
    args,
    split: str,
    shuffle: bool,
    is_train: bool,
):
    """
    DLP TrainPack dataloader factory.
    - split: "train"/"val"/"test"
    """
    task_cfg = getattr(args, "task", {})
    data_cfg = getattr(args, "data", {})
    image_cfg = getattr(args, "image", {})
    filters_cfg = getattr(args, "filters", {})

    trainpack_root = Path(getattr(args, "data_trainpack_root"))
    manifest_csv = Path(getattr(args, "data_manifest_csv"))
    splits_dir = Path(getattr(args, "data_splits_dir"))
    split_source = getattr(args, "data_split_source", "manifest")
    modes = list(getattr(args, "data_modes", ["binary", "gray"]))
    ds = TrainPackManifestDataset(
        trainpack_root=trainpack_root,
        manifest_csv=manifest_csv,
        splits_dir=splits_dir,
        split_source=split_source,
        split=split,
        modes=modes,
        task_cfg=task_cfg,
        image_cfg=image_cfg,
        filters_cfg=filters_cfg,
       seed=int(getattr(args, "seed", 1234)),
        shuffle_thr=bool(is_train),
    )

    batch_size = int(getattr(args, "batch_size", 4)) if is_train else int(getattr(args, "val_batch_size", 4))
    num_workers = int(getattr(args, "num_workers", 0))

    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(is_train),
    )