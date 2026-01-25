# utils/data/trainpack_datasets.py
from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from torch.utils.data import Dataset

from utils.data.trainpack_manifest import TrainPackRow
from utils.data.trainpack_io import (
    read_gray_u8,
    resize_u8,
    u8_to_float,
    binarize01,
    to_chw_torch,
    load_thr_file_index,
    pick_thr_fixed_file,
)


@dataclass
class CommonImageConfig:
    normalize: str = "0_1"
    binarize_target: bool = True
    size: int = 1280


class InverseThr2MaskDataset(Dataset):
    """
    inverse_1:
      x: thr_random | thr_fixed | ld_1280_aligned | mask_target
      y: mask_{128|160|1280}
    """
    def __init__(
        self,
        *,
        root: Path,
        rows: List[TrainPackRow],
        task_cfg: Dict,
        image_cfg: CommonImageConfig,
        thr_index_path: Optional[Path] = None,
        is_train: bool = True,
        seed: int = 1234,
    ):
        self.root = Path(root)
        self.rows = rows
        self.task_cfg = task_cfg or {}
        self.image_cfg = image_cfg
        self.is_train = bool(is_train)
        self.rng = random.Random(int(seed))

        inv = (self.task_cfg.get("inverse") or {})
        self.input_source = inv.get("input_source", "thr_random")              # thr_random|thr_fixed|ld_1280_aligned|mask_target
        self.thr_random_policy = inv.get("thr_random_policy", "expand_all")   # expand_all|random_one
        self.thr_fixed_values = list(inv.get("thr_fixed_values", []) or [])
        self.target_key = (inv.get("target_key", "mask_1280") or "mask_1280")
        pre = (inv.get("preprocess") or {})
        self.out_size = int(pre.get("out_size", self.image_cfg.size))

        # thr index (for thr_random)
        self.thr_index: Dict[str, List[str]] = {}
        if self.input_source == "thr_random" and thr_index_path is not None and Path(thr_index_path).exists():
            self.thr_index = load_thr_file_index(Path(thr_index_path))

        self.items: List[Dict] = self._build_items()

        # ✅ DEBUG: items 구성 통계 출력(옵션)
        inv = (self.task_cfg.get("inverse") or {})
        sub_data = (inv.get("data") or {})
        dbg_cfg = dict(sub_data.get("debug", {}) or {})
        dbg_enable = bool(dbg_cfg.get("enable", False))
        if dbg_enable:
            print(f"[DBG][InverseDataset] input_source={self.input_source} thr_random_policy={self.thr_random_policy} "
                  f"target_key={self.target_key} out_size={self.out_size}")
            print(f"[DBG][InverseDataset] rows_in={len(self.rows)} items_out={len(self.items)}")
            # item 타입별 카운트
            kind = defaultdict(int)
            for it in self.items:
                if "x_paths" in it:
                    kind["random_one"] += 1
                elif "x_path" in it:
                    kind["single_path"] += 1
                else:
                    kind["unknown"] += 1
            print(f"[DBG][InverseDataset] item_kinds={dict(kind)}")

        if len(self.items) == 0:
            raise RuntimeError("No samples after building inverse dataset items.")

    def _resolve_target_path(self, r: TrainPackRow) -> Path:
        if self.target_key == "mask_128":
            return self.root / r.mask_128_path
        if self.target_key == "mask_160":
            return self.root / r.mask_160_path
        return self.root / r.mask_1280_path

    def _build_items(self) -> List[Dict]:
        items: List[Dict] = []
        for r in self.rows:
            y_path = self._resolve_target_path(r)

            # ✅ NEW: mask_target (use target mask as input x)
            if self.input_source == "mask_target":
                x_path = y_path
                items.append({"row": r, "x_path": x_path, "y_path": y_path})
                continue

            if self.input_source == "ld_1280_aligned":
                x_path = self.root / r.ld_1280_aligned_path
                items.append({"row": r, "x_path": x_path, "y_path": y_path})
                continue

            if self.input_source == "thr_fixed":
                x_path = pick_thr_fixed_file(self.root, r.thr_fixed_map_json, self.thr_fixed_values)
                if x_path is not None:
                    items.append({"row": r, "x_path": x_path, "y_path": y_path})
                continue

            # thr_random
            rel_paths = self.thr_index.get(r.sample_key, [])
            if not rel_paths:
                continue
            thr_files = [self.root / p for p in rel_paths]

            if self.thr_random_policy == "random_one":
                items.append({"row": r, "x_paths": thr_files, "y_path": y_path})
            else:
                for p in thr_files:
                    items.append({"row": r, "x_path": p, "y_path": y_path})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _get_x_path(self, item: Dict) -> Path:
        if "x_path" in item:
            return item["x_path"]
        # random_one
        paths: List[Path] = item["x_paths"]
        if self.is_train:
            return paths[self.rng.randrange(0, len(paths))]
        return paths[0]

    def __getitem__(self, idx: int):
        item = self.items[idx]
        r: TrainPackRow = item["row"]
        x_path = self._get_x_path(item)
        y_path = item["y_path"]

        x_u8 = read_gray_u8(x_path)
        y_u8 = read_gray_u8(y_path)

        # resize
        x_u8 = resize_u8(x_u8, self.out_size, is_mask=False)
        y_u8 = resize_u8(y_u8, self.out_size, is_mask=True)

        x = u8_to_float(x_u8, self.image_cfg.normalize)
        y = u8_to_float(y_u8, self.image_cfg.normalize)
        if self.image_cfg.binarize_target:
            y = binarize01(y, 0.5)

        meta = {
            "sample_key": r.sample_key,
            "dataset": r.dataset,
            "mode": r.mode,
            "x_path": str(x_path),
            "y_path": str(y_path),
            "input_source": self.input_source,
            "target_key": self.target_key,
            "out_size": self.out_size,
        }
        return to_chw_torch(x), to_chw_torch(y), meta


class ForwardMask2LDDataset(Dataset):
    """
    forward_1 (향후):
      x: mask_{128|160|1280} (보통 160 추천)
      y: ld_1280_aligned (출력은 항상 1280(혹은 out_size)로 학습하게 만들 수 있음)
    """
    def __init__(
        self,
        *,
        root: Path,
        rows: List[TrainPackRow],
        task_cfg: Dict,
        image_cfg: CommonImageConfig,
        is_train: bool = True,
        seed: int = 1234,
    ):
        self.root = Path(root)
        self.rows = rows
        self.task_cfg = task_cfg or {}
        self.image_cfg = image_cfg
        self.is_train = bool(is_train)
        self.rng = random.Random(int(seed))

        fwd = (self.task_cfg.get("forward") or {})
        self.input_key = (fwd.get("input_key", "mask_160") or "mask_160")  # mask_128|mask_160|mask_1280
        self.binarize_input = bool(fwd.get("binarize_input", False))
        pre = (fwd.get("preprocess") or {})
        self.out_size = int(pre.get("out_size", self.image_cfg.size))      # LD output size

        self.items = [{"row": r} for r in self.rows]
        if len(self.items) == 0:
            raise RuntimeError("No samples after building forward dataset items.")

    def _resolve_mask_path(self, r: TrainPackRow) -> Path:
        if self.input_key == "mask_128":
            return self.root / r.mask_128_path
        if self.input_key == "mask_1280":
            return self.root / r.mask_1280_path
        return self.root / r.mask_160_path

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        r: TrainPackRow = self.items[idx]["row"]

        x_path = self._resolve_mask_path(r)
        y_path = self.root / r.ld_1280_aligned_path

        x_u8 = read_gray_u8(x_path)
        y_u8 = read_gray_u8(y_path)

        # forward에서는 mask는 nearest, ld는 area로 리사이즈
        x_u8 = resize_u8(x_u8, self.out_size, is_mask=True)
        y_u8 = resize_u8(y_u8, self.out_size, is_mask=False)

        x = u8_to_float(x_u8, self.image_cfg.normalize)
        y = u8_to_float(y_u8, self.image_cfg.normalize)

        # ✅ forward input policy: optional binarize to match config
        if self.binarize_input:
            x = binarize01(x, 0.5)

        meta = {
            "sample_key": r.sample_key,
            "dataset": r.dataset,
            "mode": r.mode,
            "x_path": str(x_path),
            "y_path": str(y_path),
            "input_key": self.input_key,
            "binarize_input": self.binarize_input,
            "out_size": self.out_size,
        }
        return to_chw_torch(x), to_chw_torch(y), meta
