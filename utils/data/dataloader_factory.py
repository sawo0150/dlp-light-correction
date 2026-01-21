# utils/data/dataloader_factory.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from utils.data.trainpack_manifest import TrainPackManifest
from utils.data.trainpack_datasets import (
    CommonImageConfig,
    InverseThr2MaskDataset,
    ForwardMask2LDDataset,
)


def _get_task_name(task_cfg: Dict) -> str:
    return str((task_cfg or {}).get("name", "")).strip()


def _common_image_cfg(args) -> CommonImageConfig:
    image_cfg = getattr(args, "image", {}) or {}
    return CommonImageConfig(
        normalize=str(image_cfg.get("normalize", "0_1")),
        binarize_target=bool(image_cfg.get("binarize_target", True)),
        size=int(image_cfg.get("size", 1280)),
    )


def create_data_loaders(args, split: str, shuffle: bool, is_train: bool) -> DataLoader:
    task_cfg = getattr(args, "task", {}) or {}
    data_cfg = getattr(args, "data", {}) or {}
    filters_cfg = getattr(args, "filters", {}) or {}

    root = Path(getattr(args, "data_trainpack_root"))
    manifest_csv = Path(getattr(args, "data_manifest_csv"))

    # file index path: root/thr_file_index.json (현재 네 코드랑 동일한 전제)
    thr_index_path = root / "thr_file_index.json"

    # task별 데이터 정책 읽기
    task_name = _get_task_name(task_cfg)

    # split/mode/allow는 task 하위 data를 최우선
    # (inverse/forward 공통적으로 "data" 키를 갖도록 설계)
    if task_name.startswith("inverse"):
        sub = (task_cfg.get("inverse") or {})
    elif task_name.startswith("forward"):
        sub = (task_cfg.get("forward") or {})
    else:
        sub = {}

    sub_data = (sub.get("data") or {})

    split_source = str(sub_data.get("split_source", "manifest"))
    modes = list(sub_data.get("modes", filters_cfg.get("modes", ["binary", "gray"])))
    dataset_allow = list(sub_data.get("dataset_allow", filters_cfg.get("dataset_allow", [])))

    manifest = TrainPackManifest(manifest_csv)

    # require flags: task가 필요한 것만 선택적으로 강제
    require_flags = {}
    if task_name.startswith("inverse"):
        input_source = str((task_cfg.get("inverse") or {}).get("input_source", "thr_random"))
        if input_source == "thr_random":
            require_flags["has_inv_random"] = 1
        elif input_source == "thr_fixed":
            require_flags["has_inv_fixed"] = 1
        elif input_source == "ld_1280_aligned":
            require_flags["has_fwd"] = 1

    if task_name.startswith("forward"):
        require_flags["has_fwd"] = 1

    rows = manifest.filter(
        split=split,
        split_source=split_source,
        modes=modes,
        dataset_allow=dataset_allow if dataset_allow else None,
        require_flags=require_flags if require_flags else None,
    )

    image_cfg = _common_image_cfg(args)

    if task_name.startswith("inverse"):
        ds = InverseThr2MaskDataset(
            root=root,
            rows=rows,
            task_cfg=task_cfg,
            image_cfg=image_cfg,
            thr_index_path=thr_index_path,
            is_train=is_train,
            seed=int(getattr(args, "seed", 1234)),
        )
    elif task_name.startswith("forward"):
        ds = ForwardMask2LDDataset(
            root=root,
            rows=rows,
            task_cfg=task_cfg,
            image_cfg=image_cfg,
            is_train=is_train,
            seed=int(getattr(args, "seed", 1234)),
        )
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    batch_size = int(getattr(args, "batch_size", 4)) if is_train else int(getattr(args, "val_batch_size", 4))
    num_workers = int(getattr(args, "num_workers", 0))

    print(
        f"[DataLoader] task={task_name} split={split} modes={modes} "
        f"split_source={split_source} len={len(ds)} batch={batch_size} "
        f"require_flags={require_flags}"
    )

    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(is_train),
    )
