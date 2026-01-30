# utils/data/dataloader_factory.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import json

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils.data.trainpack_manifest import TrainPackManifest
from utils.data.trainpack_datasets import (
    CommonImageConfig,
    InverseThr2MaskDataset,
    InverseThr2LDMaskDataset,
    ForwardMask2LDDataset,
)


def _dataset_weight(name: str, weights: Dict[str, float]) -> float:
    """
    weights key can be:
      - exact dataset name: "B1_Shape_Cutout"
      - prefix group: "B1" -> matches "B1_*"
    fallback: 1.0
    """
    if not weights:
        return 1.0
    if name in weights:
        return float(weights[name])
    # prefix match: "B1" matches "B1_..."
    for k, v in weights.items():
        k = str(k)
        if name.startswith(k + "_"):
            return float(v)
    return 1.0

def _apply_domain_policy(
    rows,
    *,
    domain_weights: Dict[str, float],
    max_n: int | None,
    seed: int,
    with_replacement: bool,
):
    """
    1) drop datasets with weight<=0
    2) if max_n is None: keep all remaining
    3) else: allocate quota per dataset ~ weight ratio, then sample within each dataset
    """
    if not rows:
        return rows

    # group by dataset
    buckets = defaultdict(list)
    for r in rows:
        w = _dataset_weight(r.dataset, domain_weights)
        if w > 0:
            buckets[r.dataset].append(r)

    if not buckets:
        return []

    if max_n is None or max_n <= 0:
        # keep all (but already dropped weight<=0)
        out = []
        for ds in sorted(buckets.keys()):
            out.extend(buckets[ds])
        return out

    rng = np.random.default_rng(int(seed))

    # dataset-level weights
    ds_names = sorted(buckets.keys())
    ws = np.array([max(0.0, _dataset_weight(ds, domain_weights)) for ds in ds_names], dtype=np.float64)
    if ws.sum() <= 0:
        return []
    ws = ws / ws.sum()

    # quota allocation
    quotas = np.floor(ws * max_n).astype(int)
    # distribute remainder
    rem = int(max_n - quotas.sum())
    if rem > 0:
        order = np.argsort(-ws)  # larger weight first
        for i in range(rem):
            quotas[order[i % len(order)]] += 1

    out = []
    for ds, q in zip(ds_names, quotas.tolist()):
        if q <= 0:
            continue
        pool = buckets[ds]
        if (not with_replacement) and q >= len(pool):
            out.extend(pool)  # cap
            continue
        idx = rng.choice(len(pool), size=q, replace=bool(with_replacement))
        out.extend([pool[i] for i in idx.tolist()])
    return out

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

    # ✅ NEW: debug options
    dbg_cfg = dict(sub_data.get("debug", {}) or {})
    dbg_enable = bool(dbg_cfg.get("enable", False))
    dbg_examples = int(dbg_cfg.get("examples", 3) or 3)

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
        elif input_source == "mask_target":
            # ✅ target mask만 있으면 되므로 require_flags 강제 없음
            pass

        # ✅ NEW: pipeline(chain) mode면 LD GT가 필요하므로 has_fwd도 강제
        inv_cfg = (task_cfg.get("inverse") or {})
        pipe_cfg = (inv_cfg.get("pipeline") or {})
        pipe_enable = bool(pipe_cfg.get("enable", False))
        if pipe_enable:
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

    if dbg_enable:
        # dataset/mode 분포 출력 (filter 직후)
        cnt_ds = defaultdict(int)
        cnt_mode = defaultdict(int)
        for r in rows:
            cnt_ds[r.dataset] += 1
            cnt_mode[r.mode] += 1
        top_ds = sorted(cnt_ds.items(), key=lambda x: -x[1])[:10]
        print(f"[DBG] split={split} after manifest.filter: rows={len(rows)}")
        print(f"[DBG]   mode_counts={dict(cnt_mode)}")
        print(f"[DBG]   top_dataset_counts={top_ds}")

    # ✅ NEW: domain mix + max samples
    domain_weights = dict(sub_data.get("domain_weights", {}) or {})
    domain_cfg = dict(sub_data.get("domain", {}) or {})
    domain_enable = bool(domain_cfg.get("enable", True))
    domain_weights = dict(domain_cfg.get("weights", sub_data.get("domain_weights", {})) or {})

    max_samples_cfg = dict(sub_data.get("max_samples", {}) or {})
    max_n = int(max_samples_cfg.get(split, 0) or 0)
    max_n = None if max_n <= 0 else max_n
    with_replacement = bool(sub_data.get("with_replacement", False))

    # ✅ NEW: max_samples 기준 단위 선택 (rows vs items)
    # 기본은 rows (기존 동작 유지). items로 설정하면 Dataset len(iter) 기준으로 제한됨.
    max_unit = str(sub_data.get("max_samples_unit", "rows")).lower().strip()
    if max_unit not in {"rows", "items"}:
        max_unit = "rows"

    # optional epoch-dependent resampling (train only)
    resample_each_epoch = bool(sub_data.get("resample_each_epoch", False))
    epoch = int(getattr(args, "_epoch", 0))
    split_off = {"train": 0, "val": 1, "test": 2}.get(str(split), 9)
    base_seed = int(getattr(args, "seed", 1234))
    eff_seed = base_seed + split_off + (epoch * 10007 if (is_train and resample_each_epoch) else 0)

    rows_before = len(rows)
    if domain_enable:
        rows = _apply_domain_policy(
            rows,
            domain_weights=domain_weights,
            max_n=max_n,
            seed=eff_seed,
            with_replacement=with_replacement,
        )
        if rows_before != len(rows):
            print(f"[DomainPolicy] split={split} rows {rows_before} -> {len(rows)} | enable={domain_enable} | weights={domain_weights} | max_n={max_n} | repl={with_replacement}")
        elif dbg_enable:
            print(f"[DBG] split={split} domain_policy no change: rows={len(rows)} | enable={domain_enable} | weights={domain_weights} | max_n={max_n} | repl={with_replacement}")
    else:
        # domain off: weights는 무시하고 전체 rows 유지
        # (단, max_samples_unit='items'면 아래에서 items 제한은 여전히 가능)
        if dbg_enable:
            print(f"[DBG] split={split} domain policy DISABLED -> keep all rows={len(rows)} (weights ignored)")
        # 만약 'rows' 단위로도 max_n을 걸고 싶으면, 여기서 uniform 샘플링을 하면 됨(선택)
        # 현재는: rows는 그대로 두고, 필요시 아래 items cap으로 제한하도록 둠.
    
    if dbg_enable:
        # domain policy 이후 분포 출력
        cnt_ds2 = defaultdict(int)
        cnt_mode2 = defaultdict(int)
        for r in rows:
            cnt_ds2[r.dataset] += 1
            cnt_mode2[r.mode] += 1
        top_ds2 = sorted(cnt_ds2.items(), key=lambda x: -x[1])[:10]
        print(f"[DBG] split={split} after domain_policy: rows={len(rows)}")
        print(f"[DBG]   mode_counts={dict(cnt_mode2)}")
        print(f"[DBG]   top_dataset_counts={top_ds2}")

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

    # ✅ DEBUG: rows -> items 확장 원인(특히 thr_random expand_all) 확인
    if dbg_enable and task_name.startswith("inverse"):
        inv = (task_cfg.get("inverse") or {})
        input_source = str(inv.get("input_source", "thr_random"))
        thr_pol = str(inv.get("thr_random_policy", "expand_all"))
        print(f"[DBG] inverse input_source={input_source} thr_random_policy={thr_pol}")
        print(f"[DBG] built dataset: rows={len(rows)} -> items(len(ds))={len(ds)}")

        # thr_random + expand_all인 경우, thr_index로 평균 확장량 추정
        if input_source == "thr_random" and thr_pol == "expand_all" and thr_index_path.exists():
            try:
                with open(thr_index_path, "r", encoding="utf-8") as f:
                    thr_index = json.load(f)
                counts = []
                for r in rows:
                    counts.append(len(thr_index.get(r.sample_key, [])))
                if counts:
                    avg = float(np.mean(counts))
                    mn = int(np.min(counts))
                    mx = int(np.max(counts))
                    print(f"[DBG] thr_random expansion stats over selected rows: "
                          f"avg_thr_files_per_row={avg:.3f} min={mn} max={mx}")
                    # 예시 몇 개
                    ex = []
                    for r in rows[:dbg_examples]:
                        ex.append((r.sample_key, r.dataset, r.mode, len(thr_index.get(r.sample_key, []))))
                    print(f"[DBG] examples(sample_key,dataset,mode,thr_count)={ex}")
            except Exception as e:
                print(f"[DBG] failed to load thr_index for debug: {e}")

    # ✅ NEW: max_samples_unit == "items"이면, Dataset 길이를 max_n으로 제한 (iter 제한)
    if (max_unit == "items") and (max_n is not None) and (max_n > 0):
        if len(ds) > max_n:
            rng = np.random.default_rng(int(eff_seed))
            if with_replacement:
                idx = rng.choice(len(ds), size=max_n, replace=True).tolist()
            else:
                idx = rng.choice(len(ds), size=max_n, replace=False).tolist()
            ds = Subset(ds, idx)
            print(f"[DomainPolicy-Items] split={split} items capped: {len(idx)} (max_n={max_n}, repl={with_replacement})")

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
