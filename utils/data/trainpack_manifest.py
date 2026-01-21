# utils/data/trainpack_manifest.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TrainPackRow:
    # core
    sample_key: str
    dataset: str
    mode: str
    split: str

    # paths (root-relative)
    mask_128_path: str
    mask_160_path: str
    mask_1280_path: str
    ld_1280_aligned_path: str
    thr_random_dir: str
    thr_random_count: int
    thr_fixed_map_json: str

    # availability flags (manifest.csv에 있는 컬럼)
    has_fwd: int
    has_inv_random: int
    has_inv_fixed: int


class TrainPackManifest:
    """
    manifest.csv reader + common filtering.
    """
    def __init__(self, manifest_csv: Path):
        self.manifest_csv = Path(manifest_csv)
        self.rows: List[TrainPackRow] = self._read()

    def _read(self) -> List[TrainPackRow]:
        out: List[TrainPackRow] = []
        with open(self.manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(
                    TrainPackRow(
                        sample_key=row["sample_key"],
                        dataset=row.get("dataset", ""),
                        mode=row.get("mode", ""),
                        split=row.get("split", ""),
                        mask_128_path=row.get("mask_128_path", ""),
                        mask_160_path=row.get("mask_160_path", ""),
                        mask_1280_path=row.get("mask_1280_path", ""),
                        ld_1280_aligned_path=row.get("ld_1280_aligned_path", ""),
                        thr_random_dir=row.get("thr_random_dir", ""),
                        thr_random_count=int(row.get("thr_random_count", "0") or "0"),
                        thr_fixed_map_json=row.get("thr_fixed_map_json", "{}"),
                        has_fwd=int(row.get("has_fwd", "0") or "0"),
                        has_inv_random=int(row.get("has_inv_random", "0") or "0"),
                        has_inv_fixed=int(row.get("has_inv_fixed", "0") or "0"),
                    )
                )
        return out

    def filter(
        self,
        *,
        split: str,
        split_source: str = "manifest",   # 현재는 manifest만 지원(추후 txt도 가능)
        modes: Optional[Sequence[str]] = None,
        dataset_allow: Optional[Sequence[str]] = None,
        require_flags: Optional[Dict[str, int]] = None,
    ) -> List[TrainPackRow]:
        rows = self.rows

        # split
        if split_source == "manifest":
            rows = [r for r in rows if r.split == split]
        else:
            raise NotImplementedError("split_source=txt는 이후 확장 포인트로 남김")

        # modes
        if modes:
            mset = set(modes)
            rows = [r for r in rows if r.mode in mset]

        # dataset allowlist
        if dataset_allow:
            aset = set(dataset_allow)
            rows = [r for r in rows if r.dataset in aset]

        # require flags (has_fwd/has_inv_random/has_inv_fixed 등)
        if require_flags:
            for k, v in require_flags.items():
                rows = [r for r in rows if int(getattr(r, k)) == int(v)]

        return rows
