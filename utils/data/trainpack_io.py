# utils/data/trainpack_io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


def read_gray_u8(path: Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        return img.astype(np.uint8)
    if Image is not None:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
    raise RuntimeError("Need opencv-python or pillow to read png.")


def resize_u8(img_u8: np.ndarray, out_size: int, is_mask: bool) -> np.ndarray:
    h, w = img_u8.shape[:2]
    if h == out_size and w == out_size:
        return img_u8
    if cv2 is not None:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        return cv2.resize(img_u8, (out_size, out_size), interpolation=interp).astype(np.uint8)
    if Image is not None:
        pil = Image.fromarray(img_u8)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        pil = pil.resize((out_size, out_size), resample=resample)
        return np.array(pil, dtype=np.uint8)
    raise RuntimeError("Need opencv-python or pillow to resize images.")


def u8_to_float(img_u8: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "0_1":
        return img_u8.astype(np.float32) / 255.0
    if normalize == "-1_1":
        return (img_u8.astype(np.float32) / 127.5) - 1.0
    return img_u8.astype(np.float32)


def binarize01(img01: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (img01 >= thr).astype(np.float32)


def to_chw_torch(img01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img01).unsqueeze(0)  # [1,H,W]


def load_thr_file_index(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_thr_fixed_file(root: Path, thr_fixed_map_json: str, thr_fixed_values: List[int]) -> Optional[Path]:
    try:
        m = json.loads(thr_fixed_map_json or "{}")
    except Exception:
        m = {}
    if not m:
        return None
    if thr_fixed_values:
        for t in thr_fixed_values:
            key = str(int(t))
            if key in m and m[key]:
                return root / m[key]
    # stable fallback
    k = sorted(m.keys(), key=lambda x: int(x) if str(x).isdigit() else 999999)[0]
    return root / m[k]
