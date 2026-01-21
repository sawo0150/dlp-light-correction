from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None


class BenchmarkDataset(Dataset):
    """
    Loads benchmark images from:
      root/
        category_1/
          img1.png
        category_2/
          ...
    """
    def __init__(self, root: str | Path, normalize: str = "0_1"):
        self.root = Path(root)
        self.normalize = normalize
        self.items: List[Dict] = []
        self._scan()

    def _scan(self):
        if not self.root.exists():
            return

        # Sort categories for consistent order
        categories = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        
        for cat in categories:
            cat_dir = self.root / cat
            # Filter image files
            files = sorted([f for f in cat_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
            
            for f in files:
                self.items.append({
                    "path": f,
                    "name": f.stem,
                    "category": cat
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = str(item["path"])
        
        if cv2 is None:
             raise ImportError("opencv-python is required for BenchmarkDataset")

        # Read grayscale
        img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
             raise FileNotFoundError(f"Failed to read {path}")

        # Normalize
        if self.normalize == "0_1":
            img_f = img_u8.astype(np.float32) / 255.0
        elif self.normalize == "-1_1":
             img_f = (img_u8.astype(np.float32) / 127.5) - 1.0
        else:
             img_f = img_u8.astype(np.float32)

        # To Tensor [1, H, W]
        img_t = torch.from_numpy(img_f).unsqueeze(0)

        return img_t, item["name"], item["category"]

    def get_categories(self) -> List[str]:
        return sorted(list(set(x["category"] for x in self.items)))

    def get_subset_by_category(self, category: str) -> "BenchmarkDataset":
        """Returns a new dataset instance for a specific category."""
        subset = BenchmarkDataset(self.root, self.normalize)
        subset.items = [x for x in self.items if x["category"] == category]
        return subset