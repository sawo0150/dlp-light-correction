# utils/common/utils.py

from skimage.metrics import structural_similarity
from torchmetrics.functional import structural_similarity_index_measure as ssim_f

import h5py
import numpy as np
import torch
import random, math
import os

def seed_fix(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)
    random.seed(n)

# -----------------------------------------------------------------------------
# ✅ Metric Helper Functions (Moved from train_part.py)
# -----------------------------------------------------------------------------
def compute_segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    Inverse Task용: Dice & IoU 계산
    logits: [B,1,H,W] (unbounded)
    target: [B,1,H,W] in {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

def compute_regression_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Forward Task용: MAE, RMSE, PSNR 계산
    logits -> pred01 = sigmoid(logits) 가정 (DLP Forward Output은 0~1 normalized)
    """
    pred = torch.sigmoid(logits)
    diff = pred - target
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    rmse = float(math.sqrt(mse + eps))
    # PSNR (0~1 scale): 20 * log10(MAX / RMSE) -> MAX=1.0
    psnr = float(20.0 * math.log10(1.0 / (rmse + eps)))
    return mae, rmse, psnr

