# utils/common/utils.py

from skimage.metrics import structural_similarity
from torchmetrics.functional import structural_similarity_index_measure as ssim_f

import h5py
import numpy as np
import torch
import random, math
import os
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

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

def _get_pipe_cfg(args) -> Dict[str, Any]:
    task_cfg = getattr(args, "task", {}) or {}
    inv_cfg  = (task_cfg.get("inverse") or {}) if isinstance(task_cfg, dict) else {}
    return dict(inv_cfg.get("pipeline", {}) or {})


def _apply_stage_input_policy(policy: str, *, gt: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    policy:
      - "gt": use gt directly
      - "pred_detach": use sigmoid(pred_logits).detach()
    """
    p = str(policy).lower().strip()
    if p == "pred_detach":
        return torch.sigmoid(pred_logits).detach()
    return gt


def _gaussian_kernel2d(ks: int, sigma: float, device, dtype) -> torch.Tensor:
    ks = int(ks)
    if ks % 2 == 0:
        ks += 1
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma) + 1e-12))
    k = k / (k.sum() + 1e-12)
    return k


def _blur_01(x01: torch.Tensor, *, ks: int, sigma: float, n_iter: int = 1) -> torch.Tensor:
    if ks <= 1 or sigma <= 0:
        return x01
    k = _gaussian_kernel2d(ks, sigma, device=x01.device, dtype=x01.dtype)
    w = k[None, None, :, :]
    pad = ks // 2
    out = x01
    for _ in range(max(1, int(n_iter))):
        out = F.conv2d(out, w, bias=None, stride=1, padding=pad)
    return out


def _make_doc_pseudo(ld_gt_01: torch.Tensor, doc_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    ld_gt_01: [B,1,H,W] in [0,1]
    doc_pseudo: blur(ld_gt_01) (baseline)
    """
    g = dict((doc_cfg or {}).get("gauss", {}) or {})
    if bool(g.get("enable", True)):
        ks = int(g.get("kernel_size", 11))
        sigma = float(g.get("sigma", 10))
        n_iter = int(g.get("n_iter", 1))
        ld_gt_01 = _blur_01(ld_gt_01, ks=ks, sigma=sigma, n_iter=n_iter)
    return torch.clamp(ld_gt_01, 0.0, 1.0)

class _StageSequentialWrapper(nn.Module):
    """
    Pipeline(stage dict) -> single nn.Module wrapper for inference.
    - For benchmark/eval: always run with predicted outputs (no teacher forcing).
    - Assumes stage I/O is sequential: out(stage_i) -> in(stage_{i+1})
      (thr2ld -> ld2doc -> doc2mask, or thr2ld -> ld2mask, etc.)
    """
    def __init__(self, models: Dict[str, nn.Module], stages: List[str]):
        super().__init__()
        # keep deterministic order
        self.stages = list(stages)
        # ModuleDict for proper .eval(), .to(), state_dict recursion
        self.models = nn.ModuleDict({s: models[s] for s in self.stages})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, s in enumerate(self.stages):
            h = self.models[s](h)

            # ✅ stage between sigmoid: thr2ld(또는 ld2doc)의 logits -> 0~1로 변환
            if i < len(self.stages) - 1:
                if s in ("thr2ld", "ld2doc"):   # 필요시 stage명 확장
                    h = torch.sigmoid(h)
        return h