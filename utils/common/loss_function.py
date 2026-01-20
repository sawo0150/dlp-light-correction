# utils/common/loss_function.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W] -> [B, C, HW]
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(f"Expected 3D/4D tensor, got {tuple(x.shape)}")
    b, c, h, w = x.shape
    return x.view(b, c, h * w)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Expects:
      logits: [B,1,H,W] (raw) or probs if from_probs=True
      target: [B,1,H,W] in {0,1} (float ok)
    """
    def __init__(self, eps: float = 1e-6, from_probs: bool = False):
        super().__init__()
        self.eps = eps
        self.from_probs = from_probs

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        probs = logits if self.from_probs else torch.sigmoid(logits)
        probs_f = _flatten_hw(probs)
        targ_f  = _flatten_hw(target)

        inter = (probs_f * targ_f).sum(dim=-1)                      # [B,C]
        denom = probs_f.sum(dim=-1) + targ_f.sum(dim=-1)            # [B,C]
        dice = (2.0 * inter + self.eps) / (denom + self.eps)        # [B,C]
        loss = 1.0 - dice                                           # [B,C]
        return loss.mean()                                          # scalar


class IoULoss(nn.Module):
    """
    Soft IoU/Jaccard loss for binary segmentation.
    """
    def __init__(self, eps: float = 1e-6, from_probs: bool = False):
        super().__init__()
        self.eps = eps
        self.from_probs = from_probs

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        probs = logits if self.from_probs else torch.sigmoid(logits)
        probs_f = _flatten_hw(probs)
        targ_f  = _flatten_hw(target)

        inter = (probs_f * targ_f).sum(dim=-1)
        union = probs_f.sum(dim=-1) + targ_f.sum(dim=-1) - inter
        iou = (inter + self.eps) / (union + self.eps)
        loss = 1.0 - iou
        return loss.mean()


class BCEWithLogitsDiceLoss(nn.Module):
    """
    Recommended default for DLP inverse (thr->mask).
    loss = w_bce * BCEWithLogits + w_dice * DiceLoss
    """
    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.dice = DiceLoss()
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            # pos_weight is a scalar; will be broadcasted to [1] and applied to positive class
            pw = torch.tensor([float(pos_weight)])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        bce = self.bce(logits, target)
        dice = self.dice(logits, target)
        return self.w_bce * bce + self.w_dice * dice


class BCEWithLogitsLoss(nn.Module):
    """
    Plain BCE loss wrapper (for convenience in Hydra).
    """
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        if pos_weight is None:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            pw = torch.tensor([float(pos_weight)])
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return self.loss(logits, target.float())


class BCEWithLogitsL1Loss(nn.Module):
    """
    loss = w_bce * BCEWithLogits(logits, target) + w_l1 * L1(sigmoid(logits), target)

    - BCEWithLogits: raw logits를 직접 사용 (안정적)
    - L1: 확률(probs) 공간에서의 절대오차(경계/면적 차이를 부드럽게 페널티)
    """
    def __init__(self, w_bce: float = 1.0, w_l1: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_l1 = float(w_l1)

        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            pw = torch.tensor([float(pos_weight)])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.l1 = nn.L1Loss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        l1 = self.l1(probs, target)
        return self.w_bce * bce + self.w_l1 * l1