# utils/common/physics_loss.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Optional

from utils.evaluation.proxy_forward import ProxyForwardModel

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for DLP 3D Printing.
    
    Pipeline:
    1. Inverse Output (Logits) -> Sigmoid -> 0~1 Mask
    2. Pixel Grid Constraint: Mean Downsample (160) -> Nearest Upsample (640)
    3. Frozen Forward Model: Simulates Light Distribution (Logits)
    4. Soft Curing: Piecewise Linear Function (50~80 intensity)
    5. Loss: L1 distance between Soft-Cured Shape and Target Mask
    """
    # NOTE:
    # - "DoC" (degree of cure) stage is optional.
    # - Keep LD->CI mapping via existing soft_curing() unchanged.
 

    def __init__(
        self,
        forward_checkpoint: str,
        grid_size: int = 160,
        infer_size: int = 640,
        cure_thr_low: float = 50.0,
        cure_thr_high: float = 80.0,
        # ✅ NEW: curing config (selectable)
        curing: Optional[Dict[str, Any]] = None,
        # ✅ [NEW] DoC options (Hydra로 조절)
        doc_enable: bool = False,
        doc_gauss_enable: bool = True,
        doc_gauss_kernel_size: int = 51,
        doc_gauss_sigma: float = 12.0,
        doc_gauss_n_iter: int = 1,
        # ✅ forward model output policy (robust)
        forward_output_is_logits: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.grid_size = grid_size
        self.infer_size = infer_size
        self.cure_thr_low = cure_thr_low
        self.cure_thr_high = cure_thr_high
        
        # ✅ curing config
        curing = dict(curing or {})
        self.curing_mode = str(curing.get("mode", "soft_piecewise")).strip().lower()
        # threshold is in normalized LD domain: sigmoid(ld_logits) in [0,1]
        self.curing_threshold = float(curing.get("threshold", 0.23529))
        self.curing_sigmoid_temp = float(curing.get("sigmoid_temp", 20.0))

        valid_modes = {"soft_piecewise", "hard_threshold", "sigmoid_threshold"}
        if self.curing_mode not in valid_modes:
            raise ValueError(f"Unknown curing.mode='{self.curing_mode}'. Valid: {sorted(valid_modes)}")

        # ✅ DoC config
        self.doc_enable = bool(doc_enable)
        self.doc_gauss_enable = bool(doc_gauss_enable)
        self.doc_gauss_kernel_size = int(doc_gauss_kernel_size)
        self.doc_gauss_sigma = float(doc_gauss_sigma)
        self.doc_gauss_n_iter = int(doc_gauss_n_iter)
        self.forward_output_is_logits = bool(forward_output_is_logits)
 
        # Load Proxy Forward Model
        print(f"[PhysicsLoss] Loading Frozen Forward Model from: {forward_checkpoint}")
        self.forward_model = ProxyForwardModel(ckpt_path=forward_checkpoint, device=torch.device(device))
        self.forward_model.eval()
        
        # Explicitly freeze parameters just in case
        for param in self.forward_model.parameters():
            param.requires_grad = False

    def _apply_curing(self, ld_logits: torch.Tensor) -> torch.Tensor:
        """
        ld_logits: (B,1,H,W) logits (LD)
        Returns cured image in [0,1].

        Modes:
          - soft_piecewise: 기존 _soft_curing() (미분 가능)
          - hard_threshold: (sigmoid(ld_logits) >= thr).float() (거의 미분 불가)
          - sigmoid_threshold: sigmoid(temp*(sigmoid(ld)-thr)) (미분 가능, hard 근사)
        """
        if self.curing_mode == "soft_piecewise":
            return self._soft_curing(ld_logits)

        ld_01 = torch.sigmoid(ld_logits)  # normalized intensity in [0,1]
        thr = float(self.curing_threshold)

        if self.curing_mode == "hard_threshold":
            # ⚠️ hard step -> gradient 거의 0 (학습이 멈출 수 있음)
            return (ld_01 >= thr).to(ld_01.dtype)

        # sigmoid_threshold
        temp = float(self.curing_sigmoid_temp)
        # larger temp -> sharper transition
        return torch.sigmoid(temp * (ld_01 - thr))

    # -----------------------------
    # ✅ [NEW] DoC utils
    # -----------------------------
    @staticmethod
    def _logit_safe(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    @staticmethod
    def _gaussian_kernel2d(ks: int, sigma: float, device, dtype) -> torch.Tensor:
        ks = int(ks)
        sigma = float(sigma)
        if ks % 2 == 0:
            ks += 1  # enforce odd
        ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma + 1e-12))
        kernel = kernel / (kernel.sum() + 1e-12)
        return kernel

    def _apply_gaussian_blur(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        img_01: (B,1,H,W) in [0,1]
        """
        if (not self.doc_gauss_enable) or (self.doc_gauss_sigma <= 0) or (self.doc_gauss_kernel_size <= 1):
            return img_01
        ks = int(self.doc_gauss_kernel_size)
        if ks % 2 == 0:
            ks += 1
        k = self._gaussian_kernel2d(ks, self.doc_gauss_sigma, device=img_01.device, dtype=img_01.dtype)
        w = k[None, None, :, :]  # (1,1,ks,ks)
        pad = ks // 2
        x = img_01
        n_iter = max(1, int(self.doc_gauss_n_iter))
        for _ in range(n_iter):
            x = F.conv2d(x, w, bias=None, stride=1, padding=pad)
        return x

    def _pixelize_constraint(self, mask_01: torch.Tensor) -> torch.Tensor:
        """
        Simulate physical pixel constraint (160x160 resolution).
        Differentiable approximation: Mean Down -> Nearest Up.
        """
        # 1. Mean Downsample: Approximates "Area Curing" in a coarse pixel
        x_small = F.adaptive_avg_pool2d(mask_01, output_size=(self.grid_size, self.grid_size))
        
        # 2. Nearest Upsample: Projects back to simulation grid (640x640)
        x_back = F.interpolate(x_small, size=(self.infer_size, self.infer_size), mode="nearest")
        
        return x_back

    def _degree_of_cure(self, ld_logits: torch.Tensor) -> torch.Tensor:
        """
        ✅ DoC stage (optional): LD(logits) -> LD(prob) -> (Gaussian blur etc.) -> back to logits
        Why return logits?
        - To preserve existing LD->CI path via _soft_curing(ld_logits) without changing its behavior.
        """
        # logits -> prob [0,1]
        ld_01 = torch.sigmoid(ld_logits)

        # spatial spread / proximity as Gaussian convolution
        ld_01 = self._apply_gaussian_blur(ld_01)

        # stay valid range and return to logits domain
        ld_01 = torch.clamp(ld_01, 0.0, 1.0)
        ld_logits_doc = self._logit_safe(ld_01)
        return ld_logits_doc
 
    def _soft_curing(self, ld_logits: torch.Tensor) -> torch.Tensor:
        """
        Differentiable piecewise-linear curing.

        Map intensity (0~255) to cured (0~1) with 3 segments:
          [0,  50]  -> [0.0, 0.1]
          [50, 80]  -> [0.1, 1.0]
          [80, 255] -> 1.0 (saturate)
        """
        # Logits -> Normalized Intensity (0.0 ~ 1.0)
        ld_norm = torch.sigmoid(ld_logits)
        
        # Normalized -> Real Intensity (0.0 ~ 255.0)
        ld_real = ld_norm * 255.0
        
        # Segment breakpoints
        x0 = 0.0
        x1 = float(self.cure_thr_low)   # 50
        x2 = float(self.cure_thr_high)  # 80

        # Output levels at breakpoints (requested)
        y0, y1, y2 = 0.0, 0.001, 1.0

        # Slopes for first two segments
        m01 = (y1 - y0) / (x1 - x0 + 1e-6)
        m12 = (y2 - y1) / (x2 - x1 + 1e-6)

        x = ld_real
        # piecewise linear: seg3 is saturation at 1.0
        y_seg1 = y0 + m01 * (x - x0)
        y_seg2 = y1 + m12 * (x - x1)

        cured = torch.where(
            x < x1, y_seg1,
            torch.where(x < x2, y_seg2, torch.ones_like(x))
        )
        cured = torch.clamp(cured, 0.0, 1.0)
        return cured

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: Output from BandUnet [B, 1, H, W] (Logits)
            target_mask: Ground Truth Target [B, 1, H, W] (Binary 0/1)
        """
        
        # 1. Inverse Model Output (Logits) -> Probabilities (0~1)
        pred_probs = torch.sigmoid(pred_logits)
        
        # 2. Apply Physical Pixel Grid Constraint (160px -> 640px)
        #    This forces the generator to output masks that respect the DMD resolution
        pred_pixelized = self._pixelize_constraint(pred_probs)
        
        # 3. Forward Simulation (Forward Model is Frozen)
        #    Input: 0~1 Mask, Output: Light Distribution Logits
        #    Note: ProxyForwardModel internally handles normalization if trained properly,
        #          but usually expects 0~1 float inputs.
        # IMPORTANT:
        # - ProxyForwardModel.forward() is decorated with no_grad for benchmark.
        # - For training, use forward_with_grad() to keep gradient to pred_pixelized.
        if hasattr(self.forward_model, "forward_with_grad"):
            ld_out = self.forward_model.forward_with_grad(pred_pixelized)
        else:
            # fallback: direct call (may be no_grad if forward() is decorated)
            ld_out = self.forward_model.model(pred_pixelized.to(self.forward_model.device))
   
        # ✅ normalize forward output representation:
        # - if forward outputs prob(0~1), convert to logits to keep soft_curing unchanged
        if self.forward_output_is_logits:
            ld_logits = ld_out
        else:
            ld_logits = self._logit_safe(torch.clamp(ld_out, 0.0, 1.0))
 
        # 4. ✅ (Optional) DoC stage: LD -> DoC(LD) (still logits)
        if self.doc_enable:
            ld_logits = self._degree_of_cure(ld_logits)

        # 5. ✅ Selectable curing (yaml controlled)
        cured_img = self._apply_curing(ld_logits)
          
        # safety: ensure target is same spatial size as cured_img
        if target_mask.shape[-2:] != cured_img.shape[-2:]:
            target_mask = F.interpolate(target_mask, size=cured_img.shape[-2:], mode="nearest")

        # 6. Compute Loss (L1)
        #    Compare the "Simulated Cured Result" with the "Desired Target"
        loss = F.l1_loss(cured_img, target_mask)
        
        return loss