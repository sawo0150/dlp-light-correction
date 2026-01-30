from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import cv2
import wandb
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.evaluation.benchmark_dataset import BenchmarkDataset
from utils.evaluation.proxy_forward import ProxyForwardModel
import torch.nn.functional as F
import math


class BenchmarkReporter:
    def __init__(
        self,
        inverse_model: nn.Module,
        forward_model: ProxyForwardModel,
        device: torch.device,
        image_size: int = 160,      # Visualization size
        model_input_size: int = 640, # Model inference size
        # ✅ NEW: inverse output post-process options
        inverse_apply_sigmoid: bool = True,
        inverse_binarize: bool = True,
        inverse_binarize_thr: float = 0.5,
        # ✅ NEW: curing threshold options (forward output -> cured)
        curing_threshold: float = 0.5,
        curing_binarize: bool = True,
        # ✅ NEW: forward input policy align
        forward_binarize_input: bool = False,
        forward_apply_sigmoid: bool = False,
        # ✅ NEW: mask pixelization (coarse grid constraint)
        mask_pixelize_enable: bool = False,
        mask_pixelize_size: int = 160,
        mask_pixelize_downsample: str = "mean",   # "mean"(area/avg) | "nearest"
        mask_pixelize_upsample: str = "nearest",  # "nearest" | "bilinear"
        mask_pixelize_apply_to: str = "both",     # "both" | "naive" | "corr"

        # ✅ [NEW] DoC options (eval-only)
        doc_enable: bool = False,
        doc_gauss_enable: bool = True,
        doc_gauss_kernel_size: int = 51,
        doc_gauss_sigma: float = 12.0,
        doc_gauss_n_iter: int = 1,
    ):
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        self.device = device
        self.viz_size = image_size
        self.infer_size = model_input_size

        # postprocess configs
        self.inv_apply_sigmoid = bool(inverse_apply_sigmoid)
        self.inv_binarize = bool(inverse_binarize)
        self.inv_bin_thr = float(inverse_binarize_thr)

        self.curing_thr = float(curing_threshold)
        self.curing_binarize = bool(curing_binarize)
        self.fwd_binarize_input = bool(forward_binarize_input)
        self.fwd_apply_sigmoid = bool(forward_apply_sigmoid)

        # pixelize configs
        self.mp_enable = bool(mask_pixelize_enable)
        self.mp_size = int(mask_pixelize_size)
        self.mp_down = str(mask_pixelize_downsample).lower()
        self.mp_up = str(mask_pixelize_upsample).lower()
        self.mp_apply_to = str(mask_pixelize_apply_to).lower()

        self.doc_enable = bool(doc_enable)
        self.doc_gauss_enable = bool(doc_gauss_enable)
        self.doc_gauss_kernel_size = int(doc_gauss_kernel_size)
        self.doc_gauss_sigma = float(doc_gauss_sigma)
        self.doc_gauss_n_iter = int(doc_gauss_n_iter)

    @staticmethod
    def _logit_safe(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    @staticmethod
    def _gaussian_kernel2d(ks: int, sigma: float, device, dtype) -> torch.Tensor:
        ks = int(ks)
        if ks % 2 == 0:
            ks += 1
        ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma) + 1e-12))
        kernel = kernel / (kernel.sum() + 1e-12)
        return kernel

    def _apply_gaussian_blur(self, img_01: torch.Tensor) -> torch.Tensor:
        if (not self.doc_gauss_enable) or (self.doc_gauss_sigma <= 0) or (self.doc_gauss_kernel_size <= 1):
            return img_01
        ks = int(self.doc_gauss_kernel_size)
        if ks % 2 == 0:
            ks += 1
        k = self._gaussian_kernel2d(ks, self.doc_gauss_sigma, device=img_01.device, dtype=img_01.dtype)
        w = k[None, None, :, :]
        pad = ks // 2
        x = img_01
        for _ in range(max(1, int(self.doc_gauss_n_iter))):
            x = F.conv2d(x, w, bias=None, stride=1, padding=pad)
        return x

    def _apply_doc_to_ld_logits(self, ld_logits: torch.Tensor) -> torch.Tensor:
        """
        LD(logits) -> LD(prob) -> blur -> back to logits
        (soft_curing이 logits->sigmoid를 기대하는 구조를 최대한 유지하려고 logits로 복귀)
        """
        # ✅ forward output policy:
        # - fwd_apply_sigmoid=True  => ld_logits is logits -> sigmoid
        # - fwd_apply_sigmoid=False => ld_logits is already prob(0~1)
        ld_01 = torch.sigmoid(ld_logits) if self.fwd_apply_sigmoid else ld_logits
        ld_01 = self._apply_gaussian_blur(ld_01)
        ld_01 = torch.clamp(ld_01, 0.0, 1.0)
        return self._logit_safe(ld_01)

    # ---------------------------------------------------------------------
    # ✅ Metrics helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _bin01(x: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
        """x: [B,1,H,W] float -> {0,1} float"""
        return (x >= thr).float()

    def _count_error_pixels(self, pred01: torch.Tensor, target01: torch.Tensor) -> Tuple[int, int]:
        """
        Count mismatched pixels between pred and target (both mask-like).
        Returns:
          (err_px, total_px) as python ints
        """
        # 안전하게 binary화 (curing_binarize=False인 경우도 대비)
        p = self._bin01(pred01, 0.5)
        t = self._bin01(target01, 0.5)
        # XOR mismatch
        err = (p != t).sum()
        total = t.numel()
        return int(err.item()), int(total)

    def _postprocess_forward(self, out: torch.Tensor) -> torch.Tensor:
        # forward model output이 logits인 경우에만 sigmoid
        if self.fwd_apply_sigmoid:
            return torch.sigmoid(out)
        return out

    def _pixelize_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce coarse pixel grid on a mask-like tensor at inference resolution.
        Policy (requested):
          - downsample: mean (area/avg)  ✅
          - upsample: nearest           ✅

        x: [B,1,H,W] (H=W=self.infer_size expected)
        returns: [B,1,self.infer_size,self.infer_size]
        """
        if (not self.mp_enable) or (self.mp_size <= 0):
            return x

        # 1) downsample to (mp_size, mp_size)
        if self.mp_down in ("mean", "area", "avg"):
            # adaptive_avg_pool2d == mean over regions (robust even when not divisible)
            x_small = F.adaptive_avg_pool2d(x, output_size=(self.mp_size, self.mp_size))
        elif self.mp_down in ("nearest",):
            x_small = F.interpolate(x, size=(self.mp_size, self.mp_size), mode="nearest")
        else:
            # safe fallback
            x_small = F.adaptive_avg_pool2d(x, output_size=(self.mp_size, self.mp_size))

        # 2) upsample back to infer_size
        if self.mp_up in ("nearest",):
            x_back = F.interpolate(x_small, size=(self.infer_size, self.infer_size), mode="nearest")
        elif self.mp_up in ("bilinear", "linear"):
            x_back = F.interpolate(x_small, size=(self.infer_size, self.infer_size), mode="bilinear", align_corners=False)
        else:
            x_back = F.interpolate(x_small, size=(self.infer_size, self.infer_size), mode="nearest")

        return x_back

    def _apply_forward_input_policy(self, x: torch.Tensor) -> torch.Tensor:
        if self.fwd_binarize_input:
            return (x >= 0.5).float()
        return x

    def _to_numpy_u8(self, tensor: torch.Tensor) -> np.ndarray:
        """ [1, H, W] float 0~1 -> [H, W] uint8 0~255 """
        arr = tensor.squeeze().detach().cpu().numpy()
        arr = np.clip(arr, 0, 1) * 255.0
        return arr.astype(np.uint8)

    def _resize_tensor(self, tensor: torch.Tensor, size: int) -> torch.Tensor:
        """ Resize [B, 1, H, W] tensor (default bilinear for continuous maps) """
        return torch.nn.functional.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)

    def _resize_mask_tensor(self, tensor: torch.Tensor, size: int) -> torch.Tensor:
        """ Resize mask-like [B,1,H,W] with nearest to match training forward mask preprocessing """
        return torch.nn.functional.interpolate(tensor, size=(size, size), mode='nearest')


    def _postprocess_inverse(self, logits: torch.Tensor) -> torch.Tensor:
        """
        inverse_model output postprocess:
          logits -> (optional sigmoid) -> (optional binarize)
        Returns:
          mask_like: [B,1,H,W] float in [0,1] if sigmoid used, else raw.
                    if binarize True -> {0,1}
        """
        x = logits
        if self.inv_apply_sigmoid:
            x = torch.sigmoid(x)
        # If inverse outputs grayscale mask, do NOT binarize.
        if self.inv_binarize:
            x = (x >= self.inv_bin_thr).float()
        return x

    def _apply_curing(self, ld: torch.Tensor) -> torch.Tensor:
        """
        forward output(LD/print proxy) -> cured representation
        If curing_binarize: threshold -> {0,1}
        Else: return ld as-is (0~1 expected).
        """
        # 1) LD를 0~1로 정규화 (logits이면 sigmoid, 아니면 그대로)
        ld = self._postprocess_forward(ld)

        # 2) ✅ (Optional) DoC stage: LD -> DoC(LD) in 0~1 domain
        #    LD->CI vs LD->DoC->CI 를 yaml로 토글 가능하게.
        if self.doc_enable:
            ld = self._apply_gaussian_blur(ld)
            ld = torch.clamp(ld, 0.0, 1.0)

        if self.curing_binarize:
            return (ld >= self.curing_thr).float()
        return ld

    def _get_error_heatmap(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Generates RGB heatmap for (Pred - Target).
        Red (>0) = Over-cure, Blue (<0) = Under-cure, White (0) = Match.
        """
        diff = pred.astype(np.float32) - target.astype(np.float32) # Range -255 ~ 255
        norm_diff = diff / 255.0  # [-1, 1]
        diff_shifted = (norm_diff + 1) / 2.0 # [0, 1]
        
        cmap = plt.get_cmap('bwr')
        heatmap = cmap(diff_shifted)[:, :, :3] # RGBA -> RGB
        
        return (heatmap * 255).astype(np.uint8)

    def _process_category(self, dataset: BenchmarkDataset) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process category -> Horizontal Stitch of columns.
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        row_images = []
        
        # ✅ Metrics accumulators (category-level)
        naive_err_sum = 0
        corr_err_sum  = 0
        total_px_sum  = 0
        n_samples     = 0

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_color, thickness = 0.4, (0,0,0), 1
        
        self.inverse_model.eval()

        for i, (img_t, name_t, _) in enumerate(loader):
            name = name_t[0]
            
            # 1. Inputs
            T_viz_t = img_t.to(self.device) # Target [1, 1, H, W] (benchmark는 보통 160)
            # ✅ Naive path input should mimic forward training input => nearest resize for mask
            T_infer_t = self._resize_mask_tensor(T_viz_t, self.infer_size)
            # ✅ FIX: visualization/heatmap용 target도 viz_size로 맞춤
            if int(T_viz_t.shape[-1]) != int(self.viz_size):
                T_vis_t = self._resize_mask_tensor(T_viz_t, self.viz_size)
            else:
                T_vis_t = T_viz_t

            with torch.no_grad():
                # 2. Naive Path: T -> Forward -> P(T) -> (optional curing threshold)
                Naive_in = T_infer_t
                if self.mp_enable and self.mp_apply_to in ("both", "naive"):
                    Naive_in = self._pixelize_mask(Naive_in)
                Naive_in = self._apply_forward_input_policy(Naive_in)

                Naive_Print_infer = self.forward_model(Naive_in)
                Naive_Print_infer = self._apply_curing(Naive_Print_infer)           

                # 3. Correction Path: T -> Inverse -> M (postprocess configurable)
                M_logits = self.inverse_model(T_infer_t)
                M_out = self._postprocess_inverse(M_logits)  # may be gray or binary
   
                # 4. Corrected Print: M -> Forward -> P(M) -> (optional curing threshold)
                Corr_in = M_out
                if self.mp_enable and self.mp_apply_to in ("both", "corr"):
                    Corr_in = self._pixelize_mask(Corr_in)
                Corr_in = self._apply_forward_input_policy(Corr_in)
                Corrected_Print_infer = self.forward_model(Corr_in)
                Corrected_Print_infer = self._apply_curing(Corrected_Print_infer)
 
                # ✅ ----------------------------------------------------------
                # ✅ Metrics per-sample (error pixels vs target at infer resolution)
                # ✅ ----------------------------------------------------------
                # target도 infer resolution에서 비교(학습/forward 입력과 같은 해상도)
                T_bin_infer = self._bin01(T_infer_t, 0.5)
                # pred는 curing 결과가 binary일 확률이 높지만, 안전하게 binarize
                naive_err, total_px = self._count_error_pixels(Naive_Print_infer, T_bin_infer)
                corr_err,  _        = self._count_error_pixels(Corrected_Print_infer, T_bin_infer)

                naive_err_sum += naive_err
                corr_err_sum  += corr_err
                total_px_sum  += total_px
                n_samples     += 1
  
            # --- Visualization Prep (Resize back to 160) ---
            
            # Row 1: Target
            # ✅ FIX: viz_size 기반 target 사용
            img_T = self._to_numpy_u8(T_vis_t)
            img_T_rgb = cv2.cvtColor(img_T, cv2.COLOR_GRAY2RGB)

            # Row 2: Naive Print
            img_Naive_P = self._to_numpy_u8(self._resize_tensor(Naive_Print_infer, self.viz_size))
            img_Naive_P_rgb = cv2.cvtColor(img_Naive_P, cv2.COLOR_GRAY2RGB)

            # Row 3: Naive Error (P(T) - T) [SWAPPED]
            img_Naive_Err_rgb = self._get_error_heatmap(img_Naive_P, img_T)

            # Row 4: Corrected Mask (M) (gray or binary depending on config)
            img_M = self._to_numpy_u8(self._resize_mask_tensor(M_out, self.viz_size))
            img_M_rgb = cv2.cvtColor(img_M, cv2.COLOR_GRAY2RGB)

            # Row 5: Corrected Print (P(M))
            img_Corr_P = self._to_numpy_u8(self._resize_tensor(Corrected_Print_infer, self.viz_size))
            img_Corr_P_rgb = cv2.cvtColor(img_Corr_P, cv2.COLOR_GRAY2RGB)

            # Row 6: Corrected Error (P(M) - T)
            img_Corr_Err_rgb = self._get_error_heatmap(img_Corr_P, img_T)

            # --- Stitching ---
            rows = [
                (img_T_rgb, "Target"),
                (img_Naive_P_rgb, "Naive Prt"),
                (img_Naive_Err_rgb, "Naive Err"),
                (img_M_rgb, "Corr Mask"),
                (img_Corr_P_rgb, "Corr Prt"),
                (img_Corr_Err_rgb, "Corr Err")
            ]
            
            annotated_rows = []
            for img, lbl in rows:
                canvas = img.copy()
                # Text Background
                cv2.rectangle(canvas, (0, 0), (70, 15), (255, 255, 255), -1)
                cv2.putText(canvas, lbl, (5, 10), font, font_scale, font_color, thickness)
                annotated_rows.append(canvas)

            # ✅ Row/Column swap:
            #   - before: one sample = vertical column (vconcat of 6)
            #   - after : one sample = horizontal row  (hconcat of 6)
            row_img = cv2.hconcat(annotated_rows)

            # ✅ Add left-side header (Image Name) as a vertical strip
            # header width = 120px (tweak if needed)
            header_w = 120
            header = np.ones((self.viz_size, header_w, 3), dtype=np.uint8) * 255
            # Center-ish text vertically
            cv2.putText(
                header,
                name,
                (5, min(self.viz_size - 5, 20)),
                font,
                font_scale,
                font_color,
                thickness,
            )

            full_row = cv2.hconcat([header, row_img])
            row_images.append(full_row)
  
        if not row_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # ✅ stack samples vertically
        grid = cv2.vconcat(row_images)

        # ✅ Category metrics
        if total_px_sum <= 0:
            metrics = {
                "corr_err_px": 0.0,
                "corr_err_rate": 0.0,
                "err_reduction": 0.0,
            }
            return grid, metrics

        naive_rate = float(naive_err_sum) / float(total_px_sum)
        corr_rate  = float(corr_err_sum)  / float(total_px_sum)
        # reduction = (naive - corr) / naive
        if naive_err_sum > 0:
            reduction = float(naive_err_sum - corr_err_sum) / float(naive_err_sum)
        else:
            # naive가 완벽(0 error)이면 reduction 정의가 애매하므로 0으로 둠
            reduction = 0.0

        metrics = {
            "corr_err_px": float(corr_err_sum),
            "corr_err_rate": float(corr_rate),
            "err_reduction": float(reduction),
        }
        return grid, metrics

    def generate_report(self, full_dataset: BenchmarkDataset) -> Dict[str, Any]:
        categories = full_dataset.get_categories()
        report = {}
        # ✅ overall accumulator
        overall = {
            "corr_err_px": 0.0,
            "total_px": 0.0,
            "samples": 0.0,  # overall rate 계산용(=infer_size^2 * samples)
        }

        for cat in categories:
            subset = full_dataset.get_subset_by_category(cat)
            grid_img, m = self._process_category(subset)
            report[f"Report/{cat}"] = wandb.Image(grid_img, caption=f"Category: {cat}")

            # ✅ log category metrics as scalars
            report[f"Metrics/{cat}/corr_err_px"]    = m["corr_err_px"]
            report[f"Metrics/{cat}/corr_err_rate"]  = m["corr_err_rate"]
            report[f"Metrics/{cat}/err_reduction"]  = m["err_reduction"]

            # accumulate overall
            overall["corr_err_px"]  += m["corr_err_px"]
            # total_px는 category에서 직접 안 주고 있어서 rate로부터 역추정은 위험함
            # -> 여기서는 samples만 합하고, overall_rate는 아래에서 "infer_size*infer_size*samples"로 계산
            # category별 샘플 수를 metrics에서 빼버렸으므로, overall은 별도 집계가 필요
            # 가장 안전한 방법: _process_category에서 n_samples를 반환하지 않으니 여기서는 category dataset 길이로 추정
            overall["samples"]      += float(len(subset))

        # ✅ Overall metrics (rate 계산은 infer_size 기반으로 확정 가능)
        if overall["samples"] > 0:
            total_px = float(self.infer_size * self.infer_size) * float(overall["samples"])
            report["Metrics/_overall/corr_err_px"]    = float(overall["corr_err_px"])
            report["Metrics/_overall/corr_err_rate"]  = float(overall["corr_err_px"]) / total_px
            # overall err_reduction은 naive_err_px를 로깅에서 제외했으므로 계산/로깅도 제외
            # (원하면 다시 overall_naive_err_px만 내부적으로 집계해서 reduction만 남길 수도 있음)


        return report