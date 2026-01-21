from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import cv2
import wandb
from typing import Dict, List
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.evaluation.benchmark_dataset import BenchmarkDataset
from utils.evaluation.proxy_forward import ProxyForwardModel


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

    def _postprocess_forward(self, out: torch.Tensor) -> torch.Tensor:
        # forward model output이 logits인 경우에만 sigmoid
        if self.fwd_apply_sigmoid:
            return torch.sigmoid(out)
        return out

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
        ld = self._postprocess_forward(ld)   # ✅ curing 전에 정규화
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

    def _process_category(self, dataset: BenchmarkDataset) -> np.ndarray:
        """
        Process category -> Horizontal Stitch of columns.
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        row_images = []
        
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
                Naive_in = self._apply_forward_input_policy(T_infer_t)
                Naive_Print_infer = self.forward_model(Naive_in)
                Naive_Print_infer = self._apply_curing(Naive_Print_infer)           

                # 3. Correction Path: T -> Inverse -> M (postprocess configurable)
                M_logits = self.inverse_model(T_infer_t)
                M_out = self._postprocess_inverse(M_logits)  # may be gray or binary
   
                # 4. Corrected Print: M -> Forward -> P(M) -> (optional curing threshold)
                Corr_in = self._apply_forward_input_policy(M_out)
                Corrected_Print_infer = self.forward_model(Corr_in)
                Corrected_Print_infer = self._apply_curing(Corrected_Print_infer)
 
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
        return cv2.vconcat(row_images)

    def generate_report(self, full_dataset: BenchmarkDataset) -> Dict[str, wandb.Image]:
        categories = full_dataset.get_categories()
        report = {}
        for cat in categories:
            subset = full_dataset.get_subset_by_category(cat)
            grid_img = self._process_category(subset)
            report[f"Report/{cat}"] = wandb.Image(grid_img, caption=f"Category: {cat}")
        return report