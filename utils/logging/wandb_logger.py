# utils/logging/wandb_logger.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt  # ✅ 시각화를 위해 추가

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


@dataclass
class WandbLogConfig:
    log_every_n_iters: int = 0
    log_images_per_epoch: int = 0
    log_images_split: str = "val"   # "train" | "val"
    threshold: float = 0.5
    image_group: str = "viz"        # ✅ W&B panel prefix (e.g., "viz", "media", "qual")


def _wandb_on() -> bool:
    return (wandb is not None) and (getattr(wandb, "run", None) is not None)


def _as_numpy_01(x: torch.Tensor) -> np.ndarray:
    """Tensor -> numpy float32 in [0,1]. Expects [H,W] or [1,H,W]."""
    x = x.detach().float().cpu().squeeze()
    x = torch.clamp(x, 0.0, 1.0)
    return x.numpy().astype(np.float32)

def _to_rgb(img_gray: np.ndarray) -> np.ndarray:
    """
    (H, W) -> (H, W, 3) RGB로 변환 (Concatenation을 위해)
    """
    if img_gray.ndim == 2:
        return np.stack([img_gray]*3, axis=-1)
    return img_gray

def _get_error_heatmap(diff: np.ndarray) -> np.ndarray:
    """
    diff: (H, W) range approx [-1, 1]
    Returns: (H, W, 3) RGB numpy array using 'bwr' colormap
    - Red (>0): Pred > Target (Over / False Positive)
    - Blue (<0): Pred < Target (Under / False Negative)
    - White (0): Match
    """
    # 1. Normalize differences to [-1, 1] for colormap
    #    For binary, diff is exactly -1, 0, 1.
    #    For continuous, we might want to clip or normalize by max abs.
    max_val = np.max(np.abs(diff)) + 1e-6
    norm_diff = diff / (max_val) # Now inside [-1, 1]
    
    # 2. Shift to [0, 1] for matplotlib colormap (0.5 is center)
    #    -1 -> 0.0 (Blue), 0 -> 0.5 (White), 1 -> 1.0 (Red)
    diff_shifted = (norm_diff + 1) / 2.0
    
    # 3. Apply Colormap (bwr: Blue-White-Red)
    cmap = plt.get_cmap('bwr')
    heatmap = cmap(diff_shifted)[:, :, :3] # RGBA -> RGB
    return heatmap.astype(np.float32)

def _concat_row(imgs: List[np.ndarray]) -> np.ndarray:
    """
    imgs: list of [H,W,3] float32 in [0,1]
    returns: [H, W*len(imgs), 3]
    """
    # ensure same H,W
    h, w, c = imgs[0].shape
    out = []
    for im in imgs:
        if im.shape != (h, w, c):
            raise ValueError(f"Image shape mismatch: expected {(h,w)}, got {im.shape}")
        out.append(im)
    return np.concatenate(out, axis=1)

class WandbLogger:
    """
    - main.py에서 wandb.init()을 이미 호출했다고 가정.
    - 여기서는 wandb.run 유무만 보고 log만 담당.
    """
    def __init__(self, args: Any):
        self.args = args
        self.enabled = bool(getattr(args, "use_wandb", False)) and _wandb_on()
        self.cfg = self._parse_cfg(args)

        task_cfg = getattr(args, "task", {})
        if isinstance(task_cfg, dict):
            self.task_name = str(task_cfg.get("name", "")).strip()
        else:
            self.task_name = str(task_cfg).strip()
    
    def _parse_cfg(self, args: Any) -> WandbLogConfig:
        wb = getattr(args, "wandb", {})
        if isinstance(wb, dict):
            return WandbLogConfig(
                log_every_n_iters=int(wb.get("log_every_n_iters", 0) or 0),
                log_images_per_epoch=int(wb.get("log_images_per_epoch", 0) or 0),
                log_images_split=str(wb.get("log_images_split", "val") or "val"),
                threshold=float(wb.get("threshold", 0.5) or 0.5),
                image_group=str(wb.get("image_group", "viz") or "viz"),
            )
        # fallback (flattened args)
        return WandbLogConfig(
            log_every_n_iters=int(getattr(args, "wandb_log_every_n_iters", 0) or 0),
            log_images_per_epoch=int(getattr(args, "wandb_log_images_per_epoch", 0) or 0),
            log_images_split=str(getattr(args, "wandb_log_images_split", "val") or "val"),
            threshold=float(getattr(args, "wandb_threshold", 0.5) or 0.5),
            image_group=str(getattr(args, "wandb_image_group", "viz") or "viz"),
        )

    # -----------------------------
    # Train: iter metric logging
    # -----------------------------
    def log_train_iter(
        self,
        *,
        step: int,
        epoch: int,
        it: int,
        loss: float,
        dice: float,
        iou: float,
        lr: float,
    ) -> None:
        if not self.enabled:
            return
        wandb.log(
            {
                "train/iter_loss": float(loss),
                "train/iter_dice": float(dice),
                "train/iter_iou": float(iou),
                "train/lr": float(lr),
                "epoch": int(epoch),
                "iter": int(it),
            },
            step=int(step),
        )

    # -----------------------------
    # Val/Test: epoch metric logging
    # -----------------------------
    def log_epoch_metrics(
        self,
        *,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_dice: Optional[float] = None,
        val_iou: Optional[float] = None,
        lr: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        d: Dict[str, float] = {"epoch": float(epoch)}
        if train_loss is not None:
            d["train/loss"] = float(train_loss)
        if val_loss is not None:
            d["val/loss"] = float(val_loss)
        if val_dice is not None:
            d["val/dice"] = float(val_dice)
        if val_iou is not None:
            d["val/iou"] = float(val_iou)
        if lr is not None:
            d["lr"] = float(lr)
        wandb.log(d, step=int(step if step is not None else epoch))

    # -----------------------------
    # Images: epoch logging
    # -----------------------------
    @torch.no_grad()
    def log_epoch_images(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        split_name: str,
        device: torch.device,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        k = int(self.cfg.log_images_per_epoch)
        if k <= 0:
            return

        thr = float(self.cfg.threshold)
        model.eval()

        images = []     # each element is ONE composite image per sample: [input|target|pred|error]
        count = 0
        # ✅ Forward Task인지 확인
        is_forward = self.task_name.startswith("forward")

        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)
            # ✅ pipeline dataset: y can be dict (e.g., {"ld":..., "mask":...})
            #    For inverse visualization, we want the mask target by default.
            if isinstance(y, dict):
                if "mask" in y:
                    y = y["mask"]
                else:
                    # fallback: pick first tensor value
                    # (keeps compatibility with other dict-shaped targets)
                    for v in y.values():
                        y = v
                        break
            # also handle list/tuple targets defensively
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.to(device, non_blocking=True)

            logits = model(x)
            prob = torch.sigmoid(logits)

            # 🔴 [FIX] Forward면 이진화 금지 (Continuous), Inverse면 이진화 (Binary)
            # SigmoidL1Loss를 사용하므로 prob(0~1)가 곧 prediction입니다.
            if is_forward:
                pred = prob  # 있는 그대로 (0.0 ~ 1.0)
            else:
                pred = (prob >= thr).float() # 이진화 (0 or 1) 

            # ✅ [Modified] error map: Signed Difference
            # ✅ Error Map: 단순 차이 (Pred - Target)
            diff_tensor = pred - y

            bsz = x.shape[0]
            for bi in range(bsz):
                if count >= k:
                    break

                sample_key = ""
                # meta는 collate에 따라 dict-of-list 또는 list-of-dict일 수 있음
                if isinstance(meta, dict) and "sample_key" in meta:
                    try:
                        sample_key = str(meta["sample_key"][bi])
                    except Exception:
                        sample_key = str(meta["sample_key"])
                elif isinstance(meta, (list, tuple)) and len(meta) > bi and isinstance(meta[bi], dict):
                    sample_key = str(meta[bi].get("sample_key", ""))

                x_img = _as_numpy_01(x[bi, 0])
                y_img = _as_numpy_01(y[bi, 0])

                p_img = _as_numpy_01(pred[bi, 0])   # ✅ pred는 binary로 (요청: input/target/pred/error)
                
                d_img = diff_tensor[bi, 0].detach().cpu().numpy()
                e_img = _get_error_heatmap(d_img)   # ✅ RGB Heatmap 변환

                # ✅ RGB로 통일하여 붙이기: [input | target | pred | error]
                row = _concat_row([_to_rgb(x_img), _to_rgb(y_img), _to_rgb(p_img), e_img])
                images.append(
                    wandb.Image(
                        row,
                        caption=f"{sample_key} | [input | target | pred@{thr:.2f} | error]"
                    )
                )

                count += 1

            if count >= k:
                break

        # ✅ step은 반드시 단조 증가해야 함.
        # epoch(0,1,2,...)로 찍으면 train iter step(1000,2000,...)보다 작아져 무시됨.
        s = int(step if step is not None else epoch)
        group = str(self.cfg.image_group).strip() or "viz"
        # ✅ val/ 아래가 아니라 viz/ 아래로 들어가게
        wandb.log({f"{group}/{split_name}_samples": images, "epoch": int(epoch)}, step=s)