from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
from datetime import datetime
from hydra.utils import instantiate
from omegaconf import OmegaConf

def _safe_torch_load(path: Path, map_location: Any):
    """
    PyTorch 2.6+ default weights_only=True 때문에,
    SimpleNamespace(args) 등이 들어있는 legacy checkpoint 로딩이 실패할 수 있음.
    우리가 만든(신뢰 가능한) 체크포인트라면 weights_only=False로 로드한다.
    """
    # 1) PyTorch 2.6+ (weights_only 인자 지원)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 2) 구버전 PyTorch: weights_only 인자 자체가 없음
        return torch.load(path, map_location=map_location)

def _fmt_num(x: Any, ndigits: int = 6) -> str:
    try:
        if x is None:
            return "None"
        if isinstance(x, (float, int)):
            return f"{float(x):.{ndigits}g}"
        return str(x)
    except Exception:
        return str(x)


def _print_ckpt_summary(path: Path, ckpt: Any) -> None:
    """Best-effort checkpoint metadata summary (non-fatal)."""
    try:
        if not isinstance(ckpt, dict):
            print(f"[ProxyForward] ckpt type={type(ckpt)} (not dict) — skip meta summary.")
            return

        epoch = ckpt.get("epoch", None)
        best_val_loss = ckpt.get("best_val_loss", None)
        best_val_score = ckpt.get("best_val_score", ckpt.get("best_val_dice", None))
        exp_dir = ckpt.get("exp_dir", None)
        keys_hint = []
        for k in ["epoch", "best_val_loss", "best_val_score", "exp_dir", "scheduler", "optimizer", "scaler"]:
            if k in ckpt:
                keys_hint.append(k)

        # args/config에서 task/model 정보가 있으면 추가로 보여줌
        task_name = None
        if "args" in ckpt:
            a = ckpt["args"]
            if hasattr(a, "task"):
                t = getattr(a, "task")
                if isinstance(t, dict):
                    task_name = t.get("name", None)
        elif "config" in ckpt and isinstance(ckpt["config"], dict):
            task_name = (ckpt["config"].get("task") or {}).get("name", None)

        print(
            f"[ProxyForward] ckpt meta: epoch={_fmt_num(epoch)} "
            f"best_val_loss={_fmt_num(best_val_loss)} best_val_score={_fmt_num(best_val_score)}"
            + (f" task={task_name}" if task_name else "")
        )
        if keys_hint:
            print(f"[ProxyForward] ckpt keys: {', '.join(keys_hint)}")
    except Exception as e:
        print(f"[ProxyForward] (meta) failed to summarize ckpt: {e}")

class ProxyForwardModel(nn.Module):
    """
    Wraps a pre-trained Forward Model to act as a simulator (Digital Twin).
    """
    def __init__(self, ckpt_path: str | Path, device: torch.device):
        super().__init__()
        self.device = device
        self.model = self._load_model(ckpt_path)
        self.model.eval()
        self.model.to(device)

    def _load_model(self, ckpt_path: str | Path) -> nn.Module:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[ProxyForward] Warning: Checkpoint {path} not found. Using Identity (Pass-through).")
            return nn.Identity()

        print(f"[ProxyForward] Loading forward model from {path} ...")
        
        try:
            ckpt = _safe_torch_load(path, map_location=self.device)
            
            # ✅ Print basic checkpoint info (epoch/best loss/score/exp_dir...)
            _print_ckpt_summary(path, ckpt)

            # 1. Attempt to find model config
            model_cfg = None
            
            # Case A: Saved as 'args' (SimpleNamespace or dict)
            if 'args' in ckpt:
                args = ckpt['args']
                # If args is an object, try accessing .model
                if hasattr(args, 'model'):
                    model_cfg = args.model
                # If args is a dict (unlikely based on your main.py but possible)
                elif isinstance(args, dict) and 'model' in args:
                    model_cfg = args['model']

            # Case B: Saved as 'config' (Hydra style)
            elif 'config' in ckpt:
                 model_cfg = ckpt['config'].get('model')

            if model_cfg is None:
                print("[ProxyForward] Could not find 'model' config in checkpoint. Using Identity.")
                return nn.Identity()

            # 2. Instantiate Model
            # Ensure it's OmegaConf for instantiate
            if isinstance(model_cfg, dict):
                model_cfg = OmegaConf.create(model_cfg)
            
            model = instantiate(model_cfg)
            
            # 3. Load Weights
            model.load_state_dict(ckpt['model'])
            print("[ProxyForward] Model loaded successfully (state_dict).")
            return model
            
        except Exception as e:
            print(f"[ProxyForward] Failed to instantiate/load model: {e}")
            return nn.Identity()

    @torch.no_grad()
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args: mask [B, 1, H, W]
        Returns: light_dist [B, 1, H, W]
        """
        x = mask.to(self.device)
        return self.model(x)