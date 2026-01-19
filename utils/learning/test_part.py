# utils/learning/test_part.py

import numpy as np
import torch

from tqdm import tqdm  # 추가
from utils.data.load_data import create_data_loaders
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
import os
import cv2

def _sigmoid_to_u8(logits: torch.Tensor) -> np.ndarray:
    probs = torch.sigmoid(logits).detach().cpu().numpy()  # [B,1,H,W]
    probs = np.clip(probs, 0.0, 1.0)
    u8 = (probs * 255.0).astype(np.uint8)
    return u8


def inference(args, model, data_loader, out_dir: Path):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # 진행률 표시를 위한 tqdm 래퍼
        
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Infer", ncols=70, leave=False, total=len(data_loader)):
            x, y, meta = batch
            x = x.cuda(non_blocking=True)
            logits = model(x)  # [B,1,H,W]
            pred_u8 = _sigmoid_to_u8(logits)

            # save each sample
            B = pred_u8.shape[0]
            for i in range(B):
                sample_key = meta["sample_key"][i] if isinstance(meta["sample_key"], (list, tuple)) else meta["sample_key"]
                out_path = out_dir / f"{sample_key}.png"
                if cv2 is None:
                    raise RuntimeError("opencv-python required to save png in inference().")
                cv2.imwrite(str(out_path), pred_u8[i, 0])


def run_inference(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # 모델 설정은 args.model dict (_target_ + 파라미터) 기준으로 instantiate
    model_cfg = getattr(args, "model")
    model = instantiate(OmegaConf.create(model_cfg))
    model.to(device=device)

    ckpt_path = args.exp_dir / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"[Inference] loaded: {ckpt_path} (epoch={checkpoint.get('epoch', -1)})")

    # test split loader
    test_loader = create_data_loaders(args=args, split="test", shuffle=False, is_train=False)
    out_dir = Path(args.val_loss_dir) / "inference_test"
    inference(args, model, test_loader, out_dir=out_dir)

    del test_loader
    torch.cuda.empty_cache()