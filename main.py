# main.py

"""
Hydra + W&B 진입 스크립트
"""
from omegaconf import DictConfig, OmegaConf
import math, operator
# 안전한 계산용 최소 환경만 넘겨 줌 (※ __builtins__ 비우기)
OmegaConf.register_new_resolver(
    "calc",
    lambda expr: eval(expr, {"__builtins__": {}, "math": math, "operator": operator.__dict__})
)
import hydra
import torch
import sys
import wandb
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

# repo 내부 모듈 임포트 경로 확보 (train.py 방식과 동일) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

from utils.learning.train_part import train      # 기존 학습 루프
from utils.common.utils import seed_fix          # seed 고정 함수

# ───────────────────────────────────────────────────────────────────────────────

def _flatten_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    """
    ✔️  *재귀적으로* 모든 항목을 평탄화하여 SimpleNamespace 로 변환한다.
       - model / data 서브트리는 “옛 argparser 스타일” 유지 → prefix 미부여
       - 그 외 서브트리는 `<tree>_<leaf>` 형식으로 충돌 방지
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace()

    def recurse(prefix: str, node: Mapping):
        for k, v in node.items():
            new_key = f"{prefix}_{k}" if prefix else k

            # ✅ DLP 최소 구성: train.yaml에 남긴 것만 preserve
            PRESERVE = {"task", "model", "data", "LossFunction", "optimizer", 
                        "LRscheduler", "image", "filters", "wandb"}
 

            # (1) 보존용 속성 먼저 세팅
            if prefix == "" and k in PRESERVE and isinstance(v, Mapping):
                # args.LRscheduler = {...}  처럼 딕셔너리 그대로 유지
                setattr(args, k, v)
                # (2) 동시에 하위 키도 평탄화해 주기
                recurse("", v) if k in {"model", "data"} else recurse(k, v)
            elif isinstance(v, Mapping):
                recurse(new_key, v)        # 일반 dict: prefix 유지
            else:
                setattr(args, new_key, v)

    recurse("", container)

    # ✅ 최소 필드만 alias로 세팅
    args.GPU_NUM = container.get("GPU_NUM", 0)
    args.use_wandb = container.get("wandb", {}).get("use_wandb", False)
    args.max_vis_per_cat = container.get("wandb", {}).get("max_vis_per_cat", 0)

    # ─────────────────────────────────────────────────────────────
    # [FIX] 변수명 매핑 추가
    # main.py는 'trainpack_root'로 만들었지만, load_data.py는 'data_trainpack_root'를 찾음
    # ─────────────────────────────────────────────────────────────
    if hasattr(args, "trainpack_root") and not hasattr(args, "data_trainpack_root"):
        args.data_trainpack_root = args.trainpack_root
    
    if hasattr(args, "manifest_csv") and not hasattr(args, "data_manifest_csv"):
        args.data_manifest_csv = args.manifest_csv
        
    if hasattr(args, "splits_dir") and not hasattr(args, "data_splits_dir"):
        args.data_splits_dir = args.splits_dir
        
    # ─────────────────────────────────────────────────────────────
    # ✅ DLP TrainPack 경로도 Path로 변환 (Dataset에서 join할 때 편하게)
    if hasattr(args, "data_trainpack_root"):
        args.data_trainpack_root = Path(args.data_trainpack_root)
    if hasattr(args, "data_manifest_csv"):
        args.data_manifest_csv = Path(args.data_manifest_csv)
    if hasattr(args, "data_splits_dir"):
        args.data_splits_dir = Path(args.data_splits_dir)

    # ✅ 결과 경로 세팅 (data.PROJECT_ROOT가 없으면 현재 작업 디렉토리 사용)
    project_root = Path(getattr(cfg, "data", {}).get("PROJECT_ROOT", Path.cwd()))
    result_dir = project_root / "result" / args.exp_name

    args.exp_dir = result_dir / "checkpoints"
    args.val_dir = result_dir / "reconstructions_val"
    args.main_dir = result_dir / Path(__file__).name
    args.val_loss_dir = result_dir
    for p in [args.exp_dir, args.val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return args

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # ── 1. reproducibility ────────────────────────────────────────────────────
    if cfg.seed is not None:
        seed_fix(cfg.seed)

    # ── 2. cfg → args 변환 -----------------------------------------------------
    args = _flatten_cfg_to_args(cfg)

    # ── 3. W&B 초기화 (조건부) -------------------------------------------------
    if args.use_wandb and wandb is not None:
        wandb_cfg = getattr(cfg, "wandb", None)
        
        # ✅ [Log Path Fix] wandb 로그가 result 폴더 안에 쌓이도록 설정
        wandb_dir = args.val_loss_dir / "wandb_logs"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=args.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
 
    # ── 4. 학습 (task 라우팅) -------------------------------------------------
    task_name = getattr(cfg, "task", {}).get("name", "train")
    print(f"[Task] {task_name}")

    # 현재는 train_part.train 하나로 진입
    train(args)

    # ── 5. 마무리 -------------------------------------------------------------
    if args.use_wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
