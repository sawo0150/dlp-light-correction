# utils/learning/train_part.py

import shutil
import numpy as np
import torch
import torch.nn as nn
import time, math
from pathlib import Path
from typing import Dict, Optional
from importlib import import_module
import inspect
from typing import Any

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

import os
#---------------------------------------------------------#
# 1. ❶ 메모리 단편화 완화용 CUDA allocator 옵션 ― 반드시
#    torch import *이전*에 export 해야 효과가 납니다.  :contentReference[oaicite:0]{index=0}
#---------------------------------------------------------#
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:64,garbage_collection_threshold:0.6"
)

import shutil

from tqdm import tqdm
from hydra.utils import instantiate        
from omegaconf import OmegaConf     
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_  
import torch.nn.functional as F

from utils.data.dataloader_factory import create_data_loaders
from utils.logging.wandb_logger import WandbLogger
from utils.common.utils import compute_segmentation_metrics, compute_regression_metrics
from utils.evaluation.benchmark_dataset import BenchmarkDataset
from utils.evaluation.proxy_forward import ProxyForwardModel
from utils.evaluation.reporter import BenchmarkReporter


# (선택) metric logger는 간단히 숫자만 찍도록 축소

def _get_task_name_from_args(args) -> str:
    task_cfg = getattr(args, "task", {}) or {}
    if isinstance(task_cfg, dict):
        return str(task_cfg.get("name", "")).strip()
    return str(task_cfg)


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler,
                loss_fn,
                scaler, amp_enabled, accum_steps,
                logger: WandbLogger,
                global_iter_start: int = 0):

    model.train()
    # reset peak memory counter at the start of each epoch
    torch.cuda.reset_peak_memory_stats()

    # start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    grad_clip_enabled  = getattr(args, "training_grad_clip_enable", False)
    grad_clip_max_norm = getattr(args, "training_grad_clip_max_norm", 1.0)
    grad_clip_norm_t   = getattr(args, "training_grad_clip_norm_type", 2)

    pbar = tqdm(enumerate(data_loader),
                total=len_loader,
                dynamic_ncols=True,   # ✅ 터미널 폭에 맞춰 자동 확장
                leave=False,
                desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
    task_name = _get_task_name_from_args(args)
    for iter, data in pbar:
        # DLP: (x, y, meta)
        x, y, meta = data
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)        

        with autocast(enabled=amp_enabled):
            logits = model(x)

        # ✅ [VRAM 로깅 추가] report_interval 마다 VRAM 사용량 출력
        if iter > 0 and iter % args.report_interval == 0:
            torch.cuda.synchronize() # 정확한 측정을 위한 동기화
            vram_alloc = torch.cuda.memory_allocated() / 1024**2
            vram_peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\n  [VRAM at iter {iter}] Allocated: {vram_alloc:.2f} MB | Peak: {vram_peak:.2f} MB")
            
        # loss_fn은 (logits, target) 형태를 가정
        loss = loss_fn(logits, y) / accum_steps

        # print("max alloc MB:", torch.cuda.max_memory_allocated() / 1024**2)
        # print("max memory_reserved MB:", torch.cuda.torch.cuda.max_memory_reserved() / 1024**2)
        # ─── Accumulation ──────────────────────────────────────────────
        if iter % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True) # [수정] DeepSpeed 분기 제거

        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # step & update?
        if (iter + 1) % accum_steps == 0 or (iter + 1) == len_loader:
            if amp_enabled:
                # # unscale / clip_grad 등 필요 시 여기서
                # scaler.step(optimizer)
                # scaler.update()
                
                scaler.unscale_(optimizer)
                if grad_clip_enabled:
                    clip_grad_norm_(model.parameters(),
                                    grad_clip_max_norm,
                                    norm_type=grad_clip_norm_t)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            else:
                if grad_clip_enabled:
                    clip_grad_norm_(model.parameters(),
                                    grad_clip_max_norm,
                                    norm_type=grad_clip_norm_t)
                optimizer.step()

                # free gradient buffers right after the optimizer step
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            #   scheduler.step() 을 이미 호출하므로 별도 호출 금지
            if scheduler is not None:  # [수정] 조건문 단순화
                 if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,
                                           torch.optim.lr_scheduler.CyclicLR)):
                    scheduler.step()

        total_loss += float(loss.detach().item()) * accum_steps
        if task_name.startswith("inverse"):
            m1, m2 = compute_segmentation_metrics(logits.detach(), y.detach())
            postfix = f"loss {total_loss/(iter+1):.4g} | D {m1:.3f} | I {m2:.3f}"
        elif task_name.startswith("forward"):
            mae, rmse, psnr = compute_regression_metrics(logits.detach(), y.detach())
            postfix = f"loss {total_loss/(iter+1):.4g} | MAE {mae:.3f} | RMSE {rmse:.3f} | PSNR {psnr:.2f}"
        else:
            postfix = f"loss {total_loss/(iter+1):.4g}"

        # ✅ postfix를 문자열로 넣으면 잘림이 덜하고, 출력 제어가 쉬움
        pbar.set_postfix_str(postfix)

        # ✅ Train metric: iter 단위 로깅 (N iters마다)
        if logger.enabled:
            wb_cfg = getattr(args, "wandb", {}) if hasattr(args, "wandb") else {}
            log_every = int(wb_cfg.get("log_every_n_iters", 0) or 0) if isinstance(wb_cfg, dict) else 0
            if log_every and (iter > 0) and (iter % log_every == 0):
                lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
                # inverse/forward 공통으로 loss는 기록하고, 나머지는 task별로 다르게
                if task_name.startswith("inverse"):
                    dice, iou = compute_segmentation_metrics(logits.detach(), y.detach())
                    logger.log_train_iter(
                        step=global_iter_start + iter,
                        epoch=epoch,
                        it=iter,
                        loss=float(loss.detach().item() * accum_steps),
                        dice=float(dice),
                        iou=float(iou),
                        lr=float(lr),
                    )
                elif task_name.startswith("forward"):
                    mae, rmse, psnr = compute_regression_metrics(logits.detach(), y.detach())
                    # wandb_logger에 forward용 키가 없을 수도 있으니, 최소 호환으로 dice/iou 자리에 mae/rmse를 넣어도 되고
                    # (권장) wandb_logger 쪽을 확장하는 게 맞음. 일단 안전하게 meta만 남기고 loss 위주로 기록.
                    logger.log_train_iter(
                        step=global_iter_start + iter,
                        epoch=epoch,
                        it=iter,
                        loss=float(loss.detach().item() * accum_steps),
                        dice=float(mae),   # backward compat slot
                        iou=float(rmse),    # backward compat slot
                        lr=float(lr),
                    )
                else:
                    logger.log_train_iter(
                        step=global_iter_start + iter,
                        epoch=epoch,
                        it=iter,
                        loss=float(loss.detach().item() * accum_steps),
                        dice=0.0,
                        iou=0.0,
                        lr=float(lr),
                    )

        del logits, loss
        torch.cuda.empty_cache()
        
    epoch_time = time.perf_counter() - start_iter
    return total_loss / max(1, len_loader), epoch_time


def validate(args, model, data_loader, epoch, loss_fn):
    model.eval()
    start = time.perf_counter()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    n_batches  = 0
    task_name = _get_task_name_from_args(args)
    total_m1 = 0.0
    total_m2 = 0.0
    total_m3 = 0.0
    
    len_loader = len(data_loader)                 # ← 전체 길이
    pbar = tqdm(enumerate(data_loader),           # ← tqdm 래퍼
                total=len_loader,
                dynamic_ncols=True,   # ✅ 터미널 폭에 맞춰 자동 확장
                leave=False,
                desc=f"Val  [{epoch:2d}/{args.num_epochs}]")
    
    with torch.no_grad():
        for idx, data in pbar:
            x, y, meta = data
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)
            if task_name.startswith("inverse"):
                m1, m2 = compute_segmentation_metrics(logits, y)  # dice/iou
                total_m1 += float(m1)
                total_m2 += float(m2)
            elif task_name.startswith("forward"):
                mae, rmse, psnr = compute_regression_metrics(logits, y)
                total_m1 += float(mae)
                total_m2 += float(rmse)
                total_m3 += float(psnr)

            total_loss += float(loss.item())
            n_batches += 1

            avg_loss = total_loss / n_batches
            if task_name.startswith("inverse"):
                avg_dice = total_m1 / n_batches
                avg_iou  = total_m2 / n_batches
                pbar.set_postfix_str(f"loss {avg_loss:.4g} | D {avg_dice:.3f} | I {avg_iou:.3f}")
            elif task_name.startswith("forward"):
                avg_mae  = total_m1 / n_batches
                avg_rmse = total_m2 / n_batches
                avg_psnr = total_m3 / n_batches
                pbar.set_postfix_str(f"loss {avg_loss:.4g} | MAE {avg_mae:.3f} | RMSE {avg_rmse:.3f} | PSNR {avg_psnr:.2f}")
            else:
                pbar.set_postfix_str(f"loss {avg_loss:.4g}")

        torch.cuda.empty_cache()

    metric_loss = total_loss / max(1, n_batches)
    if task_name.startswith("inverse"):
        metric_m1 = total_m1 / max(1, n_batches)  # dice
        metric_m2 = total_m2 / max(1, n_batches)  # iou
        metric_m3 = 0.0
    elif task_name.startswith("forward"):
        metric_m1 = total_m1 / max(1, n_batches)  # mae
        metric_m2 = total_m2 / max(1, n_batches)  # rmse
        metric_m3 = total_m3 / max(1, n_batches)  # psnr
    else:
        metric_m1 = 0.0
        metric_m2 = 0.0
        metric_m3 = 0.0

    return metric_loss, metric_m1, metric_m2, metric_m3, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    
    checkpoint = {
        'epoch':         epoch,
        'args':          args,                              # ← SimpleNamespace 통째로
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir':       str(exp_dir),                      # Path는 문자열로 저장해도 OK
    }
    torch.save(checkpoint, exp_dir / 'model.pt')
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    print('Current cuda device: ', torch.cuda.current_device())
    # DLP에서는 maskDuplicate/maskAugment/collator/compressor/sampler 등 MRI 전용 기능 제거
    dup_mul = 1

    # ✅ Logger (W&B)
    logger = WandbLogger(args)

    # ▸ 0. 옵션 파싱 (기본값 유지)
    # ───────── Gradient Accumulation 기본값 및 스케줄러 설정 ─────────
    accum_steps_default = getattr(args, "training_accum_steps", 1)
    ga_sched_cfg    = getattr(args, "training_grad_accum_scheduler", {"enable": False})
    ga_sched_enable = ga_sched_cfg.get("enable", False)
    ga_milestones   = sorted(ga_sched_cfg.get("milestones", []), key=lambda x: x.get("epoch", 0))

    def _accum_steps_for_epoch(ep: int) -> int:
        """현재 epoch에서 사용할 accum_steps 반환."""
        if not ga_sched_enable or not ga_milestones:
            return accum_steps_default
        curr = accum_steps_default
        for m in ga_milestones:          # epoch 오름차순
            if ep >= m.get("epoch", 0):
                curr = m.get("steps", curr)
            else:
                break
        return max(1, int(curr))

    accum_steps = _accum_steps_for_epoch(0)   # 첫 epoch 기준 초기화

    checkpointing = getattr(args, "training_checkpointing", False)
    amp_enabled   = getattr(args, "training_amp",           False)
    print(f"[Hydra] training: accum_steps={accum_steps} "
        f"checkpointing={checkpointing} amp_enabled={amp_enabled}")
    if ga_sched_enable:
        print(f"[Hydra] grad_accum_scheduler 활성화 → milestones={ga_milestones}")

    # ⬇︎ 추가 ─────────────────────────────────────────────
    grad_clip_enabled  = getattr(args, "training_grad_clip_enable", False)
    grad_clip_max_norm = getattr(args, "training_grad_clip_max_norm", 1.0)
    grad_clip_norm_t   = getattr(args, "training_grad_clip_norm_type", 2)
    print(f"[Hydra] grad_clip: enable={grad_clip_enabled} "
        f"max_norm={grad_clip_max_norm} norm_type={grad_clip_norm_t}")
    # ──────────────────────────────────────────────────


    early_cfg = getattr(args, "early_stop", {})
    early_enabled = early_cfg.get("enable", False)
    stage_table = {s["epoch"]: s["ssim"] for s in early_cfg.get("stages", [])}
    # ex) {10:0.90, 20:0.95, 25:0.96}
    print(f"[Hydra-eval] {early_cfg}")
    print(f"[Hydra-eval] early_enabled={early_enabled}, stage_table={stage_table}")


    # instantiate(cfg.model) 로 모든 모델 파라미터 주입,
    # use_checkpoint 은 training.checkpointing 에 따름
    model_cfg = getattr(args, "model")
    model = instantiate(OmegaConf.create(model_cfg))
    model.to(device)
    print(f"[Hydra-model] model_cfg={model_cfg}")

    loss_cfg = getattr(args, "LossFunction")
    loss_fn = instantiate(OmegaConf.create(loss_cfg)).to(device=device)
    print(f"[Hydra] loss_func ▶ {loss_fn}")

    # ──────────────────────────────────────────────────────────────────────────
    # ✅ [FIX] Optimizer 초기화 (UnboundLocalError 해결)
    # 스케줄러 생성 전에 반드시 Optimizer가 있어야 합니다.
    # ──────────────────────────────────────────────────────────────────────────
    optim_cfg = getattr(args, "optimizer")
    # Hydra instantiate로 생성 (params 인자로 모델 파라미터 전달)
    optimizer = instantiate(OmegaConf.create(optim_cfg), params=model.parameters())
    print(f"[Hydra] optimizer ▶ {optimizer.__class__.__name__}")

    # ──────────────── LR Scheduler (옵션) ────────────────
    # [수정] DeepSpeed 분기 제거하고 바로 Scheduler 생성 로직 수행
    
    # 0) Config → OmegaConf
    sched_cfg_raw = getattr(args, "LRscheduler", None)
    scheduler = None
    if sched_cfg_raw is not None:
        sched_cfg = OmegaConf.create(sched_cfg_raw)   # dict → OmegaConf

        print("[Info] Skip calculating exact effective_steps to save time.")
        effective_steps = 1000 

        # 2) target class 로드 & 시그니처 분석
        target_path = sched_cfg["_target_"]
        mod_name, cls_name = target_path.rsplit(".", 1)
        SchedulerCls = getattr(import_module(mod_name), cls_name)
        sig = inspect.signature(SchedulerCls.__init__)
        valid_keys = set(sig.parameters.keys())       

        # 3) 필요한 key만 conditionally 추가
        if "effective_steps" in sched_cfg and "effective_steps" not in valid_keys:
            del sched_cfg["effective_steps"]

        # CyclicLR
        if cls_name == "CyclicLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["step_size_up"] = sched_cfg.get(
                "step_size_up", effective_steps * 2
            )
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 6)

        # OneCycleLR
        if cls_name == "OneCycleLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["total_steps"] = sched_cfg.get(
                "total_steps", effective_steps * args.num_epochs
            )
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 10)

        # 4) instantiate (optimizer 전달)
        clean_dict = {k: v for k, v in sched_cfg.items() if k in valid_keys or k.startswith("_")}
        scheduler = instantiate(OmegaConf.create(clean_dict), optimizer=optimizer)
        print(f"[Hydra] Scheduler ▶ {scheduler}")

    # DLP inverse MVP에서는 augmenter/MaskAugmenter 기능은 일단 제거 (필요하면 나중에 DLP용으로 다시 추가)
    augmenter = None
    mask_augmenter = None

    # ✅ [Bug Fix] Initialize val_loss_history BEFORE loading check
    val_loss_history = []
    
    # ── Resume logic (이제 model, optimizer가 정의된 이후) ──
    start_epoch   = 0
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    best_val_score = -float('inf')  # inverse: dice 최대, forward: -val_loss 최대(=val_loss 최소)

    print(f"[Resume] {getattr(args, 'resume_checkpoint', None)}")
    if getattr(args, 'resume_checkpoint', None):
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_ssim = ckpt.get('best_val_ssim', 0.0)
        start_epoch = ckpt.get('epoch', 0)
        val_loss_history = ckpt.get('val_loss_history', []) # 체크포인트에서 기록 복원
        print(f"[Resume] Loaded '{args.resume_checkpoint}' → epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        # 재개 시, augmenter의 상태도 복원
        # --- resume 영역 ---
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if augmenter:
            augmenter.val_loss_history.clear()
            augmenter.val_loss_history.extend(val_loss_history)
            print(f"[Resume] Augmenter val_loss history 복원 완료 ({len(val_loss_history)}개 항목)")
        if mask_augmenter:
            mask_augmenter.val_hist.clear()
            mask_augmenter.val_hist.extend(val_loss_history)
            print(f"[Resume] MaskAugmenter val_loss history 복원 완료 ({len(val_loss_history)}개 항목)")


    # ▸ 2. AMP scaler (옵션)
    scaler = GradScaler(enabled=amp_enabled)
    # Resume 시 GradScaler 상태 복원
    if getattr(args, 'resume_checkpoint', None) and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
        print(f"[Resume] Loaded GradScaler state")

    print("[Data] trainpack_root:", getattr(args, "data_trainpack_root", None))
    print("[Data] manifest_csv  :", getattr(args, "data_manifest_csv", None))

    # ───────────────── evaluation 서브트리 언랩 ─────────────────
    # ①  args.evaluation 이 이미 {"enable": …} 형태라면 그대로 사용
    # ②  args.evaluation 이 {"evaluation": {...}} 처럼 한 번 더
    #     래핑돼 있으면 내부 dict 를 꺼낸다.
    _raw_eval = getattr(args, "evaluation", {})
    eval_cfg  = _raw_eval.get("evaluation", _raw_eval)

    lb_enable = eval_cfg.get("enable", False)
    lb_every  = eval_cfg.get("every_n_epochs", 999_999)   # 기본 매우 크게

    print(f"[Hydra-eval] {eval_cfg}")
    print(f"[Hydra-eval] lb_enable={lb_enable}, lb_every={lb_every}")

    # ──────────────────────────────────────────────────────────────────────────
    # ✅ [Benchmark Init] 벤치마크 리포터 초기화
    # ──────────────────────────────────────────────────────────────────────────
    bench_cfg = eval_cfg.get("benchmark", {})
    bench_enabled = bench_cfg.get("enable", False)
    benchmark_reporter = None

    if bench_enabled:
        print(f"[Benchmark] Initializing Reporter... (Proxy: {bench_cfg.get('proxy_checkpoint')})")

        # 1. Dataset & Proxy Load
        bench_loader = BenchmarkDataset(root=bench_cfg.get("data_root", "data/benchmark_160"))
        proxy_model = ProxyForwardModel(ckpt_path=bench_cfg.get("proxy_checkpoint", ""), device=device)
        # 1.5. Post-process configs (robust parsing)
        inv_post: Dict[str, Any] = bench_cfg.get("inverse_post", {}) or {}
        curing_cfg: Dict[str, Any] = bench_cfg.get("curing", {}) or {}
        fwd_post: Dict[str, Any] = bench_cfg.get("forward_post", {}) or {}
        fwd_apply_sigmoid = bool(fwd_post.get("apply_sigmoid", False))

        inv_apply_sigmoid = bool(inv_post.get("apply_sigmoid", True))
        inv_binarize      = bool(inv_post.get("binarize", True))
        inv_bin_thr       = float(inv_post.get("binarize_thr", 0.5))

        curing_thr        = float(curing_cfg.get("threshold", 0.5))
        curing_binarize   = bool(curing_cfg.get("binarize", True))

        print(
            f"[Benchmark] inverse_post: apply_sigmoid={inv_apply_sigmoid}, "
            f"binarize={inv_binarize}, thr={inv_bin_thr} | "
            f"curing: thr={curing_thr}, binarize={curing_binarize}"
        )

        # 2. Model Input Size Resolution (✅ forward 학습 out_size를 우선 사용)
        task_cfg = getattr(args, "task", {}) or {}
        fwd_cfg = task_cfg.get("forward", {}) if isinstance(task_cfg, dict) else {}
        fwd_pre = (fwd_cfg.get("preprocess", {}) or {}) if isinstance(fwd_cfg, dict) else {}
        model_in_size = int(fwd_pre.get("out_size", 640))  # Default 640
        fwd_binarize_input = bool((fwd_cfg or {}).get("binarize_input", False))

        # 3. Reporter Init
        benchmark_reporter = BenchmarkReporter(
            inverse_model=model,
            forward_model=proxy_model,
            device=device,
            image_size=640,          # Visualization Size (Canvas)
            model_input_size=model_in_size,
            inverse_apply_sigmoid=inv_apply_sigmoid,
            inverse_binarize=inv_binarize,
            inverse_binarize_thr=inv_bin_thr,
            curing_threshold=curing_thr,
            curing_binarize=curing_binarize,
            forward_binarize_input=fwd_binarize_input,
            forward_apply_sigmoid=fwd_apply_sigmoid,
        )
        bench_freq = int(bench_cfg.get("log_every_n_epochs", 1))


    val_loader = create_data_loaders(args=args, split="val", shuffle=False, is_train=False)
    task_name = _get_task_name_from_args(args)
    global_iter = 0

    # ▲ Resume 시 기존 val_loss_log를 불러와 이어서 기록
    val_loss_log_file = os.path.join(args.val_loss_dir, "val_loss_log.npy")
    if getattr(args, 'resume_checkpoint', None) and os.path.exists(val_loss_log_file):
        val_loss_log = np.load(val_loss_log_file)
        print(f"[Resume] 기존 val_loss_log 불러옴, shape={val_loss_log.shape}")
    else:
        val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.exp_name} ...............')
        torch.cuda.empty_cache()
 
        train_loader = create_data_loaders(args=args, split="train", shuffle=True, is_train=True)
 

        # ── epoch별 accum_steps 갱신 ──
        accum_steps_epoch = _accum_steps_for_epoch(epoch)
        if accum_steps_epoch != accum_steps:
            print(f"[GradAccum] Epoch {epoch}: accum_steps {accum_steps} → {accum_steps_epoch}")
            accum_steps = accum_steps_epoch

        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, scheduler,
            loss_fn, scaler, amp_enabled, accum_steps,
            logger=logger,
            global_iter_start=global_iter
         )
        global_iter += len(train_loader)    # ✅ 이 시점의 global_iter가 "epoch 끝 step"
        
        val_loss, val_m1, val_m2, val_m3, val_time = validate(args, model, val_loader, epoch, loss_fn)

        # ✨ val_loss 기록 (스케줄러 및 체크포인트용)
        val_loss_history.append(val_loss)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        # num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        # inverse에서는 dice 기준으로 best 선택 (원하면 iou/val_loss로 변경)
        if task_name.startswith("inverse"):
            # val_m1 = dice
            score = float(val_m1)
            is_new_best = score > best_val_score
            best_val_score = max(best_val_score, score)
            best_val_loss = min(best_val_loss, val_loss)
        elif task_name.startswith("forward"):
            # forward는 loss 최소화
            score = -float(val_loss)
            is_new_best = score > best_val_score
            best_val_score = max(best_val_score, score)
            best_val_loss = min(best_val_loss, val_loss)
        else:
            is_new_best = False

        # ✨ save_model에 val_loss_history 추가 (상태 저장을 위해)
        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            'scaler': scaler.state_dict(),          
            'best_val_score': best_val_score,
            'best_val_loss': best_val_loss,
            'exp_dir': str(args.exp_dir),
            'val_loss_history': val_loss_history 
        }
        torch.save(checkpoint, args.exp_dir / 'model.pt')

        if is_new_best:
            shutil.copyfile(args.exp_dir / 'model.pt', args.exp_dir / 'best_model.pt')
        
        # ──────────── LR 스케줄러 업데이트 ────────────
        if scheduler is not None:
            # ReduceLROnPlateau 는 val_metric 이 필요함
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)      # 여기서는 “낮을수록 좋음” metric
            else:
                scheduler.step()

        # ✅ Val/Test metric: epoch 단위 로깅
        if logger.enabled:
            lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
            # backward compat: val_dice/val_iou 슬롯을 task별로 재사용
            if task_name.startswith("inverse"):
                v1, v2 = float(val_m1), float(val_m2)  # dice/iou
            elif task_name.startswith("forward"):
                v1, v2 = float(val_m1), float(val_m2)  # mae/rmse
            else:
                v1, v2 = 0.0, 0.0

            logger.log_epoch_metrics(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                val_dice=v1,
                val_iou=v2,
                lr=float(lr),
                step=global_iter,
            )

        # ✅ Epoch 이미지 로깅 (train/val 선택)
        if logger.enabled:
            wb_cfg = getattr(args, "wandb", {}) if hasattr(args, "wandb") else {}
            split_name = str(wb_cfg.get("log_images_split", "val")).lower() if isinstance(wb_cfg, dict) else "val"
            if split_name == "train":
                logger.log_epoch_images(model=model, loader=train_loader, epoch=epoch, split_name="train", device=device, step=global_iter)
            else:
                logger.log_epoch_images(model=model, loader=val_loader, epoch=epoch, split_name="val", device=device, step=global_iter)

        # ──────────────────────────────────────────────────────────────────────
        # ✅ [Benchmark Report] 벤치마크 리포트 생성 및 W&B 전송
        # ──────────────────────────────────────────────────────────────────────
        if bench_enabled and benchmark_reporter and ((epoch + 1) % bench_freq == 0):
            print(f"[Benchmark] Generating Visual Report (Epoch {epoch})...")
            # Report 생성 (Dict[str, wandb.Image])
            report_imgs = benchmark_reporter.generate_report(bench_loader)
            
            # W&B Logging
            if logger.enabled and wandb.run is not None:
                wandb.log(report_imgs, step=global_iter)

        # best 모델 저장은 기존 로직 유지 (wandb artifact는 main/train에서 선택적으로 추가 가능)
        if getattr(args, "use_wandb", False) and (wandb is not None) and (getattr(wandb, "run", None) is not None):
            if is_new_best:
                wandb.save(str(args.exp_dir / "best_model.pt"))
                               
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g} '
            f'ValM1 = {val_m1:.4g} ValM2 = {val_m2:.4g} '
            f'TrainTime = {train_time:.2f}s ValTime = {val_time:.2f}s',
        )

        #  (원하면 DLP용 “몇 장 PNG 저장”으로 별도 구현)

        # ────────── epoch 루프 내부, val 계산·로그 이후 ──────────
        current_epoch = epoch + 1         # 사람 눈금 1-base
        # release dataloader workers & cached memory before next epoch
        del train_loader
        torch.cuda.empty_cache()
        if early_enabled and current_epoch in stage_table:
            req = stage_table[current_epoch]
            if val_dice < req:
                print(f"[EarlyStop] Epoch {current_epoch}: "
                    f"val_dice={val_dice:.4f} < target={req:.4f}. 학습 중단!")
                break                     # for epoch 루프 탈출