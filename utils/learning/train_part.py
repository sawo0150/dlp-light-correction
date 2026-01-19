# utils/learning/train_part.py

import shutil
import numpy as np
import torch
import torch.nn as nn
import time, math
from pathlib import Path
from typing import Dict, Optional

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

from utils.data.load_data import create_data_loaders
from utils.logging.metric_accumulator import MetricAccumulator
from utils.logging.vis_logger import log_epoch_samples

# (선택) metric logger는 간단히 숫자만 찍도록 축소

def _dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    logits: [B,1,H,W] (unbounded) or probs
    target: [B,1,H,W] in {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()
 
def train_epoch(args, epoch, model, data_loader, optimizer, scheduler,
                loss_fn,
                scaler, amp_enabled, use_deepspeed, accum_steps):
    # print(sum(p.numel() for p in model.parameters()))
    # from utils.model.feature_varnet.modules import DLKAConvBlock
    # for n,m in model.named_modules():
    #     if isinstance(m, DLKAConvBlock): print("DLKA at", n)
    # for n, p in model.named_parameters(): print(n, p.numel())
    # torch.autograd.set_detect_anomaly(True)

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
                ncols=70,
                leave=False,
                desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
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
            if use_deepspeed:
                model.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

        if use_deepspeed:
            model.backward(loss)
        elif amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # step & update?
        if (iter + 1) % accum_steps == 0 or (iter + 1) == len_loader:
            if use_deepspeed:
                model.step()
                model.zero_grad()
            elif amp_enabled:
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

            # ▶ DeepSpeed 엔진은 model.step() 안에서
            #   scheduler.step() 을 이미 호출하므로 별도 호출 금지
            if (not use_deepspeed) and scheduler is not None:
                if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,
                                          torch.optim.lr_scheduler.CyclicLR)):
                    scheduler.step()

        total_loss += float(loss.detach().item()) * accum_steps
        dice, iou = _dice_iou_from_logits(logits.detach(), y.detach())
        pbar.set_postfix(loss=f"{(total_loss/(iter+1)):.4g}", dice=f"{dice:.3f}", iou=f"{iou:.3f}")

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
    
    len_loader = len(data_loader)                 # ← 전체 길이
    pbar = tqdm(enumerate(data_loader),           # ← tqdm 래퍼
                total=len_loader,
                ncols=70,
                leave=False,
                desc=f"Val  [{epoch:2d}/{args.num_epochs}]")
    
    with torch.no_grad():
        for idx, data in pbar:
            x, y, meta = data
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)
            dice, iou = _dice_iou_from_logits(logits, y)

            total_loss += float(loss.item())
            total_dice += float(dice)
            total_iou  += float(iou)
            n_batches += 1
            pbar.set_postfix(loss=f"{(total_loss/n_batches):.4g}", dice=f"{(total_dice/n_batches):.3f}", iou=f"{(total_iou/n_batches):.3f}")

        torch.cuda.empty_cache()

    metric_loss = total_loss / max(1, n_batches)
    metric_dice = total_dice / max(1, n_batches)
    metric_iou  = total_iou  / max(1, n_batches)
    return metric_loss, metric_dice, metric_iou, time.perf_counter() - start
 

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

    # ── 1. Optimizer 선택 (fallback: Adam) ─────────────────────────────
    # DeepSpeed 활성화 여부를 미리 확인
    ds_cfg = getattr(args, "deepspeed", None)
    use_deepspeed = ds_cfg is not None and ds_cfg.get("enable", False)

    temp_loader = create_data_loaders(args=args, split="train", shuffle=True, is_train=True)

    # ❷ DeepSpeed 스케줄러 파라미터 동적 채우기
    ds_cfg = getattr(args, "deepspeed", None)
    if ds_cfg and ds_cfg["config"].get("scheduler"):
        sched_p = ds_cfg["config"]["scheduler"]["params"]
        sched_p["warmup_num_steps"] = args.warmup_epochs * effective_steps
        sched_p["total_num_steps"]  = args.num_epochs    * effective_steps

    # ──────────────── LR Scheduler (옵션) ────────────────
    # DeepSpeed가 활성화된 경우, DeepSpeed의 scheduler 설정을 사용
    if use_deepspeed:
        # DeepSpeed가 ds_cfg에서 scheduler를 읽어오므로, 여기서는 별도로 instantiate하지 않음
        scheduler = None # DeepSpeed가 자체적으로 스케줄러를 관리하도록 None으로 설정
        # ds_cfg["config"]["scheduler"] 블록을 직접 사용할 것임
    else:
        # 0) Config → OmegaConf
        sched_cfg_raw = getattr(args, "LRscheduler", None)
        scheduler = None
        if sched_cfg_raw is not None:
            sched_cfg = OmegaConf.create(sched_cfg_raw)   # dict → OmegaConf

            # 1) 공통 계산
            temp_loader = create_data_loaders(
                data_path=args.data_path_train,
                args=args,
                shuffle=True,
                augmenter=None,
                mask_augmenter=None,
                is_train=True,      # ★ train만 True
                domain_filter=getattr(args, "domain_filter", None),
            )
            effective_steps = math.ceil(len(temp_loader) / accum_steps)
            del temp_loader

            # 2) target class 로드 & 시그니처 분석
            target_path = sched_cfg["_target_"]
            mod_name, cls_name = target_path.rsplit(".", 1)
            SchedulerCls = getattr(import_module(mod_name), cls_name)
            sig = inspect.signature(SchedulerCls.__init__)
            valid_keys = set(sig.parameters.keys())        # 허용 인수 목록

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

            # 4) instantiate (불필요 인수 제거 후)
            clean_dict = {k: v for k, v in sched_cfg.items() if k in valid_keys or k.startswith("_")}
            scheduler = instantiate(OmegaConf.create(clean_dict), optimizer=optimizer)
            print(f"[Hydra] Scheduler ▶ {scheduler}")

    # ── ❷ DeepSpeed 옵션 적용 ────────────────────────────────
    print("[DeepSpeed] use_deepspeed",use_deepspeed)
    print(ds_cfg)
    if use_deepspeed:
        import deepspeed                                       # :contentReference[oaicite:3]{index=3}
        model, optimizer, scheduler, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,      
            model_parameters=model.parameters(),
            config=ds_cfg["config"]
        )
        print("[DeepSpeed] enabled → ZeRO-Stage",
              ds_cfg["config"]["zero_optimization"]["stage"])
        print(f"[DeepSpeed] 최종 Optimizer: {optimizer.__class__.__name__}")
        print(f"[DeepSpeed] 최종 LR Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")

    # DLP inverse MVP에서는 augmenter/MaskAugmenter 기능은 일단 제거 (필요하면 나중에 DLP용으로 다시 추가)
    augmenter = None
    mask_augmenter = None

    # ── Resume logic (이제 model, optimizer가 정의된 이후) ──
    start_epoch   = 0
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    val_loss_history = [] # val loss기록 for augmenter

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

    # train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True) # 매 에폭마다 생성 예정
    val_loader = create_data_loaders(args=args, split="val", shuffle=False, is_train=False)
    
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
            loss_fn, scaler, amp_enabled, use_deepspeed, accum_steps
        )
        val_loss, val_dice, val_iou, val_time = validate(args, model, val_loader, epoch, loss_fn)

        # ✨ val_loss 기록 (스케줄러 및 체크포인트용)
        val_loss_history.append(val_loss)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        # inverse에서는 dice 기준으로 best 선택 (원하면 iou/val_loss로 변경)
        is_new_best = val_dice > best_val_ssim
        best_val_loss = min(best_val_loss, val_loss)
        best_val_ssim = max(best_val_ssim, val_dice)

        # ✨ save_model에 val_loss_history 추가 (상태 저장을 위해)
        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            'scaler': scaler.state_dict(),                  # ← 추가
            'best_val_ssim': best_val_ssim,
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

        # ---------------- W&B epoch log (간단) ----------------
        if getattr(args, "use_wandb", False) and wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "val/dice": float(val_dice),
                    "val/iou": float(val_iou),
                    "lr": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
                },
                step=epoch*dup_mul
            )
            if is_new_best:
                wandb.save(str(args.exp_dir / "best_model.pt"))
                                
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g} '
            f'ValDice = {val_dice:.4g} ValIoU = {val_iou:.4g} '
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