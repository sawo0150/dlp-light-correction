#!/bin/bash
# trainingScripts/run_inverse_1_forwardLoss_1280.sh
# =================================================================
# DLP Inverse Model Training (Physics-Informed Forward Loss)
# Task: Target Mask -> (Band constrained) Corrected Mask -> Forward(Freeze) -> LD -> Soft Curing -> L1(Target)
#
# [핵심]
# - train.yaml "수정" 없이 Hydra override로 설정 주입
# - ProxyForwardModel.forward()는 no_grad 유지 (benchmark용)
# - PhysicsInformedLoss 내부에서는 forward_with_grad() 사용 (학습용)
# =================================================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    task=inverse_physics_binary \
    model=band_unet_small \
    LossFunction=physics_informedL1 \
    exp_name=dlp_inverse_physics_bandopt_1280_v1 \
    wandb.project=dlp_inverse_peoject \
    \
    image.binarize_target=true \
    task.inverse.preprocess.out_size=1280 \
    \
    num_epochs=10 \
    training_amp=true \
    training_accum_steps=8 \
    batch_size=1 \
    val_batch_size=1 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false