#!/bin/bash
# trainingScripts/run_inverse_1_binary.sh
# =================================================================
# DLP Inverse Model Training (Binary Mode)
# Task: Thresholded LD -> Binary Mask
# 
# [설명]
# - task=inverse_1_binary: Binary 데이터셋만 로드하는 설정 사용
# - image.binarize_target=true: Target Mask를 강제로 0 또는 1로 변환하여 로드
# =================================================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    task=inverse_1_binary \
    LossFunction=BCEWithLogitsL1Loss \
    exp_name=dlp_inverse_binary_baseline_v1 \
    wandb.project=dlp_inverse_peoject \
    \
    image.binarize_target=true \
    \
    num_epochs=10 \
    training_accum_steps=4 \
    batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.inverse_post.binarize=true \
    hydra.job.chdir=false