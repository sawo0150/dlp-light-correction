#!/bin/bash
# trainingScripts/run_inverse_1_gray.sh
# =================================================================
# DLP Inverse Model Training (Gray Mode)
# Task: Thresholded LD -> Grayscale Mask (Soft Mask)
# 
# [설명]
# - task=inverse_1_gray: Gray 및 Binary 데이터셋 모두 로드
# - image.binarize_target=false: 중요! Target Mask의 0~1 사이 실수값을 유지함
# 
# [Evaluation 옵션 조정]
# - inverse_post.binarize=false: 벤치마크 리포트에서 모델이 예측한 Gray Mask를 
#   이진화하지 않고 그대로 시각화하여 Soft한 값을 잘 예측하는지 확인
# =================================================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    task=inverse_1_gray \
    LossFunction=BCEWithLogitsL1Loss \
    exp_name=dlp_inverse_gray_baseline_1280_v1 \
    wandb.project=dlp_inverse_peoject \
    \
    image.binarize_target=false \
    image.size=1280 \
    task.inverse.preprocess.out_size=1280 \
    \
    num_epochs=4 \
    training_accum_steps=8 \
    batch_size=1 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false