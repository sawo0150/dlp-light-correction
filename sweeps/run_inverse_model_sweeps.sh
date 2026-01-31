#!/bin/bash
# sweeps/dlp_models_comparasion.sh
# =================================================================
# Task1: th -> gray mask
# Task2: th -> LD -> gray mask
# Task3: Mask -> corrected mask (forwardLoss)
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
    exp_name=inverse_gray_1_BCEL1 \
    wandb.project=dlp_models_comparasion \
    \
    image.binarize_target=false \
    \
    num_epochs=4 \
    training_accum_steps=4 \
    batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.doc.enable=false \
    evaluation.benchmark.bandopt.enable=false \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false


python main.py \
    task=inverse_1_gray \
    LossFunction=SigmoidL1Loss \
    exp_name=inverse_gray_1_SigmoidL1 \
    wandb.project=dlp_models_comparasion \
    \
    image.binarize_target=false \
    \
    num_epochs=4 \
    training_accum_steps=4 \
    batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.doc.enable=false \
    evaluation.benchmark.bandopt.enable=false \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false

python main.py \
    task=inverse_chain_2_gray \
    LossFunction=BCEWithLogitsL1Loss \
    exp_name=inverse_gray_chain_2_BCEL1 \
    wandb.project=dlp_models_comparasion \
    \
    image.binarize_target=false \
    \
    num_epochs=4 \
    training_accum_steps=4 \
    batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.doc.enable=false \
    evaluation.benchmark.bandopt.enable=false \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false

python main.py \
    task=inverse_chain_2_gray \
    LossFunction=SigmoidL1Loss \
    exp_name=inverse_gray_chain_2_SigmoidL1  \
    wandb.project=dlp_models_comparasion \
    \
    image.binarize_target=false \
    \
    num_epochs=4 \
    training_accum_steps=4 \
    batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.doc.enable=false \
    evaluation.benchmark.bandopt.enable=false \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false

python main.py \
    task=inverse_physics_binary \
    model=unet_small \
    LossFunction=physics_informedL1 \
    LossFunction.doc_enable=false \
    exp_name=inverse_forwardLoss \
    wandb.project=dlp_models_comparasion \
    \
    image.binarize_target=true \
    \
    num_epochs=10 \
    training_amp=true \
    training_accum_steps=4 \
    batch_size=2 \
    val_batch_size=2 \
    \
    evaluation.benchmark.enable=true \
    evaluation.benchmark.doc.enable=false \
    evaluation.benchmark.bandopt.enable=true \
    evaluation.benchmark.inverse_post.binarize=false \
    hydra.job.chdir=false