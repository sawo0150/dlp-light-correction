#!/bin/bash
# trainingScripts/run_forward.sh
# =================================================================
# DLP Forward Model Training Script (Mask -> LD)
# 
# [설명]
# train.yaml을 수정하지 않고, Hydra CLI Override 기능을 사용하여
# Forward Task용 설정으로 학습을 시작합니다.
# =================================================================

# 1. 사용할 GPU 번호 지정
export CUDA_VISIBLE_DEVICES=0

# 2. Python 실행
# - task=forward_1           : configs/task/forward_1.yaml 로드
# - LossFunction=SigmoidL1Loss      : configs/LossFunction/L1Loss.yaml 로드 (없으면 아래 설명 참조)
# - image.binarize_target=false : LD는 0~1 사이의 실수값이므로 이진화 끔
# - exp_name=...             : 결과 저장 폴더명 변경

python main.py \
    task=forward_1 \
    LossFunction=SigmoidL1Loss \
    exp_name=dlp_forward_mask2ld_baseline_v0 \
    image.binarize_target=false \
    training_accum_steps=4 \
    batch_size=1 \
    hydra.job.chdir=false

# Tip: 로그를 파일로 남기며 백그라운드에서 돌리려면 아래 주석 해제 후 사용
# nohup python main.py ... (위 옵션들) ... > forward_log.txt 2>&1 &