#!/bin/bash

# set -e: 스크립트 실행 중 에러가 발생하면 즉시 중단 (불필요한 시간 낭비 방지)
set -e

echo "=========================================="
echo "1. Physics Informed Loss 시작"
echo "=========================================="
bash trainingScripts/run_inverse_1_forwardLoss.sh

echo "=========================================="
echo "2. Binary Baseline 시작"
echo "=========================================="
bash trainingScripts/run_inverse_1_binary.sh

echo "=========================================="
echo "3. Gray Baseline 시작"
echo "=========================================="
bash trainingScripts/run_inverse_1_gray.sh

echo "=========================================="
echo "모든 실험이 완료되었습니다."
echo "=========================================="