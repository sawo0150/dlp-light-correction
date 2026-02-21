#!/bin/bash
# trainingScripts/run_inverse_2_binary_domain_sweep.sh
# ================================================================
# Sweep DLP inverse_chain_2_binary over multiple (B1~B5) domain mixtures.
#
# - max_samples.train = 10000 (items 기준)
# - domain.enable = true
# - 각 조합을 순차 실행
# ================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PROJECT="dlp_inverse_DataComparasion"
BASE_EXP_PREFIX="dlp_inverse_binary_2_dom"
EPOCHS=4
ACCUM=4
BS=2
MAX_SAMPLES_TRAIN=10000

# 공통 hydra overrides
COMMON_ARGS=(
  "task=inverse_chain_2_binary"
  "LossFunction=BCEWithLogitsL1Loss"
  "wandb.project=${PROJECT}"

  # ✅ binary target
  "image.binarize_target=true"

  "num_epochs=${EPOCHS}"
  "training_accum_steps=${ACCUM}"
  "batch_size=${BS}"
  "hydra.job.chdir=false"

  # ✅ benchmark viz/eval: corrected mask를 binary로 만들어 proxy forward에 넣기
  "evaluation.benchmark.enable=true"
  "evaluation.benchmark.doc.enable=false"
  "evaluation.benchmark.bandopt.enable=false"
  "evaluation.benchmark.inverse_post.apply_sigmoid=true"
  "evaluation.benchmark.inverse_post.binarize=true"
  "evaluation.benchmark.inverse_post.binarize_thr=0.5"

  # data cap (items 기준 10000)
  "task.inverse.data.max_samples.train=${MAX_SAMPLES_TRAIN}"
  "task.inverse.data.max_samples_unit=items"

  # domain mix 사용
  "task.inverse.data.domain.enable=true"
)

run_case () {
  local TAG="$1"
  local B1="$2"
  local B2="$3"
  local B3="$4"
  local B4="$5"
  local B5="$6"

  local EXP_NAME="${BASE_EXP_PREFIX}_${TAG}"

  echo "================================================"
  echo "[RUN] ${EXP_NAME}"
  echo "      weights: B1=${B1}, B2=${B2}, B3=${B3}, B4=${B4}, B5=${B5}"
  echo "================================================"

  python main.py \
    "${COMMON_ARGS[@]}" \
    "exp_name=${EXP_NAME}" \
    "task.inverse.data.domain.weights.B1=${B1}" \
    "task.inverse.data.domain.weights.B2=${B2}" \
    "task.inverse.data.domain.weights.B3=${B3}" \
    "task.inverse.data.domain.weights.B4=${B4}" \
    "task.inverse.data.domain.weights.B5=${B5}"
}

# ------------------------------------------------
# Sweeps (B1 B2 B3 B4 B5)
# ------------------------------------------------
run_case "B1_1"          1.0       0.0       0.0       0.0       0.0
run_case "B1B2_50_50"    0.5       0.5       0.0       0.0       0.0
run_case "B2_1"          0.0       1.0       0.0       0.0       0.0
run_case "B1B2B3_50_25_25" 0.5     0.25      0.25      0.0       0.0
run_case "B1B2B3B4_40_20_20_20" 0.4 0.2      0.2       0.2       0.0
run_case "B5_1"          0.0       0.0       0.0       0.0       1.0
run_case "mix_2_1_1_1_1_over6" 0.3333333333 0.1666666667 0.1666666667 0.1666666667 0.1666666667

echo "✅ All sweeps finished."
