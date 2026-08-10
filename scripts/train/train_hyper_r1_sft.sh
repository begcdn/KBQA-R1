#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_SFT:?Set SOURCE_SFT to successful rollout SFT parquet}"
REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
HYPER_SFT=${HYPER_SFT:-"${REPO_ROOT}/data/grailqa_hyper_r1_sft/train.parquet"}

python3 "${REPO_ROOT}/scripts/data_process/build_hyper_r1_sft.py" \
  --input "${SOURCE_SFT}" \
  --output "${HYPER_SFT}"

export TRAIN_PARQUET="${HYPER_SFT}"
export DATASET_TYPE=${DATASET_TYPE:-grailqa}
export PROJECT_NAME=${PROJECT_NAME:-HyPER-R1-SFT}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grailqa-hyper-r1-sft}

exec "${REPO_ROOT}/scripts/train/run_sft_from_rejection.sh"
