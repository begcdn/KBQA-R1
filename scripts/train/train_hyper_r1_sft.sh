#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
: "${BASE_MODEL:?Set BASE_MODEL to the released KBQA-R1 SFT checkpoint}"
HYPER_SFT=${HYPER_SFT:-"${REPO_ROOT}/data/grailqa_hyper_r1_demonstrations/train.parquet"}
if [[ ! -f "${HYPER_SFT}" ]]; then
  echo "Missing verified HyPER-R1 trajectory data: ${HYPER_SFT}" >&2
  echo "Run scripts/data_process/build_hyper_demonstrations.py first." >&2
  exit 1
fi

export TRAIN_PARQUET="${HYPER_SFT}"
export DATASET_TYPE=${DATASET_TYPE:-grailqa}
export PROJECT_NAME=${PROJECT_NAME:-HyPER-R1-SFT}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grailqa-hyper-r1-sft}

exec "${REPO_ROOT}/scripts/train/run_sft_from_rejection.sh"
