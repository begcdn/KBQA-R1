#!/usr/bin/env bash
set -euo pipefail

# Full held-out evaluation through the same executable environment used for
# training. Set HYPER_R1_ENABLE=false for the released linear control.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/.." && pwd)"}
DATASET_TYPE=${DATASET_TYPE:-grailqa}
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint being evaluated}"
: "${TEST_FILE:?Set TEST_FILE to the held-out parquet}"

export REPO_ROOT DATASET_TYPE MODEL_PATH
export BASE_MODEL="${MODEL_PATH}"
export INPUT_FILE_HINT="${TEST_FILE}"
export OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/runs/hyper_r1_eval"}
export HYPER_R1_ENABLE=${HYPER_R1_ENABLE:-true}
export MAX_SAMPLES=1
export REWARD_THRESHOLD=0
export NUM_SAMPLES=null

exec bash "${REPO_ROOT}/scripts/data_process/rejection_sampling_simple.sh"
