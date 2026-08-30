#!/usr/bin/env bash
set -euo pipefail

# Full held-out evaluation through the same executable environment used for
# training. Set HYPER_R1_ENABLE=false for the released linear control.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/.." && pwd)"}
DATASET_TYPE=${DATASET_TYPE:-grailqa}
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint being evaluated}"
: "${TEST_FILE:?Set TEST_FILE to the held-out parquet}"
: "${HYPER_RELATION_MODEL:?Set HYPER_RELATION_MODEL to the v15 relation ranker}"

export REPO_ROOT DATASET_TYPE MODEL_PATH
export HYPER_RELATION_MODEL
export BASE_MODEL="${MODEL_PATH}"
export OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/runs/hyper_r1_eval"}
export HYPER_R1_ENABLE=${HYPER_R1_ENABLE:-true}
export HYPER_R1_FRONTIER_WIDTH=${HYPER_R1_FRONTIER_WIDTH:-6}
export SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-32}
export MAX_SAMPLES=1
export REWARD_THRESHOLD=0
export NUM_SAMPLES=null

mkdir -p "${OUTPUT_DIR}"
if [[ "${HYPER_R1_ENABLE}" == "true" ]]; then
  AUGMENTED_TEST_FILE=${AUGMENTED_TEST_FILE:-"${OUTPUT_DIR}/test_hyper_r1.parquet"}
  python3 "${REPO_ROOT}/scripts/data_process/enable_hyper_r1_prompts.py" \
    --input "${TEST_FILE}" --output "${AUGMENTED_TEST_FILE}"
  export INPUT_FILE_HINT="${AUGMENTED_TEST_FILE}"
else
  export INPUT_FILE_HINT="${TEST_FILE}"
fi

exec bash "${REPO_ROOT}/scripts/data_process/rejection_sampling_simple.sh"
