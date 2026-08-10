#!/usr/bin/env bash
set -euo pipefail

# Collect successful executable hypothesis-graph traces with the released
# KBQA-R1 policy, then convert them into the HyPER-R1 SFT corpus.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
DATASET_TYPE=${DATASET_TYPE:-grailqa}
SOURCE_DATA=${SOURCE_DATA:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset_sft/train_with_hints.parquet"}
WORK_DIR=${WORK_DIR:-"${REPO_ROOT}/data/${DATASET_TYPE}_hyper_r1_sft"}
PROMPT_DATA="${WORK_DIR}/train_with_hyper_prompts.parquet"
LINEAR_SFT="${WORK_DIR}/successful_rollouts.parquet"
HYPER_SFT="${WORK_DIR}/train.parquet"

: "${MODEL_PATH:?Set MODEL_PATH to the released KBQA-R1 SFT checkpoint}"
mkdir -p "${WORK_DIR}"

python3 "${REPO_ROOT}/scripts/data_process/enable_hyper_r1_prompts.py" \
  --input "${SOURCE_DATA}" \
  --output "${PROMPT_DATA}"

export REPO_ROOT DATASET_TYPE MODEL_PATH
export BASE_MODEL="${MODEL_PATH}"
export INPUT_FILE_HINT="${PROMPT_DATA}"
export OUTPUT_DIR="${WORK_DIR}/rollouts"
export HYPER_R1_ENABLE=true
export MAX_SAMPLES=${MAX_SAMPLES:-4}
export REWARD_THRESHOLD=${REWARD_THRESHOLD:-0.8}

bash "${REPO_ROOT}/scripts/data_process/rejection_sampling_simple.sh"

LATEST_DUMP=$(find "${OUTPUT_DIR}" -type d -path '*/validation' | sort | tail -n 1)
if [[ -z "${LATEST_DUMP}" ]]; then
  echo "No validation rollout directory was produced" >&2
  exit 1
fi

python3 "${REPO_ROOT}/scripts/data_process/build_sft_from_dumps.py" \
  --dump_dir "${LATEST_DUMP}" \
  --output_file "${LINEAR_SFT}" \
  --reward_threshold "${REWARD_THRESHOLD}" \
  --min_mid_f1 0.9 \
  --info_role tool

python3 "${REPO_ROOT}/scripts/data_process/build_hyper_r1_sft.py" \
  --input "${LINEAR_SFT}" \
  --output "${HYPER_SFT}"

echo "HyPER-R1 SFT corpus: ${HYPER_SFT}"
