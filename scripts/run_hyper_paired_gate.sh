#!/usr/bin/env bash
set -euo pipefail

: "${BASELINE_MODEL:?Set BASELINE_MODEL to the pre-corrective checkpoint}"
: "${METHOD_MODEL:?Set METHOD_MODEL to the corrective checkpoint}"
: "${TEST_FILE:?Set TEST_FILE to the held-out evaluation parquet}"
: "${HYPER_RELATION_MODEL:?Set HYPER_RELATION_MODEL to the relation ranker}"

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/.." && pwd)"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/runs/hyper_r1_paired_gate"}
EXPECTED=${EXPECTED:-50}

run_arm() {
    local name=$1
    local model=$2
    local output="${OUTPUT_ROOT}/${name}"
    local progress="${output}/${name}/validation/progress.jsonl"

    mkdir -p "${output}"
    if [[ -f "${progress}" ]] && [[ "$(wc -l < "${progress}")" -ge "${EXPECTED}" ]]; then
        echo "${name}: already complete"
    else
        MODEL_PATH="${model}" \
        OUTPUT_DIR="${output}" \
        EXPERIMENT_NAME="${name}" \
        TEST_FILE="${TEST_FILE}" \
        HYPER_RELATION_MODEL="${HYPER_RELATION_MODEL}" \
        HYPER_R1_STRUCTURAL_CONSTRAINTS=true \
        HYPER_RELATION_DEVICE=cpu \
        TRAINER_LOGGER='[console]' \
        INCREMENTAL_VALIDATION_DUMP=true \
        VAL_BATCH_SIZE=2 \
        TRAIN_BATCH_SIZE=2 \
        MICRO_BATCH_SIZE=1 \
        FSDP_MODEL_DTYPE=bf16 \
        GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55} \
        MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-16384} \
        bash "${REPO_ROOT}/scripts/evaluate_hyper_r1.sh"
    fi

    python3 "${REPO_ROOT}/scripts/summarize_hyper_gate.py" \
        --progress "${progress}" \
        --expected "${EXPECTED}" \
        --output "${output}/analysis"
}

mkdir -p "${OUTPUT_ROOT}"
run_arm pre_corrective "${BASELINE_MODEL}" \
    > "${OUTPUT_ROOT}/pre_corrective.console.log" 2>&1
run_arm corrective "${METHOD_MODEL}" \
    > "${OUTPUT_ROOT}/corrective.console.log" 2>&1

python3 "${REPO_ROOT}/scripts/compare_hyper_r1_eval.py" \
    --baseline "${OUTPUT_ROOT}/pre_corrective/pre_corrective/validation/progress.jsonl" \
    --method "${OUTPUT_ROOT}/corrective/corrective/validation/progress.jsonl" \
    --output "${OUTPUT_ROOT}/paired_comparison.json"

echo "Paired HyPER-R1 gate complete: ${OUTPUT_ROOT}"
