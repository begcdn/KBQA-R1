#!/usr/bin/env bash
set -euo pipefail

# Prepare Rejection Sampling Datasets for All KBQA Datasets
# 为所有 KBQA 数据集添加 action hints，用于 rejection sampling
#
# 支持的数据集:
# - WebQSP
# - GrailQA
# - GraphQ

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="/ossfs/workspace/kbqa-r1"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HINT_STYLE=${HINT_STYLE:-"reference"}   # Hint 风格: reference, hidden, example
USE_ODBC=${USE_ODBC:-false}             # 是否使用 ODBC 获取实体标签
NUM_SAMPLES=${NUM_SAMPLES:-}            # 处理样本数（留空 = 全部）
SKIP_NO_HINT=${SKIP_NO_HINT:-false}     # 是否跳过没有 hint 的样本（默认保留，用于 RL 训练）

# Datasets to process
# DATASETS=${DATASETS:-"webqsp grailqa graphq"}
DATASETS=${DATASETS:-"grailqa"}

echo "========================================"
echo "Prepare Rejection Sampling Datasets"
echo "========================================"
echo "Datasets     : ${DATASETS}"
echo "Hint Style   : ${HINT_STYLE}"
echo "Use ODBC     : ${USE_ODBC}"
echo "Num Samples  : ${NUM_SAMPLES:-all}"
echo "Skip No Hint : ${SKIP_NO_HINT}"
echo "========================================"
echo ""

# Track statistics
TOTAL_DATASETS=0
SUCCESS_DATASETS=0
FAILED_DATASETS=0

# Process each dataset
for dataset in ${DATASETS}; do
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    
    echo "========================================"
    echo "Processing Dataset: ${dataset}"
    echo "========================================"
    
    # Set dataset-specific paths
    DATA_DIR="${REPO_ROOT}/data/${dataset}_rl_dataset"
    OUTPUT_DIR="${DATA_DIR}_sft"
    # INPUT_TRAIN="${DATA_DIR}/train.parquet"
    INPUT_TEST="${DATA_DIR}/test.parquet"
    # OUTPUT_TRAIN="${OUTPUT_DIR}/train_with_hints.parquet"
    OUTPUT_TEST="${OUTPUT_DIR}/test_with_hints.parquet"
    
    # Check if input files exist
    # if [[ ! -f "${INPUT_TRAIN}" ]]; then
    #     echo "❌ Error: Training file not found: ${INPUT_TRAIN}"
    #     echo "   Please run prepare_rl_dataset.py first for ${dataset}"
    #     FAILED_DATASETS=$((FAILED_DATASETS + 1))
    #     echo ""
    #     continue
    # fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Build preprocessing command
    PREPROCESS_CMD="python3 ${SCRIPT_DIR}/prepare_rejection_sampling_dataset.py \
        --hint_style=${HINT_STYLE}"
    
    # Add optional arguments
    if [[ -n "${NUM_SAMPLES}" ]]; then
        PREPROCESS_CMD="${PREPROCESS_CMD} --num_samples=${NUM_SAMPLES}"
    fi
    
    if [[ "${USE_ODBC}" == "true" ]]; then
        PREPROCESS_CMD="${PREPROCESS_CMD} --use_odbc"
    fi
    
    if [[ "${SKIP_NO_HINT}" == "true" ]]; then
        PREPROCESS_CMD="${PREPROCESS_CMD} --skip_no_hint"
    fi
    
    # Process training set
    # echo ""
    # echo "Processing training set..."
    # TRAIN_CMD="${PREPROCESS_CMD} \
    #     --input_path='${INPUT_TRAIN}' \
    #     --output_path='${OUTPUT_TRAIN}'"
    
    # if eval ${TRAIN_CMD} 2>&1 | tee "${OUTPUT_DIR}/preprocessing_train.log"; then
    #     echo "✓ Training set processed: ${OUTPUT_TRAIN}"
    # else
    #     echo "❌ Error processing training set"
    #     FAILED_DATASETS=$((FAILED_DATASETS + 1))
    #     echo ""
    #     continue
    # fi
    
    # Process test set (if exists)
    if [[ -f "${INPUT_TEST}" ]]; then
        echo ""
        echo "Processing test set..."
        TEST_CMD="${PREPROCESS_CMD} \
            --input_path='${INPUT_TEST}' \
            --output_path='${OUTPUT_TEST}'"
        
        if eval ${TEST_CMD} 2>&1 | tee "${OUTPUT_DIR}/preprocessing_test.log"; then
            echo "✓ Test set processed: ${OUTPUT_TEST}"
        else
            echo "⚠ Warning: Error processing test set (continuing anyway)"
        fi
    else
        echo ""
        echo "⚠ Test file not found: ${INPUT_TEST} (skipping)"
    fi
    
    SUCCESS_DATASETS=$((SUCCESS_DATASETS + 1))
    echo ""
    echo "✓ Dataset ${dataset} completed successfully"
    echo ""
done

# Print summary
echo "========================================"
echo "Processing Summary"
echo "========================================"
echo "Total Datasets:   ${TOTAL_DATASETS}"
echo "Successful:       ${SUCCESS_DATASETS}"
echo "Failed:           ${FAILED_DATASETS}"
echo "========================================"
echo ""
echo "Output directories:"
for dataset in ${DATASETS}; do
    echo "  - ${REPO_ROOT}/data/${dataset}_rl_dataset_sft/"
done
echo ""

# Exit with appropriate code
if [[ ${FAILED_DATASETS} -gt 0 ]]; then
    echo "⚠ Warning: Some datasets failed to process"
    exit 1
else
    echo "✓ All datasets processed successfully!"
    exit 0
fi
