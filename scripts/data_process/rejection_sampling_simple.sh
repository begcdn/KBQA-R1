#!/usr/bin/env bash
set -xeuo pipefail


# Simplified Rejection Sampling for KBQA-R1
# 直接使用 main_ppo_kbqa.py 的 val_only 模式进行 rollout
# 这样可以复用所有的环境交互逻辑，而不需要重新实现

# export RAY_DEBUG_POST_MORTEM=1
export RAY_DEBUG=1
export RAY_memory_monitor_refresh_ms=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
use_dynamic_bsz=true
# 检测 GPU 数量
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected ${GPU_COUNT} GPUs using nvidia-smi"
        return 0
    fi
    if command -v python &> /dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected ${GPU_COUNT} GPUs using PyTorch"
        return 0
    fi
    echo "Warning: Could not detect GPU count, defaulting to 8"
    GPU_COUNT=8
    return 1
}

detect_gpu_count

if [ "$GPU_COUNT" -eq 16 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    N_GPUS_PER_NODE=16
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.6
    export BASE_MODEL=${BASE_MODEL:-'/aml/share/aml/sota_models/Qwen3/Qwen3-32B'}
    export MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}

    echo "PPU machine detected: Using 16 GPUs"
elif [ "$GPU_COUNT" -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    N_GPUS_PER_NODE=8
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.6
    # export BASE_MODEL='/ossfs/workspace/aml0/Qwen3/Qwen3-32B'
    # export MODEL_PATH='/ossfs/workspace/aml0/Qwen3/Qwen3-32B'
    export BASE_MODEL=${BASE_MODEL:-'/ossfs/workspace/aml0/Qwen3/Qwen2.5-72B-Instruct'}
    export MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}
    echo "Standard machine detected: Using 8 GPUs"
else
    N_GPUS_PER_NODE=${GPU_COUNT}
    TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
    : "${MODEL_PATH:?Set MODEL_PATH when running on fewer than 8 GPUs}"
    export BASE_MODEL=${BASE_MODEL:-"${MODEL_PATH}"}
    echo "Small GPU host detected: Using ${GPU_COUNT} GPU(s)"
fi

# 基础配置
DATASET_TYPE=${DATASET_TYPE:-'webqsp'}
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset"}
# 使用带 hint 的数据集与正常数据集共同进行 rejection sampling
# 可通过环境变量覆盖：INPUT_FILE_HINT, INPUT_FILE_NORMAL
INPUT_FILE_HINT=${INPUT_FILE_HINT:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset_sft/train_with_hints.parquet"}
INPUT_FILE_NORMAL=''
# INPUT_FILE_HINT=''
# INPUT_FILE_NORMAL=${INPUT_FILE_NORMAL:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset/train.parquet"}

OUTPUT_DIR=${OUTPUT_DIR:-"${DATA_DIR}_sft"}

# 组装训练/验证数据文件列表（优先使用存在的文件）
TRAIN_FILES=()
if [[ -f "${INPUT_FILE_HINT}" ]]; then
    TRAIN_FILES+=("${INPUT_FILE_HINT}")
else
    echo "[WARN] Hint dataset not found: ${INPUT_FILE_HINT}"
fi
if [[ -f "${INPUT_FILE_NORMAL}" ]]; then
    TRAIN_FILES+=("${INPUT_FILE_NORMAL}")
else
    echo "[WARN] Normal dataset not found: ${INPUT_FILE_NORMAL}"
fi

if [[ ${#TRAIN_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] No input datasets found. Checked:"
    echo "  - ${INPUT_FILE_HINT}"
    echo "  - ${INPUT_FILE_NORMAL}"
    exit 1
fi

# 构造 Hydra 可解析的 Python 列表字符串，例如 ['file1','file2']
TRAIN_FILES_STR="["
for i in "${!TRAIN_FILES[@]}"; do
    if (( i > 0 )); then TRAIN_FILES_STR+=", "; fi
    TRAIN_FILES_STR+="'${TRAIN_FILES[$i]}'"
done
TRAIN_FILES_STR+="]"

# Rejection sampling 参数
MAX_SAMPLES=${MAX_SAMPLES:-3}           # 每个样本最多采样次数
REWARD_THRESHOLD=${REWARD_THRESHOLD:-0.8}  # 奖励阈值
NUM_SAMPLES=${NUM_SAMPLES:-null}         # null processes the full split
HYPER_R1_ENABLE=${HYPER_R1_ENABLE:-false}
HYPER_R1_MAX_ACTIVE=${HYPER_R1_MAX_ACTIVE:-6}
HYPER_R1_MAX_NODES=${HYPER_R1_MAX_NODES:-24}
HYPER_R1_FRONTIER_WIDTH=${HYPER_R1_FRONTIER_WIDTH:-3}
HYPER_R1_MAX_FRONTIER_WIDTH=${HYPER_R1_MAX_FRONTIER_WIDTH:-6}

# GPU 配置
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPU_COUNT}}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-8}
if (( NGPUS_PER_NODE < 4 )); then
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
    MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
else
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1024}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-48}
    MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-16384}
fi

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="rejection_sampling_${DATASET_TYPE}_${TIMESTAMP}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/rejection_sampling_${TIMESTAMP}.log"

# Create experiment subdirectories for validation and rollout dumps
EXPERIMENT_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"
VALIDATION_DUMP_DIR="${EXPERIMENT_DIR}/validation"
ROLLOUT_DUMP_DIR="${EXPERIMENT_DIR}/rollout"
mkdir -p "${VALIDATION_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"

# 环境变量
export WANDB_MODE=offline
export SEXPR_MODE=true
export ENABLE_ACTION_REASONING=true
export ENABLE_RELATION_RETRIEVAL=true
export SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-7}
export SEXPR_VALIDATION_LEVEL=STANDARD
export PYTHON_LOG_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Simplified Rejection Sampling"
echo "========================================"
echo "Dataset Type : ${DATASET_TYPE}"
echo "Input Files  : ${TRAIN_FILES_STR}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Model        : ${BASE_MODEL}"
echo "Max Samples  : ${MAX_SAMPLES}"
echo "Reward Thresh: ${REWARD_THRESHOLD}"
echo "Num Samples  : ${NUM_SAMPLES}"
echo "GPUs         : ${NGPUS_PER_NODE}"
echo "Experiment   : ${EXPERIMENT_NAME}"
echo "HyPER-R1     : ${HYPER_R1_ENABLE}"
echo "========================================"
# 核心思路：使用 main_ppo_kbqa.py 的 val_only=true 模式
# 这样会：
# 1. 初始化所有 workers（actor_rollout_wg 等）
# 2. 加载数据集
# 3. 进行 rollout（通过 SExprLLMGenerationManager）
# 4. 计算 reward
# 5. 不进行训练（因为 val_only=true）
#
# 然后我们可以：
# - 从日志中提取高质量样本（reward > threshold）
# - 或者修改 val_reward_fn 来保存高质量样本

use_dynamic_bsz=true

python3 -m verl.trainer.main_ppo_kbqa \
    data.train_files="${TRAIN_FILES_STR}" \
    data.val_files="${TRAIN_FILES_STR}" \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=true \
    data.truncation='left' \
    data.train_batch_size=512 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=4196 \
    algorithm.adv_estimator=grpo \
    algorithm.hyper_r1_enable=${HYPER_R1_ENABLE} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.n=${MAX_SAMPLES} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.project_name="KBQA-R1-Rejection-Sampling" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VALIDATION_DUMP_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}" \
    trainer.log_val_generations=1000 \
    +trainer.max_val_samples=${NUM_SAMPLES} \
    sexpr_config.enable_sexpr_mode=true \
    sexpr_config.enable_action_reasoning=true \
    sexpr_config.enable_relation_retrieval=true \
    sexpr_config.validation_level=${SEXPR_VALIDATION_LEVEL} \
    sexpr_config.max_function_calls=10 \
    sexpr_config.enable_entity_linking=true \
    sexpr_config.enable_semantic_validation=true \
    sexpr_config.use_complete_sparql_converter=true \
    hyper_r1.enable=${HYPER_R1_ENABLE} \
    hyper_r1.max_active=${HYPER_R1_MAX_ACTIVE} \
    hyper_r1.max_nodes=${HYPER_R1_MAX_NODES} \
    hyper_r1.frontier_width=${HYPER_R1_FRONTIER_WIDTH} \
    hyper_r1.max_frontier_width=${HYPER_R1_MAX_FRONTIER_WIDTH} \
    max_turns=${SEXPR_MAX_TURNS} \
    use_odbc=true \
    use_aioodbc=false \
    odbc_config.driver_path="Virtuoso" \
    odbc_config.host="localhost" \
    odbc_config.port=13001 \
    odbc_config.uid="dba" \
    odbc_config.pwd="dba" \
    odbc_config.pool_size=4 \
    odbc_config.max_pool_size=20 \
    odbc_config.pool_timeout=30 \
    odbc_config.query_timeout=600 \
    odbc_config.max_concurrent=16 \
    odbc_config.max_retries=1 \
    odbc_config.retry_delay=1.0 \
    sparql_batch_size=128 \
    sparql_max_concurrent=16 \
    sparql.url="http://0.0.0.0:8000/execute" \
    reward_model.reward_manager=kbqa \
    reward_model.reward_kwargs.mid_f1_weight=1.0 \
    reward_model.reward_kwargs.structure_format_score=0.0 \
    reward_model.val_reward_kwargs.mid_f1_weight=1.0 \
    reward_model.val_reward_kwargs.structure_format_score=0.0 \
    custom_reward_function.path="${REPO_ROOT}/kbqa_custom_reward.py" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "Rollout Completed"
echo "========================================"
echo "Log file: ${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo ""
echo "Next steps:"
echo "1. 从日志中提取高质量样本（reward >= ${REWARD_THRESHOLD}）"
echo "2. 转换为 SFT 格式"
echo ""
echo "提取高质量样本的命令："
echo "python scripts/data_process/extract_high_quality_samples.py \\"
echo "    --log_file ${LOG_FILE} \\"
echo "    --reward_threshold ${REWARD_THRESHOLD} \\"
echo "    --output_file ${OUTPUT_DIR}/train_sft.parquet"
echo "========================================"
