#!/bin/bash

# S-Expression Generation Training Script for KBQA-R1 
# Uses sexpr_generation.py as the main generation component
# Usage: 
#   ./train_kbqa_sexpr_generation.sh                        # Auto-detect GPU count, WebQSP dataset (default)
#   ./train_kbqa_sexpr_generation.sh experiment1            # Set EXPERIMENT_NAME=experiment1
#
# Dataset Selection (set DATASET_TYPE environment variable):
#   DATASET_TYPE=webqsp ./train_kbqa_sexpr_generation.sh    # WebQSP dataset (default)
#   DATASET_TYPE=grailqa ./train_kbqa_sexpr_generation.sh   # GrailQA dataset  
#   DATASET_TYPE=graphq ./train_kbqa_sexpr_generation.sh    # GraphQ dataset

# Suppress Ray FutureWarning about GPU environment variable override
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
REPO_ROOT="/ossfs/workspace/kbqa-r1"
# Auto-detect GPU count and set configuration
export KBQA_DEBUG_BP=1
detect_gpu_count() {
    # Try nvidia-smi first
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected $GPU_COUNT GPUs using nvidia-smi"
        return 0
    fi
    
    # Fallback to python if nvidia-smi not available
    if command -v python &> /dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected $GPU_COUNT GPUs using PyTorch"
        return 0
    fi
    
    # Fallback to python3
    if command -v python3 &> /dev/null; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected $GPU_COUNT GPUs using PyTorch (python3)"
        return 0
    fi
    
    echo "Warning: Could not detect GPU count, defaulting to 8"
    GPU_COUNT=8
    return 1
}

# Detect GPU count
detect_gpu_count

# Set GPU configuration based on detected GPU count
if [ "$GPU_COUNT" -eq 16 ]; then
    # PPU machine: 16 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    N_GPUS_PER_NODE=16
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.6
    echo "PPU machine detected: Using 16 GPUs"
elif [ "$GPU_COUNT" -eq 8 ]; then
    # Standard machine: 8 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    N_GPUS_PER_NODE=8
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.6
    echo "Standard machine detected: Using 8 GPUs"
elif [ "$GPU_COUNT" -eq 0 ]; then
    # Default to 8 GPUs if no GPUs detected (e.g., on Mac)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    N_GPUS_PER_NODE=8
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.6
    echo "No GPUs detected, using default 8-GPU configuration"
else
    # For other GPU counts, use all available GPUs
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)) | sed 's/,$//')
    N_GPUS_PER_NODE=$GPU_COUNT
    TENSOR_MODEL_PARALLEL_SIZE=4
    GPU_MEM_UTIL=0.8
    echo "Custom GPU configuration: Using $GPU_COUNT GPUs"
fi

# Detect GPU type (H20 vs A100)
detect_gpu_type() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        if [[ "${GPU_NAME}" == *"H20"* ]]; then
            echo "H20"
        elif [[ "${GPU_NAME}" == *"A100"* ]]; then
            echo "A100"
        else
            echo "UNKNOWN"
        fi
    else
        echo "UNKNOWN"
    fi
}

GPU_TYPE=$(detect_gpu_type)
echo "Detected GPU Type: ${GPU_TYPE}"

# Data configuration - VERL format for reinforcement learning
# Support multiple datasets: webqsp, grailqa, graphq
DATASET_TYPE=${DATASET_TYPE:-'webqsp'}
export DATASET_TYPE="${DATASET_TYPE}"
# Set total epochs based on dataset type
if [[ "${DATASET_TYPE}" == "grailqa" ]]; then
    TOTAL_EPOCHS=3
else
    TOTAL_EPOCHS=8
fi

export DATA_DIR="data/${DATASET_TYPE}_rl_dataset"
export WANDB_MODE=offline
export WANDB_API_KEY='fcd289b147d05c0afe3ca10bef15610cdee020ff'
WAND_PROJECT='KBQA-R1-SExpr'

# Checkpoint configuration
# Checkpoint directory - configured based on GPU type
if [[ "${GPU_TYPE}" == "H20" ]]; then
    # H20 configuration: use nas_zhongqi
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-'/ossfs/workspace/nas_zhongqi/checkpoints/'}
    echo "[INFO] Checkpoints will be saved to nas_zhongqi: ${CHECKPOINT_DIR}"
else
    # A100 configuration: use aml/share
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-'/aml/share/aml/465910'}
    echo "[INFO] Checkpoints will be saved to aml/share: ${CHECKPOINT_DIR}"
fi

# Generate timestamp for unique directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Model configuration
# By default we use a base Llama-3 model. Set USE_SFT_MODEL=true to use a dataset-specific SFT model.
# Example: USE_SFT_MODEL=true DATASET_TYPE=grailqa ./train_kbqa_sexpr_generation.sh
# Default base model (Llama-3)
DEFAULT_BASE_MODEL='/ossfs/workspace/aml2/aml_ri/fengyi/Llama-3.1-8B-Instruct'

# Dataset-specific SFT checkpoints
SFT_MODEL_WEBQSP='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/webqsp-sft-from-rs/20251202_034557/global_step_94/huggingface'
# SFT_MODEL_WEBQSP='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/webqsp-sft-from-rs/20251031_172831/global_step_84/huggingface'
# SFT_MODEL_GRAILQA='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251102_221931/global_step_1342/huggingface'
# SFT_MODEL_GRAPHQ='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/graphq-sft-from-rs/20251102_193738/global_step_144/huggingface'
SFT_MODEL_GRAPHQ='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/graphq-sft-from-rs/20251202_022546/global_step_96/huggingface'

if [[ "${GPU_TYPE}" == "H20" ]]; then
    # H20 configuration: use nas_zhongqi paths
    SFT_MODEL_GRAILQA='/ossfs/workspace/nas_zhongqi/huggingface' #2ep
    echo "[INFO] Using H20 GPU configuration with nas_zhongqi paths"
else
    # A100 configuration: use kbqa-r1 paths
    # SFT_MODEL_GRAILQA='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251102_221931/global_step_1342/huggingface'
    SFT_MODEL_GRAILQA='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251202_025127/global_step_909/huggingface'

    echo "[INFO] Using A100 GPU configuration with kbqa-r1 paths"
fi
# Control whether to use the SFT model for RL (default: true)

# Control whether to use the SFT model for RL (default: false)
USE_SFT_MODEL=${USE_SFT_MODEL:-true}

if [[ "${USE_SFT_MODEL}" == "true" ]]; then
    case "${DATASET_TYPE}" in
        webqsp)
            BASE_MODEL="${SFT_MODEL_WEBQSP}"
            ;;
        grailqa)
            BASE_MODEL="${SFT_MODEL_GRAILQA}"
            ;;
        graphq)
            BASE_MODEL="${SFT_MODEL_GRAPHQ}"
            ;;
        *)
            echo "[WARN] Unknown DATASET_TYPE='${DATASET_TYPE}', falling back to default base model."
            BASE_MODEL="${DEFAULT_BASE_MODEL}"
            ;;
    esac
else
    BASE_MODEL="${DEFAULT_BASE_MODEL}"
fi
export BASE_MODEL
echo "[INFO] Using BASE_MODEL=${BASE_MODEL} (USE_SFT_MODEL=${USE_SFT_MODEL}, DATASET_TYPE=${DATASET_TYPE})"

# Set experiment name from positional args
EXPERIMENT_PREFIX="${DATASET_TYPE}-kbqa-r1-ppo-sexpr-generation"
EXP_ARG="$1"
if [[ -n "$EXP_ARG" ]]; then
    # If user provided an experiment name, suffix a precise timestamp to unify naming across logs and TensorBoard
    export EXPERIMENT_NAME="${EXP_ARG}__${TIMESTAMP}"
else
    # Default experiment name already includes timestamp
    export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}/${TIMESTAMP}"
fi
export RAY_memory_monitor_refresh_ms=0

# Set TensorBoard directory to align with EXPERIMENT_NAME for unified naming
export TENSORBOARD_DIR=tensorboard_log/${EXPERIMENT_NAME}

# Ensure directories exist (统一把日志集中到 logs/ 目录；不再在当前目录落盘实验日志)
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
# 仍保留变量但默认不写入文件；如需恢复文件日志，可把管道 tee 打开
LOG_PATH="${LOG_DIR}/${EXPERIMENT_NAME}.log"
VALIDATION_DUMP_DIR="${LOG_DIR}/${EXPERIMENT_NAME}/validation"
mkdir -p "${VALIDATION_DUMP_DIR}"

# Enable XFORMERS backend for better performance

rollout_is=${rollout_is:-false}                   # apply weights to policy loss (false = metrics only)
rollout_is_level=${rollout_is_level:-sequence}          # token | sequence | geometric
rollout_is_mode=${rollout_is_mode:-mask}         # truncate | mask
rollout_is_threshold=${rollout_is_threshold:-2.0}    # float => enable; "" or null => disable
rollout_is_threshold_lower=${rollout_is_threshold_lower:-0.5}  # optional; only meaningful for mask mode
rollout_is_veto_threshold=${rollout_is_veto_threshold:-1e-4} # per-token veto (null to disable)

# S-Expression specific environment variables
export SEXPR_MODE=true
export ENABLE_ACTION_REASONING=true
export ENABLE_RELATION_RETRIEVAL=true
export SEXPR_MAX_TURNS=6
export SEXPR_VALIDATION_LEVEL=STANDARD


# Logging configuration for S-Expression testing
export PYTHON_LOG_LEVEL=INFO
export SEXPR_DEBUG_MODE=true
export VERL_LOGGING_LEVEL=INFO
export VERL_PPO_LOGGING_LEVEL=INFO

# Print configuration summary for testing
echo "========================================"
echo "S-Expression Training Configuration:"
echo "========================================"
echo "Dataset: $DATASET_TYPE"
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $N_GPUS_PER_NODE (Tensor Parallel: $TENSOR_MODEL_PARALLEL_SIZE)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Data Dir: $DATA_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "S-Expression Mode: ENABLED"
echo "Max Turns: $SEXPR_MAX_TURNS"
echo "Validation Level: $SEXPR_VALIDATION_LEVEL"
echo "SPARQL URL: http://0.0.0.0:8000/execute"
echo "Tensorboard: $TENSORBOARD_DIR"
echo "========================================"

echo "[SCRIPT] Starting S-Expression training with VERL..."
echo "[SCRIPT] Key S-Expression parameters to check in logs:"
echo "  - sexpr_config.enable_sexpr_mode=true"
echo "  - sexpr_config.enable_action_reasoning=true" 
echo "  - sexpr_config.enable_semantic_validation=true"
echo "[SCRIPT] Looking for log markers: [VERL], [SEXPR], [REWARDS]"

use_dynamic_bsz=true

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_kbqa \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +trainer.max_val_samples=500 \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=3072 \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.dtype=float16 \
    actor_rollout_ref.ref.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.3 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.top_p=0.99 \
	actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    trainer.validation_data_dir="${VALIDATION_DUMP_DIR}" \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CHECKPOINT_DIR/$EXPERIMENT_NAME \
    max_turns=$SEXPR_MAX_TURNS \
    use_odbc=True \
    use_aioodbc=False \
    odbc_config.driver_path="Virtuoso" \
    odbc_config.host="localhost" \
    odbc_config.port=13001 \
    odbc_config.uid="dba" \
    odbc_config.pwd="dba" \
    odbc_config.pool_size=4 \
    odbc_config.max_pool_size=20 \
    odbc_config.pool_timeout=30 \
    odbc_config.query_timeout=600 \
    odbc_config.max_concurrent=4 \
    odbc_config.max_retries=1 \
    odbc_config.retry_delay=1.0 \
    sparql_batch_size=128 \
    sparql_max_concurrent=16 \
    sparql.url="http://0.0.0.0:8000/execute" \
    sexpr_config.enable_sexpr_mode=true \
    sexpr_config.enable_action_reasoning=true \
    sexpr_config.enable_relation_retrieval=true \
    sexpr_config.validation_level=STANDARD \
    sexpr_config.max_function_calls=10 \
    sexpr_config.enable_entity_linking=true \
    sexpr_config.enable_semantic_validation=true \
    sexpr_config.use_complete_sparql_converter=true \
    +sexpr_config.enable_logging=false \
    reward_model.reward_manager=kbqa \
    reward_model.reward_kwargs.mid_f1_weight=1.0 \
    reward_model.reward_kwargs.structure_format_score=0.1 \
    reward_model.val_reward_kwargs.mid_f1_weight=1.0 \
    reward_model.val_reward_kwargs.structure_format_score=0.0 \
    custom_reward_function.path="${REPO_ROOT}/kbqa_custom_reward.py" \
    2>&1 | tee "${LOG_PATH}"  # 只输出到控制台，不落盘日志文件

# Post-training summary
echo ""
echo "========================================"
echo "S-Expression Training Completed"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: (disabled; no .log saved, only TensorBoard & checkpoints)"
echo "Tensorboard: $TENSORBOARD_DIR"
echo "Checkpoints: $CHECKPOINT_DIR/$EXPERIMENT_NAME"
echo ""
echo "(File logs disabled; use TensorBoard or console output for monitoring)"
echo ""
echo "To start tensorboard:"
echo "  tensorboard --logdir $TENSORBOARD_DIR --port 6006"
echo "========================================" 

cd /ossfs/workspace
pkill -f main_ppo_kbqa
sleep 100
python train.py
# S-Expression Training Notes:
# ==========================
# 
# Key differences from format training:
# 1. Uses sexpr_generation.py for action-based reasoning
# 2. Generates S-Expressions instead of SPARQL directly
# 3. Uses SExprExecutor for query execution
# 4. Employs action parsing and function building
# 5. Includes relation retrieval for dynamic schema exploration
#
# S-Expression Configuration:
# - enable_sexpr_mode=true: Activates S-Expression generation mode
# - enable_action_reasoning=true: Uses action-based reasoning
# - enable_relation_retrieval=true: Dynamic relation discovery
# - validation_level=STANDARD: S-Expression validation level
# - max_function_calls=10: Limit for function call chains
# - use_complete_sparql_converter=true: Uses KBQA-o1 integrated converter
#
# Reward Model for S-Expression:
# - mid_f1_weight=3.0: Base reward for correct answers (unchanged)
# - structure_format_score=0.2: Format reward for valid KBQA sequence structure
#
# Data Requirements:
# - Question-Answer pairs with gold S-Expressions
# - Entity linking annotations (optional but recommended)
# - Relation type information for retrieval
# - Expected format: see DATA_PREPARATION.md
