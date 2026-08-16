#!/usr/bin/env bash
set -xeuo pipefail

# KBQA-R1 S-Expression Training 


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
export VLLM_LOGGING_LEVEL=DEBUG

# export PYTHONPATH="${REPO_ROOT}/verl_newest:${PYTHONPATH:-}"

# Debugging: print PYTHONPATH and which verl module will be used when running Python

# -----------------------------------------------------------------------------
# GPU discovery helpers (reuse logic from the legacy script)
# -----------------------------------------------------------------------------
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
    if command -v python3 &> /dev/null; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected ${GPU_COUNT} GPUs using PyTorch (python3)"
        return 0
    fi
    echo "Warning: Could not detect GPU count, defaulting to 8"
    GPU_COUNT=8
    return 1
}

detect_gpu_count

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

# export VLLM_USE_V1=${VLLM_USE_V1:-1}
# export VLLM_ATTENTION_BACKEND=XFORMERS
# FIXED: Enable XFORMERS for better performance (was commented out)
# export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# -----------------------------------------------------------------------------
# Dataset, model, and bookkeeping configuration
# -----------------------------------------------------------------------------
DATASET_TYPE=${DATASET_TYPE:-'webqsp'}
export DATASET_TYPE="${DATASET_TYPE}"

# Set total epochs based on dataset type
if [[ "${DATASET_TYPE}" == "grailqa" ]]; then
    TOTAL_EPOCHS=3
else
    TOTAL_EPOCHS=8
fi
# Mirror the sexpr_generation script naming: DATA_DIR and BASE_MODEL
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test.parquet"}

# Default base model (Llama-3)
DEFAULT_BASE_MODEL=${DEFAULT_BASE_MODEL:-'/ossfs/workspace/aml2/aml_ri/fengyi/Llama-3.1-8B-Instruct'}


# Dataset-specific SFT checkpoints
SFT_MODEL_WEBQSP=${SFT_MODEL_WEBQSP:-'/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/webqsp-sft-from-rs/20251202_034557/global_step_94/huggingface'}
# SFT_MODEL_WEBQSP='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/webqsp-sft-from-rs/20251031_172831/global_step_84/huggingface'
# SFT_MODEL_GRAILQA='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251102_221931/global_step_1342/huggingface'
# SFT_MODEL_GRAPHQ='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/graphq-sft-from-rs/20251102_193738/global_step_144/huggingface'
SFT_MODEL_GRAPHQ=${SFT_MODEL_GRAPHQ:-'/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/graphq-sft-from-rs/20251202_022546/global_step_96/huggingface'}

# if [[ "${GPU_TYPE}" == "H20" ]]; then
#     # H20 configuration: use nas_zhongqi paths
#     SFT_MODEL_GRAILQA='/ossfs/workspace/nas_zhongqi/huggingface' #2ep
#     echo "[INFO] Using H20 GPU configuration with nas_zhongqi paths"
# else
    # A100 configuration: use kbqa-r1 paths
    # SFT_MODEL_GRAILQA='/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251102_221931/global_step_1342/huggingface'
SFT_MODEL_GRAILQA=${SFT_MODEL_GRAILQA:-'/ossfs/workspace/kbqa-r1/checkpoints/KBQA-R1-SFT/grailqa-sft-from-rs/20251202_025127/global_step_1818/huggingface'}


# Control whether to use the SFT model for RL (default: true)
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

# Base/model path naming consistent with sexpr script
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# allow script to be called without positional args while running with `set -u`
EXP_ARG="${1:-}"
if [[ -n "${EXP_ARG}" ]]; then
    EXPERIMENT_NAME="${EXP_ARG}__${TIMESTAMP}"
else
    EXPERIMENT_NAME="${DATASET_TYPE}-kbqa-r1/${TIMESTAMP}"
fi

# Checkpoint directory - configured based on GPU type
if [[ "${GPU_TYPE}" == "H20" ]]; then
    # H20 configuration: use nas_zhongqi
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-'/ossfs/workspace/nas_zhongqi/checkpoints/'}
    echo "[INFO] Checkpoints will be saved to nas_zhongqi: ${CHECKPOINT_DIR}"
else
    # A100 configuration: use aml/share
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${REPO_ROOT}/checkpoints"}
    echo "[INFO] Checkpoints will be saved to aml/share: ${CHECKPOINT_DIR}"
fi

# TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-"tensorboard_log_sexpr_generation"}
# 统一日志到仓库根目录下的 logs/ 目录，避免分散 & NAS空间膨胀；用户仍要求保留主运行日志 (tee)
LOG_DIR_ROOT="logs"
mkdir -p "${LOG_DIR_ROOT}"  # 创建集中日志目录
LOG_FILE=${LOG_FILE:-"${LOG_DIR_ROOT}/${EXPERIMENT_NAME}.log"}
# LOG_DIR="${TENSORBOARD_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" "${LOG_DIR_ROOT}"

WANDB_MODE=${WANDB_MODE:-offline}
WANDB_PROJECT=${WANDB_PROJECT:-'KBQA-R1-SExpr'}
export WANDB_MODE
export WANDB_PROJECT
export WANDB_API_KEY=${WANDB_API_KEY:-''}
export TENSORBOARD_DIR=tensorboard_log/${EXPERIMENT_NAME}
VALIDATION_DUMP_DIR="${LOG_DIR_ROOT}/${EXPERIMENT_NAME}/validation"
mkdir -p "${VALIDATION_DUMP_DIR}"
# -----------------------------------------------------------------------------
# Training hyper-parameters (aligned with run_qwen2_7b defaults)
# -----------------------------------------------------------------------------
adv_estimator=${adv_estimator:-grpo}
use_kl_in_reward=${use_kl_in_reward:-false}
kl_coef=${kl_coef:-0.0}
use_kl_loss=${use_kl_loss:-false}
kl_loss_coef=${kl_loss_coef:-0.001}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}

# -----------------------------------------------------------------------------
# Rollout Importance Sampling (mismatch correction) configuration
# These env vars let you toggle mismatch_helper.py features without editing code.
# Set rollout_is_threshold to a float (e.g. 3.0) to ENABLE; leave empty or "" to disable.
# Recommended initial settings:
#   token-level + truncate: stable baseline
#   threshold 2.5~5.0; monitor mismatch/rollout_is_eff_sample_size & mismatch/rollout_is_p95
# To just observe metrics first: set rollout_is=false but keep threshold.
# To switch to mask mode: set rollout_is_mode=mask and optionally rollout_is_threshold_lower (default 1/upper).
# To disable veto: set rollout_is_veto_threshold="null" (Hydra will parse null) or very small (1e-6).
# -----------------------------------------------------------------------------
rollout_is=${rollout_is:-false}                       # apply weights to policy loss (false = metrics only)
rollout_is_level=${rollout_is_level:-token}          # token | sequence | geometric
rollout_is_mode=${rollout_is_mode:-truncate}         # truncate | mask
rollout_is_threshold=${rollout_is_threshold:-4.0}    # float => enable; "" or null => disable
rollout_is_threshold_lower=${rollout_is_threshold_lower:-null}  # optional; only meaningful for mask mode
rollout_is_veto_threshold=${rollout_is_veto_threshold:-null} # per-token veto (null to disable)

max_turns=${max_turns:-6}
max_prompt_length=${max_prompt_length:-14336}
# Reduce max_response_length to avoid GPU memory issues
max_response_length=${max_response_length:-1024}
actor_lr=${actor_lr:-1e-6}
hyper_r1_enable=${hyper_r1_enable:-false}
hyper_r1_max_active=${hyper_r1_max_active:-6}
hyper_r1_max_nodes=${hyper_r1_max_nodes:-24}
hyper_r1_frontier_width=${hyper_r1_frontier_width:-3}
hyper_r1_max_frontier_width=${hyper_r1_max_frontier_width:-6}
hyper_r1_relation_model=${hyper_r1_relation_model:-}
hyper_r1_credit_weight=${hyper_r1_credit_weight:-1.0}
hyper_r1_budget_cost=${hyper_r1_budget_cost:-0.05}
hyper_r1_invalid_commit_penalty=${hyper_r1_invalid_commit_penalty:-0.25}
hyper_r1_invalid_action_penalty=${hyper_r1_invalid_action_penalty:-0.25}
structure_format_score=${structure_format_score:-0.1}

if (( GPU_COUNT < 4 )); then
    train_batch_size=${train_batch_size:-32}
    ppo_mini_batch_size=${ppo_mini_batch_size:-16}
    n_resp_per_prompt=${n_resp_per_prompt:-4}
    actor_micro_batch_size=${actor_micro_batch_size:-1}
    log_prob_micro_batch_size=${log_prob_micro_batch_size:-2}
    rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.45}
    max_num_batched_tokens=${max_num_batched_tokens:-8192}
else
    train_batch_size=${train_batch_size:-256}
    ppo_mini_batch_size=${ppo_mini_batch_size:-128}
    n_resp_per_prompt=${n_resp_per_prompt:-5}
    actor_micro_batch_size=${actor_micro_batch_size:-40}
    log_prob_micro_batch_size=${log_prob_micro_batch_size:-128}
    rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.70}
    max_num_batched_tokens=${max_num_batched_tokens:-16384}
fi
# n_resp_per_prompt_val=${n_resp_per_prompt_val:-5}

# PERFORMANCE FIX: Increase tensor parallel from 2 to 4 to match original script
# This improves inference throughput significantly on 8-GPU setup
infer_tp=${infer_tp:-4}
offload=${offload:-true}

enable_overlong_buffer=${enable_overlong_buffer:-false}
overlong_buffer_len=${overlong_buffer_len:-$((256 * 4))}
overlong_penalty_factor=${overlong_penalty_factor:-1.0}


loss_agg_mode="token-mean"

actor_max_token_len_per_gpu=16384
log_prob_max_token_len_per_gpu=$((actor_max_token_len_per_gpu))

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPU_COUNT}}
TRAINING_GPUS=${TRAINING_GPUS:-${NGPUS_PER_NODE}}

use_dynamic_bsz=true

if (( NGPUS_PER_NODE < 1 )); then
    NGPUS_PER_NODE=1
fi
if (( TRAINING_GPUS < 1 )); then
    TRAINING_GPUS=1
fi
if (( TRAINING_GPUS > NGPUS_PER_NODE )); then
    TRAINING_GPUS=${NGPUS_PER_NODE}
fi


if (( infer_tp > NGPUS_PER_NODE )); then
    echo "Adjusting infer_tp (${infer_tp}) down to available GPUs per node (${NGPUS_PER_NODE})."
    infer_tp=${NGPUS_PER_NODE}
fi
if (( infer_tp < 1 )); then
    infer_tp=1
fi

echo "========================================"
echo "KBQA-R1 Configuration"
echo "========================================"
echo "Dataset Type : ${DATASET_TYPE}"
echo "Train File   : ${TRAIN_FILE}"
echo "Val File     : ${VAL_FILE}"
echo "Model Path   : ${MODEL_PATH}"
echo "Experiment   : ${EXPERIMENT_NAME}"
echo "HyPER-R1     : ${hyper_r1_enable} (active=${hyper_r1_max_active}, nodes=${hyper_r1_max_nodes})"
echo "Checkpoints  : ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo "Log File     : ${LOG_FILE} (only main run log; internal sexpr file logs disabled)"
echo "GPUs / node  : ${NGPUS_PER_NODE} (using ${TRAINING_GPUS} for training)"
echo "========================================"

# -----------------------------------------------------------------------------
# KBQA specific toggles
# -----------------------------------------------------------------------------
export SEXPR_MODE=true
export ENABLE_ACTION_REASONING=true
export ENABLE_RELATION_RETRIEVAL=true
export SEXPR_MAX_TURNS=${max_turns}
export SEXPR_VALIDATION_LEVEL=${SEXPR_VALIDATION_LEVEL:-STANDARD}
export PYTHON_LOG_LEVEL=${PYTHON_LOG_LEVEL:-INFO}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_PPO_LOGGING_LEVEL=${VERL_PPO_LOGGING_LEVEL:-INFO}

export PYTHONUNBUFFERED=1
export DISABLE_SEXPR_FILE_LOGS=1  # 禁用内部写盘 (sexpr_executor / logging manager)
python3 -m verl.trainer.main_ppo_kbqa \
    +trainer.max_val_samples=500 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.hyper_r1_enable=${hyper_r1_enable} \
    algorithm.hyper_r1_credit_weight=${hyper_r1_credit_weight} \
    algorithm.hyper_r1_budget_cost=${hyper_r1_budget_cost} \
    algorithm.hyper_r1_invalid_commit_penalty=${hyper_r1_invalid_commit_penalty} \
    algorithm.hyper_r1_invalid_action_penalty=${hyper_r1_invalid_action_penalty} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.dtype=float16 \
    actor_rollout_ref.ref.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16 \
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    algorithm.filter_groups.enable=false \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=true \
    data.truncation='left' \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.max_start_length=2048 \
    data.max_obs_length=3072 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.3 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    trainer.logger=['console','tensorboard'] \
    trainer.resume_mode="auto" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VALIDATION_DUMP_DIR}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME//\//-}" \
    trainer.n_gpus_per_node=${TRAINING_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=false \
    trainer.log_val_generations=10 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    sexpr_config.enable_sexpr_mode=true \
    sexpr_config.enable_action_reasoning=true \
    sexpr_config.enable_relation_retrieval=true \
    sexpr_config.validation_level=${SEXPR_VALIDATION_LEVEL} \
    sexpr_config.max_function_calls=10 \
    sexpr_config.enable_entity_linking=true \
    sexpr_config.enable_semantic_validation=true \
    sexpr_config.use_complete_sparql_converter=true \
    hyper_r1.enable=${hyper_r1_enable} \
    hyper_r1.max_active=${hyper_r1_max_active} \
    hyper_r1.max_nodes=${hyper_r1_max_nodes} \
    hyper_r1.frontier_width=${hyper_r1_frontier_width} \
    hyper_r1.max_frontier_width=${hyper_r1_max_frontier_width} \
    hyper_r1.relation_model="${hyper_r1_relation_model}" \
    +sexpr_config.enable_logging=false \
    max_turns=${max_turns} \
    use_odbc=true \
    use_aioodbc=false \
    odbc_config.driver_path="Virtuoso" \
    odbc_config.host="localhost" \
    odbc_config.port=13001 \
    odbc_config.uid="dba" \
    odbc_config.pwd="dba" \
    odbc_config.pool_size=48 \
    odbc_config.max_pool_size=48 \
    odbc_config.pool_timeout=30 \
    odbc_config.query_timeout=600 \
    odbc_config.max_concurrent=32 \
    odbc_config.max_retries=1 \
    odbc_config.retry_delay=1.0 \
    sparql_batch_size=128 \
    sparql_max_concurrent=16 \
    sparql.url="http://0.0.0.0:8000/execute" \
    reward_model.reward_manager=kbqa \
    reward_model.reward_kwargs.mid_f1_weight=1.0 \
    reward_model.reward_kwargs.structure_format_score=${structure_format_score} \
    reward_model.val_reward_kwargs.mid_f1_weight=1.0 \
    reward_model.val_reward_kwargs.structure_format_score=0.0 \
    custom_reward_function.path="${REPO_ROOT}/kbqa_custom_reward.py" \
    2>&1 | tee "${LOG_FILE}"

echo "========================================"
echo "KBQA-R1  training finished (exit code $?)"
echo "Logs        : ${LOG_FILE}"
echo "Checkpoints : ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo "========================================"
