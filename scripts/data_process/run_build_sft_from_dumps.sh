#!/usr/bin/env bash
set -euo pipefail

# Launcher for build_sft_from_dumps.py
# Allows you to easily run the converter with environment-variable overrides
# Defaults chosen to match repository conventions used elsewhere in this project.



# Defaults (can override via env or CLI args below)
INPUT_DIR=${INPUT_DIR:-"/ossfs/workspace/kbqa-r1/data/graphq_rl_dataset_sft/rejection_sampling_graphq_20251031_155131/validation"}
# INPUT_DIR=${INPUT_DIR:-"/ossfs/workspace/kbqa-r1/data/webqsp_rl_dataset_sft/rejection_sampling_webqsp_20251030_201622/validation"}
# INPUT_DIR=${INPUT_DIR:-"/ossfs/workspace/kbqa-r1/data/graphq_rl_dataset_sft/rejection_sampling_graphq_20251202_011801/validation"}
# INPUT_DIR=${INPUT_DIR:-"/ossfs/workspace/kbqa-r1/data/grailqa_rl_dataset_sft/rejection_sampling_grailqa_20251201_051339/validation"}

python3 scripts/data_process/build_sft_from_dumps.py \
  --dump_dir "${INPUT_DIR}" \
  --output_file "/ossfs/workspace/kbqa-r1/data/graphq_rl_dataset_sft/train_sft___.parquet" \
  --min_mid_f1 0.9 \
  --structure_reward_eq 0.1 \
  --info_role user

