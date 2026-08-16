#!/usr/bin/env bash
set -euo pipefail

# Native HyPER-R1 training entrypoint. It requires the policy checkpoint from
# verified frontier SFT, then augments the ordinary RL prompts before GRPO.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
DATASET_TYPE=${DATASET_TYPE:-grailqa}
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset"}
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${HYPER_SFT_MODEL:?Set HYPER_SFT_MODEL to the verified HyPER-R1 SFT checkpoint}"
: "${HYPER_RELATION_MODEL:?Set HYPER_RELATION_MODEL to the explicit relation-ranker checkpoint}"
: "${HYPER_SFT_SCREEN:?Set HYPER_SFT_SCREEN to the behavioral screen metrics.json}"

"${PYTHON_BIN}" - "${HYPER_SFT_SCREEN}" "${HYPER_SFT_MODEL}" <<'PY'
import json
import os
import sys

screen_path, model_path = sys.argv[1:]
payload = json.load(open(screen_path, encoding="utf-8"))
expected = os.path.realpath(model_path)
matches = []
for row in payload.get("checkpoints", []):
    candidates = {
        os.path.realpath(str(row.get("checkpoint", ""))),
        os.path.realpath(str(row.get("resolved_model_dir", ""))),
    }
    if expected in candidates or any(candidate.startswith(expected + os.sep) for candidate in candidates):
        matches.append(row)
if len(matches) != 1:
    raise SystemExit(
        f"Behavioral screen does not identify HYPER_SFT_MODEL exactly: {model_path}"
    )
screen = matches[0].get("minimum_behavior_screen", {})
if not screen.get("passed", False):
    failed = [name for name, passed in screen.get("checks", {}).items() if not passed]
    raise SystemExit(
        "HyPER-R1 SFT checkpoint failed the minimum behavioral screen: "
        + ", ".join(failed)
    )
print("HyPER-R1 SFT checkpoint passed the minimum behavioral screen.")
print("It still requires executable held-out evaluation during GRPO validation.")
PY

RAW_TRAIN_FILE=${RAW_TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
RAW_VAL_FILE=${RAW_VAL_FILE:-"${DATA_DIR}/test.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train_hyper_r1.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test_hyper_r1.parquet"}

for pair in "${RAW_TRAIN_FILE}:${TRAIN_FILE}" "${RAW_VAL_FILE}:${VAL_FILE}"; do
  source_file=${pair%%:*}
  output_file=${pair#*:}
  if [[ ! -f "${output_file}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/data_process/enable_hyper_r1_prompts.py" \
      --input "${source_file}" --output "${output_file}"
  fi
done

export DATASET_TYPE DATA_DIR TRAIN_FILE VAL_FILE
export MODEL_PATH=${MODEL_PATH:-"${HYPER_SFT_MODEL}"}

export hyper_r1_enable=true
export hyper_r1_max_active=${hyper_r1_max_active:-6}
export hyper_r1_max_nodes=${hyper_r1_max_nodes:-24}
export hyper_r1_frontier_width=${hyper_r1_frontier_width:-3}
export hyper_r1_max_frontier_width=${hyper_r1_max_frontier_width:-6}
export hyper_r1_relation_model=${hyper_r1_relation_model:-"${HYPER_RELATION_MODEL}"}
export hyper_r1_credit_weight=${hyper_r1_credit_weight:-1.0}
export hyper_r1_budget_cost=${hyper_r1_budget_cost:-0.05}
export hyper_r1_invalid_commit_penalty=${hyper_r1_invalid_commit_penalty:-0.25}
export hyper_r1_invalid_action_penalty=${hyper_r1_invalid_action_penalty:-0.25}
export max_turns=${max_turns:-14}
# The released format bonus validates the old linear S-expression protocol.
# HyPER's structured actions are enforced by the environment itself, so that
# bonus would reward the wrong syntax.
export structure_format_score=${structure_format_score:-0.0}

exec "$(dirname "$0")/train_kbqa_sexpr_generation_grpo.sh" "${1:-hyper-r1}"
