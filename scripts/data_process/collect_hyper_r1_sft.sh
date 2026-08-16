#!/usr/bin/env bash
set -euo pipefail

# Build the actual HyPER-R1 behavior-cloning corpus. Demonstrations are derived
# from training-set gold programs, proposed through the normal relation
# retriever, executed against Freebase, and replay-validated before export.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
: "${PROCESSED_GRAILQA:?Set PROCESSED_GRAILQA to processed GrailQA train JSON/parquet}"
: "${HYPER_RELATION_MODEL:?Set HYPER_RELATION_MODEL to the explicit relation-ranker checkpoint}"
OUTPUT=${OUTPUT:-"${REPO_ROOT}/data/grailqa_hyper_r1_demonstrations"}
# Zero means the complete training split/corpus. Set small positive values only
# for an explicit smoke test.
MAX_DEMONSTRATIONS=${MAX_DEMONSTRATIONS:-0}
MAX_INPUT_ROWS=${MAX_INPUT_ROWS:-0}
HYPER_BUILD_WORKERS=${HYPER_BUILD_WORKERS:-8}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARNING}

exec python3 "${REPO_ROOT}/scripts/data_process/build_hyper_demonstrations.py" \
  --input "${PROCESSED_GRAILQA}" \
  --output "${OUTPUT}" \
  --relation-model "${HYPER_RELATION_MODEL}" \
  --max-demonstrations "${MAX_DEMONSTRATIONS}" \
  --max-input-rows "${MAX_INPUT_ROWS}" \
  --workers "${HYPER_BUILD_WORKERS}" \
  --relation-topk 20 \
  --frontier-width 3 \
  --max-frontier-width 6 \
  --max-active 6 \
  --max-turns 14
