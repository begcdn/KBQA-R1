#!/usr/bin/env bash
set -euo pipefail

# Build the actual HyPER-R1 behavior-cloning corpus. Demonstrations are derived
# from training-set gold programs, proposed through the normal relation
# retriever, executed against Freebase, and replay-validated before export.

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
: "${PROCESSED_GRAILQA:?Set PROCESSED_GRAILQA to processed GrailQA train JSON/parquet}"
OUTPUT=${OUTPUT:-"${REPO_ROOT}/data/grailqa_hyper_r1_demonstrations"}
MAX_DEMONSTRATIONS=${MAX_DEMONSTRATIONS:-3000}
MAX_INPUT_ROWS=${MAX_INPUT_ROWS:-20000}

exec python3 "${REPO_ROOT}/scripts/data_process/build_hyper_demonstrations.py" \
  --input "${PROCESSED_GRAILQA}" \
  --output "${OUTPUT}" \
  --max-demonstrations "${MAX_DEMONSTRATIONS}" \
  --max-input-rows "${MAX_INPUT_ROWS}" \
  --relation-topk 20 \
  --frontier-width 3 \
  --max-active 6 \
  --max-turns 10
