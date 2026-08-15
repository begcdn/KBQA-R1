#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
: "${BASE_MODEL:?Set BASE_MODEL to the released KBQA-R1 SFT checkpoint}"
HYPER_SFT=${HYPER_SFT:-"${REPO_ROOT}/data/grailqa_hyper_r1_demonstrations/train_decision.parquet"}
HYPER_REPORT=${HYPER_REPORT:-"$(dirname "${HYPER_SFT}")/report.json"}
if [[ ! -f "${HYPER_SFT}" ]]; then
  echo "Missing verified HyPER-R1 trajectory data: ${HYPER_SFT}" >&2
  echo "Run scripts/data_process/build_hyper_demonstrations.py first." >&2
  exit 1
fi
if [[ ! -f "${HYPER_REPORT}" ]]; then
  echo "Missing HyPER-R1 corpus report: ${HYPER_REPORT}" >&2
  exit 1
fi
python3 - "${HYPER_REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
quality = report.get("quality_assessment", {})
if not quality.get("structurally_ready_for_sft", False):
    failed = [
        name for name, passed in quality.get("checks", {}).items() if not passed
    ]
    raise SystemExit(
        "HyPER-R1 corpus is not structurally ready for SFT: " + ", ".join(failed)
    )
print("HyPER-R1 corpus passed structural checks.")
print(json.dumps({
    "accepted_demonstrations": report.get("accepted_demonstrations"),
    "training_demonstrations": report.get("training_demonstrations"),
    "validation_rows": report.get("validation_rows"),
    "families": report.get("families"),
    "proposal_recall": report.get("proposal_recall"),
}, indent=2))
PY

export TRAIN_PARQUET="${HYPER_SFT}"
export VAL_PARQUET=${VAL_PARQUET:-"$(dirname "${HYPER_SFT}")/validation_decision.parquet"}
export DATASET_TYPE=${DATASET_TYPE:-grailqa}
export PROJECT_NAME=${PROJECT_NAME:-HyPER-R1-SFT}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grailqa-hyper-r1-sft}

exec "${REPO_ROOT}/scripts/train/run_sft_from_rejection.sh"
