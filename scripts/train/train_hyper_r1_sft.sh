#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
: "${BASE_MODEL:?Set BASE_MODEL to the Llama-3.1-8B-Instruct backbone}"
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
if report.get("quality_schema") != "hyper_r1_v12_public_count_prune_parity":
    raise SystemExit(
        "HyPER-R1 corpus predates public Count-aware Prune parity; "
        "rebuild it."
    )
expected_contract = {
    "relation_page_size": 6,
    "relation_rank_cutoff": None,
    "max_active": 24,
    "max_nodes": 128,
    "max_execution_attempts": 24,
    "max_turns": 32,
}
contract_mismatches = {
    key: {"expected": expected, "actual": report.get(key)}
    for key, expected in expected_contract.items()
    if report.get(key) != expected
}
if contract_mismatches:
    raise SystemExit(
        "HyPER-R1 corpus does not match the training/inference contract: "
        + json.dumps(contract_mismatches, sort_keys=True)
    )
quality = report.get("quality_assessment", {})
if not quality.get("structurally_ready_for_sft", False):
    failed = [
        name for name, passed in quality.get("checks", {}).items() if not passed
    ]
    raise SystemExit(
        "HyPER-R1 corpus is not structurally ready for SFT: " + ", ".join(failed)
    )
if quality.get("deep_training_trajectories", 0) < quality.get(
    "minimum_deep_trajectories", 1
):
    raise SystemExit("HyPER-R1 corpus lacks enough verified deep trajectories for SFT")
print("HyPER-R1 corpus passed structural checks.")
print(json.dumps({
    "accepted_demonstrations": report.get("accepted_demonstrations"),
    "training_demonstrations": report.get("training_demonstrations"),
    "validation_rows": report.get("validation_rows"),
    "families": report.get("families"),
    "proposal_recall": report.get("proposal_recall"),
    "deep_training_trajectories": quality.get("deep_training_trajectories"),
    "minimum_deep_trajectories": quality.get("minimum_deep_trajectories"),
}, indent=2))
PY

export TRAIN_PARQUET="${HYPER_SFT}"
export VAL_PARQUET=${VAL_PARQUET:-"$(dirname "${HYPER_SFT}")/validation_decision.parquet"}
export DATASET_TYPE=${DATASET_TYPE:-grailqa}
export PROJECT_NAME=${PROJECT_NAME:-HyPER-R1-SFT}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grailqa-hyper-r1-sft}

exec "${REPO_ROOT}/scripts/train/run_sft_from_rejection.sh"
