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
schema = report.get("quality_schema")
if report.get("schema_version") == "hyper-corrective-assembly-v1":
    required = {
        "question_disjoint": True,
        "test_question_overlap": 0,
        "student_visible_recovery_marker": False,
        "fixed_state_weight": 1.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in required.items()
        if report.get(key) != expected
    }
    mixture = report.get("mixture_percent", {})
    actual = report.get("actual_rows", {})
    if sum(mixture.values()) != 100:
        mismatches["mixture_percent"] = mixture
    if mixture.get("semantic_recovery", 0) <= 0:
        mismatches["semantic_recovery"] = mixture.get("semantic_recovery")
    for partition in ("train", "dev"):
        expected_rows = report.get("requested_rows", {}).get(partition)
        if sum(actual.get(partition, {}).values()) != expected_rows:
            mismatches[f"actual_rows.{partition}"] = actual.get(partition)
    expected_outputs = {"train_decision.parquet", "dev_decision.parquet"}
    if not expected_outputs.issubset(report.get("output_sha256", {})):
        mismatches["output_sha256"] = sorted(report.get("output_sha256", {}))
    if mismatches:
        raise SystemExit(
            "HyPER-R1 corrective corpus failed its invariants: "
            + json.dumps(mismatches, sort_keys=True)
        )
    print("HyPER-R1 corrective corpus passed structural checks.")
    print(json.dumps({
        "schema_version": report["schema_version"],
        "mixture_percent": mixture,
        "actual_rows": actual,
        "question_counts": report.get("question_counts"),
    }, indent=2))
    raise SystemExit(0)
if schema == "hyper_r1_v23_runtime_comparison":
    assertions = report.get("assertions", {})
    failed_assertions = [name for name, passed in assertions.items() if not passed]
    contradictions = report.get("decision_consistency", {}).get(
        "contradictory_states", -1
    )
    if failed_assertions or contradictions != 0:
        raise SystemExit(
            "HyPER-R1 runtime-comparison corpus failed its invariants: "
            + json.dumps(
                {
                    "failed_assertions": failed_assertions,
                    "contradictory_states": contradictions,
                },
                sort_keys=True,
            )
        )
    structural_report = report.get("source_contract", {})
    if structural_report.get("quality_schema") != "hyper_r1_v20_f1_control":
        raise SystemExit("HyPER-R1 v23 source contract is invalid")
elif schema in {
    "hyper_r1_v20_f1_control",
    "hyper_r1_v17_truthful_control",
    "hyper_r1_v18_runtime_aligned_control",
    "hyper_r1_v19_runtime_exact_control",
    "hyper_r1_v13_state_covered_recovery",
}:
    raise SystemExit(
        "HyPER-R1 corpus schema " + str(schema) +
        " is superseded; regenerate the F1-aligned v20 corpus before SFT."
    )
else:
    raise SystemExit(
        "HyPER-R1 corpus is not the validated state-covered recovery corpus; "
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
    key: {"expected": expected, "actual": structural_report.get(key)}
    for key, expected in expected_contract.items()
    if structural_report.get(key) != expected
}
if contract_mismatches:
    raise SystemExit(
        "HyPER-R1 corpus does not match the training/inference contract: "
        + json.dumps(contract_mismatches, sort_keys=True)
    )
quality = structural_report.get("quality_assessment", {})
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
    "quality_schema": schema,
    "training_demonstrations": report.get(
        "train_demonstrations", structural_report.get("training_demonstrations")
    ),
    "validation_rows": report.get(
        "validation_decisions", structural_report.get("validation_rows")
    ),
    "families": structural_report.get("families"),
    "proposal_recall": structural_report.get("proposal_recall"),
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
