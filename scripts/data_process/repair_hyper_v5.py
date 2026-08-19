#!/usr/bin/env python3
"""Inspect the retired HyPER-v5 rationale repair.

HyPER-v5 cannot be upgraded to proof-carrying supervision without replaying the
graph and comparing its terminal program with the gold logical program.  The
historical rationale transformation remains available for audit, but export is
blocked for uncertified corpora rather than silently relabelling them as valid.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_data import (  # noqa: E402
    DemonstrationStep,
    HyperDemonstration,
    decision_sft_records,
    trajectory_sft_record,
)
from scripts.data_process.build_hyper_demonstrations import _question_split  # noqa: E402
from scripts.data_process.repair_hyper_curriculum import _load_demo  # noqa: E402


def _assert_only_rationales_changed(
    before: HyperDemonstration, after: HyperDemonstration
) -> None:
    before_payload = before.to_dict()
    after_payload = after.to_dict()
    if set(before_payload) != set(after_payload):
        raise AssertionError("repair changed demonstration fields")
    for field in before_payload:
        if field != "steps" and before_payload[field] != after_payload[field]:
            raise AssertionError(f"repair changed field: {field}")
    if len(before.steps) != len(after.steps):
        raise AssertionError("repair changed the number of steps")
    for old_step, new_step in zip(before.steps, after.steps):
        old_payload = old_step.__dict__.copy()
        new_payload = new_step.__dict__.copy()
        old_payload.pop("rationale_facts")
        new_payload.pop("rationale_facts")
        if old_payload != new_payload or old_step.action != new_step.action:
            raise AssertionError("repair changed a step field or action")


def repair_demonstration(
    demo: HyperDemonstration,
) -> Tuple[HyperDemonstration, Counter]:
    counts: Counter = Counter()
    repaired_steps: List[DemonstrationStep] = []
    max_active = int(demo.private_metadata.get("max_active", 6))
    for step in demo.steps:
        facts = step.rationale_facts
        mismatch_facts = tuple(
            fact for fact in facts if fact.startswith("question_path_mismatch:")
        )
        if step.action != "Prune" or not mismatch_facts:
            repaired_steps.append(step)
            continue
        node = demo.hypotheses[step.arguments[0]]
        if not node.denotation:
            counts["empty_adaptive_or_other_mismatch_unchanged"] += 1
            repaired_steps.append(step)
            continue
        counts["nonempty_mismatch_prunes_seen"] += 1
        if demo.family != "adaptive_frontier_widen":
            if demo.family == "semantic_frontier_recovery":
                counts["semantic_recovery_mismatch_unchanged"] += 1
            else:
                counts["non_adaptive_mismatch_unchanged"] += 1
            repaired_steps.append(step)
            continue

        fact = (
            "frontier_capacity_eviction"
            if len(step.visible_before) == max_active
            else "frontier_capacity_reservation"
        )
        new_facts = tuple(
            fact if old_fact.startswith("question_path_mismatch:") else old_fact
            for old_fact in facts
        )
        repaired_steps.append(replace(step, rationale_facts=new_facts))
        counts[f"converted_to_{fact.removeprefix('frontier_capacity_')}"] += 1

    repaired = replace(demo, steps=repaired_steps)
    _assert_only_rationales_changed(demo, repaired)
    return repaired, counts


def _open_jsonl(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _read_demos(path: Path) -> List[HyperDemonstration]:
    with _open_jsonl(path, "r") as handle:
        return [_load_demo(json.loads(line)) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_compatible_report(
    source_report_path: Path | None,
    output: Path,
    repair_report: Dict[str, Any],
) -> None:
    if source_report_path is None:
        return
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_report["capacity_rationale_repair"] = repair_report
    source_report["training_demonstrations"] = repair_report[
        "train_trajectory_rows"
    ]
    source_report["sft_rows"] = repair_report["train_trajectory_rows"]
    source_report["decision_sft_rows"] = repair_report["train_decision_rows"]
    source_report["validation_rows"] = repair_report[
        "validation_trajectory_rows"
    ]
    (output / "report.json").write_text(
        json.dumps(source_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_repaired_corpus(
    input_path: Path,
    output: Path,
    source_report_path: Path | None = None,
) -> Dict[str, Any]:
    demos = _read_demos(input_path)
    uncertified = [
        (demo.demo_id, step.action)
        for demo in demos
        for step in demo.steps
        if step.action in {"Prune", "Commit"} and not step.certificate_kind
    ]
    if uncertified:
        examples = ", ".join(
            f"{demo_id}:{action}" for demo_id, action in uncertified[:3]
        )
        raise RuntimeError(
            "HyPER-v5 cannot be made proof-carrying by a rationale-only repair; "
            "rebuild the corpus with graph replay and logical-program certificates "
            f"(uncertified examples: {examples})"
        )
    repaired: List[HyperDemonstration] = []
    counts: Counter = Counter()
    for demo in demos:
        fixed, demo_counts = repair_demonstration(demo)
        repaired.append(fixed)
        counts.update(demo_counts)

    trajectory_rows = [trajectory_sft_record(demo) for demo in repaired]
    decision_rows = [
        row for demo in repaired for row in decision_sft_records(demo)
    ]
    train_pairs = [
        (demo, row)
        for demo, row in zip(repaired, trajectory_rows)
        if _question_split(demo.question_id) == "train"
    ]
    validation_pairs = [
        (demo, row)
        for demo, row in zip(repaired, trajectory_rows)
        if _question_split(demo.question_id) == "validation"
    ]
    train_demos = [demo for demo, _ in train_pairs]
    validation_demos = [demo for demo, _ in validation_pairs]
    train_trajectory = [row for _, row in train_pairs]
    validation_trajectory = [row for _, row in validation_pairs]
    train_decision = [
        row for demo in train_demos for row in decision_sft_records(demo)
    ]
    validation_decision = [
        row for demo in validation_demos for row in decision_sft_records(demo)
    ]

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "demonstrations.jsonl", (demo.to_dict() for demo in repaired))
    _write_jsonl(output / "trajectory_sft.jsonl", trajectory_rows)
    _write_jsonl(output / "decision_sft.jsonl", decision_rows)
    _write_jsonl(output / "train_trajectory_sft.jsonl", train_trajectory)
    _write_jsonl(output / "validation_trajectory_sft.jsonl", validation_trajectory)
    _write_jsonl(output / "train_decision_sft.jsonl", train_decision)
    _write_jsonl(output / "validation_decision_sft.jsonl", validation_decision)

    from datasets import Dataset

    Dataset.from_list(train_trajectory).to_parquet(str(output / "train.parquet"))
    Dataset.from_list(validation_trajectory).to_parquet(
        str(output / "validation.parquet")
    )
    Dataset.from_list(train_decision).to_parquet(
        str(output / "train_decision.parquet")
    )
    Dataset.from_list(validation_decision).to_parquet(
        str(output / "validation_decision.parquet")
    )

    split_by_question = {}
    for demo in repaired:
        split = _question_split(demo.question_id)
        prior = split_by_question.setdefault(demo.question_id, split)
        if prior != split:
            raise AssertionError("question appears in both train and validation")

    report = {
        "input": str(input_path),
        "output": str(output),
        "input_demonstrations": len(demos),
        "output_demonstrations": len(repaired),
        "repair_counts": dict(sorted(counts.items())),
        "converted_total": sum(
            value
            for key, value in counts.items()
            if key.startswith("converted_to_")
        ),
        "semantic_recovery_mismatch_prunes_unchanged": counts[
            "semantic_recovery_mismatch_unchanged"
        ],
        "train_trajectory_rows": len(train_trajectory),
        "validation_trajectory_rows": len(validation_trajectory),
        "train_decision_rows": len(train_decision),
        "validation_decision_rows": len(validation_decision),
        "question_disjoint_split": "sha256_question_id_95_5",
        "graph_execution_performed": False,
        "validation_and_serialization": "current trajectory and decision serializers",
        "assertions": {
            "only_rationale_facts_changed": True,
            "actions_unchanged": True,
            "question_disjoint_train_validation": True,
        },
    }
    (output / "repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_compatible_report(source_report_path, output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-report",
        type=Path,
        help="Copy the audited source report and append this repair record.",
    )
    args = parser.parse_args()
    input_path = args.input
    if input_path.is_dir():
        input_path = input_path / "demonstrations.jsonl"
        if not input_path.exists() and input_path.with_suffix(".jsonl.gz").exists():
            input_path = input_path.with_suffix(".jsonl.gz")
    report = export_repaired_corpus(
        input_path,
        args.output,
        source_report_path=args.source_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
