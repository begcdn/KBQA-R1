#!/usr/bin/env python3
"""Regenerate HyPER-R1 SFT from structured demonstrations.

This exporter adds only transitions whose legality can be reconstructed from
the stored graph state: reversible Park/Recall recovery, budget-exhausted
Abstain, and masked invalid-action recovery contexts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_data import (  # noqa: E402
    DemonstrationStep,
    HyperDemonstration,
    decision_sft_records,
)
from scripts.data_process.build_hyper_demonstrations import _question_split  # noqa: E402
from scripts.data_process.repair_hyper_curriculum import _load_demo  # noqa: E402


_EXECUTING_ACTIONS = {
    "Inspect",
    "Combine",
    "Merge",
    "Order",
    "Compare",
    "Time_constraint",
    "Count",
}

_SOURCE_CONTRACT_FIELDS = (
    "quality_schema",
    "relation_page_size",
    "relation_rank_cutoff",
    "max_active",
    "max_nodes",
    "max_execution_attempts",
    "max_turns",
    "quality_assessment",
    "proposal_recall",
    "families",
)


def _load_source_contract(input_path: Path) -> dict:
    report_path = input_path.parent / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"source corpus contract is missing: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing = [field for field in _SOURCE_CONTRACT_FIELDS if field not in report]
    if missing:
        raise ValueError(
            "source corpus report is missing contract fields: "
            + ", ".join(missing)
        )
    contract = {field: report[field] for field in _SOURCE_CONTRACT_FIELDS}
    contract["report"] = str(report_path)
    contract["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return contract


def _read_demos(path: Path) -> list[HyperDemonstration]:
    with path.open(encoding="utf-8") as handle:
        return [_load_demo(json.loads(line)) for line in handle if line.strip()]


def _iter_demos(path: Path) -> Iterator[HyperDemonstration]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _load_demo(json.loads(line))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _augment_recall(demo: HyperDemonstration) -> tuple[HyperDemonstration, bool]:
    """Park the preserved branch during a real probe, then Recall it."""
    steps = list(demo.steps)
    max_turns = int(demo.private_metadata.get("max_turns", 24))
    if len(steps) + 2 > max_turns:
        return demo, False

    for start, step in enumerate(steps):
        if step.action != "Select" or step.supervision != "intervention":
            continue
        resolution_seen = False
        resume = None
        resume_index = None
        for index in range(start + 1, len(steps)):
            candidate = steps[index]
            resolution_seen = resolution_seen or candidate.action in {"Park", "Prune"}
            if resolution_seen and candidate.action == "Select":
                resume = candidate.arguments[0]
                resume_index = index
                break
            if candidate.action in {"Commit", "Abstain"}:
                break
        if resume is None or resume_index is None:
            continue
        if resume == step.arguments[0] or resume not in step.visible_before:
            continue

        parked_state = tuple(value for value in step.visible_before if value != resume)
        rewritten = steps[:start]
        rewritten.append(
            DemonstrationStep(
                "Park",
                (resume,),
                step.visible_before,
                (),
                (
                    "preserve_required_branch_during_probe",
                    "storage_only_not_semantic_rejection",
                ),
            )
        )
        for index in range(start, resume_index):
            current = steps[index]
            rewritten.append(
                replace(
                    current,
                    visible_before=tuple(
                        value for value in current.visible_before if value != resume
                    ),
                )
            )
        resume_step = steps[resume_index]
        recall_before = tuple(
            value for value in resume_step.visible_before if value != resume
        )
        rewritten.append(
            DemonstrationStep(
                "Recall",
                (resume,),
                recall_before,
                (),
                ("restore_preserved_branch_after_probe",),
            )
        )
        rewritten.extend(steps[resume_index:])
        metadata = dict(demo.private_metadata)
        metadata["recall_augmented"] = True
        return replace(demo, steps=rewritten, private_metadata=metadata), True
    return demo, False


def _budget_abstain_demo(
    demo: HyperDemonstration,
) -> HyperDemonstration | None:
    """Create a terminal state where the next required Inspect is unaffordable."""
    executions = 0
    for index, step in enumerate(demo.steps):
        if step.action == "Inspect" and executions > 0:
            complete = any(
                demo.hypotheses[node_id].denotation == demo.gold_answers
                for node_id in step.visible_before
                if node_id in demo.hypotheses
            )
            if not complete:
                metadata = dict(demo.private_metadata)
                metadata.update(
                    {
                        "max_execution_attempts": executions,
                        "max_turns": index + 1,
                        "execution_attempts": executions,
                        "verified_abstain_reason": "execution_budget_exhausted",
                    }
                )
                abstain = DemonstrationStep(
                    "Abstain",
                    (),
                    step.visible_before,
                    (),
                    ("execution_budget_exhausted_without_complete_hypothesis",),
                )
                retained_steps = [*demo.steps[:index], abstain]
                retained_hypotheses = {
                    node_id
                    for retained_step in retained_steps
                    for node_id in (*retained_step.visible_before, *retained_step.created)
                    if node_id in demo.hypotheses
                }
                retained_proposals = {
                    proposal_id
                    for retained_step in retained_steps
                    for proposal_id in retained_step.exposed
                    if proposal_id in demo.proposals
                }
                return replace(
                    demo,
                    demo_id=f"{demo.demo_id}:budget_abstain:{index}",
                    family="budget_exhausted_abstain",
                    hypotheses={
                        node_id: demo.hypotheses[node_id]
                        for node_id in sorted(retained_hypotheses)
                    },
                    steps=retained_steps,
                    private_metadata=metadata,
                    proposals={
                        proposal_id: demo.proposals[proposal_id]
                        for proposal_id in sorted(retained_proposals)
                    },
                )
        executions += int(step.action in _EXECUTING_ACTIONS)
    return None


def _target_action(row: dict) -> str:
    content = str(row["messages"][-1]["content"])
    match = re.search(r"<action>\s*([^\s\[]+)", content)
    if match is None:
        raise ValueError("decision row has no target action")
    return match.group(1)


def _invalid_recovery_record(row: dict, ordinal: int) -> dict:
    """Insert a rejected action with zero loss, then retain the valid target."""
    messages = [dict(message) for message in row["messages"]]
    if len(messages) < 2 or messages[-2].get("role") != "user":
        raise ValueError("decision record lacks a current user observation")
    observation = str(messages[-2]["content"])
    if "<information>" not in observation:
        raise ValueError("invalid recovery requires an environment observation")
    if "<proposal_catalog>" in observation:
        invalid_action = "Inspect [ P999999 ]"
        failure = "unknown or unexposed proposal: P999999"
    else:
        invalid_action = "Select [ H999999 ]"
        failure = "unknown hypothesis: H999999"
    failed_observation = observation.replace(
        "<information>",
        f"<information>\nGraph action failed: {failure}",
        1,
    )
    messages[-1:-1] = [
        {
            "role": "assistant",
            "content": f"<think>I must use only currently listed IDs.</think>\n<action>{invalid_action}</action>",
            "loss_mask": 0,
        },
        {"role": "user", "content": failed_observation, "loss_mask": 0},
    ]
    extra = dict(row.get("extra_info", {}))
    extra.update(
        {
            "invalid_recovery": True,
            "invalid_recovery_ordinal": ordinal,
        }
    )
    return {**row, "messages": messages, "extra_info": extra}


def _supports_invalid_recovery(row: dict) -> bool:
    messages = row.get("messages", ())
    return bool(
        len(messages) >= 2
        and messages[-2].get("role") == "user"
        and "<information>" in str(messages[-2].get("content", ""))
    )


def _balanced_invalid_recoveries(rows: list[dict], limit: int) -> list[dict]:
    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        action = _target_action(row)
        if action not in {"Commit", "Abstain"} and _supports_invalid_recovery(row):
            by_action[action].append(row)
    selected = []
    while len(selected) < limit and any(by_action.values()):
        for action in sorted(by_action):
            if by_action[action] and len(selected) < limit:
                selected.append(by_action[action].pop(0))
    return [_invalid_recovery_record(row, index) for index, row in enumerate(selected)]


def _decision_contradictions(rows: list[dict]) -> dict:
    state_actions: dict[str, set[str]] = {}
    for row in rows:
        history = []
        for message in row["messages"]:
            content = str(message.get("content", ""))
            if (
                message.get("role") == "assistant"
                and message.get("loss_mask") == 1
                and "<action>" in content
            ):
                key = json.dumps(history, sort_keys=True, ensure_ascii=True)
                state_actions.setdefault(key, set()).add(content)
            history.append(message)
    conflicts = sum(len(actions) > 1 for actions in state_actions.values())
    return {
        "decision_states": len(state_actions),
        "contradictory_states": conflicts,
        "contradiction_rate": conflicts / len(state_actions) if state_actions else 0.0,
    }


def _normalized_decision_row(row: dict) -> dict:
    extra = dict(row.get("extra_info", {}))
    extra.setdefault("recovery_stratum", None)
    extra.setdefault("invalid_recovery", False)
    extra.setdefault("invalid_recovery_ordinal", -1)
    return {**row, "extra_info": extra}


def _decision_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                            pa.field("loss_mask", pa.int64()),
                        ]
                    )
                ),
            ),
            pa.field("data_source", pa.string()),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("demo_id", pa.string()),
                        pa.field("question_id", pa.string()),
                        pa.field("family", pa.string()),
                        pa.field("recovery_stratum", pa.string()),
                        pa.field("replay_verified", pa.bool_()),
                        pa.field("gold_injected_into_proposals", pa.bool_()),
                        pa.field("terminal_action", pa.string()),
                        pa.field("decision_index", pa.int64()),
                        pa.field("trajectory_step_index", pa.int64()),
                        pa.field("target_is_graph_action", pa.bool_()),
                        pa.field("invalid_recovery", pa.bool_()),
                        pa.field("invalid_recovery_ordinal", pa.int64()),
                    ]
                ),
            ),
        ]
    )


class _ParquetSink:
    """Write bounded compressed row groups and publish only a complete file."""

    def __init__(self, path: Path, batch_size: int = 512):
        import pyarrow.parquet as pq

        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.temporary.unlink(missing_ok=True)
        self.schema = _decision_schema()
        self.writer = pq.ParquetWriter(
            self.temporary,
            self.schema,
            compression="snappy",
            use_dictionary=True,
        )
        self.batch_size = int(batch_size)
        self.rows: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.rows.append(_normalized_decision_row(row))
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa

        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        self.writer.close()
        self.temporary.unlink(missing_ok=True)


class _DecisionConsistency:
    def __init__(self):
        self.actions: dict[bytes, str] = {}
        self.conflicts: set[bytes] = set()

    def add(self, row: dict) -> None:
        messages = row["messages"]
        if not messages or messages[-1].get("loss_mask") != 1:
            raise ValueError("decision row must end in exactly one trained action")
        state = json.dumps(messages[:-1], sort_keys=True, ensure_ascii=True)
        key = hashlib.sha256(state.encode()).digest()
        action = str(messages[-1]["content"])
        previous = self.actions.setdefault(key, action)
        if previous != action:
            self.conflicts.add(key)

    def report(self) -> dict:
        states = len(self.actions)
        conflicts = len(self.conflicts)
        return {
            "decision_states": states,
            "contradictory_states": conflicts,
            "contradiction_rate": conflicts / states if states else 0.0,
        }


def _valid_recall_augmentation(demo: HyperDemonstration) -> bool:
    parked = set()
    recalls = 0
    for step in demo.steps:
        if step.action == "Park":
            parked.add(step.arguments[0])
        elif step.action == "Recall":
            node_id = step.arguments[0]
            if node_id not in parked or node_id not in demo.hypotheses:
                return False
            parked.remove(node_id)
            recalls += 1
    return recalls == 1


def _valid_budget_abstain(demo: HyperDemonstration) -> bool:
    if not demo.steps or demo.steps[-1].action != "Abstain":
        return False
    executions = sum(step.action in _EXECUTING_ACTIONS for step in demo.steps[:-1])
    visible = demo.steps[-1].visible_before
    complete_visible = any(
        demo.hypotheses[node_id].denotation == demo.gold_answers
        for node_id in visible
        if node_id in demo.hypotheses
    )
    return (
        executions > 0
        and int(demo.private_metadata.get("max_execution_attempts", -1)) == executions
        and demo.private_metadata.get("verified_abstain_reason")
        == "execution_budget_exhausted"
        and not complete_visible
    )


def _valid_invalid_recovery(row: dict) -> bool:
    messages = row["messages"]
    return (
        len(messages) >= 3
        and messages[-3].get("role") == "assistant"
        and messages[-3].get("loss_mask") == 0
        and "999999" in str(messages[-3].get("content", ""))
        and messages[-2].get("role") == "user"
        and messages[-2].get("loss_mask") == 0
        and "Graph action failed:" in str(messages[-2].get("content", ""))
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("loss_mask") == 1
    )


def _regenerate_unlocked(input_path: Path, output: Path) -> dict:
    source_contract = _load_source_contract(input_path)
    source_demonstrations = 0
    action_counts: Counter = Counter()
    for demo in _iter_demos(input_path):
        source_demonstrations += 1
        action_counts.update(step.action for step in demo.steps)
    rare_action_target = max(action_counts["Widen"], action_counts["Prune"], 1)
    output.mkdir(parents=True, exist_ok=True)
    demo_path = output / "demonstrations.jsonl"
    demo_temporary = demo_path.with_suffix(demo_path.suffix + ".tmp")
    demo_temporary.unlink(missing_ok=True)
    train_sink = _ParquetSink(output / "train_decision.parquet")
    validation_sink = _ParquetSink(output / "validation_decision.parquet")
    consistency = _DecisionConsistency()
    eligible_recovery_actions = sorted(
        (set(action_counts) | {"Recall"}) - {"Commit", "Abstain"}
    )
    per_action_limit = (
        rare_action_target + len(eligible_recovery_actions) - 1
    ) // len(eligible_recovery_actions)
    recovery_candidates: dict[str, list[dict]] = {
        action: [] for action in eligible_recovery_actions
    }
    recovery_overflow: list[dict] = []
    seen_abstain_questions = set()
    train_questions = set()
    validation_questions = set()
    final_actions: Counter = Counter()
    counters: Counter = Counter()
    recall_valid = True
    abstain_valid = True

    def write_decision(row: dict, *, collect_recovery: bool) -> None:
        split = _question_split(row["extra_info"]["question_id"])
        (train_sink if split == "train" else validation_sink).append(row)
        consistency.add(row)
        if collect_recovery:
            action = _target_action(row)
            candidates = recovery_candidates.get(action)
            if candidates is not None and _supports_invalid_recovery(row):
                if len(candidates) < per_action_limit:
                    candidates.append(row)
                elif len(recovery_overflow) < rare_action_target:
                    recovery_overflow.append(row)

    try:
        with demo_temporary.open("w", encoding="utf-8") as demo_handle:
            for source_demo in _iter_demos(input_path):
                updated, augmented = _augment_recall(source_demo)
                variants = [updated]
                counters["recall"] += int(augmented)
                if augmented:
                    recall_valid = recall_valid and _valid_recall_augmentation(updated)
                if (
                    counters["abstain"] < rare_action_target
                    and updated.question_id not in seen_abstain_questions
                ):
                    candidate = _budget_abstain_demo(updated)
                    if candidate is not None:
                        variants.append(candidate)
                        seen_abstain_questions.add(updated.question_id)
                        counters["abstain"] += 1
                        abstain_valid = abstain_valid and _valid_budget_abstain(candidate)

                for demo in variants:
                    split = _question_split(demo.question_id)
                    (train_questions if split == "train" else validation_questions).add(
                        demo.question_id
                    )
                    counters[f"{split}_demonstrations"] += 1
                    final_actions.update(step.action for step in demo.steps)
                    demo_handle.write(
                        json.dumps(demo.to_dict(), ensure_ascii=False) + "\n"
                    )
                    for row in decision_sft_records(demo):
                        write_decision(row, collect_recovery=True)

        selected = []
        while len(selected) < rare_action_target and any(recovery_candidates.values()):
            for action in eligible_recovery_actions:
                candidates = recovery_candidates[action]
                if candidates and len(selected) < rare_action_target:
                    selected.append(candidates.pop(0))
        selected.extend(recovery_overflow[: rare_action_target - len(selected)])
        invalid_rows = [
            _invalid_recovery_record(row, index)
            for index, row in enumerate(selected)
        ]
        invalid_count_complete = len(invalid_rows) == rare_action_target
        invalid_masks_valid = all(
            _valid_invalid_recovery(row) for row in invalid_rows
        )
        for row in invalid_rows:
            write_decision(row, collect_recovery=False)

        consistency_report = consistency.report()
        assertions = {
            "regenerated_from_structured_demonstrations": source_demonstrations > 0,
            "invalid_recovery_count_reached": invalid_count_complete,
            "invalid_actions_have_zero_loss": invalid_masks_valid,
            "recall_restores_a_parked_executable_hypothesis": recall_valid,
            "abstain_requires_exhausted_execution_budget": abstain_valid,
            "question_disjoint_split": not train_questions.intersection(
                validation_questions
            ),
        }
        if not all(assertions.values()) or consistency_report["contradictory_states"]:
            raise RuntimeError(
                "generated corpus failed a truthfulness invariant: "
                f"assertions={assertions}, consistency={consistency_report}"
            )

        train_sink.close()
        validation_sink.close()
        os.replace(demo_temporary, demo_path)
    except Exception:
        train_sink.abort()
        validation_sink.abort()
        demo_temporary.unlink(missing_ok=True)
        raise

    report = {
        "quality_schema": "hyper_r1_v17_truthful_control",
        "source": str(input_path),
        "source_contract": source_contract,
        "output": str(output),
        "source_demonstrations": source_demonstrations,
        "output_demonstrations": source_demonstrations + counters["abstain"],
        "recall_augmented_demonstrations": counters["recall"],
        "budget_exhausted_abstain_demonstrations": counters["abstain"],
        "masked_invalid_recovery_decisions": len(invalid_rows),
        "rare_action_reference_count": rare_action_target,
        "action_counts": dict(sorted(final_actions.items())),
        "train_demonstrations": counters["train_demonstrations"],
        "validation_demonstrations": counters["validation_demonstrations"],
        "train_decisions": train_sink.count,
        "validation_decisions": validation_sink.count,
        "decision_consistency": consistency_report,
        "assertions": assertions,
    }
    report_path = output / "report.json"
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(report_temporary, report_path)
    return report


def regenerate(input_path: Path, output: Path) -> dict:
    """Regenerate under an exclusive lock so concurrent exports cannot mix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another corpus export owns {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        return _regenerate_unlocked(input_path, output)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
