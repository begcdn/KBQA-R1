#!/usr/bin/env python3
"""Measure recoverable HyPER states under a bounded gold continuation oracle.

This is an audit, not a corpus generator.  It restores exact pre-action
snapshots from failed rollouts and asks whether the production transition
system can still reach an explicit answer-exact, intent-equivalent Commit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kbqa_r1.hyper_gold_oracle import (
    GoldContinuationOracle,
    GoldContinuationUnavailable,
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def _response(action: str) -> str:
    return (
        "<think>I will follow the verified continuation from this exact "
        "executable state.</think>\n"
        f"<action>{action}</action>"
    )


def _reset_terminal(manager, sample_id: int) -> None:
    for field in (
        "_hyper_commit_certificates",
        "_hyper_valid_answer_turns",
        "_hyper_protocol_valid_answer_turns",
    ):
        getattr(manager, field).pop(sample_id, None)
    manager._hyper_premature_answers.discard(sample_id)


def run_continuation(
    manager,
    decision: Mapping[str, Any],
    *,
    first_action: str | None = None,
) -> dict[str, Any]:
    snapshot = decision.get("private_execution_state")
    if not isinstance(snapshot, Mapping):
        return {"success": False, "status": "missing_snapshot"}
    turn = int(decision.get("turn", -1))
    if turn < 0:
        return {"success": False, "status": "invalid_turn"}

    contract = snapshot.get("private_gold_contract")
    if not isinstance(contract, Mapping):
        return {"success": False, "status": "missing_gold_contract"}
    try:
        oracle = GoldContinuationOracle.from_contract(contract)
    except (GoldContinuationUnavailable, ValueError) as exc:
        return {
            "success": False,
            "status": "invalid_gold_contract",
            "detail": str(exc),
        }

    sample_id = 0
    _reset_terminal(manager, sample_id)
    try:
        manager._restore_hyper_execution_state(sample_id, snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "status": "snapshot_restore_failed",
            "detail": str(exc),
        }

    actions = []
    reasons = []
    initial_constraint_digest = ""
    max_turns = int(manager.config.max_turns)
    while turn < max_turns:
        graph = manager.hyper_graph.state(sample_id)
        manager.hyper_graph.set_clock(
            sample_id,
            turns_used=min(turn + 1, max_turns),
            max_turns=max_turns,
        )
        constraint = manager._hyper_action_constraint(sample_id, turn)
        if not actions:
            initial_constraint_digest = constraint.digest
        if not actions and first_action is not None:
            if not constraint.accepts_response(_response(first_action)):
                return {
                    "success": False,
                    "status": "first_action_outside_constraint",
                    "detail": first_action,
                    "actions": actions,
                    "reasons": reasons,
                    "semantic_q": 0.0,
                    "first_action_accepted": False,
                    "constraint_digest": initial_constraint_digest,
                    "start_turn": int(decision.get("turn", -1)),
                    "max_turns": max_turns,
                    "continuation_policy": "bounded_gold_oracle_after_first_action",
                }
            action = first_action
            reason = "replay the observed first action before gold continuation"
        else:
            try:
                choice = oracle.choose(
                    nodes=tuple(graph.nodes.values()),
                    selected_id=graph.selected_id,
                    frontiers=tuple(manager._hyper_frontiers.get(sample_id, ())),
                    constraint=constraint,
                )
            except GoldContinuationUnavailable as exc:
                return {
                    "success": False,
                    "status": "oracle_unavailable",
                    "detail": str(exc),
                    "actions": actions,
                    "reasons": reasons,
                    "semantic_q": 0.0,
                    "first_action_accepted": bool(actions),
                    "constraint_digest": initial_constraint_digest,
                    "start_turn": int(decision.get("turn", -1)),
                    "max_turns": max_turns,
                    "continuation_policy": "bounded_gold_oracle_after_first_action",
                }
            action = choice.action
            reason = choice.reason

        records_before = len(manager._hyper_action_records.get(sample_id, ()))
        progress_before = manager._hyper_progress_hash(sample_id)
        _, dones = manager.execute_predictions(
            [_response(action)],
            pad_token=str(manager.tokenizer.pad_token or ""),
            turn=turn,
        )
        records = manager._hyper_action_records.get(sample_id, ())
        accepted = len(records[records_before:]) == 1
        progressed = manager._hyper_progress_hash(sample_id) != progress_before
        actions.append(action)
        reasons.append(reason)
        if not accepted or not progressed:
            return {
                "success": False,
                "status": "live_action_rejected",
                "detail": action,
                "actions": actions,
                "reasons": reasons,
                "semantic_q": 0.0,
                "first_action_accepted": accepted if len(actions) == 1 else True,
                "constraint_digest": initial_constraint_digest,
                "start_turn": int(decision.get("turn", -1)),
                "max_turns": max_turns,
                "continuation_policy": "bounded_gold_oracle_after_first_action",
            }

        graph = manager.hyper_graph.state(sample_id)
        if dones[0]:
            certificate = manager._hyper_commit_certificates.get(sample_id, {})
            answer_f1 = float(certificate.get("answer_f1", 0.0))
            explicit = (
                graph.terminal_kind == "explicit_commit"
                and graph.committed_id is not None
            )
            exact = (
                certificate.get("answer_exact") is True
                and abs(answer_f1 - 1.0) <= 1e-9
            )
            intent = certificate.get("intent_equivalent") is True
            semantic_q = answer_f1 if explicit and intent else 0.0
            return {
                "success": explicit and exact and intent,
                "status": (
                    "exact_intent_commit"
                    if explicit and exact and intent
                    else "terminal_not_exact_intent"
                ),
                "actions": actions,
                "reasons": reasons,
                "answer_f1": answer_f1,
                "answer_exact": exact,
                "intent_equivalent": intent,
                "explicit_commit": explicit,
                "semantic_q": semantic_q,
                "first_action_accepted": True,
                "constraint_digest": initial_constraint_digest,
                "start_turn": int(decision.get("turn", -1)),
                "max_turns": max_turns,
                "continuation_policy": "bounded_gold_oracle_after_first_action",
            }
        turn += 1

    return {
        "success": False,
        "status": "turn_budget_exhausted",
        "actions": actions,
        "reasons": reasons,
        "semantic_q": 0.0,
        "first_action_accepted": bool(actions),
        "constraint_digest": initial_constraint_digest,
        "start_turn": int(decision.get("turn", -1)),
        "max_turns": max_turns,
        "continuation_policy": "bounded_gold_oracle_after_first_action",
    }


def main() -> None:
    from scripts.data_process.replay_hyper_trace_corrections import _manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--relation-model", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id")
    args = parser.parse_args()

    manager = _manager(args.tokenizer, args.relation_model)
    counts: Counter[str] = Counter()
    output = []
    for row in _read_jsonl(args.input):
        if args.question_id and str(row.get("question_id")) != args.question_id:
            continue
        if row.get("explicit_model_commit") is not True:
            continue
        if abs(float(row.get("commit_answer_f1", 0.0)) - 1.0) <= 1e-9:
            continue
        if args.limit is not None and counts["failed_rollouts"] >= args.limit:
            break
        counts["failed_rollouts"] += 1
        decisions = sorted(
            (
                value
                for value in row.get("decisions", ())
                if isinstance(value, Mapping)
            ),
            key=lambda value: int(value.get("turn", -1)),
            reverse=True,
        )
        attempts = []
        recovered = None
        for decision in decisions:
            result = run_continuation(manager, decision)
            attempts.append(
                {
                    "turn": int(decision.get("turn", -1)),
                    "model_action": str(decision.get("raw_action") or ""),
                    **result,
                }
            )
            counts[f"state:{result['status']}"] += 1
            if result["success"]:
                recovered = attempts[-1]
                break
        if recovered is not None:
            counts["recoverable_rollouts"] += 1
        else:
            counts["unrecoverable_rollouts"] += 1
        output.append(
            {
                "rollout_id": str(row.get("rollout_id") or ""),
                "question_id": str(row.get("question_id") or ""),
                "family": str(row.get("family") or "unknown"),
                "model_commit_answer_f1": float(row.get("commit_answer_f1", 0.0)),
                "latest_recoverable_state": recovered,
                "attempts": attempts,
            }
        )

    report = {
        **dict(sorted(counts.items())),
        "output_rows": len(output),
        "scope": "latest exact pre-action state with bounded gold continuation",
        "training_rows_emitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
