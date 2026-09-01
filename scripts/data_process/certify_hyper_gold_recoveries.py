#!/usr/bin/env python3
"""Certify the earliest same-state semantic recovery in failed HyPER rollouts.

For each accepted, legal model action, replay two branches from the identical
saved execution state and turn budget:

1. the next action selected by the private gold-program oracle, followed by
   that same bounded oracle; and
2. the model's observed action, followed by that same bounded oracle.

A row is emitted only when the first branch reaches an explicit, answer-exact,
formally intent-equivalent Commit and the observed branch has strictly lower
semantic utility.  The oracle may select only actions exposed by the production
runtime; it cannot insert a hidden relation, entity, proposal, or graph node.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from scripts.data_process.assemble_hyper_corrective_sft import (
    PAIRWISE_SEMANTIC_CERTIFICATE_VERSION,
)
from scripts.data_process.audit_hyper_gold_continuations import run_continuation
from scripts.data_process.certify_hyper_trace_corrections import (
    _canonical_json,
    _digest,
    _state_hash,
)


PAIRWISE_CERTIFIER_VERSION = "hyper-pairwise-gold-regret-v1"
PairwiseEvaluator = Callable[
    [Mapping[str, Any], str | None], Mapping[str, Any]
]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def _source_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _response(action: str) -> str:
    return (
        "<think>I will take the verified action that preserves a complete "
        "route to the question's answer.</think>\n"
        f"<action>{action}</action>"
    )


def _semantic_q(outcome: Mapping[str, Any]) -> float:
    try:
        return float(outcome.get("semantic_q", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _comparable_branches(
    target: Mapping[str, Any], failed: Mapping[str, Any]
) -> bool:
    keys = ("constraint_digest", "start_turn", "max_turns", "continuation_policy")
    return all(target.get(key) == failed.get(key) for key in keys)


def certify_pairwise_rollout(
    row: Mapping[str, Any],
    evaluate: PairwiseEvaluator,
    *,
    ranker_sha256: str,
    freebase_sha256: str,
    executor_hash: str,
    epsilon: float = 0.0,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    """Return the earliest state with certified regret over the observed action."""
    decisions = sorted(
        (value for value in row.get("decisions", ()) if isinstance(value, Mapping)),
        key=lambda value: int(value.get("turn", -1)),
    )
    diagnostics: list[dict[str, Any]] = []
    for decision in decisions:
        turn = int(decision.get("turn", -1))
        model_action = str(decision.get("raw_action") or "").strip()
        diagnostic = {"turn": turn, "model_action": model_action}
        if (
            turn < 0
            or decision.get("accepted") is not True
            or decision.get("no_progress") is True
            or str(decision.get("failure_kind") or "").strip()
        ):
            diagnostic["status"] = "protocol_failure_precedes_semantic_regret"
            diagnostics.append(diagnostic)
            return None, "protocol_failure_precedes_semantic_regret", diagnostics
        if not model_action:
            diagnostic["status"] = "invalid_decision_trace"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics

        messages = deepcopy(decision.get("messages"))
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            diagnostic["status"] = "missing_messages"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics
        state_hash = _state_hash(messages)
        if str(decision.get("state_before_hash") or "") != state_hash:
            diagnostic["status"] = "state_hash_mismatch"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics
        payload = decision.get("constraint_spec")
        if not isinstance(payload, Mapping):
            diagnostic["status"] = "missing_constraint"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics
        try:
            spec = HyPERActionConstraintSpec.from_dict(payload)
        except (TypeError, ValueError):
            diagnostic["status"] = "invalid_constraint"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics
        model_response = str(decision.get("raw_response") or _response(model_action))
        if not spec.accepts_response(model_response):
            diagnostic["status"] = "observed_action_outside_constraint"
            diagnostics.append(diagnostic)
            return None, "invalid_decision_trace", diagnostics

        target = dict(evaluate(decision, None))
        target_actions = target.get("actions")
        target_action = (
            str(target_actions[0]).strip()
            if isinstance(target_actions, list) and target_actions
            else ""
        )
        diagnostic["target_action"] = target_action
        diagnostic["target_status"] = str(target.get("status") or "")
        if (
            target.get("success") is not True
            or target.get("explicit_commit") is not True
            or target.get("answer_exact") is not True
            or target.get("intent_equivalent") is not True
            or abs(_semantic_q(target) - 1.0) > 1e-9
        ):
            diagnostic["status"] = "target_not_exact_intent_completion"
            diagnostics.append(diagnostic)
            continue
        if not target_action or target_action == model_action:
            diagnostic["status"] = "no_distinct_target"
            diagnostics.append(diagnostic)
            continue
        if not spec.accepts_response(_response(target_action)):
            diagnostic["status"] = "target_outside_constraint"
            diagnostics.append(diagnostic)
            continue
        if str(target.get("constraint_digest") or "") != spec.digest:
            diagnostic["status"] = "target_constraint_mismatch"
            diagnostics.append(diagnostic)
            continue

        failed = dict(evaluate(decision, model_action))
        target_q = _semantic_q(target)
        failed_q = _semantic_q(failed)
        diagnostic.update(
            failed_status=str(failed.get("status") or ""),
            target_q=target_q,
            failed_q=failed_q,
        )
        if failed.get("first_action_accepted") is not True:
            diagnostic["status"] = "observed_action_not_replayable"
            diagnostics.append(diagnostic)
            return None, "branch_replay_mismatch", diagnostics
        if not _comparable_branches(target, failed):
            diagnostic["status"] = "branch_contract_mismatch"
            diagnostics.append(diagnostic)
            return None, "branch_replay_mismatch", diagnostics
        if target_q - failed_q <= epsilon + 1e-12:
            diagnostic["status"] = "no_positive_regret"
            diagnostics.append(diagnostic)
            continue

        corrected_messages = deepcopy(messages)
        corrected_messages[-1] = {
            "role": "assistant",
            "content": _response(target_action),
            "loss_mask": 1,
        }
        certificate = {
            "schema_version": PAIRWISE_SEMANTIC_CERTIFICATE_VERSION,
            "certifier_version": PAIRWISE_CERTIFIER_VERSION,
            "certified": True,
            "legal_action": True,
            "execution_verified": True,
            "budget_respected": True,
            "intent_equivalent": True,
            "first_meaningful_failure": True,
            "state_hash": state_hash,
            "target_action": target_action,
            "failed_action": model_action,
            "constraint_digest": spec.digest,
            "ranker_sha256": str(ranker_sha256),
            "freebase_sha256": str(freebase_sha256),
            "executor_hash": str(executor_hash),
            "epsilon": float(epsilon),
            "target_q": target_q,
            "failed_q": failed_q,
            "utility_upper_bound": 1.0,
            "global_upper_bound_achieved": True,
            "decision_scope": "paired_teacher_and_observed_action_at_exact_state",
            "continuation_policy": "bounded_gold_oracle_after_first_action",
            "teacher_source": "private_gold_program_runtime_actions_only",
            "no_hidden_action_injection": True,
            "earlier_states_examined": len(diagnostics),
            "target_outcome": target,
            "failed_outcome": failed,
        }
        certificate["certifier_hash"] = _digest(certificate)
        diagnostic["status"] = "certified"
        diagnostics.append(diagnostic)
        return {
            "rollout_id": str(row.get("rollout_id") or ""),
            "question_id": str(row.get("question_id") or ""),
            "source_split": str(row.get("source_split") or ""),
            "rollout_mode": "masked" if row.get("masked") is True else "unmasked",
            "turn": turn,
            "family": str(row.get("family") or "unknown"),
            "messages": corrected_messages,
            "constraint_spec": dict(payload),
            "certification": certificate,
        }, "certified", diagnostics

    return None, "no_certified_pairwise_regret", diagnostics


class LivePairwiseEvaluator:
    def __init__(self, manager):
        self.manager = manager

    def __call__(
        self, decision: Mapping[str, Any], first_action: str | None
    ) -> Mapping[str, Any]:
        return run_continuation(
            self.manager,
            decision,
            first_action=first_action,
        )


def main() -> None:
    from scripts.data_process.replay_hyper_trace_corrections import _manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--relation-model", type=Path, required=True)
    parser.add_argument("--ranker-sha256", required=True)
    parser.add_argument("--freebase-sha256", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manager = _manager(args.tokenizer, args.relation_model)
    evaluator = LivePairwiseEvaluator(manager)
    root = Path(__file__).resolve().parents[2]
    executor_hash = _source_hash(
        (
            root / "kbqa_r1" / "llm_agent" / "sexpr_generation.py",
            root / "kbqa_r1" / "llm_agent" / "sexpr_action_processor.py",
            root / "kbqa_r1" / "hyper_r1.py",
            root / "kbqa_r1" / "hyper_gold_oracle.py",
        )
    )
    counts: Counter[str] = Counter()
    output = []
    diagnostics = []
    for row in _read_jsonl(args.input):
        if args.limit is not None and counts["input_rollouts"] >= args.limit:
            break
        counts["input_rollouts"] += 1
        if (
            row.get("explicit_model_commit") is not True
            or row.get("forced_terminal") is True
            or abs(float(row.get("commit_answer_f1", 0.0)) - 1.0) <= 1e-9
        ):
            counts["not_failed_explicit_commit"] += 1
            continue
        certified, status, attempts = certify_pairwise_rollout(
            row,
            evaluator,
            ranker_sha256=args.ranker_sha256,
            freebase_sha256=args.freebase_sha256,
            executor_hash=executor_hash,
        )
        counts[status] += 1
        diagnostics.append(
            {
                "rollout_id": str(row.get("rollout_id") or ""),
                "question_id": str(row.get("question_id") or ""),
                "status": status,
                "attempts": attempts,
            }
        )
        if certified is not None:
            output.append(certified)

    report = {
        **dict(sorted(counts.items())),
        "output_rows": len(output),
        "certifier_version": PAIRWISE_CERTIFIER_VERSION,
        "executor_hash": executor_hash,
        "decision_scope": "earliest_certified_pairwise_regret_per_rollout",
        "training_rows_emitted": len(output),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(_canonical_json(row) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
