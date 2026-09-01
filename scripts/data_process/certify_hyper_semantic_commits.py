#!/usr/bin/env python3
"""Certify semantic regret among terminal Commit actions.

This is the finite, teacher-free semantic case.  Given the exact state before
an accepted but wrong Commit, replay every Commit action advertised by that
state's production constraint.  A correction is retained only when a distinct
candidate terminates with an answer-exact, formally intent-equivalent program.

The resulting certificate is intentionally scoped to the terminal Commit
decision.  It does not claim to rank nonterminal graph actions or open-ended
operator strings; those require a common bounded continuation policy and a
finite runtime candidate menu.
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
    SEMANTIC_CERTIFICATE_VERSION,
)
from scripts.data_process.certify_hyper_trace_corrections import (
    _canonical_json,
    _digest,
    _state_hash,
)
COMMIT_CERTIFIER_VERSION = "hyper-terminal-commit-regret-v1"
CommitEvaluator = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _source_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _accepted_wrong_commit(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if row.get("explicit_model_commit") is not True:
        return None
    if abs(float(row.get("commit_answer_f1", 0.0)) - 1.0) <= 1e-9:
        return None
    decisions = sorted(
        (value for value in row.get("decisions", ()) if isinstance(value, Mapping)),
        key=lambda value: int(value.get("turn", -1)),
    )
    commits = [
        value
        for value in decisions
        if value.get("accepted") is True
        and str(value.get("raw_action") or "").strip().startswith("Commit [")
    ]
    return commits[0] if len(commits) == 1 else None


def _target_response(action: str) -> str:
    return (
        "<think>This retained hypothesis is the strongest complete "
        "interpretation of the question.</think>\n"
        f"<action>{action}</action>"
    )


def certify_commit_rollout(
    row: Mapping[str, Any],
    evaluate: CommitEvaluator,
    *,
    ranker_sha256: str,
    freebase_sha256: str,
    executor_hash: str,
    epsilon: float = 0.0,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Return one correction when exhaustive terminal Commit replay proves regret."""
    failed = _accepted_wrong_commit(row)
    if failed is None:
        return None, "not_wrong_explicit_commit", {}

    messages = deepcopy(failed.get("messages"))
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None, "missing_messages", {}
    state_hash = _state_hash(messages)
    if str(failed.get("state_before_hash") or "") != state_hash:
        return None, "state_hash_mismatch", {}

    payload = failed.get("constraint_spec")
    if not isinstance(payload, Mapping):
        return None, "missing_constraint", {}
    try:
        spec = HyPERActionConstraintSpec.from_dict(payload)
    except (TypeError, ValueError):
        return None, "invalid_constraint", {}

    commit_actions = tuple(
        action for action in spec.exact_actions if action.startswith("Commit [")
    )
    failed_action = str(failed.get("raw_action") or "").strip()
    if len(commit_actions) < 2 or failed_action not in commit_actions:
        return None, "insufficient_commit_candidates", {
            "rollout_id": str(row.get("rollout_id") or ""),
            "question_id": str(row.get("question_id") or ""),
            "failed_action": failed_action,
            "candidate_count": len(commit_actions),
            "failed_answer_f1": float(row.get("commit_answer_f1", 0.0)),
        }

    outcomes: dict[str, dict[str, Any]] = {}
    q_values: dict[str, float] = {}
    for action in commit_actions:
        raw = dict(evaluate(failed, action))
        answer_f1 = float(raw.get("answer_f1", 0.0))
        accepted = raw.get("accepted") is True
        explicit_commit = raw.get("explicit_commit") is True
        intent_equivalent = raw.get("intent_equivalent") is True
        answer_exact = raw.get("answer_exact") is True and abs(answer_f1 - 1.0) <= 1e-9
        verified = accepted and explicit_commit
        semantic_q = answer_f1 if verified and intent_equivalent else 0.0
        q_values[action] = semantic_q
        outcomes[action] = {
            "accepted": accepted,
            "explicit_commit": explicit_commit,
            "answer_f1": answer_f1,
            "answer_exact": answer_exact,
            "intent_equivalent": intent_equivalent,
            "semantic_q": semantic_q,
            "constraint_digest": str(raw.get("constraint_digest") or ""),
        }

    maximum = max(q_values.values())
    optimal = tuple(
        action
        for action in commit_actions
        if q_values[action] >= maximum - epsilon - 1e-12
    )
    eligible = [
        action
        for action in optimal
        if outcomes[action]["answer_exact"]
        and outcomes[action]["intent_equivalent"]
        and outcomes[action]["constraint_digest"] == spec.digest
    ]
    diagnostic = {
        "rollout_id": str(row.get("rollout_id") or ""),
        "question_id": str(row.get("question_id") or ""),
        "failed_action": failed_action,
        "candidate_count": len(commit_actions),
        "failed_answer_f1": float(row.get("commit_answer_f1", 0.0)),
        "accepted_candidates": sum(
            value["accepted"] and value["explicit_commit"]
            for value in outcomes.values()
        ),
        "answer_exact_candidates": sum(
            value["answer_exact"] for value in outcomes.values()
        ),
        "intent_equivalent_candidates": sum(
            value["intent_equivalent"] for value in outcomes.values()
        ),
        "best_answer_f1": max(value["answer_f1"] for value in outcomes.values()),
        "best_semantic_q": maximum,
    }
    if not eligible:
        return None, "no_exact_intent_equivalent_alternative", diagnostic
    if failed_action in optimal or maximum - q_values[failed_action] <= epsilon + 1e-12:
        return None, "no_positive_regret", diagnostic

    target_action = eligible[0]
    corrected_messages = deepcopy(messages)
    corrected_messages[-1] = {
        "role": "assistant",
        "content": _target_response(target_action),
        "loss_mask": 1,
    }
    certificate = {
        "schema_version": SEMANTIC_CERTIFICATE_VERSION,
        "certifier_version": COMMIT_CERTIFIER_VERSION,
        "certified": True,
        "legal_action": True,
        "execution_verified": True,
        "budget_respected": True,
        "intent_equivalent": True,
        "first_meaningful_failure": True,
        "state_hash": state_hash,
        "target_action": target_action,
        "constraint_digest": spec.digest,
        "ranker_sha256": str(ranker_sha256),
        "freebase_sha256": str(freebase_sha256),
        "executor_hash": str(executor_hash),
        "epsilon": float(epsilon),
        "failed_action": failed_action,
        "legal_actions": list(commit_actions),
        "q_values": q_values,
        "optimal_actions": list(optimal),
        "decision_scope": "all_terminal_commit_actions_in_live_contract",
        "continuation_policy": "terminal_action_no_continuation",
        "utility": "answer_f1_if_formally_intent_equivalent_else_zero",
        "candidate_outcomes": outcomes,
    }
    certificate["certifier_hash"] = _digest(certificate)
    return {
        "rollout_id": str(row.get("rollout_id") or ""),
        "question_id": str(row.get("question_id") or ""),
        "source_split": str(row.get("source_split") or ""),
        "rollout_mode": "masked" if row.get("masked") is True else "unmasked",
        "turn": int(failed.get("turn", -1)),
        "family": str(row.get("family") or "unknown"),
        "messages": corrected_messages,
        "constraint_spec": dict(payload),
        "certification": certificate,
    }, "certified", diagnostic


class LiveCommitEvaluator:
    """Replay terminal candidates through the production transition system."""

    def __init__(self, manager):
        self.manager = manager

    def _reset(self, sample_id: int) -> None:
        for field in (
            "_hyper_commit_certificates",
            "_hyper_valid_answer_turns",
            "_hyper_protocol_valid_answer_turns",
        ):
            getattr(self.manager, field).pop(sample_id, None)
        self.manager._hyper_premature_answers.discard(sample_id)

    def __call__(self, decision: Mapping[str, Any], action: str) -> Mapping[str, Any]:
        snapshot = decision.get("private_execution_state")
        if not isinstance(snapshot, Mapping):
            return {}
        sample_id = 0
        turn = int(decision.get("turn", -1))
        if turn < 0:
            return {}
        self._reset(sample_id)
        self.manager._restore_hyper_execution_state(sample_id, snapshot)
        self.manager.hyper_graph.set_clock(
            sample_id,
            turns_used=min(turn + 1, self.manager.config.max_turns),
            max_turns=self.manager.config.max_turns,
        )
        spec = self.manager._hyper_action_constraint(sample_id, turn)
        response = _target_response(action)
        if not spec.accepts_response(response):
            return {"constraint_digest": spec.digest}
        records_before = len(self.manager._hyper_action_records.get(sample_id, ()))
        _, dones = self.manager.execute_predictions(
            [response],
            pad_token=str(self.manager.tokenizer.pad_token or ""),
            turn=turn,
        )
        records = self.manager._hyper_action_records.get(sample_id, ())
        accepted = len(records[records_before:]) == 1
        graph = self.manager.hyper_graph.state(sample_id)
        certificate = self.manager._hyper_commit_certificates.get(sample_id, {})
        return {
            "accepted": accepted,
            "explicit_commit": bool(
                dones[0]
                and graph.terminal_kind == "explicit_commit"
                and graph.committed_id is not None
            ),
            "answer_f1": float(certificate.get("answer_f1", 0.0)),
            "answer_exact": certificate.get("answer_exact") is True,
            "intent_equivalent": certificate.get("intent_equivalent") is True,
            "constraint_digest": spec.digest,
        }


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
    args = parser.parse_args()

    manager = _manager(args.tokenizer, args.relation_model)
    root = Path(__file__).resolve().parents[2]
    executor_hash = _source_hash(
        (
            root / "kbqa_r1" / "llm_agent" / "sexpr_generation.py",
            root / "kbqa_r1" / "llm_agent" / "sexpr_action_processor.py",
            root / "kbqa_r1" / "hyper_r1.py",
        )
    )
    evaluate = LiveCommitEvaluator(manager)
    counts: Counter[str] = Counter()
    output = []
    diagnostics = []
    for row in _read_jsonl(args.input):
        counts["input_rollouts"] += 1
        certified, status, diagnostic = certify_commit_rollout(
            row,
            evaluate,
            ranker_sha256=args.ranker_sha256,
            freebase_sha256=args.freebase_sha256,
            executor_hash=executor_hash,
        )
        counts[status] += 1
        if diagnostic:
            diagnostics.append({**diagnostic, "status": status})
        if certified is not None:
            output.append(certified)

    report = {
        **dict(sorted(counts.items())),
        "output_rows": len(output),
        "certifier_version": COMMIT_CERTIFIER_VERSION,
        "executor_hash": executor_hash,
        "decision_scope": "terminal_commit_only",
        "wrong_commit_diagnostics": diagnostics,
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
