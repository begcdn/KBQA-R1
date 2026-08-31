#!/usr/bin/env python3
"""Certify same-state protocol corrections from autonomous HyPER traces.

The certifier never invents a target.  It retains a rollout only when its
first failed decision made no executable progress and a later policy decision
is accepted from that unchanged state, belongs to the failed state's original
legal-action language, and the full trajectory ends in an explicit exact
answer.  All other failed trajectories are dropped fail-closed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kbqa_r1.action_constraints import HyPERActionConstraintSpec


CERTIFIER_VERSION = "hyper-same-state-correction-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _state_hash(messages: list[Mapping[str, Any]]) -> str:
    return _digest(messages[:-1])


def _is_exact_explicit_success(row: Mapping[str, Any]) -> bool:
    return (
        row.get("trajectory_success") is True
        and row.get("explicit_model_commit") is True
        and row.get("forced_terminal") is False
        and abs(float(row.get("commit_answer_f1", -1.0)) - 1.0) <= 1e-9
    )


def _failure_kind(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> str:
    kind = str(current.get("failure_kind") or "").strip()
    if kind:
        return kind
    if previous is None:
        return ""
    same_action = str(previous.get("raw_action") or "").strip() == str(
        current.get("raw_action") or ""
    ).strip()
    unchanged = (
        previous.get("no_progress") is True
        and previous.get("progress_before_hash")
        == previous.get("progress_after_hash")
        == current.get("progress_before_hash")
    )
    return "no_progress_loop" if same_action and unchanged else ""


def _first_failure(decisions: list[Mapping[str, Any]]) -> tuple[int, str] | None:
    previous = None
    for index, decision in enumerate(decisions):
        kind = _failure_kind(previous, decision)
        if kind:
            return index, kind
        previous = decision
    return None


def _same_state_correction(
    row: Mapping[str, Any],
    decisions: list[Mapping[str, Any]],
    failure_index: int,
    failure_kind: str,
) -> dict[str, Any] | None:
    failed = decisions[failure_index]
    before = str(failed.get("progress_before_hash") or "")
    after = str(failed.get("progress_after_hash") or "")
    if not before or before != after or failed.get("no_progress") is not True:
        return None

    messages = deepcopy(failed.get("messages"))
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    state_hash = _state_hash(messages)
    if str(failed.get("state_before_hash") or "") != state_hash:
        return None

    payload = failed.get("constraint_spec")
    if not isinstance(payload, Mapping):
        return None
    try:
        spec = HyPERActionConstraintSpec.from_dict(payload)
    except (TypeError, ValueError):
        return None

    failed_response = str(failed.get("raw_response") or "")
    # Existing traces do not contain a restorable full executor snapshot.  A
    # same-state certificate is therefore sound only for structural rejection:
    # the response was outside the original action contract and no legal
    # transition was entered.  Legal actions that later failed execution need
    # the separate counterfactual replayer and must not be relabeled here.
    if spec.accepts_response(failed_response):
        return None

    for candidate in decisions[failure_index + 1 :]:
        if str(candidate.get("progress_before_hash") or "") != after:
            break
        response = str(candidate.get("raw_response") or "")
        if candidate.get("accepted") is not True or not spec.accepts_response(response):
            continue
        corrected = deepcopy(messages)
        corrected[-1] = {
            "role": "assistant",
            "content": response,
            "loss_mask": 1,
        }
        evidence = {
            "schema_version": CERTIFIER_VERSION,
            "rollout_id": str(row.get("rollout_id") or ""),
            "question_id": str(row.get("question_id") or ""),
            "failed_turn": int(failed.get("turn", -1)),
            "correcting_turn": int(candidate.get("turn", -1)),
            "failure_kind": failure_kind,
            "state_hash": state_hash,
            "progress_hash": after,
            "constraint_digest": spec.digest,
            "failed_response_outside_contract": True,
            "target_response": response,
            "explicit_exact_completion": True,
        }
        return {
            "messages": corrected,
            "legal_target_certified": True,
            "state_hash": state_hash,
            "constraint_digest": spec.digest,
            "certifier_hash": _digest(evidence),
            "certifier_version": CERTIFIER_VERSION,
            "evidence": evidence,
        }
    return None


def certify_rollout(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    decisions = sorted(
        (deepcopy(value) for value in row.get("decisions", [])),
        key=lambda value: int(value.get("turn", -1)),
    )
    failure = _first_failure(decisions)
    if failure is None:
        result = deepcopy(dict(row))
        result["decisions"] = decisions
        return result, "clean"
    if not _is_exact_explicit_success(row):
        return None, "failed_trajectory_not_exact"

    failure_index, failure_kind = failure
    correction = _same_state_correction(
        row,
        decisions,
        failure_index,
        failure_kind,
    )
    if correction is None:
        return None, "first_failure_uncertified"
    decisions[failure_index]["correction"] = correction
    result = deepcopy(dict(row))
    result["decisions"] = decisions
    return result, "certified_recovery"


def certify_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    counts: dict[str, int] = {
        "input_rollouts": 0,
        "clean_rollouts": 0,
        "certified_recoveries": 0,
        "dropped_nonexact": 0,
        "dropped_uncertified": 0,
    }
    for row in rows:
        counts["input_rollouts"] += 1
        certified, status = certify_rollout(row)
        if status == "clean":
            counts["clean_rollouts"] += 1
        elif status == "certified_recovery":
            counts["certified_recoveries"] += 1
        elif status == "failed_trajectory_not_exact":
            counts["dropped_nonexact"] += 1
        else:
            counts["dropped_uncertified"] += 1
        if certified is not None:
            output.append(certified)
    counts["output_rollouts"] = len(output)
    return output, counts


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows, report = certify_rows(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
