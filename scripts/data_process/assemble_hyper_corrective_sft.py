#!/usr/bin/env python3
"""Assemble the compact HyPER-R1 corrective SFT mixture.

This module does not create prompts or infer corrective actions.  Every output
row copies a canonical Markov decision record from one of four pre-existing
sources.  Protocol recoveries require a certified legal correction, and
semantic recoveries require a complete executable-regret certificate.

Autonomous JSONL uses one object per rollout::

    {
      "rollout_id": "...", "question_id": "...", "source_split": "train",
      "masked": true, "trajectory_success": true,
      "explicit_model_commit": true, "forced_terminal": false,
      "commit_answer_f1": 1.0,
      "decisions": [{"turn": 0, "messages": [...], ...}]
    }

Each decision may carry ``failure_kind`` (``protocol``, ``stale_id``, or
``deadline``), state hashes and ``no_progress``.  The first meaningful failure
must also carry ``correction`` with canonical ``messages``,
``legal_target_certified=true``, the matching ``state_hash``, and a nonempty
``certifier_hash``.  A second identical no-progress action is detected from
the ordered decision records.

Semantic JSONL uses one certified state per line.  Its ``messages`` contain the
chosen target and ``certification`` follows ``hyper-semantic-recovery-v1``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


MIXTURE = {
    "ordinary_v23": 50,
    "autonomous_success": 25,
    "protocol_recovery": 15,
    "semantic_recovery": 10,
}
ASSEMBLY_SCHEMA_VERSION = "hyper-corrective-assembly-v1"
SEMANTIC_CERTIFICATE_VERSION = "hyper-semantic-recovery-v1"
REQUIRED_PROVENANCE = {
    "repository_commit",
    "checkpoint_sha256",
    "tokenizer_sha256",
    "prompt_schema_version",
    "ranker_sha256",
    "freebase_sha256",
    "horizon",
    "execution_budget",
    "frontier_size",
    "page_size",
}
_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_FAILURE_PRIORITY = {
    "protocol": 0,
    "stale_id": 1,
    "no_progress_loop": 2,
    "deadline": 3,
}


@dataclass(frozen=True)
class Candidate:
    category: str
    question_id: str
    partition: str
    state_hash: str
    record_hash: str
    record_id: str
    stratum: tuple[str, ...]
    source_path: str
    row_index: int = -1
    rollout_id: str = ""
    rollout_mode: str = ""
    failure_kind: str = ""
    turn: int = -1
    family: str = "unknown"
    record: Mapping[str, Any] | None = None
    certificate_hash: str = ""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            value["__line_number__"] = line_number
            yield value


def _load_test_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        ids = {str(item).strip() for item in value if str(item).strip()}
    elif isinstance(value, dict):
        raw = value.get("question_ids") or value.get("ids")
        if not isinstance(raw, list):
            raise ValueError("test-ID JSON must contain a question_ids or ids list")
        ids = {str(item).strip() for item in raw if str(item).strip()}
    else:
        ids = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                ids.add(line.strip())
                continue
            if isinstance(row, str):
                ids.add(row.strip())
            elif isinstance(row, dict):
                question_id = row.get("question_id") or row.get("id")
                if question_id is None:
                    raise ValueError(f"{path}:{line_number}: test row has no ID")
                ids.add(str(question_id))
            else:
                raise ValueError(f"{path}:{line_number}: invalid test-ID row")
    if not ids:
        raise ValueError("test question-ID set is empty")
    return ids


def _question_partition(question_id: str, seed: int, dev_fraction: float) -> str:
    value = int(
        hashlib.sha256(f"{seed}:question:{question_id}".encode()).hexdigest()[:16],
        16,
    ) / float(16**16)
    return "dev" if value < dev_fraction else "train"


def _message_copy(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("decision record is missing messages")
    copied = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("message must be an object")
        if set(message) != {"role", "content", "loss_mask"}:
            raise ValueError("canonical messages require role, content, and loss_mask only")
        copied.append(
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
                "loss_mask": int(message["loss_mask"]),
            }
        )
    return copied


def _validate_context(messages: Sequence[Mapping[str, Any]]) -> None:
    roles = [message["role"] for message in messages]
    if roles == ["user"]:
        pass
    elif roles == ["user", "assistant", "user"]:
        if messages[1]["content"] != "":
            raise ValueError("canonical Markov bridge assistant must be empty")
        if "<information>" not in messages[2]["content"]:
            raise ValueError("latest runtime observation is missing <information>")
    else:
        raise ValueError(f"noncanonical Markov message roles: {roles}")
    if any(int(message["loss_mask"]) != 0 for message in messages):
        raise ValueError("context messages must have zero loss mask")
    if "HyPER-R1 executable hypothesis graph:" not in messages[0]["content"]:
        raise ValueError("initial message is not the canonical HyPER-R1 prompt")


def _target_action(messages: Any) -> tuple[list[dict[str, Any]], str]:
    copied = _message_copy(messages)
    if len(copied) not in {2, 4}:
        raise ValueError("canonical decision must contain two or four messages")
    _validate_context(copied[:-1])
    target = copied[-1]
    if target["role"] != "assistant" or target["loss_mask"] != 1:
        raise ValueError("decision must end in one supervised assistant action")
    matches = _ACTION.findall(target["content"])
    if len(matches) != 1 or "<answer>" in target["content"]:
        raise ValueError("target must contain exactly one graph action and no answer")
    return copied, matches[0].strip()


def _state_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    return _digest(list(messages[:-1]))


def _value(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    containers = [record]
    for name in ("metadata", "extra_info", "outcome"):
        value = record.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
    for key in keys:
        for container in containers:
            if key in container:
                return container[key]
    return default


def _question_id(record: Mapping[str, Any]) -> str:
    value = _value(record, "question_id", "id")
    if value is None or not str(value).strip():
        raise ValueError("record has no question_id")
    return str(value)


def _turn_band(turn: int) -> str:
    if turn < 0:
        return "unknown"
    if turn <= 3:
        return "early"
    if turn <= 15:
        return "middle"
    return "late"


def _depth_band(value: Any) -> str:
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "0" if depth <= 0 else "1" if depth == 1 else "2+"


def _stratum(record: Mapping[str, Any], action: str, turn: int) -> tuple[str, ...]:
    action_type = action.split("[", 1)[0].strip() or "unknown"
    return (
        str(_value(record, "level", "difficulty", default="unknown")),
        str(_value(record, "family", default="unknown")),
        action_type,
        _turn_band(turn),
        _depth_band(_value(record, "state_depth", "depth", default=None)),
    )


def _candidate(
    *,
    category: str,
    question_id: str,
    seed: int,
    dev_fraction: float,
    source_path: Path,
    messages: Any,
    record_for_hash: Mapping[str, Any],
    row_index: int = -1,
    rollout_id: str = "",
    rollout_mode: str = "",
    failure_kind: str = "",
    turn: int = -1,
    family: str = "unknown",
    record: Mapping[str, Any] | None = None,
    certificate_hash: str = "",
) -> Candidate:
    canonical_messages, action = _target_action(messages)
    state_hash = _state_hash(canonical_messages)
    record_hash = _digest(record_for_hash)
    record_id = _digest(
        [category, str(source_path), rollout_id, turn, state_hash, action, record_hash]
    )
    materialized = None
    if record is not None:
        materialized = {**deepcopy(dict(record)), "messages": canonical_messages}
    return Candidate(
        category=category,
        question_id=question_id,
        partition=_question_partition(question_id, seed, dev_fraction),
        state_hash=state_hash,
        record_hash=record_hash,
        record_id=record_id,
        stratum=_stratum(record_for_hash, action, turn),
        source_path=str(source_path),
        row_index=row_index,
        rollout_id=rollout_id,
        rollout_mode=rollout_mode,
        failure_kind=failure_kind,
        turn=turn,
        family=family,
        record=materialized,
        certificate_hash=certificate_hash,
    )


def _index_ordinary(
    path: Path,
    *,
    seed: int,
    dev_fraction: float,
    test_ids: set[str],
) -> tuple[list[Candidate], set[str]]:
    candidates = []
    question_ids = set()
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(columns=["messages", "data_source", "extra_info"], batch_size=256):
        for row in batch.to_pylist():
            question_id = _question_id(row)
            if question_id in test_ids:
                raise ValueError(f"ordinary v23 input contains test question {question_id}")
            question_ids.add(question_id)
            family = str(_value(row, "family", default="unknown"))
            turn = int(_value(row, "decision_index", default=-1))
            candidates.append(
                _candidate(
                    category="ordinary_v23",
                    question_id=question_id,
                    seed=seed,
                    dev_fraction=dev_fraction,
                    source_path=path,
                    messages=row["messages"],
                    record_for_hash=row,
                    row_index=row_index,
                    turn=turn,
                    family=family,
                )
            )
            row_index += 1
    if not candidates:
        raise ValueError("ordinary v23 parquet is empty")
    return candidates, question_ids


def _rollout_objects(path: Path, *, expected_masked: bool) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    flat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat_roots: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        masked = _value(row, "masked", "structural_constraints")
        if not isinstance(masked, bool) or masked is not expected_masked:
            raise ValueError(f"{path}:{row['__line_number__']}: rollout mask mode mismatch")
        rollout_id = str(_value(row, "rollout_id", "request_id", default="")).strip()
        if not rollout_id:
            raise ValueError(f"{path}:{row['__line_number__']}: rollout_id is required")
        decisions = row.get("decisions")
        if decisions is not None:
            if not isinstance(decisions, list) or not decisions:
                raise ValueError(f"{path}:{row['__line_number__']}: decisions must be nonempty")
            trajectories.append(row)
            continue
        if "messages" not in row:
            raise ValueError(f"{path}:{row['__line_number__']}: no decisions or messages")
        flat[rollout_id].append(row)
        flat_roots.setdefault(rollout_id, row)
    for rollout_id, decisions in flat.items():
        root = dict(flat_roots[rollout_id])
        root["rollout_id"] = rollout_id
        root["decisions"] = decisions
        trajectories.append(root)
    return trajectories


def _source_train(record: Mapping[str, Any]) -> None:
    split = str(_value(record, "source_split", "dataset_split", "split", default=""))
    if split.lower() != "train":
        raise ValueError(f"autonomous/certified record must declare source_split=train, got {split!r}")


def _decision_action(decision: Mapping[str, Any]) -> str:
    value = _value(decision, "policy_action", "action", default="")
    if str(value).strip():
        return str(value).strip()
    messages = _message_copy(decision.get("messages"))
    matches = _ACTION.findall(messages[-1]["content"]) if messages else []
    return matches[0].strip() if len(matches) == 1 else ""


def _failure_kind(decision: Mapping[str, Any]) -> str:
    raw = str(_value(decision, "failure_kind", default="")).lower()
    if bool(_value(decision, "stale_id", "stale_reference", default=False)) or raw in {
        "stale", "stale_id", "unknown_id", "unknown_or_stale_id"
    }:
        return "stale_id"
    if bool(_value(decision, "protocol_error", default=False)) or raw in {
        "protocol", "malformed_action", "invalid_action", "parse_error"
    }:
        return "protocol"
    if bool(_value(decision, "deadline", default=False)) or raw in {
        "deadline", "turn_deadline", "turn_exhausted"
    }:
        return "deadline"
    return ""


def _second_identical_no_progress(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> bool:
    if previous is None:
        return False
    previous_action = _decision_action(previous)
    current_action = _decision_action(current)
    if not previous_action or previous_action != current_action:
        return False
    if not bool(_value(previous, "no_progress", default=False)):
        return False
    before = str(_value(previous, "state_before_hash", default=""))
    after = str(_value(previous, "state_after_hash", default=""))
    current_before = str(_value(current, "state_before_hash", default=""))
    return bool(before and before == after == current_before)


def _correction_messages(
    decision: Mapping[str, Any],
    *,
    failure_kind: str,
) -> tuple[list[dict[str, Any]], str]:
    correction = decision.get("correction")
    if not isinstance(correction, Mapping):
        raise ValueError(f"{failure_kind} recovery lacks correction metadata")
    if correction.get("legal_target_certified") is not True:
        raise ValueError(f"{failure_kind} correction is not legally certified")
    certifier_hash = str(correction.get("certifier_hash", "")).strip()
    if not certifier_hash:
        raise ValueError(f"{failure_kind} correction lacks certifier_hash")
    expected_state = str(_value(decision, "state_before_hash", default=""))
    if not expected_state or str(correction.get("state_hash", "")) != expected_state:
        raise ValueError(f"{failure_kind} correction state hash mismatch")
    corrected, _ = _target_action(correction.get("messages"))
    observed = _message_copy(decision.get("messages"))
    if corrected[:-1] != observed[:-1]:
        raise ValueError(f"{failure_kind} correction changes the student-visible state")
    if failure_kind == "deadline" and correction.get("completion_reachable") is not True:
        raise ValueError("deadline correction is not certified reachable")
    return corrected, certifier_hash


def _load_rollout_candidates(
    path: Path,
    *,
    expected_masked: bool,
    seed: int,
    dev_fraction: float,
    test_ids: set[str],
    train_ids: set[str],
) -> tuple[list[Candidate], list[Candidate]]:
    successes: list[Candidate] = []
    recoveries: list[Candidate] = []
    mode = "masked" if expected_masked else "unmasked"
    for trajectory in _rollout_objects(path, expected_masked=expected_masked):
        _source_train(trajectory)
        question_id = _question_id(trajectory)
        if question_id in test_ids:
            raise ValueError(f"{mode} rollout contains test question {question_id}")
        if question_id not in train_ids:
            raise ValueError(f"{mode} rollout question {question_id} is absent from v23 train")
        rollout_id = str(_value(trajectory, "rollout_id", "request_id"))
        decisions = sorted(
            trajectory["decisions"],
            key=lambda value: int(_value(value, "turn", "decision_index", default=-1)),
        )
        if len({int(_value(value, "turn", "decision_index", default=-1)) for value in decisions}) != len(decisions):
            raise ValueError(f"rollout {rollout_id} has duplicate decision turns")

        successful = (
            _value(trajectory, "trajectory_success", default=False) is True
            and _value(trajectory, "explicit_model_commit", default=False) is True
            and _value(trajectory, "forced_terminal", default=False) is False
            and abs(float(_value(trajectory, "commit_answer_f1", "hyper_r1_commit_answer_f1", default=-1.0)) - 1.0) <= 1e-9
        )
        clean_success = successful
        previous = None
        for decision in decisions:
            if (
                _value(decision, "accepted", "action_accepted", default=False) is not True
                or bool(_failure_kind(decision))
                or bool(_value(decision, "no_progress", default=False))
                or _second_identical_no_progress(previous, decision)
            ):
                clean_success = False
                break
            previous = decision
        if clean_success:
            for decision in decisions:
                turn = int(_value(decision, "turn", "decision_index", default=-1))
                successes.append(
                    _candidate(
                        category="autonomous_success",
                        question_id=question_id,
                        seed=seed,
                        dev_fraction=dev_fraction,
                        source_path=path,
                        messages=decision.get("messages"),
                        record_for_hash=decision,
                        rollout_id=rollout_id,
                        rollout_mode=mode,
                        turn=turn,
                        family=str(_value(trajectory, "family", default="unknown")),
                        record=decision,
                    )
                )

        failures: list[tuple[int, int, str, Mapping[str, Any]]] = []
        previous = None
        for decision in decisions:
            turn = int(_value(decision, "turn", "decision_index", default=-1))
            kind = _failure_kind(decision)
            if not kind and _second_identical_no_progress(previous, decision):
                kind = "no_progress_loop"
            if kind:
                failures.append((turn, _FAILURE_PRIORITY[kind], kind, decision))
            previous = decision
        if failures:
            turn, _, kind, decision = min(failures, key=lambda value: (value[0], value[1]))
            corrected, certifier_hash = _correction_messages(decision, failure_kind=kind)
            recovery_record = {**deepcopy(dict(decision)), "messages": corrected}
            recoveries.append(
                _candidate(
                    category="protocol_recovery",
                    question_id=question_id,
                    seed=seed,
                    dev_fraction=dev_fraction,
                    source_path=path,
                    messages=corrected,
                    record_for_hash=recovery_record,
                    rollout_id=rollout_id,
                    rollout_mode=mode,
                    failure_kind=kind,
                    turn=turn,
                    family=str(_value(trajectory, "family", default="unknown")),
                    record=recovery_record,
                    certificate_hash=certifier_hash,
                )
            )
    return successes, recoveries


def _validate_semantic_certificate(
    row: Mapping[str, Any],
    *,
    action: str,
    state_hash: str,
    provenance: Mapping[str, Any],
) -> str:
    certificate = row.get("certification")
    if not isinstance(certificate, Mapping):
        raise ValueError("semantic recovery has no certification object")
    if certificate.get("schema_version") != SEMANTIC_CERTIFICATE_VERSION:
        raise ValueError("semantic recovery has an unsupported certificate schema")
    required_true = (
        "certified",
        "legal_action",
        "execution_verified",
        "budget_respected",
        "intent_equivalent",
        "first_meaningful_failure",
    )
    missing = [name for name in required_true if certificate.get(name) is not True]
    if missing:
        raise ValueError("uncertified semantic recovery: " + ", ".join(missing))
    if str(certificate.get("state_hash", "")) != state_hash:
        raise ValueError("semantic recovery certificate state hash mismatch")
    if str(certificate.get("target_action", "")).strip() != action:
        raise ValueError("semantic recovery target differs from certified action")
    for key in ("ranker_sha256", "freebase_sha256"):
        if str(certificate.get(key, "")) != str(provenance[key]):
            raise ValueError(f"semantic recovery {key} differs from run provenance")
    for key in ("executor_hash", "certifier_hash"):
        if not str(certificate.get(key, "")).strip():
            raise ValueError(f"semantic recovery lacks {key}")
    try:
        epsilon = float(certificate["epsilon"])
        q_values = {str(key): float(value) for key, value in certificate["q_values"].items()}
        optimal = {str(value) for value in certificate["optimal_actions"]}
        failed_action = str(certificate["failed_action"]).strip()
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("semantic recovery has malformed executable-regret values") from exc
    if (
        not math.isfinite(epsilon)
        or not (0.0 <= epsilon <= 0.1)
        or not q_values
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in q_values.values())
    ):
        raise ValueError("semantic recovery epsilon/Q values are invalid")
    if not failed_action or failed_action == action:
        raise ValueError("semantic recovery must identify a distinct failed action")
    if failed_action not in q_values:
        raise ValueError("semantic recovery failed action is absent from Q values")
    maximum = max(q_values.values())
    expected = {candidate for candidate, value in q_values.items() if value >= maximum - epsilon - 1e-12}
    if optimal != expected or action not in optimal:
        raise ValueError("semantic recovery optimal action set is inconsistent with Q values")
    if failed_action in optimal or maximum - q_values[failed_action] <= epsilon + 1e-12:
        raise ValueError("semantic recovery does not certify positive executable regret")
    return _digest(certificate)


def _load_semantic_candidates(
    path: Path,
    *,
    seed: int,
    dev_fraction: float,
    test_ids: set[str],
    train_ids: set[str],
    provenance: Mapping[str, Any],
) -> list[Candidate]:
    candidates = []
    seen_rollouts = set()
    for row in _read_jsonl(path):
        _source_train(row)
        question_id = _question_id(row)
        if question_id in test_ids:
            raise ValueError(f"semantic recovery contains test question {question_id}")
        if question_id not in train_ids:
            raise ValueError(f"semantic recovery question {question_id} is absent from v23 train")
        rollout_id = str(_value(row, "rollout_id", "request_id", default="")).strip()
        if not rollout_id:
            raise ValueError("semantic recovery requires rollout_id")
        if rollout_id in seen_rollouts:
            raise ValueError(
                f"semantic recovery contains more than one state for rollout {rollout_id}"
            )
        seen_rollouts.add(rollout_id)
        messages, action = _target_action(row.get("messages"))
        state_hash = _state_hash(messages)
        certificate_hash = _validate_semantic_certificate(
            row,
            action=action,
            state_hash=state_hash,
            provenance=provenance,
        )
        turn = int(_value(row, "turn", "decision_index", default=-1))
        candidate = _candidate(
            category="semantic_recovery",
            question_id=question_id,
            seed=seed,
            dev_fraction=dev_fraction,
            source_path=path,
            messages=messages,
            record_for_hash=row,
            rollout_id=rollout_id,
            rollout_mode=str(_value(row, "rollout_mode", default="masked")),
            failure_kind="semantic_regret",
            turn=turn,
            family=str(_value(row, "family", default="unknown")),
            record=row,
            certificate_hash=certificate_hash,
        )
        candidates.append(candidate)
    return candidates


def _rank(seed: int, category: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{value}".encode()).hexdigest()


def _stratified_sample(
    candidates: Sequence[Candidate],
    count: int,
    *,
    seed: int,
    excluded_states: set[str],
    recovery_rollouts: set[str],
) -> list[Candidate]:
    groups: dict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.state_hash in excluded_states:
            continue
        if "recovery" in candidate.category and candidate.rollout_id in recovery_rollouts:
            continue
        groups[candidate.stratum].append(candidate)
    queues = []
    for stratum, values in groups.items():
        values.sort(key=lambda value: _rank(seed, value.category, value.record_id))
        heapq.heappush(queues, (0, _rank(seed, "stratum", repr(stratum)), stratum, values, 0))
    chosen = []
    while queues and len(chosen) < count:
        used, stratum_rank, stratum, values, offset = heapq.heappop(queues)
        candidate = values[offset]
        if candidate.state_hash not in excluded_states and not (
            "recovery" in candidate.category and candidate.rollout_id in recovery_rollouts
        ):
            chosen.append(candidate)
            excluded_states.add(candidate.state_hash)
            if "recovery" in candidate.category:
                recovery_rollouts.add(candidate.rollout_id)
        offset += 1
        if offset < len(values):
            heapq.heappush(queues, (used + 1, stratum_rank, stratum, values, offset))
    if len(chosen) != count:
        raise ValueError(
            f"insufficient unique {candidates[0].category if candidates else 'candidate'} "
            f"states: requested {count}, selected {len(chosen)}"
        )
    return chosen


def _mixture_counts(size: int) -> dict[str, int]:
    if size <= 0 or size % 20:
        raise ValueError("train/dev sizes must be positive multiples of 20")
    return {name: size * share // 100 for name, share in MIXTURE.items()}


def _materialize_ordinary(path: Path, indexes: set[int]) -> dict[int, dict[str, Any]]:
    rows = {}
    row_index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row_index in indexes:
                rows[row_index] = row
            row_index += 1
    if set(rows) != indexes:
        raise RuntimeError("failed to materialize selected ordinary rows")
    return rows


def _output_schema() -> pa.Schema:
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
                        pa.field("question_id", pa.string()),
                        pa.field("family", pa.string()),
                        pa.field("correction_source", pa.string()),
                        pa.field("source_record_id", pa.string()),
                        pa.field("source_record_sha256", pa.string()),
                        pa.field("source_state_sha256", pa.string()),
                        pa.field("source_rollout_id", pa.string()),
                        pa.field("source_rollout_mode", pa.string()),
                        pa.field("failure_kind", pa.string()),
                        pa.field("certificate_sha256", pa.string()),
                        pa.field("turn", pa.int64()),
                        pa.field("state_weight", pa.float64()),
                        pa.field("target_is_graph_action", pa.bool_()),
                    ]
                ),
            ),
        ]
    )


def _output_row(candidate: Candidate, source: Mapping[str, Any]) -> dict[str, Any]:
    messages, _ = _target_action(source.get("messages"))
    if _state_hash(messages) != candidate.state_hash:
        raise RuntimeError("source state changed during assembly")
    return {
        "messages": messages,
        "data_source": "hyper_r1_corrective_decision",
        "extra_info": {
            "question_id": candidate.question_id,
            "family": candidate.family,
            "correction_source": candidate.category,
            "source_record_id": candidate.record_id,
            "source_record_sha256": candidate.record_hash,
            "source_state_sha256": candidate.state_hash,
            "source_rollout_id": candidate.rollout_id,
            "source_rollout_mode": candidate.rollout_mode,
            "failure_kind": candidate.failure_kind,
            "certificate_sha256": candidate.certificate_hash,
            "turn": candidate.turn,
            "state_weight": 1.0,
            "target_is_graph_action": True,
        },
    }


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assemble_corrective_dataset(
    *,
    ordinary_v23: Path,
    masked_rollouts: Path,
    unmasked_rollouts: Path,
    semantic_recoveries: Path,
    test_question_ids: Path,
    provenance_config: Path,
    output: Path,
    train_size: int,
    dev_size: int,
    seed: int,
    dev_fraction: float,
) -> dict[str, Any]:
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError("dev_fraction must lie strictly between zero and one")
    counts = {"train": _mixture_counts(train_size), "dev": _mixture_counts(dev_size)}
    paths = {
        "ordinary_v23": ordinary_v23,
        "masked_rollouts": masked_rollouts,
        "unmasked_rollouts": unmasked_rollouts,
        "semantic_recoveries": semantic_recoveries,
        "test_question_ids": test_question_ids,
        "provenance_config": provenance_config,
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("missing input files: " + ", ".join(missing_paths))
    provenance = json.loads(provenance_config.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("provenance config must be a JSON object")
    incomplete_provenance = sorted(
        key for key in REQUIRED_PROVENANCE if provenance.get(key) in (None, "")
    )
    if incomplete_provenance:
        raise ValueError(
            "provenance config is incomplete: " + ", ".join(incomplete_provenance)
        )

    test_ids = _load_test_ids(test_question_ids)
    ordinary, train_ids = _index_ordinary(
        ordinary_v23,
        seed=seed,
        dev_fraction=dev_fraction,
        test_ids=test_ids,
    )
    masked_success, masked_recovery = _load_rollout_candidates(
        masked_rollouts,
        expected_masked=True,
        seed=seed,
        dev_fraction=dev_fraction,
        test_ids=test_ids,
        train_ids=train_ids,
    )
    unmasked_success, unmasked_recovery = _load_rollout_candidates(
        unmasked_rollouts,
        expected_masked=False,
        seed=seed,
        dev_fraction=dev_fraction,
        test_ids=test_ids,
        train_ids=train_ids,
    )
    if not masked_success or not unmasked_success:
        raise ValueError("successful autonomous pool must contain masked and unmasked states")
    semantic = _load_semantic_candidates(
        semantic_recoveries,
        seed=seed,
        dev_fraction=dev_fraction,
        test_ids=test_ids,
        train_ids=train_ids,
        provenance=provenance,
    )
    pools = {
        "ordinary_v23": ordinary,
        "autonomous_success": [*masked_success, *unmasked_success],
        "protocol_recovery": [*masked_recovery, *unmasked_recovery],
        "semantic_recovery": semantic,
    }

    selected: dict[str, list[Candidate]] = {"train": [], "dev": []}
    for partition in ("train", "dev"):
        excluded_states: set[str] = set()
        recovery_rollouts: set[str] = set()
        # Select scarce recoveries first; later pools cannot duplicate their
        # public states. Protocol recovery has precedence over semantic rows
        # from the same rollout.
        for category in (
            "protocol_recovery",
            "semantic_recovery",
            "autonomous_success",
            "ordinary_v23",
        ):
            partition_pool = [value for value in pools[category] if value.partition == partition]
            selected[partition].extend(
                _stratified_sample(
                    partition_pool,
                    counts[partition][category],
                    seed=seed,
                    excluded_states=excluded_states,
                    recovery_rollouts=recovery_rollouts,
                )
            )

    train_qids = {value.question_id for value in selected["train"]}
    dev_qids = {value.question_id for value in selected["dev"]}
    if train_qids & dev_qids:
        raise RuntimeError("corrective train/dev question overlap")
    if (train_qids | dev_qids) & test_ids:
        raise RuntimeError("test question leaked into corrective output")
    for partition in ("train", "dev"):
        recoveries = [
            value.rollout_id
            for value in selected[partition]
            if "recovery" in value.category
        ]
        if len(recoveries) != len(set(recoveries)):
            raise RuntimeError("more than one recovery state selected from a rollout")

    ordinary_indexes = {
        value.row_index
        for values in selected.values()
        for value in values
        if value.category == "ordinary_v23"
    }
    materialized_ordinary = _materialize_ordinary(ordinary_v23, ordinary_indexes)

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        output_hashes = {}
        category_counts = {}
        for partition in ("train", "dev"):
            values = sorted(
                selected[partition],
                key=lambda value: _rank(seed, f"output:{partition}", value.record_id),
            )
            rows = []
            category_counts[partition] = defaultdict(int)
            for candidate in values:
                source = (
                    materialized_ordinary[candidate.row_index]
                    if candidate.category == "ordinary_v23"
                    else candidate.record
                )
                if source is None:
                    raise RuntimeError("selected candidate has no source record")
                rows.append(_output_row(candidate, source))
                category_counts[partition][candidate.category] += 1
            parquet_path = temporary / f"{partition}_decision.parquet"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=_output_schema()),
                parquet_path,
                compression="snappy",
            )
            output_hashes[parquet_path.name] = _file_sha256(parquet_path)

        for partition, qids in (("train", train_qids), ("dev", dev_qids)):
            qid_path = temporary / f"{partition}_question_ids.txt"
            qid_path.write_text("".join(f"{qid}\n" for qid in sorted(qids)), encoding="utf-8")
            output_hashes[qid_path.name] = _file_sha256(qid_path)

        repository_root = Path(__file__).resolve().parents[2]
        manifest = {
            "schema_version": ASSEMBLY_SCHEMA_VERSION,
            "assembler_commit": _git_commit(repository_root),
            "student_visible_recovery_marker": False,
            "fixed_state_weight": 1.0,
            "seed": seed,
            "dev_fraction": dev_fraction,
            "mixture_percent": MIXTURE,
            "requested_rows": {"train": train_size, "dev": dev_size},
            "actual_rows": {
                partition: dict(sorted(values.items()))
                for partition, values in category_counts.items()
            },
            "question_counts": {"train": len(train_qids), "dev": len(dev_qids)},
            "question_disjoint": True,
            "test_question_overlap": 0,
            "input_files": {
                name: {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for name, path in paths.items()
            },
            "run_provenance": provenance,
            "output_sha256": output_hashes,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-v23", type=Path, required=True)
    parser.add_argument("--masked-rollouts", type=Path, required=True)
    parser.add_argument("--unmasked-rollouts", type=Path, required=True)
    parser.add_argument("--semantic-recoveries", type=Path, required=True)
    parser.add_argument("--test-question-ids", type=Path, required=True)
    parser.add_argument("--provenance-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=40_000)
    parser.add_argument("--dev-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = assemble_corrective_dataset(
        ordinary_v23=args.ordinary_v23,
        masked_rollouts=args.masked_rollouts,
        unmasked_rollouts=args.unmasked_rollouts,
        semantic_recoveries=args.semantic_recoveries,
        test_question_ids=args.test_question_ids,
        provenance_config=args.provenance_config,
        output=args.output,
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
