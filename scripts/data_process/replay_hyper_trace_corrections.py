#!/usr/bin/env python3
"""Replay and certify protocol recoveries from exact HyPER execution states.

This tool does not invent corrective actions.  For the first failed decision
in an exact autonomous trajectory, it considers a later action that the same
policy executed successfully, restores the saved pre-failure state, and then:

1. reproduces the original rejection without executable progress;
2. verifies that the candidate is legal in the restored live state;
3. executes the candidate and the accepted trajectory suffix; and
4. retains the correction only if an explicit commit is answer-exact and
   formally intent-equivalent to the private gold program.

Semantic-regret examples are deliberately out of scope.  They require Q values
for every legal action under a common bounded continuation policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer

from kbqa_r1.llm_agent.sexpr_config import SExprGenerationConfig
from kbqa_r1.llm_agent.sexpr_generation import SExprLLMGenerationManager
from kbqa_r1.sparql.odbc_config import ODBCConfig
from scripts.data_process.certify_hyper_trace_corrections import (
    _canonical_json,
    _digest,
    certify_rows,
)


REPLAY_VERSION = "hyper-exact-state-protocol-replay-v1"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class ExactStateReplayVerifier:
    """Use the production transition system to verify one corrective suffix."""

    def __init__(self, manager: SExprLLMGenerationManager, executor_hash: str):
        self.manager = manager
        self.executor_hash = executor_hash
        self.counts: Counter[str] = Counter()

    def _reset_terminal_state(self, sample_id: int) -> None:
        for field in (
            "_hyper_commit_certificates",
            "_hyper_valid_answer_turns",
            "_hyper_protocol_valid_answer_turns",
        ):
            getattr(self.manager, field).pop(sample_id, None)
        self.manager._hyper_premature_answers.discard(sample_id)

    def _execute(
        self,
        response: str,
        *,
        sample_id: int,
        turn: int,
        require_legal: bool,
    ) -> dict[str, Any]:
        self.manager.hyper_graph.set_clock(
            sample_id,
            turns_used=min(turn + 1, self.manager.config.max_turns),
            max_turns=self.manager.config.max_turns,
        )
        constraint = self.manager._hyper_action_constraint(sample_id, turn)
        legal = constraint.accepts_response(response)
        if require_legal and not legal:
            return {"legal": False, "accepted": False, "made_progress": False}

        progress_before = self.manager._hyper_progress_hash(sample_id)
        records_before = len(self.manager._hyper_action_records.get(sample_id, ()))
        observations, dones = self.manager.execute_predictions(
            [response],
            pad_token=str(self.manager.tokenizer.pad_token or ""),
            turn=turn,
        )
        records = self.manager._hyper_action_records.get(sample_id, ())
        accepted_records = records[records_before:]
        progress_after = self.manager._hyper_progress_hash(sample_id)
        return {
            "legal": legal,
            "constraint_digest": constraint.digest,
            "accepted": len(accepted_records) == 1,
            "made_progress": progress_before != progress_after,
            "terminal": bool(dones[0]),
            "observation": self.manager._trace_observation(observations[0]),
            "progress_before_hash": progress_before,
            "progress_after_hash": progress_after,
        }

    def __call__(
        self,
        row: Mapping[str, Any],
        failed: Mapping[str, Any],
        candidate: Mapping[str, Any],
        suffix: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        self.counts["attempted"] += 1
        snapshot = failed.get("private_execution_state")
        if not isinstance(snapshot, Mapping):
            self.counts["missing_snapshot"] += 1
            return None

        sample_id = 0
        failed_turn = int(failed.get("turn", -1))
        if failed_turn < 0:
            self.counts["invalid_turn"] += 1
            return None

        try:
            self._reset_terminal_state(sample_id)
            self.manager._restore_hyper_execution_state(sample_id, snapshot)
            failed_replay = self._execute(
                str(failed.get("raw_response") or ""),
                sample_id=sample_id,
                turn=failed_turn,
                require_legal=False,
            )
            if (
                failed_replay["accepted"]
                or failed_replay["made_progress"]
                or failed_replay["terminal"]
            ):
                self.counts["failure_not_reproduced"] += 1
                return None

            self._reset_terminal_state(sample_id)
            self.manager._restore_hyper_execution_state(sample_id, snapshot)
            replayed_actions = []
            target_step = None
            for decision in suffix:
                if decision.get("accepted") is not True:
                    continue
                response = str(decision.get("raw_response") or "")
                turn = failed_turn + len(replayed_actions)
                step = self._execute(
                    response,
                    sample_id=sample_id,
                    turn=turn,
                    require_legal=True,
                )
                if not step["legal"]:
                    self.counts["suffix_outside_live_contract"] += 1
                    return None
                if not step["accepted"]:
                    self.counts["suffix_rejected"] += 1
                    return None
                action = str(decision.get("raw_action") or "").strip()
                replayed_actions.append(action)
                if target_step is None:
                    target_step = step
                if step["terminal"]:
                    break

            if not replayed_actions or target_step is None:
                self.counts["empty_suffix"] += 1
                return None
            if not target_step["made_progress"]:
                self.counts["target_no_progress"] += 1
                return None

            graph = self.manager.hyper_graph.state(sample_id)
            certificate = self.manager._hyper_commit_certificates.get(sample_id, {})
            answer_f1 = float(certificate.get("answer_f1", 0.0))
            exact = certificate.get("answer_exact") is True and abs(answer_f1 - 1.0) <= 1e-9
            intent_equivalent = certificate.get("intent_equivalent") is True
            explicit_commit = (
                graph.terminal_kind == "explicit_commit"
                and graph.committed_id is not None
            )
            if not explicit_commit:
                self.counts["suffix_not_explicit_commit"] += 1
                return None
            if not exact:
                self.counts["suffix_not_answer_exact"] += 1
                return None
            if not intent_equivalent:
                self.counts["suffix_not_intent_equivalent"] += 1
                return None

            evidence = {
                "schema_version": REPLAY_VERSION,
                "certified": True,
                "failed_reproduced": True,
                "target_accepted": True,
                "target_made_progress": True,
                "explicit_exact_completion": True,
                "intent_equivalent": True,
                "answer_f1": answer_f1,
                "failed_action": str(failed.get("raw_action") or "").strip(),
                "target_action": str(candidate.get("raw_action") or "").strip(),
                "replayed_actions": replayed_actions,
                "failed_observation": failed_replay["observation"],
                "target_constraint_digest": target_step["constraint_digest"],
                "executor_hash": self.executor_hash,
                "snapshot_hash": _digest(snapshot),
                "final_progress_hash": self.manager._hyper_progress_hash(sample_id),
            }
            evidence["replay_hash"] = _digest(evidence)
            self.counts["certified"] += 1
            return evidence
        except Exception as exc:
            self.counts[f"exception:{type(exc).__name__}"] += 1
            return None


def _manager(tokenizer_path: Path, relation_model: Path) -> SExprLLMGenerationManager:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    odbc = asdict(ODBCConfig.from_env())
    config = SExprGenerationConfig(
        max_turns=32,
        max_start_length=1536,
        max_prompt_length=8192,
        max_response_length=1024,
        max_obs_length=4096,
        num_gpus=0,
        hyper_r1_enable=True,
        hyper_r1_structural_constraints=False,
        hyper_r1_relation_model=str(relation_model),
        hyper_r1_relation_device="cpu",
        use_odbc=True,
        use_aioodbc=False,
        odbc_config=odbc,
        enable_logging=False,
    )
    return SExprLLMGenerationManager(
        tokenizer,
        None,
        config,
        is_validation=True,
        sparql_config={
            "use_odbc": True,
            "use_aioodbc": False,
            "odbc_config": odbc,
        },
        dataset="grailqa",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--relation-model", type=Path, required=True)
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
    verifier = ExactStateReplayVerifier(manager, executor_hash)
    rows, report = certify_rows(_read_jsonl(args.input), verifier)
    report["replay"] = dict(sorted(verifier.counts.items()))
    report["replay_version"] = REPLAY_VERSION
    report["executor_hash"] = executor_hash

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
