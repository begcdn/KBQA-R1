from pathlib import Path
from types import SimpleNamespace

from kbqa_r1.action_constraints import HyPERActionConstraintSpec
from kbqa_r1.hyper_gold_oracle import GoldContinuationUnavailable
from kbqa_r1.hyper_r1 import GraphActionAffordances
from scripts.data_process import audit_hyper_gold_continuations as MODULE
from scripts.data_process.audit_hyper_gold_continuations import run_continuation


def test_missing_snapshot_fails_closed():
    assert run_continuation(object(), {}) == {
        "success": False,
        "status": "missing_snapshot",
    }


def test_script_is_explicitly_an_audit_not_a_corpus_generator():
    source = Path(
        "scripts/data_process/audit_hyper_gold_continuations.py"
    ).read_text(encoding="utf-8")
    assert '"training_rows_emitted": 0' in source


def test_first_action_acceptance_survives_later_oracle_failure(monkeypatch):
    class Oracle:
        def choose(self, **_kwargs):
            raise GoldContinuationUnavailable("no continuation")

    monkeypatch.setattr(
        MODULE.GoldContinuationOracle,
        "from_contract",
        classmethod(lambda _cls, _contract: Oracle()),
    )

    class Manager:
        def __init__(self):
            self.config = SimpleNamespace(max_turns=3)
            self.tokenizer = SimpleNamespace(pad_token="")
            self.hyper_graph = SimpleNamespace(
                state=lambda _sample: SimpleNamespace(
                    nodes={}, selected_id=None, terminal_kind="", committed_id=None
                ),
                set_clock=lambda *_args, **_kwargs: None,
            )
            self._hyper_frontiers = {}
            self._hyper_action_records = {0: []}
            self._hyper_commit_certificates = {}
            self._hyper_valid_answer_turns = {}
            self._hyper_protocol_valid_answer_turns = {}
            self._hyper_premature_answers = set()
            self.progress = "before"

        def _restore_hyper_execution_state(self, _sample, _snapshot):
            return None

        def _hyper_progress_hash(self, _sample):
            return self.progress

        def _hyper_action_constraint(self, _sample, turn):
            return HyPERActionConstraintSpec.build(
                state_key=f"state-{turn}",
                turn=turn,
                affordances=GraphActionAffordances(select=("H0",)),
                allow_open_operators=False,
            )

        def execute_predictions(self, _responses, **_kwargs):
            self._hyper_action_records[0].append({"accepted": True})
            self.progress = "after"
            return [""], [False]

    outcome = run_continuation(
        Manager(),
        {
            "turn": 0,
            "private_execution_state": {"private_gold_contract": {"gold": True}},
        },
        first_action="Select [ H0 ]",
    )

    assert outcome["status"] == "oracle_unavailable"
    assert outcome["actions"] == ["Select [ H0 ]"]
    assert outcome["first_action_accepted"] is True
