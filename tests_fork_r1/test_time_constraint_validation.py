import sys
import types
from pathlib import Path
from types import SimpleNamespace


if "pyodbc" not in sys.modules:
    pyodbc = types.ModuleType("pyodbc")
    pyodbc.Connection = object
    pyodbc.Error = Exception
    pyodbc.SQL_CHAR = 1
    pyodbc.SQL_WCHAR = 2
    sys.modules["pyodbc"] = pyodbc

if "kbqa_r1.llm_agent" not in sys.modules:
    llm_agent = types.ModuleType("kbqa_r1.llm_agent")
    llm_agent.__path__ = [
        str(Path(__file__).resolve().parents[1] / "kbqa_r1" / "llm_agent")
    ]
    sys.modules["kbqa_r1.llm_agent"] = llm_agent

from kbqa_r1.llm_agent.sexpr_action_processor import SExprActionProcessor
from kbqa_r1.sexpr.action_parser import ActionResult, ActionType


class _RelationRetrieval:
    def get_candidate_relations(self, function_state, allow_literal_relations=False):
        return [("time.event.start_date", "time.event.start_date")]

    def select_best_relations(self, query, candidates, source=None):
        return [SimpleNamespace(relation_id="time.event.start_date")]


def _processor():
    processor = SExprActionProcessor.__new__(SExprActionProcessor)
    processor.relation_retrieval = _RelationRetrieval()
    return processor


def _action(arguments):
    return ActionResult(
        action_type=ActionType.TIME_CONSTRAINT,
        arguments=list(arguments),
        raw_text="Time_constraint",
        step_number=0,
    )


def test_time_constraint_rejects_invalid_time_after_relation_resolution():
    result = _processor().process_time_constraint_action(
        _action(["start_date", "not-a-time"]),
        ["expression1 = START('m.topic')"],
    )

    assert not result.is_valid
    assert "Invalid time" in result.error_message


def test_time_constraint_requires_exactly_two_arguments():
    result = _processor().process_time_constraint_action(
        _action(["start_date"]),
        ["expression1 = START('m.topic')"],
    )

    assert not result.is_valid
    assert "exactly 2 arguments" in result.error_message
