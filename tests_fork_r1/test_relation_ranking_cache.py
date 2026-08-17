import sys
import types
from pathlib import Path

import numpy as np

if "pyodbc" not in sys.modules:
    pyodbc = types.ModuleType("pyodbc")
    pyodbc.Connection = object
    pyodbc.Error = Exception
    pyodbc.SQL_CHAR = 1
    pyodbc.SQL_WCHAR = 2
    sys.modules["pyodbc"] = pyodbc

# Import the action processor without executing kbqa_r1.llm_agent.__init__,
# which eagerly imports the optional distributed-training stack.
if "kbqa_r1.llm_agent" not in sys.modules:
    llm_agent = types.ModuleType("kbqa_r1.llm_agent")
    llm_agent.__path__ = [
        str(Path(__file__).resolve().parents[1] / "kbqa_r1" / "llm_agent")
    ]
    sys.modules["kbqa_r1.llm_agent"] = llm_agent

from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval
from kbqa_r1.llm_agent.sexpr_action_processor import SExprActionProcessor
from kbqa_r1.sexpr.action_parser import ActionResult, ActionType


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        vectors = {
            "where was the person born": [1.0, 0.0],
            "place of birth": [1.0, 0.0],
            "place of death": [0.0, 1.0],
            "nationality": [0.5, 0.5],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_rank_relations_reuses_question_and_relation_embeddings():
    retrieval = RelationRetrieval.__new__(RelationRetrieval)
    retrieval.simcse_rel = FakeEncoder()
    retrieval._ranking_embedding_cache = {}

    first = retrieval.rank_relations_no_threshold(
        "where was the person born",
        [("place of birth", "born"), ("place of death", "died")],
    )
    second = retrieval.rank_relations_no_threshold(
        "where was the person born",
        [("place of birth", "born"), ("nationality", "nationality")],
    )

    assert [candidate.relation_id for candidate in first] == ["born", "died"]
    assert [candidate.relation_id for candidate in second] == ["born", "nationality"]
    assert retrieval.simcse_rel.calls == [
        ["where was the person born", "place of birth", "place of death"],
        ["nationality"],
    ]


def test_hyper_ranking_can_preserve_every_structural_candidate():
    retrieval = RelationRetrieval.__new__(RelationRetrieval)
    retrieval.simcse_rel = None
    candidates = [(f"unrelated relation {index}", f"r.{index}") for index in range(13)]
    candidates.append(("where was the person born", "r.required"))

    ranked = retrieval.rank_relations_no_threshold(
        "where was the person born",
        candidates,
        topk=None,
    )

    assert len(ranked) == len(candidates)
    assert {candidate.relation_id for candidate in ranked} == {
        relation_id for _, relation_id in candidates
    }
    assert [candidate.relation_id for candidate in ranked].count("r.required") == 1


class FakeHyperState:
    def get_sample_prompt(self, _sample_id):
        return (
            "Candidate Entities: ['Ada Lovelace' (m.ada)]\n"
            "Question: Where was Ada Lovelace born?\n\n"
            "HyPER-R1 executable hypothesis graph:\n- instructions"
        )

    def get_sample_function_state(self, _sample_id):
        return []

    def snapshot_sample_state(self, _sample_id):
        return {
            "function_state": [],
            "expression_counter": 0,
            "entities": [("Ada Lovelace", "m.ada")],
            "prompt": self.get_sample_prompt(_sample_id),
        }


class FakeHyperRetrieval:
    dataset = "grailqa"

    def __init__(self):
        self.queries = []
        self.last_similarity_scores = []

    def rank_relations_no_threshold(self, query, candidates, topk=None):
        self.queries.append((query, tuple(candidates), topk))
        ranked = [
            types.SimpleNamespace(
                relation_name=name,
                relation_id=relation_id,
                score=1.0 - index * 0.1,
            )
            for index, (name, relation_id) in enumerate(candidates)
        ]
        return ranked


def test_hyper_find_relation_uses_immutable_question_and_one_public_argument():
    retrieval = FakeHyperRetrieval()
    processor = SExprActionProcessor(
        retrieval,
        FakeHyperState(),
        hyper_frontier_width=6,
    )
    processor._get_candidate_relations_by_entity_type = lambda *_args: [
        ("people.person.place_of_birth", "people.person.place_of_birth"),
        ("people.deceased_person.place_of_death", "people.deceased_person.place_of_death"),
    ]
    action = ActionResult(
        action_type=ActionType.FIND_RELATION,
        arguments=["m.ada"],
        raw_text="Find_relation [ m.ada ]",
        step_number=1,
    )
    action.action_index = 0

    processed = processor.process_find_relation_action(action, [], sample_id=0)

    assert processed.is_valid is True
    assert processed.arguments == ["m.ada", "people.person.place_of_birth"]
    assert retrieval.queries == [
        (
            "Where was Ada Lovelace born?",
            (
                ("people.person.place_of_birth", "people.person.place_of_birth"),
                ("people.deceased_person.place_of_death", "people.deceased_person.place_of_death"),
            ),
            None,
        )
    ]
    decision = processor.get_fork_r1_decisions(0)[0]
    assert decision.relation_prompt == "Where was Ada Lovelace born?"


def test_hyper_find_relation_rejects_student_supplied_relation_text():
    processor = SExprActionProcessor(
        FakeHyperRetrieval(),
        FakeHyperState(),
        hyper_frontier_width=6,
    )
    action = ActionResult(
        action_type=ActionType.FIND_RELATION,
        arguments=["m.ada", "place of birth"],
        raw_text="Find_relation [ m.ada | place of birth ]",
        step_number=1,
    )

    processed = processor.process_find_relation_action(action, [], sample_id=0)

    assert processed.is_valid is False
    assert "environment owns relation ranking" in processed.error_message
