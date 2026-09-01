import sys
import types
from pathlib import Path

import numpy as np
import pytest

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


class FakeMergeState:
    def __init__(self):
        self.functions = [
            "expression1 = START('m.topic')",
            "expression1 = JOIN('r.people', expression1)",
        ]
        self.current = 1

    def get_sample_entities(self, _sample_id):
        return [("Topic", "m.topic")]

    def get_sample_function_state(self, _sample_id):
        return list(self.functions)

    def get_next_expression_id(self, _sample_id):
        self.current += 1
        return self.current

    def update_sample_function_state(self, _sample_id, function):
        self.functions.append(function)


def test_merge_accepts_question_inferred_ontology_type_without_gold_candidate():
    state = FakeMergeState()
    processor = SExprActionProcessor(
        FakeHyperRetrieval(),
        state,
        hyper_frontier_width=6,
    )
    action = ActionResult(
        action_type=ActionType.MERGE,
        arguments=["expression1", "dining.chef"],
        raw_text="Merge [ expression1 | dining.chef ]",
        step_number=1,
    )

    processed = processor.process_merge_action(
        action,
        state.get_sample_function_state(0),
        sample_id=0,
    )

    assert processed.is_valid is True
    assert processed.arguments == ["expression1", "expression2"]
    assert state.functions[-1] == "expression2 = START('dining.chef')"


@pytest.mark.parametrize(
    "ontology_type",
    (
        "base.exoplanetology.exoplanet",
        "user.patrick.default_domain.warship_v1_1",
    ),
)
def test_merge_accepts_multisegment_freebase_ontology_types(ontology_type):
    state = FakeMergeState()
    processor = SExprActionProcessor(
        FakeHyperRetrieval(),
        state,
        hyper_frontier_width=6,
    )
    action = ActionResult(
        action_type=ActionType.MERGE,
        arguments=["expression1", ontology_type],
        raw_text=f"Merge [ expression1 | {ontology_type} ]",
        step_number=1,
    )

    processed = processor.process_merge_action(
        action,
        state.get_sample_function_state(0),
        sample_id=0,
    )

    assert processed.is_valid is True
    assert processed.arguments == ["expression1", "expression2"]
    assert state.functions[-1] == f"expression2 = START('{ontology_type}')"


@pytest.mark.parametrize(
    "ontology_type",
    ("people", "m.0123", "people..person", "people.person-name"),
)
def test_merge_rejects_malformed_ontology_types(ontology_type):
    state = FakeMergeState()
    processor = SExprActionProcessor(
        FakeHyperRetrieval(),
        state,
        hyper_frontier_width=6,
    )
    action = ActionResult(
        action_type=ActionType.MERGE,
        arguments=["expression1", ontology_type],
        raw_text=f"Merge [ expression1 | {ontology_type} ]",
        step_number=1,
    )

    processed = processor.process_merge_action(
        action,
        state.get_sample_function_state(0),
        sample_id=0,
    )

    assert processed.is_valid is False


class FakeCompareRetrieval:
    dataset = "grailqa"
    last_similarity_scores = []

    def select_best_relation_for_cmp(self, relation, _mode):
        return relation


def test_compare_accepts_gold_datetime_with_timezone_offset():
    processor = SExprActionProcessor(
        FakeCompareRetrieval(),
        FakeMergeState(),
        hyper_frontier_width=6,
    )
    value = "2010-06-29T20:30:00-08:00^^http://www.w3.org/2001/XMLSchema#dateTime"
    action = ActionResult(
        action_type=ActionType.COMPARE,
        arguments=["le", "time.event.start_date", value],
        raw_text=f"Compare [ le | time.event.start_date | {value} ]",
        step_number=1,
    )

    processed = processor.process_compare_action(action, sample_id=0)

    assert processed.is_valid is True
    assert processed.arguments[2] == value


@pytest.mark.parametrize(
    "value",
    (
        "2010-06-29T20:30:00-25:00^^http://www.w3.org/2001/XMLSchema#dateTime",
        "2010-13-29T20:30:00Z^^http://www.w3.org/2001/XMLSchema#dateTime",
        "2010-06-29T20:30:00+0800^^http://www.w3.org/2001/XMLSchema#dateTime",
    ),
)
def test_compare_rejects_malformed_typed_datetimes(value):
    processor = SExprActionProcessor(
        FakeCompareRetrieval(),
        FakeMergeState(),
        hyper_frontier_width=6,
    )
    action = ActionResult(
        action_type=ActionType.COMPARE,
        arguments=["le", "time.event.start_date", value],
        raw_text=f"Compare [ le | time.event.start_date | {value} ]",
        step_number=1,
    )

    processed = processor.process_compare_action(action, sample_id=0)

    assert processed.is_valid is False


class FakeOrderRetrieval:
    dataset = "grailqa"
    last_similarity_scores = []

    def __init__(self, exact_relation):
        self.literal_relation_list = ["time.event.start_date", exact_relation]
        self.ranker_calls = []

    def get_candidate_relations(self, _function_state, allow_literal_relations=False):
        assert allow_literal_relations is True
        return [
            ("time.event.start_date", "time.event.start_date"),
        ]

    def select_best_relations(self, query, candidates, source=None):
        self.ranker_calls.append((query, tuple(candidates), source))
        return [
            types.SimpleNamespace(
                relation_name="time.event.start_date",
                relation_id="time.event.start_date",
                score=0.99,
            )
        ]


def test_order_preserves_exact_nested_live_candidate_before_fuzzy_ranking():
    exact = (
        "(JOIN soccer.football_match.substitution "
        "soccer.football_player_substitution.minute)"
    )
    retrieval = FakeOrderRetrieval(exact)
    state = FakeMergeState()
    processor = SExprActionProcessor(retrieval, state, hyper_frontier_width=6)
    action = ActionResult(
        action_type=ActionType.ORDER,
        arguments=["ARGMIN", "expression1", exact],
        raw_text=f"Order [ ARGMIN | expression1 | {exact} ]",
        step_number=1,
    )

    processed = processor.process_order_action(
        action,
        state.get_sample_function_state(0),
        sample_id=0,
    )

    assert processed.is_valid is True
    assert processed.arguments == ["ARGMIN", "expression1", exact]
    assert retrieval.ranker_calls == []


def test_compare_ranker_preserves_an_exact_literal_relation():
    retrieval = RelationRetrieval.__new__(RelationRetrieval)
    retrieval.literal_relation_list = [
        "boats.ship.length_overall",
        "time.event.start_date",
    ]
    retrieval.last_similarity_scores = []

    candidate = retrieval.select_best_relation_for_cmp(
        "time.event.start_date",
        "le",
    )

    assert candidate.relation_id == "time.event.start_date"
    assert candidate.score == 1.0
    assert retrieval.last_similarity_scores == [0.0, 1.0]


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
