import sys
import types

import numpy as np

if "pyodbc" not in sys.modules:
    pyodbc = types.ModuleType("pyodbc")
    pyodbc.Connection = object
    pyodbc.Error = Exception
    pyodbc.SQL_CHAR = 1
    pyodbc.SQL_WCHAR = 2
    sys.modules["pyodbc"] = pyodbc

from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval


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
