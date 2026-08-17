import pytest

pytest.importorskip("pyodbc")

from kbqa_r1.sexpr.dynamic_relation_retrieval import DynamicRelationRetrieval


def _retrieval_without_backends():
    return DynamicRelationRetrieval.__new__(DynamicRelationRetrieval)


def test_detects_grailqa_class_and_typed_literal_starts():
    retrieval = _retrieval_without_backends()

    assert retrieval._detect_entity_type("people.person") == "onto"
    assert retrieval._detect_entity_type("m.0123") == "entity"
    assert (
        retrieval._detect_entity_type(
            "7.0^^http://www.w3.org/2001/XMLSchema#float"
        )
        == "literal"
    )


def test_ontology_frontier_queries_class_instances_in_both_directions():
    retrieval = _retrieval_without_backends()

    forward = retrieval._generate_ontology_query("people.person", "forward")
    reverse = retrieval._generate_ontology_query("people.person", "reverse")

    type_clause = "?x ns:type.object.type ns:people.person ."
    assert type_clause in forward
    assert "?y ?relation ?x ." in forward
    assert type_clause in reverse
    assert "?x ?relation ?y ." in reverse


def test_typed_literal_uses_relation_template_query():
    retrieval = _retrieval_without_backends()
    retrieval._generate_template_query = lambda functions, expression, direction: (
        f"template:{expression}:{direction}"
    )

    query = retrieval._generate_relation_query(
        "7.0^^http://www.w3.org/2001/XMLSchema#float",
        "forward",
        "",
        ["expression = START('7.0^^http://www.w3.org/2001/XMLSchema#float')"],
    )

    assert query == "template::forward"
