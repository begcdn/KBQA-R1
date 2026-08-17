from kbqa_r1.hyper_prompt import (
    augment_dataset_row,
    dataset_candidate_entities,
    extract_hyper_question,
    question_candidate_literals,
)


def test_augment_rl_prompt_once():
    row = {"prompt": [{"role": "user", "content": "Question"}]}
    first = augment_dataset_row(row)
    second = augment_dataset_row(first)
    content = second["prompt"][0]["content"]
    assert content.count("HyPER-R1 executable hypothesis graph:") == 1
    assert "Commit [ Hn ]" in content
    assert "second root" in content
    assert "Merge [ expression | ontology_type ]" in content


def test_augment_sft_messages_without_mutating_source():
    row = {"messages": [{"role": "user", "content": "Question"}]}
    result = augment_dataset_row(row)
    assert "HyPER-R1" not in row["messages"][0]["content"]
    assert "HyPER-R1" in result["messages"][0]["content"]


def test_extracts_immutable_question_before_protocol_instructions():
    content = augment_dataset_row(
        {"prompt": [{"role": "user", "content": "Context\nQuestion: Who founded Apple?"}]}
    )["prompt"][0]["content"]
    assert extract_hyper_question(content) == "Who founded Apple?"
    assert "Find_relation [ source ]" in content


def test_augment_rl_prompt_uses_available_entity_names():
    row = {
        "prompt": [{
            "role": "user",
            "content": "Candidate Entities: ['m.apple' (m.apple)]\nQuestion: Who founded Apple?",
        }],
        "extra_info": {
            "extracted_entities": [["Apple Inc.", "m.apple"]],
        },
    }

    content = augment_dataset_row(row)["prompt"][0]["content"]

    assert "Candidate Entities: ['Apple Inc.' (m.apple)]" in content
    assert "['m.apple' (m.apple)]" not in content


def test_candidate_entities_fall_back_to_reward_metadata():
    entities = [["Apple Inc.", "m.apple"]]
    row = {
        "reward_model": {"ground_truth": {"candidate_entities": entities}},
    }

    assert dataset_candidate_entities(row) == entities


def test_question_literals_are_public_and_deterministic():
    literals = question_candidate_literals(
        "What recipe takes at most 120.0 minutes and uses 1.5 of the ingredient?"
    )

    assert literals == [
        ("120.0", "120.0^^http://www.w3.org/2001/XMLSchema#float"),
        ("1.5", "1.5^^http://www.w3.org/2001/XMLSchema#float"),
    ]


def test_question_literals_normalize_dates_without_gold_programs():
    literals = dict(
        question_candidate_literals(
            "Which events occurred on 07/01/1970 or Feb. the 10th, 2008?"
        )
    )

    assert literals["07/01/1970"].endswith("#date")
    assert literals["07/01/1970"].startswith("1970-07-01^^")
    assert literals["Feb. the 10th, 2008"].startswith("2008-02-10^^")


def test_augment_prompt_exposes_question_literals_as_sources():
    row = {
        "prompt": [{
            "role": "user",
            "content": "Candidate Entities: []\nQuestion: Which recipe uses 1.5 units?",
        }],
        "extra_info": {"original_question": "Which recipe uses 1.5 units?"},
    }

    content = augment_dataset_row(row)["prompt"][0]["content"]

    assert (
        "Candidate Literals: ['1.5' "
        "(1.5^^http://www.w3.org/2001/XMLSchema#float)]"
    ) in content
