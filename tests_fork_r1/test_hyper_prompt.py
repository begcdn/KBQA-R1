from kbqa_r1.hyper_prompt import (
    augment_dataset_row,
    dataset_candidate_entities,
    extract_hyper_question,
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
