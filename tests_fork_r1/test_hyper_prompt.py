from kbqa_r1.hyper_prompt import augment_dataset_row


def test_augment_rl_prompt_once():
    row = {"prompt": [{"role": "user", "content": "Question"}]}
    first = augment_dataset_row(row)
    second = augment_dataset_row(first)
    content = second["prompt"][0]["content"]
    assert content.count("HyPER-R1 executable hypothesis graph:") == 1
    assert "Commit [ Hn ]" in content


def test_augment_sft_messages_without_mutating_source():
    row = {"messages": [{"role": "user", "content": "Question"}]}
    result = augment_dataset_row(row)
    assert "HyPER-R1" not in row["messages"][0]["content"]
    assert "HyPER-R1" in result["messages"][0]["content"]
