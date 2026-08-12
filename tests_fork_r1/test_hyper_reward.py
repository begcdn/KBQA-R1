from kbqa_custom_reward import compute_score


def test_custom_reward_exposes_kbqa_manager_score_key():
    result = compute_score(
        data_source="grailqa",
        solution_str="<answer>m.answer</answer>",
        ground_truth={"target": ["m.answer"]},
        structure_format_score=0.0,
    )

    assert result["score"] == 1.0
    assert result["mid_f1"] == 1.0


def test_custom_reward_accepts_space_separated_multi_answer_commit():
    result = compute_score(
        data_source="grailqa",
        solution_str="<answer>m.one m.two</answer>",
        ground_truth={"target": ["m.two", "m.one"]},
        structure_format_score=0.0,
    )

    assert result["score"] == 1.0
