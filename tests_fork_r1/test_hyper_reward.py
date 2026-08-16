import importlib.util
from pathlib import Path

from kbqa_custom_reward import compute_score
from kbqa_r1.answer_utils import extract_last_answer_values


MODULE_PATH = Path(__file__).parents[1] / "verl" / "utils" / "reward_score" / "mid_reward.py"
SPEC = importlib.util.spec_from_file_location("hyper_mid_reward", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
extract_mid_list = MODULE.extract_mid_list


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


def test_runtime_and_reward_share_answer_normalization():
    text = "prefix <AnSwEr>['m.two', 'm.one']</aNsWeR>"

    assert extract_last_answer_values(text) == ("m.one", "m.two")
    assert extract_mid_list(text) == ["m.one", "m.two"]


def test_answer_parser_distinguishes_missing_and_empty_tags():
    assert extract_last_answer_values("no answer") is None
    assert extract_last_answer_values("<answer> </answer>") == ()
    assert extract_last_answer_values("<answer>m.one, m.two</answer>") == (
        "m.one",
        "m.two",
    )
