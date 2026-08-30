from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray")

from kbqa_r1.llm_agent.sexpr_batch_utils import SExprBatchUtils


class _TensorHelper:
    @staticmethod
    def create_attention_mask(values):
        return values != 0


def _batch_utils(max_length=10):
    value = SExprBatchUtils.__new__(SExprBatchUtils)
    value.tensor_fn = _TensorHelper()
    value.config = SimpleNamespace(max_prompt_length=max_length)
    value.tokenizer = SimpleNamespace(pad_token_id=0)
    return value


def test_multiturn_masks_and_log_probs_follow_response_compaction():
    utils = _batch_utils()
    right = {
        "responses": torch.tensor([[11, 0]]),
        "responses_with_info_mask": torch.tensor([[11, 0]]),
        "rollout_log_probs": torch.tensor([[0.1, 0.0]]),
        "hyper_action_ids": torch.tensor([[1, 0]]),
        "hyper_invalid_action_mask": torch.tensor([[False, False]]),
        "hyper_constraint_turn_ids": torch.tensor([[0, -1]]),
        "hyper_tail_truncated": torch.tensor([False]),
    }

    result = utils.update_right_side(
        right,
        torch.tensor([[22, 0]]),
        cur_rollout_log_probs=torch.tensor([[0.2, 0.0]]),
        next_obs_ids=torch.tensor([[33, 0]]),
        cur_action_ids=torch.tensor([[2, 0]]),
        cur_invalid_action_mask=torch.tensor([[True, False]]),
        cur_constraint_turn_ids=torch.tensor([[1, -1]]),
    )

    assert result["responses"].tolist() == [[11, 22, 33]]
    assert torch.allclose(
        result["rollout_log_probs"], torch.tensor([[0.1, 0.2, 0.0]])
    )
    assert result["hyper_action_ids"].tolist() == [[1, 2, 0]]
    assert result["hyper_invalid_action_mask"].tolist() == [[False, True, False]]
    assert result["hyper_constraint_turn_ids"].tolist() == [[0, 1, -1]]


def test_hyper_rollout_marks_policy_token_truncation():
    utils = _batch_utils(max_length=2)
    right = {
        "responses": torch.tensor([[11]]),
        "responses_with_info_mask": torch.tensor([[11]]),
        "hyper_action_ids": torch.tensor([[1]]),
        "hyper_invalid_action_mask": torch.tensor([[False]]),
        "hyper_tail_truncated": torch.tensor([False]),
    }

    result = utils.update_right_side(
        right,
        torch.tensor([[22, 33]]),
        cur_action_ids=torch.tensor([[2, 2]]),
        cur_invalid_action_mask=torch.tensor([[False, False]]),
    )

    assert result["hyper_tail_truncated"].tolist() == [True]
