from types import SimpleNamespace

import pytest
import torch

from kbqa_r1.token_replay import align_structural_rollout_log_probs


TOKENIZER = SimpleNamespace(pad_token_id=0, all_special_ids=[0, 99])


def test_structural_log_probs_follow_exact_replayed_tokens():
    sampled = torch.tensor([[11, 12, 99, 0], [21, 22, 23, 0]])
    log_probs = torch.tensor(
        [[-0.1, -0.2, -0.3, -1.0], [-0.4, -0.5, -0.6, -1.0]]
    )
    replay = torch.tensor([[11, 12, 0], [21, 22, 23]])

    aligned = align_structural_rollout_log_probs(
        TOKENIZER, sampled, log_probs, replay
    )

    assert torch.allclose(
        aligned,
        torch.tensor([[-0.1, -0.2, 0.0], [-0.4, -0.5, -0.6]]),
    )


def test_structural_replay_rejects_changed_tokenization():
    with pytest.raises(RuntimeError, match="re-tokenize changed"):
        align_structural_rollout_log_probs(
            TOKENIZER,
            torch.tensor([[11, 12, 0]]),
            torch.tensor([[-0.1, -0.2, -1.0]]),
            torch.tensor([[11, 13]]),
        )


def test_structural_replay_rejects_discarded_ordinary_tokens():
    with pytest.raises(RuntimeError, match="discarded non-special"):
        align_structural_rollout_log_probs(
            TOKENIZER,
            torch.tensor([[11, 12, 13]]),
            torch.tensor([[-0.1, -0.2, -0.3]]),
            torch.tensor([[11, 12]]),
        )
