"""Exact token replay checks for structurally constrained HyPER rollouts."""

from __future__ import annotations

import torch


def align_structural_rollout_log_probs(
    tokenizer,
    sampled_responses: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    replay_responses: torch.Tensor,
) -> torch.Tensor:
    """Align vLLM probabilities to the exact tokens replayed by PPO.

    A stop token may disappear because postprocessing skips special tokens.
    Any other disagreement fails the run instead of silently shifting rollout
    probabilities onto different action or observation tokens.
    """
    if sampled_responses.shape != sampled_log_probs.shape:
        raise RuntimeError("HyPER sampled tokens and rollout log probabilities differ")
    if sampled_responses.shape[0] != replay_responses.shape[0]:
        raise RuntimeError("HyPER replay batch does not match sampled responses")

    pad_id = int(tokenizer.pad_token_id)
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    aligned = torch.zeros(
        replay_responses.shape,
        dtype=sampled_log_probs.dtype,
        device=sampled_log_probs.device,
    )
    sampled_cpu = sampled_responses.detach().cpu()
    replay_cpu = replay_responses.detach().cpu()
    for row in range(replay_responses.shape[0]):
        replay_mask = replay_cpu[row] != pad_id
        replay_length = int(replay_mask.sum().item())
        if bool(replay_mask[replay_length:].any()):
            raise RuntimeError("HyPER replay response is not right padded")
        if replay_length > sampled_responses.shape[1]:
            raise RuntimeError("HyPER replay response is longer than vLLM output")
        if not torch.equal(
            sampled_cpu[row, :replay_length], replay_cpu[row, :replay_length]
        ):
            raise RuntimeError("HyPER decode/re-tokenize changed sampled action tokens")
        trailing = sampled_cpu[row, replay_length:].tolist()
        if any(
            int(token) != pad_id and int(token) not in special_ids
            for token in trailing
        ):
            raise RuntimeError(
                "HyPER postprocessing discarded non-special sampled tokens"
            )
        aligned[row, :replay_length] = sampled_log_probs[row, :replay_length]
    return aligned
