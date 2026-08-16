import json

import pytest

from scripts.evaluate_hyper_sft_checkpoints import (
    action_signature,
    depth_band,
    resolve_checkpoint,
    state_depth,
    stratified_sample,
    summarize,
)


def _row(family, depth, action, index=0):
    graph = "" if depth < 0 else f"H0 [active] depth={depth} answers=1: x"
    return {
        "messages": [
            {"role": "user", "content": graph},
            {"role": "assistant", "content": f"<action>{action}</action>"},
        ],
        "extra_info": {"family": family, "decision_index": index},
    }


def test_action_signature_requires_one_tagged_action():
    assert action_signature("<action>Combine [ H4 | H2 ]</action>") == (
        "Combine",
        ("H2", "H4"),
    )
    assert action_signature("Commit [ H1 ]") is None
    assert action_signature(
        "<action>Select [ H1 ]</action><action>Commit [ H1 ]</action>"
    ) is None
    assert action_signature(
        "<action>Select [ H1 ] Commit [ H1 ]</action>"
    ) is None
    assert action_signature(
        "<action>Select [ H1 ] trailing commentary</action>"
    ) is None


def test_state_depth_uses_latest_public_graph():
    messages = [
        {"role": "user", "content": "H0 [active] depth=0"},
        {"role": "assistant", "content": "<action>Select [ H0 ]</action>"},
        {"role": "user", "content": "H1 [active] depth=1\nH2 [active] depth=3"},
    ]
    assert state_depth(messages) == 3
    assert depth_band(3) == "depth_2_plus"


def test_stratified_sample_keeps_rare_deep_state():
    rows = [_row("direct", 0, "Commit [ H0 ]", index=i) for i in range(20)]
    rows.append(_row("deep_frontier_progress", 3, "Find_relation [ expression3 ]"))
    selected = stratified_sample(rows, 2)
    assert any(state_depth(row["messages"][:-1]) == 3 for row in selected)


def test_resolve_checkpoint_rejects_missing_indexed_shard(tmp_path):
    model = tmp_path / "global_step_1" / "huggingface"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (model / "model-00001-of-00002.safetensors").write_bytes(b"one")
    with pytest.raises(ValueError, match="missing 1 indexed weight shard"):
        resolve_checkpoint(tmp_path / "global_step_1")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert resolve_checkpoint(tmp_path / "global_step_1") == model


def test_behavior_screen_rejects_absent_deep_evidence():
    row = {
        "family": "direct",
        "depth_band": "depth_0",
        "target_type": "Commit",
        "predicted_type": "Commit",
        "decision_index": 1,
        "parsable_single_action": True,
        "action_type_correct": True,
        "exact_action": True,
        "premature_commit": False,
    }
    metrics = summarize([row])
    assert metrics["exact_action_accuracy"] == 1.0
    assert not metrics["minimum_behavior_screen"]["passed"]
    assert not metrics["minimum_behavior_screen"]["checks"]["deep_states_present"]
