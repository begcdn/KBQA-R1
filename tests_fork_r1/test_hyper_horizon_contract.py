from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_hyper_entrypoints_default_to_32_turns():
    evaluation = _read("scripts/evaluate_hyper_r1.sh")
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-32}" in evaluation
    assert "HYPER_R1_STRUCTURAL_CONSTRAINTS=${HYPER_R1_STRUCTURAL_CONSTRAINTS:-true}" in evaluation

    rejection = _read("scripts/data_process/rejection_sampling_simple.sh")
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-32}" in rejection
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-7}" in rejection
    assert "hyper_r1.structural_constraints=${HYPER_R1_STRUCTURAL_CONSTRAINTS}" in rejection
    assert "guided_decoding_backend=${GUIDED_DECODING_BACKEND}" in rejection
    assert "guided_decoding_disable_fallback=${GUIDED_DECODING_DISABLE_FALLBACK}" in rejection

    grpo = _read("scripts/train/train_kbqa_sexpr_generation_grpo.sh")
    assert "max_turns=${max_turns:-32}" in grpo
    assert "max_turns=${max_turns:-6}" in grpo
    assert "guided_decoding_backend=${guided_decoding_backend}" in grpo
    assert "guided_decoding_disable_fallback=${guided_decoding_disable_fallback}" in grpo

    hyper_grpo = _read("scripts/train/train_hyper_r1_grpo.sh")
    assert "hyper_r1_structural_constraints=${hyper_r1_structural_constraints:-true}" in hyper_grpo


def test_vllm_version_bridge_moves_backend_selection_to_engine():
    rollout = _read("verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")
    assert 'minver="0.10.0"' in rollout
    assert 'guided_kwargs["backend"] = "xgrammar:no-fallback"' in rollout


def test_hyper_corpus_helpers_do_not_use_old_24_turn_fallback():
    for path in (
        "kbqa_r1/hyper_data.py",
        "scripts/data_process/regenerate_hyper_control_corpus.py",
    ):
        assert 'get("max_turns", 24)' not in _read(path)
