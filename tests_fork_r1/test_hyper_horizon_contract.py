from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_hyper_entrypoints_default_to_32_turns():
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-32}" in _read(
        "scripts/evaluate_hyper_r1.sh"
    )

    rejection = _read("scripts/data_process/rejection_sampling_simple.sh")
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-32}" in rejection
    assert "SEXPR_MAX_TURNS=${SEXPR_MAX_TURNS:-7}" in rejection

    grpo = _read("scripts/train/train_kbqa_sexpr_generation_grpo.sh")
    assert "max_turns=${max_turns:-32}" in grpo
    assert "max_turns=${max_turns:-6}" in grpo


def test_hyper_corpus_helpers_do_not_use_old_24_turn_fallback():
    for path in (
        "kbqa_r1/hyper_data.py",
        "scripts/data_process/regenerate_hyper_control_corpus.py",
    ):
        assert 'get("max_turns", 24)' not in _read(path)
