"""KBQA-R1 answer reward exposed through VERL's custom reward contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REWARD_DIR = Path(__file__).parent / "verl" / "utils" / "reward_score"
_MID_REWARD = _load_module("kbqa_mid_reward", _REWARD_DIR / "mid_reward.py")
_SEXPR_FORMAT = _load_module("kbqa_sexpr_format", _REWARD_DIR / "sexpr_format.py")


def _answers(ground_truth: Any) -> Any:
    if isinstance(ground_truth, Mapping):
        for key in ("target", "answer", "answers", "answer_mids"):
            if key in ground_truth:
                return ground_truth[key]
    return ground_truth


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    mid_f1_weight: float = 1.0,
    structure_format_score: float = 0.1,
    **_: Any,
) -> dict[str, float]:
    """Return answer F1 and format components in KBQARewardManager's schema."""
    # Compute the released answer reward without importing VERL's top-level
    # package (which initializes Ray). Format validation is added explicitly.
    result = _MID_REWARD.compute_mid_reward(
        solution_str=solution_str,
        ground_truth=_answers(ground_truth),
        correct_reward=float(mid_f1_weight),
        structure_format_score=0.0,
        training_step=int((extra_info or {}).get("training_step", 0) or 0),
    )
    structure_reward = 0.0
    if result["mid_f1"] > 0 and float(structure_format_score):
        valid, _ = _SEXPR_FORMAT.is_valid_sexpr_sequence(solution_str)
        if valid:
            structure_reward = float(structure_format_score)
    return {
        "score": float(result["total"]) + structure_reward,
        "mid_f1": float(result["mid_f1"]),
        "structure_reward": structure_reward,
        "timeout_penalty": float(result["timeout_penalty"]),
    }
