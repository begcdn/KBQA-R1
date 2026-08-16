"""Shared answer-tag parsing for HyPER execution and reward scoring."""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional, Tuple


def normalize_answer_values(values: Iterable[object]) -> Tuple[str, ...]:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sorted(normalized))


def extract_last_answer_values(text: str) -> Optional[Tuple[str, ...]]:
    """Return the final answer tag, distinguishing no tag from an empty tag."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", str(text), re.I | re.S))
    if not matches:
        return None
    content = matches[-1].group(1).strip()
    if not content:
        return ()
    if content.startswith("[") and content.endswith("]"):
        try:
            values = json.loads(content.replace("'", '"'))
        except json.JSONDecodeError:
            values = None
        if isinstance(values, list):
            return normalize_answer_values(values)
    return normalize_answer_values(re.split(r"[\s,]+", content))
