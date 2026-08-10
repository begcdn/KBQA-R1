"""Prompt contract for HyPER-R1 graph actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


HYPER_R1_INSTRUCTIONS = """

HyPER-R1 executable hypothesis graph:
- A Find_relation action executes your selected relation and one hard alternative from the exact same state. The environment returns both as H-numbered hypotheses.
- Use exactly one `Select [ Hn ]` action to continue reasoning from a hypothesis.
- Use `Prune [ Hn ]` to reject an active hypothesis.
- Use `Combine [ Hn | Hm ]` to intersect two active hypotheses.
- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. After the environment confirms it, return its values inside <answer>.
- Hypothesis IDs and execution results are owned by the environment. Never invent or edit them.
Preserve plausible alternatives until execution gives you a reason to select or prune them.
""".rstrip()


def append_hyper_instructions(text: str) -> str:
    if "HyPER-R1 executable hypothesis graph:" in text:
        return text
    return str(text).rstrip() + HYPER_R1_INSTRUCTIONS


def augment_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add the protocol to common KBQA-R1 RL and SFT parquet schemas."""
    result = deepcopy(row)
    prompt = result.get("prompt")
    if isinstance(prompt, str):
        result["prompt"] = append_hyper_instructions(prompt)
    elif isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = append_hyper_instructions(message.get("content", ""))
                break

    messages = result.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = append_hyper_instructions(message.get("content", ""))
                break
    return result

