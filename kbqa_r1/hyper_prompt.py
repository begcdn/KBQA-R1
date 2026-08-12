"""Prompt contract for HyPER-R1 graph actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Sequence


HYPER_R1_INSTRUCTIONS = """

HyPER-R1 executable hypothesis graph:
- A Find_relation action proposes and executes a small ranked frontier of relations from the exact same state. The environment returns the successful alternatives as H-numbered hypotheses.
- Use exactly one `Select [ Hn ]` action to continue reasoning from a hypothesis.
- Use `Prune [ Hn ]` only when its visible path or execution result contradicts the question. An empty result is direct negative evidence; a merely different nonempty answer is not.
- Use `Combine [ Hn | Hm ]` to intersect two active hypotheses.
- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. After the environment confirms it, return its values inside <answer>.
- Hypothesis IDs and execution results are owned by the environment. Never invent or edit them.
- The active frontier is bounded. Prune unsupported hypotheses before exploring when it is full.
Preserve plausible alternatives until later execution distinguishes them. Select is not Commit: selecting one hypothesis for expansion does not reject the others. After Commit, perform no more graph actions and copy the committed values exactly into <answer>.
""".rstrip()


def append_hyper_instructions(text: str) -> str:
    if "HyPER-R1 executable hypothesis graph:" in text:
        return text
    return str(text).rstrip() + HYPER_R1_INSTRUCTIONS


def build_hyper_prompt(
    question: str,
    candidate_entities: Sequence[Sequence[str]] = (),
    base_prompt: str = "",
) -> str:
    """Build the same student-visible protocol used by SFT and RL datasets."""
    if base_prompt:
        return append_hyper_instructions(base_prompt)
    entities = []
    for entity in candidate_entities:
        if not entity:
            continue
        name = str(entity[0]) if len(entity) > 1 else str(entity[-1])
        entity_id = str(entity[-1])
        entities.append(f"'{name}' ({entity_id})")
    entity_text = ", ".join(entities) or "none"
    prompt = (
        "You are an expert assistant for querying Freebase with executable actions.\n"
        "Find_relation [ source | relation intent ] opens a relation frontier; "
        "source must be one of the candidate entity IDs or a selected expression.\n"
        f"Candidate Entities: [{entity_text}]\n"
        f"Question: {question}"
    )
    return append_hyper_instructions(prompt)


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
