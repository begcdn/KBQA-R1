"""Prompt contract for HyPER-R1 graph actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Sequence
import re


HYPER_R1_INSTRUCTIONS = """

HyPER-R1 executable hypothesis graph:
- These instructions replace the legacy two-argument Find_relation format. Never write a relation name or free-text relation intent; relation proposals belong to the environment.
- `Find_relation [ source ]` asks the environment to rank relations from the immutable question and that exact executable state. The environment executes a small frontier and returns H-numbered alternatives.
- `Widen [ source ]` asks for the next ranked page from the same open frontier when the visible candidates do not cover the question. It is repeatable while another page exists and must occur before Select.
- When a question requires an intersection, `Find_relation` may open a second root from another supplied candidate entity while earlier hypotheses remain active. Continuing an existing hypothesis still requires `Select` first.
- Use exactly one `Select [ Hn ]` action to continue reasoning from a hypothesis.
- Use `Prune [ Hn ]` only for a visible path or execution contradiction. An empty result is direct negative evidence; low rank, low confidence, and a merely different nonempty answer are not contradiction certificates.
- Use `Combine [ Hn | Hm ]` to intersect two active hypotheses.
- Use `Merge [ expression | ontology_type ]` after Select when the question restricts a retained hypothesis to a Freebase type, such as `religion.religious_leader`. Infer the type from the question; it need not appear among Candidate Entities.
- The existing executable logical actions remain available: `Order [ mode | expression_or_type | relation ]`, `Compare [ mode | relation | value ]`, `Time_constraint [ relation | time ]`, and `Count [ expression ]`. Select an existing hypothesis before applying a continuation operator; Compare and ontology-rooted Order may open an independent branch that can later be combined.
- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. After the environment confirms it, return its values inside <answer>.
- Hypothesis IDs and execution results are owned by the environment. Never invent or edit them.
- Entity labels help interpret evidence; bracketed MIDs remain the exact executable identities.
- The executed-node and action budgets are fixed. Widen only when the extra evidence is needed; never manufacture room by deleting a still-plausible branch.
Preserve plausible alternatives until later execution distinguishes them. Select is not Commit: selecting one hypothesis for expansion does not reject the others. After Commit, perform no more graph actions and copy the committed values exactly into <answer>.
""".rstrip()


def append_hyper_instructions(text: str) -> str:
    if "HyPER-R1 executable hypothesis graph:" in text:
        return text
    return str(text).rstrip() + HYPER_R1_INSTRUCTIONS


def extract_hyper_question(prompt: str) -> str:
    """Recover the immutable benchmark question from a HyPER prompt."""
    matches = list(re.finditer(r"(?:^|\n)Question:\s*", str(prompt)))
    if not matches:
        raise ValueError("HyPER-R1 prompt is missing its Question field")
    remainder = str(prompt)[matches[-1].end():]
    question = remainder.split("\n\nHyPER-R1 executable hypothesis graph:", 1)[0].strip()
    if not question:
        raise ValueError("HyPER-R1 Question field is empty")
    return question


def build_hyper_prompt(
    question: str,
    candidate_entities: Sequence[Sequence[str]] = (),
    base_prompt: str = "",
) -> str:
    """Build the same student-visible protocol used by SFT and RL datasets."""
    entities = []
    for entity in candidate_entities:
        if not entity:
            continue
        name = str(entity[0]) if len(entity) > 1 else str(entity[-1])
        entity_id = str(entity[-1])
        entities.append(f"'{name}' ({entity_id})")
    entity_text = ", ".join(entities) or "none"
    if base_prompt:
        candidate_line = f"Candidate Entities: [{entity_text}]"
        if re.search(r"(?m)^Candidate Entities:\s*.*$", base_prompt):
            base_prompt = re.sub(
                r"(?m)^Candidate Entities:\s*.*$",
                lambda _match: candidate_line,
                base_prompt,
                count=1,
            )
        else:
            base_prompt = re.sub(
                r"(?m)^(Question:\s*)",
                lambda match: candidate_line + "\n" + match.group(1),
                base_prompt,
                count=1,
            )
        return append_hyper_instructions(base_prompt)
    prompt = (
        "You are an expert assistant for querying Freebase with executable actions.\n"
        "Find_relation [ source ] opens a question-conditioned relation frontier; "
        "source must be one of the candidate entity IDs or a selected expression.\n"
        f"Candidate Entities: [{entity_text}]\n"
        f"Question: {question}"
    )
    return append_hyper_instructions(prompt)


def dataset_candidate_entities(row: Dict[str, Any]) -> Sequence[Sequence[str]]:
    """Read candidate entity label/identity pairs from supported dataset schemas."""
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    entities = extra.get("extracted_entities") or extra.get("candidate_entities") or ()
    if entities:
        return entities
    reward = row.get("reward_model")
    ground_truth = reward.get("ground_truth", {}) if isinstance(reward, dict) else {}
    return ground_truth.get("candidate_entities") or ()


def augment_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add the protocol to common KBQA-R1 RL and SFT parquet schemas."""
    result = deepcopy(row)
    entities = dataset_candidate_entities(result)

    def augment_content(content: str) -> str:
        if entities:
            return build_hyper_prompt("", entities, str(content))
        return append_hyper_instructions(content)

    prompt = result.get("prompt")
    if isinstance(prompt, str):
        result["prompt"] = augment_content(prompt)
    elif isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = augment_content(message.get("content", ""))
                break

    messages = result.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = augment_content(message.get("content", ""))
                break
    return result
