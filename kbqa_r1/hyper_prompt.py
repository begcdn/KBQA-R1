"""Prompt contract for HyPER-R1 graph actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple
import re


HYPER_R1_INSTRUCTIONS = """

HyPER-R1 executable hypothesis graph:
- These instructions replace the legacy two-argument Find_relation format. Never write a relation name or free-text relation intent; relation proposals belong to the environment.
- `Find_relation [ source ]` asks the environment to rank every structurally legal relation from the immutable question and that exact executable state. It exposes P-numbered symbolic proposals and executes nothing.
- `Widen [ source ]` exposes the next stable proposal page. Browsing proposals does not consume graph-execution budget.
- `Inspect [ Pn ]` executes exactly one visible proposal and creates an H-numbered hypothesis. Inspect only proposals that could discharge part of the question.
- When a question requires an intersection, `Find_relation` may open a second root from another supplied candidate entity or candidate literal while earlier hypotheses remain active. Continuing an existing hypothesis still requires `Select` first.
- Use exactly one `Select [ Hn ]` action to continue reasoning from a hypothesis.
- `Park [ Hn ]` reversibly removes an unresolved hypothesis from the visible workspace; `Recall [ Hn ]` restores it. Parking is not evidence that a path is wrong.
- Use `Prune [ Hn ]` only for a visible path or execution contradiction. An empty result is terminal negative evidence only when no pending `Count` obligation can turn it into the valid answer zero; otherwise Park it. Low rank, low confidence, and a merely different nonempty answer are not contradiction certificates.
- Use `Combine [ Hn | Hm ]` to intersect two active hypotheses.
- Use `Merge [ expression | ontology_type ]` after Select when the question restricts a retained hypothesis to a Freebase type, such as `religion.religious_leader`. Infer the type from the question; it need not appear among Candidate Entities.
- The existing executable logical actions remain available: `Order [ mode | expression_or_type | relation ]`, `Compare [ mode | relation | value ]`, `Time_constraint [ relation | time ]`, and `Count [ expression ]`. Select an existing hypothesis before applying a continuation operator; Compare and ontology-rooted Order may open an independent branch that can later be combined.
- Use `Commit [ Hn ]` when one hypothesis expresses the complete question. Commit is terminal; the environment returns that hypothesis's values.
- Before resources expire, select or recall the strongest supported nonempty hypothesis and `Commit` it; returning the best available answer is always preferred to abstaining.
- Hypothesis IDs and execution results are owned by the environment. Never invent or edit them.
- Every observation lists current action targets. Never reuse a hypothesis or proposal ID that is absent from the corresponding target list, even if it appeared earlier in the conversation.
- Entity labels help interpret evidence; bracketed MIDs remain the exact executable identities.
- Proposal browsing, visible workspace, stored hypotheses, execution attempts, and policy turns have separate budgets. Never manufacture room by Pruning a still-plausible branch.
Preserve plausible alternatives until later execution distinguishes them. Select is not Commit: selecting one hypothesis for expansion does not reject the others. Commit ends the search.
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
    candidate_literals: Sequence[Sequence[str]] = (),
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
    literals = []
    for literal in candidate_literals:
        if not literal:
            continue
        surface = str(literal[0]) if len(literal) > 1 else str(literal[-1])
        literal_id = str(literal[-1])
        literals.append(f"'{surface}' ({literal_id})")
    literal_text = ", ".join(literals) or "none"
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
        literal_line = f"Candidate Literals: [{literal_text}]"
        if re.search(r"(?m)^Candidate Literals:\s*.*$", base_prompt):
            base_prompt = re.sub(
                r"(?m)^Candidate Literals:\s*.*$",
                lambda _match: literal_line,
                base_prompt,
                count=1,
            )
        else:
            base_prompt = re.sub(
                r"(?m)^(Question:\s*)",
                lambda match: literal_line + "\n" + match.group(1),
                base_prompt,
                count=1,
            )
        return append_hyper_instructions(base_prompt)
    prompt = (
        "You are an expert assistant for querying Freebase with executable actions.\n"
        "Find_relation [ source ] opens a question-conditioned relation frontier; "
        "source must be a supplied candidate entity, candidate literal, or selected expression.\n"
        f"Candidate Entities: [{entity_text}]\n"
        f"Candidate Literals: [{literal_text}]\n"
        f"Question: {question}"
    )
    return append_hyper_instructions(prompt)


def dataset_candidate_entities(row: Dict[str, Any]) -> Sequence[Sequence[str]]:
    """Return the explicit topic-entity field shared by SFT and RL.

    Oracle entity linking is a declared benchmark setting, but the runtime
    prompt builder must not recover those entities from private gold programs.
    Corpus construction stores the oracle roots in these public candidate
    fields before training or evaluation begins.
    """
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    entities = (
        extra.get("extracted_entities")
        or extra.get("candidate_entities")
        or ()
    )
    return [
        [str(entity[0]), str(entity[-1])]
        for entity in entities
        if entity and str(entity[-1]).startswith(("m.", "g."))
    ]


_XSD = "http://www.w3.org/2001/XMLSchema#"
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def question_candidate_literals(question: str) -> List[Tuple[str, str]]:
    """Return executable literal sources derived only from question text."""
    text = str(question)
    occupied: List[Tuple[int, int]] = []
    values: List[Tuple[str, str]] = []

    def add(match: re.Match, lexical: str, datatype: str) -> None:
        occupied.append(match.span())
        canonical = f"{lexical}^^{_XSD}{datatype}"
        if canonical not in {value for _, value in values}:
            values.append((match.group(0), canonical))

    datetime_pattern = re.compile(
        r"(?<!\w)(\d{4}-\d{2}-\d{2})[tT](\d{2}:\d{2}:\d{2})([zZ]|[+-]\d{2}:\d{2})(?!\w)"
    )
    for match in datetime_pattern.finditer(text):
        lexical = f"{match.group(1)}T{match.group(2)}{match.group(3).upper()}"
        add(match, lexical, "dateTime")

    for match in re.finditer(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?![\dTt])", text):
        add(match, match.group(0), "date")

    for match in re.finditer(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)", text):
        month, day, year = map(int, match.groups())
        add(match, f"{year:04d}-{month:02d}-{day:02d}", "date")

    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    month_pattern = re.compile(
        rf"(?i)(?<!\w)({month_names})\.?\s+(?:the\s+)?"
        r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(-?\d{1,4})(?!\d)"
    )
    for match in month_pattern.finditer(text):
        month = _MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        add(match, f"{year:04d}-{month:02d}-{day:02d}", "date")

    for match in re.finditer(r"(?<!\d)(-?\d{3,4})-(\d{2})(?!-?\d)", text):
        if any(start <= match.start() < stop for start, stop in occupied):
            continue
        add(match, match.group(0), "gYearMonth")

    number_pattern = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
    for match in number_pattern.finditer(text):
        if any(start <= match.start() < stop for start, stop in occupied):
            continue
        lexical = match.group(0)
        if "." in lexical:
            add(match, lexical, "float")
            continue
        add(match, lexical, "integer")
        if 3 <= len(lexical.lstrip("-")) <= 4:
            canonical = f"{lexical}^^{_XSD}gYear"
            if canonical not in {value for _, value in values}:
                values.append((match.group(0), canonical))
    return values


def augment_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add the protocol to common KBQA-R1 RL and SFT parquet schemas."""
    result = deepcopy(row)
    entities = dataset_candidate_entities(result)
    extra = result.get("extra_info") if isinstance(result.get("extra_info"), dict) else {}
    original_question = str(extra.get("original_question", ""))

    def augment_content(content: str) -> str:
        question = original_question
        if not question:
            match = re.search(r"(?m)^Question:\s*(.+)$", str(content))
            question = match.group(1).strip() if match else ""
        literals = question_candidate_literals(question)
        if entities or literals:
            return build_hyper_prompt(
                question, entities, str(content), candidate_literals=literals
            )
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
