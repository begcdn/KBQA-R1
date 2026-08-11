"""Verified demonstration construction for HyPER-R1.

Gold programs determine the executable trajectory.  A language model may later
verbalize a fixed action, but it is never allowed to choose relations, graph
actions, denotations, or the committed answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_ASSIGNMENT = re.compile(r"^\s*(expression\d+)\s*=\s*(.+?)\s*$")
_START = re.compile(r"^START\('(.+)'\)$")
_JOIN = re.compile(r"^JOIN\('(.+)'\s*,\s*(expression\d+|'[^']+')\)$")
_AND = re.compile(r"^AND\((expression\d+)\s*,\s*(expression\d+)\)$")
_STOP = re.compile(r"^STOP\((expression\d+)\)$")


def normalize_values(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values or () if str(value).strip()}))


@dataclass(frozen=True)
class ProgramStatement:
    index: int
    target: str
    kind: str
    sources: Tuple[str, ...]
    relation: Optional[str]
    raw: str


@dataclass(frozen=True)
class GoldPlan:
    statements: Tuple[ProgramStatement, ...]
    executable_functions: Tuple[str, ...]
    target_expression: str

    @property
    def joins(self) -> Tuple[ProgramStatement, ...]:
        return tuple(statement for statement in self.statements if statement.kind == "join")

    @property
    def intersections(self) -> Tuple[ProgramStatement, ...]:
        return tuple(statement for statement in self.statements if statement.kind == "and")


class IneligibleProgram(ValueError):
    pass


def compile_gold_plan(function_list: Sequence[str]) -> GoldPlan:
    """Compile the initial HyPER-R1 action subset: START, JOIN, AND, STOP."""
    statements: List[ProgramStatement] = []
    executable: List[str] = []
    defined = set()
    final_target: Optional[str] = None
    stopped = False

    for index, raw_value in enumerate(function_list):
        raw = str(raw_value).strip()
        if stopped:
            raise IneligibleProgram("STOP must be the terminal statement")
        match = _ASSIGNMENT.match(raw)
        if not match:
            raise IneligibleProgram(f"invalid function statement at {index}: {raw}")
        target, rhs = match.groups()

        start = _START.match(rhs)
        join = _JOIN.match(rhs)
        combine = _AND.match(rhs)
        stop = _STOP.match(rhs)
        if start:
            statement = ProgramStatement(index, target, "start", (), None, raw)
        elif join:
            relation, source = join.groups()
            sources = (source,) if source.startswith("expression") else ()
            statement = ProgramStatement(index, target, "join", sources, relation, raw)
        elif combine:
            statement = ProgramStatement(index, target, "and", combine.groups(), None, raw)
        elif stop:
            source = stop.group(1)
            if source not in defined:
                raise IneligibleProgram(f"STOP references undefined {source}")
            final_target = source
            statements.append(ProgramStatement(index, target, "stop", (source,), None, raw))
            stopped = True
            continue
        else:
            operator = rhs.split("(", 1)[0]
            raise IneligibleProgram(f"unsupported operator {operator or rhs}")

        missing = [source for source in statement.sources if source not in defined]
        if missing:
            raise IneligibleProgram(f"statement {index} references undefined {missing}")
        statements.append(statement)
        executable.append(raw)
        defined.add(target)
        final_target = target

    if not executable or final_target is None:
        raise IneligibleProgram("program has no executable result")
    return GoldPlan(tuple(statements), tuple(executable), final_target)


@dataclass(frozen=True)
class RelationOption:
    relation: str
    score: float
    rank: int


@dataclass(frozen=True)
class ExecutedHypothesis:
    hypothesis_id: str
    function_state: Tuple[str, ...]
    target_expression: str
    denotation: Tuple[str, ...]
    relation: Optional[str] = None
    role: str = "gold"


@dataclass(frozen=True)
class DemonstrationStep:
    action: str
    arguments: Tuple[str, ...]
    visible_before: Tuple[str, ...]
    rationale_facts: Tuple[str, ...] = ()


@dataclass
class HyperDemonstration:
    demo_id: str
    question_id: str
    question: str
    family: str
    hypotheses: Dict[str, ExecutedHypothesis]
    steps: List[DemonstrationStep]
    gold_answers: Tuple[str, ...]
    private_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "question_id": self.question_id,
            "question": self.question,
            "family": self.family,
            "hypotheses": {key: asdict(value) for key, value in self.hypotheses.items()},
            "steps": [asdict(step) for step in self.steps],
            "gold_answers": list(self.gold_answers),
            "private_metadata": self.private_metadata,
        }


ProgramExecutor = Callable[[Sequence[str], str], Iterable[Any]]
CandidateProvider = Callable[
    [str, Sequence[str], ProgramStatement], Sequence[RelationOption]
]


def replace_join_relation(raw: str, relation: str) -> str:
    match = _ASSIGNMENT.match(raw)
    if not match:
        raise ValueError(f"invalid JOIN statement: {raw}")
    target, rhs = match.groups()
    join = _JOIN.match(rhs)
    if not join:
        raise ValueError(f"not a JOIN statement: {raw}")
    _, source = join.groups()
    return f"{target} = JOIN('{relation}', {source})"


def _digest(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class DemonstrationBuilder:
    """Construct verified decision demonstrations from gold programs."""

    def __init__(
        self,
        executor: ProgramExecutor,
        candidate_provider: CandidateProvider,
        max_active: int = 3,
    ):
        if max_active < 2:
            raise ValueError("max_active must permit at least two hypotheses")
        self.executor = executor
        self.candidate_provider = candidate_provider
        self.max_active = int(max_active)

    def build(self, row: Mapping[str, Any]) -> List[HyperDemonstration]:
        question = str(row.get("question") or row.get("original_question") or "").strip()
        question_id = str(row.get("ID") or row.get("id") or _digest(question))
        plan = compile_gold_plan(row.get("function_list") or ())
        annotated = row.get("answer")
        if annotated is None:
            annotated = row.get("answers")
        if annotated is None:
            annotated = row.get("target")
        gold_answers = normalize_values(() if annotated is None else annotated)
        if not gold_answers:
            return []
        executed_gold = normalize_values(self.executor(plan.executable_functions, plan.target_expression))
        if not executed_gold:
            return []
        if gold_answers and executed_gold != gold_answers:
            return []
        demos: List[HyperDemonstration] = []
        for join in plan.joins:
            demo = self._build_relation_demo(question_id, question, plan, join, gold_answers)
            if demo is not None:
                demos.append(demo)
        for combine in plan.intersections:
            demo = self._build_intersection_demo(question_id, question, plan, combine, gold_answers)
            if demo is not None:
                demos.append(demo)
        return demos

    def _build_relation_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        join: ProgramStatement,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        state_before = list(plan.executable_functions[: join.index])
        options = list(self.candidate_provider(question, state_before, join))
        gold_option = next((option for option in options if option.relation == join.relation), None)
        if gold_option is None:
            return None

        gold_prefix = state_before + [join.raw]
        gold_prefix_values = normalize_values(self.executor(gold_prefix, join.target))
        if not gold_prefix_values:
            return None

        accepted: List[Tuple[RelationOption, List[str], Tuple[str, ...], Tuple[str, ...]]] = []
        for option in options:
            if option.relation == join.relation:
                continue
            alternative_statement = replace_join_relation(join.raw, option.relation)
            alternative_prefix = state_before + [alternative_statement]
            prefix_values = normalize_values(self.executor(alternative_prefix, join.target))
            if not prefix_values or prefix_values == gold_prefix_values:
                continue
            full_alternative = list(plan.executable_functions)
            full_alternative[join.index] = alternative_statement
            terminal_values = normalize_values(
                self.executor(full_alternative, plan.target_expression)
            )
            if not terminal_values or terminal_values == gold_answers:
                continue
            accepted.append((option, alternative_prefix, prefix_values, terminal_values))

        # Recovery supervision is useful only when the inference-time retriever
        # would actually prefer the distractor.  Lower-ranked negatives are
        # ordinary contrastive examples, not evidence for delayed commitment.
        accepted = [item for item in accepted if item[0].rank < gold_option.rank]
        if not accepted:
            return None
        accepted.sort(key=lambda item: (-item[0].score, item[0].rank, item[0].relation))
        option, alternative_prefix, prefix_values, terminal_values = accepted[0]
        wrong = ExecutedHypothesis(
            "H0", tuple(alternative_prefix), join.target, prefix_values,
            relation=option.relation, role="distractor",
        )
        gold = ExecutedHypothesis(
            "H1", tuple(gold_prefix), join.target, gold_prefix_values,
            relation=join.relation, role="gold",
        )
        hypotheses = {wrong.hypothesis_id: wrong, gold.hypothesis_id: gold}
        steps = [
            DemonstrationStep("Prune", (wrong.hypothesis_id,), tuple(hypotheses), ("fails_full_intent",)),
            DemonstrationStep("Select", (gold.hypothesis_id,), (gold.hypothesis_id,), ("matches_full_intent",)),
        ]
        if join.target == plan.target_expression and gold_prefix_values == gold_answers:
            steps.append(
                DemonstrationStep(
                    "Commit", (gold.hypothesis_id,), (gold.hypothesis_id,),
                    ("complete", "executable"),
                )
            )
        return HyperDemonstration(
            demo_id=f"{question_id}:join:{join.index}:{_digest(option.relation)}",
            question_id=question_id,
            question=question,
            family="wrong_sibling_recovery",
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "gold_relation": join.relation,
                "gold_rank": gold_option.rank,
                "distractor_relation": option.relation,
                "distractor_rank": option.rank,
                "distractor_score": option.score,
                "distractor_terminal": list(terminal_values),
                "decision_index": join.index,
            },
        )

    def _build_intersection_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        combine: ProgramStatement,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        prefix = list(plan.executable_functions[: combine.index])
        left_target, right_target = combine.sources
        left_values = normalize_values(self.executor(prefix, left_target))
        right_values = normalize_values(self.executor(prefix, right_target))
        combined_prefix = prefix + [combine.raw]
        combined_values = normalize_values(self.executor(combined_prefix, combine.target))
        if not left_values or not right_values or not combined_values:
            return None
        if combined_values != normalize_values(set(left_values) & set(right_values)):
            return None
        if combined_values == left_values or combined_values == right_values:
            return None
        # The first corpus contains complete, replayable conjunction decisions.
        # A downstream suffix needs additional continuation-state supervision
        # and is deliberately deferred instead of being represented falsely.
        if combine.target != plan.target_expression or combined_values != gold_answers:
            return None

        left = ExecutedHypothesis("H0", tuple(prefix), left_target, left_values, role="required_branch")
        right = ExecutedHypothesis("H1", tuple(prefix), right_target, right_values, role="required_branch")
        combined = ExecutedHypothesis(
            "H2", tuple(combined_prefix), combine.target, combined_values, role="combined"
        )
        hypotheses = {node.hypothesis_id: node for node in (left, right, combined)}
        steps = [
            DemonstrationStep("Select", (left.hypothesis_id,), (left.hypothesis_id, right.hypothesis_id), ("required_branch",)),
            DemonstrationStep("Select", (right.hypothesis_id,), (left.hypothesis_id, right.hypothesis_id), ("required_branch",)),
            DemonstrationStep("Combine", (left.hypothesis_id, right.hypothesis_id), (left.hypothesis_id, right.hypothesis_id), ("both_branches_necessary",)),
        ]
        steps.append(
            DemonstrationStep(
                "Commit", (combined.hypothesis_id,), (combined.hypothesis_id,),
                ("complete", "executable"),
            )
        )
        return HyperDemonstration(
            demo_id=f"{question_id}:and:{combine.index}",
            question_id=question_id,
            question=question,
            family="conjunction",
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={"decision_index": combine.index},
        )

class DemonstrationValidator:
    """Replay private executable states and verify graph-action consistency."""

    def __init__(self, executor: ProgramExecutor, max_active: int = 3):
        self.executor = executor
        self.max_active = int(max_active)

    def validate(self, demo: HyperDemonstration) -> List[str]:
        errors: List[str] = []
        for node in demo.hypotheses.values():
            replayed = normalize_values(self.executor(node.function_state, node.target_expression))
            if replayed != node.denotation:
                errors.append(f"{node.hypothesis_id}: replay mismatch")

        active = set(demo.steps[0].visible_before if demo.steps else ())
        committed = None
        for step in demo.steps:
            unknown = [argument for argument in step.arguments if argument not in demo.hypotheses]
            if unknown:
                errors.append(f"{step.action}: unknown hypotheses {unknown}")
                continue
            if set(step.visible_before) != active:
                errors.append(
                    f"{step.action}: visible state {sorted(step.visible_before)} "
                    f"does not match active state {sorted(active)}"
                )
            if step.action == "Prune":
                if step.arguments[0] not in active:
                    errors.append(f"Prune targets inactive {step.arguments[0]}")
                active.discard(step.arguments[0])
            elif step.action == "Select":
                if step.arguments[0] not in active:
                    errors.append(f"Select targets inactive {step.arguments[0]}")
            elif step.action == "Combine":
                left, right = step.arguments
                if left == right:
                    errors.append("Combine requires distinct hypotheses")
                required = {
                    node.hypothesis_id
                    for node in demo.hypotheses.values()
                    if node.role == "required_branch"
                }
                if {left, right} != required:
                    errors.append(
                        f"Combine parents {sorted((left, right))} do not match "
                        f"required branches {sorted(required)}"
                    )
                if left not in active or right not in active:
                    errors.append("Combine requires two active parents")
                active.difference_update((left, right))
                combined = next(
                    (node.hypothesis_id for node in demo.hypotheses.values() if node.role == "combined"),
                    None,
                )
                if combined:
                    active.add(combined)
            elif step.action == "Commit":
                node = demo.hypotheses[step.arguments[0]]
                if node.denotation != demo.gold_answers:
                    errors.append(f"Commit {node.hypothesis_id} does not return gold answers")
                committed = node.hypothesis_id
            else:
                errors.append(f"unsupported action {step.action}")
            if len(active) > self.max_active:
                errors.append(f"active hypothesis budget exceeded: {len(active)}")

        if demo.family == "conjunction":
            combined = next((node for node in demo.hypotheses.values() if node.role == "combined"), None)
            parents = [node for node in demo.hypotheses.values() if node.role == "required_branch"]
            if combined is None or len(parents) != 2:
                errors.append("conjunction requires two parents and one combined hypothesis")
            elif combined.denotation in {parents[0].denotation, parents[1].denotation}:
                errors.append("conjunction branch is not causally necessary")
            elif combined.denotation != normalize_values(
                set(parents[0].denotation) & set(parents[1].denotation)
            ):
                errors.append("combined denotation is not the parent intersection")
            if committed != (combined.hypothesis_id if combined else None):
                errors.append("conjunction must commit its combined hypothesis")
        return errors


def step_sft_records(demo: HyperDemonstration) -> List[Dict[str, Any]]:
    """Export public step supervision; private gold metadata is deliberately omitted."""
    records = []
    for index, step in enumerate(demo.steps):
        visible = []
        for hypothesis_id in step.visible_before:
            node = demo.hypotheses[hypothesis_id]
            visible.append(
                {
                    "id": hypothesis_id,
                    "relation": node.relation,
                    "answer_count": len(node.denotation),
                    # These are ordinary executor observations available at
                    # inference.  The private role (gold/distractor) is omitted.
                    "answer_examples": list(node.denotation[:3]),
                }
            )
        records.append(
            {
                "demo_id": demo.demo_id,
                "step": index,
                "input": {"question": demo.question, "active_hypotheses": visible},
                "target": {"action": step.action, "arguments": list(step.arguments)},
                "metadata": {"family": demo.family, "trajectory_weight": 1.0 / len(demo.steps)},
            }
        )
    return records
