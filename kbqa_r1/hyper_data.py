"""Verified demonstration construction for HyPER-R1.

Gold programs determine the executable trajectory.  A language model may later
verbalize a fixed action, but it is never allowed to choose relations, graph
actions, denotations, or the committed answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hyper_prompt import append_hyper_instructions
from .hyper_r1 import combine_function_states


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
    created: Tuple[str, ...] = ()
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


def _join_source(raw: str) -> str:
    match = _ASSIGNMENT.match(raw)
    if not match:
        raise ValueError(f"invalid JOIN statement: {raw}")
    join = _JOIN.match(match.group(2))
    if not join:
        raise ValueError(f"not a JOIN statement: {raw}")
    return join.group(2).strip("'")


def _action_source(state_before: Sequence[str], raw: str) -> str:
    source = _join_source(raw)
    if not source.startswith("expression"):
        return source
    for statement in reversed(state_before):
        match = _ASSIGNMENT.match(statement)
        if not match or match.group(1) != source:
            continue
        start = _START.match(match.group(2))
        if start:
            return start.group(1)
        break
    return source


def relation_hint(relation: str) -> str:
    """Render the same semantic relation request used by the student policy."""
    value = str(relation).strip()
    reverse = value.startswith("(R ") and value.endswith(")")
    if reverse:
        value = value[3:-1]
    phrase = value.rsplit(".", 1)[-1].replace("_", " ")
    return f"reverse {phrase}" if reverse else phrase


def _is_immediate_linear_terminal(
    plan: GoldPlan,
    current: ProgramStatement,
    following: Optional[ProgramStatement],
) -> bool:
    """Return whether one more JOIN completes this linear gold branch."""
    return bool(
        following is not None
        and following.index == current.index + 1
        and following.sources == (current.target,)
        and following.target == plan.target_expression
        and following.raw == plan.executable_functions[-1]
    )


def _digest(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class DemonstrationBuilder:
    """Construct replay-verified frontier trajectories from gold programs.

    Candidate relations always come from the normal retriever ranking. Gold is
    used only to verify trajectories and choose teacher actions; it is never
    inserted into a proposal set that failed to retrieve it.
    """

    def __init__(
        self,
        executor: ProgramExecutor,
        candidate_provider: CandidateProvider,
        max_active: int = 6,
        frontier_width: int = 3,
    ):
        if frontier_width < 2:
            raise ValueError("frontier_width must permit alternatives")
        if max_active < frontier_width * 2 - 1:
            raise ValueError("max_active must hold a frontier while another is explored")
        self.executor = executor
        self.candidate_provider = candidate_provider
        self.max_active = int(max_active)
        self.frontier_width = int(frontier_width)
        self.stats: Counter = Counter()

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
        joins = list(plan.joins)
        for position, join in enumerate(joins):
            next_join = joins[position + 1] if position + 1 < len(joins) else None
            demo = self._build_relation_demo(
                question_id, question, plan, join, next_join, gold_answers
            )
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
        next_join: Optional[ProgramStatement],
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        state_before = list(plan.executable_functions[: join.index])
        # A standalone SFT conversation cannot begin from a hidden gold prefix.
        # Longer paths are learned by repeating the verified two-step protocol
        # during RL rather than fabricating an unseen initial state.
        if any(_JOIN.match(_ASSIGNMENT.match(raw).group(2)) for raw in state_before):
            return None
        self.stats["relation_decisions"] += 1
        query = relation_hint(str(join.relation))
        options = list(self.candidate_provider(query, state_before, join))[
            : self.frontier_width
        ]
        gold_option = next((option for option in options if option.relation == join.relation), None)
        if gold_option is None:
            self.stats["proposal_miss"] += 1
            return None
        self.stats["proposal_hit"] += 1

        candidates: List[Tuple[RelationOption, ExecutedHypothesis, Tuple[str, ...]]] = []
        seen_denotations = set()
        for option in options:
            statement = replace_join_relation(join.raw, option.relation)
            prefix = state_before + [statement]
            prefix_values = normalize_values(self.executor(prefix, join.target))
            if not prefix_values:
                continue
            if prefix_values in seen_denotations:
                continue
            seen_denotations.add(prefix_values)
            full_program = list(plan.executable_functions)
            full_program[join.index] = statement
            terminal = normalize_values(self.executor(full_program, plan.target_expression))
            node = ExecutedHypothesis(
                f"H{len(candidates)}",
                tuple(prefix),
                join.target,
                prefix_values,
                relation=option.relation,
                role="gold" if option.relation == join.relation else "alternative",
            )
            candidates.append((option, node, terminal))
        gold_entry = next(
            (item for item in candidates if item[0].relation == join.relation), None
        )
        if gold_entry is None or len(candidates) < 2:
            return None

        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        created = tuple(hypotheses)
        source = _action_source(state_before, join.raw)
        steps = [
            DemonstrationStep(
                "Find_relation", (source, query), (), created,
                ("open_ranked_frontier",),
            )
        ]
        gold = gold_entry[1]

        terminal_gold = join.target == plan.target_expression and gold.denotation == gold_answers
        if terminal_gold:
            steps.extend(
                [
                    DemonstrationStep(
                        "Select", (gold.hypothesis_id,), created, (),
                        ("investigate_without_erasing_alternatives",),
                    ),
                    DemonstrationStep(
                        "Commit", (gold.hypothesis_id,), created, (),
                        ("complete", "executable"),
                    ),
                ]
            )
            family = "frontier_commit"
            probe_relation = None
        else:
            # A genuine delayed-recovery trace requires exactly one remaining
            # linear JOIN. This keeps every action replayable by the runtime.
            if not _is_immediate_linear_terminal(plan, join, next_join):
                return None
            wrong_entries = [
                item
                for item in candidates
                if item[0].relation != join.relation
                and item[2]
                and item[2] != gold_answers
            ]
            if not wrong_entries:
                return None
            wrong_entries.sort(key=lambda item: (item[0].rank, -item[0].score))
            wrong = wrong_entries[0][1]
            next_relation = relation_hint(str(next_join.relation))

            steps.append(
                DemonstrationStep(
                    "Select", (wrong.hypothesis_id,), created, (),
                    ("plausible_branch_requires_more_evidence",),
                )
            )
            active = list(created)
            wrong_children = self._expand_terminal_frontier(
                wrong, next_join, hypotheses
            )
            if not wrong_children or any(
                hypotheses[node_id].denotation == gold_answers for node_id in wrong_children
            ):
                return None
            active.remove(wrong.hypothesis_id)
            active.extend(wrong_children)
            steps.append(
                DemonstrationStep(
                    "Find_relation", (wrong.target_expression, next_relation), created,
                    tuple(wrong_children), ("test_plausible_alternative",),
                )
            )
            for child_id in wrong_children:
                before = tuple(active)
                active.remove(child_id)
                steps.append(
                    DemonstrationStep(
                        "Prune", (child_id,), before, (),
                        ("continuation_fails_full_intent",),
                    )
                )
            steps.append(
                DemonstrationStep(
                    "Select", (gold.hypothesis_id,), tuple(active), (),
                    ("return_to_preserved_alternative",),
                )
            )
            gold_children = self._expand_terminal_frontier(
                gold, next_join, hypotheses
            )
            final_gold = next(
                (
                    hypotheses[node_id]
                    for node_id in gold_children
                    if hypotheses[node_id].relation == next_join.relation
                    and hypotheses[node_id].denotation == gold_answers
                ),
                None,
            )
            if final_gold is None:
                return None
            before_gold_expansion = tuple(active)
            active.remove(gold.hypothesis_id)
            active.extend(gold_children)
            if len(active) > self.max_active:
                return None
            steps.extend(
                [
                    DemonstrationStep(
                        "Find_relation",
                        (gold.target_expression, next_relation),
                        before_gold_expansion,
                        tuple(gold_children),
                        ("continue_preserved_alternative",),
                    ),
                    DemonstrationStep(
                        "Select", (final_gold.hypothesis_id,), tuple(active), (),
                        ("full_intent_supported",),
                    ),
                    DemonstrationStep(
                        "Commit", (final_gold.hypothesis_id,), tuple(active), (),
                        ("complete", "executable"),
                    ),
                ]
            )
            family = "delayed_frontier_recovery"
            probe_relation = next_relation

        return HyperDemonstration(
            demo_id=f"{question_id}:join:{join.index}:{_digest(tuple(option.relation for option, _, _ in candidates))}",
            question_id=question_id,
            question=question,
            family=family,
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "gold_relation": join.relation,
                "gold_rank": gold_option.rank,
                "proposal_relations": [option.relation for option, _, _ in candidates],
                "proposal_recall_at_frontier": True,
                "probe_relation": probe_relation,
                "decision_index": join.index,
            },
        )

    def _expand_terminal_frontier(
        self,
        parent: ExecutedHypothesis,
        join: ProgramStatement,
        hypotheses: Dict[str, ExecutedHypothesis],
    ) -> List[str]:
        options = list(
            self.candidate_provider(
                relation_hint(str(join.relation)), parent.function_state, join
            )
        )[
            : self.frontier_width
        ]
        if not any(option.relation == join.relation for option in options):
            return []
        children: List[str] = []
        seen_denotations = set()
        for option in options:
            statement = replace_join_relation(join.raw, option.relation)
            state = list(parent.function_state) + [statement]
            values = normalize_values(self.executor(state, join.target))
            if not values:
                continue
            if values in seen_denotations:
                continue
            seen_denotations.add(values)
            node_id = f"H{len(hypotheses)}"
            hypotheses[node_id] = ExecutedHypothesis(
                node_id,
                tuple(state),
                join.target,
                values,
                relation=option.relation,
                role="continuation",
            )
            children.append(node_id)
        return children

    def _build_intersection_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        combine: ProgramStatement,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        self.stats["conjunction_decisions"] += 1
        left_target, right_target = combine.sources
        definitions = {statement.target: statement for statement in plan.statements}
        left_join = definitions.get(left_target)
        right_join = definitions.get(right_target)
        if (
            left_join is None
            or right_join is None
            or left_join.kind != "join"
            or right_join.kind != "join"
            or left_join.sources != right_join.sources
            or not left_join.sources
        ):
            return None
        shared_source = left_join.sources[0]
        base_state = self._dependency_state(plan, shared_source)
        action_source = _action_source(base_state, left_join.raw)
        query = (
            relation_hint(str(left_join.relation))
            + " and "
            + relation_hint(str(right_join.relation))
        )
        options = list(self.candidate_provider(query, base_state, left_join))[
            : self.frontier_width
        ]
        required_relations = {left_join.relation, right_join.relation}
        if not required_relations.issubset({option.relation for option in options}):
            self.stats["conjunction_proposal_miss"] += 1
            return None
        self.stats["conjunction_proposal_hit"] += 1

        hypotheses: Dict[str, ExecutedHypothesis] = {}
        relation_nodes: Dict[str, ExecutedHypothesis] = {}
        seen_denotations = set()
        for option in options:
            statement = replace_join_relation(left_join.raw, option.relation)
            state = base_state + [statement]
            values = normalize_values(self.executor(state, left_join.target))
            if not values:
                continue
            if values in seen_denotations:
                continue
            seen_denotations.add(values)
            node = ExecutedHypothesis(
                f"H{len(hypotheses)}", tuple(state), left_join.target, values,
                relation=option.relation,
                role=(
                    "required_branch"
                    if option.relation in required_relations
                    else "alternative"
                ),
            )
            hypotheses[node.hypothesis_id] = node
            relation_nodes[option.relation] = node
        if not required_relations.issubset(relation_nodes):
            return None

        left = relation_nodes[left_join.relation]
        right = relation_nodes[right_join.relation]
        combined_state, combined_target = combine_function_states(
            left.function_state, left.target_expression,
            right.function_state, right.target_expression,
        )
        combined_values = normalize_values(self.executor(combined_state, combined_target))
        if (
            combine.target != plan.target_expression
            or combined_values != gold_answers
            or combined_values != normalize_values(set(left.denotation) & set(right.denotation))
            or combined_values in {left.denotation, right.denotation}
        ):
            return None
        combined = ExecutedHypothesis(
            f"H{len(hypotheses)}", tuple(combined_state), combined_target,
            combined_values, role="combined",
        )
        hypotheses[combined.hypothesis_id] = combined
        created = tuple(node_id for node_id in hypotheses if node_id != combined.hypothesis_id)
        steps = [
            DemonstrationStep(
                "Find_relation", (action_source, query), (), created,
                ("open_shared_conjunction_frontier",),
            )
        ]
        active = list(created)
        for node_id in created:
            if node_id in {left.hypothesis_id, right.hypothesis_id}:
                continue
            before = tuple(active)
            active.remove(node_id)
            steps.append(
                DemonstrationStep(
                    "Prune", (node_id,), before, (),
                    ("not_a_required_conjunct",),
                )
            )
        steps.extend(
            [
                DemonstrationStep(
                    "Select", (left.hypothesis_id,), tuple(active), (),
                    ("required_branch",),
                ),
                DemonstrationStep(
                    "Select", (right.hypothesis_id,), tuple(active), (),
                    ("required_branch",),
                ),
                DemonstrationStep(
                    "Combine", (left.hypothesis_id, right.hypothesis_id),
                    tuple(active), created=(combined.hypothesis_id,),
                    rationale_facts=("both_branches_necessary",),
                ),
            ]
        )
        steps.append(
            DemonstrationStep(
                "Commit", (combined.hypothesis_id,), (combined.hypothesis_id,),
                rationale_facts=("complete", "executable"),
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

    def _dependency_state(self, plan: GoldPlan, target: str) -> List[str]:
        definitions = {statement.target: statement for statement in plan.statements}
        ordered: List[ProgramStatement] = []
        seen = set()

        def visit(expression: str) -> None:
            if expression in seen:
                return
            statement = definitions.get(expression)
            if statement is None or statement.kind == "stop":
                return
            for source in statement.sources:
                visit(source)
            seen.add(expression)
            ordered.append(statement)

        visit(target)
        return [statement.raw for statement in ordered]

class DemonstrationValidator:
    """Replay private executable states and verify graph-action consistency."""

    def __init__(self, executor: ProgramExecutor, max_active: int = 6):
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
        selected = None
        for step in demo.steps:
            hypothesis_arguments = (
                step.arguments if step.action in {"Prune", "Select", "Combine", "Commit"} else ()
            )
            unknown = [
                argument for argument in hypothesis_arguments
                if argument not in demo.hypotheses
            ]
            if unknown:
                errors.append(f"{step.action}: unknown hypotheses {unknown}")
                continue
            unknown_created = [
                node_id for node_id in step.created if node_id not in demo.hypotheses
            ]
            if unknown_created:
                errors.append(f"{step.action}: unknown created hypotheses {unknown_created}")
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
                selected = step.arguments[0]
            elif step.action == "Find_relation":
                if selected is not None:
                    if selected not in active:
                        errors.append(f"Find_relation expands inactive {selected}")
                    active.discard(selected)
                active.update(step.created)
                selected = None
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
                if len(step.created) != 1:
                    errors.append("Combine must create exactly one hypothesis")
                else:
                    active.add(step.created[0])
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
                    "function_state": list(node.function_state),
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


def _public_graph(demo: HyperDemonstration, active: Sequence[str]) -> str:
    lines = ["<hypothesis_graph>", f"active={len(active)}"]
    for node_id in active:
        node = demo.hypotheses[node_id]
        answers = ", ".join(node.denotation[:3]) or "empty"
        lines.append(
            f"{node_id} [active] via={node.relation or 'derived'} "
            f"answers={len(node.denotation)}: {answers}"
        )
    lines.append("</hypothesis_graph>")
    return "\n".join(lines)


def _action_text(step: DemonstrationStep) -> str:
    if step.action == "Find_relation":
        body = f"Find_relation [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "I will execute a bounded relation frontier and retain its alternatives."
    elif step.action == "Combine":
        body = f"Combine [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "Both active branches express required parts of the question."
    else:
        body = f"{step.action} [ {step.arguments[0]} ]"
        thoughts = {
            "Select": "I will investigate this hypothesis without discarding the others.",
            "Prune": "Execution has made this continuation unsupported.",
            "Commit": "This executable hypothesis covers the full question.",
        }
        thought = thoughts[step.action]
    return f"<think>{thought}</think>\n<action>{body}</action>"


def trajectory_sft_record(demo: HyperDemonstration) -> Dict[str, Any]:
    """Export one complete policy trajectory in the runtime's multi-turn format."""
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": append_hyper_instructions(
                f"Question: {demo.question}\nUse executable KG reasoning to answer it."
            ),
        }
    ]
    active: List[str] = list(demo.steps[0].visible_before if demo.steps else ())
    if active:
        messages.append(
            {
                "role": "tool",
                "content": "<information>Executable branches are available.\n"
                + _public_graph(demo, active)
                + "\n</information>",
            }
        )
    selected: Optional[str] = None
    for step in demo.steps:
        if tuple(active) != step.visible_before:
            raise ValueError(
                f"trajectory {demo.demo_id} has inconsistent visible state before {step.action}"
            )
        messages.append({"role": "assistant", "content": _action_text(step)})
        if step.action == "Find_relation":
            if selected is not None and selected in active:
                active.remove(selected)
            active.extend(step.created)
            selected = None
            event = "Executed the requested relation frontier."
        elif step.action == "Select":
            selected = step.arguments[0]
            event = f"Selected {selected}; other active hypotheses remain available."
        elif step.action == "Prune":
            active.remove(step.arguments[0])
            event = f"Pruned {step.arguments[0]}."
        elif step.action == "Combine":
            active.remove(step.arguments[0])
            active.remove(step.arguments[1])
            active.extend(step.created)
            selected = None
            event = f"Combined {step.arguments[0]} and {step.arguments[1]}."
        elif step.action == "Commit":
            active = [step.arguments[0]]
            selected = step.arguments[0]
            event = f"Committed {selected}."
        else:
            raise ValueError(f"unsupported action {step.action}")
        messages.append(
            {
                "role": "tool",
                "content": f"<information>{event}\n{_public_graph(demo, active)}\n</information>",
            }
        )
    committed = demo.hypotheses[selected] if selected else None
    if committed is None or committed.denotation != demo.gold_answers:
        raise ValueError(f"trajectory {demo.demo_id} did not finish on verified answers")
    messages.append(
        {
            "role": "assistant",
            "content": "<answer>" + ", ".join(committed.denotation) + "</answer>",
        }
    )
    return {
        "messages": messages,
        "data_source": "hyper_r1_verified_frontier",
        "extra_info": {
            "demo_id": demo.demo_id,
            "question_id": demo.question_id,
            "family": demo.family,
            "replay_verified": True,
            "gold_injected_into_proposals": False,
        },
    }
