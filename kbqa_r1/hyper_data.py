"""Verified demonstration construction for HyPER-R1.

Gold programs determine the executable trajectory.  A language model may later
verbalize a fixed action, but it is never allowed to choose relations, graph
actions, denotations, or the committed answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hyper_prompt import build_hyper_prompt
from .hyper_r1 import combine_function_states, relation_path, serialize_frontier
from .relation_paging import relation_page, serialize_relation_page_state


_EXPRESSION = r"expression\d*"
_ASSIGNMENT = re.compile(rf"^\s*({_EXPRESSION})\s*=\s*(.+?)\s*$")
_START = re.compile(r"^START\('(.+)'\)$")
_JOIN = re.compile(rf"^JOIN\('(.+)'\s*,\s*({_EXPRESSION}|'[^']+')\)$")
_AND = re.compile(rf"^AND\(({_EXPRESSION})\s*,\s*({_EXPRESSION})\)$")
_ARG = re.compile(
    rf"^ARG\('([^']+)'\s*,\s*({_EXPRESSION})\s*,\s*'([^']+)'\)$"
)
_CMP = re.compile(
    rf"^CMP\('([^']+)'\s*,\s*'([^']+)'\s*,\s*({_EXPRESSION})\)$"
)
_TC = re.compile(
    rf"^TC\(({_EXPRESSION})\s*,\s*'([^']+)'\s*,\s*'([^']+)'\)$"
)
_COUNT = re.compile(rf"^COUNT\(({_EXPRESSION})\)$")
_STOP = re.compile(rf"^STOP\(({_EXPRESSION})\)$")
_ONTOLOGY_TYPE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
)


def _is_ontology_type(value: str) -> bool:
    return bool(_ONTOLOGY_TYPE.fullmatch(str(value).strip()))


def normalize_values(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values or () if str(value).strip()}))


def normalize_gold_answers(values: Iterable[Any]) -> Tuple[str, ...]:
    """Extract answer identities from official GrailQA answer records."""
    normalized = []
    for value in values or ():
        if isinstance(value, Mapping):
            value = value.get("answer_argument")
        if value is not None and str(value).strip():
            normalized.append(value)
    return normalize_values(normalized)


def answer_set_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    predicted_set = set(normalize_values(predicted))
    gold_set = set(normalize_values(gold))
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


@dataclass(frozen=True)
class ProgramStatement:
    index: int
    target: str
    kind: str
    sources: Tuple[str, ...]
    relation: Optional[str]
    raw: str
    arguments: Tuple[str, ...] = ()


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

    @property
    def operators(self) -> Tuple[ProgramStatement, ...]:
        return tuple(
            statement
            for statement in self.statements
            if statement.kind in {"order", "compare", "time_constraint", "count"}
        )


class IneligibleProgram(ValueError):
    pass


class ProgramExecutionError(RuntimeError):
    """The candidate did not produce a valid executable KG observation."""


def compile_gold_plan(function_list: Sequence[str]) -> GoldPlan:
    """Compile the released runtime grammar into canonical expression names."""
    raw_functions = [str(value).strip() for value in function_list]
    has_bare_expression = any(
        re.search(r"\bexpression\b", raw) for raw in raw_functions
    )
    statements: List[ProgramStatement] = []
    executable: List[str] = []
    defined = set()
    expression_names: Dict[str, str] = (
        {"expression": "expression1"} if has_bare_expression else {}
    )
    final_target: Optional[str] = None
    stopped = False

    def expression_name(source_name: str) -> str:
        if not has_bare_expression:
            return source_name
        if source_name not in expression_names:
            expression_names[source_name] = f"expression{len(expression_names) + 1}"
        return expression_names[source_name]

    for index, raw in enumerate(raw_functions):
        if stopped:
            raise IneligibleProgram("STOP must be the terminal statement")
        match = _ASSIGNMENT.match(raw)
        if not match:
            raise IneligibleProgram(f"invalid function statement at {index}: {raw}")
        source_target, rhs = match.groups()
        target = expression_name(source_target)

        start = _START.match(rhs)
        join = _JOIN.match(rhs)
        combine = _AND.match(rhs)
        order = _ARG.match(rhs)
        compare = _CMP.match(rhs)
        time_constraint = _TC.match(rhs)
        count = _COUNT.match(rhs)
        stop = _STOP.match(rhs)
        if start:
            entity = start.group(1)
            canonical_raw = f"{target} = START('{entity}')"
            statement = ProgramStatement(
                index, target, "start", (), None, canonical_raw, (entity,)
            )
        elif join:
            relation, source = join.groups()
            if source.startswith("expression"):
                if source not in defined:
                    raise IneligibleProgram(
                        f"statement {index} references undefined {[source]}"
                    )
                canonical_source = expression_name(source)
                sources = (canonical_source,)
            else:
                canonical_source = source
                sources = ()
            canonical_raw = (
                f"{target} = JOIN('{relation}', {canonical_source})"
            )
            statement = ProgramStatement(
                index,
                target,
                "join",
                sources,
                relation,
                canonical_raw,
                (relation, canonical_source),
            )
        elif combine:
            source_names = combine.groups()
            missing = [source for source in source_names if source not in defined]
            if missing:
                raise IneligibleProgram(
                    f"statement {index} references undefined {missing}"
                )
            sources = tuple(expression_name(source) for source in source_names)
            canonical_raw = f"{target} = AND({sources[0]}, {sources[1]})"
            statement = ProgramStatement(
                index, target, "and", sources, None, canonical_raw, sources
            )
        elif order:
            mode, source, relation = order.groups()
            if source not in defined:
                raise IneligibleProgram(
                    f"statement {index} references undefined {[source]}"
                )
            canonical_source = expression_name(source)
            canonical_raw = (
                f"{target} = ARG('{mode}', {canonical_source}, '{relation}')"
            )
            statement = ProgramStatement(
                index,
                target,
                "order",
                (canonical_source,),
                relation,
                canonical_raw,
                (mode, canonical_source, relation),
            )
        elif compare:
            mode, relation, source = compare.groups()
            if source not in defined:
                raise IneligibleProgram(
                    f"statement {index} references undefined {[source]}"
                )
            canonical_source = expression_name(source)
            canonical_raw = (
                f"{target} = CMP('{mode}', '{relation}', {canonical_source})"
            )
            statement = ProgramStatement(
                index,
                target,
                "compare",
                (canonical_source,),
                relation,
                canonical_raw,
                (mode, relation, canonical_source),
            )
        elif time_constraint:
            source, relation, time = time_constraint.groups()
            if source not in defined:
                raise IneligibleProgram(
                    f"statement {index} references undefined {[source]}"
                )
            canonical_source = expression_name(source)
            canonical_raw = (
                f"{target} = TC({canonical_source}, '{relation}', '{time}')"
            )
            statement = ProgramStatement(
                index,
                target,
                "time_constraint",
                (canonical_source,),
                relation,
                canonical_raw,
                (relation, time),
            )
        elif count:
            source = count.group(1)
            if source not in defined:
                raise IneligibleProgram(
                    f"statement {index} references undefined {[source]}"
                )
            canonical_source = expression_name(source)
            canonical_raw = f"{target} = COUNT({canonical_source})"
            statement = ProgramStatement(
                index,
                target,
                "count",
                (canonical_source,),
                None,
                canonical_raw,
                (canonical_source,),
            )
        elif stop:
            source = stop.group(1)
            if source not in defined:
                raise IneligibleProgram(f"STOP references undefined {source}")
            canonical_source = expression_name(source)
            final_target = canonical_source
            canonical_raw = f"{target} = STOP({canonical_source})"
            statements.append(
                ProgramStatement(
                    index, target, "stop", (canonical_source,), None, canonical_raw
                )
            )
            stopped = True
            continue
        else:
            operator = rhs.split("(", 1)[0]
            raise IneligibleProgram(f"unsupported operator {operator or rhs}")

        statements.append(statement)
        executable.append(statement.raw)
        defined.add(source_target)
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
class ExecutedRelationPage:
    node_ids: Tuple[str, ...]
    start: int
    stop: int
    total: int


@dataclass(frozen=True)
class ExecutedHypothesis:
    hypothesis_id: str
    function_state: Tuple[str, ...]
    target_expression: str
    denotation: Tuple[str, ...]
    denotation_labels: Tuple[Tuple[str, str], ...] = ()
    relation: Optional[str] = None
    role: str = "gold"
    parent_id: Optional[str] = None
    parent_ids: Tuple[str, ...] = ()
    operation: str = "expand"
    depth: int = 0
    provenance: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DemonstrationStep:
    action: str
    arguments: Tuple[str, ...]
    visible_before: Tuple[str, ...]
    created: Tuple[str, ...] = ()
    rationale_facts: Tuple[str, ...] = ()
    relation_page: Tuple[int, int, int] = ()


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
EntityDisplayProvider = Callable[[Sequence[str]], Mapping[str, str]]


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


def _is_immediate_linear_terminal(
    plan: GoldPlan,
    current: ProgramStatement,
    following: Optional[ProgramStatement],
) -> bool:
    """Return whether one more JOIN completes this linear gold branch."""
    if (
        following is None
        or following.index != current.index + 1
        or following.sources != (current.target,)
    ):
        return False
    # GrailQA commonly appends START(answer_type) + AND after the semantic
    # path. The child execution still has to equal the annotated answers, so
    # allowing that non-relational tail does not weaken trajectory validation.
    return not any(
        statement.kind == "join" and statement.index > following.index
        for statement in plan.statements
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
        max_active: int = 24,
        max_nodes: int = 24,
        frontier_width: int = 6,
        max_turns: int = 16,
        entity_display_provider: Optional[EntityDisplayProvider] = None,
    ):
        if frontier_width < 2:
            raise ValueError("frontier_width must permit alternatives")
        if max_active < frontier_width * 2:
            raise ValueError("max_active must hold two complete relation frontiers")
        if max_nodes < max_active:
            raise ValueError("max_nodes must be at least max_active")
        self.executor = executor
        self.candidate_provider = candidate_provider
        self.max_active = int(max_active)
        self.max_nodes = int(max_nodes)
        self.frontier_width = int(frontier_width)
        self.max_turns = int(max_turns)
        self.entity_display_provider = entity_display_provider
        self._display_cache: Dict[str, str] = {}
        if self.max_turns < 2:
            raise ValueError("max_turns must include at least one action and the answer")
        self.stats: Counter = Counter()
        self._execution_cache: Dict[
            Tuple[Tuple[str, ...], str], Optional[Tuple[str, ...]]
        ] = {}

    def _resolve_display_labels(self, values: Sequence[str]) -> None:
        identities = tuple(str(value) for value in values)
        missing = [value for value in identities if value not in self._display_cache]
        if missing and self.entity_display_provider is not None:
            resolved = self.entity_display_provider(missing)
            for identity, label in resolved.items():
                key = str(identity).strip()
                value = str(label).strip()
                if key and value and key != value:
                    self._display_cache[key] = value

    def _display_pairs(self, values: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
        visible_values = tuple(str(value) for value in values[:4])
        self._resolve_display_labels(visible_values)
        return tuple(
            (value, self._display_cache[value])
            for value in visible_values
            if value in self._display_cache
        )

    def _execute(
        self, functions: Sequence[str], target: str
    ) -> Optional[Tuple[str, ...]]:
        """Distinguish a valid empty answer set from execution failure."""
        key = (tuple(str(item) for item in functions), str(target))
        if key in self._execution_cache:
            self.stats["execution_cache_hit"] += 1
            return self._execution_cache[key]
        try:
            result = normalize_values(self.executor(functions, target))
        except ProgramExecutionError:
            self.stats["execution_failure"] += 1
            result = None
        self._execution_cache[key] = result
        self.stats["execution_cache_miss"] += 1
        return result

    def _pages_through_required_relation(
        self,
        options: Sequence[RelationOption],
        required_relation: str,
    ) -> Tuple[Tuple[Tuple[RelationOption, ...], ...], Optional[RelationOption]]:
        """Expose complete pages until the required relation becomes visible.

        Gold selects how many pages a teacher demonstration needs, but never
        changes their contents or ordering. At inference the policy obtains the
        same pages through repeated Widen actions.
        """
        pages, required = self._pages_through_required_relations(
            options, (required_relation,)
        )
        return pages, required.get(required_relation)

    def _pages_through_required_relations(
        self,
        options: Sequence[RelationOption],
        required_relations: Sequence[str],
    ) -> Tuple[
        Tuple[Tuple[RelationOption, ...], ...], Dict[str, RelationOption]
    ]:
        """Return stable pages through the last of several required relations."""
        ranked = tuple(options)
        positions: Dict[str, int] = {}
        required: Dict[str, RelationOption] = {}
        wanted = {str(relation) for relation in required_relations}
        for index, option in enumerate(ranked):
            if option.relation in wanted and option.relation not in positions:
                positions[option.relation] = index
                required[option.relation] = option
        if set(positions) != wanted:
            return (), {}

        last_page = max(positions.values()) // self.frontier_width
        pages = []
        for page_index in range(last_page + 1):
            page = relation_page(
                ranked,
                offset=page_index * self.frontier_width,
                page_size=self.frontier_width,
            )
            if page.items:
                pages.append(page.items)
        return tuple(pages), required

    def build(self, row: Mapping[str, Any]) -> List[HyperDemonstration]:
        question = str(row.get("question") or row.get("original_question") or "").strip()
        question_id = str(row.get("ID") or row.get("id") or _digest(question))
        plan = compile_gold_plan(row.get("function_list") or ())
        annotated = row.get("answer")
        if annotated is None:
            annotated = row.get("answers")
        if annotated is None:
            annotated = row.get("target")
        gold_answers = normalize_gold_answers(() if annotated is None else annotated)
        if not gold_answers:
            return []
        if len(gold_answers) > 100:
            self.stats["large_answer_row_skipped"] += 1
            return []
        executed_gold = self._execute(plan.executable_functions, plan.target_expression)
        if not executed_gold:
            return []
        if gold_answers and executed_gold != gold_answers:
            return []
        intersection_demos: List[HyperDemonstration] = []
        semantic_branch_indexes = set()
        if plan.operators or self._has_bare_type_intersection(plan):
            operator_demo = self._build_operator_program_demo(
                question_id, question, plan, gold_answers
            )
            demos = [operator_demo] if operator_demo is not None else []
        else:
            for combine in plan.intersections:
                semantic_branch_indexes.update(
                    self._semantic_intersection_join_indexes(plan, combine)
                )
                demo = self._build_intersection_demo(
                    question_id, question, plan, combine, gold_answers
                )
                if demo is None:
                    continue
                intersection_demos.append(demo)

            demos = []
            joins = list(plan.joins)
            for position, join in enumerate(joins):
                if join.index in semantic_branch_indexes:
                    continue
                next_join = joins[position + 1] if position + 1 < len(joins) else None
                demo = self._build_relation_demo(
                    question_id,
                    question,
                    plan,
                    join,
                    next_join,
                    gold_answers,
                    following_joins=tuple(joins[position + 1 :]),
                )
                if demo is not None:
                    demos.append(demo)
            demos.extend(intersection_demos)
        within_budget = []
        for demo in demos:
            if len(demo.hypotheses) > self.max_nodes:
                self.stats["trajectory_node_budget_miss"] += 1
                continue
            # Every graph action occupies one model turn and the committed
            # answer occupies a final turn of its own.
            if len(demo.steps) + 1 > self.max_turns:
                self.stats["trajectory_turn_budget_miss"] += 1
                continue
            within_budget.append(demo)
        demos = within_budget
        candidate_entities = self._row_candidate_entities(row, plan, question_id)
        base_prompt = self._row_base_prompt(row)
        for demo in demos:
            demo.private_metadata["candidate_entities"] = candidate_entities
            demo.private_metadata["candidate_entity_order"] = "stable_question_hash"
            demo.private_metadata["base_prompt"] = base_prompt
            demo.private_metadata["max_active"] = self.max_active
            demo.private_metadata["max_nodes"] = self.max_nodes
            demo.private_metadata["relation_page_size"] = self.frontier_width
            demo.private_metadata["relation_rank_cutoff"] = None
            demo.private_metadata["max_turns"] = self.max_turns
        return demos

    def _candidate_future_value(
        self,
        plan: GoldPlan,
        join: ProgramStatement,
        option: RelationOption,
        gold_answers: Tuple[str, ...],
    ) -> Dict[str, Any]:
        """Private teacher-only outcome for one naturally retrieved action."""
        program = list(plan.executable_functions)
        program[join.index] = replace_join_relation(join.raw, option.relation)
        answers = self._execute(program, plan.target_expression)
        return {
            "execution_success": answers is not None,
            "answer_count": len(answers or ()),
            "answer_exact": answers == gold_answers,
            "answer_f1": answer_set_f1(answers or (), gold_answers),
        }

    def _build_operator_program_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        """Replay one complete operator-bearing program through public actions."""
        hypotheses: Dict[str, ExecutedHypothesis] = {}
        steps: List[DemonstrationStep] = []
        active: List[str] = []
        expression_nodes: Dict[str, ExecutedHypothesis] = {}
        start_states: Dict[str, Tuple[str, str]] = {}
        widen_sources: List[str] = []
        proposal_relations: List[str] = []

        def add_node(
            state: Sequence[str],
            target: str,
            values: Tuple[str, ...],
            *,
            parent: Optional[ExecutedHypothesis] = None,
            parents: Sequence[ExecutedHypothesis] = (),
            relation: Optional[str] = None,
            role: str = "operator_progress",
            operation: str,
            provenance: Sequence[str] = (),
        ) -> Optional[ExecutedHypothesis]:
            if len(hypotheses) >= self.max_nodes:
                self.stats["operator_program_node_budget_miss"] += 1
                return None
            node_id = f"H{len(hypotheses)}"
            node = ExecutedHypothesis(
                node_id,
                tuple(state),
                target,
                values,
                denotation_labels=self._display_pairs(values),
                relation=relation,
                role=role,
                parent_id=parent.hypothesis_id if parent is not None else None,
                parent_ids=tuple(item.hypothesis_id for item in parents),
                operation=operation,
                depth=(max((item.depth for item in parents), default=-1) + 1)
                if parents
                else (parent.depth + 1 if parent is not None else 0),
                provenance=tuple(provenance),
            )
            hypotheses[node_id] = node
            return node

        def select_parent(parent: ExecutedHypothesis, rationale: str) -> bool:
            if parent.hypothesis_id not in active or not parent.denotation:
                return False
            steps.append(
                DemonstrationStep(
                    "Select",
                    (parent.hypothesis_id,),
                    tuple(active),
                    (),
                    (rationale,),
                )
            )
            return True

        for statement in plan.statements:
            if statement.kind == "start":
                start_states[statement.target] = (
                    statement.raw,
                    statement.arguments[0],
                )
                # START is an assignment and may deliberately reuse an old
                # expression name. Do not mistake the overwritten branch for
                # the new constant during a later AND.
                expression_nodes.pop(statement.target, None)
                continue
            if statement.kind == "stop":
                continue

            if statement.kind == "join":
                source_expression = statement.sources[0] if statement.sources else None
                parent = expression_nodes.get(source_expression or "")
                if parent is not None:
                    if not select_parent(parent, "continue_required_program_branch"):
                        return None
                    state_before = list(parent.function_state)
                    action_source = parent.target_expression
                else:
                    if source_expression is not None:
                        start = start_states.get(source_expression)
                        if start is None:
                            return None
                        state_before = [start[0]]
                    else:
                        state_before = []
                    action_source = _action_source(state_before, statement.raw)

                ranked = list(
                    self.candidate_provider(question.strip(), state_before, statement)
                )
                pages, required = self._pages_through_required_relation(
                    ranked, statement.relation or ""
                )
                self.stats["operator_program_relation_decisions"] += 1
                if required is None:
                    self.stats["operator_program_proposal_miss"] += 1
                    return None
                if sum(len(page) for page in pages) > self.max_active:
                    self.stats["operator_program_within_budget_miss"] += 1
                    return None
                self.stats["operator_program_within_budget_hit"] += 1
                if len(pages) == 1:
                    self.stats["operator_program_proposal_hit"] += 1
                else:
                    self.stats["operator_program_proposal_miss"] += 1
                    self.stats["operator_program_recovered_by_widen"] += 1
                if required.rank == 1:
                    self.stats["operator_program_top1_hit"] += 1
                proposal_relations.extend(option.relation for option in ranked)

                created_pages: List[Tuple[str, ...]] = []
                required_node: Optional[ExecutedHypothesis] = None
                for page in pages:
                    page_nodes = []
                    for option in page:
                        raw = replace_join_relation(statement.raw, option.relation)
                        state = [*state_before, raw]
                        values = self._execute(state, statement.target)
                        if values is None:
                            continue
                        node = add_node(
                            state,
                            statement.target,
                            values,
                            parent=parent,
                            relation=option.relation,
                            role=(
                                "required_program_branch"
                                if option.relation == statement.relation
                                else "alternative"
                            ),
                            operation="expand",
                            provenance=(
                                "policy_choice"
                                if option.rank == 1
                                else "ranked_alternative",
                            ),
                        )
                        if node is None:
                            return None
                        page_nodes.append(node.hypothesis_id)
                        if option.relation == statement.relation:
                            required_node = node
                    created_pages.append(tuple(page_nodes))

                if (
                    required_node is None
                    or not required_node.denotation
                    or not created_pages
                    or len(created_pages[0]) < 2
                ):
                    return None
                before = tuple(active)
                if parent is not None:
                    active.remove(parent.hypothesis_id)
                if len(active) + sum(len(page) for page in created_pages) > self.max_active:
                    self.stats["operator_program_active_budget_miss"] += 1
                    return None
                active.extend(created_pages[0])
                first_bounds = relation_page(
                    ranked, offset=0, page_size=self.frontier_width
                )
                steps.append(
                    DemonstrationStep(
                        "Find_relation",
                        (action_source,),
                        before,
                        created_pages[0],
                        ("open_operator_program_frontier",),
                        (first_bounds.start, first_bounds.stop, first_bounds.total),
                    )
                )
                for page_index, created in enumerate(created_pages[1:], start=1):
                    bounds = relation_page(
                        ranked,
                        offset=page_index * self.frontier_width,
                        page_size=self.frontier_width,
                    )
                    steps.append(
                        DemonstrationStep(
                            "Widen",
                            (action_source,),
                            tuple(active),
                            created,
                            (
                                "required_relation_missing_from_visible_pages",
                                f"relation_page:{page_index + 1}",
                            ),
                            (bounds.start, bounds.stop, bounds.total),
                        )
                    )
                    active.extend(created)
                    widen_sources.append(action_source)
                expression_nodes[statement.target] = required_node
                continue

            if statement.kind == "and":
                left = expression_nodes.get(statement.sources[0])
                right = expression_nodes.get(statement.sources[1])
                if (left is None) != (right is None):
                    missing_source = (
                        statement.sources[0] if left is None else statement.sources[1]
                    )
                    preserved = right if left is None else left
                    bare_start = start_states.get(missing_source)
                    if preserved is None or bare_start is None:
                        self.stats["operator_program_unrepresented_intersection"] += 1
                        return None
                    ontology_type = bare_start[1]
                    if not _is_ontology_type(ontology_type):
                        self.stats["operator_program_unrepresented_intersection"] += 1
                        return None
                    if not select_parent(
                        preserved, "apply_required_ontology_type_constraint"
                    ):
                        return None

                    # Match the released Merge runtime exactly: create a START
                    # for the inferred type, then execute the AND in the same
                    # query. This avoids enumerating every instance of broad
                    # classes such as people.person.
                    expression_numbers = [
                        int(number)
                        for raw in preserved.function_state
                        for number in re.findall(r"expression(\d+)", raw)
                    ]
                    type_expression = f"expression{max(expression_numbers, default=0) + 1}"
                    combined_target = preserved.target_expression
                    combined_state = [
                        *preserved.function_state,
                        f"{type_expression} = START('{ontology_type}')",
                        f"{combined_target} = AND({combined_target}, {type_expression})",
                    ]
                    combined_values = self._execute(combined_state, combined_target)
                    if combined_values is None:
                        return None
                    filtered = add_node(
                        combined_state,
                        combined_target,
                        combined_values,
                        parent=preserved,
                        role="type_constrained",
                        operation="merge",
                        provenance=(f"ontology_type:{ontology_type}",),
                    )
                    if filtered is None:
                        return None
                    before = tuple(active)
                    active.remove(preserved.hypothesis_id)
                    active.append(filtered.hypothesis_id)
                    steps.append(
                        DemonstrationStep(
                            "Merge",
                            (preserved.target_expression, ontology_type),
                            before,
                            (filtered.hypothesis_id,),
                            ("explicit_gold_type_constraint",),
                        )
                    )
                    self.stats["operator_program_type_constraint"] += 1
                    expression_nodes[statement.target] = filtered
                    continue
                if (
                    left is None
                    or right is None
                    or left.hypothesis_id not in active
                    or right.hypothesis_id not in active
                    or left.hypothesis_id == right.hypothesis_id
                ):
                    return None
                combined_state, combined_target = combine_function_states(
                    left.function_state,
                    left.target_expression,
                    right.function_state,
                    right.target_expression,
                )
                values = self._execute(combined_state, combined_target)
                if values is None:
                    return None
                combined = add_node(
                    combined_state,
                    combined_target,
                    values,
                    parents=(left, right),
                    role="combined",
                    operation="combine",
                    provenance=(f"combined_with:{right.hypothesis_id}",),
                )
                if combined is None:
                    return None
                steps.append(
                    DemonstrationStep(
                        "Combine",
                        (left.hypothesis_id, right.hypothesis_id),
                        tuple(active),
                        (combined.hypothesis_id,),
                        ("both_branches_required_by_gold_program",),
                    )
                )
                active.remove(left.hypothesis_id)
                active.remove(right.hypothesis_id)
                active.append(combined.hypothesis_id)
                expression_nodes[statement.target] = combined
                continue

            source = statement.sources[0]
            parent = expression_nodes.get(source)
            root_operator = parent is None and statement.kind in {"compare", "order"}
            if root_operator:
                start = start_states.get(source)
                if start is None:
                    return None
                if statement.kind == "order" and (
                    start[1].startswith("m.") or start[1].startswith("g.")
                ):
                    return None
                state_before = [start[0]]
                target = source
            else:
                if parent is None or not select_parent(
                    parent, "apply_required_logical_operator"
                ):
                    return None
                state_before = list(parent.function_state)
                target = parent.target_expression

            if statement.kind == "order":
                mode, _, relation = statement.arguments
                runtime_raw = f"{target} = ARG('{mode}', {target}, '{relation}')"
                action = "Order"
                action_arguments = (mode, target if parent is not None else start[1], relation)
            elif statement.kind == "compare":
                mode, relation, _ = statement.arguments
                runtime_raw = f"{target} = CMP('{mode}', '{relation}', {target})"
                action = "Compare"
                action_arguments = (mode, relation, start_states[source][1])
            elif statement.kind == "time_constraint":
                relation, time = statement.arguments
                runtime_raw = f"{target} = TC({target}, '{relation}', '{time}')"
                action = "Time_constraint"
                action_arguments = (relation, time)
            elif statement.kind == "count":
                runtime_raw = f"{target} = COUNT({target})"
                action = "Count"
                action_arguments = (target,)
            else:
                return None

            state = [*state_before, runtime_raw]
            values = self._execute(state, target)
            if values is None:
                return None
            node = add_node(
                state,
                target,
                values,
                parent=parent,
                operation=action.lower(),
                provenance=("gold_program_operator",),
            )
            if node is None:
                return None
            before = tuple(active)
            if parent is not None:
                active.remove(parent.hypothesis_id)
            if len(active) >= self.max_active:
                return None
            active.append(node.hypothesis_id)
            steps.append(
                DemonstrationStep(
                    action,
                    action_arguments,
                    before,
                    (node.hypothesis_id,),
                    (f"gold_program_operator:{statement.kind}",),
                )
            )
            expression_nodes[statement.target] = node
            expression_nodes[source] = node

        final = expression_nodes.get(plan.target_expression)
        if final is None or final.denotation != gold_answers:
            self.stats["operator_program_terminal_mismatch"] += 1
            return None
        steps.append(
            DemonstrationStep(
                "Commit",
                (final.hypothesis_id,),
                tuple(active),
                (),
                ("complete", "executable"),
            )
        )
        self.stats["operator_program_built"] += 1
        return HyperDemonstration(
            demo_id=f"{question_id}:operator:{_digest(plan.executable_functions)}",
            question_id=question_id,
            question=question,
            family="operator_program" if plan.operators else "typed_program",
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "retrieval_intent_source": "question",
                "logical_operators": [statement.kind for statement in plan.operators],
                "proposal_relations": proposal_relations,
                "proposal_recall_within_budget": True,
                "widen_sources": widen_sources,
                "path_hops": len(plan.joins),
            },
        )

    def _build_relation_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        join: ProgramStatement,
        next_join: Optional[ProgramStatement],
        gold_answers: Tuple[str, ...],
        following_joins: Sequence[ProgramStatement] = (),
    ) -> Optional[HyperDemonstration]:
        source = join.sources[0] if join.sources else _join_source(join.raw)
        state_before = self._dependency_state(plan, source, before=join.index)
        # A standalone SFT conversation cannot begin from a hidden gold prefix.
        # Deep demonstrations therefore start at the first relation and replay
        # every later frontier through public executable observations.
        if any(_JOIN.match(_ASSIGNMENT.match(raw).group(2)) for raw in state_before):
            return None
        self.stats["relation_decisions"] += 1
        # Gold labels may judge the resulting frontier, but must not shape the
        # semantic request that the student has to reproduce at inference.
        query = question.strip()
        ranked_options = list(self.candidate_provider(query, state_before, join))
        pages, gold_option = self._pages_through_required_relation(
            ranked_options, join.relation
        )
        if gold_option is None:
            self.stats["proposal_structural_or_ranking_miss"] += 1
            self.stats["proposal_miss"] += 1
            return None
        if sum(len(page) for page in pages) > self.max_active:
            self.stats["proposal_within_budget_miss"] += 1
            return None
        self.stats["proposal_within_budget_hit"] += 1
        initial_options = pages[0]
        initial_relations = {option.relation for option in initial_options}
        needs_widen = len(pages) > 1
        if needs_widen:
            self.stats["proposal_miss"] += 1
            self.stats["proposal_recovered_by_widen"] += 1
            options = tuple(option for page in pages for option in page)
        else:
            self.stats["proposal_hit"] += 1
            options = initial_options
        self.stats[f"proposal_gold_rank_{gold_option.rank}"] += 1
        if gold_option.rank == 1:
            self.stats["proposal_top1_hit"] += 1

        candidates: List[Tuple[RelationOption, ExecutedHypothesis]] = []
        for option in options:
            statement = replace_join_relation(join.raw, option.relation)
            prefix = state_before + [statement]
            prefix_values = self._execute(prefix, join.target)
            if prefix_values is None:
                continue
            self.stats["candidate_execution_success"] += 1
            self.stats[
                "candidate_nonempty" if prefix_values else "candidate_empty"
            ] += 1
            node = ExecutedHypothesis(
                f"H{len(candidates)}",
                tuple(prefix),
                join.target,
                prefix_values,
                denotation_labels=self._display_pairs(prefix_values),
                relation=option.relation,
                role="gold" if option.relation == join.relation else "alternative",
                provenance=(
                    "policy_choice" if option.rank == 1 else "ranked_alternative",
                ),
            )
            candidates.append((option, node))
        gold_entry = next(
            (item for item in candidates if item[0].relation == join.relation), None
        )
        node_by_relation = {
            option.relation: node.hypothesis_id for option, node in candidates
        }
        created_pages = tuple(
            tuple(
                node_by_relation[option.relation]
                for option in page
                if option.relation in node_by_relation
            )
            for page in pages
        )
        initial_created = created_pages[0]
        widened_created = tuple(
            node_id for page in created_pages[1:] for node_id in page
        )
        if (
            gold_entry is None
            or len(initial_created) < 2
            or (needs_widen and gold_entry[1].hypothesis_id not in widened_created)
        ):
            return None

        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        created = tuple(node_id for page in created_pages for node_id in page)
        source = _action_source(state_before, join.raw)
        steps = [
            DemonstrationStep(
                "Find_relation", (source,), (), initial_created,
                ("open_ranked_frontier",),
                (0, len(pages[0]), len(ranked_options)),
            )
        ]
        visible = list(initial_created)
        for page_number, page_created in enumerate(created_pages[1:], start=2):
            steps.append(
                DemonstrationStep(
                    "Widen",
                    (source,),
                    tuple(visible),
                    page_created,
                    (
                        "required_relation_missing_from_visible_pages",
                        f"relation_page:{page_number}",
                    ),
                    (
                        (page_number - 1) * self.frontier_width,
                        min(page_number * self.frontier_width, len(ranked_options)),
                        len(ranked_options),
                    ),
                )
            )
            visible.extend(page_created)
        gold = gold_entry[1]
        candidate_future_values = {
            option.relation: self._candidate_future_value(
                plan, join, option, gold_answers
            )
            for option, _ in candidates
        }
        best_alternative_score = max(
            (
                option.score
                for option, _ in candidates
                if option.relation != join.relation
            ),
            default=None,
        )

        terminal_gold = join.target == plan.target_expression and gold.denotation == gold_answers
        if terminal_gold:
            steps.extend(
                [
                    DemonstrationStep(
                        "Commit", (gold.hypothesis_id,), created, (),
                        ("complete", "executable"),
                    ),
                ]
            )
            family = "adaptive_frontier_widen" if needs_widen else "frontier_commit"
            probe_relation = None
        else:
            if len(following_joins) > 1:
                return self._build_deep_progress_demo(
                    question_id=question_id,
                    question=question,
                    join=join,
                    following_joins=following_joins,
                    candidates=candidates,
                    gold_answers=gold_answers,
                    gold_option=gold_option,
                    best_alternative_score=best_alternative_score,
                    initial_steps=steps,
                    needs_initial_widen=needs_widen,
                    candidate_future_values=candidate_future_values,
                )
            if not _is_immediate_linear_terminal(plan, join, next_join):
                return None
            if needs_widen:
                gold_page = self._expand_terminal_frontier(
                    gold,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="continuation",
                )
                if gold_page is None:
                    return None
                gold_children = list(gold_page.node_ids)
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
                active = list(created)
                required_prunes = max(
                    0,
                    len(active) - 1 + len(gold_children) - self.max_active,
                )
                if required_prunes:
                    self.stats["trajectory_node_budget_miss"] += 1
                    return None
                steps.append(
                    DemonstrationStep(
                        "Select",
                        (gold.hypothesis_id,),
                        tuple(active),
                        (),
                        (f"question_relation_match:{gold.relation}",),
                    )
                )
                before_expansion = tuple(active)
                active.remove(gold.hypothesis_id)
                active.extend(gold_children)
                if len(active) > self.max_active:
                    return None
                steps.extend(
                    [
                        DemonstrationStep(
                            "Find_relation",
                            (gold.target_expression,),
                            before_expansion,
                            tuple(gold_children),
                            ("continue_widened_branch",),
                            (gold_page.start, gold_page.stop, gold_page.total),
                        ),
                        DemonstrationStep(
                            "Commit",
                            (final_gold.hypothesis_id,),
                            tuple(active),
                            (),
                            ("complete", "executable"),
                        ),
                    ]
                )
                family = "adaptive_frontier_widen"
                probe_relation = None
            else:
                # Recovery candidates are higher-ranked natural proposals whose
                # complete counterfactual program is not answer-equivalent.
                wrong_entries = []
                for option, node in candidates:
                    future = candidate_future_values[option.relation]
                    if (
                        option.relation == join.relation
                        or option.rank >= gold_option.rank
                        or not node.denotation
                        or not future["execution_success"]
                        or future["answer_exact"]
                    ):
                        continue
                    wrong_entries.append((option, node))
                if not wrong_entries:
                    direct = self._build_direct_progress_demo(
                        question_id,
                        question,
                        join,
                        next_join,
                        candidates,
                        gold_answers,
                        gold_option,
                        best_alternative_score,
                        len(ranked_options),
                    )
                    if direct is not None:
                        direct.private_metadata["candidate_future_values"] = candidate_future_values
                    return direct
                self.stats["recovery_opportunity"] += 1
                wrong_entries.sort(key=lambda item: (item[0].rank, -item[0].score))
                wrong = wrong_entries[0][1]
                steps.append(
                    DemonstrationStep(
                        "Select", (wrong.hypothesis_id,), created, (),
                        ("plausible_branch_requires_more_evidence",),
                    )
                )
                active = list(created)
                wrong_page = self._expand_terminal_frontier(
                    wrong,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="recovery_probe",
                    require_required_relation=False,
                )
                wrong_children = list(wrong_page.node_ids) if wrong_page else []
                top_probe = next(
                    (
                        hypotheses[node_id]
                        for node_id in wrong_children
                        if "policy_choice" in hypotheses[node_id].provenance
                    ),
                    None,
                )
                if top_probe is None:
                    self.stats["recovery_probe_without_visible_failure"] += 1
                    direct = self._build_direct_progress_demo(
                        question_id,
                        question,
                        join,
                        next_join,
                        candidates,
                        gold_answers,
                        gold_option,
                        best_alternative_score,
                        len(ranked_options),
                    )
                    if direct is not None:
                        direct.private_metadata["candidate_future_values"] = candidate_future_values
                    return direct
                active.remove(wrong.hypothesis_id)
                active.extend(wrong_children)
                steps.append(
                    DemonstrationStep(
                        "Find_relation", (wrong.target_expression,), created,
                        tuple(wrong_children), ("test_top_ranked_continuation",),
                        (
                            (wrong_page.start, wrong_page.stop, wrong_page.total)
                            if wrong_page else ()
                        ),
                    )
                )
                semantic_mismatch = bool(top_probe.denotation)
                prune_ids = [
                    node_id
                    for node_id in wrong_children
                    if not hypotheses[node_id].denotation
                ]
                if semantic_mismatch and top_probe.hypothesis_id not in prune_ids:
                    prune_ids.append(top_probe.hypothesis_id)
                if not prune_ids:
                    self.stats["recovery_probe_without_visible_failure"] += 1
                    direct = self._build_direct_progress_demo(
                        question_id,
                        question,
                        join,
                        next_join,
                        candidates,
                        gold_answers,
                        gold_option,
                        best_alternative_score,
                        len(ranked_options),
                    )
                    if direct is not None:
                        direct.private_metadata["candidate_future_values"] = candidate_future_values
                    return direct
                self.stats[
                    "recovery_probe_visible_semantic_mismatch"
                    if semantic_mismatch
                    else "recovery_probe_visible_empty"
                ] += 1
                for child_id in prune_ids:
                    before = tuple(active)
                    active.remove(child_id)
                    rationale = (
                        (f"question_path_mismatch:{wrong.relation}",)
                        if hypotheses[child_id].denotation
                        else ("empty_execution",)
                    )
                    steps.append(
                        DemonstrationStep(
                            "Prune", (child_id,), before, (), rationale,
                        )
                    )
                steps.append(
                    DemonstrationStep(
                        "Select", (gold.hypothesis_id,), tuple(active), (),
                        ("return_after_top_probe_failed",),
                    )
                )
                gold_page = self._expand_terminal_frontier(
                    gold,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="continuation",
                )
                if gold_page is None:
                    return None
                gold_children = list(gold_page.node_ids)
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
                            (gold.target_expression,),
                            before_gold_expansion,
                            tuple(gold_children),
                            ("continue_preserved_alternative",),
                            (gold_page.start, gold_page.stop, gold_page.total),
                        ),
                        DemonstrationStep(
                            "Commit", (final_gold.hypothesis_id,), tuple(active), (),
                            ("complete", "executable"),
                        ),
                    ]
                )
                family = (
                    "semantic_frontier_recovery"
                    if semantic_mismatch
                    else "delayed_frontier_recovery"
                )
                probe_relation = "question_conditioned"
                self.stats["recovery_built"] += 1

        if needs_widen:
            self.stats["widen_demo_built"] += 1
        return HyperDemonstration(
            demo_id=f"{question_id}:join:{join.index}:{_digest(tuple(option.relation for option, _ in candidates))}",
            question_id=question_id,
            question=question,
            family=family,
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "gold_relation": join.relation,
                "gold_rank": gold_option.rank,
                "gold_score": gold_option.score,
                "gold_vs_best_alternative_margin": (
                    gold_option.score - best_alternative_score
                    if best_alternative_score is not None else None
                ),
                "proposal_relations": [option.relation for option, _ in candidates],
                "proposal_recall_at_frontier": not needs_widen,
                "proposal_recall_within_budget": True,
                "candidate_future_values": candidate_future_values,
                "widen_sources": [source] if needs_widen else [],
                "retrieval_intent_source": "question",
                "probe_relation": probe_relation,
                "decision_index": join.index,
            },
        )

    def _append_capacity_prunes(
        self,
        steps: List[DemonstrationStep],
        hypotheses: Mapping[str, ExecutedHypothesis],
        active: List[str],
        *,
        maximum_active: int,
        protected: Sequence[str] = (),
    ) -> bool:
        """Reject traces that need confidence-based deletion to fit the budget."""
        del steps, hypotheses, protected
        if len(active) > maximum_active:
            self.stats["trajectory_node_budget_miss"] += 1
            return False
        return True

    def _build_deep_progress_demo(
        self,
        *,
        question_id: str,
        question: str,
        join: ProgramStatement,
        following_joins: Sequence[ProgramStatement],
        candidates: Sequence[Tuple[RelationOption, ExecutedHypothesis]],
        gold_answers: Tuple[str, ...],
        gold_option: RelationOption,
        best_alternative_score: Optional[float],
        initial_steps: Sequence[DemonstrationStep],
        needs_initial_widen: bool,
        candidate_future_values: Mapping[str, Any],
    ) -> Optional[HyperDemonstration]:
        """Replay a complete public linear path instead of stopping at hop two."""
        if not following_joins:
            return None
        previous = join
        for following in following_joins:
            if following.sources != (previous.target,):
                return None
            previous = following

        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        active = list(hypotheses)
        steps = list(initial_steps)
        current_gold = next(
            item[1] for item in candidates if item[0].relation == join.relation
        )
        widen_sources = [
            step.arguments[0] for step in steps if step.action == "Widen"
        ]

        for continuation in following_joins:
            # Runtime checks capacity before replacing the selected parent, so
            # reserve a complete initial frontier first.
            if not self._append_capacity_prunes(
                steps,
                hypotheses,
                active,
                maximum_active=self.max_active - self.frontier_width,
                protected=(current_gold.hypothesis_id,),
            ):
                return None
            steps.append(
                DemonstrationStep(
                    "Select",
                    (current_gold.hypothesis_id,),
                    tuple(active),
                    (),
                    (f"question_relation_match:{current_gold.relation}",),
                )
            )
            child_pages = self._expand_terminal_frontier_batches(
                current_gold,
                continuation,
                hypotheses,
                question,
                stat_scope="deep_continuation",
            )
            if not child_pages:
                self.stats["deep_progress_miss"] += 1
                return None
            initial_page = child_pages[0]
            initial_children = list(initial_page.node_ids)
            widened_pages = child_pages[1:]
            all_children = [
                node_id for page in child_pages for node_id in page.node_ids
            ]
            next_gold = next(
                (
                    hypotheses[node_id]
                    for node_id in all_children
                    if hypotheses[node_id].relation == continuation.relation
                ),
                None,
            )
            if next_gold is None or len(all_children) < 2:
                self.stats["deep_progress_miss"] += 1
                return None

            before_expansion = tuple(active)
            active.remove(current_gold.hypothesis_id)
            active.extend(initial_children)
            steps.append(
                DemonstrationStep(
                    "Find_relation",
                    (current_gold.target_expression,),
                    before_expansion,
                    tuple(initial_children),
                    ("continue_supported_branch",),
                    (initial_page.start, initial_page.stop, initial_page.total),
                )
            )
            for page_number, widened_page in enumerate(widened_pages, start=2):
                widened_children = list(widened_page.node_ids)
                if not self._append_capacity_prunes(
                    steps,
                    hypotheses,
                    active,
                    maximum_active=self.max_active - len(widened_children),
                ):
                    return None
                steps.append(
                    DemonstrationStep(
                        "Widen",
                        (current_gold.target_expression,),
                        tuple(active),
                        tuple(widened_children),
                        (
                            "required_relation_missing_from_visible_pages",
                            f"relation_page:{page_number}",
                        ),
                        (
                            widened_page.start,
                            widened_page.stop,
                            widened_page.total,
                        ),
                    )
                )
                active.extend(widened_children)
                widen_sources.append(current_gold.target_expression)
            if len(active) > self.max_active:
                return None
            current_gold = next_gold

        if current_gold.denotation != gold_answers:
            self.stats["deep_progress_terminal_mismatch"] += 1
            return None
        steps.append(
            DemonstrationStep(
                "Commit",
                (current_gold.hypothesis_id,),
                tuple(active),
                (),
                ("complete", "executable"),
            )
        )
        self.stats["deep_progress_built"] += 1
        return HyperDemonstration(
            demo_id=(
                f"{question_id}:join:{join.index}:"
                f"{_digest(tuple(option.relation for option, _ in candidates))}:deep"
            ),
            question_id=question_id,
            question=question,
            family="deep_frontier_progress",
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "gold_relation": join.relation,
                "gold_rank": gold_option.rank,
                "gold_score": gold_option.score,
                "gold_vs_best_alternative_margin": (
                    gold_option.score - best_alternative_score
                    if best_alternative_score is not None else None
                ),
                "proposal_relations": [option.relation for option, _ in candidates],
                "proposal_recall_at_frontier": not needs_initial_widen,
                "proposal_recall_within_budget": True,
                "candidate_future_values": dict(candidate_future_values),
                "widen_sources": widen_sources,
                "retrieval_intent_source": "question",
                "probe_relation": None,
                "decision_index": join.index,
                "path_hops": 1 + len(following_joins),
            },
        )

    def _build_direct_progress_demo(
        self,
        question_id: str,
        question: str,
        join: ProgramStatement,
        next_join: ProgramStatement,
        candidates: Sequence[
            Tuple[RelationOption, ExecutedHypothesis]
        ],
        gold_answers: Tuple[str, ...],
        gold_option: RelationOption,
        best_alternative_score: Optional[float],
        ranked_relation_total: int,
    ) -> Optional[HyperDemonstration]:
        """Build ordinary two-hop progress from the same natural frontier."""
        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        created = tuple(hypotheses)
        gold = next(
            item[1] for item in candidates if item[0].relation == join.relation
        )
        child_pages = self._expand_terminal_frontier_batches(
            gold,
            next_join,
            hypotheses,
            question,
            stat_scope="continuation",
        )
        if not child_pages:
            self.stats["direct_progress_miss"] += 1
            return None
        initial_page = child_pages[0]
        initial_children = list(initial_page.node_ids)
        widened_pages = child_pages[1:]
        gold_children = [
            node_id for page in child_pages for node_id in page.node_ids
        ]
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
            self.stats["direct_progress_miss"] += 1
            return None
        source = _action_source(gold.function_state[:-1], join.raw)
        steps = [
            DemonstrationStep(
                "Find_relation", (source,), (), created,
                ("open_ranked_frontier",),
                (
                    0,
                    min(self.frontier_width, ranked_relation_total),
                    ranked_relation_total,
                ),
            ),
            DemonstrationStep(
                "Select",
                (gold.hypothesis_id,),
                created,
                rationale_facts=(f"question_relation_match:{gold.relation}",),
            ),
            DemonstrationStep(
                "Find_relation",
                (gold.target_expression,),
                created,
                tuple(initial_children),
                ("continue_supported_branch",),
                (initial_page.start, initial_page.stop, initial_page.total),
            ),
        ]
        active = [node_id for node_id in created if node_id != gold.hypothesis_id]
        active.extend(initial_children)
        family = "direct_frontier_progress"
        widen_sources = []
        for page_number, widened_page in enumerate(widened_pages, start=2):
            widened_children = list(widened_page.node_ids)
            if not self._append_capacity_prunes(
                steps,
                hypotheses,
                active,
                maximum_active=self.max_active - len(widened_children),
            ):
                return None
            steps.append(
                DemonstrationStep(
                    "Widen",
                    (gold.target_expression,),
                    tuple(active),
                    tuple(widened_children),
                    (
                        "required_relation_missing_from_visible_pages",
                        f"relation_page:{page_number}",
                    ),
                    (
                        widened_page.start,
                        widened_page.stop,
                        widened_page.total,
                    ),
                )
            )
            active.extend(widened_children)
            family = "adaptive_frontier_widen"
            widen_sources.append(gold.target_expression)
            self.stats["widen_demo_built"] += 1
        if len(active) > self.max_active:
            self.stats["direct_progress_capacity_miss"] += 1
            return None
        steps.append(
            DemonstrationStep(
                "Commit",
                (final_gold.hypothesis_id,),
                tuple(active),
                rationale_facts=("complete", "executable"),
            )
        )
        self.stats["direct_progress_built"] += 1
        return HyperDemonstration(
            demo_id=(
                f"{question_id}:join:{join.index}:"
                f"{_digest(tuple(option.relation for option, _ in candidates))}:direct"
            ),
            question_id=question_id,
            question=question,
            family=family,
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "gold_relation": join.relation,
                "gold_rank": gold_option.rank,
                "gold_score": gold_option.score,
                "gold_vs_best_alternative_margin": (
                    gold_option.score - best_alternative_score
                    if best_alternative_score is not None else None
                ),
                "proposal_relations": [
                    option.relation for option, _ in candidates
                ],
                "proposal_recall_at_frontier": True,
                "proposal_recall_within_budget": True,
                "widen_sources": widen_sources,
                "retrieval_intent_source": "question",
                "probe_relation": None,
                "decision_index": join.index,
            },
        )

    def _expand_terminal_frontier_batches(
        self,
        parent: ExecutedHypothesis,
        join: ProgramStatement,
        hypotheses: Dict[str, ExecutedHypothesis],
        question: str,
        stat_scope: str,
    ) -> List[ExecutedRelationPage]:
        """Execute stable relation pages through the required continuation."""
        options = list(
            self.candidate_provider(question.strip(), parent.function_state, join)
        )
        pages, required = self._pages_through_required_relation(
            options, join.relation
        )
        self.stats[f"{stat_scope}_decisions"] += 1
        if required is None:
            self.stats[f"{stat_scope}_proposal_miss"] += 1
            self.stats[f"{stat_scope}_structural_or_ranking_miss"] += 1
            return []
        if sum(len(page) for page in pages) > self.max_active:
            self.stats[f"{stat_scope}_within_budget_miss"] += 1
            return []
        self.stats[f"{stat_scope}_within_budget_hit"] += 1
        if len(pages) == 1:
            self.stats[f"{stat_scope}_proposal_hit"] += 1
        else:
            self.stats[f"{stat_scope}_proposal_miss"] += 1
            self.stats[f"{stat_scope}_recovered_by_widen"] += 1
        self.stats[f"{stat_scope}_gold_rank_{required.rank}"] += 1
        if required.rank == 1:
            self.stats[f"{stat_scope}_top1_hit"] += 1

        child_pages: List[ExecutedRelationPage] = []
        for page_index, page in enumerate(pages):
            children: List[str] = []
            for option in page:
                statement = replace_join_relation(join.raw, option.relation)
                state = list(parent.function_state) + [statement]
                values = self._execute(state, join.target)
                if values is None:
                    continue
                node_id = f"H{len(hypotheses)}"
                hypotheses[node_id] = ExecutedHypothesis(
                    node_id,
                    tuple(state),
                    join.target,
                    values,
                    denotation_labels=self._display_pairs(values),
                    relation=option.relation,
                    role="continuation",
                    parent_id=parent.hypothesis_id,
                    depth=parent.depth + 1,
                    provenance=(
                        "policy_choice" if option.rank == 1 else "ranked_alternative",
                    ),
                )
                children.append(node_id)
            bounds = relation_page(
                options,
                offset=page_index * self.frontier_width,
                page_size=self.frontier_width,
            )
            child_pages.append(
                ExecutedRelationPage(
                    tuple(children), bounds.start, bounds.stop, bounds.total
                )
            )
        return child_pages

    def _expand_terminal_frontier(
        self,
        parent: ExecutedHypothesis,
        join: ProgramStatement,
        hypotheses: Dict[str, ExecutedHypothesis],
        question: str,
        stat_scope: str,
        require_required_relation: bool = True,
    ) -> Optional[ExecutedRelationPage]:
        ranked_options = list(
            self.candidate_provider(question.strip(), parent.function_state, join)
        )
        page = relation_page(
            ranked_options, offset=0, page_size=self.frontier_width
        )
        options = page.items
        self.stats[f"{stat_scope}_decisions"] += 1
        if require_required_relation:
            required = next(
                (option for option in options if option.relation == join.relation), None
            )
            if required is None:
                self.stats[f"{stat_scope}_proposal_miss"] += 1
                return None
            self.stats[f"{stat_scope}_proposal_hit"] += 1
            self.stats[f"{stat_scope}_gold_rank_{required.rank}"] += 1
            if required.rank == 1:
                self.stats[f"{stat_scope}_top1_hit"] += 1
        else:
            self.stats[f"{stat_scope}_natural_frontier"] += 1
        children: List[str] = []
        for option in options:
            statement = replace_join_relation(join.raw, option.relation)
            state = list(parent.function_state) + [statement]
            values = self._execute(state, join.target)
            if values is None:
                continue
            node_id = f"H{len(hypotheses)}"
            hypotheses[node_id] = ExecutedHypothesis(
                node_id,
                tuple(state),
                join.target,
                values,
                denotation_labels=self._display_pairs(values),
                relation=option.relation,
                role="continuation",
                parent_id=parent.hypothesis_id,
                depth=parent.depth + 1,
                provenance=(
                    "policy_choice" if option.rank == 1 else "ranked_alternative",
                ),
            )
            children.append(node_id)
        return ExecutedRelationPage(
            tuple(children), page.start, page.stop, page.total
        )

    def _build_intersection_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
        combine: ProgramStatement,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        branches = self._immediate_intersection_branches(plan, combine)
        if len(branches) != 2:
            return None
        left_join, right_join = branches
        left_source = left_join.sources[0]
        right_source = right_join.sources[0]
        left_state = self._dependency_state(plan, left_source, before=left_join.index)
        right_state = self._dependency_state(plan, right_source, before=right_join.index)
        # The SFT trajectory must start from public candidate entities rather
        # than from a hidden multi-hop gold prefix.
        if any("JOIN(" in raw for raw in (*left_state, *right_state)):
            self.stats["conjunction_hidden_prefix"] += 1
            return self._build_deep_intersection_demo(
                question_id=question_id,
                question=question,
                plan=plan,
                combine=combine,
                gold_answers=gold_answers,
                terminal_branches=(left_join, right_join),
            )
        left_action_source = _action_source(left_state, left_join.raw)
        right_action_source = _action_source(right_state, right_join.raw)
        same_frontier = left_state == right_state and left_action_source == right_action_source
        if same_frontier and left_join.relation == right_join.relation:
            return None

        self.stats["conjunction_decisions"] += 1
        query = question.strip()
        hypotheses: Dict[str, ExecutedHypothesis] = {}
        frontiers = []
        if same_frontier:
            options = list(self.candidate_provider(query, left_state, left_join))
            pages, required = self._pages_through_required_relations(
                options, (left_join.relation, right_join.relation)
            )
            if not pages:
                self.stats["conjunction_proposal_miss"] += 1
                return None
            if sum(len(page) for page in pages) > self.max_active:
                self.stats["conjunction_within_budget_miss"] += 1
                return None
            relation_nodes = {}
            created_pages = []
            for page in pages:
                created = []
                for option in page:
                    statement = replace_join_relation(left_join.raw, option.relation)
                    state = left_state + [statement]
                    values = self._execute(state, left_join.target)
                    if values is None:
                        continue
                    node = ExecutedHypothesis(
                        f"H{len(hypotheses)}",
                        tuple(state),
                        left_join.target,
                        values,
                        denotation_labels=self._display_pairs(values),
                        relation=option.relation,
                        role=(
                            "required_branch"
                            if option.relation
                            in {left_join.relation, right_join.relation}
                            else "alternative"
                        ),
                        provenance=(
                            "policy_choice"
                            if option.rank == 1
                            else "ranked_alternative",
                        ),
                    )
                    hypotheses[node.hypothesis_id] = node
                    relation_nodes[option.relation] = node
                    created.append(node.hypothesis_id)
                created_pages.append(tuple(created))
            left = relation_nodes.get(left_join.relation)
            right = relation_nodes.get(right_join.relation)
            ranks = {
                "left": required[left_join.relation].rank,
                "right": required[right_join.relation].rank,
            }
            frontiers.append(
                (
                    left_action_source,
                    tuple(
                        ExecutedRelationPage(
                            node_ids=created,
                            start=page_index * self.frontier_width,
                            stop=min(
                                (page_index + 1) * self.frontier_width,
                                len(options),
                            ),
                            total=len(options),
                        )
                        for page_index, created in enumerate(created_pages)
                    ),
                )
            )
        else:
            required_nodes = []
            ranks = {}
            for label, state_before, action_source, join in (
                ("left", left_state, left_action_source, left_join),
                ("right", right_state, right_action_source, right_join),
            ):
                options = list(self.candidate_provider(query, state_before, join))
                pages, required = self._pages_through_required_relation(
                    options, join.relation
                )
                if required is None:
                    self.stats["conjunction_proposal_miss"] += 1
                    return None
                if sum(len(page) for page in pages) > self.max_active:
                    self.stats["conjunction_within_budget_miss"] += 1
                    return None
                ranks[label] = required.rank
                created_pages = []
                required_node = None
                for page in pages:
                    created = []
                    for option in page:
                        statement = replace_join_relation(join.raw, option.relation)
                        state = state_before + [statement]
                        values = self._execute(state, join.target)
                        if values is None:
                            continue
                        node = ExecutedHypothesis(
                            f"H{len(hypotheses)}",
                            tuple(state),
                            join.target,
                            values,
                            denotation_labels=self._display_pairs(values),
                            relation=option.relation,
                            role=(
                                "required_branch"
                                if option.relation == join.relation
                                else "alternative"
                            ),
                            provenance=(
                                "policy_choice"
                                if option.rank == 1
                                else "ranked_alternative",
                            ),
                        )
                        hypotheses[node.hypothesis_id] = node
                        created.append(node.hypothesis_id)
                        if option.relation == join.relation:
                            required_node = node
                    created_pages.append(tuple(created))
                if required_node is None:
                    return None
                required_nodes.append(required_node)
                frontiers.append(
                    (
                        action_source,
                        tuple(
                            ExecutedRelationPage(
                                node_ids=created,
                                start=page_index * self.frontier_width,
                                stop=min(
                                    (page_index + 1) * self.frontier_width,
                                    len(options),
                                ),
                                total=len(options),
                            )
                            for page_index, created in enumerate(created_pages)
                        ),
                    )
                )
            left, right = required_nodes

        if left is None or right is None:
            return None
        self.stats["conjunction_proposal_hit"] += 1
        self.stats[f"conjunction_worst_required_rank_{max(ranks.values())}"] += 1
        combined_state, combined_target = combine_function_states(
            left.function_state, left.target_expression,
            right.function_state, right.target_expression,
        )
        combined_values = self._execute(combined_state, combined_target)
        if combined_values is None:
            return None
        reference_state = self._dependency_state(
            plan, combine.target, before=combine.index + 1
        )
        reference_values = self._execute(reference_state, combine.target)
        if (
            reference_values != gold_answers
            or combined_values != gold_answers
            or combined_values != normalize_values(set(left.denotation) & set(right.denotation))
            or combined_values in {left.denotation, right.denotation}
        ):
            return None
        combined = ExecutedHypothesis(
            f"H{len(hypotheses)}", tuple(combined_state), combined_target,
            combined_values, denotation_labels=self._display_pairs(combined_values),
            role="combined", parent_id=left.hypothesis_id,
            parent_ids=(left.hypothesis_id, right.hypothesis_id),
            operation="combine", depth=max(left.depth, right.depth) + 1,
            provenance=(f"combined_with:{right.hypothesis_id}",),
        )
        hypotheses[combined.hypothesis_id] = combined
        steps = []
        active = []
        if sum(
            len(page.node_ids) for _, pages in frontiers for page in pages
        ) > self.max_active:
            self.stats["conjunction_within_budget_miss"] += 1
            return None
        for index, (action_source, pages) in enumerate(frontiers):
            initial_page = pages[0]
            initial_created = initial_page.node_ids
            steps.append(
                DemonstrationStep(
                    "Find_relation",
                    (action_source,),
                    tuple(active),
                    initial_created,
                    (
                        "open_shared_conjunction_frontier"
                        if len(frontiers) == 1
                        else f"open_conjunction_branch_{index + 1}"
                    ,),
                    (initial_page.start, initial_page.stop, initial_page.total),
                )
            )
            active.extend(initial_created)
            for page_number, page in enumerate(pages[1:], start=2):
                created = page.node_ids
                steps.append(
                    DemonstrationStep(
                        "Widen",
                        (action_source,),
                        tuple(active),
                        created,
                        (
                            "required_relation_missing_from_visible_pages",
                            f"relation_page:{page_number}",
                        ),
                        (page.start, page.stop, page.total),
                    )
                )
                active.extend(created)
        steps.extend(
            [
                DemonstrationStep(
                    "Combine", (left.hypothesis_id, right.hypothesis_id),
                    tuple(active), created=(combined.hypothesis_id,),
                    rationale_facts=("both_branches_necessary",),
                ),
            ]
        )
        steps.append(
            DemonstrationStep(
                "Commit", (combined.hypothesis_id,),
                tuple(
                    node_id
                    for node_id in active
                    if node_id not in {left.hypothesis_id, right.hypothesis_id}
                ) + (combined.hypothesis_id,),
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
            private_metadata={
                "decision_index": combine.index,
                "retrieval_intent_source": "question",
                "required_relation_ranks": {
                    "left": ranks["left"],
                    "right": ranks["right"],
                },
                "conjunction_roots": len(frontiers),
            },
        )

    def _build_deep_intersection_demo(
        self,
        *,
        question_id: str,
        question: str,
        plan: GoldPlan,
        combine: ProgramStatement,
        gold_answers: Tuple[str, ...],
        terminal_branches: Sequence[ProgramStatement],
    ) -> Optional[HyperDemonstration]:
        """Expose complete branch exploration before combining and continuing."""
        branch_chains = [
            self._join_chain_to_root(plan, terminal)
            for terminal in terminal_branches
        ]
        if any(not chain for chain in branch_chains):
            return None

        query = question.strip()
        hypotheses: Dict[str, ExecutedHypothesis] = {}
        active: List[str] = []
        steps: List[DemonstrationStep] = []
        required_terminals: List[ExecutedHypothesis] = []
        widen_sources: List[str] = []

        def open_root_frontier(
            join: ProgramStatement,
            protected: Sequence[str],
            branch_label: str,
        ) -> Optional[ExecutedHypothesis]:
            source = join.sources[0] if join.sources else _join_source(join.raw)
            state_before = self._dependency_state(plan, source, before=join.index)
            if any("JOIN(" in raw or "AND(" in raw for raw in state_before):
                return None
            options = list(self.candidate_provider(query, state_before, join))
            pages, required = self._pages_through_required_relation(
                options, join.relation
            )
            self.stats["deep_conjunction_root_decisions"] += 1
            if required is None:
                self.stats["deep_conjunction_root_structural_or_ranking_miss"] += 1
                return None
            if sum(len(page) for page in pages) > self.max_active:
                self.stats["deep_conjunction_root_within_budget_miss"] += 1
                return None
            self.stats["deep_conjunction_root_within_budget_hit"] += 1
            initial_options = pages[0]
            initial_relations = {option.relation for option in initial_options}
            needs_widen = len(pages) > 1
            self.stats[
                "deep_conjunction_root_proposal_miss"
                if needs_widen else "deep_conjunction_root_proposal_hit"
            ] += 1
            if needs_widen:
                self.stats["deep_conjunction_root_recovered_by_widen"] += 1
            created_pages: List[List[str]] = [[] for _ in pages]
            required_node = None
            for page_index, page in enumerate(pages):
                for option in page:
                    statement = replace_join_relation(join.raw, option.relation)
                    state = state_before + [statement]
                    values = self._execute(state, join.target)
                    if values is None:
                        continue
                    node_id = f"H{len(hypotheses)}"
                    node = ExecutedHypothesis(
                        node_id,
                        tuple(state),
                        join.target,
                        values,
                        denotation_labels=self._display_pairs(values),
                        relation=option.relation,
                        role="continuation",
                        provenance=(
                            "policy_choice"
                            if option.rank == 1
                            else "ranked_alternative",
                        ),
                    )
                    hypotheses[node_id] = node
                    created_pages[page_index].append(node_id)
                    if option.relation == join.relation:
                        required_node = node
            initial_created = created_pages[0]
            if required_node is None or len(initial_created) < 2:
                return None
            if not self._append_capacity_prunes(
                steps,
                hypotheses,
                active,
                maximum_active=self.max_active - len(initial_created),
                protected=protected,
            ):
                return None
            action_source = _action_source(state_before, join.raw)
            steps.append(
                DemonstrationStep(
                    "Find_relation",
                    (action_source,),
                    tuple(active),
                    tuple(initial_created),
                    (f"open_deep_conjunction_{branch_label}_root",),
                    (0, min(self.frontier_width, len(options)), len(options)),
                )
            )
            active.extend(initial_created)
            for page_number, widened_created in enumerate(
                created_pages[1:], start=2
            ):
                if not self._append_capacity_prunes(
                    steps,
                    hypotheses,
                    active,
                    maximum_active=self.max_active - len(widened_created),
                    protected=protected,
                ):
                    return None
                steps.append(
                    DemonstrationStep(
                        "Widen",
                        (action_source,),
                        tuple(active),
                        tuple(widened_created),
                        (
                            "required_relation_missing_from_visible_pages",
                            f"relation_page:{page_number}",
                        ),
                        (
                            (page_number - 1) * self.frontier_width,
                            min(page_number * self.frontier_width, len(options)),
                            len(options),
                        ),
                    )
                )
                active.extend(widened_created)
                widen_sources.append(action_source)
            return required_node

        def continue_chain(
            current: ExecutedHypothesis,
            remaining: Sequence[ProgramStatement],
            protected: Sequence[str],
            stat_scope: str,
        ) -> Optional[ExecutedHypothesis]:
            for continuation in remaining:
                if not self._append_capacity_prunes(
                    steps,
                    hypotheses,
                    active,
                    maximum_active=self.max_active - self.frontier_width,
                    protected=(*protected, current.hypothesis_id),
                ):
                    return None
                steps.append(
                    DemonstrationStep(
                        "Select",
                        (current.hypothesis_id,),
                        tuple(active),
                        (),
                        (f"question_relation_match:{current.relation}",),
                    )
                )
                child_pages = self._expand_terminal_frontier_batches(
                    current,
                    continuation,
                    hypotheses,
                    question,
                    stat_scope=stat_scope,
                )
                if not child_pages:
                    return None
                initial_page = child_pages[0]
                initial_children = list(initial_page.node_ids)
                widened_pages = child_pages[1:]
                children = [
                    node_id for page in child_pages for node_id in page.node_ids
                ]
                next_required = next(
                    (
                        hypotheses[node_id]
                        for node_id in children
                        if hypotheses[node_id].relation == continuation.relation
                    ),
                    None,
                )
                if next_required is None or len(initial_children) < 2:
                    return None
                before = tuple(active)
                active.remove(current.hypothesis_id)
                active.extend(initial_children)
                steps.append(
                    DemonstrationStep(
                        "Find_relation",
                        (current.target_expression,),
                        before,
                        tuple(initial_children),
                        ("continue_required_branch",),
                        (initial_page.start, initial_page.stop, initial_page.total),
                    )
                )
                for page_number, widened_page in enumerate(
                    widened_pages, start=2
                ):
                    widened_children = list(widened_page.node_ids)
                    if not self._append_capacity_prunes(
                        steps,
                        hypotheses,
                        active,
                        maximum_active=self.max_active - len(widened_children),
                        protected=protected,
                    ):
                        return None
                    steps.append(
                        DemonstrationStep(
                            "Widen",
                            (current.target_expression,),
                            tuple(active),
                            tuple(widened_children),
                            (
                                "required_relation_missing_from_visible_pages",
                                f"relation_page:{page_number}",
                            ),
                            (
                                widened_page.start,
                                widened_page.stop,
                                widened_page.total,
                            ),
                        )
                    )
                    active.extend(widened_children)
                    widen_sources.append(current.target_expression)
                current = next_required
            return current

        for index, chain in enumerate(branch_chains):
            label = "left" if index == 0 else "right"
            protected = tuple(node.hypothesis_id for node in required_terminals)
            current = open_root_frontier(chain[0], protected, label)
            if current is None:
                return None
            current = continue_chain(
                current,
                chain[1:],
                protected,
                "deep_conjunction_continuation",
            )
            if current is None:
                return None
            hypotheses[current.hypothesis_id] = replace(
                current, role="required_branch"
            )
            required_terminals.append(hypotheses[current.hypothesis_id])

        left, right = required_terminals
        combined_state, combined_target = combine_function_states(
            left.function_state,
            left.target_expression,
            right.function_state,
            right.target_expression,
        )
        combined_values = self._execute(combined_state, combined_target)
        reference_state = self._dependency_state(
            plan, combine.target, before=combine.index + 1
        )
        reference_values = self._execute(reference_state, combine.target)
        if (
            combined_values is None
            or combined_values != reference_values
            or combined_values
            != normalize_values(set(left.denotation) & set(right.denotation))
            or combined_values in {left.denotation, right.denotation}
        ):
            return None
        combined = ExecutedHypothesis(
            f"H{len(hypotheses)}",
            tuple(combined_state),
            combined_target,
            combined_values,
            denotation_labels=self._display_pairs(combined_values),
            role="combined",
            parent_id=left.hypothesis_id,
            parent_ids=(left.hypothesis_id, right.hypothesis_id),
            operation="combine",
            depth=max(left.depth, right.depth) + 1,
            provenance=(f"combined_with:{right.hypothesis_id}",),
        )
        hypotheses[combined.hypothesis_id] = combined
        steps.append(
            DemonstrationStep(
                "Combine",
                (left.hypothesis_id, right.hypothesis_id),
                tuple(active),
                created=(combined.hypothesis_id,),
                rationale_facts=("both_branches_necessary",),
            )
        )
        active.remove(left.hypothesis_id)
        active.remove(right.hypothesis_id)
        active.append(combined.hypothesis_id)

        tail = self._linear_tail_after_combine(plan, combine)
        final = continue_chain(
            combined,
            tail,
            (),
            "deep_conjunction_tail",
        )
        if final is None or final.denotation != gold_answers:
            return None
        steps.append(
            DemonstrationStep(
                "Commit",
                (final.hypothesis_id,),
                tuple(active),
                (),
                ("complete", "executable"),
            )
        )
        self.stats["deep_conjunction_built"] += 1
        return HyperDemonstration(
            demo_id=f"{question_id}:and:{combine.index}:deep",
            question_id=question_id,
            question=question,
            family="deep_conjunction_progress",
            hypotheses=hypotheses,
            steps=steps,
            gold_answers=gold_answers,
            private_metadata={
                "decision_index": combine.index,
                "retrieval_intent_source": "question",
                "conjunction_roots": 2,
                "widen_sources": widen_sources,
                "path_hops": sum(len(chain) for chain in branch_chains)
                + len(tail),
            },
        )

    def _join_chain_to_root(
        self, plan: GoldPlan, terminal: ProgramStatement
    ) -> Tuple[ProgramStatement, ...]:
        chain = [terminal]
        current = terminal
        while current.sources:
            dependency = self._definition_before(
                plan, current.sources[0], current.index
            )
            if dependency is None:
                return ()
            if dependency.kind == "start":
                return tuple(reversed(chain))
            if dependency.kind != "join":
                return ()
            chain.append(dependency)
            current = dependency
        return ()

    @staticmethod
    def _linear_tail_after_combine(
        plan: GoldPlan, combine: ProgramStatement
    ) -> Tuple[ProgramStatement, ...]:
        tail = []
        current_target = combine.target
        for statement in plan.statements:
            if statement.index <= combine.index:
                continue
            if statement.kind == "join" and statement.sources == (current_target,):
                tail.append(statement)
                current_target = statement.target
            elif statement.kind == "and" and current_target in statement.sources:
                break
        return tuple(tail)

    def _definition_before(
        self, plan: GoldPlan, target: str, before: int
    ) -> Optional[ProgramStatement]:
        return next(
            (
                statement
                for statement in reversed(plan.statements)
                if statement.index < before
                and statement.target == target
                and statement.kind != "stop"
            ),
            None,
        )

    def _has_bare_type_intersection(self, plan: GoldPlan) -> bool:
        for combine in plan.intersections:
            for source in combine.sources:
                definition = self._definition_before(plan, source, combine.index)
                if (
                    definition is not None
                    and definition.kind == "start"
                    and definition.arguments
                    and _is_ontology_type(definition.arguments[0])
                ):
                    return True
        return False

    def _immediate_intersection_branches(
        self, plan: GoldPlan, combine: ProgramStatement
    ) -> Tuple[ProgramStatement, ...]:
        branches = tuple(
            self._definition_before(plan, source, combine.index)
            for source in combine.sources
        )
        if len(branches) != 2 or any(
            branch is None or branch.kind != "join" or not branch.sources
            for branch in branches
        ):
            return ()
        return branches

    def _semantic_intersection_join_indexes(
        self, plan: GoldPlan, combine: ProgramStatement
    ) -> Tuple[int, ...]:
        """Return every JOIN that belongs to a real two-branch intersection."""
        branches = self._immediate_intersection_branches(plan, combine)
        if len(branches) != 2:
            return ()
        indexes = set()

        def visit(statement: ProgramStatement) -> None:
            if statement.kind == "join":
                indexes.add(statement.index)
            for source in statement.sources:
                dependency = self._definition_before(plan, source, statement.index)
                if dependency is not None:
                    visit(dependency)

        for branch in branches:
            visit(branch)
        return tuple(sorted(indexes))

    def _row_candidate_entities(
        self, row: Mapping[str, Any], plan: GoldPlan, question_id: str
    ) -> List[Tuple[str, str]]:
        extra = row.get("extra_info") or {}
        if isinstance(extra, Mapping):
            entities = extra.get("extracted_entities") or extra.get("candidate_entities")
            if entities:
                values = [
                    (str(item[0]), str(item[-1]))
                    for item in entities
                    if item
                    and str(item[-1]).startswith(("m.", "g."))
                ]
            else:
                values = []
        else:
            values = []
        if not values:
            for statement in plan.statements:
                if statement.kind == "start":
                    assignment = _ASSIGNMENT.match(statement.raw)
                    start = _START.match(assignment.group(2)) if assignment else None
                    if start is None:
                        continue
                    value = start.group(1)
                    if value.startswith(("m.", "g.")):
                        values.append((value, value))

        unique: Dict[str, str] = {}
        for name, identity in values:
            unique.setdefault(identity, name)
        unresolved = [
            identity for identity, name in unique.items()
            if name == identity and (identity.startswith("m.") or identity.startswith("g."))
        ]
        if unresolved:
            self._resolve_display_labels(unresolved)
            for identity in unresolved:
                if identity in self._display_cache:
                    unique[identity] = self._display_cache[identity]
        ordered = [(name, identity) for identity, name in unique.items()]
        ordered.sort(key=lambda item: _digest(question_id, item[1]))
        return ordered

    @staticmethod
    def _row_base_prompt(row: Mapping[str, Any]) -> str:
        prompt = row.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, Sequence):
            for message in prompt:
                if isinstance(message, Mapping) and message.get("role") == "user":
                    return str(message.get("content", ""))
        return ""

    def _dependency_state(
        self, plan: GoldPlan, target: str, before: Optional[int] = None
    ) -> List[str]:
        ordered: List[ProgramStatement] = []
        seen = set()

        def visit(expression: str, limit: int) -> None:
            statement = self._definition_before(plan, expression, limit)
            if statement is None or statement.index in seen:
                return
            for source in statement.sources:
                visit(source, statement.index)
            seen.add(statement.index)
            ordered.append(statement)

        visit(target, len(plan.statements) + 1 if before is None else before)
        return [statement.raw for statement in ordered]

def _has_visible_path_mismatch(
    demo: HyperDemonstration,
    step: DemonstrationStep,
    node: ExecutedHypothesis,
) -> bool:
    if demo.family not in {"semantic_frontier_recovery", "adaptive_frontier_widen"}:
        return False
    visible_relations = set(relation_path(node.function_state))
    return any(
        fact.split(":", 1)[1] in visible_relations
        for fact in step.rationale_facts
        if fact.startswith("question_path_mismatch:") and ":" in fact
    )


def _has_public_prune_reason(
    demo: HyperDemonstration,
    step: DemonstrationStep,
    node: ExecutedHypothesis,
    active_count: Optional[int] = None,
    max_active: Optional[int] = None,
) -> bool:
    del active_count, max_active
    capacity_facts = {
        "frontier_capacity_eviction",
        "frontier_capacity_reservation",
    }.intersection(step.rationale_facts)
    if capacity_facts:
        # Historical v5 exports used capacity as a pruning rationale. Keep
        # those artifacts readable, but never admit that behavior into the
        # paged relation contract used by new SFT, RL, and inference runs.
        return "relation_page_size" not in demo.private_metadata
    return (
        _has_visible_path_mismatch(demo, step, node)
    )


class DemonstrationValidator:
    """Replay private executable states and verify graph-action consistency."""

    def __init__(
        self,
        executor: ProgramExecutor,
        max_active: int = 24,
        execution_cache: Optional[
            Dict[Tuple[Tuple[str, ...], str], Optional[Tuple[str, ...]]]
        ] = None,
    ):
        self.executor = executor
        self.max_active = int(max_active)
        self.execution_cache = execution_cache if execution_cache is not None else {}

    def _replay(
        self, functions: Sequence[str], target: str
    ) -> Optional[Tuple[str, ...]]:
        key = (tuple(str(item) for item in functions), str(target))
        if key in self.execution_cache:
            return self.execution_cache[key]
        try:
            result = normalize_values(self.executor(functions, target))
        except ProgramExecutionError:
            result = None
        self.execution_cache[key] = result
        return result

    def validate(self, demo: HyperDemonstration) -> List[str]:
        errors: List[str] = []
        for node in demo.hypotheses.values():
            replayed = self._replay(node.function_state, node.target_expression)
            if replayed is None:
                errors.append(f"{node.hypothesis_id}: replay execution failed")
                continue
            if replayed != node.denotation:
                errors.append(f"{node.hypothesis_id}: replay mismatch")

        active = set(demo.steps[0].visible_before if demo.steps else ())
        committed = None
        selected = None
        relation_page_size = int(
            demo.private_metadata.get("relation_page_size", 6)
        )
        max_nodes = int(demo.private_metadata.get("max_nodes", 24))
        paged_contract = "relation_page_size" in demo.private_metadata
        frontiers: List[Dict[str, Any]] = []
        known_nodes = set(active)
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
            if len(known_nodes.union(step.created)) > max_nodes:
                errors.append(
                    f"{step.action}: executed-node budget would exceed {max_nodes}"
                )
            known_nodes.update(step.created)
            if set(step.visible_before) != active:
                errors.append(
                    f"{step.action}: visible state {sorted(step.visible_before)} "
                    f"does not match active state {sorted(active)}"
                )
            if step.action == "Prune":
                if step.arguments[0] not in active:
                    errors.append(f"Prune targets inactive {step.arguments[0]}")
                node = demo.hypotheses[step.arguments[0]]
                if node.denotation:
                    if (
                        "frontier_capacity_eviction" in step.rationale_facts
                        and len(active) != self.max_active
                    ):
                        errors.append(
                            f"Prune {step.arguments[0]} uses capacity eviction when the "
                            f"frontier is {len(active)}/{self.max_active}, not full"
                        )
                    elif (
                        "frontier_capacity_reservation" in step.rationale_facts
                        and len(active) >= self.max_active
                    ):
                        errors.append(
                            f"Prune {step.arguments[0]} uses capacity reservation when the "
                            f"frontier is already {len(active)}/{self.max_active}"
                        )
                    elif not _has_public_prune_reason(
                        demo, step, node, len(active), self.max_active
                    ):
                        errors.append(
                            f"Prune {step.arguments[0]} lacks visible contradictory evidence"
                        )
                active.discard(step.arguments[0])
                if selected == step.arguments[0]:
                    selected = None
            elif step.action == "Select":
                if step.arguments[0] not in active:
                    errors.append(f"Select targets inactive {step.arguments[0]}")
                if not demo.hypotheses[step.arguments[0]].denotation:
                    errors.append(f"Select targets empty {step.arguments[0]}")
                selected = step.arguments[0]
            elif step.action == "Widen":
                if selected is not None:
                    errors.append("Widen must occur before Select")
                if not active:
                    errors.append("Widen requires an open active frontier")
                if not step.created:
                    errors.append("Widen must create at least one additional hypothesis")
                if paged_contract and len(step.relation_page) != 3:
                    errors.append("Widen must expose an explicit relation page cursor")
                elif step.relation_page:
                    start, stop, total = step.relation_page
                    frontier = next(
                        (
                            item
                            for item in reversed(frontiers)
                            if item["source"] == step.arguments[0]
                            and item["exposed"] == start
                            and set(item["node_ids"]).intersection(active)
                        ),
                        None,
                    )
                    if frontier is None:
                        errors.append(
                            f"Widen source {step.arguments[0]} does not continue "
                            "an open stable relation page"
                        )
                    elif frontier["total"] != total:
                        errors.append("Widen changed the ranked relation-list size")
                    elif stop <= start or stop - start > relation_page_size or stop > total:
                        errors.append("Widen exposed an invalid relation page span")
                    else:
                        frontier["exposed"] = stop
                        frontier["node_ids"].extend(step.created)
                active.update(step.created)
            elif step.action == "Find_relation":
                if len(step.arguments) != 1:
                    errors.append(
                        "Find_relation must expose only its source; relation ranking is environment-owned"
                    )
                if paged_contract and len(step.relation_page) != 3:
                    errors.append("Find_relation must expose an explicit relation page cursor")
                elif step.relation_page:
                    start, stop, total = step.relation_page
                    if (
                        start != 0
                        or stop <= 0
                        or stop > total
                        or stop > relation_page_size
                    ):
                        errors.append("Find_relation exposed an invalid first relation page")
                    else:
                        frontiers.append(
                            {
                                "source": step.arguments[0],
                                "node_ids": list(step.created),
                                "exposed": stop,
                                "total": total,
                            }
                        )
                candidate_sources = {
                    str(entity[-1])
                    for entity in demo.private_metadata.get("candidate_entities", ())
                    if entity
                }
                opens_new_root = (
                    bool(active)
                    and selected is None
                    and step.arguments[0] in candidate_sources
                )
                if active and selected is None and not opens_new_root:
                    errors.append("Find_relation requires a selected active hypothesis")
                if selected is not None:
                    if selected not in active:
                        errors.append(f"Find_relation expands inactive {selected}")
                    expected_source = demo.hypotheses[selected].target_expression
                    if step.arguments[0] != expected_source:
                        errors.append(
                            f"Find_relation source {step.arguments[0]} does not match "
                            f"selected {selected} target {expected_source}"
                        )
                    active.discard(selected)
                active.update(step.created)
                selected = None
            elif step.action == "Combine":
                left, right = step.arguments
                if left == right:
                    errors.append("Combine requires distinct hypotheses")
                if left not in active or right not in active:
                    errors.append("Combine requires two active parents")
                active.difference_update((left, right))
                if len(step.created) != 1:
                    errors.append("Combine must create exactly one hypothesis")
                else:
                    combined = demo.hypotheses[step.created[0]]
                    if set(combined.parent_ids) != {left, right}:
                        errors.append(
                            f"Combine parents {sorted((left, right))} do not match "
                            f"required branches {sorted(combined.parent_ids)}"
                        )
                    active.add(combined.hypothesis_id)
                selected = None
            elif step.action == "Merge":
                if len(step.arguments) != 2:
                    errors.append("Merge requires an expression and an ontology type")
                elif not _is_ontology_type(step.arguments[1]):
                    errors.append(f"Merge has invalid ontology type {step.arguments[1]}")
                if len(step.created) != 1:
                    errors.append("Merge must create exactly one hypothesis")
                elif selected is None:
                    errors.append("Merge requires Select")
                else:
                    child = demo.hypotheses[step.created[0]]
                    if selected not in active:
                        errors.append(f"Merge expands inactive {selected}")
                    parent = demo.hypotheses[selected]
                    if step.arguments[0] != parent.target_expression:
                        errors.append(
                            f"Merge source {step.arguments[0]} does not match "
                            f"selected {selected} target {parent.target_expression}"
                        )
                    if child.parent_id != selected:
                        errors.append(
                            f"Merge child {child.hypothesis_id} has wrong parent"
                        )
                    active.discard(selected)
                    active.add(child.hypothesis_id)
                selected = None
            elif step.action in {"Order", "Compare", "Time_constraint", "Count"}:
                expected_arguments = {
                    "Order": 3,
                    "Compare": 3,
                    "Time_constraint": 2,
                    "Count": 1,
                }
                if len(step.arguments) != expected_arguments[step.action]:
                    errors.append(
                        f"{step.action} requires {expected_arguments[step.action]} arguments"
                    )
                if len(step.created) != 1:
                    errors.append(f"{step.action} must create exactly one hypothesis")
                else:
                    child = demo.hypotheses[step.created[0]]
                    if selected is not None:
                        if selected not in active:
                            errors.append(f"{step.action} expands inactive {selected}")
                        if child.parent_id != selected:
                            errors.append(
                                f"{step.action} child {child.hypothesis_id} has wrong parent"
                            )
                        active.discard(selected)
                    elif step.action not in {"Order", "Compare"}:
                        errors.append(f"{step.action} requires Select")
                    elif child.parent_id is not None:
                        errors.append(f"root {step.action} must not have a parent")
                    active.add(child.hypothesis_id)
                selected = None
            elif step.action == "Commit":
                node = demo.hypotheses[step.arguments[0]]
                if step.arguments[0] not in active:
                    errors.append(f"Commit targets inactive {step.arguments[0]}")
                if node.denotation != demo.gold_answers:
                    errors.append(f"Commit {node.hypothesis_id} does not return gold answers")
                committed = node.hypothesis_id
                active = {node.hypothesis_id}
                selected = None
            else:
                errors.append(f"unsupported action {step.action}")
            if len(active) > self.max_active:
                errors.append(f"active hypothesis budget exceeded: {len(active)}")

        if demo.family in {"conjunction", "deep_conjunction_progress"}:
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
            if demo.family == "conjunction":
                if committed != (combined.hypothesis_id if combined else None):
                    errors.append("conjunction must commit its combined hypothesis")
            elif combined is not None and committed is not None:
                ancestor = demo.hypotheses[committed]
                while ancestor.parent_id is not None and ancestor.hypothesis_id != combined.hypothesis_id:
                    ancestor = demo.hypotheses[ancestor.parent_id]
                if ancestor.hypothesis_id != combined.hypothesis_id:
                    errors.append("deep conjunction must commit the combined branch or its descendant")
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
                    "answer_labels": dict(node.denotation_labels),
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


def _public_graph(
    demo: HyperDemonstration,
    active: Sequence[str],
    *,
    selected: Optional[str] = None,
    committed: Optional[str] = None,
    executions: int = 0,
    known_ids: Optional[Sequence[str]] = None,
) -> str:
    known = set(known_ids or ())
    nodes = []
    for node in demo.hypotheses.values():
        if node.hypothesis_id not in known:
            continue
        nodes.append(
            {
                "node_id": node.hypothesis_id,
                "function_state": node.function_state,
                "denotation": node.denotation,
                "denotation_labels": dict(node.denotation_labels),
                "parent_id": node.parent_id,
                "parent_ids": node.parent_ids,
                "operation": node.operation,
                "relation_id": node.relation,
                "depth": node.depth,
                "provenance": node.provenance,
            }
        )
    return serialize_frontier(
        nodes,
        active_ids=active,
        selected_id=selected,
        committed_id=committed,
        max_active=int(demo.private_metadata.get("max_active", 24)),
        node_count=len(known),
        execution_calls=executions,
    )


def _action_text(step: DemonstrationStep) -> str:
    if step.action == "Find_relation":
        body = f"Find_relation [ {step.arguments[0]} ]"
        if "test_top_ranked_continuation" in step.rationale_facts:
            thought = (
                "I will test this branch's top continuation while its alternatives remain available."
            )
        elif "continue_preserved_alternative" in step.rationale_facts:
            thought = "I will continue the preserved branch after the failed probe."
        else:
            thought = "I will execute a bounded relation frontier and retain its alternatives."
    elif step.action == "Widen":
        body = f"Widen [ {step.arguments[0]} ]"
        thought = (
            "The visible relation pages do not cover the question, so I will inspect "
            "the next ranked page before selecting a path."
        )
    elif step.action == "Combine":
        body = f"Combine [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "Both active branches express required parts of the question."
    elif step.action == "Merge":
        body = f"Merge [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "The question requires this retained branch to have a specific Freebase type."
    elif step.action in {"Order", "Compare"}:
        body = f"{step.action} [ {' | '.join(step.arguments)} ]"
        thought = (
            "The retained executable branch now needs the logical operation "
            "specified by the question."
        )
    elif step.action == "Time_constraint":
        body = f"Time_constraint [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "I will apply the question's time condition to the retained branch."
    elif step.action == "Count":
        body = f"Count [ {step.arguments[0]} ]"
        thought = "The question asks for the number of results in this retained branch."
    else:
        body = f"{step.action} [ {step.arguments[0]} ]"
        thoughts = {
            "Select": (
                "The probe supplied negative evidence, so I will return to a preserved "
                "alternative."
                if "return_after_top_probe_failed" in step.rationale_facts
                else next(
                    (
                        "This relation best matches the question, so I will test it while "
                        "preserving the other hypotheses: " + fact.split(":", 1)[1]
                        for fact in step.rationale_facts
                        if fact.startswith("question_relation_match:")
                    ),
                    "I will investigate this hypothesis without discarding the others.",
                )
            ),
            "Prune": (
                "The frontier is full, so I will remove this lower-ranked branch to make room for a higher-priority continuation."
                if "frontier_capacity_eviction" in step.rationale_facts
                else "The current frontier does not have enough room for the incoming continuation, so I will remove a lower-ranked branch before expanding."
                if "frontier_capacity_reservation" in step.rationale_facts
                else "This visible relation path conflicts with what the question asks, so I will remove it."
                if any(
                    fact.startswith("question_path_mismatch:")
                    for fact in step.rationale_facts
                )
                else "This hypothesis has an empty execution result, so it cannot answer the question."
            ),
            "Commit": "This executable hypothesis covers the full question.",
        }
        thought = thoughts[step.action]
    return f"<think>{thought}</think>\n<action>{body}</action>"


def trajectory_sft_record(demo: HyperDemonstration) -> Dict[str, Any]:
    """Export one complete policy trajectory in the runtime's multi-turn format."""
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": build_hyper_prompt(
                demo.question,
                demo.private_metadata.get("candidate_entities", ()),
                demo.private_metadata.get("base_prompt", ""),
            ),
        }
    ]
    active: List[str] = list(demo.steps[0].visible_before if demo.steps else ())
    selected: Optional[str] = None
    committed_id: Optional[str] = None
    known = set(active)
    executions = 0
    open_frontiers: List[Dict[str, Any]] = []
    if active:
        messages.append(
            {
                "role": "user",
                "content": "<information>Executable branches are available.\n"
                + _public_graph(
                    demo,
                    active,
                    selected=selected,
                    executions=executions,
                    known_ids=known,
                )
                + "\n</information>",
            }
        )
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
            known.update(step.created)
            executions += len(step.created)
            selected = None
            event = "Executed the requested relation frontier."
            if step.relation_page:
                start, stop, total = step.relation_page
                if start != 0:
                    raise ValueError("Find_relation must expose the first relation page")
                open_frontiers.append(
                    {
                        "source": step.arguments[0],
                        "node_ids": list(step.created),
                        "exposed": stop,
                        "total": total,
                    }
                )
        elif step.action == "Widen":
            active.extend(step.created)
            known.update(step.created)
            executions += len(step.created)
            event = (
                f"Widened the frontier from {step.arguments[0]} with "
                f"{len(step.created)} additional executable hypotheses."
            )
            if step.relation_page:
                start, stop, total = step.relation_page
                frontier = next(
                    (
                        item
                        for item in reversed(open_frontiers)
                        if item["source"] == step.arguments[0]
                        and item["exposed"] == start
                    ),
                    None,
                )
                if frontier is None or frontier["total"] != total:
                    raise ValueError("Widen does not continue an open stable relation page")
                frontier["node_ids"].extend(step.created)
                frontier["exposed"] = stop
        elif step.action == "Select":
            selected = step.arguments[0]
            event = (
                f"Selected {selected}. Further Find_relation actions now expand "
                "this hypothesis."
            )
        elif step.action == "Prune":
            node = demo.hypotheses[step.arguments[0]]
            if node.denotation and not _has_public_prune_reason(
                demo,
                step,
                node,
                len(active),
                int(demo.private_metadata.get("max_active", 24)),
            ):
                raise ValueError(
                    f"trajectory {demo.demo_id} prunes a nonempty branch without public evidence"
                )
            active.remove(step.arguments[0])
            event = f"Pruned {step.arguments[0]}."
        elif step.action == "Combine":
            active.remove(step.arguments[0])
            active.remove(step.arguments[1])
            active.extend(step.created)
            known.update(step.created)
            executions += len(step.created)
            selected = None
            event = (
                f"Combined {step.arguments[0]} and {step.arguments[1]} into "
                f"{step.created[0]}."
            )
        elif step.action == "Merge":
            if len(step.created) != 1:
                raise ValueError("Merge must create one executable hypothesis")
            if selected is None:
                raise ValueError("Merge requires a selected hypothesis")
            active.remove(selected)
            active.extend(step.created)
            known.update(step.created)
            executions += 1
            selected = None
            event = (
                f"Applied ontology type {step.arguments[1]} into {step.created[0]}."
            )
        elif step.action in {"Order", "Compare", "Time_constraint", "Count"}:
            if len(step.created) != 1:
                raise ValueError(f"{step.action} must create one executable hypothesis")
            if selected is not None:
                active.remove(selected)
            elif step.action not in {"Order", "Compare"}:
                raise ValueError(f"{step.action} requires a selected hypothesis")
            active.extend(step.created)
            known.update(step.created)
            executions += 1
            selected = None
            event = f"Executed {step.action} into {step.created[0]}."
        elif step.action == "Commit":
            active = [step.arguments[0]]
            committed_id = step.arguments[0]
            selected = None
            values = " ".join(demo.hypotheses[committed_id].denotation)
            event = (
                f"Committed {committed_id}. Return exactly these values in <answer>: "
                f"{values}"
            )
        else:
            raise ValueError(f"unsupported action {step.action}")
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<information>\n{event}\n"
                    + _public_graph(
                        demo,
                        active,
                        selected=selected,
                        committed=(committed_id if step.action == "Commit" else None),
                        executions=executions,
                        known_ids=known,
                    )
                    + (
                        "\n"
                        + "\n".join(
                            serialize_relation_page_state(
                                str(frontier["source"]),
                                exposed=int(frontier["exposed"]),
                                total=int(frontier["total"]),
                                page_size=int(
                                    demo.private_metadata.get(
                                        "relation_page_size", 6
                                    )
                                ),
                            )
                            for frontier in open_frontiers
                            if committed_id is None
                            and set(frontier["node_ids"]).intersection(active)
                        )
                        if any(
                            committed_id is None
                            and set(frontier["node_ids"]).intersection(active)
                            for frontier in open_frontiers
                        )
                        else ""
                    )
                    + "\n</information>"
                ),
            }
        )
    committed = demo.hypotheses[committed_id] if committed_id else None
    if committed is None or committed.denotation != demo.gold_answers:
        raise ValueError(f"trajectory {demo.demo_id} did not finish on verified answers")
    messages.append(
        {
            "role": "assistant",
            "content": "<answer>" + " ".join(committed.denotation) + "</answer>",
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


def decision_sft_records(demo: HyperDemonstration) -> List[Dict[str, Any]]:
    """Export one exact conversation prefix per graph decision.

    The final answer-copy turn is deliberately excluded. This makes behavior
    cloning optimize the policy actions that SFT is meant to install, while the
    complete trajectory remains available for replay checks and inspection.
    """
    trajectory = trajectory_sft_record(demo)
    records = []
    decision_index = 0
    for message_index, message in enumerate(trajectory["messages"]):
        if message.get("role") != "assistant" or "<action>" not in message.get("content", ""):
            continue
        prefix = [dict(item) for item in trajectory["messages"][: message_index + 1]]
        for prior in prefix:
            # Keep the nested Parquet field non-nullable. Pandas otherwise
            # promotes nullable integer masks to floats before VERL sees them.
            prior["loss_mask"] = 0
        prefix[-1]["loss_mask"] = 1
        records.append(
            {
                "messages": prefix,
                "data_source": "hyper_r1_verified_decision",
                "extra_info": {
                    **trajectory["extra_info"],
                    "decision_index": decision_index,
                    "target_is_graph_action": True,
                },
            }
        )
        decision_index += 1
    return records
