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

from .hyper_prompt import build_hyper_prompt
from .hyper_r1 import combine_function_states, relation_path, serialize_frontier


_EXPRESSION = r"expression\d*"
_ASSIGNMENT = re.compile(rf"^\s*({_EXPRESSION})\s*=\s*(.+?)\s*$")
_START = re.compile(r"^START\('(.+)'\)$")
_JOIN = re.compile(rf"^JOIN\('(.+)'\s*,\s*({_EXPRESSION}|'[^']+')\)$")
_AND = re.compile(rf"^AND\(({_EXPRESSION})\s*,\s*({_EXPRESSION})\)$")
_STOP = re.compile(rf"^STOP\(({_EXPRESSION})\)$")


def normalize_values(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values or () if str(value).strip()}))


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


class ProgramExecutionError(RuntimeError):
    """The candidate did not produce a valid executable KG observation."""


def compile_gold_plan(function_list: Sequence[str]) -> GoldPlan:
    """Compile START/JOIN/AND/STOP into runtime-compatible expression names."""
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
        stop = _STOP.match(rhs)
        if start:
            entity = start.group(1)
            canonical_raw = f"{target} = START('{entity}')"
            statement = ProgramStatement(
                index, target, "start", (), None, canonical_raw
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
                index, target, "join", sources, relation, canonical_raw
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
                index, target, "and", sources, None, canonical_raw
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
        max_active: int = 6,
        frontier_width: int = 3,
        max_frontier_width: int = 6,
        max_turns: int = 10,
        entity_display_provider: Optional[EntityDisplayProvider] = None,
    ):
        if frontier_width < 2:
            raise ValueError("frontier_width must permit alternatives")
        if max_active < frontier_width * 2:
            raise ValueError("max_active must hold two complete relation frontiers")
        if max_frontier_width < frontier_width:
            raise ValueError("max_frontier_width cannot be smaller than frontier_width")
        if max_frontier_width > max_active:
            raise ValueError("max_frontier_width cannot exceed max_active")
        self.executor = executor
        self.candidate_provider = candidate_provider
        self.max_active = int(max_active)
        self.frontier_width = int(frontier_width)
        self.max_frontier_width = int(max_frontier_width)
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

        demos: List[HyperDemonstration] = []
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
            demo.private_metadata["max_frontier_width"] = self.max_frontier_width
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
        ranked_options = list(self.candidate_provider(query, state_before, join))[
            : self.max_frontier_width
        ]
        initial_options = ranked_options[: self.frontier_width]
        gold_option = next(
            (option for option in ranked_options if option.relation == join.relation), None
        )
        if gold_option is None:
            self.stats["proposal_max_frontier_miss"] += 1
            self.stats["proposal_miss"] += 1
            return None
        self.stats["proposal_max_frontier_hit"] += 1
        initial_relations = {option.relation for option in initial_options}
        needs_widen = gold_option.relation not in initial_relations
        if needs_widen:
            self.stats["proposal_miss"] += 1
            self.stats["proposal_recovered_by_widen"] += 1
            options = ranked_options
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
        initial_created = tuple(
            node.hypothesis_id
            for option, node in candidates
            if option.relation in initial_relations
        )
        widened_created = tuple(
            node.hypothesis_id
            for option, node in candidates
            if option.relation not in initial_relations
        )
        if (
            gold_entry is None
            or len(initial_created) < 2
            or (needs_widen and gold_entry[1].hypothesis_id not in widened_created)
        ):
            return None

        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        created = initial_created + widened_created
        source = _action_source(state_before, join.raw)
        steps = [
            DemonstrationStep(
                "Find_relation", (source,), (), initial_created,
                ("open_ranked_frontier",),
            )
        ]
        if needs_widen:
            steps.append(
                DemonstrationStep(
                    "Widen",
                    (source,),
                    initial_created,
                    widened_created,
                    ("required_relation_missing_from_initial_frontier",),
                )
            )
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
                gold_children = self._expand_terminal_frontier(
                    gold,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="continuation",
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
                active = list(created)
                required_prunes = max(
                    0,
                    len(active) - 1 + len(gold_children) - self.max_active,
                )
                rank_by_relation = {
                    option.relation: option.rank for option, _ in candidates
                }
                prunable = sorted(
                    (
                        hypotheses[node_id]
                        for node_id in active
                        if node_id != gold.hypothesis_id
                    ),
                    key=lambda node: (
                        bool(node.denotation),
                        -rank_by_relation.get(node.relation, 0),
                    ),
                )
                for victim in prunable[:required_prunes]:
                    before = tuple(active)
                    active.remove(victim.hypothesis_id)
                    rationale = (
                        ("empty_execution",)
                        if not victim.denotation
                        else (
                            ("frontier_capacity_eviction",)
                            if len(before) == self.max_active
                            else ("frontier_capacity_reservation",)
                        )
                    )
                    steps.append(
                        DemonstrationStep(
                            "Prune",
                            (victim.hypothesis_id,),
                            before,
                            (),
                            rationale,
                        )
                    )
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
                wrong_children = self._expand_terminal_frontier(
                    wrong,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="recovery_probe",
                    require_required_relation=False,
                )
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
                gold_children = self._expand_terminal_frontier(
                    gold,
                    next_join,
                    hypotheses,
                    question,
                    stat_scope="continuation",
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
                            (gold.target_expression,),
                            before_gold_expansion,
                            tuple(gold_children),
                            ("continue_preserved_alternative",),
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
                "proposal_recall_at_max_frontier": True,
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
        """Free visible frontier slots using only public capacity evidence."""
        protected_ids = set(protected)
        while len(active) > maximum_active:
            candidates = [
                hypotheses[node_id]
                for node_id in active
                if node_id not in protected_ids
            ]
            if not candidates:
                return False
            victim = min(
                candidates,
                key=lambda node: (
                    bool(node.denotation),
                    "policy_choice" in node.provenance,
                    -node.depth,
                    node.hypothesis_id,
                ),
            )
            before = tuple(active)
            active.remove(victim.hypothesis_id)
            rationale = (
                ("empty_execution",)
                if not victim.denotation
                else (
                    ("frontier_capacity_eviction",)
                    if len(before) == self.max_active
                    else ("frontier_capacity_reservation",)
                )
            )
            steps.append(
                DemonstrationStep(
                    "Prune",
                    (victim.hypothesis_id,),
                    before,
                    (),
                    rationale,
                )
            )
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
            initial_children, widened_children = self._expand_terminal_frontier_batches(
                current_gold,
                continuation,
                hypotheses,
                question,
                stat_scope="deep_continuation",
            )
            all_children = initial_children + widened_children
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
                )
            )
            if widened_children:
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
                        ("required_relation_missing_from_initial_frontier",),
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
                "proposal_recall_at_max_frontier": True,
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
    ) -> Optional[HyperDemonstration]:
        """Build ordinary two-hop progress from the same natural frontier."""
        hypotheses = {item[1].hypothesis_id: item[1] for item in candidates}
        created = tuple(hypotheses)
        gold = next(
            item[1] for item in candidates if item[0].relation == join.relation
        )
        initial_children, widened_children = self._expand_terminal_frontier_batches(
            gold,
            next_join,
            hypotheses,
            question,
            stat_scope="continuation",
        )
        gold_children = initial_children + widened_children
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
            ),
        ]
        active = [node_id for node_id in created if node_id != gold.hypothesis_id]
        active.extend(initial_children)
        family = "direct_frontier_progress"
        widen_sources = []
        if widened_children:
            required_prunes = max(
                0, len(active) + len(widened_children) - self.max_active
            )
            prunable = sorted(
                (hypotheses[node_id] for node_id in active),
                key=lambda node: (
                    bool(node.denotation),
                    -node.depth,
                    node.hypothesis_id,
                ),
            )
            for victim in prunable[:required_prunes]:
                before = tuple(active)
                active.remove(victim.hypothesis_id)
                rationale = (
                    ("empty_execution",)
                    if not victim.denotation
                    else (
                        ("frontier_capacity_eviction",)
                        if len(before) == self.max_active
                        else ("frontier_capacity_reservation",)
                    )
                )
                steps.append(
                    DemonstrationStep(
                        "Prune", (victim.hypothesis_id,), before, (), rationale,
                    )
                )
            steps.append(
                DemonstrationStep(
                    "Widen",
                    (gold.target_expression,),
                    tuple(active),
                    tuple(widened_children),
                    ("required_relation_missing_from_initial_frontier",),
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
                "proposal_recall_at_max_frontier": True,
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
    ) -> Tuple[List[str], List[str]]:
        """Execute initial and optional widened batches from one relation ranking."""
        options = list(
            self.candidate_provider(question.strip(), parent.function_state, join)
        )[: self.max_frontier_width]
        initial_options = options[: self.frontier_width]
        required = next(
            (option for option in options if option.relation == join.relation), None
        )
        self.stats[f"{stat_scope}_decisions"] += 1
        if required is None:
            self.stats[f"{stat_scope}_proposal_miss"] += 1
            self.stats[f"{stat_scope}_max_frontier_miss"] += 1
            return [], []
        self.stats[f"{stat_scope}_max_frontier_hit"] += 1
        initial_relations = {option.relation for option in initial_options}
        if required.relation in initial_relations:
            self.stats[f"{stat_scope}_proposal_hit"] += 1
        else:
            self.stats[f"{stat_scope}_proposal_miss"] += 1
            self.stats[f"{stat_scope}_recovered_by_widen"] += 1
        self.stats[f"{stat_scope}_gold_rank_{required.rank}"] += 1
        if required.rank == 1:
            self.stats[f"{stat_scope}_top1_hit"] += 1

        visible_options = (
            initial_options if required.relation in initial_relations else options
        )
        initial_children: List[str] = []
        widened_children: List[str] = []
        for option in visible_options:
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
            target_batch = (
                initial_children
                if option.relation in initial_relations
                else widened_children
            )
            target_batch.append(node_id)
        return initial_children, widened_children

    def _expand_terminal_frontier(
        self,
        parent: ExecutedHypothesis,
        join: ProgramStatement,
        hypotheses: Dict[str, ExecutedHypothesis],
        question: str,
        stat_scope: str,
        require_required_relation: bool = True,
    ) -> List[str]:
        options = list(
            self.candidate_provider(question.strip(), parent.function_state, join)
        )[
            : self.frontier_width
        ]
        self.stats[f"{stat_scope}_decisions"] += 1
        if require_required_relation:
            required = next(
                (option for option in options if option.relation == join.relation), None
            )
            if required is None:
                self.stats[f"{stat_scope}_proposal_miss"] += 1
                return []
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
        return children

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
            return None
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
            options = list(self.candidate_provider(query, left_state, left_join))[
                : self.frontier_width
            ]
            relation_set = {option.relation for option in options}
            if not {left_join.relation, right_join.relation}.issubset(relation_set):
                self.stats["conjunction_proposal_miss"] += 1
                return None
            relation_nodes = {}
            created = []
            for option in options:
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
                        if option.relation in {left_join.relation, right_join.relation}
                        else "alternative"
                    ),
                    provenance=(
                        "policy_choice" if option.rank == 1 else "ranked_alternative",
                    ),
                )
                hypotheses[node.hypothesis_id] = node
                relation_nodes[option.relation] = node
                created.append(node.hypothesis_id)
            left = relation_nodes.get(left_join.relation)
            right = relation_nodes.get(right_join.relation)
            ranks = {
                "left": next(option.rank for option in options if option.relation == left_join.relation),
                "right": next(option.rank for option in options if option.relation == right_join.relation),
            }
            frontiers.append((left_action_source, tuple(created)))
        else:
            required_nodes = []
            ranks = {}
            for label, state_before, action_source, join in (
                ("left", left_state, left_action_source, left_join),
                ("right", right_state, right_action_source, right_join),
            ):
                options = list(self.candidate_provider(query, state_before, join))[
                    : self.frontier_width
                ]
                required = next(
                    (option for option in options if option.relation == join.relation), None
                )
                if required is None:
                    self.stats["conjunction_proposal_miss"] += 1
                    return None
                ranks[label] = required.rank
                created = []
                required_node = None
                for option in options:
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
                            if option.relation == join.relation else "alternative"
                        ),
                        provenance=(
                            "policy_choice" if option.rank == 1 else "ranked_alternative",
                        ),
                    )
                    hypotheses[node.hypothesis_id] = node
                    created.append(node.hypothesis_id)
                    if option.relation == join.relation:
                        required_node = node
                if required_node is None:
                    return None
                required_nodes.append(required_node)
                frontiers.append((action_source, tuple(created)))
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
        for index, (action_source, created) in enumerate(frontiers):
            steps.append(
                DemonstrationStep(
                    "Find_relation",
                    (action_source,),
                    tuple(active),
                    created,
                    (
                        "open_shared_conjunction_frontier"
                        if len(frontiers) == 1
                        else f"open_conjunction_branch_{index + 1}"
                    ,),
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
                values = [(str(item[0]), str(item[-1])) for item in entities if item]
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
    if "frontier_capacity_eviction" in step.rationale_facts:
        return active_count is not None and max_active is not None and active_count == max_active
    if "frontier_capacity_reservation" in step.rationale_facts:
        return active_count is not None and max_active is not None and active_count < max_active
    return (
        _has_visible_path_mismatch(demo, step, node)
    )


class DemonstrationValidator:
    """Replay private executable states and verify graph-action consistency."""

    def __init__(
        self,
        executor: ProgramExecutor,
        max_active: int = 6,
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
                if step.arguments[0] not in demo.private_metadata.get("widen_sources", ()):
                    errors.append(f"Widen source {step.arguments[0]} is not an open frontier")
                if not step.created:
                    errors.append("Widen must create at least one additional hypothesis")
                active.update(step.created)
            elif step.action == "Find_relation":
                if len(step.arguments) != 1:
                    errors.append(
                        "Find_relation must expose only its source; relation ranking is environment-owned"
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
        max_active=int(demo.private_metadata.get("max_active", 6)),
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
            "The initial frontier does not cover the question, so I will inspect the next "
            "ranked batch before selecting a path."
        )
    elif step.action == "Combine":
        body = f"Combine [ {step.arguments[0]} | {step.arguments[1]} ]"
        thought = "Both active branches express required parts of the question."
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
        elif step.action == "Widen":
            active.extend(step.created)
            known.update(step.created)
            executions += len(step.created)
            event = (
                f"Widened the frontier from {step.arguments[0]} with "
                f"{len(step.created)} additional executable hypotheses."
            )
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
                int(demo.private_metadata.get("max_active", 6)),
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
