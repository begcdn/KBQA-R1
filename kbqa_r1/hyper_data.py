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

from .hyper_prompt import build_hyper_prompt, question_candidate_literals
from .hyper_r1 import (
    PruneCertificate,
    combine_function_states,
    public_empty_prune_certificate,
    public_question_contract,
    relation_path,
    serialize_frontier,
    validate_public_prune_certificate,
)
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


def logical_program_signature(
    function_list: Sequence[str], target_expression: Optional[str] = None
) -> Tuple[Any, ...]:
    """Return a variable-name-independent logical signature for one result.

    Denotation equality is not semantic equivalence: two different Freebase
    programs can happen to return the same entities.  This signature preserves
    roots, directed relations, branch structure, and logical operators while
    ignoring expression names and the order of conjunction operands.
    """
    plan = compile_gold_plan(function_list)
    target = str(target_expression or plan.target_expression)
    statements = tuple(
        statement for statement in plan.statements if statement.kind != "stop"
    )

    def definition(expression: str, before: int) -> ProgramStatement:
        match = next(
            (
                statement
                for statement in reversed(statements)
                if statement.index < before and statement.target == expression
            ),
            None,
        )
        if match is None:
            raise IneligibleProgram(
                f"logical target {expression} has no definition before {before}"
            )
        return match

    def visit(expression: str, before: int) -> Tuple[Any, ...]:
        statement = definition(expression, before)
        if statement.kind == "start":
            return ("start", statement.arguments[0])
        if statement.kind == "join":
            if statement.sources:
                source = visit(statement.sources[0], statement.index)
            else:
                source = ("literal", statement.arguments[1].strip("'"))
            return ("join", statement.relation, source)
        if statement.kind == "and":
            branches = sorted(
                (visit(source, statement.index) for source in statement.sources),
                key=repr,
            )
            return ("and", *branches)
        if statement.kind == "order":
            mode, source, relation = statement.arguments
            return ("order", mode, relation, visit(source, statement.index))
        if statement.kind == "compare":
            mode, relation, source = statement.arguments
            return ("compare", mode, relation, visit(source, statement.index))
        if statement.kind == "time_constraint":
            relation, time = statement.arguments
            return (
                "time_constraint",
                relation,
                time,
                visit(statement.sources[0], statement.index),
            )
        if statement.kind == "count":
            return ("count", visit(statement.sources[0], statement.index))
        raise IneligibleProgram(f"unsupported logical signature kind {statement.kind}")

    return visit(target, len(plan.statements) + 1)


@dataclass(frozen=True)
class _ConjunctiveQuery:
    head: Tuple[str, str]
    atoms: Tuple[Tuple[str, Tuple[str, str], Tuple[str, str]], ...]


def _signature_to_conjunctive_query(
    signature: Tuple[Any, ...]
) -> _ConjunctiveQuery:
    """Compile the supported set-valued fragment into a conjunctive query."""
    next_variable = 0
    atoms: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = []
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def variable() -> Tuple[str, str]:
        nonlocal next_variable
        value = ("var", f"v{next_variable}")
        next_variable += 1
        parent[value] = value
        return value

    def constant(value: Any) -> Tuple[str, str]:
        return ("const", str(value))

    def find(term: Tuple[str, str]) -> Tuple[str, str]:
        if term[0] == "const":
            return term
        root = parent.setdefault(term, term)
        while parent[root] != root:
            root = parent[root]
        cursor = term
        while parent[cursor] != cursor:
            following = parent[cursor]
            parent[cursor] = root
            cursor = following
        return root

    def substitute(old: Tuple[str, str], new: Tuple[str, str]) -> None:
        nonlocal atoms
        atoms = [
            (
                relation,
                new if left == old else left,
                new if right == old else right,
            )
            for relation, left, right in atoms
        ]

    def unify(left: Tuple[str, str], right: Tuple[str, str]) -> Tuple[str, str]:
        left = find(left)
        right = find(right)
        if left == right:
            return left
        if left[0] == "const" and right[0] == "const":
            raise IneligibleProgram("conjunction equates two distinct constants")
        if left[0] == "const":
            substitute(right, left)
            parent[right] = right
            return left
        if right[0] == "const":
            substitute(left, right)
            parent[left] = left
            return right
        parent[right] = left
        substitute(right, left)
        return left

    def visit(node: Tuple[Any, ...]) -> Tuple[str, str]:
        kind = node[0]
        if kind in {"start", "literal"}:
            return constant(node[1])
        if kind == "join":
            source = visit(node[2])
            output = variable()
            atoms.append((f"JOIN:{node[1]}", output, source))
            return output
        if kind == "and":
            if len(node) < 3:
                raise IneligibleProgram("AND requires two branches")
            output = visit(node[1])
            for branch in node[2:]:
                output = unify(output, visit(branch))
            return output
        if kind == "compare":
            output = variable()
            source = visit(node[3])
            atoms.append((f"COMPARE:{node[1]}:{node[2]}", output, source))
            return output
        if kind == "time_constraint":
            output = visit(node[3])
            atoms.append(
                (
                    f"TIME:{node[1]}",
                    output,
                    constant(node[2]),
                )
            )
            return output
        raise IneligibleProgram(f"{kind} is outside the conjunctive-query fragment")

    head = find(visit(signature))
    normalized_atoms = {
        (relation, find(left), find(right))
        for relation, left, right in atoms
    }
    return _ConjunctiveQuery(
        head=head,
        atoms=tuple(sorted(normalized_atoms, key=repr)),
    )


def _has_query_homomorphism(
    source: _ConjunctiveQuery, target: _ConjunctiveQuery
) -> bool:
    """Return whether ``source`` maps homomorphically into ``target``."""
    mapping: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def bind(
        source_term: Tuple[str, str], target_term: Tuple[str, str]
    ) -> Optional[Dict[Tuple[str, str], Tuple[str, str]]]:
        if source_term[0] == "const":
            return {} if source_term == target_term else None
        existing = mapping.get(source_term)
        if existing is not None:
            return {} if existing == target_term else None
        return {source_term: target_term}

    initial = bind(source.head, target.head)
    if initial is None:
        return False
    mapping.update(initial)
    target_by_relation: Dict[
        str, List[Tuple[str, Tuple[str, str], Tuple[str, str]]]
    ] = {}
    for atom in target.atoms:
        target_by_relation.setdefault(atom[0], []).append(atom)

    ordered = sorted(
        source.atoms,
        key=lambda atom: (len(target_by_relation.get(atom[0], ())), repr(atom)),
    )

    def search(index: int) -> bool:
        if index == len(ordered):
            return True
        relation, source_left, source_right = ordered[index]
        for _, target_left, target_right in target_by_relation.get(relation, ()):
            additions: Dict[Tuple[str, str], Tuple[str, str]] = {}
            valid = True
            for source_term, target_term in (
                (source_left, target_left),
                (source_right, target_right),
            ):
                if source_term[0] == "const":
                    if source_term != target_term:
                        valid = False
                        break
                    continue
                existing = mapping.get(source_term)
                if existing is not None and existing != target_term:
                    valid = False
                    break
                if existing is None:
                    pending = additions.get(source_term)
                    if pending is not None and pending != target_term:
                        valid = False
                        break
                    additions[source_term] = target_term
            if not valid:
                continue
            mapping.update(additions)
            if search(index + 1):
                return True
            for term in additions:
                mapping.pop(term, None)
        return False

    return search(0)


def _signatures_are_formally_equivalent(
    left: Tuple[Any, ...], right: Tuple[Any, ...]
) -> bool:
    if left == right:
        return True
    if left[0] == right[0] == "count":
        return _signatures_are_formally_equivalent(left[1], right[1])
    if left[0] == right[0] == "order":
        return (
            left[1:3] == right[1:3]
            and _signatures_are_formally_equivalent(left[3], right[3])
        )
    try:
        left_query = _signature_to_conjunctive_query(left)
        right_query = _signature_to_conjunctive_query(right)
    except IneligibleProgram:
        return False
    # Q1 is contained in Q2 iff there is a homomorphism from Q2 to Q1.
    return _has_query_homomorphism(left_query, right_query) and _has_query_homomorphism(
        right_query, left_query
    )


def programs_are_intent_equivalent(
    candidate_functions: Sequence[str],
    candidate_target: str,
    gold_functions: Sequence[str],
    gold_target: str,
) -> bool:
    """Conservatively prove supported-query equivalence.

    Conjunctive programs use bidirectional query containment. Aggregating or
    ordering operators must match exactly and recurse into equivalent inputs.
    Unsupported constructs fail closed.
    """
    try:
        candidate = logical_program_signature(candidate_functions, candidate_target)
        gold = logical_program_signature(gold_functions, gold_target)
        return _signatures_are_formally_equivalent(candidate, gold)
    except IneligibleProgram:
        return False


@dataclass(frozen=True)
class ProgramCommitCertificate:
    answer_exact: bool
    intent_equivalent: bool

    @property
    def valid(self) -> bool:
        return self.answer_exact and self.intent_equivalent


def certify_program_commit(
    candidate_functions: Sequence[str],
    candidate_target: str,
    candidate_answers: Sequence[str],
    gold_functions: Sequence[str],
    gold_target: str,
    gold_answers: Sequence[str],
) -> ProgramCommitCertificate:
    """Require both denotational correctness and formal query equivalence."""
    return ProgramCommitCertificate(
        answer_exact=(
            normalize_values(candidate_answers) == normalize_values(gold_answers)
        ),
        intent_equivalent=programs_are_intent_equivalent(
            candidate_functions,
            candidate_target,
            gold_functions,
            gold_target,
        ),
    )


@dataclass(frozen=True)
class RelationOption:
    relation: str
    score: float
    rank: int


@dataclass(frozen=True)
class RelationProposal:
    proposal_id: str
    frontier_id: str
    source: str
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
    supervision: str = "policy_target"
    certificate_kind: Optional[str] = None
    certificate_evidence: Tuple[str, ...] = ()
    exposed: Tuple[str, ...] = ()


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
    proposals: Dict[str, RelationProposal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "question_id": self.question_id,
            "question": self.question,
            "family": self.family,
            "hypotheses": {key: asdict(value) for key, value in self.hypotheses.items()},
            "steps": [asdict(step) for step in self.steps],
            "gold_answers": list(self.gold_answers),
            "proposals": {key: asdict(value) for key, value in self.proposals.items()},
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


def replace_join_relation_and_source(
    raw: str, relation: str, source_expression: str
) -> str:
    """Apply a relation to the selected runtime hypothesis expression."""
    match = _ASSIGNMENT.match(raw)
    if not match or not _JOIN.match(match.group(2)):
        raise ValueError(f"not a JOIN statement: {raw}")
    target = match.group(1)
    return f"{target} = JOIN('{relation}', {source_expression})"


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
        max_nodes: int = 128,
        max_execution_attempts: int = 24,
        frontier_width: int = 6,
        max_turns: int = 32,
        entity_display_provider: Optional[EntityDisplayProvider] = None,
    ):
        if frontier_width < 2:
            raise ValueError("frontier_width must permit alternatives")
        if max_active < 2:
            raise ValueError("max_active must permit competing hypotheses")
        if max_nodes < max_active:
            raise ValueError("max_nodes must be at least max_active")
        if max_execution_attempts < 1:
            raise ValueError("max_execution_attempts must be positive")
        self.executor = executor
        self.candidate_provider = candidate_provider
        self.max_active = int(max_active)
        self.max_nodes = int(max_nodes)
        self.max_execution_attempts = int(max_execution_attempts)
        self.frontier_width = int(frontier_width)
        self.max_turns = int(max_turns)
        self.entity_display_provider = entity_display_provider
        self._display_cache: Dict[str, str] = {}
        if self.max_turns < 1:
            raise ValueError("max_turns must permit at least one executable policy action")
        self.stats: Counter = Counter()
        self._execution_cache: Dict[
            Tuple[Tuple[str, ...], str], Optional[Tuple[str, ...]]
        ] = {}
        self._proposal_cache: Dict[
            Tuple[str, Tuple[str, ...], str], Tuple[RelationOption, ...]
        ] = {}
        self._proposal_state_cache: Dict[
            Tuple[str, Tuple[str, ...]], Tuple[RelationOption, ...]
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

    def _ranked_options(
        self,
        question: str,
        state_before: Sequence[str],
        decision: ProgramStatement,
    ) -> Tuple[RelationOption, ...]:
        """Cache the inference-time proposal frontier for one visible state."""
        key = (
            str(question).strip(),
            tuple(str(item) for item in state_before),
            str(decision.raw),
        )
        if key not in self._proposal_cache:
            self._proposal_cache[key] = tuple(
                self.candidate_provider(key[0], key[1], decision)
            )
        self._proposal_state_cache[(key[0], key[1])] = self._proposal_cache[key]
        return self._proposal_cache[key]

    def _audit_gold_program_proposals(
        self, question: str, plan: GoldPlan
    ) -> Dict[str, Any]:
        """Measure exact gold-program reachability under the lazy runtime budgets."""
        decisions = []
        for join in plan.joins:
            source = join.sources[0] if join.sources else _join_source(join.raw)
            state_before = self._dependency_state(plan, source, before=join.index)
            options = self._ranked_options(question, state_before, join)
            position = next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.relation == join.relation
                ),
                None,
            )
            pages_required = (
                position // self.frontier_width + 1
                if position is not None
                else None
            )
            decisions.append(
                {
                    "decision_index": join.index,
                    "relation": join.relation,
                    "rank": options[position].rank if position is not None else None,
                    "position": position + 1 if position is not None else None,
                    "present": position is not None,
                    "pages_required": pages_required,
                    "catalog_actions": pages_required,
                    "inspection_actions": 1 if position is not None else None,
                    "within_budget": bool(
                        position is not None
                        and (pages_required or 0) + 2 <= self.max_turns
                    ),
                }
            )

        # Programs rooted directly in an ontology type can begin with an
        # operator such as ARGMAX and require no relation frontier at all.
        # In that case every required relation is present vacuously; the
        # operator simulation and runtime budgets below decide reachability.
        all_present = all(row["present"] for row in decisions)
        decision_by_index = {
            join.index: row for join, row in zip(plan.joins, decisions)
        }
        expression_tokens: Dict[str, Optional[str]] = {}
        active_tokens: set[str] = set()
        next_token = 0
        actions = 0
        execution_attempts = 0
        nodes = 0
        peak_active = 0
        supported = True

        def new_token(target: str) -> str:
            nonlocal next_token, nodes, peak_active
            token = f"N{next_token}"
            next_token += 1
            nodes += 1
            active_tokens.add(token)
            expression_tokens[target] = token
            peak_active = max(peak_active, len(active_tokens))
            return token

        for statement in plan.statements:
            if statement.kind in {"stop"}:
                continue
            if statement.kind == "start":
                previous = expression_tokens.get(statement.target)
                if previous in active_tokens:
                    active_tokens.remove(previous)
                    actions += 1  # Park an overwritten unresolved branch.
                expression_tokens[statement.target] = None
                continue
            if statement.kind == "join":
                decision = decision_by_index[statement.index]
                if not decision["present"]:
                    supported = False
                    continue
                parent = (
                    expression_tokens.get(statement.sources[0])
                    if statement.sources
                    else None
                )
                if parent is not None:
                    actions += 1  # Select
                actions += int(decision["pages_required"]) + 1  # Find/Widen + Inspect
                execution_attempts += 1
                child = new_token(statement.target)
                if parent is not None and parent in active_tokens:
                    active_tokens.remove(parent)
                    actions += 1  # Park after opening the child catalog.
                active_tokens.add(child)
                peak_active = max(peak_active, len(active_tokens))
                continue
            if statement.kind == "and":
                parents = [expression_tokens.get(source) for source in statement.sources]
                executable_parents = [value for value in parents if value is not None]
                if len(executable_parents) == 2:
                    actions += 1  # Combine
                elif len(executable_parents) == 1:
                    actions += 2  # Select + ontology-type Merge
                else:
                    supported = False
                    continue
                execution_attempts += 1
                active_tokens.difference_update(executable_parents)
                new_token(statement.target)
                continue
            if statement.kind in {"order", "compare", "time_constraint", "count"}:
                parent = expression_tokens.get(statement.sources[0])
                if parent is not None:
                    actions += 1  # Select
                elif statement.kind not in {"order", "compare"}:
                    supported = False
                    continue
                actions += 1
                execution_attempts += 1
                if parent is not None:
                    active_tokens.discard(parent)
                new_token(statement.target)
                continue
            supported = False

        target = expression_tokens.get(plan.target_expression)
        if target is None:
            supported = False
        else:
            actions += 1  # Commit
        # The runtime grants ``max_turns`` executable policy decisions and, if
        # the rollout is still active, generates the answer in one separate
        # answer-only turn.  Keep both numbers explicit so the oracle ceiling
        # cannot silently reserve one fewer action than inference receives.
        turns_with_answer = actions + 1
        budget_checks = {
            "turns": actions <= self.max_turns,
            "execution_attempts": execution_attempts <= self.max_execution_attempts,
            "stored_nodes": nodes <= self.max_nodes,
            "active_workspace": peak_active <= self.max_active,
        }
        runtime_reachable = all_present and supported and all(budget_checks.values())
        self.stats["gold_program_proposal_audit_rows"] += 1
        self.stats["gold_program_proposal_decisions"] += len(decisions)
        self.stats["gold_program_all_relations_present"] += int(all_present)
        self.stats["gold_program_all_relations_within_budget"] += int(
            runtime_reachable
        )
        self.stats["gold_program_runtime_reachable"] += int(runtime_reachable)
        self.stats["gold_program_relation_present"] += sum(
            row["present"] for row in decisions
        )
        self.stats["gold_program_relation_within_budget"] += sum(
            row["within_budget"] for row in decisions
        )
        for name, passed in budget_checks.items():
            self.stats[f"gold_program_{name}_budget_hit"] += int(passed)
        return {
            "decisions": decisions,
            "all_relations_present": all_present,
            # Backward-compatible key now means the complete lazy protocol,
            # not a relation-rank cutoff tied to active-node capacity.
            "all_relations_within_budget": runtime_reachable,
            "runtime_reachable": runtime_reachable,
            "supported_program": supported,
            "required_actions": actions,
            "required_turns_with_answer": turns_with_answer,
            "required_execution_attempts": execution_attempts,
            "required_stored_nodes": nodes,
            "required_peak_active": peak_active,
            "budget_checks": budget_checks,
        }

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
        public_contract = public_question_contract(question)
        gold_has_count = any(statement.kind == "count" for statement in plan.statements)
        self.stats["public_count_contract_rows"] += 1
        self.stats["gold_count_rows"] += int(gold_has_count)
        self.stats["public_count_required_rows"] += int(
            public_contract.count_required is True
        )
        if gold_has_count and public_contract.count_required is not True:
            # A teacher action that runtime cannot license from public input is
            # ineligible rather than silently relying on private gold syntax.
            self.stats["public_count_contract_false_negative"] += 1
            return []
        if not gold_has_count and public_contract.count_required is True:
            # Conservative false positives only preserve more empty branches;
            # report them because they reduce useful Prune supervision.
            self.stats["public_count_contract_false_positive"] += 1
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
        proposal_audit = self._audit_gold_program_proposals(question, plan)
        intersection_demos: List[HyperDemonstration] = []
        semantic_branch_indexes = set()
        terminal_type = self._terminal_type_constraint(plan)
        has_bare_type = self._has_bare_type_intersection(plan)
        if plan.operators or (has_bare_type and terminal_type is None):
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
        if terminal_type is not None and not plan.operators:
            constrained_demos = []
            for demo in demos:
                constrained = self._append_terminal_type_constraint(
                    demo, plan, gold_answers
                )
                if constrained is not None:
                    constrained_demos.append(constrained)
            demos = constrained_demos
        certified_demos = []
        for demo in demos:
            certified = self._certify_irreversible_actions(
                demo, plan, gold_answers
            )
            if certified is None:
                continue
            runtime_demo = self._to_lazy_runtime_demo(certified)
            if runtime_demo is not None:
                certified_demos.append(runtime_demo)
        demos = certified_demos
        within_budget = []
        for demo in demos:
            if len(demo.hypotheses) > self.max_nodes:
                self.stats["trajectory_node_budget_miss"] += 1
                continue
            # ``max_turns`` counts graph actions.  The runtime emits the final
            # answer in a distinct, answer-only generation afterwards.
            if len(demo.steps) > self.max_turns:
                self.stats["trajectory_turn_budget_miss"] += 1
                continue
            within_budget.append(demo)
        demos = within_budget
        candidate_entities, root_provenance = self._row_candidate_entities(
            row, plan, question_id
        )
        candidate_literals = question_candidate_literals(question)
        requires_entity_root = any(
            statement.kind == "start"
            and statement.arguments
            and statement.arguments[0].startswith(("m.", "g."))
            for statement in plan.statements
        )
        if requires_entity_root and not candidate_entities:
            self.stats["public_root_entity_missing"] += 1
            return []
        public_sources = {
            str(source[-1])
            for source in (*candidate_entities, *candidate_literals)
            if source
        }
        public_demos = []
        for demo in demos:
            unavailable = [
                step.arguments[0]
                for step in demo.steps
                if step.action == "Find_relation"
                and step.arguments
                and not step.arguments[0].startswith("expression")
                and step.arguments[0] not in public_sources
            ]
            if unavailable:
                self.stats["public_source_proposal_miss"] += 1
                continue
            public_demos.append(demo)
        demos = public_demos
        base_prompt = self._row_base_prompt(row)
        for demo in demos:
            demo.private_metadata["candidate_entities"] = candidate_entities
            demo.private_metadata["root_entity_provenance"] = root_provenance
            demo.private_metadata["candidate_literals"] = candidate_literals
            demo.private_metadata["candidate_entity_order"] = "stable_question_hash"
            demo.private_metadata["base_prompt"] = base_prompt
            demo.private_metadata["max_active"] = self.max_active
            demo.private_metadata["max_nodes"] = self.max_nodes
            demo.private_metadata["max_execution_attempts"] = (
                self.max_execution_attempts
            )
            demo.private_metadata["relation_page_size"] = self.frontier_width
            demo.private_metadata["relation_rank_cutoff"] = None
            demo.private_metadata["max_turns"] = self.max_turns
            demo.private_metadata["gold_program_proposal_audit"] = proposal_audit
        return demos

    def _certify_irreversible_actions(
        self,
        demo: HyperDemonstration,
        plan: GoldPlan,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        """Attach independently recomputable proofs to Prune and Commit.

        Gold may certify a teacher target, but it must never be converted into
        fabricated public evidence.  Unsupported nonempty branches therefore
        remain unresolved, and answer equality alone is insufficient to Commit.
        """
        certified_steps: List[DemonstrationStep] = []
        contract = public_question_contract(demo.question)
        for step in demo.steps:
            if step.action == "Prune":
                node = demo.hypotheses[step.arguments[0]]
                if node.denotation:
                    self.stats["uncertified_nonempty_prune_rejected"] += 1
                    return None
                certificate = public_empty_prune_certificate(
                    node.hypothesis_id,
                    node.denotation,
                    contract,
                )
                if certificate is None:
                    self.stats["uncertified_empty_before_count_rejected"] += 1
                    return None
                step = replace(
                    step,
                    rationale_facts=("empty_execution", "empty_is_terminal"),
                    certificate_kind=certificate.kind,
                    certificate_evidence=certificate.evidence,
                )
                self.stats["certified_empty_prune"] += 1
            elif step.action == "Commit":
                node = demo.hypotheses[step.arguments[0]]
                commit_certificate = certify_program_commit(
                    node.function_state,
                    node.target_expression,
                    node.denotation,
                    plan.executable_functions,
                    plan.target_expression,
                    gold_answers,
                )
                if not commit_certificate.answer_exact:
                    self.stats["commit_answer_certificate_miss"] += 1
                    return None
                if not commit_certificate.intent_equivalent:
                    self.stats["commit_intent_certificate_miss"] += 1
                    return None
                step = replace(
                    step,
                    rationale_facts=(
                        "answer_exact",
                        "gold_program_formally_equivalent",
                    ),
                    certificate_kind="answer_and_supported_query_equivalent",
                    certificate_evidence=(
                        "denotation:exact",
                        "supported_query:bidirectional_containment",
                    ),
                )
                self.stats["certified_commit"] += 1
            certified_steps.append(step)

        demo.steps = certified_steps
        demo.private_metadata["gold_program"] = list(plan.executable_functions)
        demo.private_metadata["gold_target_expression"] = plan.target_expression
        demo.private_metadata["gold_intent_signature"] = logical_program_signature(
            plan.executable_functions, plan.target_expression
        )
        demo.private_metadata["irreversible_action_contract"] = (
            "proof_carrying_semantic_storage_split_v2"
        )
        return demo

    def _to_lazy_runtime_demo(
        self, demo: HyperDemonstration
    ) -> Optional[HyperDemonstration]:
        """Compile an eager oracle trace into the exact lazy runtime protocol.

        Eager construction is useful for verifying candidate outcomes, but it
        is not a policy interface.  At runtime Find/Widen expose symbolic
        proposals, Inspect executes one proposal, and Park changes storage
        without asserting that a branch is false.  This compiler is the only
        boundary at which verified oracle traces become student trajectories.
        """
        referenced = {
            argument
            for step in demo.steps
            for argument in step.arguments
            if re.fullmatch(r"H\d+", str(argument))
        }
        required = set(referenced)
        changed = True
        while changed:
            changed = False
            for node_id in tuple(required):
                node = demo.hypotheses.get(node_id)
                if node is None:
                    continue
                parents = tuple(node.parent_ids) or (
                    (node.parent_id,) if node.parent_id else ()
                )
                for parent_id in parents:
                    if parent_id not in required:
                        required.add(parent_id)
                        changed = True

        proposals: Dict[str, RelationProposal] = {}
        frontiers: List[Dict[str, Any]] = []
        steps: List[DemonstrationStep] = []
        active: set[str] = set(demo.steps[0].visible_before if demo.steps else ())
        parked: set[str] = set()
        known: set[str] = set(active)
        selected: Optional[str] = None
        next_proposal = 0
        execution_attempts = 0

        def append_park(node_id: str, reason: str) -> None:
            nonlocal selected
            if node_id not in active:
                return
            steps.append(
                DemonstrationStep(
                    "Park",
                    (node_id,),
                    tuple(sorted(active)),
                    (),
                    (reason, "storage_only_not_semantic_rejection"),
                )
            )
            active.remove(node_id)
            parked.add(node_id)
            if selected == node_id:
                selected = None

        def make_active(node_id: str, protected: Sequence[str] = ()) -> bool:
            if node_id in active:
                return True
            if node_id not in parked:
                return False
            if len(active) >= self.max_active:
                candidates = sorted(active.difference(protected))
                if not candidates:
                    return False
                append_park(candidates[0], "free_visible_workspace_for_recall")
            steps.append(
                DemonstrationStep(
                    "Recall",
                    (node_id,),
                    tuple(sorted(active)),
                    (),
                    ("resume_preserved_hypothesis",),
                )
            )
            parked.remove(node_id)
            active.add(node_id)
            return True

        def page_options(step: DemonstrationStep) -> Optional[Tuple[RelationOption, ...]]:
            if not step.created:
                return None
            child = demo.hypotheses.get(step.created[0])
            if child is None or child.relation is None:
                return None
            if child.parent_id:
                parent = demo.hypotheses.get(child.parent_id)
                if parent is None:
                    return None
                state_before = tuple(parent.function_state)
            else:
                state_before = tuple(child.function_state[:-1])
            ranked = self._proposal_state_cache.get((demo.question, state_before))
            if ranked is None:
                return None
            if len(step.relation_page) == 3:
                start, stop, total = step.relation_page
                if total != len(ranked) or not (0 <= start < stop <= total):
                    return None
            else:
                start = 0
                stop = min(self.frontier_width, len(ranked))
            return tuple(ranked[start:stop])

        def inspect_page(
            eager: DemonstrationStep,
            frontier: Dict[str, Any],
            exposed_ids: Sequence[str],
        ) -> bool:
            nonlocal execution_attempts
            children = [
                demo.hypotheses[node_id]
                for node_id in eager.created
                if node_id in demo.hypotheses
            ]
            chosen = [child for child in children if child.hypothesis_id in required]
            # Preserve one strong executable competitor when the gold-derived
            # trace otherwise uses only one branch from this catalog.
            if len(frontier["inspected"]) + len(chosen) < 2:
                alternative = next(
                    (
                        child
                        for child in children
                        if child.hypothesis_id not in {item.hypothesis_id for item in chosen}
                        and child.denotation
                    ),
                    None,
                )
                if alternative is not None:
                    chosen.append(alternative)
            relation_to_proposal = {
                proposals[proposal_id].relation: proposal_id
                for proposal_id in exposed_ids
            }
            for child in chosen:
                if child.hypothesis_id in known:
                    continue
                proposal_id = relation_to_proposal.get(child.relation or "")
                if proposal_id is None:
                    return False
                if execution_attempts >= self.max_execution_attempts:
                    return False
                if len(known) >= self.max_nodes:
                    return False
                if len(active) >= self.max_active:
                    candidates = sorted(active.difference({child.parent_id or ""}))
                    if not candidates:
                        return False
                    append_park(candidates[0], "free_visible_workspace_for_inspection")
                steps.append(
                    DemonstrationStep(
                        "Inspect",
                        (proposal_id,),
                        tuple(sorted(active)),
                        (child.hypothesis_id,),
                        (
                            "execute_visible_ranked_proposal",
                            f"relation_rank:{proposals[proposal_id].rank}",
                        ),
                    )
                )
                execution_attempts += 1
                known.add(child.hypothesis_id)
                active.add(child.hypothesis_id)
                frontier["inspected"].add(child.hypothesis_id)
            return True

        for eager in demo.steps:
            if eager.action in {"Find_relation", "Widen"}:
                options = page_options(eager)
                if not options:
                    self.stats["lazy_protocol_catalog_reconstruction_miss"] += 1
                    return None
                source = eager.arguments[0]
                if eager.action == "Find_relation":
                    frontier = {
                        "frontier_id": f"F{len(frontiers)}",
                        "source": source,
                        "inspected": set(),
                    }
                    frontiers.append(frontier)
                else:
                    frontier = next(
                        (
                            item
                            for item in reversed(frontiers)
                            if item["source"] == source
                        ),
                        None,
                    )
                    if frontier is None or selected is not None:
                        return None
                exposed_ids = []
                for option in options:
                    proposal_id = f"P{next_proposal}"
                    next_proposal += 1
                    proposals[proposal_id] = RelationProposal(
                        proposal_id=proposal_id,
                        frontier_id=frontier["frontier_id"],
                        source=source,
                        relation=option.relation,
                        score=float(option.score),
                        rank=int(option.rank),
                    )
                    exposed_ids.append(proposal_id)
                steps.append(
                    replace(
                        eager,
                        visible_before=tuple(sorted(active)),
                        created=(),
                        exposed=tuple(exposed_ids),
                    )
                )
                parent_id = selected if eager.action == "Find_relation" else None
                selected = None
                if parent_id is not None:
                    append_park(parent_id, "preserve_parent_after_opening_child_catalog")
                if not inspect_page(eager, frontier, exposed_ids):
                    self.stats["lazy_protocol_inspection_budget_miss"] += 1
                    return None
                continue

            if eager.action == "Select":
                node_id = eager.arguments[0]
                if not make_active(node_id):
                    return None
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                selected = node_id
                continue

            if eager.action == "Park":
                append_park(eager.arguments[0], "teacher_requested_storage_move")
                continue

            if eager.action == "Recall":
                if not make_active(eager.arguments[0]):
                    return None
                continue

            if eager.action == "Prune":
                node_id = eager.arguments[0]
                if not make_active(node_id):
                    return None
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                active.remove(node_id)
                if selected == node_id:
                    selected = None
                continue

            if eager.action == "Combine":
                left, right = eager.arguments
                if not make_active(left, (right,)) or not make_active(right, (left,)):
                    return None
                if execution_attempts >= self.max_execution_attempts:
                    return None
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                active.difference_update((left, right))
                active.update(eager.created)
                known.update(eager.created)
                execution_attempts += 1
                selected = None
                continue

            if eager.action in {"Merge", "Order", "Compare", "Time_constraint", "Count"}:
                if execution_attempts >= self.max_execution_attempts:
                    return None
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                if selected is not None:
                    active.discard(selected)
                active.update(eager.created)
                known.update(eager.created)
                execution_attempts += 1
                selected = None
                continue

            if eager.action == "Commit":
                node_id = eager.arguments[0]
                if not make_active(node_id):
                    return None
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                active = {node_id}
                parked.clear()
                selected = None
                continue

            if eager.action == "Abstain":
                steps.append(replace(eager, visible_before=tuple(sorted(active))))
                active.clear()
                parked.clear()
                selected = None
                continue

            return None

        if not steps or steps[-1].action not in {"Commit", "Abstain"}:
            return None
        demo.steps = steps
        demo.proposals = proposals
        demo.hypotheses = {
            node_id: node
            for node_id, node in demo.hypotheses.items()
            if node_id in known
        }
        demo.private_metadata["runtime_protocol"] = "lazy_relation_inspection_v1"
        demo.private_metadata["execution_attempts"] = execution_attempts
        demo.private_metadata["max_execution_attempts"] = self.max_execution_attempts
        return demo

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
        recovery_outcome = None

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
                    self._ranked_options(question, state_before, statement)
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
                if parent is None:
                    return None
                recovered_here = None
                if recovery_outcome is None:
                    sibling_anchor = parent
                    while (
                        sibling_anchor.role == "type_constrained"
                        and sibling_anchor.parent_id is not None
                        and sibling_anchor.parent_id in hypotheses
                    ):
                        sibling_anchor = hypotheses[sibling_anchor.parent_id]
                    recovered_here = self._append_natural_recovery_probe(
                        question=question,
                        required_parent=parent,
                        sibling_anchor=sibling_anchor,
                        continuation=None,
                        hypotheses=hypotheses,
                        active=active,
                        steps=steps,
                        stat_scope="operator_adjacent_recovery",
                    )
                    if recovered_here is not None:
                        recovery_outcome = recovered_here
                if recovered_here is None and not select_parent(
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
                "probe_outcome": recovery_outcome,
                "recovery_stratum": (
                    "operator_adjacent" if recovery_outcome else None
                ),
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
        ranked_options = list(self._ranked_options(query, state_before, join))
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

        terminal_gold = (
            join.target == plan.target_expression
            and self._matches_terminal_answers(plan, gold, gold_answers)
        )
        recovery_stratum = None
        probe_outcome = None
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
                    plan=plan,
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
                        and self._matches_terminal_answers(
                            plan, hypotheses[node_id], gold_answers
                        )
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
                probe_outcome = self._append_natural_recovery_probe(
                    question=question,
                    required_parent=gold,
                    continuation=next_join,
                    hypotheses=hypotheses,
                    active=active,
                    steps=steps,
                    stat_scope="post_widen_recovery",
                )
                if probe_outcome is None:
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
                if probe_outcome is None:
                    family = "adaptive_frontier_widen"
                    probe_relation = None
                else:
                    family = (
                        "certified_empty_recovery"
                        if probe_outcome == "proved_false_empty"
                        else "non_destructive_nonempty_recovery"
                    )
                    probe_relation = "question_conditioned"
                    recovery_stratum = "post_widen"
                    self.stats["recovery_built"] += 1
            else:
                # A higher-ranked nonempty alternative is a useful recovery
                # intervention state.  Do not call it wrong by forcing the gold
                # suffix onto it: its own natural continuation may still be
                # useful, equivalent, or simply unresolved.
                wrong_entries = []
                for option, node in candidates:
                    if (
                        option.relation == join.relation
                        or option.rank >= gold_option.rank
                        or not node.denotation
                    ):
                        continue
                    wrong_entries.append((option, node))
                if not wrong_entries:
                    direct = self._build_direct_progress_demo(
                        question_id,
                        question,
                        plan,
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
                        supervision="intervention",
                    )
                )
                self.stats["recovery_intervention_select"] += 1
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
                        plan,
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
                prune_ids = [
                    node_id
                    for node_id in wrong_children
                    if not hypotheses[node_id].denotation
                ]
                top_probe_answer_exact = self._matches_terminal_answers(
                    plan, top_probe, gold_answers
                )
                top_probe_intent_exact = programs_are_intent_equivalent(
                    top_probe.function_state,
                    top_probe.target_expression,
                    plan.executable_functions,
                    plan.target_expression,
                )
                if top_probe_answer_exact and not top_probe_intent_exact:
                    self.stats["recovery_answer_exact_without_intent_proof"] += 1
                if top_probe.denotation:
                    self.stats["recovery_probe_unresolved_nonempty"] += 1
                else:
                    self.stats["recovery_probe_visible_empty"] += 1
                for child_id in prune_ids:
                    before = tuple(active)
                    active.remove(child_id)
                    steps.append(
                        DemonstrationStep(
                            "Prune",
                            (child_id,),
                            before,
                            (),
                            ("empty_execution", "empty_is_terminal"),
                        )
                    )
                steps.append(
                    DemonstrationStep(
                        "Select", (gold.hypothesis_id,), tuple(active), (),
                        (
                            "return_after_certified_failure"
                            if not top_probe.denotation
                            else "switch_while_preserving_unresolved_probe",
                        ),
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
                        and self._matches_terminal_answers(
                            plan, hypotheses[node_id], gold_answers
                        )
                    ),
                    None,
                )
                if final_gold is None:
                    return None
                before_gold_expansion = tuple(active)
                active.remove(gold.hypothesis_id)
                active.extend(gold_children)
                if len(active) > self.max_active:
                    self.stats["non_destructive_recovery_budget_miss"] += 1
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
                    "certified_empty_recovery"
                    if not top_probe.denotation
                    else "non_destructive_nonempty_recovery"
                )
                recovery_stratum = "immediate_linear"
                probe_outcome = (
                    "proved_false_empty"
                    if not top_probe.denotation
                    else "unresolved_nonempty"
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
                "probe_outcome": probe_outcome,
                "recovery_stratum": recovery_stratum,
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

    @staticmethod
    def _natural_probe_statement(parent: ExecutedHypothesis) -> ProgramStatement:
        """Create a structurally legal, question-ranked continuation slot.

        The placeholder relation is never executed or shown to the student.
        The normal candidate provider fills the slot from the public question
        and executable parent state, exactly as it does during inference.
        """
        numbers = [
            int(number)
            for raw in parent.function_state
            for number in re.findall(r"expression(\d+)", raw)
        ]
        target = f"expression{max(numbers, default=0) + 1}"
        raw = f"{target} = JOIN('__question_conditioned_probe__', {parent.target_expression})"
        return ProgramStatement(
            index=-1,
            target=target,
            kind="join",
            sources=(parent.target_expression,),
            relation="__question_conditioned_probe__",
            raw=raw,
        )

    def _append_natural_recovery_probe(
        self,
        *,
        question: str,
        required_parent: ExecutedHypothesis,
        continuation: Optional[ProgramStatement],
        hypotheses: Dict[str, ExecutedHypothesis],
        active: List[str],
        steps: List[DemonstrationStep],
        stat_scope: str,
        sibling_anchor: Optional[ExecutedHypothesis] = None,
    ) -> Optional[str]:
        """Test a naturally preferred sibling, preserve uncertainty, and recover.

        Gold identifies the branch the teacher must eventually resume, but it
        does not fabricate the competing action or its outcome. The competing
        sibling and continuation both come from the inference-time ranker and
        are executed against the graph. Empty results may be pruned only under
        the public Count contract; nonempty results are parked, never called
        false merely because they differ from gold.
        """
        # Unary semantic transforms such as a type constraint replace the
        # required relation node in active memory. Their alternatives remain
        # siblings of the pre-transform node, not of the transformed result.
        # Keep the resume target separate from the node that identifies the
        # naturally competing relation frontier.
        anchor = sibling_anchor or required_parent
        siblings = [
            node
            for node_id in active
            if (node := hypotheses.get(node_id)) is not None
            and node.hypothesis_id != anchor.hypothesis_id
            and node.parent_id == anchor.parent_id
            and node.denotation
            and node.role
            not in {
                "gold",
                "required_branch",
                "required_program_branch",
                "combined",
                "type_constrained",
            }
            and "policy_choice" in node.provenance
        ]
        if not siblings:
            return None
        sibling = sorted(siblings, key=lambda node: node.hypothesis_id)[0]
        if len(steps) + 4 > self.max_turns:
            self.stats[f"{stat_scope}_turn_budget_miss"] += 1
            return None

        trial_hypotheses = dict(hypotheses)
        probe_statement = continuation or self._natural_probe_statement(sibling)
        probe_page = self._expand_terminal_frontier(
            sibling,
            probe_statement,
            trial_hypotheses,
            question,
            stat_scope=stat_scope,
            require_required_relation=False,
        )
        if probe_page is None:
            return None
        probe_children = [
            trial_hypotheses[node_id]
            for node_id in probe_page.node_ids
            if node_id in trial_hypotheses
        ]
        if not probe_children:
            return None
        probe = next(
            (
                node
                for node in probe_children
                if "policy_choice" in node.provenance
            ),
            probe_children[0],
        )
        if len(trial_hypotheses) > self.max_nodes:
            self.stats[f"{stat_scope}_node_budget_miss"] += 1
            return None

        trial_active = list(active)
        trial_steps = [
            DemonstrationStep(
                "Select",
                (sibling.hypothesis_id,),
                tuple(trial_active),
                (),
                ("plausible_higher_ranked_branch_requires_evidence",),
                supervision="intervention",
            )
        ]
        before_probe = tuple(trial_active)
        trial_active.remove(sibling.hypothesis_id)
        # Eager construction executes the page for verification, while the
        # lazy runtime materializes only the inspected proposal as a node.
        trial_active.append(probe.hypothesis_id)
        trial_steps.append(
            DemonstrationStep(
                "Find_relation",
                (sibling.target_expression,),
                before_probe,
                probe_page.node_ids,
                ("test_question_conditioned_continuation",),
                (probe_page.start, probe_page.stop, probe_page.total),
            )
        )

        can_prune = public_empty_prune_certificate(
            probe.hypothesis_id,
            probe.denotation,
            public_question_contract(question),
        ) is not None
        if can_prune:
            trial_steps.append(
                DemonstrationStep(
                    "Prune",
                    (probe.hypothesis_id,),
                    tuple(trial_active),
                    (),
                    ("empty_execution", "empty_is_terminal"),
                )
            )
            outcome = "proved_false_empty"
        else:
            trial_steps.append(
                DemonstrationStep(
                    "Park",
                    (probe.hypothesis_id,),
                    tuple(trial_active),
                    (),
                    (
                        "preserve_unresolved_probe",
                        "storage_only_not_semantic_rejection",
                    ),
                )
            )
            outcome = (
                "empty_preserved_for_count"
                if not probe.denotation
                else "unresolved_nonempty"
            )
        trial_active.remove(probe.hypothesis_id)
        trial_steps.append(
            DemonstrationStep(
                "Select",
                (required_parent.hypothesis_id,),
                tuple(trial_active),
                (),
                ("resume_gold_supported_hypothesis_after_probe",),
            )
        )

        hypotheses.clear()
        hypotheses.update(trial_hypotheses)
        active[:] = trial_active
        steps.extend(trial_steps)
        self.stats[f"{stat_scope}_built"] += 1
        self.stats[f"{stat_scope}_{outcome}"] += 1
        return outcome

    def _build_deep_progress_demo(
        self,
        *,
        question_id: str,
        question: str,
        plan: GoldPlan,
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
        recovery_outcome = None

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
            recovered_here = None
            if recovery_outcome is None and current_gold.depth > 0:
                recovered_here = self._append_natural_recovery_probe(
                    question=question,
                    required_parent=current_gold,
                    continuation=continuation,
                    hypotheses=hypotheses,
                    active=active,
                    steps=steps,
                    stat_scope="deep_recovery",
                )
                if recovered_here is not None:
                    recovery_outcome = recovered_here
            if recovered_here is None:
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

        if not self._matches_terminal_answers(plan, current_gold, gold_answers):
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
                "probe_outcome": recovery_outcome,
                "recovery_stratum": "deep" if recovery_outcome else None,
                "decision_index": join.index,
                "path_hops": 1 + len(following_joins),
            },
        )

    def _build_direct_progress_demo(
        self,
        question_id: str,
        question: str,
        plan: GoldPlan,
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
                and self._matches_terminal_answers(
                    plan, hypotheses[node_id], gold_answers
                )
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
            self._ranked_options(question, parent.function_state, join)
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
                statement = replace_join_relation_and_source(
                    join.raw, option.relation, parent.target_expression
                )
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
            self._ranked_options(question, parent.function_state, join)
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
            statement = replace_join_relation_and_source(
                join.raw, option.relation, parent.target_expression
            )
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
            options = list(self._ranked_options(query, left_state, left_join))
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
                options = list(self._ranked_options(query, state_before, join))
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
            reference_values != combined_values
            or combined_values
            != normalize_values(set(left.denotation) & set(right.denotation))
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
        if not self._matches_terminal_answers(plan, combined, gold_answers):
            return None
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
        recovery_outcome = self._append_natural_recovery_probe(
            question=question,
            required_parent=left,
            continuation=None,
            hypotheses=hypotheses,
            active=active,
            steps=steps,
            stat_scope="conjunction_recovery",
        )
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
                "probe_outcome": recovery_outcome,
                "recovery_stratum": "conjunction" if recovery_outcome else None,
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
            options = list(self._ranked_options(query, state_before, join))
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
        recovery_outcome = self._append_natural_recovery_probe(
            question=question,
            required_parent=left,
            continuation=None,
            hypotheses=hypotheses,
            active=active,
            steps=steps,
            stat_scope="conjunction_recovery",
        )
        if recovery_outcome is None:
            recovery_outcome = self._append_natural_recovery_probe(
                question=question,
                required_parent=right,
                continuation=None,
                hypotheses=hypotheses,
                active=active,
                steps=steps,
                stat_scope="conjunction_recovery",
            )
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
        if (
            final is None
            or not self._matches_terminal_answers(plan, final, gold_answers)
        ):
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
                "probe_outcome": recovery_outcome,
                "recovery_stratum": "conjunction" if recovery_outcome else None,
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
        return any(
            self._bare_type_for_intersection(plan, combine) is not None
            for combine in plan.intersections
        )

    def _bare_type_for_intersection(
        self, plan: GoldPlan, combine: ProgramStatement
    ) -> Optional[str]:
        types = []
        for source in combine.sources:
            definition = self._definition_before(plan, source, combine.index)
            if (
                definition is not None
                and definition.kind == "start"
                and definition.arguments
                and _is_ontology_type(definition.arguments[0])
            ):
                types.append(definition.arguments[0])
        return types[0] if len(types) == 1 else None

    def _terminal_type_constraint(self, plan: GoldPlan) -> Optional[str]:
        """Return one explicit answer type that follows the semantic path."""
        non_stop = tuple(
            statement for statement in plan.statements if statement.kind != "stop"
        )
        if not non_stop or non_stop[-1].kind != "and":
            return None
        terminal = non_stop[-1]
        if terminal.target != plan.target_expression:
            return None
        return self._bare_type_for_intersection(plan, terminal)

    @staticmethod
    def _type_merge_state(
        functions: Sequence[str], target: str, ontology_type: str
    ) -> Tuple[Tuple[str, ...], str]:
        expression_numbers = [
            int(number)
            for raw in functions
            for number in re.findall(r"expression(\d+)", str(raw))
        ]
        type_expression = f"expression{max(expression_numbers, default=0) + 1}"
        state = (
            *tuple(str(raw) for raw in functions),
            f"{type_expression} = START('{ontology_type}')",
            f"{target} = AND({target}, {type_expression})",
        )
        return state, type_expression

    def _terminal_values(
        self,
        plan: GoldPlan,
        functions: Sequence[str],
        target: str,
        untyped_values: Tuple[str, ...],
    ) -> Optional[Tuple[str, ...]]:
        ontology_type = self._terminal_type_constraint(plan)
        if ontology_type is None:
            return untyped_values
        state, _ = self._type_merge_state(functions, target, ontology_type)
        return self._execute(state, target)

    def _matches_terminal_answers(
        self,
        plan: GoldPlan,
        node: ExecutedHypothesis,
        gold_answers: Tuple[str, ...],
    ) -> bool:
        return (
            self._terminal_values(
                plan,
                node.function_state,
                node.target_expression,
                node.denotation,
            )
            == gold_answers
        )

    def _append_terminal_type_constraint(
        self,
        demo: HyperDemonstration,
        plan: GoldPlan,
        gold_answers: Tuple[str, ...],
    ) -> Optional[HyperDemonstration]:
        ontology_type = self._terminal_type_constraint(plan)
        if ontology_type is None:
            return demo
        if not demo.steps or demo.steps[-1].action != "Commit":
            return None
        commit = demo.steps[-1]
        committed = demo.hypotheses.get(commit.arguments[0])
        if committed is None or committed.hypothesis_id not in commit.visible_before:
            return None
        state, _ = self._type_merge_state(
            committed.function_state,
            committed.target_expression,
            ontology_type,
        )
        values = self._execute(state, committed.target_expression)
        if values != gold_answers:
            self.stats["frontier_type_constraint_terminal_mismatch"] += 1
            return None
        if len(demo.hypotheses) >= self.max_nodes:
            self.stats["trajectory_node_budget_miss"] += 1
            return None
        constrained_step_count = len(demo.steps) + 2
        if constrained_step_count > self.max_turns:
            self.stats["trajectory_turn_budget_miss"] += 1
            return None

        node_id = f"H{len(demo.hypotheses)}"
        filtered = ExecutedHypothesis(
            node_id,
            state,
            committed.target_expression,
            values,
            denotation_labels=self._display_pairs(values),
            role="type_constrained",
            parent_id=committed.hypothesis_id,
            operation="merge",
            depth=committed.depth + 1,
            provenance=(f"ontology_type:{ontology_type}",),
        )
        demo.hypotheses[node_id] = filtered
        active_before = tuple(commit.visible_before)
        active_after = tuple(
            item for item in active_before if item != committed.hypothesis_id
        ) + (node_id,)
        demo.steps[-1:] = [
            DemonstrationStep(
                "Select",
                (committed.hypothesis_id,),
                active_before,
                (),
                ("apply_required_ontology_type_constraint",),
            ),
            DemonstrationStep(
                "Merge",
                (committed.target_expression, ontology_type),
                active_before,
                (node_id,),
                ("explicit_gold_type_constraint",),
            ),
            DemonstrationStep(
                "Commit",
                (node_id,),
                active_after,
                (),
                ("complete", "executable"),
            ),
        ]
        demo.private_metadata["ontology_type_constraints"] = [ontology_type]
        self.stats["frontier_type_constraint"] += 1
        return demo

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
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Return the benchmark-provided oracle topic entities.

        BoG and the matched KGQA protocol assume correct initial entity
        linking.  GrailQA records those entities as MID-valued START nodes in
        the annotated program.  Linker output is used only to recover readable
        labels for those oracle roots; it must not add unrelated candidates.
        """
        root_ids = []
        for statement in plan.statements:
            if (
                statement.kind == "start"
                and statement.arguments
                and statement.arguments[0].startswith(("m.", "g."))
            ):
                root_ids.append(str(statement.arguments[0]))

        extra = row.get("extra_info") or {}
        linked_names: Dict[str, str] = {}
        if isinstance(extra, Mapping):
            entities = extra.get("extracted_entities") or extra.get("candidate_entities")
            if entities:
                linked_names = {
                    str(item[-1]): str(item[0])
                    for item in entities
                    if item and str(item[-1]).startswith(("m.", "g."))
                }

        values = [(linked_names.get(identity, identity), identity) for identity in root_ids]
        provenance = "oracle_gold_program" if values else "no_entity_root"
        if values:
            self.stats["oracle_root_entities_used"] += 1

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
        return ordered, provenance

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

def _has_public_prune_reason(
    demo: HyperDemonstration,
    step: DemonstrationStep,
    node: ExecutedHypothesis,
    active_count: Optional[int] = None,
    max_active: Optional[int] = None,
) -> bool:
    del active_count, max_active
    if step.certificate_kind != "empty_monotone" or node.denotation:
        return False
    certificate = PruneCertificate(
        kind=step.certificate_kind,
        node_id=node.hypothesis_id,
        evidence=tuple(step.certificate_evidence),
        empty_preserving_completion=(
            "public_count_obligation:false" in step.certificate_evidence
        ),
    )
    return validate_public_prune_certificate(
        node.hypothesis_id,
        node.denotation,
        public_question_contract(demo.question),
        certificate,
    )


def _has_valid_commit_certificate(
    demo: HyperDemonstration,
    step: DemonstrationStep,
    node: ExecutedHypothesis,
) -> bool:
    valid_kind = step.certificate_kind == "answer_and_supported_query_equivalent"
    legacy_kind = (
        step.certificate_kind == "answer_and_intent_exact"
        and demo.private_metadata.get("runtime_protocol")
        != "lazy_relation_inspection_v1"
    )
    if not (valid_kind or legacy_kind):
        return False
    gold_program = demo.private_metadata.get("gold_program")
    gold_target = demo.private_metadata.get("gold_target_expression")
    if not gold_program or not gold_target or node.denotation != demo.gold_answers:
        return False
    return programs_are_intent_equivalent(
        node.function_state,
        node.target_expression,
        gold_program,
        str(gold_target),
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

    def _validate_lazy(self, demo: HyperDemonstration) -> List[str]:
        errors: List[str] = []
        for node in demo.hypotheses.values():
            replayed = self._replay(node.function_state, node.target_expression)
            if replayed is None:
                errors.append(f"{node.hypothesis_id}: replay execution failed")
            elif replayed != node.denotation:
                errors.append(f"{node.hypothesis_id}: replay mismatch")

        active = set(demo.steps[0].visible_before if demo.steps else ())
        parked: set[str] = set()
        known = set(active)
        selected: Optional[str] = None
        committed: Optional[str] = None
        abstained = False
        proposal_status: Dict[str, str] = {}
        frontiers: Dict[str, Dict[str, Any]] = {}
        max_nodes = int(demo.private_metadata.get("max_nodes", 128))
        max_execution_attempts = int(
            demo.private_metadata.get("max_execution_attempts", 24)
        )
        execution_attempts = 0

        def recall(node_id: str) -> None:
            if node_id not in parked:
                errors.append(f"Recall targets non-parked {node_id}")
                return
            if len(active) >= self.max_active:
                errors.append("Recall exceeds active hypothesis budget")
                return
            parked.remove(node_id)
            active.add(node_id)

        for step in demo.steps:
            if set(step.visible_before) != active:
                errors.append(
                    f"{step.action}: visible state {sorted(step.visible_before)} "
                    f"does not match active state {sorted(active)}"
                )
            if step.supervision not in {"policy_target", "intervention"}:
                errors.append(
                    f"{step.action}: invalid supervision mode {step.supervision}"
                )
            if step.supervision == "intervention" and step.action != "Select":
                errors.append(
                    f"{step.action}: only a teacher-forced Select may be an intervention"
                )
            if step.action not in {"Find_relation", "Widen"} and step.exposed:
                errors.append(f"{step.action}: only catalog actions may expose proposals")

            if step.action in {"Find_relation", "Widen"}:
                if step.created:
                    errors.append(f"{step.action}: symbolic catalog action executed a node")
                if not step.exposed:
                    errors.append(f"{step.action}: no proposals were exposed")
                    continue
                page = [demo.proposals.get(value) for value in step.exposed]
                if any(item is None for item in page):
                    errors.append(f"{step.action}: unknown proposal id")
                    continue
                proposal_page = [item for item in page if item is not None]
                frontier_ids = {item.frontier_id for item in proposal_page}
                if len(frontier_ids) != 1:
                    errors.append(f"{step.action}: page spans multiple frontiers")
                    continue
                frontier_id = next(iter(frontier_ids))
                if any(item.source != step.arguments[0] for item in proposal_page):
                    errors.append(f"{step.action}: proposal source mismatch")
                if len(step.relation_page) != 3:
                    errors.append(f"{step.action}: missing relation page cursor")
                    continue
                start, stop, total = step.relation_page
                if stop - start != len(proposal_page) or not (0 <= start < stop <= total):
                    errors.append(f"{step.action}: invalid relation page span")
                if [item.rank for item in proposal_page] != list(
                    range(start + 1, stop + 1)
                ):
                    errors.append(f"{step.action}: proposal ranks do not match page")
                if any(item.proposal_id in proposal_status for item in proposal_page):
                    errors.append(f"{step.action}: proposal exposed more than once")
                for item in proposal_page:
                    proposal_status[item.proposal_id] = "visible"

                if step.action == "Find_relation":
                    if len(step.arguments) != 1:
                        errors.append(
                            "Find_relation must expose only its source; relation ranking is environment-owned"
                        )
                    if start != 0:
                        errors.append("Find_relation must open the first page")
                    candidate_sources = {
                        str(source[-1])
                        for key in ("candidate_entities", "candidate_literals")
                        for source in demo.private_metadata.get(key, ())
                        if source
                    }
                    opens_new_root = (
                        bool(active)
                        and selected is None
                        and step.arguments[0] in candidate_sources
                    )
                    if active and selected is None and not opens_new_root:
                        errors.append("Find_relation requires Select or a public new root")
                    parent_id = selected
                    if parent_id is not None:
                        expected = demo.hypotheses[parent_id].target_expression
                        if step.arguments[0] != expected:
                            errors.append(
                                f"Find_relation source {step.arguments[0]} does not match "
                                f"selected {parent_id} target {expected}"
                            )
                    frontiers[frontier_id] = {
                        "source": step.arguments[0],
                        "parent_id": parent_id,
                        "exposed": stop,
                        "total": total,
                    }
                    selected = None
                else:
                    if selected is not None:
                        errors.append("Widen must occur before Select")
                    frontier = frontiers.get(frontier_id)
                    if frontier is None:
                        errors.append("Widen targets an unopened frontier")
                    elif (
                        frontier["source"] != step.arguments[0]
                        or frontier["exposed"] != start
                        or frontier["total"] != total
                    ):
                        errors.append("Widen does not continue the stable catalog")
                    else:
                        frontier["exposed"] = stop
                continue

            if step.action == "Inspect":
                if len(step.arguments) != 1:
                    errors.append("Inspect requires one proposal")
                    continue
                proposal_id = step.arguments[0]
                proposal = demo.proposals.get(proposal_id)
                if proposal is None or proposal_status.get(proposal_id) != "visible":
                    errors.append(f"Inspect targets unavailable {proposal_id}")
                    continue
                if len(step.created) != 1:
                    errors.append("certified Inspect must create exactly one hypothesis")
                    continue
                node_id = step.created[0]
                node = demo.hypotheses.get(node_id)
                frontier = frontiers.get(proposal.frontier_id)
                if node is None or frontier is None:
                    errors.append("Inspect lacks a node or open frontier")
                    continue
                if node.relation != proposal.relation:
                    errors.append(f"Inspect {proposal_id} relation mismatch")
                if node.parent_id != frontier["parent_id"]:
                    errors.append(f"Inspect {proposal_id} parent mismatch")
                if node_id in known:
                    errors.append(f"Inspect recreates known {node_id}")
                if len(active) >= self.max_active:
                    errors.append("Inspect exceeds active hypothesis budget")
                known.add(node_id)
                active.add(node_id)
                proposal_status[proposal_id] = "inspected"
                execution_attempts += 1
            elif step.action == "Park":
                node_id = step.arguments[0]
                if node_id not in active:
                    errors.append(f"Park targets inactive {node_id}")
                else:
                    active.remove(node_id)
                    parked.add(node_id)
                if selected == node_id:
                    selected = None
            elif step.action == "Recall":
                recall(step.arguments[0])
            elif step.action == "Select":
                node_id = step.arguments[0]
                if node_id not in active:
                    errors.append(f"Select targets inactive {node_id}")
                elif not demo.hypotheses[node_id].denotation:
                    errors.append(f"Select targets empty {node_id}")
                else:
                    selected = node_id
            elif step.action == "Prune":
                node_id = step.arguments[0]
                node = demo.hypotheses.get(node_id)
                if node_id not in active or node is None:
                    errors.append(f"Prune targets inactive {node_id}")
                elif not _has_public_prune_reason(
                    demo, step, node, len(active), self.max_active
                ):
                    errors.append(
                        f"Prune {node_id} lacks a verified public contradiction certificate"
                    )
                active.discard(node_id)
                if selected == node_id:
                    selected = None
            elif step.action == "Combine":
                left, right = step.arguments
                if left == right or left not in active or right not in active:
                    errors.append("Combine requires two distinct active parents")
                if len(step.created) != 1:
                    errors.append("Combine must create exactly one hypothesis")
                else:
                    child = demo.hypotheses.get(step.created[0])
                    if child is None or set(child.parent_ids) != {left, right}:
                        errors.append("Combine child has wrong parents")
                    active.difference_update((left, right))
                    active.add(step.created[0])
                    known.add(step.created[0])
                selected = None
                execution_attempts += 1
            elif step.action in {"Merge", "Order", "Compare", "Time_constraint", "Count"}:
                if len(step.created) != 1:
                    errors.append(f"{step.action} must create exactly one hypothesis")
                else:
                    child = demo.hypotheses.get(step.created[0])
                    if child is None:
                        errors.append(f"{step.action} creates an unknown hypothesis")
                    elif selected is not None and child.parent_id != selected:
                        errors.append(f"{step.action} child has wrong parent")
                    if selected is not None:
                        active.discard(selected)
                    active.add(step.created[0])
                    known.add(step.created[0])
                selected = None
                execution_attempts += 1
            elif step.action == "Commit":
                node_id = step.arguments[0]
                node = demo.hypotheses.get(node_id)
                if node_id not in active or node is None:
                    errors.append(f"Commit targets inactive {node_id}")
                else:
                    if node.denotation != demo.gold_answers:
                        errors.append(f"Commit {node_id} does not return gold answers")
                    if not _has_valid_commit_certificate(demo, step, node):
                        errors.append(
                            f"Commit {node_id} lacks exact answer-and-intent proof"
                        )
                    committed = node_id
                active = {node_id} if node is not None else set()
                parked.clear()
                selected = None
            elif step.action == "Abstain":
                abstained = True
                active.clear()
                parked.clear()
                selected = None
            else:
                errors.append(f"unsupported action {step.action}")

            if len(known) > max_nodes:
                errors.append(f"executed-node budget exceeded: {len(known)}/{max_nodes}")
            if execution_attempts > max_execution_attempts:
                errors.append(
                    "execution-attempt budget exceeded: "
                    f"{execution_attempts}/{max_execution_attempts}"
                )
            if len(active) > self.max_active:
                errors.append(f"active hypothesis budget exceeded: {len(active)}")

        expected_attempts = demo.private_metadata.get("execution_attempts")
        if expected_attempts is not None and execution_attempts != int(expected_attempts):
            errors.append(
                f"execution-attempt count mismatch: {execution_attempts}/{expected_attempts}"
            )
        if committed is None and not abstained:
            errors.append("lazy trajectory ends without Commit or Abstain")
        if set(demo.hypotheses) != known:
            errors.append("demonstration contains hypotheses never created by public actions")
        return errors

    def validate(self, demo: HyperDemonstration) -> List[str]:
        if demo.private_metadata.get("runtime_protocol") == "lazy_relation_inspection_v1":
            return self._validate_lazy(demo)
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
            if step.supervision not in {"policy_target", "intervention"}:
                errors.append(
                    f"{step.action}: invalid supervision mode {step.supervision}"
                )
            if step.supervision == "intervention" and step.action != "Select":
                errors.append(
                    f"{step.action}: only a teacher-forced Select may be an intervention"
                )
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
                if not _has_public_prune_reason(
                    demo, step, node, len(active), self.max_active
                ):
                    errors.append(
                        f"Prune {step.arguments[0]} lacks a verified public contradiction certificate"
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
                    str(source[-1])
                    for key in ("candidate_entities", "candidate_literals")
                    for source in demo.private_metadata.get(key, ())
                    if source
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
                if not _has_valid_commit_certificate(demo, step, node):
                    errors.append(
                        f"Commit {node.hypothesis_id} lacks exact answer-and-intent proof"
                    )
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
            if combined is not None and committed is not None:
                ancestor = demo.hypotheses[committed]
                while ancestor.parent_id is not None and ancestor.hypothesis_id != combined.hypothesis_id:
                    ancestor = demo.hypotheses[ancestor.parent_id]
                if ancestor.hypothesis_id != combined.hypothesis_id:
                    errors.append(
                        "conjunction must commit the combined branch or its descendant"
                    )
        return errors


def step_sft_records(demo: HyperDemonstration) -> List[Dict[str, Any]]:
    """Export public step supervision; private gold metadata is deliberately omitted."""
    records = []
    proposal_status: Dict[str, str] = {}
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
        visible_proposals = [
            asdict(demo.proposals[proposal_id])
            for proposal_id, status in proposal_status.items()
            if status == "visible" and proposal_id in demo.proposals
        ]
        if step.supervision == "policy_target":
            records.append(
                {
                    "demo_id": demo.demo_id,
                    "step": index,
                    "input": {
                        "question": demo.question,
                        "active_hypotheses": visible,
                        "visible_proposals": visible_proposals,
                    },
                    "target": {"action": step.action, "arguments": list(step.arguments)},
                    "metadata": {
                        "family": demo.family,
                        "recovery_stratum": demo.private_metadata.get(
                            "recovery_stratum"
                        ),
                        "trajectory_weight": 1.0 / max(
                            1,
                            sum(
                                item.supervision == "policy_target"
                                for item in demo.steps
                            ),
                        ),
                    },
                }
            )
        for proposal_id in step.exposed:
            proposal_status[proposal_id] = "visible"
        if step.action == "Inspect" and step.arguments:
            proposal_status[step.arguments[0]] = "inspected"
    return records


def _public_graph(
    demo: HyperDemonstration,
    active: Sequence[str],
    *,
    selected: Optional[str] = None,
    committed: Optional[str] = None,
    executions: int = 0,
    known_ids: Optional[Sequence[str]] = None,
    parked_ids: Optional[Sequence[str]] = None,
) -> str:
    known = set(known_ids or ())
    parked = set(parked_ids or ())
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
        parked_nodes=[node for node in nodes if node["node_id"] in parked],
        execution_budget=int(
            demo.private_metadata.get("max_execution_attempts", 24)
        ),
    )


def _public_proposal_catalogs(
    demo: HyperDemonstration,
    frontiers: Mapping[str, Mapping[str, Any]],
    proposal_status: Mapping[str, str],
) -> str:
    blocks = []
    for frontier_id, frontier in frontiers.items():
        lines = [
            "<proposal_catalog>",
            f"source={frontier['source']} exposed={frontier['exposed']}/{frontier['total']} "
            f"page_size={int(demo.private_metadata.get('relation_page_size', 6))}",
        ]
        for proposal in demo.proposals.values():
            status = proposal_status.get(proposal.proposal_id)
            if proposal.frontier_id != frontier_id or status != "visible":
                continue
            lines.append(
                f"{proposal.proposal_id} rank={proposal.rank} "
                f"relation={proposal.relation} score={proposal.score:.4f} status=visible"
            )
        lines.append(
            "Use Inspect [ Pn ] to execute one visible proposal; Widen only reveals the next page."
        )
        lines.append("</proposal_catalog>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


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
            thought = "I will open a ranked relation catalog without executing every option."
    elif step.action == "Widen":
        body = f"Widen [ {step.arguments[0]} ]"
        thought = (
            "The visible relation pages do not cover the question, so I will inspect "
            "the next ranked page before selecting a path."
        )
    elif step.action == "Inspect":
        body = f"Inspect [ {step.arguments[0]} ]"
        thought = "I will spend one graph query to test this visible relation proposal."
    elif step.action == "Park":
        body = f"Park [ {step.arguments[0]} ]"
        thought = "I will free visible workspace while preserving this unresolved hypothesis."
    elif step.action == "Recall":
        body = f"Recall [ {step.arguments[0]} ]"
        thought = "I will restore this preserved hypothesis for further reasoning."
    elif step.action == "Abstain":
        body = "Abstain"
        thought = "The search budget ended without a complete justified program."
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
                "The probe is still unresolved, so I will preserve it while inspecting "
                "another active interpretation."
                if "switch_while_preserving_unresolved_probe" in step.rationale_facts
                else "The probe has a verified terminal contradiction, so I will return "
                "to a preserved alternative."
                if "return_after_certified_failure" in step.rationale_facts
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
                "This hypothesis is empty and all remaining operations preserve "
                "emptiness, so it is proved unable to answer the question."
                if step.certificate_kind == "empty_monotone"
                else "This hypothesis has a verified public contradiction."
            ),
            "Commit": (
                "This executable hypothesis has an exact answer and its logical "
                "program covers the complete question."
            ),
        }
        thought = thoughts[step.action]
    return f"<think>{thought}</think>\n<action>{body}</action>"


def trajectory_sft_record(demo: HyperDemonstration) -> Dict[str, Any]:
    """Export one complete policy trajectory in the runtime's multi-turn format."""
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": build_hyper_prompt(
                demo.question,
                demo.private_metadata.get("candidate_entities", ()),
                demo.private_metadata.get("base_prompt", ""),
                candidate_literals=demo.private_metadata.get("candidate_literals", ()),
            ),
        }
    ]
    active: List[str] = list(demo.steps[0].visible_before if demo.steps else ())
    parked: set[str] = set()
    selected: Optional[str] = None
    committed_id: Optional[str] = None
    known = set(active)
    executions = 0
    open_frontiers: List[Dict[str, Any]] = []
    lazy_protocol = (
        demo.private_metadata.get("runtime_protocol")
        == "lazy_relation_inspection_v1"
    )
    lazy_frontiers: Dict[str, Dict[str, Any]] = {}
    proposal_status: Dict[str, str] = {}
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
                    parked_ids=parked,
                )
                + "\n</information>",
            }
        )
    for step in demo.steps:
        if set(active) != set(step.visible_before):
            raise ValueError(
                f"trajectory {demo.demo_id} has inconsistent visible state before {step.action}"
            )
        action_message: Dict[str, Any] = {
            "role": "assistant",
            "content": _action_text(step),
        }
        if step.supervision == "intervention":
            action_message["loss_mask"] = 0
        messages.append(action_message)
        if step.action == "Find_relation":
            if lazy_protocol:
                if step.created or not step.exposed:
                    raise ValueError("lazy Find_relation must expose proposals only")
                first = demo.proposals[step.exposed[0]]
                start, stop, total = step.relation_page
                if start != 0:
                    raise ValueError("Find_relation must expose the first relation page")
                lazy_frontiers[first.frontier_id] = {
                    "source": step.arguments[0],
                    "parent_id": selected,
                    "exposed": stop,
                    "total": total,
                }
                for proposal_id in step.exposed:
                    proposal_status[proposal_id] = "visible"
                selected = None
                event = (
                    f"Opened a symbolic relation catalog from {step.arguments[0]}; "
                    f"exposed proposals {start + 1}-{stop} of {total}. "
                    "No graph query was executed."
                )
            else:
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
            if lazy_protocol:
                if step.created or not step.exposed:
                    raise ValueError("lazy Widen must expose proposals only")
                first = demo.proposals[step.exposed[0]]
                frontier = lazy_frontiers.get(first.frontier_id)
                start, stop, total = step.relation_page
                if (
                    frontier is None
                    or frontier["source"] != step.arguments[0]
                    or frontier["exposed"] != start
                    or frontier["total"] != total
                ):
                    raise ValueError("Widen does not continue an open stable proposal catalog")
                frontier["exposed"] = stop
                for proposal_id in step.exposed:
                    proposal_status[proposal_id] = "visible"
                event = (
                    f"Widened the symbolic catalog from {step.arguments[0]}; "
                    f"exposed proposals {start + 1}-{stop} of {total}. "
                    "No graph query was executed."
                )
            else:
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
        elif step.action == "Inspect":
            proposal_id = step.arguments[0]
            if proposal_status.get(proposal_id) != "visible" or len(step.created) != 1:
                raise ValueError("Inspect requires one visible proposal and one replayed node")
            active.extend(step.created)
            known.update(step.created)
            proposal_status[proposal_id] = "inspected"
            executions += 1
            event = (
                f"Inspected {proposal_id} ({demo.proposals[proposal_id].relation}) "
                f"and created {step.created[0]}."
            )
        elif step.action == "Park":
            node_id = step.arguments[0]
            active.remove(node_id)
            parked.add(node_id)
            if selected == node_id:
                selected = None
            event = f"Parked {node_id} without making a semantic judgment."
        elif step.action == "Recall":
            node_id = step.arguments[0]
            parked.remove(node_id)
            active.append(node_id)
            event = f"Recalled {node_id} to the visible workspace."
        elif step.action == "Select":
            selected = step.arguments[0]
            event = (
                f"Selected {selected}. Further Find_relation actions now expand "
                "this hypothesis."
            )
        elif step.action == "Prune":
            node = demo.hypotheses[step.arguments[0]]
            if not _has_public_prune_reason(
                demo,
                step,
                node,
                len(active),
                int(demo.private_metadata.get("max_active", 24)),
            ):
                raise ValueError(
                    f"trajectory {demo.demo_id} prunes without a verified public certificate"
                )
            active.remove(step.arguments[0])
            if selected == step.arguments[0]:
                selected = None
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
            node = demo.hypotheses[step.arguments[0]]
            if not _has_valid_commit_certificate(demo, step, node):
                raise ValueError(
                    f"trajectory {demo.demo_id} commits without exact answer-and-intent proof"
                )
            active = [step.arguments[0]]
            parked.clear()
            committed_id = step.arguments[0]
            selected = None
            values = " ".join(demo.hypotheses[committed_id].denotation)
            event = (
                f"Committed {committed_id}. Return exactly these values in <answer>: "
                f"{values}"
            )
        elif step.action == "Abstain":
            active = []
            parked.clear()
            selected = None
            event = "Closed the search without a certified complete answer."
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
                        parked_ids=parked,
                    )
                    + (
                        "\n"
                        + (
                            _public_proposal_catalogs(
                                demo, lazy_frontiers, proposal_status
                            )
                            if lazy_protocol
                            else "\n".join(
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
                        )
                        if committed_id is None
                        and (
                            (lazy_protocol and lazy_frontiers)
                            or any(
                                set(frontier["node_ids"]).intersection(active)
                                for frontier in open_frontiers
                            )
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
            "recovery_stratum": demo.private_metadata.get("recovery_stratum"),
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
    step_index = 0
    for message_index, message in enumerate(trajectory["messages"]):
        if message.get("role") != "assistant" or "<action>" not in message.get("content", ""):
            continue
        step = demo.steps[step_index]
        step_index += 1
        if step.supervision != "policy_target":
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
                    "trajectory_step_index": step_index - 1,
                    "target_is_graph_action": True,
                },
            }
        )
        decision_index += 1
    return records
