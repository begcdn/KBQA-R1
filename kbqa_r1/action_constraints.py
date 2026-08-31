"""State-conditioned structural action language for HyPER-R1.

The constraint contains only public runtime state.  It removes impossible
protocol actions while leaving semantic choices, including wrong-but-legal
ones, to the policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .hyper_r1 import GraphActionAffordances


HYPER_ACTION_CONSTRAINT_VERSION = "hyper-action-v2"


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _exact_graph_actions(
    affordances: GraphActionAffordances,
    *,
    inspect: Sequence[str] = (),
    widen: Sequence[str] = (),
) -> Tuple[str, ...]:
    actions = []
    actions.extend(f"Select [ {value} ]" for value in affordances.select)
    actions.extend(f"Park [ {value} ]" for value in affordances.park)
    actions.extend(f"Commit [ {value} ]" for value in affordances.commit)
    for value in affordances.combine:
        left, right = (part.strip() for part in value.split("|", 1))
        actions.append(f"Combine [ {left} | {right} ]")
        actions.append(f"Combine [ {right} | {left} ]")
    actions.extend(f"Prune [ {value} ]" for value in affordances.prune)
    actions.extend(f"Recall [ {value} ]" for value in affordances.recall)
    actions.extend(f"Find_relation [ {value} ]" for value in affordances.find_relation)
    actions.extend(f"Inspect [ {value} ]" for value in inspect)
    actions.extend(f"Widen [ {value} ]" for value in widen)
    return _unique(actions)


@dataclass(frozen=True)
class HyPERActionConstraintSpec:
    """Compact public-state contract reconstructed by rollout and actor workers."""

    state_key: str
    turn: int
    exact_actions: Tuple[str, ...]
    selected_expression: str = ""
    allow_open_operators: bool = True
    version: str = HYPER_ACTION_CONSTRAINT_VERSION

    @classmethod
    def build(
        cls,
        *,
        state_key: str,
        turn: int,
        affordances: GraphActionAffordances,
        inspect: Sequence[str] = (),
        widen: Sequence[str] = (),
        selected_expression: str = "",
        allow_open_operators: bool = True,
    ) -> "HyPERActionConstraintSpec":
        return cls(
            state_key=str(state_key),
            turn=int(turn),
            exact_actions=_exact_graph_actions(
                affordances, inspect=inspect, widen=widen
            ),
            selected_expression=str(selected_expression or "").strip(),
            allow_open_operators=bool(allow_open_operators),
        )

    @property
    def action_pattern(self) -> str:
        alternatives = [re.escape(action) for action in self.exact_actions]

        if self.allow_open_operators:
            ontology = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
            relation = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
            relation_step = rf"(?:{relation}|\(R\s+{relation}\))"
            relation_path = relation_step
            for _ in range(4):
                relation_path = (
                    rf"(?:{relation_step}|\(JOIN\s+{relation_path}\s+{relation_path}\))"
                )
            number = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
            date = r"\d{4}(?:-(?:0[1-9]|1[0-2]))?(?:-(?:0[1-9]|[12]\d|3[01]))?(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?"
            typed = rf"(?:{number}|{date})(?:\^\^(?:https?://[^\s\]]+|xsd:[A-Za-z]+))?"

            if self.selected_expression:
                # Continuation operators consume the currently selected
                # hypothesis. Other stored expressions remain reachable only
                # through their public Select target.
                expression = re.escape(self.selected_expression)
                alternatives.append(rf"Count\s*\[\s*{expression}\s*\]")
                alternatives.append(rf"Merge\s*\[\s*{expression}\s*\|\s*{ontology}\s*\]")
                alternatives.append(
                    rf"Time_constraint\s*\[\s*{relation}\s*\|\s*(?:NOW|{date})\s*\]"
                )
                order_source = expression
            else:
                # Compare and ontology-rooted Order are the two public
                # operators that may open an independent executable branch.
                order_source = ontology

            alternatives.append(
                rf"Order\s*\[\s*(?:ARGMAX|ARGMIN)\s*\|\s*{order_source}\s*\|\s*{relation_path}\s*\]"
            )
            alternatives.append(
                rf"Compare\s*\[\s*(?:le|ge|lt|gt)\s*\|\s*{relation}\s*\|\s*{typed}\s*\]"
            )

        if not alternatives:
            raise ValueError("HyPER constraint has no executable action")
        return "(?:" + "|".join(alternatives) + ")"

    @property
    def response_pattern(self) -> str:
        # Reasoning remains a policy choice. The decoder constrains the complete
        # envelope and exactly one action payload.  Excluding ``<`` makes the
        # first closing tag an unambiguous boundary for streaming grammar
        # engines; an unrestricted ``.*`` can swallow ``</think><action>`` and
        # leave the supposedly constrained action in the free-form region.
        thinking = r"(?:<think>[^<]*</think>\s*)?"
        return rf"{thinking}<action>\s*{self.action_pattern}\s*</action>"

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exact_actions"] = list(self.exact_actions)
        payload["response_pattern"] = self.response_pattern
        payload["digest"] = self.digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HyPERActionConstraintSpec":
        version = str(payload.get("version", ""))
        if version != HYPER_ACTION_CONSTRAINT_VERSION:
            raise ValueError(f"unsupported HyPER constraint version: {version}")
        spec = cls(
            state_key=str(payload["state_key"]),
            turn=int(payload["turn"]),
            exact_actions=tuple(str(value) for value in payload.get("exact_actions", ())),
            selected_expression=str(payload.get("selected_expression", "")),
            allow_open_operators=bool(payload.get("allow_open_operators", True)),
            version=version,
        )
        supplied_digest = payload.get("digest")
        if supplied_digest is not None and str(supplied_digest) != spec.digest:
            raise ValueError("HyPER constraint digest mismatch")
        return spec

    def accepts_response(self, text: str) -> bool:
        return re.fullmatch(self.response_pattern, str(text), flags=re.DOTALL) is not None
