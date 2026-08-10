"""Convert successful executable rollouts into HyPER-R1 behavior-cloning traces."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional

from .hyper_prompt import augment_dataset_row


_POLICY_NODE = re.compile(
    r"^(H\d+) \[(?:active|committed)\].*\bsource=policy\b", re.MULTILINE
)
_ANSWER_TAG = re.compile(r"<answer>[\s\S]*?</answer>", re.IGNORECASE)
_GRAPH_ACTION = re.compile(r"\b(?:Select|Prune|Combine|Commit)\s*\[", re.IGNORECASE)


def latest_policy_node(text: str) -> Optional[str]:
    """Return the newest factual hypothesis exposed by an environment message."""
    matches = _POLICY_NODE.findall(str(text or ""))
    if not matches:
        return None
    return max(matches, key=lambda node_id: int(node_id[1:]))


def _assistant(content: str) -> Dict[str, str]:
    return {"role": "assistant", "content": content}


def _tool(content: str) -> Dict[str, str]:
    return {"role": "tool", "content": content}


def add_graph_decisions(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Insert supervised Select/Commit turns into one successful rollout.

    The chosen node is read from the environment's executed graph. No relation,
    denotation, or hypothesis identifier is fabricated by this transformation.
    """
    source = [deepcopy(message) for message in messages]
    result: List[Dict[str, Any]] = []
    selected: Optional[str] = None

    for index, message in enumerate(source):
        role = message.get("role")
        content = str(message.get("content", ""))

        if role == "assistant" and _ANSWER_TAG.search(content) and selected:
            if not _GRAPH_ACTION.search(content):
                result.append(
                    _assistant(
                        "<think>The selected hypothesis is complete and executable.</think>\n"
                        f"<action>Commit [ {selected} ]</action>"
                    )
                )
                result.append(
                    _tool(
                        f"<information>Committed {selected}. Return its values in "
                        "<answer>.</information>"
                    )
                )

        result.append(message)

        if role != "tool" or "<hypothesis_graph>" not in content:
            continue
        factual = latest_policy_node(content)
        if factual is None:
            continue

        next_message = source[index + 1] if index + 1 < len(source) else {}
        next_content = str(next_message.get("content", ""))
        if _GRAPH_ACTION.search(next_content):
            selected = factual
            continue

        selected = factual
        result.append(
            _assistant(
                "<think>I will preserve the executed alternatives and continue from "
                "the factual branch.</think>\n"
                f"<action>Select [ {factual} ]</action>"
            )
        )
        result.append(
            _tool(
                f"<information>Selected {factual}. Further reasoning expands this "
                "hypothesis.\n"
                f"{_extract_graph(content)}\n</information>"
            )
        )

    return result


def _extract_graph(text: str) -> str:
    match = re.search(
        r"<hypothesis_graph>[\s\S]*?</hypothesis_graph>", str(text), re.IGNORECASE
    )
    return match.group(0) if match else ""


def convert_sft_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Augment one successful multi-turn SFT row with graph decisions."""
    converted = augment_dataset_row(row)
    messages = converted.get("messages")
    if not isinstance(messages, list):
        raise ValueError("HyPER-R1 SFT rows require a multi-turn messages column")
    converted["messages"] = add_graph_decisions(messages)
    converted.setdefault("extra_info", {})
    if isinstance(converted["extra_info"], dict):
        converted["extra_info"]["hyper_r1_sft"] = True
    return converted
