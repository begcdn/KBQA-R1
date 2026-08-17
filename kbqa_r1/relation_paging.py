"""Deterministic relation pages shared by HyPER training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, Tuple, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RelationPage(Generic[T]):
    """One stable slice of a complete ranked relation list."""

    items: Tuple[T, ...]
    start: int
    stop: int
    total: int

    @property
    def has_more(self) -> bool:
        return self.stop < self.total


def relation_page(
    ranked_relations: Sequence[T], *, offset: int, page_size: int
) -> RelationPage[T]:
    """Return the page starting at ``offset`` without applying a rank cutoff."""
    if page_size < 1:
        raise ValueError("relation page size must be positive")
    if offset < 0:
        raise ValueError("relation page offset cannot be negative")

    total = len(ranked_relations)
    start = min(int(offset), total)
    stop = min(start + int(page_size), total)
    return RelationPage(
        items=tuple(ranked_relations[start:stop]),
        start=start,
        stop=stop,
        total=total,
    )


def serialize_relation_page_state(
    source: str, *, exposed: int, total: int, page_size: int
) -> str:
    """Public page cursor shown identically during SFT and live rollouts."""
    if exposed < 0 or total < 0 or exposed > total:
        raise ValueError("invalid exposed/total relation counts")
    next_count = min(page_size, total - exposed)
    return (
        "<relation_frontier>"
        f"source={source} exposed={exposed}/{total} "
        f"next_page={next_count if next_count else 'none'}"
        "</relation_frontier>"
    )
