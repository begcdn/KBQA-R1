import pytest

from kbqa_r1.relation_paging import (
    relation_page,
    serialize_relation_page_state,
)


def test_relation_pages_are_complete_stable_and_nonoverlapping():
    ranked = tuple(f"r.{index}" for index in range(14))

    first = relation_page(ranked, offset=0, page_size=6)
    second = relation_page(ranked, offset=first.stop, page_size=6)
    third = relation_page(ranked, offset=second.stop, page_size=6)

    assert first.items == ranked[:6]
    assert second.items == ranked[6:12]
    assert third.items == ranked[12:]
    assert first.has_more and second.has_more and not third.has_more
    assert first.items + second.items + third.items == ranked


def test_relation_page_rejects_invalid_interface_parameters():
    with pytest.raises(ValueError, match="positive"):
        relation_page(("r.one",), offset=0, page_size=0)
    with pytest.raises(ValueError, match="negative"):
        relation_page(("r.one",), offset=-1, page_size=6)


def test_public_page_state_reveals_cursor_but_not_unseen_relations():
    state = serialize_relation_page_state(
        "expression2", exposed=6, total=17, page_size=6
    )

    assert "source=expression2" in state
    assert "exposed=6/17" in state
    assert "next_page=6" in state
    assert "r.unseen" not in state
