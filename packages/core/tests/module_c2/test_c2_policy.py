"""Pure sync rules: freshness (ADR-0007), cursor monotonicity, content hash."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragforge_core.domain.models import NormalizedBlock, SourceDocument, SourceItemRef, SourceKind
from ragforge_core.services.sync.policy import (
    is_fresh,
    listing_since,
    next_cursor,
    page_sha256,
)


def t(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 10, 16, 9, minute, second, tzinfo=UTC)


@pytest.mark.parametrize(
    ("edited", "fetched", "fresh"),
    [
        (t(10), t(20, 30), True),  # edited well before the fetch's minute
        (t(19, 59), t(20, 30), True),
        (t(20), t(20, 30), False),  # same minute as the fetch: must be refetched
        (t(21), t(20, 30), False),
        (None, t(20), False),
        (t(10), None, False),
    ],
)
def test_freshness_rule(edited: datetime | None, fetched: datetime | None, fresh: bool) -> None:
    assert is_fresh(edited, fetched) is fresh


def test_cursor_never_moves_backwards() -> None:
    assert next_cursor(t(30), t(25)) == t(30)
    assert next_cursor(t(30), t(35)) == t(35)
    assert next_cursor(t(30), None) == t(30)
    assert next_cursor(None, t(5)) == t(5)
    assert listing_since(t(30), timedelta(minutes=5)) == t(25)
    assert listing_since(None, timedelta(minutes=5)) is None


def _doc(text: str, fetched: datetime, title: str = "T") -> SourceDocument:
    return SourceDocument(
        ref=SourceItemRef(SourceKind.notion, "p", fetched),
        title=title,
        external_url=None,
        blocks=(NormalizedBlock("a", "paragraph", text),),
        fetched_at=fetched,
    )


def test_page_hash_ignores_fetch_metadata_but_not_content() -> None:
    base = page_sha256(_doc("x", t(1)))
    assert len(base) == 32
    assert page_sha256(_doc("x", t(9))) == base
    assert page_sha256(_doc("y", t(1))) != base
    assert page_sha256(_doc("x", t(1), title="U")) != base
