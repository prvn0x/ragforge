"""Keyset (cursor) pagination for every list endpoint.

Cursors are opaque base64url JSON of the last row's sort key. Keyset (not OFFSET) so pages stay
stable while rows are inserted and cost O(limit) at any depth. A cursor is not a capability: it is
only a position, and every page is still filtered by RLS, so a cursor from another tenant yields
nothing of theirs.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Query
from sqlalchemy import RowMapping

from ragforge_contracts.api import Page
from ragforge_core.domain.errors import ValidationFailed

Limit = Annotated[int, Query(ge=1, le=200)]
CursorParam = Annotated[str | None, Query(max_length=512)]
DEFAULT_LIMIT = 50


def _encode(parts: list[str]) -> str:
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(cursor: str) -> list[Any]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        parts = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise ValidationFailed("invalid cursor") from exc
    if not isinstance(parts, list):
        raise ValidationFailed("invalid cursor")
    return parts


def time_id_after(cursor: str | None) -> tuple[str, dict[str, Any]]:
    """SQL fragment + params for `ORDER BY created_at DESC, id DESC` pages."""
    if cursor is None:
        return "", {}
    parts = _decode(cursor)
    try:
        created_at, row_id = datetime.fromisoformat(parts[0]), UUID(parts[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValidationFailed("invalid cursor") from exc
    return " AND (created_at, id) < (:c_ts, :c_id)", {"c_ts": created_at, "c_id": row_id}


def int_id_after(cursor: str | None) -> tuple[str, dict[str, Any]]:
    """SQL fragment + params for `ORDER BY id DESC` pages over bigint ids."""
    if cursor is None:
        return "", {}
    parts = _decode(cursor)
    try:
        row_id = int(parts[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValidationFailed("invalid cursor") from exc
    return " AND id < :c_id", {"c_id": row_id}


def time_id_cursor(row: RowMapping) -> str:
    return _encode([row["created_at"].isoformat(), str(row["id"])])


def int_id_cursor(row: RowMapping) -> str:
    return _encode([str(row["id"])])


def page[T](
    rows: Sequence[RowMapping],
    limit: int,
    to_item: Callable[[RowMapping], T],
    cursor_of: Callable[[RowMapping], str] = time_id_cursor,
) -> Page[T]:
    """`rows` was fetched with LIMIT limit + 1: the extra row only signals that more exist."""
    visible = rows[:limit]
    more = len(rows) > limit
    return Page[T](
        items=[to_item(r) for r in visible],
        next_cursor=cursor_of(visible[-1]) if more and visible else None,
    )
