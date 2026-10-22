"""Module C1 test support: recorded-fixture loading, a respx "Notion server", a fake limiter.

Imported as `module_c1.c1_support` (a namespace package under packages/core/tests), the same
name mypy resolves through mypy_path, so pytest and `make typecheck` agree.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import respx

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://api.notion.com/v1"
TOKEN = "ntn_TESTsecretTOKEN_9f8e7d6c5b4a3f2e1d0c"  # fake; must never leak anywhere

HANDBOOK_ID = "59833787-2cf9-4fdf-8782-e53db20768a5"
ROW_ID = "d5000000-0000-4000-8000-000000000001"
SYNCED_ORIGINAL_ID = "5e000000-0000-4000-8000-000000000001"
MENTIONED_PAGE_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
CHILD_PAGE_ID = "b1000000-0000-4000-8000-000000000018"
LINKED_PAGE_ID = "a5000000-0000-4000-8000-000000000001"

FIRST_DATE = "Fri, 16 Oct 2026 12:00:00 GMT"
LATER_DATE = "Fri, 16 Oct 2026 12:07:30 GMT"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def error_body(status: int) -> dict[str, Any]:
    body: dict[str, Any] = load("errors.json")[str(status)]
    return body


def bid(n: int) -> str:
    """Handbook block id by number, e.g. bid(15) -> b1000000-...-000000000015."""
    return f"b1000000-0000-4000-8000-{n:012d}"


@dataclass
class FakeLimiter:
    """RateLimiter test double: records every acquire; never blocks."""

    calls: list[tuple[str, int]] = field(default_factory=list)

    async def acquire(self, key: str, tokens: int = 1) -> None:
        self.calls.append((key, tokens))


@dataclass
class FakePenalizingLimiter(FakeLimiter):
    """FakeLimiter that also implements `PenalizingLimiter` (like PostgresRateLimiter)."""

    penalties: list[tuple[str, float]] = field(default_factory=list)

    async def penalize(self, key: str, seconds: float) -> float:
        self.penalties.append((key, seconds))
        return seconds


@dataclass
class FakeSleep:
    delays: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _paged(pages: Sequence[Mapping[str, Any]], date: str) -> Any:
    """respx side effect serving a recorded paginated listing by `start_cursor`."""

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("start_cursor")
        if cursor is None and request.content:
            cursor = json.loads(request.content).get("start_cursor")
        if cursor is None:
            return httpx.Response(200, json=pages[0], headers={"Date": date})
        for prev, page in itertools.pairwise(pages):
            if prev.get("next_cursor") == cursor:
                return httpx.Response(200, json=page, headers={"Date": date})
        return httpx.Response(400, json=error_body(400), headers={"Date": date})

    return handler


def serve_page(
    router: respx.MockRouter,
    page_fixture: str,
    blocks_fixture: str,
    *,
    first_date: str = FIRST_DATE,
    later_date: str = LATER_DATE,
) -> None:
    """Serve GET /pages/{id} (with `first_date`) and every recorded block listing (with
    `later_date`, so tests can tell which Date header fetched_at came from)."""
    page = load(page_fixture)
    router.get(f"/pages/{page['id']}").mock(
        return_value=httpx.Response(200, json=page, headers={"Date": first_date})
    )
    blocks: dict[str, Any] = load(blocks_fixture)
    for parent, pages in blocks.items():
        if parent.startswith("_"):
            continue
        router.get(f"/blocks/{parent}/children").mock(side_effect=_paged(pages, later_date))


def serve_search(router: respx.MockRouter, pages: Sequence[Mapping[str, Any]]) -> respx.Route:
    return router.post("/search").mock(side_effect=_paged(pages, FIRST_DATE))


def paged_handler(pages: Sequence[Mapping[str, Any]]) -> Any:
    return _paged(pages, FIRST_DATE)
