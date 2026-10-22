"""Module B local fixtures (shared ones come from packages/core/tests/conftest.py)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from pgtest import connect
from ragforge_core.config import Settings
from ragforge_core.db.engine import make_engine

Conn = psycopg.Connection[tuple[object, ...]]


@pytest.fixture
async def b_worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "worker")
    yield engine
    await engine.dispose()


@pytest.fixture
def b_session(settings: Settings) -> Iterator[Conn]:
    """A private superuser session (NOT autocommit) for SET ROLE / auto_explain experiments."""
    with connect(settings, "superuser") as conn:
        yield conn


@pytest.fixture
def b_migrator(settings: Settings) -> Iterator[Conn]:
    with connect(settings, "migrator") as conn:
        yield conn
