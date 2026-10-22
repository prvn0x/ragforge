"""Module C2 local fixtures (shared ones come from packages/core/tests/conftest.py)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.config import Settings
from ragforge_core.db.engine import make_engine


@pytest.fixture
async def worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "worker")
    yield engine
    await engine.dispose()
