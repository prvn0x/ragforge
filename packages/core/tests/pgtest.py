"""Test helpers importable from any test module (pytest puts this directory on sys.path)."""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

from ragforge_core.config import Settings
from ragforge_core.domain.models import RankedChunk


def connect(settings: Settings, role: str) -> psycopg.Connection[tuple[object, ...]]:
    """Sync psycopg connection as a role: superuser|migrator|api|worker|metrics|backup."""
    return psycopg.connect(settings.dsn(role, driver="postgresql"))


class IdentityReranker:
    """Stands in for FlashRank where a test builds the real API container: the host has no baked
    model under /opt/models/flashrank, and build_api refuses to start without one."""

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        return list(candidates)
