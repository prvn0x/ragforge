"""Which traffic a request is (`prompt_context.surface`): the owner's own chat, interview mode,
the eval harness, or the public demo (ADR-0018 R2-1).

Why it matters: the owner's dashboards and alerts (verified share, TTFT) must not move because
an eval run fired 300 questions at the eval tenant, or because the demo tenant was exercised.
The signal is the TENANT, never a tenant id: an `eval` tenant is the eval harness; the demo
tenant is the one `make eval-demo` creates under `DEMO_TENANT_NAME`; anything else is the owner.

The tenant's kind and name never change after provisioning, so one lookup per tenant per
process is enough (`TenantSurfaces` caches it). A failed lookup is answered with "other" and
not cached: a metrics label must never fail or slow a chat turn more than once.
"""

from __future__ import annotations

from typing import Final
from uuid import UUID

from sqlalchemy import text

from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.observability.logging import get_logger

# The Makefile's `eval-demo` target creates the demo tenant with this name (a test pins it).
DEMO_TENANT_NAME: Final = "ragforge-eval-demo"
_TENANT_SQL: Final = text("SELECT kind, name FROM rf.tenants WHERE id = rf.current_tenant()")
_MAX_CACHED: Final = 1024

log = get_logger(__name__)


def surface_of_tenant(kind: str, name: str) -> str:
    if kind == "eval":
        return "eval"
    if name == DEMO_TENANT_NAME:
        return "demo"
    return "chat"


def surface_of_path(path: str) -> str:
    """The surface a route implies before the caller is known (the middleware's default)."""
    if path.startswith("/v1/interview/"):
        return "interview"
    if path.startswith("/v1/eval/"):
        return "eval"
    return "chat"


class TenantSurfaces:
    """tenant id -> surface, read once per tenant through the tenant's own RLS scope."""

    def __init__(self) -> None:
        self._known: dict[UUID, str] = {}

    async def resolve(self, tx: TransactionManager, tenant_id: UUID) -> str:
        cached = self._known.get(tenant_id)
        if cached is not None:
            return cached
        try:
            async with tx.tenant(tenant_id) as t:
                row = (await connection(t).execute(_TENANT_SQL)).first()
        except Exception as exc:  # a label is never worth failing a turn
            log.warning("surface_lookup_failed", error=type(exc).__name__)
            return "other"
        found = "other" if row is None else surface_of_tenant(str(row[0]), str(row[1]))
        if len(self._known) >= _MAX_CACHED:
            self._known.clear()
        self._known[tenant_id] = found
        return found
