"""Typed callers for the slot-0004 definer functions other modules invoke.

* `route_notion_webhook` (API, module E): which (tenant, connector) pairs a verified webhook
  for (integration_id, workspace_id) belongs to. Call it in a SYSTEM transaction, then enqueue
  `sync_page` at priority `webhook` inside each tenant's own transaction.
* `purge_tombstones` (worker maintenance): hard-delete documents tombstoned > 7 days ago,
  in bounded batches; loop until it returns fewer than `limit`.

Connector config keys used for routing (owned by module C; written at connect time):
`config.integration_id` (the Notion bot id) and `config.workspace_id`.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import text

from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.ports import Tx

CONFIG_INTEGRATION_ID = "integration_id"
CONFIG_WORKSPACE_ID = "workspace_id"

_ROUTE = text(
    "SELECT tenant_id, connector_id FROM rf.route_notion_webhook(:integration_id, :workspace_id)"
)
_PURGE = text("SELECT rf.purge_tombstones(:limit)")


async def route_notion_webhook(
    tx: Tx, integration_id: str, workspace_id: str
) -> Sequence[tuple[UUID, UUID]]:
    result = await connection(tx).execute(
        _ROUTE, {"integration_id": integration_id, "workspace_id": workspace_id}
    )
    return [(UUID(str(t)), UUID(str(c))) for t, c in result.all()]


async def purge_tombstones(tx: Tx, limit: int = 1000) -> int:
    result = await connection(tx).execute(_PURGE, {"limit": limit})
    return int(result.scalar_one())
