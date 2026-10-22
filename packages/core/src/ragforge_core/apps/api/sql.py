"""Small SQL helpers shared by the API's routes and stores.

Every query here runs inside a transaction from `TransactionManager`: tenant-scoped ones rely on
RLS (another tenant's row is simply invisible, which is why routes answer 404, never 403, for
foreign ids), and only the named definer functions cross tenants.

Statements are composed only from module constants (column lists) and pagination fragments that
themselves contain bind parameters; every user-supplied value is a bind parameter. That is what
the `noqa: S608` markers in the routes assert.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_core.services.auth.keys import GeneratedKey
from ragforge_core.services.auth.principal import AuthenticatedKey
from ragforge_core.services.auth.scopes import ClientKind

SET_TENANT = text("SELECT set_config('app.tenant_id', :tenant_id, true)")


async def fetch_one(
    conn: AsyncConnection, sql: str, params: Mapping[str, Any] | None = None
) -> RowMapping | None:
    result = await conn.execute(text(sql), dict(params or {}))
    return result.mappings().one_or_none()


async def fetch_all(
    conn: AsyncConnection, sql: str, params: Mapping[str, Any] | None = None
) -> Sequence[RowMapping]:
    result = await conn.execute(text(sql), dict(params or {}))
    return result.mappings().all()


async def execute(conn: AsyncConnection, sql: str, params: Mapping[str, Any] | None = None) -> int:
    """Run a statement; returns the affected row count."""
    result = await conn.execute(text(sql), dict(params or {}))
    return result.rowcount


async def set_tenant(conn: AsyncConnection, tenant_id: UUID) -> None:
    """Re-scope the CURRENT transaction (transaction-local, bind-parameter safe)."""
    await conn.execute(SET_TENANT, {"tenant_id": str(tenant_id)})


def sqlstate(exc: BaseException) -> str | None:
    if isinstance(exc, DBAPIError):
        return getattr(exc.orig, "sqlstate", None)
    return None


async def audit(
    conn: AsyncConnection,
    actor: AuthenticatedKey | str,
    action: str,
    target: str | None = None,
    details: Mapping[str, Any] | None = None,
    *,
    tenant_id: UUID | None = None,
) -> None:
    """Insert an audit row in the caller's transaction, so it commits with the change it records.

    `details` must never contain secrets; callers pass ids and enum values only.
    """
    if isinstance(actor, AuthenticatedKey):
        tenant = tenant_id or actor.principal.tenant_id
        actor_name = actor.actor
    else:
        if tenant_id is None:
            raise ValueError("tenant_id is required for a non-key actor")
        tenant, actor_name = tenant_id, actor
    await conn.execute(
        text(
            "INSERT INTO rf.audit_log (tenant_id, actor, action, target, details) "
            "VALUES (:tenant, :actor, :action, :target, CAST(:details AS jsonb))"
        ),
        {
            "tenant": tenant,
            "actor": actor_name,
            "action": action,
            "target": target,
            "details": json.dumps(dict(details or {}), default=str),
        },
    )


API_KEY_COLUMNS = "id, name, prefix, scopes, created_at, last_used_at, revoked_at"


async def insert_api_key(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    user_id: UUID,
    name: str,
    key: GeneratedKey,
    scopes: frozenset[str],
    client_kind: ClientKind,
) -> RowMapping:
    row = await fetch_one(
        conn,
        "INSERT INTO rf.api_keys "  # noqa: S608
        "(tenant_id, user_id, name, prefix, key_sha256, scopes, client_kind) "
        "VALUES (:tenant, :user, :name, :prefix, :sha, :scopes, :kind) "
        f"RETURNING {API_KEY_COLUMNS}",
        {
            "tenant": tenant_id,
            "user": user_id,
            "name": name,
            "prefix": key.prefix,
            "sha": key.sha256,
            "scopes": sorted(scopes),
            "kind": client_kind.value if client_kind is ClientKind.mcp else None,
        },
    )
    assert row is not None  # INSERT ... RETURNING always yields the row
    return row
