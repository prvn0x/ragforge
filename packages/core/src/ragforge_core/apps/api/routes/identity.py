"""Tenants (platform provisioning) and API keys (tenant self-administration).

Authorization (services/auth/scopes.py):
  - POST /v1/tenants needs `admin` (platform scope, CLI-issued with --platform-admin only).
  - Minting and listing keys needs a key-manager key (`approvals:write` or `admin`). Without
    `admin` the caller sees and manages only its OWN user's keys, so in a team tenant one member
    cannot revoke another member's keys; `admin` sees the whole tenant.
  - Any key may revoke ITSELF (no scope needed): the holder of a leaked MCP key can kill it.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response

from ragforge_contracts.api import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyCreated,
    Page,
    Tenant,
    TenantCreate,
    TenantProvisioned,
)
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.deps import AdminKey, AnyKey, KeyManagerKey, TxManager
from ragforge_core.apps.api.pagination import (
    DEFAULT_LIMIT,
    CursorParam,
    Limit,
    page,
    time_id_after,
)
from ragforge_core.apps.api.sql import API_KEY_COLUMNS, audit, fetch_all, fetch_one, insert_api_key
from ragforge_core.apps.api.stores import SqlProvisioningStore
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.models import Scope
from ragforge_core.services.auth.keys import generate_key
from ragforge_core.services.auth.ports import NewTenant
from ragforge_core.services.auth.principal import AuthenticatedKey
from ragforge_core.services.auth.provisioning import provision_tenant
from ragforge_core.services.auth.scopes import require_key_manager, validate_new_key_scopes

router = APIRouter(prefix="/v1")
NO_STORE = "no-store"  # responses carrying a key secret must never be cached


@router.post("/tenants", status_code=201, tags=["tenants"])
async def create_tenant(
    body: TenantCreate, key: AdminKey, txm: TxManager, response: Response
) -> TenantProvisioned:
    result = await provision_tenant(
        SqlProvisioningStore(txm),
        NewTenant(
            name=body.name,
            kind=body.kind.value,
            owner_display_name=body.owner_display_name,
            team_visibility_consent=body.team_visibility_consent,
        ),
        actor=key.actor,
        actor_tenant=key.principal.tenant_id,
    )
    async with txm.tenant(result.tenant_id) as tx:
        conn = connection(tx)
        tenant = await fetch_one(
            conn, "SELECT id, name, kind, created_at FROM rf.tenants WHERE id = :id",
            {"id": result.tenant_id},
        )  # fmt: skip
        key_row = await fetch_one(
            conn, f"SELECT {API_KEY_COLUMNS} FROM rf.api_keys WHERE id = :id",  # noqa: S608
            {"id": result.key_id},
        )  # fmt: skip
    assert tenant is not None
    assert key_row is not None
    response.headers["cache-control"] = NO_STORE
    return TenantProvisioned(
        tenant=Tenant.model_validate(dict(tenant)),
        user_id=result.user_id,
        api_key=ApiKeyCreated.model_validate({**key_row, "secret": result.key.plaintext}),
    )


@router.post("/api-keys", status_code=201, tags=["api keys"])
async def create_api_key(
    body: ApiKeyCreate, key: KeyManagerKey, txm: TxManager, response: Response
) -> ApiKeyCreated:
    scopes, kind = validate_new_key_scopes(key.principal, (s.value for s in body.scopes), body.name)
    new_key = generate_key()
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        row = await insert_api_key(
            conn,
            tenant_id=key.principal.tenant_id,
            user_id=key.principal.user_id,
            name=body.name,
            key=new_key,
            scopes=scopes,
            client_kind=kind,
        )
        await audit(
            conn, key, "api_key.create", str(row["id"]),
            {"prefix": new_key.prefix, "scopes": sorted(scopes), "client_kind": kind.value},
        )  # fmt: skip
    response.headers["cache-control"] = NO_STORE
    return ApiKeyCreated.model_validate({**row, "secret": new_key.plaintext})


def _owner_filter(key: AuthenticatedKey) -> tuple[str, dict[str, UUID]]:
    """Visibility of keys within the tenant (RLS already hides other tenants): `admin` sees every
    key of the tenant, any other key manager only its own user's keys."""
    if key.principal.has(Scope.admin.value):
        return "", {}
    return " AND user_id = :owner", {"owner": key.principal.user_id}


@router.get("/api-keys", tags=["api keys"])
async def list_api_keys(
    key: KeyManagerKey, txm: TxManager, cursor: CursorParam = None, limit: Limit = DEFAULT_LIMIT
) -> Page[ApiKey]:
    after, params = time_id_after(cursor)
    owner, owner_params = _owner_filter(key)
    async with txm.tenant(key.principal.tenant_id) as tx:
        rows = await fetch_all(
            connection(tx),
            f"SELECT {API_KEY_COLUMNS} FROM rf.api_keys "  # noqa: S608
            f"WHERE tenant_id = :tenant{owner}{after} "
            "ORDER BY created_at DESC, id DESC LIMIT :n",
            {**params, **owner_params, "tenant": key.principal.tenant_id, "n": limit + 1},
        )
    return page(rows, limit, lambda r: ApiKey.model_validate(dict(r)))


@router.delete("/api-keys/{key_id}", status_code=204, tags=["api keys"])
async def revoke_api_key(key_id: UUID, key: AnyKey, txm: TxManager) -> Response:
    """Idempotent: revoking an already revoked key is a no-op 204; an invisible id is 404.

    Self-revocation needs no scope; revoking another key needs a key manager (scopes.py)."""
    owner: str = ""
    owner_params: dict[str, UUID] = {}
    if key_id != key.key_id:
        require_key_manager(key.principal)
        owner, owner_params = _owner_filter(key)
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        row = await fetch_one(
            conn,
            "UPDATE rf.api_keys SET revoked_at = coalesce(revoked_at, now()) "  # noqa: S608
            f"WHERE id = :id{owner} RETURNING prefix",
            {"id": key_id, **owner_params},
        )
        if row is None:
            raise NotFound("API key not found")
        await audit(conn, key, "api_key.revoke", str(key_id), {"prefix": row["prefix"]})
    return Response(status_code=204)
