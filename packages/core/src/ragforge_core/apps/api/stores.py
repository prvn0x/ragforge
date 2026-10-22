"""SQL implementations of the auth service's persistence ports (services/auth/ports.py)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import DBAPIError

from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.sql import audit, fetch_one, insert_api_key, set_tenant, sqlstate
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.services.auth.ports import (
    KeyRecord,
    NewTenant,
    OwnerKey,
    ProvisionedTenant,
)
from ragforge_core.services.auth.scopes import ClientKind, client_kind_for_name

# Data-rule violations raised by provision_tenant (22023) or by a table CHECK (23514, e.g.
# api_keys_mcp_never_privileged) are the caller's fault: map them to a 422 instead of letting a
# raw DBAPIError (whose message includes bound parameters such as a key digest) escape.
_CLIENT_ERROR_STATES = frozenset({"22023", "23514"})


class SqlKeyStore:
    def __init__(self, txm: TransactionManager) -> None:
        self._txm = txm

    async def authenticate(self, prefix: str, sha256: bytes) -> KeyRecord | None:
        # A system transaction: the tenant is unknown until the key is resolved, and the
        # definer function is the only (named, audited) cross-tenant path to api_keys.
        async with self._txm.system() as tx:
            row = await fetch_one(
                connection(tx),
                "SELECT api_key_id, tenant_id, user_id, scopes, client_kind "
                "FROM rf.authenticate_api_key(:prefix, :sha256)",
                {"prefix": prefix, "sha256": sha256},
            )
        if row is None:
            return None
        return KeyRecord(
            key_id=row["api_key_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            scopes=tuple(row["scopes"]),
            client_kind=ClientKind.mcp if row["client_kind"] == "mcp" else ClientKind.user,
        )


class SqlProvisioningStore:
    def __init__(self, txm: TransactionManager) -> None:
        self._txm = txm

    async def provision(
        self, tenant: NewTenant, owner_key: OwnerKey, *, actor: str, actor_tenant: UUID | None
    ) -> ProvisionedTenant:
        """One transaction: tenant + owner user (definer) -> owner key (RLS, re-scoped to the
        new tenant) -> audit in the actor's tenant. Either everything commits or nothing does,
        so a tenant never exists without its first key."""
        try:
            async with self._txm.system() as tx:
                conn = connection(tx)
                created = await fetch_one(
                    conn,
                    "SELECT tenant_id, user_id FROM rf.provision_tenant(:n, :k, :o, :c)",
                    {
                        "n": tenant.name,
                        "k": tenant.kind,
                        "o": tenant.owner_display_name,
                        "c": tenant.team_visibility_consent,
                    },
                )
                assert created is not None
                tenant_id: UUID = created["tenant_id"]
                user_id: UUID = created["user_id"]
                await set_tenant(conn, tenant_id)
                key_row = await insert_api_key(
                    conn,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    name=owner_key.name,
                    key=owner_key.key,
                    scopes=owner_key.scopes,
                    client_kind=client_kind_for_name(owner_key.name),
                )
                if actor_tenant is not None:
                    await set_tenant(conn, actor_tenant)
                    await audit(
                        conn,
                        actor,
                        "tenant.provision",
                        str(tenant_id),
                        {"kind": tenant.kind},
                        tenant_id=actor_tenant,
                    )
        except DBAPIError as exc:
            if sqlstate(exc) in _CLIENT_ERROR_STATES:
                raise ValidationFailed("invalid tenant request") from None
            raise
        return ProvisionedTenant(tenant_id=tenant_id, user_id=user_id, key_id=key_row["id"])
