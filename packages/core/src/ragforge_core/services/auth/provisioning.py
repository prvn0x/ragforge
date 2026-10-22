"""Tenant provisioning use case, shared by POST /v1/tenants and the `ragforge-admin` CLI."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.services.auth.keys import GeneratedKey, generate_key
from ragforge_core.services.auth.ports import NewTenant, OwnerKey, ProvisioningStore
from ragforge_core.services.auth.scopes import (
    OWNER_KEY_SCOPES,
    check_eval_scopes,
    check_mcp_scopes,
    client_kind_for_name,
)

TENANT_KINDS = frozenset({"personal", "team"})


@dataclass(frozen=True, slots=True)
class ProvisionResult:
    tenant_id: UUID
    user_id: UUID
    key_id: UUID
    key: GeneratedKey
    key_name: str
    scopes: frozenset[str]


def check_tenant_request(tenant: NewTenant) -> None:
    """Fail fast with a clear 422. The definer function enforces the same rules authoritatively."""
    if tenant.kind not in TENANT_KINDS:
        raise ValidationFailed("tenant kind must be personal or team")
    if tenant.kind == "team" and not tenant.team_visibility_consent:
        raise ValidationFailed(
            "team tenants require team_visibility_consent=true: team members can see each "
            "other's indexed sources"
        )


async def provision_tenant(
    store: ProvisioningStore,
    tenant: NewTenant,
    *,
    actor: str,
    actor_tenant: UUID | None = None,
    owner_scopes: frozenset[str] = OWNER_KEY_SCOPES,
    key_name: str = "owner",
) -> ProvisionResult:
    check_tenant_request(tenant)
    # The owner-key name is caller input (CLI `--key-name`): an MCP-classified name must not
    # carry human-only scopes. Checked here so the caller gets a 422, not the DB CHECK's 23514.
    kind = client_kind_for_name(key_name)
    check_mcp_scopes(kind, owner_scopes)
    check_eval_scopes(kind, owner_scopes)  # ADR-0017: never eval on an MCP-named owner key
    key = generate_key()
    owner = OwnerKey(name=key_name, key=key, scopes=owner_scopes)
    created = await store.provision(tenant, owner, actor=actor, actor_tenant=actor_tenant)
    return ProvisionResult(
        tenant_id=created.tenant_id,
        user_id=created.user_id,
        key_id=created.key_id,
        key=key,
        key_name=key_name,
        scopes=owner_scopes,
    )
