"""Module-local persistence ports for authentication and provisioning (docs/PHASE1.md:
services never import the database; the API app implements these with SQL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ragforge_core.services.auth.keys import GeneratedKey
from ragforge_core.services.auth.scopes import ClientKind


@dataclass(frozen=True, slots=True)
class KeyRecord:
    """What `authenticate_api_key` returns for a matching, unrevoked key."""

    key_id: UUID
    tenant_id: UUID
    user_id: UUID
    scopes: tuple[str, ...]
    client_kind: ClientKind


@dataclass(frozen=True, slots=True)
class NewTenant:
    name: str
    kind: str  # personal | team
    owner_display_name: str
    team_visibility_consent: bool


@dataclass(frozen=True, slots=True)
class OwnerKey:
    name: str
    key: GeneratedKey
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class ProvisionedTenant:
    tenant_id: UUID
    user_id: UUID
    key_id: UUID


class KeyStore(Protocol):
    async def authenticate(self, prefix: str, sha256: bytes) -> KeyRecord | None:
        """Constant-time lookup (definer function). None for unknown, wrong or revoked keys."""
        ...


class ProvisioningStore(Protocol):
    async def provision(
        self, tenant: NewTenant, owner_key: OwnerKey, *, actor: str, actor_tenant: UUID | None
    ) -> ProvisionedTenant:
        """Create tenant + owner user + owner key atomically (one transaction). When
        `actor_tenant` is given, also audit the action in the actor's own tenant."""
        ...
