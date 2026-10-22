"""Bearer-token authentication: header -> Principal."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ragforge_core.domain.errors import Unauthorized
from ragforge_core.domain.models import Principal
from ragforge_core.services.auth.keys import parse_key
from ragforge_core.services.auth.ports import KeyStore
from ragforge_core.services.auth.scopes import ClientKind, effective_scopes

# One message for every failure mode: callers must not learn whether a prefix exists, a key was
# revoked, or the secret was wrong.
_INVALID = "missing, malformed, revoked or unknown API key"


@dataclass(frozen=True, slots=True)
class AuthenticatedKey:
    principal: Principal
    key_id: UUID
    prefix: str
    client_kind: ClientKind

    @property
    def actor(self) -> str:
        """Audit actor string: identifies the key without any secret material."""
        return f"key:{self.prefix}"


class Authenticator:
    def __init__(self, store: KeyStore) -> None:
        self._store = store

    async def authenticate(self, bearer_token: str | None) -> AuthenticatedKey:
        parsed = parse_key(bearer_token) if bearer_token else None
        if parsed is None:
            raise Unauthorized(_INVALID)
        record = await self._store.authenticate(parsed.prefix, parsed.sha256)
        if record is None:
            raise Unauthorized(_INVALID)
        principal = Principal(
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            scopes=effective_scopes(record.scopes, record.client_kind),
        )
        return AuthenticatedKey(
            principal=principal,
            key_id=record.key_id,
            prefix=parsed.prefix,
            client_kind=record.client_kind,
        )
