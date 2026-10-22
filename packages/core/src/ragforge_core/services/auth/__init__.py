"""Authentication and authorization: API keys, principals, scopes, tenant provisioning.

Pure policy; persistence is behind `ports` (the API app implements them with SQL)."""

from ragforge_core.services.auth.keys import GeneratedKey, ParsedKey, generate_key, parse_key
from ragforge_core.services.auth.principal import AuthenticatedKey, Authenticator
from ragforge_core.services.auth.scopes import ClientKind, client_kind_for_name, require_scope

__all__ = [
    "AuthenticatedKey",
    "Authenticator",
    "ClientKind",
    "GeneratedKey",
    "ParsedKey",
    "client_kind_for_name",
    "generate_key",
    "parse_key",
    "require_scope",
]
