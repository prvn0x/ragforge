"""services/auth without a database: key format, scope policy, principal resolution."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from ragforge_core.domain.errors import Forbidden, Unauthorized, ValidationFailed
from ragforge_core.domain.models import Principal, Scope
from ragforge_core.services.auth.keys import generate_key, hash_key, parse_key
from ragforge_core.services.auth.ports import KeyRecord
from ragforge_core.services.auth.principal import Authenticator
from ragforge_core.services.auth.scopes import (
    ClientKind,
    client_kind_for_name,
    declares_human_client,
    effective_scopes,
    is_key_manager,
    require_key_manager,
    require_scope,
    validate_new_key_scopes,
)


def test_generated_keys_parse_back_and_hash_the_full_key() -> None:
    key = generate_key()
    assert key.plaintext.startswith(f"rf_{key.prefix}_")
    assert len(key.sha256) == 32
    parsed = parse_key(key.plaintext)
    assert parsed is not None
    assert (parsed.prefix, parsed.sha256) == (key.prefix, key.sha256)
    assert key.plaintext not in repr(key)


def test_keys_are_unique() -> None:
    keys = [generate_key() for _ in range(200)]
    assert len({k.prefix for k in keys}) == len({k.plaintext for k in keys}) == 200


def test_digest_binds_the_prefix() -> None:
    key = generate_key()
    secret = key.plaintext.split("_", 2)[2]
    other_prefix = f"rf_{'0' * 16}_{secret}"
    assert hash_key(other_prefix) != key.sha256


@pytest.mark.parametrize(
    "raw",
    ["", "Bearer x", "rf_short_secret", "rf_ABCDEF0123456789_" + "a" * 43, "rf_" + "a" * 70,
     "sk_0123456789abcdef_" + "a" * 43, "rf_0123456789abcdef_" + "a" * 42 + "!"],
)  # fmt: skip
def test_malformed_keys_are_rejected_without_lookup(raw: str) -> None:
    assert parse_key(raw) is None


@pytest.mark.parametrize(
    ("name", "kind"),
    [("mcp", ClientKind.mcp), ("MCP:laptop", ClientKind.mcp), ("mcp-claude", ClientKind.mcp),
     ("mcp_x", ClientKind.mcp), ("mcpserver", ClientKind.mcp), ("MCP server", ClientKind.mcp),
     ("claude-mcp", ClientKind.mcp), ("my mcp", ClientKind.mcp), ("ui:mcp", ClientKind.mcp),
     ("ui", ClientKind.user), ("cursor", ClientKind.user), ("claude-desktop", ClientKind.user)],
)  # fmt: skip
def test_client_kind_from_name_fails_closed(name: str, kind: ClientKind) -> None:
    """Any `mcp` in the name marks an MCP key: a false positive only loses human-only scopes."""
    assert client_kind_for_name(name) is kind


@pytest.mark.parametrize(
    ("name", "human"),
    [("ui", True), ("UI:laptop", True), ("user-ada", True), ("user_x", True),
     ("ui:mcp", False), ("cursor", False), ("claude-desktop", False), ("uix", False),
     ("my ui", False), ("owner", False)],
)  # fmt: skip
def test_only_names_that_declare_a_human_client_are_human(name: str, human: bool) -> None:
    assert declares_human_client(name) is human


def _principal(*scopes: str) -> Principal:
    return Principal(tenant_id=uuid4(), user_id=uuid4(), scopes=frozenset(scopes))


def test_mcp_keys_never_get_approvals_or_admin() -> None:
    admin = _principal(*(s.value for s in Scope))
    for name in ("mcp", "MCP server", "claude-mcp", "ui:mcp"):
        for forbidden in ("approvals:write", "admin"):
            with pytest.raises(ValidationFailed):
                validate_new_key_scopes(admin, ["read", forbidden], name)
    assert validate_new_key_scopes(admin, ["read", "chat"], "claude-mcp") == (
        frozenset({"read", "chat"}),
        ClientKind.mcp,
    )
    assert effective_scopes(["read", "approvals:write", "admin"], ClientKind.mcp) == {"read"}


@pytest.mark.parametrize("name", ["cursor", "claude-desktop", "k", "owner"])
def test_privileged_scopes_need_a_declared_human_client(name: str) -> None:
    """Fail closed: a name that says nothing about its client cannot get approvals:write."""
    admin = _principal(*(s.value for s in Scope))
    for privileged in ("approvals:write", "admin"):
        with pytest.raises(ValidationFailed, match="ui:"):
            validate_new_key_scopes(admin, ["read", privileged], name)
    assert validate_new_key_scopes(admin, ["read"], name) == (frozenset({"read"}), ClientKind.user)
    assert validate_new_key_scopes(admin, ["approvals:write"], "ui:laptop") == (
        frozenset({"approvals:write"}),
        ClientKind.user,
    )


def test_no_scope_escalation_and_unknown_scopes_fail_closed() -> None:
    caller = _principal("read", "admin")
    with pytest.raises(Forbidden):
        validate_new_key_scopes(caller, ["write"], "k")
    with pytest.raises(Forbidden):
        validate_new_key_scopes(_principal("approvals:write"), ["admin"], "ui")
    with pytest.raises(ValidationFailed):
        validate_new_key_scopes(caller, ["read", "superuser"], "k")
    with pytest.raises(ValidationFailed):
        validate_new_key_scopes(caller, [], "k")
    assert effective_scopes(["read", "root"], ClientKind.user) == {"read"}


def test_key_manager_is_approvals_or_admin_never_an_mcp_scope_set() -> None:
    assert is_key_manager(_principal("approvals:write"))
    assert is_key_manager(_principal("admin"))
    with pytest.raises(Forbidden):
        require_key_manager(_principal("read", "write", "chat"))
    # An MCP key's effective scopes can never make it a key manager.
    mcp = Principal(uuid4(), uuid4(), effective_scopes([s.value for s in Scope], ClientKind.mcp))
    assert not is_key_manager(mcp)


def test_require_scope() -> None:
    require_scope(_principal("approvals:write"), Scope.approvals_write)
    with pytest.raises(Forbidden):
        require_scope(_principal("read", "write", "chat"), Scope.approvals_write)


class _Store:
    def __init__(self, record: KeyRecord | None) -> None:
        self.record = record
        self.lookups: list[tuple[str, bytes]] = []

    async def authenticate(self, prefix: str, sha256: bytes) -> KeyRecord | None:
        self.lookups.append((prefix, sha256))
        return self.record


async def test_authenticator_builds_principal_and_strips_mcp_scopes() -> None:
    key = generate_key()
    tenant, user = uuid4(), uuid4()
    record = KeyRecord(uuid4(), tenant, user, ("read", "approvals:write"), ClientKind.mcp)
    store = _Store(record)
    authed = await Authenticator(store).authenticate(key.plaintext)
    assert store.lookups == [(key.prefix, key.sha256)]
    assert authed.principal == Principal(tenant, user, frozenset({"read"}))
    assert authed.actor == f"key:{key.prefix}"
    assert key.plaintext not in authed.actor


@pytest.mark.parametrize("token", [None, "", "garbage"])
async def test_authenticator_rejects_malformed_without_db(token: str | None) -> None:
    store = _Store(KeyRecord(uuid4(), UUID(int=1), UUID(int=2), ("read",), ClientKind.user))
    with pytest.raises(Unauthorized):
        await Authenticator(store).authenticate(token)
    assert store.lookups == []


async def test_authenticator_unknown_key_is_unauthorized() -> None:
    with pytest.raises(Unauthorized) as info:
        await Authenticator(_Store(None)).authenticate(generate_key().plaintext)
    assert "revoked" in str(info.value)  # one message for every failure mode
