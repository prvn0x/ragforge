"""Slot 0006 definer functions: authenticate_api_key and provision_tenant (ADR-0013 rules,
including one cross-tenant test per function)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import UUID

import psycopg
import pytest
from psycopg import errors

from module_e2.e2_support import issue_key
from pgtest import connect
from ragforge_core.config import Settings
from ragforge_core.services.auth.keys import generate_key, parse_key

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
SYSTEM_TENANT = UUID(int=0)


def _auth(settings: Settings, plaintext: str) -> list[tuple[object, ...]]:
    parsed = parse_key(plaintext)
    assert parsed is not None
    return _auth_raw(settings, parsed.prefix, parsed.sha256)


def _auth_raw(settings: Settings, prefix: str, sha: bytes) -> list[tuple[object, ...]]:
    with connect(settings, "api") as conn:  # no tenant set: exactly how the API calls it
        return conn.execute(
            "SELECT api_key_id, tenant_id, user_id, scopes, client_kind "
            "FROM rf.authenticate_api_key(%s, %s)",
            (prefix, sha),
        ).fetchall()


# ----------------------------------------------------------------- authenticate_api_key
def test_valid_key_resolves_its_own_tenant_only(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Cross-tenant: the function sees every tenant's keys, yet returns exactly the matching
    key's tenant/user, never another tenant's, even though the api role has no tenant set."""
    t1, u1, _ = make_tenant()
    t2, u2, _ = make_tenant()
    k1 = issue_key(superuser, t1, u1, ["read"])
    k2 = issue_key(superuser, t2, u2, ["read", "chat"])
    [(_, tenant1, user1, scopes1, kind1)] = _auth(settings, k1)
    [(_, tenant2, user2, scopes2, _)] = _auth(settings, k2)
    assert (tenant1, user1, scopes1, kind1) == (t1, u1, ["read"], None)
    assert (tenant2, user2, sorted(cast(list[str], scopes2))) == (t2, u2, ["chat", "read"])


def test_wrong_secret_rejected_whichever_byte_differs(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, _ = make_tenant()
    parsed = parse_key(issue_key(superuser, tenant, user))
    assert parsed is not None
    for index in (0, 15, 31):
        tampered = bytearray(parsed.sha256)
        tampered[index] ^= 0x01
        assert _auth_raw(settings, parsed.prefix, bytes(tampered)) == []
    assert _auth_raw(settings, parsed.prefix, parsed.sha256[:31]) == []  # wrong length
    other = generate_key()  # unknown prefix
    assert _auth_raw(settings, other.prefix, other.sha256) == []


def test_comparison_is_constant_time_by_construction(superuser: Conn) -> None:
    """Timing is not measurable reliably in CI, so assert the structure instead: the digest is
    never compared with `=` (which short-circuits); every call XORs all 32 bytes, including
    unknown prefixes (dummy digest) and revoked keys (checked only after the loop)."""
    row = superuser.execute(
        "SELECT pg_get_functiondef('rf.authenticate_api_key(text, bytea)'::regprocedure)"
    ).fetchone()
    assert row is not None
    body = str(row[0])
    assert "key_sha256 =" not in body
    assert "= p_sha256" not in body
    assert "FOR i IN 0..31 LOOP" in body
    assert "coalesce(k.key_sha256, decode(repeat('00', 32), 'hex'))" in body
    loop_end = body.index("END LOOP")
    assert body.index("revoked_at IS NOT NULL") > loop_end


def test_revoked_key_rejected(settings: Settings, superuser: Conn, make_tenant: MakeTenant) -> None:
    tenant, user, _ = make_tenant()
    assert _auth(settings, issue_key(superuser, tenant, user, revoked=True)) == []


def test_system_tenant_key_never_authenticates(settings: Settings, superuser: Conn) -> None:
    user = superuser.execute(
        "INSERT INTO rf.users (tenant_id, display_name) VALUES (%s, 'sys') RETURNING id",
        (SYSTEM_TENANT,),
    ).fetchone()
    assert user is not None
    key = issue_key(superuser, SYSTEM_TENANT, user[0])  # type: ignore[arg-type]
    assert _auth(settings, key) == []


def test_last_used_at_is_stamped(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, _ = make_tenant()
    key = issue_key(superuser, tenant, user)
    parsed = parse_key(key)
    assert parsed is not None
    _auth(settings, key)
    row = superuser.execute(
        "SELECT last_used_at FROM rf.api_keys WHERE prefix = %s", (parsed.prefix,)
    ).fetchone()
    assert row is not None
    assert row[0] is not None


def test_mcp_key_cannot_hold_approvals_or_admin(superuser: Conn, make_tenant: MakeTenant) -> None:
    tenant, user, _ = make_tenant()
    for scope in ("approvals:write", "admin"):
        with pytest.raises(errors.CheckViolation):
            issue_key(superuser, tenant, user, ["read", scope], client_kind="mcp")
    issue_key(superuser, tenant, user, ["read", "chat"], client_kind="mcp")


def test_only_api_role_can_execute(settings: Settings) -> None:
    for role in ("worker", "metrics"):
        with connect(settings, role) as conn, pytest.raises(errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.authenticate_api_key('x', '\\x00'::bytea)")
        with connect(settings, role) as conn, pytest.raises(errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.provision_tenant('n', 'personal', 'o', false)")


# ----------------------------------------------------------------- provision_tenant
def _provision(
    settings: Settings, kind: str, consent: bool, name: str = "acme"
) -> tuple[UUID, UUID]:
    with connect(settings, "api") as conn:
        row = conn.execute(
            "SELECT tenant_id, user_id FROM rf.provision_tenant(%s, %s, 'Ada', %s)",
            (name, kind, consent),
        ).fetchone()
        conn.commit()
    assert row is not None
    return row  # type: ignore[return-value]


def test_provision_personal_tenant(settings: Settings, superuser: Conn) -> None:
    tenant, user = _provision(settings, "personal", consent=False)
    assert superuser.execute(
        "SELECT kind, team_visibility_consent_at IS NULL FROM rf.tenants WHERE id = %s", (tenant,)
    ).fetchone() == ("personal", True)
    assert superuser.execute(
        "SELECT tenant_id FROM rf.users WHERE id = %s", (user,)
    ).fetchone() == (tenant,)
    assert superuser.execute(
        "SELECT action FROM rf.audit_log WHERE tenant_id = %s", (tenant,)
    ).fetchall() == [("tenant.provisioned",)]


def test_team_tenant_requires_consent(settings: Settings, superuser: Conn) -> None:
    with pytest.raises(errors.InvalidParameterValue):
        _provision(settings, "team", consent=False, name="no-consent-team")
    assert superuser.execute(
        "SELECT count(*) FROM rf.tenants WHERE name = 'no-consent-team'"
    ).fetchone() == (0,)
    tenant, _ = _provision(settings, "team", consent=True, name="consented-team")
    assert superuser.execute(
        "SELECT team_visibility_consent_at IS NOT NULL FROM rf.tenants WHERE id = %s", (tenant,)
    ).fetchone() == (True,)


@pytest.mark.parametrize("kind", ["system", "", "TEAM"])
def test_provision_rejects_other_kinds(settings: Settings, kind: str) -> None:
    with pytest.raises(errors.InvalidParameterValue):
        _provision(settings, kind, consent=True)


def test_provisioned_tenant_is_invisible_to_other_tenants(
    settings: Settings, make_tenant: MakeTenant
) -> None:
    """Cross-tenant: the definer creates rows in a tenant the caller is not scoped to; a caller
    scoped to any other tenant still sees none of them under RLS."""
    other, _, _ = make_tenant()
    tenant, user = _provision(settings, "personal", consent=False, name="isolated")
    with connect(settings, "api") as conn:
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(other),))
        assert conn.execute(
            "SELECT count(*) FROM rf.tenants WHERE id = %s", (tenant,)
        ).fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM rf.users WHERE id = %s", (user,)).fetchone() == (
            0,
        )
        assert conn.execute(
            "SELECT count(*) FROM rf.audit_log WHERE tenant_id = %s", (tenant,)
        ).fetchone() == (0,)
