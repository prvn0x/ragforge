"""Authentication and scopes through the HTTP API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import httpx
import psycopg
import pytest

from module_e2.e2_support import ALL, Fakes, TenantWorld, auth, issue_key
from ragforge_core.services.auth.keys import generate_key

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


KEY_MANAGER = "approvals:write|admin"  # either scope suffices (tenant self-administration)


def _endpoints(w: TenantWorld) -> list[tuple[str, str, Any, str]]:
    """(method, url, body, required scope or `a|b` alternatives) for every /v1 endpoint."""
    tenant = {"name": "t", "owner_display_name": "o"}
    connector = {"kind": "notion", "target_collection_id": str(w.collection),
                 "display_name": "n", "token": "secret_tok"}  # fmt: skip
    return [
        ("POST", "/v1/tenants", tenant, "admin"),
        ("POST", "/v1/api-keys", {"name": "k", "scopes": ["read"]}, KEY_MANAGER),
        ("GET", "/v1/api-keys", None, KEY_MANAGER),
        ("DELETE", f"/v1/api-keys/{w.tenant}", None, KEY_MANAGER),  # not the caller itself
        ("POST", "/v1/collections", {"name": "c2"}, "write"),
        ("GET", "/v1/collections", None, "read"),
        ("GET", f"/v1/collections/{w.collection}", None, "read"),
        ("POST", "/v1/connectors", connector, "write"),
        ("GET", "/v1/connectors", None, "read"),
        ("GET", f"/v1/connectors/{w.connector}", None, "read"),
        ("POST", f"/v1/connectors/{w.connector}/sync", None, "write"),
        ("DELETE", f"/v1/connectors/{w.connector}", None, "write"),
        ("GET", "/v1/documents", None, "read"),
        ("DELETE", f"/v1/documents/{w.document}", None, "write"),
        ("GET", "/v1/exclusions", None, "read"),
        ("DELETE", f"/v1/exclusions/{w.exclusion}", None, "write"),
        ("GET", f"/v1/jobs/{w.job}", None, "read"),
        ("GET", f"/v1/jobs/{w.job}/events", None, "read"),
        ("POST", "/v1/query", {"query": "q", "collection_ids": [str(w.collection)]}, "read"),
        ("POST", "/v1/chat", {"collection_ids": [str(w.collection)], "message": "m"}, "chat"),
        ("POST", f"/v1/approvals/{w.run}", {"decisions": {"call-1": "approve"}},
         "approvals:write"),
        ("GET", "/v1/audit", None, KEY_MANAGER),
    ]  # fmt: skip


async def test_each_endpoint_requires_its_scope(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    for method, url, body, scope in _endpoints(w):
        required = scope.split("|")
        others = [s for s in ALL if s not in required]
        key = issue_key(superuser, w.tenant, w.user, others)
        r = await http.request(method, url, json=body, headers=auth(key))
        assert r.status_code == 403, (method, url, r.status_code, r.text)
        assert all(s in r.json()["detail"] for s in required)
    assert fakes.jobs.enqueued == []
    assert not fakes.agent.called


async def test_missing_malformed_unknown_and_revoked_keys_are_401(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    revoked = issue_key(superuser, w.tenant, w.user, revoked=True)
    candidates: list[dict[str, str]] = [
        {},
        {"authorization": "Basic dXNlcjpwYXNz"},
        {"authorization": "Bearer not-a-key"},
        auth(generate_key().plaintext),  # well-formed, never issued
        auth(revoked),
        auth(w.key[:-1] + ("A" if w.key[-1] != "A" else "B")),  # right prefix, wrong secret
    ]
    bodies = set()
    for headers in candidates:
        for method, url, body, _ in _endpoints(w):
            r = await http.request(method, url, json=body, headers=headers)
            assert r.status_code == 401, (headers, method, url, r.text)
            assert r.headers["www-authenticate"] == 'Bearer realm="ragforge"'
            bodies.add(r.json()["detail"])
    assert len(bodies) == 1  # one indistinguishable answer for every failure mode


async def test_revocation_takes_effect_immediately(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    victim = issue_key(superuser, w.tenant, w.user, ["read"])
    assert (await http.get("/v1/collections", headers=auth(victim))).status_code == 200
    listing = (await http.get("/v1/api-keys", headers=auth(w.key))).json()["items"]
    [victim_id] = [k["id"] for k in listing if victim.startswith(f"rf_{k['prefix']}_")]
    r = await http.delete(f"/v1/api-keys/{victim_id}", headers=auth(w.key))
    assert r.status_code == 204
    assert (await http.get("/v1/collections", headers=auth(victim))).status_code == 401
    assert (await http.delete(f"/v1/api-keys/{victim_id}", headers=auth(w.key))).status_code == 204


async def test_mcp_keys_can_never_approve(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    # 1. The API refuses to mint an MCP key with approvals:write or admin, however the name
    #    spells `mcp`, and refuses those scopes to any name that does not declare a human client.
    for name in ("mcp:laptop", "MCP server", "claude-mcp", "ui:mcp", "cursor", "claude-desktop"):
        for scopes in (["read", "approvals:write"], ["admin"]):
            r = await http.post(
                "/v1/api-keys", json={"name": name, "scopes": scopes}, headers=auth(w.key)
            )
            assert r.status_code == 422, (name, r.text)
    # 2. ...an MCP key with every scope it may hold still cannot approve...
    r = await http.post(
        "/v1/api-keys",
        json={"name": "mcp:laptop", "scopes": ["read", "write", "chat"]},
        headers=auth(w.key),
    )
    assert r.status_code == 201
    mcp_key = r.json()["secret"]
    r = await http.post(
        f"/v1/approvals/{w.run}", json={"decisions": {"call-1": "approve"}}, headers=auth(mcp_key)
    )
    assert r.status_code == 403
    # 3. ...and the database refuses such a row even if the API were bypassed.
    with pytest.raises(psycopg.errors.CheckViolation):
        issue_key(superuser, w.tenant, w.user, ALL, name="mcp", client_kind="mcp")
    assert not fakes.agent.called


async def test_api_keys_cannot_escalate(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    limited_admin = issue_key(superuser, w.tenant, w.user, ["admin", "read"])
    r = await http.post(
        "/v1/api-keys", json={"name": "k", "scopes": ["write"]}, headers=auth(limited_admin)
    )
    assert r.status_code == 403
    r = await http.post(
        "/v1/api-keys", json={"name": "k", "scopes": ["read"]}, headers=auth(limited_admin)
    )
    assert r.status_code == 201
    created = r.json()
    assert created["scopes"] == ["read"]
    assert r.headers["cache-control"] == "no-store"
    # The secret works, is shown once, and is never listed again.
    assert (await http.get("/v1/collections", headers=auth(created["secret"]))).status_code == 200
    listing = (await http.get("/v1/api-keys", headers=auth(w.key))).text
    assert created["secret"] not in listing
    assert "secret" not in listing


async def _mint(http: httpx.AsyncClient, key: str, name: str, scopes: list[str]) -> dict[str, Any]:
    r = await http.post("/v1/api-keys", json={"name": name, "scopes": scopes}, headers=auth(key))
    assert r.status_code == 201, r.text
    created: dict[str, Any] = r.json()
    return created


async def test_mcp_named_key_is_stored_as_mcp(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    created = await _mint(http, w.key, "Claude MCP server", ["read", "chat"])
    assert superuser.execute(
        "SELECT client_kind FROM rf.api_keys WHERE id = %s", (created["id"],)
    ).fetchone() == ("mcp",)


async def test_owner_scoped_key_administers_its_own_keys_without_admin(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    """A provisioned owner key (no `admin`) can mint an MCP key, list and revoke keys, and read
    the audit log; it cannot provision tenants."""
    w = world()
    owner = issue_key(superuser, w.tenant, w.user, ["read", "write", "chat", "approvals:write"])
    mcp = await _mint(http, owner, "mcp:laptop", ["read", "chat"])
    ui = await _mint(http, owner, "ui:phone", ["read", "approvals:write"])
    listing = (await http.get("/v1/api-keys", headers=auth(owner))).json()["items"]
    assert {mcp["id"], ui["id"]} <= {k["id"] for k in listing}
    assert (await http.delete(f"/v1/api-keys/{mcp['id']}", headers=auth(owner))).status_code == 204
    assert (await http.get("/v1/collections", headers=auth(mcp["secret"]))).status_code == 401
    audit = (await http.get("/v1/audit", headers=auth(owner))).json()["items"]
    assert [a["action"] for a in audit[:3]] == [
        "api_key.revoke",
        "api_key.create",
        "api_key.create",
    ]
    r = await http.post(
        "/v1/tenants", json={"name": "x", "owner_display_name": "y"}, headers=auth(owner)
    )
    assert r.status_code == 403


async def test_any_key_can_revoke_itself_but_no_other_key(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    mcp = await _mint(http, w.key, "mcp", ["read"])
    sibling = await _mint(http, w.key, "mcp-2", ["read"])
    r = await http.delete(f"/v1/api-keys/{sibling['id']}", headers=auth(mcp["secret"]))
    assert r.status_code == 403
    r = await http.delete(f"/v1/api-keys/{mcp['id']}", headers=auth(mcp["secret"]))
    assert r.status_code == 204
    assert (await http.get("/v1/collections", headers=auth(mcp["secret"]))).status_code == 401
    assert (await http.get("/v1/collections", headers=auth(sibling["secret"]))).status_code == 200


async def test_non_admin_key_managers_only_see_their_own_users_keys(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    """In a team tenant one member's approvals key cannot list or revoke another member's keys
    (404, like any invisible id); a tenant `admin` key sees them all."""
    w = world()
    row = superuser.execute(
        "INSERT INTO rf.users (tenant_id, display_name) VALUES (%s, 'Bob') RETURNING id",
        (w.tenant,),
    ).fetchone()
    assert row is not None
    other_user = UUID(str(row[0]))
    member = issue_key(superuser, w.tenant, w.user, ["read", "approvals:write"])
    bob = issue_key(superuser, w.tenant, other_user, ["read", "approvals:write"], name="bob")
    bob_id = next(
        k["id"]
        for k in (await http.get("/v1/api-keys", headers=auth(w.key))).json()["items"]
        if k["name"] == "bob"
    )
    member_view = (await http.get("/v1/api-keys", headers=auth(member))).json()["items"]
    assert bob_id not in {k["id"] for k in member_view}
    assert (await http.delete(f"/v1/api-keys/{bob_id}", headers=auth(member))).status_code == 404
    assert (await http.get("/v1/collections", headers=auth(bob))).status_code == 200
    assert (await http.delete(f"/v1/api-keys/{bob_id}", headers=auth(w.key))).status_code == 204
