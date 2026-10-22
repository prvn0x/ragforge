"""Query, collections, documents, exclusions, audit, pagination and tenant provisioning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from uuid import UUID

import httpx
import psycopg
import pytest

from module_e2.e2_support import Fakes, TenantWorld, auth, make_hit
from pgtest import connect
from ragforge_core.apps.api.admin import main as admin_main
from ragforge_core.apps.api.stores import SqlProvisioningStore
from ragforge_core.config import Settings
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.services.auth.keys import generate_key
from ragforge_core.services.auth.ports import NewTenant, OwnerKey

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


# ----------------------------------------------------------------- query
async def test_query_embeds_outside_any_transaction(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.retriever.hits = [make_hit(w.collection, n=i) for i in range(3)]
    fakes.retriever.hits[2] = make_hit(w.collection, url="javascript:alert(1)", n=2)
    body = {"query": "what?", "collection_ids": [str(w.collection), str(w.collection)], "k": 2}
    r = await http.post("/v1/query", json=body, headers=auth(w.key))
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert [h["citation"]["anchor"] for h in hits] == ["block-0", "block-1"]
    assert hits[0]["rerank_score"] == 0.9
    # one embedding per active space, each computed while NO transaction was open
    assert set(fakes.embedders.by_space) == {1, 2}
    assert all(e.open_tx_during_calls == [0] for e in fakes.embedders.by_space.values())
    # candidates for the reranker (rerank_top_n), reranked with no transaction open, then cut to k
    [(query, vectors, colls, k)] = fakes.retriever.calls
    assert (query, set(vectors), colls, k) == ("what?", {1, 2}, [w.collection], 20)
    assert fakes.reranker.open_tx_during_calls == [0]
    assert fakes.reranker.pools == [3]


async def test_citation_url_dropped_when_not_http(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.retriever.hits = [make_hit(w.collection, url="javascript:alert(1)")]
    r = await http.post(
        "/v1/query", json={"query": "q", "collection_ids": [str(w.collection)]}, headers=auth(w.key)
    )
    assert r.json()["hits"][0]["citation"]["url"] is None


# ----------------------------------------------------------------- collections + pagination
async def test_collections_create_conflict_and_cursor_pagination(
    http: httpx.AsyncClient, world: World, superuser: Conn
) -> None:
    w = world()
    # New collections land on the NEWEST active space (other modules' tests add spaces to the
    # shared database, so the id is read, not assumed to be the seeded 1).
    newest = superuser.execute(
        "SELECT max(id) FROM rf.embedding_spaces WHERE status = 'active'"
    ).fetchone()
    assert newest is not None
    names = [f"c-{i:02d}" for i in range(7)]
    for name in names:
        r = await http.post("/v1/collections", json={"name": name}, headers=auth(w.key))
        assert r.status_code == 201
        assert r.json()["active_space_id"] == newest[0]
    r = await http.post("/v1/collections", json={"name": names[0]}, headers=auth(w.key))
    assert r.status_code == 409

    seen: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 3, **({"cursor": cursor} if cursor else {})}
        page = (await http.get("/v1/collections", params=params, headers=auth(w.key))).json()
        assert len(page["items"]) <= 3
        seen += [c["name"] for c in page["items"]]
        cursor = page.get("next_cursor")
        if not cursor:
            break
    assert len(seen) == len(set(seen)) == 8  # 7 + the world's own collection, no duplicates
    assert set(names) <= set(seen)

    for bad in ("garbage!", "WyJ4Il0", "W10"):
        r = await http.get("/v1/collections", params={"cursor": bad}, headers=auth(w.key))
        assert r.status_code == 422, bad


# ----------------------------------------------------------------- documents & exclusions
async def test_delete_document_excludes_and_tombstones_atomically(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    docs = (
        await http.get(f"/v1/documents?collection_id={w.collection}", headers=auth(w.key))
    ).json()
    assert [d["id"] for d in docs["items"]] == [str(w.document)]

    assert (
        await http.delete(f"/v1/documents/{w.document}", headers=auth(w.key))
    ).status_code == 204
    ext = superuser.execute(
        "SELECT external_id, deleted_at IS NOT NULL FROM rf.documents WHERE id = %s", (w.document,)
    ).fetchone()
    assert ext is not None
    assert ext[1] is True
    assert superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND external_id = %s",
        (w.tenant, ext[0]),
    ).fetchone() == (1,)
    assert (await http.get("/v1/documents", headers=auth(w.key))).json()["items"] == []
    assert (
        await http.delete(f"/v1/documents/{w.document}", headers=auth(w.key))
    ).status_code == 404

    # undo: the exclusion goes away, so the next sync may index the page again
    r = await http.delete(
        f"/v1/documents/{w.document}", params={"forget_exclusion": "true"}, headers=auth(w.key)
    )
    assert r.status_code == 204
    assert superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND external_id = %s",
        (w.tenant, ext[0]),
    ).fetchone() == (0,)


async def test_exclusions_list_and_delete(http: httpx.AsyncClient, world: World) -> None:
    w = world()
    items = (await http.get("/v1/exclusions", headers=auth(w.key))).json()["items"]
    assert [e["id"] for e in items] == [str(w.exclusion)]
    assert (
        await http.delete(f"/v1/exclusions/{w.exclusion}", headers=auth(w.key))
    ).status_code == 204
    assert (await http.get("/v1/exclusions", headers=auth(w.key))).json()["items"] == []
    audit = (await http.get("/v1/audit", headers=auth(w.key))).json()["items"]
    assert audit[0]["action"] == "exclusion.delete"
    assert audit[0]["actor"] == f"key:{w.key.split('_')[1]}"


# ----------------------------------------------------------------- provisioning (API + CLI)
async def test_admin_provisions_tenant_with_working_owner_key(
    http: httpx.AsyncClient, world: World
) -> None:
    w = world()
    r = await http.post(
        "/v1/tenants",
        json={"name": "acme", "owner_display_name": "Ada", "kind": "team"},
        headers=auth(w.key),
    )
    assert r.status_code == 422  # team without consent
    r = await http.post(
        "/v1/tenants",
        json={"name": "acme", "owner_display_name": "Ada", "kind": "team",
              "team_visibility_consent": True},
        headers=auth(w.key),
    )  # fmt: skip
    assert r.status_code == 201, r.text
    assert r.headers["cache-control"] == "no-store"
    created = r.json()
    assert created["tenant"]["kind"] == "team"
    owner_key = created["api_key"]["secret"]
    assert "admin" not in created["api_key"]["scopes"]
    # The new key works, sees only its own (empty) tenant, and cannot provision further.
    r = await http.get("/v1/collections", headers=auth(owner_key))
    assert (r.status_code, r.json()["items"]) == (200, [])
    assert (await http.post("/v1/tenants", json={"name": "x", "owner_display_name": "y"},
                            headers=auth(owner_key))).status_code == 403  # fmt: skip
    audit = (await http.get("/v1/audit", headers=auth(w.key))).json()["items"]
    assert (audit[0]["action"], audit[0]["target"]) == ("tenant.provision", created["tenant"]["id"])


def test_admin_cli_bootstraps_first_tenant(
    settings: Settings, superuser: Conn, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["create-tenant", "--name", "first", "--owner", "Root", "--platform-admin"]
    assert admin_main(argv, settings=settings) == 0
    out = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    tenant, key = UUID(out["tenant_id"]), out["api_key"]
    assert "admin" in out["scopes"].split(",")
    row = superuser.execute(
        "SELECT tenant_id, key_sha256 FROM rf.api_keys WHERE prefix = %s", (key.split("_")[1],)
    ).fetchone()
    assert row is not None
    assert row[0] == tenant
    assert key.encode() not in cast(bytes, row[1])
    with connect(settings, "api") as conn:
        from ragforge_core.services.auth.keys import parse_key

        parsed = parse_key(key)
        assert parsed is not None
        authed = conn.execute(
            "SELECT tenant_id FROM rf.authenticate_api_key(%s, %s)", (parsed.prefix, parsed.sha256)
        ).fetchall()
    assert authed == [(tenant,)]
    # team without consent fails cleanly, writes nothing
    code = admin_main(
        ["create-tenant", "--name", "t2", "--owner", "o", "--kind", "team"], settings=settings
    )
    assert code == 2
    assert superuser.execute("SELECT count(*) FROM rf.tenants WHERE name = 't2'").fetchone() == (0,)


def test_admin_cli_issues_platform_admin_only_on_request(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert admin_main(["create-tenant", "--name", "plain", "--owner", "o"], settings=settings) == 0
    out = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert out["scopes"].split(",") == ["approvals:write", "chat", "read", "write"]


def test_admin_cli_mints_a_least_privilege_key_for_an_existing_tenant(
    settings: Settings, superuser: Conn, capsys: pytest.CaptureFixture[str]
) -> None:
    assert admin_main(["create-tenant", "--name", "keyed", "--owner", "o"], settings=settings) == 0
    first = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    tenant = first["tenant_id"]

    assert admin_main(["create-key", "--tenant", tenant, "--name", "eval"], settings=settings) == 0
    out = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert out["user_id"] == first["user_id"]  # defaults to the tenant's owner
    prefix = out["api_key"].split("_")[1]
    row = superuser.execute(
        "SELECT tenant_id::text, scopes FROM rf.api_keys WHERE prefix = %s", (prefix,)
    ).fetchone()
    assert row == (tenant, ["chat", "read"])
    audited = superuser.execute(
        "SELECT actor FROM rf.audit_log "
        "WHERE action = 'api_key.create' AND details->>'prefix' = %s",
        (prefix,),
    ).fetchall()
    assert audited == [("cli:ragforge-admin",)]


@pytest.mark.parametrize("scopes", ["approvals:write", "read,admin", ""])
def test_admin_cli_create_key_never_grants_privileged_scopes(
    settings: Settings, capsys: pytest.CaptureFixture[str], scopes: str
) -> None:
    assert admin_main(["create-tenant", "--name", f"p-{scopes or 'x'}", "--owner", "o"],
                      settings=settings) == 0  # fmt: skip
    tenant = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())["tenant_id"]
    argv = ["create-key", "--tenant", tenant, "--name", "eval", "--scopes", scopes]
    assert admin_main(argv, settings=settings) == 2
    assert "api_key=" not in capsys.readouterr().out


def test_admin_cli_create_key_for_an_unknown_tenant_writes_nothing(
    settings: Settings, superuser: Conn, capsys: pytest.CaptureFixture[str]
) -> None:
    before = superuser.execute("SELECT count(*) FROM rf.api_keys").fetchone()
    argv = ["create-key", "--tenant", "00000000-0000-0000-0000-000000000001", "--name", "eval"]
    assert admin_main(argv, settings=settings) == 2
    assert superuser.execute("SELECT count(*) FROM rf.api_keys").fetchone() == before


@pytest.mark.parametrize("key_name", ["mcp", "claude-mcp"])
def test_admin_cli_rejects_mcp_key_names_cleanly(
    settings: Settings, superuser: Conn, capsys: pytest.CaptureFixture[str], key_name: str
) -> None:
    """An MCP-classified owner key would carry approvals:write: refused with a clean message
    (no traceback, no bound parameters), and nothing is written."""
    name = f"t-{key_name}"
    argv = ["create-tenant", "--name", name, "--owner", "o", "--key-name", key_name]
    assert admin_main([*argv, "--platform-admin"], settings=settings) == 2
    err = capsys.readouterr().err
    assert "MCP keys can never hold" in err
    assert "Traceback" not in err
    assert superuser.execute(
        "SELECT count(*) FROM rf.tenants WHERE name = %s", (name,)
    ).fetchone() == (0,)


async def test_provisioning_store_maps_check_violations_without_parameters(
    fakes: Fakes, superuser: Conn
) -> None:
    """Defense in depth below the use-case pre-check: the DB CHECK (23514) becomes a 422 whose
    message carries no bound parameters (key prefix, digest), and the transaction rolls back."""
    key = generate_key()
    with pytest.raises(ValidationFailed) as info:
        await SqlProvisioningStore(fakes.txm).provision(
            NewTenant(name="t-check", kind="personal", owner_display_name="o",
                      team_visibility_consent=False),
            OwnerKey(name="mcp", key=key, scopes=frozenset({"read", "approvals:write"})),
            actor="test",
            actor_tenant=None,
        )  # fmt: skip
    assert key.prefix not in str(info.value)
    assert info.value.__cause__ is None
    assert superuser.execute(
        "SELECT count(*) FROM rf.tenants WHERE name = 't-check'"
    ).fetchone() == (0,)
