"""Tenant isolation through the HTTP API: another tenant's ids are always 404 (never 403, which
would confirm they exist), lists never show them, and nothing of theirs is modified."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest

from module_e2.e2_support import Fakes, TenantWorld, auth

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


def _key_id(superuser: Conn, plaintext: str) -> UUID:
    row = superuser.execute(
        "SELECT id FROM rf.api_keys WHERE prefix = %s", (plaintext.split("_")[1],)
    ).fetchone()
    assert row is not None
    return row[0]  # type: ignore[return-value]


def _foreign_requests(a: TenantWorld, b: TenantWorld, b_key_id: UUID) -> list[tuple[str, str, Any]]:
    """(method, url, json body) for every endpoint that takes an id, aimed at tenant B."""
    return [
        ("GET", f"/v1/collections/{b.collection}", None),
        ("GET", f"/v1/connectors/{b.connector}", None),
        ("POST", f"/v1/connectors/{b.connector}/sync", None),
        ("DELETE", f"/v1/connectors/{b.connector}", None),
        ("POST", "/v1/connectors", {"kind": "notion", "target_collection_id": str(b.collection),
                                     "display_name": "x", "token": "secret_tok"}),
        ("GET", f"/v1/documents?collection_id={b.collection}", None),
        ("DELETE", f"/v1/documents/{b.document}", None),
        ("DELETE", f"/v1/documents/{b.document}?forget_exclusion=true", None),
        ("DELETE", f"/v1/exclusions/{b.exclusion}", None),
        ("GET", f"/v1/jobs/{b.job}", None),
        ("GET", f"/v1/jobs/{b.history_job}", None),
        ("GET", f"/v1/jobs/{b.job}/events", None),
        ("DELETE", f"/v1/api-keys/{b_key_id}", None),
        ("POST", "/v1/query", {"query": "q", "collection_ids": [str(b.collection)]}),
        ("POST", "/v1/query", {"query": "q",
                               "collection_ids": [str(a.collection), str(b.collection)]}),
        ("POST", "/v1/chat", {"collection_ids": [str(b.collection)], "message": "hi"}),
        ("POST", "/v1/chat", {"collection_ids": [str(a.collection)], "message": "hi",
                              "conversation_id": str(b.conversation)}),
        ("POST", f"/v1/approvals/{b.run}", {"decisions": {"call-1": "approve"}}),
    ]  # fmt: skip


async def test_every_endpoint_answers_404_for_another_tenants_ids(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    a, b = world(), world()
    for method, url, body in _foreign_requests(a, b, _key_id(superuser, b.key)):
        r = await http.request(method, url, json=body, headers=auth(a.key))
        assert r.status_code == 404, (method, url, r.status_code, r.text)
        assert r.headers["content-type"].startswith("application/problem+json")
        # Our domain NotFound (with a detail), not the router's "no such route" 404.
        assert r.json()["title"] == "Not found", (method, url, r.text)
        assert r.json()["detail"].endswith("not found"), (method, url, r.text)

    # Nothing of B's changed, nothing was enqueued, the agent never ran.
    assert superuser.execute(
        "SELECT status, secret_ciphertext IS NULL FROM rf.connectors WHERE id = %s", (b.connector,)
    ).fetchone() == ("active", True)
    assert superuser.execute(
        "SELECT deleted_at FROM rf.documents WHERE id = %s", (b.document,)
    ).fetchone() == (None,)
    assert superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE id = %s", (b.exclusion,)
    ).fetchone() == (1,)
    assert superuser.execute(
        "SELECT revoked_at FROM rf.api_keys WHERE prefix = %s", (b.key.split("_")[1],)
    ).fetchone() == (None,)
    assert superuser.execute(
        "SELECT count(*) FROM rf.connectors WHERE target_collection_id = %s", (b.collection,)
    ).fetchone() == (1,)
    assert fakes.jobs.enqueued == []
    assert not fakes.agent.called
    assert fakes.retriever.calls == []


async def test_unknown_ids_are_404_too(http: httpx.AsyncClient, world: World) -> None:
    """Same answer for 'does not exist' and 'belongs to someone else'."""
    a = world()
    r = await http.get(f"/v1/connectors/{uuid4()}", headers=auth(a.key))
    assert r.status_code == 404


async def test_lists_never_show_another_tenants_rows(http: httpx.AsyncClient, world: World) -> None:
    a, b = world(), world()
    b_ids = {str(v) for v in vars(b).values() if isinstance(v, UUID)}
    for w in (a, b):  # gives both tenants audit rows
        r = await http.post("/v1/collections", json={"name": "extra"}, headers=auth(w.key))
        assert r.status_code == 201
    for url in ("/v1/collections", "/v1/connectors", "/v1/documents", "/v1/exclusions",
                "/v1/api-keys", "/v1/audit"):  # fmt: skip
        r = await http.get(url, headers=auth(a.key))
        assert r.status_code == 200, url
        ids = {str(item["id"]) for item in r.json()["items"]}
        assert ids, url  # A's own rows are visible...
        assert not ids & b_ids, url  # ...B's never are


async def test_conversation_of_another_user_in_same_tenant_is_404(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    """LangGraph tables are outside RLS, so the API's user check is the isolation boundary."""
    from module_e2.e2_support import issue_key

    a = world()
    row = superuser.execute(
        "INSERT INTO rf.users (tenant_id, display_name) VALUES (%s, 'mallory') RETURNING id",
        (a.tenant,),
    ).fetchone()
    assert row is not None
    mallory = issue_key(superuser, a.tenant, row[0])  # type: ignore[arg-type]
    chat = {"collection_ids": [str(a.collection)], "message": "hi",
            "conversation_id": str(a.conversation)}  # fmt: skip
    r = await http.post("/v1/chat", json=chat, headers=auth(mallory))
    assert r.status_code == 404
    r = await http.post(
        f"/v1/approvals/{a.run}", json={"decisions": {"call-1": "approve"}}, headers=auth(mallory)
    )
    assert r.status_code == 404
