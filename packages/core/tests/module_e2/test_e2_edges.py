"""Edge cases from Phase 2a review round 6: an upload client that disconnects mid-body, and the
embedding space a new collection lands on while two spaces are active (blue/green rollout)."""

from __future__ import annotations

import json
from collections.abc import Callable, MutableMapping
from typing import Any
from uuid import uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import Fakes, TenantWorld, auth

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


async def test_an_upload_client_disconnecting_mid_body_is_a_400_not_a_500(
    app: FastAPI, fakes: Fakes, world: World
) -> None:
    """Starlette raises ClientDisconnect out of the multipart stream. Unmapped, that is an
    unhandled 500 with a traceback in the logs for something the client did."""
    w = world()
    part = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\n'
        b"Content-Type: text/markdown\r\n\r\n# partial"
    )
    messages: list[MutableMapping[str, Any]] = [
        {"type": "http.request", "body": part, "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> MutableMapping[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/v1/collections/{w.collection}/files",
        "raw_path": f"/v1/collections/{w.collection}/files".encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"t"),
            (b"authorization", f"Bearer {w.key}".encode()),
            (b"content-type", b"multipart/form-data; boundary=x"),
            (b"content-length", b"5000"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("t", 80),
        "state": {},
    }
    await app(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert json.loads(body)["title"] == "Client disconnected"
    assert fakes.jobs.enqueued == []


async def test_a_new_collection_lands_on_the_newest_active_space(
    http: httpx.AsyncClient, world: World, superuser: Conn
) -> None:
    """With two active spaces (after a blue/green flip, before the old one retires), the
    rollout target is the newer one; the lowest id would keep the outgoing space alive."""
    w = world()
    row = superuser.execute(
        "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status) "
        "VALUES ('ollama', %s, 4, 'cosine', 'active') RETURNING id",
        (f"test-{uuid4().hex[:10]}",),
    ).fetchone()
    assert row is not None
    newest = int(str(row[0]))
    try:
        r = await http.post(
            "/v1/collections", json={"name": f"c-{uuid4().hex[:8]}"}, headers=auth(w.key)
        )
        assert r.status_code == 201, r.text
        assert r.json()["active_space_id"] == newest
    finally:
        superuser.execute("DELETE FROM rf.collections WHERE active_space_id = %s", (newest,))
        superuser.execute("DELETE FROM rf.embedding_spaces WHERE id = %s", (newest,))
