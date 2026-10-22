"""ragforge_client round trips against the real app (ASGITransport), SSE parsing, error mapping."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from uuid import uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import Fakes, TenantWorld, make_hit, random_token
from ragforge_client import ApiError, RagforgeClient, iter_sse
from ragforge_contracts.api import ApiScope, ApprovalDecision, TenantKind
from ragforge_contracts.sse import DoneEvent, InterruptEvent, TokenEvent
from ragforge_core.domain.models import ToolCall, TurnDone, TurnInterrupt, TurnText

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


def _client(app: FastAPI, key: str) -> RagforgeClient:
    return RagforgeClient("http://test", key, transport=httpx.ASGITransport(app=app))


async def test_sdk_round_trips_every_endpoint(
    app: FastAPI, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    async with _client(app, w.key) as rf:
        # tenants & keys
        provisioned = await rf.provision_tenant(
            "sdk-team", "Ada", kind=TenantKind.team, team_visibility_consent=True
        )
        assert provisioned.api_key.secret.startswith("rf_")
        created = await rf.create_api_key("ui", [ApiScope.read, ApiScope.chat])
        assert sorted(created.scopes) == ["chat", "read"]
        keys = await rf.list_api_keys(limit=1)
        assert len(keys.items) == 1
        assert keys.next_cursor is not None
        more = await rf.list_api_keys(cursor=keys.next_cursor, limit=10)
        assert {k.id for k in keys.items}.isdisjoint({k.id for k in more.items})
        await rf.revoke_api_key(created.id)

        # collections & connectors
        coll = await rf.create_collection("sdk-docs")
        assert (await rf.get_collection(coll.id)).name == "sdk-docs"
        assert coll.id in {c.id for c in (await rf.list_collections()).items}
        token = random_token()
        conn = await rf.create_connector(
            target_collection_id=coll.id, display_name="wiki", token=token
        )
        assert (await rf.get_connector(conn.id)).display_name == "wiki"
        assert conn.id in {c.id for c in (await rf.list_connectors()).items}
        assert (await rf.sync_connector(conn.id)).job_id
        assert (await rf.delete_connector(conn.id)).job_id
        row = superuser.execute(
            "SELECT secret_ciphertext IS NULL FROM rf.connectors WHERE id = %s", (conn.id,)
        ).fetchone()
        assert row == (True,)  # sent in clear over the wire (not "**********"), then destroyed

        # documents, exclusions, jobs
        docs = await rf.list_documents(collection_id=w.collection)
        assert [d.id for d in docs.items] == [w.document]
        await rf.delete_document(w.document)
        await rf.delete_document(w.document, forget_exclusion=True)
        exclusions = await rf.list_exclusions()
        assert w.exclusion in {e.id for e in exclusions.items}
        await rf.delete_exclusion(w.exclusion)
        assert (await rf.get_job(w.history_job)).status == "done"
        events = [e async for e in rf.job_events(w.history_job)]
        assert [str(e.status) for e in events] == ["done"]

        # query & chat & approvals
        fakes.retriever.hits = [make_hit(w.collection)]
        result = await rf.query("q", [w.collection], k=1)
        assert result.hits[0].citation.anchor == "block-0"

        fakes.agent.script = [
            TurnText("hi"),
            TurnDone(conversation_id=uuid4(), usage=(), prompt_version="p"),
        ]
        chat = [e async for e in rf.chat("hello", [w.collection])]
        assert isinstance(chat[0], TokenEvent)
        assert chat[0].text == "<s>hi</s>"
        assert isinstance(chat[-1], DoneEvent)

        fakes.agent.script = [
            TurnInterrupt(run_id=w.run, pending=(ToolCall(id="c", name="t", arguments={}),))
        ]
        [interrupt] = [
            e async for e in rf.chat("again", [w.collection], conversation_id=w.conversation)
        ]
        assert isinstance(interrupt, InterruptEvent)
        fakes.agent.script = [
            TurnDone(conversation_id=w.conversation, usage=(), prompt_version="p")
        ]
        resumed = [e async for e in rf.resume(w.run, {"call-1": ApprovalDecision.approve})]
        assert isinstance(resumed[-1], DoneEvent)

        audit = await rf.list_audit(limit=100)
        actions = {a.action for a in audit.items}
        assert {
            "api_key.create",
            "connector.create",
            "document.delete",
            "approvals.submit",
        } <= actions

    assert token not in str(superuser.execute("SELECT details FROM rf.audit_log").fetchall())


async def test_sdk_maps_problems_to_api_error(app: FastAPI, world: World) -> None:
    w = world()
    async with _client(app, w.key) as rf:
        with pytest.raises(ApiError) as missing:
            await rf.get_collection(uuid4())
        assert (missing.value.status, missing.value.problem.title) == (404, "Not found")
        with pytest.raises(ApiError) as conflict:
            await rf.create_collection(
                next(iter([c.name for c in (await rf.list_collections()).items]))
            )
        assert conflict.value.status == 409
        with pytest.raises(ApiError) as stream_error:
            _ = [e async for e in rf.chat("x", [uuid4()])]
        assert stream_error.value.status == 404
    async with _client(app, "rf_0123456789abcdef_" + "a" * 43) as anon:
        with pytest.raises(ApiError) as unauthorized:
            await anon.list_collections()
        assert unauthorized.value.status == 401


async def _lines(*chunks: str) -> AsyncIterator[str]:
    for chunk in chunks:
        for line in chunk.split("\n"):
            yield line


async def test_sse_parser_skips_unknown_events_and_comments() -> None:
    from ragforge_contracts.sse import decode_sse_data

    raw = (
        ": keepalive\n\n"
        'event: token\ndata: {"type":"token","text":"a"}\n\n'
        'event: future_thing\ndata: {"type":"future_thing","x":1}\n\n'
        'event: token\ndata: {"type":"token",\ndata: "text":"b"}\n\n'
        'event: token\ndata: {"type":"token","text":"c","new_field":1}'  # no final blank line
    )
    messages = [m async for m in iter_sse(_lines(raw))]
    decoded = [decode_sse_data(m.data) for m in messages]
    known = [e for e in decoded if e is not None]
    assert [e.text for e in known if isinstance(e, TokenEvent)] == ["a", "b", "c"]
    assert len(messages) == 4
    assert decoded[1] is None  # unknown event type: skipped, not an error
