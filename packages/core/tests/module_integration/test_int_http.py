"""The real HTTP seam: `/v1/chat` -> interrupt -> `/v1/approvals/{run_id}` on the REAL container.

`test_int_composed.py` drives the AgentRunner directly. This file goes through the API instead:
key authentication, the API's own conversation pre-creation (it always hands the runner a
non-null id), the ownership checks of both routes, the approvals audit row, the sanitizer and
the TurnEvent -> SSE mapping. Only the models are scripted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import auth, issue_key
from module_integration.int_support import (
    SUPPORTED,
    FixedEmbedders,
    ScoringReranker,
    ScriptedChat,
    say,
    seed_document,
)
from ragforge_contracts.sse import ChatEvent, DoneEvent, InterruptEvent, decode_sse_data
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.domain.models import ToolCall

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
OWNER_SCOPES = ("read", "write", "chat", "approvals:write")


@pytest.fixture
async def api(settings: Settings) -> AsyncIterator[tuple[httpx.AsyncClient, ScriptedChat]]:
    chat = ScriptedChat()
    container = await build_api(
        settings, chat=chat, reranker=ScoringReranker(), embedders=FixedEmbedders()
    )
    app: FastAPI = create_app(settings, container)
    async with (
        app.router.lifespan_context(app),  # owns the container from here on
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30
        ) as client,
    ):
        yield client, chat


def _events(body: str) -> list[ChatEvent]:
    out: list[ChatEvent] = []
    for block in body.strip().split("\n\n"):
        data = "".join(
            line.removeprefix("data: ") for line in block.splitlines() if line.startswith("data: ")
        )
        event = decode_sse_data(data) if data else None
        if event is not None:
            out.append(event)
    return out


def _add_user(su: Conn, tenant: UUID) -> UUID:
    user = uuid4()
    su.execute(
        "INSERT INTO rf.users (id, tenant_id, display_name) VALUES (%s, %s, 'colleague')",
        (user, tenant),
    )
    return user


def _live(su: Conn, doc: UUID) -> bool:
    row = su.execute("SELECT deleted_at IS NULL FROM rf.documents WHERE id = %s", (doc,)).fetchone()
    return bool(row and row[0])


async def test_chat_interrupt_then_http_approval_removes_the_page_once(
    api: tuple[httpx.AsyncClient, ScriptedChat], superuser: Conn, make_tenant: MakeTenant
) -> None:
    http, chat = api
    tenant, owner, coll = make_tenant()
    doc = seed_document(superuser, tenant, coll)
    owner_key = issue_key(superuser, tenant, owner, OWNER_SCOPES, name="ui:laptop")
    colleague_key = issue_key(
        superuser, tenant, _add_user(superuser, tenant), OWNER_SCOPES, name="ui:other"
    )
    remove = ToolCall(id=uuid4().hex, name="remove_from_index", arguments={"document_id": str(doc)})

    # 1. New chat: the API pre-creates the conversation and the turn stops at the approval.
    chat.script = [say("", remove)]
    first = await http.post(
        "/v1/chat",
        json={"message": "/remove the handbook", "collection_ids": [str(coll)]},
        headers=auth(owner_key),
    )
    assert first.status_code == 200, first.text
    events = _events(first.text)
    interrupt = events[-1]
    assert isinstance(interrupt, InterruptEvent), [e.type for e in events]
    assert [p.tool_call_id for p in interrupt.pending] == [remove.id]
    assert interrupt.conversation_id is not None
    stored = superuser.execute(
        "SELECT user_id FROM rf.conversations WHERE id = %s", (interrupt.conversation_id,)
    ).fetchone()
    assert stored == (owner,), "the API created the conversation for the caller"
    assert _live(superuser, doc), "nothing executes before approval"

    # 2. A colleague in the SAME tenant can neither continue the conversation nor approve.
    follow_up = await http.post(
        "/v1/chat",
        json={
            "conversation_id": str(interrupt.conversation_id),
            "message": "and now?",
            "collection_ids": [str(coll)],
        },
        headers=auth(colleague_key),
    )
    assert follow_up.status_code == 404
    hijack = await http.post(
        f"/v1/approvals/{interrupt.run_id}",
        json={"decisions": {remove.id: "approve"}},
        headers=auth(colleague_key),
    )
    assert hijack.status_code == 404
    assert _live(superuser, doc)
    assert chat.script == [], "neither rejected request reached the model"

    # 3. The owner approves over HTTP: the removal runs once and the turn completes.
    chat.script = [say("Removed the handbook [T1]."), SUPPORTED]
    second = await http.post(
        f"/v1/approvals/{interrupt.run_id}",
        json={"decisions": {remove.id: "approve"}},
        headers=auth(owner_key),
    )
    assert second.status_code == 200, second.text
    done = _events(second.text)[-1]
    assert isinstance(done, DoneEvent), [e.type for e in _events(second.text)]
    assert done.conversation_id == interrupt.conversation_id

    assert not _live(superuser, doc), "tombstoned"
    assert superuser.execute(
        "SELECT status, executed_at IS NOT NULL FROM rf.pending_approvals WHERE tool_call_id = %s",
        (remove.id,),
    ).fetchall() == [("approved", True)]
    assert superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND source_kind = 'file'",
        (tenant,),
    ).fetchone() == (1,)
    actions: dict[str, str] = {
        str(action): str(actor)
        for action, actor in superuser.execute(
            "SELECT action, actor FROM rf.audit_log WHERE tenant_id = %s "
            "AND action IN ('approvals.submit', 'document.removed')",
            (tenant,),
        ).fetchall()
    }
    assert actions["document.removed"] == f"user:{owner}"
    assert actions["approvals.submit"].startswith("key:"), "the deciding key is recorded"

    # 4. Re-submitting the same approval is a conflict, never a second removal.
    again = await http.post(
        f"/v1/approvals/{interrupt.run_id}",
        json={"decisions": {remove.id: "approve"}},
        headers=auth(owner_key),
    )
    assert again.status_code == 409
    assert superuser.execute(
        "SELECT count(*) FROM rf.audit_log WHERE tenant_id = %s AND action = 'document.removed'",
        (tenant,),
    ).fetchone() == (1,)
