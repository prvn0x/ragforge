"""`SdkBackend`: the sync facade over the async SDK, against a fake HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator, Iterator
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from ragforge_client.client import ApiError, RagforgeClient
from ragforge_contracts.api import ApprovalDecision
from ragforge_contracts.sse import ChatEvent, DoneEvent, TokenEvent, encode_sse
from ragforge_ui.api import SdkBackend, drain

KEY = "rf_live_secret"
NOW = datetime(2026, 10, 18, tzinfo=UTC).isoformat()


def _backend(handler: httpx.MockTransport) -> SdkBackend:
    return SdkBackend("http://api.test", SecretStr(KEY), transport=handler)


def test_repr_and_url_never_show_secrets() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    backend = SdkBackend("http://user:pw@api.test", SecretStr(KEY), transport=transport)
    assert KEY not in repr(backend)
    assert "pw" not in repr(backend)
    assert backend.api_url == "http://api.test"


def test_chat_streams_typed_events_with_the_bearer_key() -> None:
    conversation_id = uuid4()
    seen: list[httpx.Request] = []
    body = encode_sse(TokenEvent(text="hi")) + encode_sse(
        DoneEvent(conversation_id=conversation_id, usage=[], prompt_version="answer@11")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    collection = uuid4()
    events = list(_backend(httpx.MockTransport(handler)).chat("q", [collection], None))
    assert [e.type for e in events] == ["token", "done"]
    request = seen[0]
    assert request.url.path == "/v1/chat"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "conversation_id": None,
        "collection_ids": [str(collection)],
        "message": "q",
    }


def test_resume_posts_the_decisions_to_the_run() -> None:
    run_id = uuid4()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"")

    decisions = {"t1": ApprovalDecision.approve, "t2": ApprovalDecision.reject}
    assert list(_backend(httpx.MockTransport(handler)).resume(run_id, decisions)) == []
    assert seen[0].url.path == f"/v1/approvals/{run_id}"
    assert json.loads(seen[0].content) == {"decisions": {"t1": "approve", "t2": "reject"}}


def test_401_raises_api_error_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"title": "Unauthorized", "status": 401})

    backend = _backend(httpx.MockTransport(handler))
    with pytest.raises(ApiError) as info:
        list(backend.chat("q", [uuid4()], None))
    assert info.value.status == 401
    assert KEY not in str(info.value)
    with pytest.raises(ApiError):
        backend.collections()


def test_collections_follow_cursors() -> None:
    def item(name: str) -> dict[str, object]:
        return {"id": str(uuid4()), "name": name, "active_space_id": 1, "created_at": NOW}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "p2":
            return httpx.Response(200, json={"items": [item("b")]})
        assert request.url.params["limit"] == "100"
        return httpx.Response(200, json={"items": [item("a")], "next_cursor": "p2"})

    names = [c.name for c in _backend(httpx.MockTransport(handler)).collections()]
    assert names == ["a", "b"]


def test_a_cursor_loop_is_bounded() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"items": [], "next_cursor": "again"})

    assert _backend(httpx.MockTransport(handler)).connectors() == []
    assert len(calls) == 10


def test_sync_returns_the_job_and_job_reads_it() -> None:
    connector_id, job_id = uuid4(), uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == f"/v1/connectors/{connector_id}/sync"
            return httpx.Response(202, json={"job_id": str(job_id)})
        job = {
            "id": str(job_id),
            "kind": "sync_connector",
            "status": "done",
            "attempts": 1,
            "last_error": None,
            "enqueued_at": NOW,
        }
        return httpx.Response(200, json=job)

    backend = _backend(httpx.MockTransport(handler))
    assert backend.sync_connector(connector_id) == job_id
    assert backend.job(job_id).status == "done"


def test_stopping_early_closes_the_response() -> None:
    """Streamlit abandons the iterator on a rerun: the HTTP stream must be closed, so the
    server sees the disconnect and cancels the turn."""
    closed: list[bool] = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield encode_sse(TokenEvent(text="a"))
            yield encode_sse(TokenEvent(text="b"))

        async def aclose(self) -> None:
            closed.append(True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Body())

    events = _backend(httpx.MockTransport(handler)).chat("q", [uuid4()], None)
    assert next(events).type == "token"
    events.close()  # type: ignore[attr-defined]
    assert closed == [True]


class _RecordingBody(httpx.AsyncByteStream):
    """An SSE body that never ends on its own and records when it is closed."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            yield encode_sse(TokenEvent(text="a"))

    async def aclose(self) -> None:
        self.closed = True


def test_a_streamlit_stop_mid_answer_closes_the_response_at_once() -> None:
    """The production path: Streamlit raises StopException (a BaseException) out of the loop
    that consumes the stream. The body must close BEFORE the exception is handled: while it
    propagates, its traceback keeps the loop's frame (and the generator) alive, so nothing may
    rely on garbage collection to disconnect."""
    body = _RecordingBody()
    backend = _backend(httpx.MockTransport(lambda r: httpx.Response(200, stream=body)))
    seen: list[ChatEvent] = []

    def handle(event: ChatEvent) -> None:
        seen.append(event)
        if len(seen) == 2:
            raise StopException()

    with pytest.raises(StopException) as info:
        drain(backend.chat("q", [uuid4()], None), handle)
    assert info.tb is not None  # the traceback (and with it the loop's frame) is still alive
    assert body.closed
    assert len(seen) == 2


def test_drain_closes_on_normal_end_and_on_errors() -> None:
    closed: list[str] = []

    def events(tag: str, fail: bool) -> Iterator[int]:
        try:
            yield 1
            if fail:
                raise ValueError("boom")
            yield 2
        finally:
            closed.append(tag)

    got: list[int] = []
    drain(events("ok", fail=False), got.append)
    with pytest.raises(ValueError, match="boom"):
        drain(events("bad", fail=True), got.append)
    assert closed == ["ok", "bad"]
    assert got == [1, 2, 1]


def test_a_failing_stream_close_still_closes_the_client() -> None:
    """Each teardown step runs even if one before it fails: no leaked connection."""
    transport_closed: list[bool] = []

    class Transport(httpx.MockTransport):
        async def aclose(self) -> None:
            transport_closed.append(True)

    class BadClose:
        def __aiter__(self) -> BadClose:
            return self

        async def __anext__(self) -> ChatEvent:
            return TokenEvent(text="x")

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    backend = SdkBackend(
        "http://api.test", SecretStr(KEY), transport=Transport(lambda r: httpx.Response(500))
    )

    def open_stream(rf: RagforgeClient) -> AsyncIterator[ChatEvent]:
        return BadClose()

    events = backend._stream(open_stream)
    assert isinstance(events, Generator)
    assert next(events).type == "token"
    events.close()  # must not raise, and must reach client.aclose()
    assert transport_closed == [True]


def test_documents_walk_every_page() -> None:
    def doc(title: str) -> dict[str, object]:
        return {
            "id": str(uuid4()),
            "collection_id": str(uuid4()),
            "title": title,
            "external_url": None,
            "external_updated_at": None,
            "fetched_at": None,
        }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents"
        if request.url.params.get("cursor") == "p2":
            return httpx.Response(200, json={"items": [doc("b")]})
        return httpx.Response(200, json={"items": [doc("a")], "next_cursor": "p2"})

    titles = [d.title for d in _backend(httpx.MockTransport(handler)).documents()]
    assert titles == ["a", "b"]


class _Stalls(httpx.AsyncByteStream):
    """Sends one SSE event, then never sends anything again (a stalled server)."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield encode_sse(TokenEvent(text="Kafka"))
        import asyncio

        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


def test_a_stalled_stream_hits_the_turn_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integrated review P2: a server that stalls after its headers must not keep the page's
    inputs disabled forever; the whole turn has a deadline (as MCP does)."""
    from ragforge_ui import api
    from ragforge_ui.errors import TurnTimeout

    monkeypatch.setattr(api, "TURN_DEADLINE_S", 0.2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Stalls(), headers={"content-type": "text/event-stream"})

    backend = _backend(httpx.MockTransport(handler))
    events = backend.chat("q", [uuid4()], None)
    assert isinstance(next(events), TokenEvent)
    with pytest.raises(TurnTimeout):
        next(events)
