"""The UI's only door to RAGForge: a small SYNCHRONOUS facade over the async SDK.

Why a facade: Streamlit runs the page script top to bottom in a plain thread, while the SDK is
async. `Backend` is the narrow, synchronous interface the page needs, so the page never touches
an event loop, and tests swap in a fake without HTTP.

Why a fresh client per call: an `httpx.AsyncClient` is bound to the event loop it first runs on,
and every Streamlit rerun is a new script run. One short-lived client per call (or per stream)
avoids sharing a client across loops; at <= 0.5 chat QPS the extra TCP connect is noise.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import SecretStr

from ragforge_client.client import RagforgeClient

# re-exported: the interview calls' timeout, shared with `make ask`
from ragforge_client.interview import INTERVIEW_TIMEOUT_S as INTERVIEW_TIMEOUT_S
from ragforge_contracts.api import (
    ApprovalDecision,
    Collection,
    Connector,
    Document,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    Job,
    Page,
)
from ragforge_contracts.sse import ChatEvent
from ragforge_ui.config import display_url
from ragforge_ui.errors import TurnTimeout

_PAGE_SIZE = 100
_MAX_PAGES = 10  # 1,000 collections or connectors is far beyond the design load (1-5 users)


class Backend(Protocol):
    """Everything the UI asks of the API. Streams are plain iterators that end at `done`."""

    @property
    def api_url(self) -> str:
        """Display-safe API URL (no credentials)."""
        ...

    def chat(
        self, message: str, collection_ids: Sequence[UUID], conversation_id: UUID | None
    ) -> Iterator[ChatEvent]: ...

    def resume(
        self, run_id: UUID, decisions: Mapping[str, ApprovalDecision]
    ) -> Iterator[ChatEvent]: ...

    def collections(self) -> list[Collection]: ...

    def connectors(self) -> list[Connector]: ...

    def documents(self) -> list[Document]:
        """Every document this key sees (bounded), to name the pages an approval would touch."""
        ...

    def sync_connector(self, connector_id: UUID) -> UUID:
        """Queue a sync; returns the job id."""
        ...

    def job(self, job_id: UUID) -> Job: ...

    def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        """One interview question (interview mode; the session lives in the page)."""
        ...

    def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade: ...


class SdkBackend:
    """`Backend` over `RagforgeClient`. Holds the key as a SecretStr; its repr never shows it."""

    def __init__(
        self,
        api_url: str,
        api_key: SecretStr,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = api_url
        self._key = api_key
        self._timeout = timeout
        self._transport = transport

    def __repr__(self) -> str:
        return f"SdkBackend(api_url={self.api_url!r})"

    @property
    def api_url(self) -> str:
        return display_url(self._url)

    def chat(
        self, message: str, collection_ids: Sequence[UUID], conversation_id: UUID | None
    ) -> Iterator[ChatEvent]:
        ids = list(collection_ids)
        return self._stream(lambda rf: rf.chat(message, ids, conversation_id=conversation_id))

    def resume(
        self, run_id: UUID, decisions: Mapping[str, ApprovalDecision]
    ) -> Iterator[ChatEvent]:
        chosen = dict(decisions)
        return self._stream(lambda rf: rf.resume(run_id, chosen))

    def collections(self) -> list[Collection]:
        return self._run(lambda rf: _all_pages(rf.list_collections))

    def connectors(self) -> list[Connector]:
        return self._run(lambda rf: _all_pages(rf.list_connectors))

    def documents(self) -> list[Document]:
        return self._run(lambda rf: _all_pages(rf.list_documents))

    def sync_connector(self, connector_id: UUID) -> UUID:
        async def call(rf: RagforgeClient) -> UUID:
            return (await rf.sync_connector(connector_id)).job_id

        return self._run(call)

    def job(self, job_id: UUID) -> Job:
        return self._run(lambda rf: rf.get_job(job_id))

    def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        return self._run(lambda rf: rf.interview_question(request), timeout=INTERVIEW_TIMEOUT_S)

    def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        return self._run(lambda rf: rf.interview_grade(request), timeout=INTERVIEW_TIMEOUT_S)

    # ------------------------------------------------------------------ plumbing
    def _client(self, timeout: float | None = None) -> RagforgeClient:
        return RagforgeClient(
            self._url,
            self._key.get_secret_value(),
            timeout=timeout if timeout is not None else self._timeout,
            transport=self._transport,
        )

    def _run[T](
        self, call: Callable[[RagforgeClient], Awaitable[T]], *, timeout: float | None = None
    ) -> T:
        async def go() -> T:
            async with self._client(timeout) as rf:
                return await call(rf)

        return asyncio.run(go())

    def _stream(
        self, open_stream: Callable[[RagforgeClient], AsyncIterator[ChatEvent]]
    ) -> Iterator[ChatEvent]:
        """Drive an async stream from synchronous code, one event at a time.

        A private event loop lives exactly as long as the iteration. When the consumer stops
        early (Streamlit reruns the script when the user clicks something), closing this
        generator closes the HTTP response, which the server sees as a disconnect and cancels
        the turn at the next node boundary (DESIGN section 8)."""
        loop = asyncio.new_event_loop()
        client = self._client()
        events = open_stream(client)
        deadline = time.monotonic() + TURN_DEADLINE_S
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TurnTimeout(TURN_DEADLINE_S)
                try:
                    event = loop.run_until_complete(asyncio.wait_for(anext(events), left))
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    # A server that stalls after its headers would otherwise keep every input
                    # disabled until the user presses Stop (integrated review P2; MCP has 180 s).
                    raise TurnTimeout(TURN_DEADLINE_S) from None
                yield event
        finally:
            _shut_down(loop, events, client)


TURN_DEADLINE_S = 180.0  # one whole turn, as the MCP server allows


def _shut_down(
    loop: asyncio.AbstractEventLoop, events: AsyncIterator[ChatEvent], client: RagforgeClient
) -> None:
    """Close the stream, then the client, then the loop: each step runs even if one before it
    fails (a failing close must not leave the HTTP connection open). Errors while tearing down
    are dropped on purpose: the turn's outcome is already decided, and cleanup can't change it."""
    closer = getattr(events, "aclose", None)
    if closer is not None:
        with contextlib.suppress(Exception):
            loop.run_until_complete(closer())
    # The SDK nests async generators; closing the outer one leaves the inner ones (which hold the
    # HTTP response) to the loop's finalizer. Run it now, while the loop is alive.
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    with contextlib.suppress(Exception):
        loop.run_until_complete(client.aclose())
    loop.close()


def drain[T](events: Iterator[T], handle: Callable[[T], None]) -> None:
    """Feed every event to `handle`, and close the stream however the loop ends.

    Why explicit: when Streamlit stops or reruns the script mid-answer it raises a BaseException
    out of the loop. The traceback keeps the loop's frame, and with it the generator, alive, so
    closing would otherwise wait for garbage collection. Closing here makes the HTTP disconnect
    (which tells the server to cancel the turn) happen right away."""
    try:
        for event in events:
            handle(event)
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()


async def _all_pages[T](
    fetch: Callable[..., Awaitable[Page[T]]],
) -> list[T]:
    """Follow `next_cursor` (bounded: a server bug must not loop the page forever)."""
    items: list[T] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        page = await fetch(cursor=cursor, limit=_PAGE_SIZE)
        items.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    return items
