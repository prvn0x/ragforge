"""Job progress wake-ups: one shared LISTEN connection per API process.

`NOTIFY rf_job_progress` carries ONLY a job id (docs/PHASE1.md), and any session may NOTIFY, so a
notification is never trusted as data: it only wakes the matching subscribers, which then re-read
the job under RLS. A notification for another tenant's job therefore reveals nothing (the re-read
returns no row for that tenant and the subscriber is not even registered for that id).

One connection for all SSE clients keeps the pool free; when it is down, subscribers still make
progress through their poll timeout (the fallback is the correctness path, LISTEN is latency).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol
from uuid import UUID

import psycopg

from ragforge_core.observability.logging import get_logger

CHANNEL = "rf_job_progress"
log = get_logger(__name__)


class JobFeed(Protocol):
    def subscribe(self, job_id: UUID) -> AbstractAsyncContextManager[asyncio.Event]:
        """An Event that is set whenever a notification for `job_id` arrives."""
        ...


class PollingJobFeed:
    """No LISTEN connection: subscribers rely on their poll interval alone."""

    @asynccontextmanager
    async def subscribe(self, job_id: UUID) -> AsyncIterator[asyncio.Event]:
        yield asyncio.Event()


class PgJobFeed:
    def __init__(
        self, dsn: str, *, retry_delay_s: float = 0.5, max_retry_delay_s: float = 30.0
    ) -> None:
        self._dsn = dsn
        self._retry_delay_s = retry_delay_s
        self._max_retry_delay_s = max_retry_delay_s
        self._subscribers: defaultdict[UUID, set[asyncio.Event]] = defaultdict(set)
        self._task: asyncio.Task[None] | None = None
        self.listening = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="job-feed")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @asynccontextmanager
    async def subscribe(self, job_id: UUID) -> AsyncIterator[asyncio.Event]:
        event = asyncio.Event()
        self._subscribers[job_id].add(event)
        try:
            yield event
        finally:
            waiters = self._subscribers.get(job_id)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    del self._subscribers[job_id]

    def dispatch(self, payload: str) -> None:
        try:
            job_id = UUID(payload.strip())
        except ValueError:
            return  # payloads are ids only; anything else is ignored, never interpreted
        for event in self._subscribers.get(job_id, ()):
            event.set()

    def _wake_all(self) -> None:
        """After (re)connecting, notifications may have been missed: everyone re-reads."""
        for waiters in self._subscribers.values():
            for event in waiters:
                event.set()

    async def _run(self) -> None:
        delay = self._retry_delay_s
        while True:
            try:
                await self._listen()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Only the exception type: driver messages can include connection details.
                log.warning("job_feed_disconnected", error=type(exc).__name__)
            if self.listening.is_set():  # it was healthy: start backing off from scratch
                delay = self._retry_delay_s
            self.listening.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_retry_delay_s)

    async def _listen(self) -> None:
        async with await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as conn:
            await conn.execute("LISTEN rf_job_progress")  # == CHANNEL
            self.listening.set()
            self._wake_all()
            async for notify in conn.notifies():
                self.dispatch(notify.payload)
