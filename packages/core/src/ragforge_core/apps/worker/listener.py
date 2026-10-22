"""Dedicated LISTEN connection that wakes the claim loop on `NOTIFY rf_jobs` (ADR-0010).

Not pooled on purpose: a pooled connection would be handed to other work (and LISTEN state is
per session), and holding a pool slot forever would shrink the pool. Notifications are only a
latency optimization: the runtime also polls every 2 s, so a lost connection degrades latency,
never correctness. After every (re)connect we wake once, because anything NOTIFYed while we were
disconnected was lost.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

import psycopg

from ragforge_core.db.engine import CONNECT_ARGS
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

JOBS_CHANNEL = "rf_jobs"


class NotificationListener:
    def __init__(
        self,
        dsn: str,
        on_notify: Callable[[], None],
        *,
        channel: str = JOBS_CHANNEL,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._dsn = dsn  # contains the password: never log it
        self._on_notify = on_notify
        self._channel = channel
        self._max_backoff_s = max_backoff_s
        self.connected = asyncio.Event()  # observable in tests and for readiness logs

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                await self._listen(stop)
                backoff = 1.0
            except (psycopg.Error, OSError) as exc:
                self.connected.clear()
                log.warning(
                    "job_listener_disconnected", error=type(exc).__name__, retry_in_s=backoff
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

    async def _listen(self, stop: asyncio.Event) -> None:
        # Keepalives: a half-open LISTEN connection would otherwise wait for notifications that
        # never come, and the worker would silently fall back to 2 s polling forever.
        conninfo = psycopg.conninfo.make_conninfo(self._dsn, **CONNECT_ARGS)
        async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
            await conn.execute(
                psycopg.sql.SQL("LISTEN {}").format(psycopg.sql.Identifier(self._channel))
            )
            self.connected.set()
            self._on_notify()
            while not stop.is_set():
                # Bounded wait so `stop` is honoured promptly; the generator ends on timeout.
                async for _ in conn.notifies(timeout=1.0):
                    self._on_notify()
        self.connected.clear()
