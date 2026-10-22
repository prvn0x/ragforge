"""Job handler registry: `kind -> async handler(lease)`.

Handler contract (ADR-0010, PHASE1 rule 8):
- Do slow work (fetch, embed, chat) OUTSIDE any transaction, then open ONE short tenant
  transaction that starts with `JobQueue.assert_lease(tx, lease)`, applies the side effects and
  ends with `JobQueue.complete(tx, lease)`. If the handler returns without completing, the
  runtime completes the job in a fenced transaction of its own.
- Raise `PermanentJobError` for failures a retry cannot fix (e.g. a definitive 404) and
  `domain.errors.UpstreamRateLimited` to honour an upstream Retry-After. Any other exception is
  retried with backoff until max_attempts.
- Expect `asyncio.CancelledError` at any await: the runtime cancels the handler when its lease is
  lost or at shutdown. Never swallow it.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from ragforge_core.domain.models import JobLease

JobHandler = Callable[[JobLease], Coroutine[Any, Any, None]]  # an `async def` handler


DEFAULT_MAX_RUNTIME_S = 30 * 60.0
"""A handler running longer than its kind's max runtime is cancelled and failed (retryable).
The keepalive renews a lease for as long as the database answers, so without a wall-clock
bound a handler stuck in a loop or an endless wait would hold its slot (and, for a listing,
its connector: the scheduler never claims a connector with a running listing) forever."""


class PermanentJobError(Exception):
    """The job can never succeed: it goes straight to `dead` (with an audit row).

    The ONE definition (services.sync.deps re-exports it). The message is written to
    jobs.last_error and shown to users, so it must never contain secrets or document content.
    """


class DuplicateHandler(ValueError):
    """A kind was registered twice; silently replacing a handler hides wiring bugs."""


class JobHandlers:
    """Mutable during startup wiring, read-only afterwards (the runtime only calls `get`)."""

    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler] = {}
        self._self_reporting: set[str] = set()
        self._max_runtime: dict[str, float] = {}

    def register(
        self,
        kind: str,
        handler: JobHandler,
        *,
        reports_outcome: bool = False,
        max_runtime_s: float = DEFAULT_MAX_RUNTIME_S,
    ) -> None:
        """`reports_outcome=True`: the handler itself always ends the job (complete/fail in
        its own fenced transactions) on every path where it returns or raises a non-lease error.
        The runtime then neither completes it again (a second fenced completion would only
        raise lease-lost in Postgres) nor turns an error from that reporting into a transient
        failure; LeaseLost and cancellation are handled as for any job."""
        if not kind:
            raise ValueError("job kind must be non-empty")
        if kind in self._handlers:
            raise DuplicateHandler(f"handler already registered for job kind {kind!r}")
        if not max_runtime_s > 0:
            raise ValueError("max_runtime_s must be positive")
        self._handlers[kind] = handler
        self._max_runtime[kind] = max_runtime_s
        if reports_outcome:
            self._self_reporting.add(kind)

    def max_runtime_s(self, kind: str) -> float:
        return self._max_runtime.get(kind, DEFAULT_MAX_RUNTIME_S)

    def reports_outcome(self, kind: str) -> bool:
        return kind in self._self_reporting

    def get(self, kind: str) -> JobHandler | None:
        return self._handlers.get(kind)

    def kinds(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def __len__(self) -> int:
        return len(self._handlers)
