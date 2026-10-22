"""Module-local ports for the worker runtime (PHASE1: module-local Protocols live in
services/<module>/ports.py and use only domain types and primitives)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragforge_core.domain.models import JobLease
from ragforge_core.domain.ports import JobQueue


@runtime_checkable
class FairJobQueue(JobQueue, Protocol):
    """The frozen `JobQueue` plus the starvation-guard flag on claim (ADR-0010). Adding a
    keyword-only parameter with a default keeps every `FairJobQueue` a valid `JobQueue`."""

    async def claim(
        self, worker_id: str, n: int, *, prefer_low: bool = False
    ) -> Sequence[JobLease]: ...

    async def defer(self, lease: JobLease, delay_s: int) -> None:
        """Give the job back WITHOUT charging an attempt, runnable again after `delay_s`
        (`rf.defer_job`). For kinds this worker cannot run during a rolling deploy."""
        ...


@runtime_checkable
class QueueMaintenance(Protocol):
    """Periodic upkeep every worker runs (all of it is safe to run concurrently)."""

    async def reclaim_expired(self, limit: int = 500) -> int: ...

    async def move_finished(self) -> int: ...
