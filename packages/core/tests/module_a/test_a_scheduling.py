"""Strict class order + every-4th-claim starvation guard (ADR-0010)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from module_a.a_support import enqueue, spec
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.models import JobPriority
from ragforge_core.services.jobs import ClaimPlan, StarvationGuard


# ------------------------------------------------------------------ pure guard
def test_guard_gives_every_fourth_job_to_the_low_classes_one_at_a_time() -> None:
    guard = StarvationGuard()
    plans = []
    for _ in range(8):
        plan = guard.plan(free_slots=1)
        plans.append(plan.prefer_low)
        guard.record(plan, 1)
    assert plans == [False, False, False, True] * 2


def test_guard_caps_batches_at_the_boundary() -> None:
    guard = StarvationGuard()
    first = guard.plan(free_slots=16)
    assert first == ClaimPlan(n=3, prefer_low=False)  # stop right before the 4th job
    guard.record(first, 3)
    second = guard.plan(free_slots=16)
    assert second == ClaimPlan(n=4, prefer_low=True)  # 4th job is the low turn
    guard.record(second, 4)  # 1 low + 3 normal -> next batch is a low turn again
    assert guard.plan(free_slots=16).prefer_low is True


def test_guard_keeps_an_unserved_low_turn_pending() -> None:
    guard = StarvationGuard()
    for _ in range(3):
        guard.record(guard.plan(1), 1)
    turn = guard.plan(1)
    assert turn.prefer_low
    guard.record(turn, 0)  # nothing claimable at all
    assert guard.plan(1).prefer_low


def test_guard_partial_batches_count_only_what_was_claimed() -> None:
    guard = StarvationGuard()
    plan = guard.plan(8)
    guard.record(plan, 1)  # asked for 3, got 1
    assert guard.plan(8) == ClaimPlan(n=2, prefer_low=False)


# ------------------------------------------------------------------ SQL class order
@pytest.mark.integration
async def test_claim_walks_classes_in_strict_order_fifo_within_class(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID
) -> None:
    # Enqueue the lowest class FIRST: no cross-class aging means it still runs last.
    order = [
        ("b1", JobPriority.backfill),
        ("m1", JobPriority.maintenance),
        ("i1", JobPriority.incremental),
        ("w1", JobPriority.webhook),
        ("x1", JobPriority.interactive),
        ("x2", JobPriority.interactive),
        ("i2", JobPriority.incremental),
    ]
    for key, priority in order:
        await enqueue(txm, queue, tenant, spec(key, priority))
    leases = await queue.claim("w", 10)
    assert [lease.payload["key"] for lease in leases] == ["x1", "x2", "w1", "i1", "i2", "m1", "b1"]


@pytest.mark.integration
async def test_prefer_low_takes_the_oldest_lower_class_head_first(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID
) -> None:
    await enqueue(txm, queue, tenant, spec("backfill-old", JobPriority.backfill))
    await enqueue(txm, queue, tenant, spec("maint-newer", JobPriority.maintenance))
    await enqueue(txm, queue, tenant, spec("interactive", JobPriority.interactive))
    await enqueue(txm, queue, tenant, spec("incremental", JobPriority.incremental))

    (first,) = await queue.claim("w", 1, prefer_low=True)
    assert first.payload["key"] == "backfill-old"  # older head wins across the two low classes
    batch = await queue.claim("w", 2, prefer_low=True)
    # Only the FIRST job of a prefer_low batch is the low turn; the rest follow class order.
    assert [lease.payload["key"] for lease in batch] == ["maint-newer", "interactive"]
    (last,) = await queue.claim("w", 1, prefer_low=True)  # no low jobs left: normal order
    assert last.payload["key"] == "incremental"


@pytest.mark.integration
async def test_future_run_after_is_not_claimable(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID
) -> None:
    later = replace(spec("later"), run_after=datetime.now(UTC) + timedelta(minutes=5))
    await enqueue(txm, queue, tenant, later)
    assert await queue.claim("w", 5) == []
    assert await queue.claim("w", 5, prefer_low=True) == []


@pytest.mark.integration
async def test_backfill_gets_a_quarter_of_claims_and_never_delays_incremental_more(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID
) -> None:
    """Freshness under a big backfill: with both queued, a worker driven by the guard serves
    exactly 1 backfill per 4 claims, and incremental work is never queued behind backfill."""
    for i in range(40):
        await enqueue(txm, queue, tenant, spec(f"bf-{i}", JobPriority.backfill))
    for i in range(12):
        await enqueue(txm, queue, tenant, spec(f"inc-{i}", JobPriority.incremental))

    guard = StarvationGuard()
    served: list[str] = []
    while len(served) < 16:
        plan = guard.plan(free_slots=1)
        leases = await queue.claim("w", plan.n, prefer_low=plan.prefer_low)
        guard.record(plan, len(leases))
        served.extend(str(lease.payload["key"]) for lease in leases)
    kinds = ["bf" if key.startswith("bf") else "inc" for key in served]
    assert kinds == ["inc", "inc", "inc", "bf"] * 4
