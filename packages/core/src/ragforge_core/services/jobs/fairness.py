"""Starvation guard bookkeeping (ADR-0010), kept pure so the 25 % guarantee is unit-testable.

Every 4th claimed job must come from the lower classes (maintenance/backfill) when any is queued.
The worker claims in batches, so the guard also bounds the batch size: a batch never spans more
than one "4th job" boundary, and the batch that reaches the boundary is the one sent with
`prefer_low=True` (the SQL takes that batch's FIRST job from the oldest-head lower class).
"""

from __future__ import annotations

from dataclasses import dataclass

LOW_CLASS_EVERY = 4


@dataclass(frozen=True, slots=True)
class ClaimPlan:
    n: int
    prefer_low: bool


class StarvationGuard:
    def __init__(self, every: int = LOW_CLASS_EVERY) -> None:
        if every < 2:
            raise ValueError("every must be >= 2")
        self._every = every
        self._since_low = 0  # jobs claimed since the last prefer_low batch that returned jobs

    def plan(self, free_slots: int) -> ClaimPlan:
        """How many jobs to ask for, and whether this batch carries the low-class turn."""
        if free_slots < 1:
            raise ValueError("free_slots must be >= 1")
        normal_left = self._every - 1 - self._since_low
        if normal_left <= 0:
            return ClaimPlan(n=min(free_slots, self._every), prefer_low=True)
        return ClaimPlan(n=min(free_slots, normal_left), prefer_low=False)

    def record(self, plan: ClaimPlan, claimed: int) -> None:
        """An empty prefer_low batch leaves the turn pending, so the next claim retries it."""
        if claimed <= 0:
            return
        if plan.prefer_low:
            self._since_low = claimed - 1
        else:
            self._since_low += claimed
