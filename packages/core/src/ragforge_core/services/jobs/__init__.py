"""Job runtime (module A): handler registry, starvation guard and the worker loop.

Other modules register handlers with `JobHandlers.register(kind, handler)`; Phase 2 wires them
in the worker entrypoint. See `registry` for the handler contract.
"""

from ragforge_core.services.jobs.fairness import ClaimPlan, StarvationGuard
from ragforge_core.services.jobs.ports import FairJobQueue, QueueMaintenance
from ragforge_core.services.jobs.registry import (
    DEFAULT_MAX_RUNTIME_S,
    DuplicateHandler,
    JobHandler,
    JobHandlers,
    PermanentJobError,
)
from ragforge_core.services.jobs.runtime import WorkerConfig, WorkerRuntime, describe_error

__all__ = [
    "DEFAULT_MAX_RUNTIME_S",
    "ClaimPlan",
    "DuplicateHandler",
    "FairJobQueue",
    "JobHandler",
    "JobHandlers",
    "PermanentJobError",
    "QueueMaintenance",
    "StarvationGuard",
    "WorkerConfig",
    "WorkerRuntime",
    "describe_error",
]
