"""The snapshot's refusals as domain errors, from the SQLSTATEs migration 0012 raises.

One table for both callers of the definer functions (the API route and the worker's store), so
a refusal means the same thing, with the same HTTP status, wherever it surfaces. The messages
are the migration's own RAISE texts: no bound parameters and no note text, so they are safe to
show as a problem detail or a job's last_error.
"""

from __future__ import annotations

from typing import Final

from ragforge_core.domain.errors import (
    Conflict,
    Forbidden,
    NotFound,
    RagforgeError,
    ValidationFailed,
)


class SnapshotNotReady(Conflict):
    """Retryable: the data the step needs is not there yet (the source is still indexing, or
    the snapshot is still being copied)."""

    title = "Snapshot not ready"


class EvalTenantExists(Conflict):
    title = "Eval tenant already exists"


_REFUSALS: Final[dict[str, type[RagforgeError]]] = {
    "42501": Forbidden,  # the r4 guard: not this tenant's eval tenant / not a personal source
    "P0002": NotFound,  # a collection or snapshot the caller cannot see
    "22023": ValidationFailed,
    "55000": SnapshotNotReady,
}

# A unique violation is ours only on the pairing column (one eval tenant per owner); any other
# 23505 (an id collision in the copy, say) is a bug and must surface as one, not as a 409.
EVAL_PAIRING_CONSTRAINT: Final = "tenants_eval_of_tenant_id_key"
UNIQUE_VIOLATION: Final = "23505"


def refusal_for(
    sqlstate: str | None, message: str | None, *, constraint: str | None = None
) -> RagforgeError | None:
    """The domain error for a refusal, or None when the error is not one of ours (the caller
    re-raises the original error then). `constraint` is the driver's diag.constraint_name."""
    if sqlstate == UNIQUE_VIOLATION:
        if constraint == EVAL_PAIRING_CONSTRAINT:
            return EvalTenantExists("this tenant already has an eval tenant")
        return None
    error = _REFUSALS.get(sqlstate or "")
    if error is None:
        return None
    return error(message or error.title)
