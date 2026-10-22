"""Pure snapshot rules: job kinds, their payloads and dedupe keys, the eval key's scopes.

Payloads are pydantic models (validation is the framework's, not hand-rolled): a payload that
does not parse can never succeed, so the job handlers turn a ValidationError into a permanent
failure. Payloads are a pure function of their dedupe key (ADR-0010), so a merge into a queued
duplicate loses nothing.
"""

from __future__ import annotations

from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, PositiveInt

from ragforge_core.domain.models import JobPriority, JobSpec, Scope

KIND_SNAPSHOT_COPY: Final = "eval_snapshot_copy"  # migration 0012 begin_eval_snapshot checks it
KIND_SNAPSHOT_REEMBED: Final = "eval_snapshot_reembed"
EVAL_SNAPSHOT_KINDS: Final = frozenset({KIND_SNAPSHOT_COPY, KIND_SNAPSHOT_REEMBED})

# The copy is SQL only (seconds even for thousands of chunks); the re-embed calls the embedder
# for every chunk text, like a full re-sync of the collection.
MAX_RUNTIME_S: Final[dict[str, float]] = {
    KIND_SNAPSHOT_COPY: 10 * 60.0,
    KIND_SNAPSHOT_REEMBED: 2 * 3600.0,
}

# ADR-0017 §1: the dedicated eval key of the eval tenant holds exactly these scopes.
EVAL_KEY_SCOPES: Final = frozenset({Scope.eval.value, Scope.chat.value, Scope.read.value})
EVAL_KEY_NAME: Final = "eval"  # no "mcp" in it: an MCP key may never hold `eval`
EVAL_OWNER_DISPLAY_NAME: Final = "eval harness"


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CopyJobPayload(_Payload):
    """Runs in the SOURCE tenant: the copy derives its source from the job's tenant."""

    snapshot_id: UUID
    source_collection_id: UUID
    target_tenant_id: UUID


class ReembedJobPayload(_Payload):
    """Runs in the EVAL tenant (where the snapshot lives)."""

    collection_id: UUID
    space_id: PositiveInt


def copy_job(snapshot_id: UUID, source_collection_id: UUID, target_tenant_id: UUID) -> JobSpec:
    """Interactive: an owner is waiting, and the job is one short SQL transaction."""
    payload = CopyJobPayload(
        snapshot_id=snapshot_id,
        source_collection_id=source_collection_id,
        target_tenant_id=target_tenant_id,
    )
    return JobSpec(
        kind=KIND_SNAPSHOT_COPY,
        priority=JobPriority.interactive,
        payload=payload.model_dump(mode="json"),
        dedupe_key=f"{KIND_SNAPSHOT_COPY}:{snapshot_id}",
    )


def reembed_job(collection_id: UUID, space_id: int) -> JobSpec:
    """Maintenance: it embeds every chunk text and must not starve live syncs."""
    payload = ReembedJobPayload(collection_id=collection_id, space_id=space_id)
    return JobSpec(
        kind=KIND_SNAPSHOT_REEMBED,
        priority=JobPriority.maintenance,
        payload=payload.model_dump(mode="json"),
        dedupe_key=f"{KIND_SNAPSHOT_REEMBED}:{collection_id}:{space_id}",
    )
