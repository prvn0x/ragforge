"""What the snapshot job handlers need from storage (implemented on Postgres in apps.worker).

Every method runs in the caller's tenant transaction. Refusals arrive as domain errors:
`Forbidden` (the r4 guard: not this tenant's eval tenant, not a personal/team source),
`NotFound` (a collection or snapshot the caller cannot see), `ValidationFailed` (bad input),
and `SnapshotNotReady` (retryable: the source is still indexing, or the snapshot is still being
copied). `errors.refusal_for` maps the migration's SQLSTATEs onto them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ragforge_core.domain.ports import Tx


@dataclass(frozen=True, slots=True)
class CopyOutcome:
    pages: int
    chunks: int
    vectors: int
    links: int
    excluded: int  # live source pages skipped because they are excluded
    content_digest: str
    index_digest: str


@dataclass(frozen=True, slots=True)
class EmbedSource:
    """One distinct embed hash of a snapshot and the stored fields its input is rebuilt from."""

    embed_sha256: bytes
    title: str
    heading_path: tuple[str, ...]
    display_text: str


@dataclass(frozen=True, slots=True)
class ReembedOutcome:
    content_digest: str
    index_digest: str
    vectors: int


class SnapshotStore(Protocol):
    async def copy(
        self, tx: Tx, *, source_collection_id: UUID, target_tenant_id: UUID, snapshot_id: UUID
    ) -> CopyOutcome:
        """rf.copy_collection_to_eval in the SOURCE tenant's transaction (idempotent)."""
        ...

    async def embed_sources(self, tx: Tx, collection_id: UUID) -> Sequence[EmbedSource]:
        """The distinct embed hashes of a READY snapshot of the transaction's tenant."""
        ...

    async def finish_reembed(self, tx: Tx, collection_id: UUID, space_id: int) -> ReembedOutcome:
        """rf.finish_eval_reembed: verify full coverage, flip the space, record the digest."""
        ...
