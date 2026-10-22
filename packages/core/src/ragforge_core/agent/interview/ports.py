"""The one persistence port interview mode needs (PHASE1: "a module that needs its own tables
accessed defines a small Protocol").

Signatures use only domain types and primitives, so the SQL implementation (in the API layer,
`apps/api/routes/interview.py`) satisfies it structurally without `agent` importing SQL.

Every method runs inside the caller's tenant `tx` and applies the SAME visibility filter, the
one retrieval applies: RLS scopes rows to the tenant; the chunk's collection is one of
`collection_ids` (already checked visible by the route); that collection's active embedding
space is `active`; the page is not removed (`documents.deleted_at IS NULL`). So an excluded,
deleted or foreign page is never asked about or graded against. Returned chunks carry
`fused_score = 0` and no rerank score: they are notes, not search results.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Tx


@runtime_checkable
class InterviewNotes(Protocol):
    async def require_collections(self, tx: Tx, collection_ids: Sequence[UUID]) -> None:
        """Raise `NotFound` unless every id is a collection this tenant can see (the same check
        and error as every other route, so a foreign collection is indistinguishable from a
        missing one)."""
        ...

    async def topic_chunks(
        self,
        tx: Tx,
        collection_ids: Sequence[UUID],
        topic: str,
        exclude: Collection[UUID],
        limit: int,
    ) -> Sequence[RankedChunk]:
        """Chunks of the pages whose TITLE or HEADING PATH matches `topic`, minus `exclude`, up
        to `limit`, in random order. Matching is full-text search with each collection's own
        text-search configuration (the one its keyword index uses), the topic passed as a bound
        parameter to `plainto_tsquery`: stemming applies ("transactions" finds "Transaction")
        and no user text is ever read as a pattern or as query syntax."""
        ...

    async def mentioning(
        self, tx: Tx, collection_ids: Sequence[UUID], chunk_ids: Collection[UUID], topic: str
    ) -> frozenset[UUID]:
        """Those of `chunk_ids` whose text, title or heading path contains EVERY word of `topic`
        (full-text search, the collection's own configuration, `plainto_tsquery`: stemmed, a
        bound parameter). The lexical gate on search-only candidates: semantic search alone let
        a Kafka note through for "Kubernetes operators" (live, 2026-10-19), so the "no notes"
        offer never appeared."""
        ...

    async def topic_phrases(
        self, tx: Tx, collection_ids: Sequence[UUID], limit: int
    ) -> Sequence[str]:
        """The DISTINCT page titles and heading-path segments of the visible chunks, at most
        `limit` (the acronym matcher's input, `acronyms.py`). Plain text out, nothing in: the
        matching happens in Python, so no user text is ever a SQL pattern. Never filtered by a
        session's exclusions (an acronym's meaning does not depend on what was asked)."""
        ...

    async def sample_chunks(
        self, tx: Tx, collection_ids: Sequence[UUID], exclude: Collection[UUID], limit: int
    ) -> Sequence[RankedChunk]:
        """At most ONE random chunk per page (a substantive one when the page has any), minus
        `exclude`, from up to `limit` pages. Pages none of whose chunks is in `exclude` come
        first (broad coverage across a session), then the others, each group shuffled."""
        ...

    async def documents_of(self, tx: Tx, chunk_ids: Collection[UUID]) -> frozenset[UUID]:
        """The pages the given (visible) chunks belong to: the pages a session already covered."""
        ...

    async def chunks_by_id(
        self, tx: Tx, chunk_ids: Sequence[UUID], collection_ids: Sequence[UUID]
    ) -> Sequence[RankedChunk]:
        """The visible chunks among `chunk_ids`, in ONE query. A chunk of another tenant, of
        another collection or of a removed page is simply absent (the caller answers 404)."""
        ...

    async def neighbours(
        self, tx: Tx, chunk_id: UUID, collection_ids: Sequence[UUID], radius: int
    ) -> Sequence[RankedChunk]:
        """The chunk and the visible chunks within `radius` positions of it on the same page,
        in page order."""
        ...
