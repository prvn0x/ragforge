"""/v1/eval: verify, chunks, resolve (ADR-0017 §1). Eval scope only, behind RAGFORGE_EVAL_API.

Every route depends on `deps.EvalKey` (or `EvalQuotaKey`): the switch first (404 while off, before
authentication), then the key, then the `eval` scope (403). Every read runs in the caller's
tenant transaction, so RLS confines it to the eval tenant: an id or key of another tenant is
simply invisible (verify: 404, chunks: absent, resolve: missing), never "forbidden", so ids
cannot be probed.

Stable identity (ADR-0017 principle 6): datasets store keys, never ids.
- page key: `(file, metadata.filename)` for uploads (their external_id is a random upload id per
  install) and `(notion, external_id)` for Notion pages, including snapshot copies, which keep
  the original page id. A file document without a filename falls back to its external_id.
- chunk key: (page key, heading_path, display_sha256, occurrence), occurrence = rank among the
  page's chunks with an equal (heading_path, display_sha256), in ordinal order.
`resolve` never re-maps: a key whose chunk is gone is `stale`, with the chunks under the same
(page, heading_path) as candidates for the owner to re-approve. A page key that names more than
one live document (a file uploaded twice) is `ambiguous` for the page and its chunks: picking one
would move a dataset item to another document with no signal (review round 1, P2).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import (
    ChunkKey,
    ClaimReason,
    EvalChunk,
    EvalChunksResponse,
    EvalCoverageUnit,
    EvalFlaggedClaim,
    EvalResolveRequest,
    EvalResolveResponse,
    EvalVerifierClaim,
    EvalVerifyRequest,
    EvalVerifyResponse,
    PageKey,
    ResolvedChunk,
    ResolvedPage,
    ResolveStatus,
    VerificationStatus,
)
from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.eval_verify import EvalVerdict, verify_answer
from ragforge_core.apps.api.deps import ChatModels, EvalKey, EvalQuotaKey, TxManager
from ragforge_core.apps.api.routes.sources import require_collections
from ragforge_core.apps.api.sql import fetch_all
from ragforge_core.domain.errors import NotFound, ValidationFailed
from ragforge_core.domain.models import RankedChunk
from ragforge_core.observability.logging import get_logger

router = APIRouter(prefix="/v1/eval", tags=["eval"])
log = get_logger(__name__)

MAX_CHUNK_IDS: Final = 200  # one drill-down page or judge batch; far above a run's shown set
# An additive value of the OPEN ResolveStatus (docs/handoff/EV-API.md): the client counts only
# `ok`, so an ambiguous key marks its item stale, exactly like a stale or missing one.
AMBIGUOUS: Final = "ambiguous"

# ----------------------------------------------------------------- SQL (RLS: the caller's tenant)
# Composed only from these constants; every user-supplied value is a bind parameter. Live pages
# only (tombstoned ones are never shown, searched or verified against).
_PAGE_KEY: Final = (
    "CASE WHEN d.source_kind = 'file' THEN COALESCE(d.metadata ->> 'filename', d.external_id) "
    "ELSE d.external_id END"
)
_LIVE: Final = (
    "FROM rf.chunks c JOIN rf.documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id "
    "AND d.deleted_at IS NULL"
)
_SHOWN_SQL: Final = f"""
    SELECT c.id, c.document_id, c.collection_id, c.anchor, d.title, c.heading_path,
           c.display_text, c.display_sha256, d.external_url
      {_LIVE}
     WHERE c.id = ANY (CAST(:ids AS uuid[])) AND c.collection_id = ANY (CAST(:colls AS uuid[]))
"""
_CHUNKS_SQL: Final = f"""
    SELECT c.id, c.document_id, d.source_kind, {_PAGE_KEY} AS page_key, c.heading_path,
           c.ordinal, c.display_text, c.display_sha256
      {_LIVE}
     WHERE c.id = ANY (CAST(:ids AS uuid[]))
"""
_PAGES_SQL: Final = f"""
    SELECT d.id, d.source_kind, {_PAGE_KEY} AS page_key
      FROM rf.documents d
     WHERE d.collection_id = ANY (CAST(:colls AS uuid[])) AND d.deleted_at IS NULL
       AND {_PAGE_KEY} = ANY (CAST(:keys AS text[]))
     ORDER BY d.created_at DESC, d.id
"""  # noqa: S608  (newest first: the order ambiguous candidates are listed in)
_PAGE_CHUNKS_SQL: Final = """
    SELECT c.id, c.document_id, c.heading_path, c.display_sha256
      FROM rf.chunks c
     WHERE c.document_id = ANY (CAST(:docs AS uuid[]))
     ORDER BY c.document_id, c.ordinal
"""


# ----------------------------------------------------------------- verify
@dataclass(frozen=True, slots=True)
class VerifyConfig:
    """The chat path's verification settings that shape the verdict E-V measures: the citation
    mode. NOT `agent_verifier_trusted`: E-V is what decides that flag, so the model under test
    is never downgraded by it (agent/eval_verify.py docstring)."""

    strict: bool = False


def get_verify_config(request: Request) -> VerifyConfig:
    settings = getattr(request.app.state.container, "settings", None)
    if settings is None:
        return VerifyConfig()
    return VerifyConfig(strict=bool(settings.agent_strict_citations))


VerifyConfigs = Annotated[VerifyConfig, Depends(get_verify_config)]


@router.post("/verify")
async def eval_verify(
    body: EvalVerifyRequest,
    key: EvalQuotaKey,
    txm: TxManager,
    chat: ChatModels,
    config: VerifyConfigs,
) -> EvalVerifyResponse:
    """The full ADR-0015 pipeline on `answer` over `shown_chunk_ids` (S1.. in order), with the
    per-layer evidence before merging (agent/eval_verify.py). The notes are re-read through RLS
    in a short transaction; the verifier runs after it closes (PHASE1: no model call in a tx)."""
    ids = list(body.shown_chunk_ids)
    if len(set(ids)) != len(ids):
        raise ValidationFailed("shown_chunk_ids: each chunk at most once (one S label each)")
    collections = list(dict.fromkeys(body.collection_ids))
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        await require_collections(conn, collections)
        notes = await _shown_notes(conn, ids, collections)
    verdict = await verify_answer(chat, notes, body.answer, strict=config.strict)
    # Counts and the outcome only, never text (the answer may hold note text).
    log.info(
        "eval_verify",
        status=verdict.status.value,
        verifier_ran=verdict.verifier_ran,
        coverage_units=len(verdict.coverage_units),
        verifier_claims=len(verdict.verifier_claims),
        shown=len(notes),
    )
    return _verify_out(verdict)


async def _shown_notes(
    conn: AsyncConnection, ids: Sequence[UUID], collections: Sequence[UUID]
) -> list[RankedChunk]:
    """The shown chunks in label order; 404 unless EVERY id is a live chunk of the requested
    collections visible to the caller (which one is never said)."""
    rows = await fetch_all(conn, _SHOWN_SQL, {"ids": list(ids), "colls": list(collections)})
    by_id = {row["id"]: _note(row) for row in rows}
    if set(by_id) != set(ids):
        raise NotFound("chunk not found")
    return [by_id[i] for i in ids]


def _note(row: RowMapping) -> RankedChunk:
    return RankedChunk(
        chunk_id=row["id"],
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        anchor=row["anchor"],
        title=row["title"],
        heading_path=tuple(row["heading_path"] or ()),
        display_text=row["display_text"],
        display_sha256=bytes(row["display_sha256"]),
        external_url=row["external_url"],
        fused_score=0.0,
    )


def _verify_out(v: EvalVerdict) -> EvalVerifyResponse:
    return EvalVerifyResponse(
        status=VerificationStatus(v.status.value),
        claims=[
            EvalFlaggedClaim(
                text=c.span.text, start=c.span.start, end=c.span.end, reason=ClaimReason(c.reason)
            )
            for c in v.claims
        ],
        coverage_units=[
            EvalCoverageUnit(text=u.span.text, start=u.span.start, end=u.span.end, kind=u.kind)
            for u in v.coverage_units
        ],
        verifier_claims=[
            EvalVerifierClaim(text=c.text, start=None, end=None)
            if c.span is None
            else EvalVerifierClaim(text=c.span.text, start=c.span.start, end=c.span.end)
            for c in v.verifier_claims
        ],
        verifier_ran=v.verifier_ran,
        verifier_verdict=(
            None if v.verifier_verdict is None else VerificationStatus(v.verifier_verdict.value)
        ),
        verifier_conclusive=v.verifier_conclusive,
        verifier_model=v.verifier_model,
        verifier_prompt=v.verifier_prompt,
    )


# ----------------------------------------------------------------- chunks
def parse_ids(raw: str) -> list[UUID]:
    """`ids=a,b,c` (the SDK's wire form), deduplicated in order; 422 on a malformed id or more
    than MAX_CHUNK_IDS. Spaces around commas are tolerated."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValidationFailed("ids: at least one chunk id is required")
    try:
        ids = list(dict.fromkeys(UUID(p) for p in parts))
    except ValueError:
        raise ValidationFailed("ids: every id must be a UUID") from None
    if len(ids) > MAX_CHUNK_IDS:
        raise ValidationFailed(f"ids: at most {MAX_CHUNK_IDS} per request")
    return ids


@router.get("/chunks")
async def eval_chunks(
    key: EvalKey,
    txm: TxManager,
    ids: Annotated[str, Query(max_length=MAX_CHUNK_IDS * 40)],
) -> EvalChunksResponse:
    """The shown TEXT of chunks the caller can see, in request order; invisible ids are absent
    (the client checks each `display_sha256` against its run before judging or labelling)."""
    wanted = parse_ids(ids)
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        rows = await fetch_all(conn, _CHUNKS_SQL, {"ids": wanted})
        documents = list(dict.fromkeys(row["document_id"] for row in rows))
        siblings = await fetch_all(conn, _PAGE_CHUNKS_SQL, {"docs": documents}) if rows else []
    occurrence = occurrences(siblings)
    by_id = {row["id"]: row for row in rows}
    return EvalChunksResponse(
        chunks=[_chunk_out(by_id[i], occurrence.get(i, 0)) for i in wanted if i in by_id]
    )


def occurrences(page_chunks: Sequence[RowMapping]) -> dict[UUID, int]:
    """Each chunk's rank among chunks of its page with an equal (heading_path, display_sha256),
    in ordinal order (rows arrive ordered by document, ordinal): the ChunkKey occurrence, by
    the same rule `match_chunk` resolves with, so a key from here always resolves back."""
    seen: dict[tuple[UUID, tuple[str, ...], bytes], int] = {}
    out: dict[UUID, int] = {}
    for row in page_chunks:
        ident = (row["document_id"], tuple(row["heading_path"] or ()), bytes(row["display_sha256"]))
        out[row["id"]] = seen.get(ident, 0)
        seen[ident] = out[row["id"]] + 1
    return out


def _chunk_out(row: RowMapping, occurrence: int = 0) -> EvalChunk:
    page = PageKey(source_kind=row["source_kind"], key=row["page_key"])
    heading = list(row["heading_path"] or ())
    digest = bytes(row["display_sha256"]).hex()
    return EvalChunk(
        chunk_id=row["id"],
        document_id=row["document_id"],
        page_key=page,
        heading_path=heading,
        ordinal=row["ordinal"],
        display_text=row["display_text"],
        display_sha256=digest,
        chunk_key=ChunkKey(
            page=page, heading_path=heading, display_sha256=digest, occurrence=occurrence
        ),
    )


# ----------------------------------------------------------------- resolve
@dataclass(frozen=True, slots=True)
class PageChunk:
    """One chunk of a resolved page, in ordinal order: what a chunk key is matched against."""

    chunk_id: UUID
    heading_path: tuple[str, ...]
    display_sha256: str  # hex


def resolve_chunk(key: ChunkKey, page: Sequence[PageChunk] | None) -> ResolvedChunk:
    """Match one chunk key against its page's chunks (ordinal order). Pure, so the occurrence
    and stale rules are testable without a database.

    - no page: `missing`;
    - the `occurrence`-th chunk (0-based, ordinal order) with an equal (heading_path, digest):
      `ok`;
    - otherwise `stale`, with every chunk under the same heading_path as a candidate (ordinal
      order), possibly none. Never re-mapped: the owner re-approves (ADR-0017 principle 6)."""
    if page is None:
        return ResolvedChunk(key=key, status=ResolveStatus.missing)
    heading = tuple(key.heading_path)
    equal = [
        c for c in page if c.heading_path == heading and c.display_sha256 == key.display_sha256
    ]
    if key.occurrence < len(equal):
        return ResolvedChunk(
            key=key, status=ResolveStatus.ok, chunk_id=equal[key.occurrence].chunk_id
        )
    candidates = [c.chunk_id for c in page if c.heading_path == heading]
    return ResolvedChunk(key=key, status=ResolveStatus.stale, candidates=candidates)


def resolve_ambiguous_chunk(key: ChunkKey, pages: Sequence[Sequence[PageChunk]]) -> ResolvedChunk:
    """A chunk key whose page key names several documents: never an id, the chunks under the
    same heading_path of EVERY such document as candidates (newest document first)."""
    heading = tuple(key.heading_path)
    candidates = [c.chunk_id for page in pages for c in page if c.heading_path == heading]
    return ResolvedChunk(key=key, status=AMBIGUOUS, candidates=candidates)


def _page_ref(key: PageKey) -> tuple[str, str]:
    return key.source_kind, key.key


@router.post("/resolve")
async def eval_resolve(
    body: EvalResolveRequest, key: EvalKey, txm: TxManager
) -> EvalResolveResponse:
    """Stable keys to current ids within `collection_ids` (each must be visible: 404 otherwise).
    Two live pages with one key (a file uploaded twice) are `ambiguous`, never resolved to one of
    them: filenames are meant to be unique per collection (ADR-0017 principle 6), so the owner
    fixes the collection or re-approves the item."""
    collections = list(dict.fromkeys(body.collection_ids))
    refs = list(
        dict.fromkeys(
            [*map(_page_ref, body.page_keys), *(_page_ref(c.page) for c in body.chunk_keys)]
        )
    )
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        await require_collections(conn, collections)
        pages = await _pages(conn, collections, refs)
        documents = [d for docs in pages.values() for d in docs]
        chunks = await _page_chunks(conn, documents) if body.chunk_keys else {}
    if any(len(docs) > 1 for docs in pages.values()):  # counts only: a filename is note data
        log.warning("eval_resolve_ambiguous", keys=sum(len(d) > 1 for d in pages.values()))
    return EvalResolveResponse(
        pages=[_page_out(k, pages.get(_page_ref(k), ())) for k in body.page_keys],
        chunks=[
            _chunk_resolved(c, pages.get(_page_ref(c.page), ()), chunks) for c in body.chunk_keys
        ],
    )


async def _pages(
    conn: AsyncConnection, collections: Sequence[UUID], refs: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], tuple[UUID, ...]]:
    """(source_kind, key) -> EVERY live document with that page key, newest first (one, unless
    the key is ambiguous)."""
    if not refs:
        return {}
    keys = sorted({k for _, k in refs})
    rows = await fetch_all(conn, _PAGES_SQL, {"colls": list(collections), "keys": keys})
    wanted = set(refs)
    found: dict[tuple[str, str], list[UUID]] = defaultdict(list)
    for row in rows:
        ref = (str(row["source_kind"]), str(row["page_key"]))
        if ref in wanted:
            found[ref].append(row["id"])
    return {ref: tuple(docs) for ref, docs in found.items()}


async def _page_chunks(
    conn: AsyncConnection, documents: Sequence[UUID]
) -> dict[UUID, list[PageChunk]]:
    if not documents:
        return {}
    rows = await fetch_all(conn, _PAGE_CHUNKS_SQL, {"docs": list(dict.fromkeys(documents))})
    out: dict[UUID, list[PageChunk]] = defaultdict(list)
    for row in rows:
        out[row["document_id"]].append(
            PageChunk(
                chunk_id=row["id"],
                heading_path=tuple(row["heading_path"] or ()),
                display_sha256=bytes(row["display_sha256"]).hex(),
            )
        )
    return out


def _chunk_resolved(
    key: ChunkKey, documents: Sequence[UUID], chunks: dict[UUID, list[PageChunk]]
) -> ResolvedChunk:
    if len(documents) > 1:
        return resolve_ambiguous_chunk(key, [chunks.get(d, []) for d in documents])
    return resolve_chunk(key, chunks.get(documents[0], []) if documents else None)


def _page_out(key: PageKey, documents: Sequence[UUID]) -> ResolvedPage:
    if not documents:
        return ResolvedPage(key=key, status=ResolveStatus.missing)
    if len(documents) > 1:
        return ResolvedPage(key=key, status=AMBIGUOUS)
    return ResolvedPage(key=key, status=ResolveStatus.ok, document_id=documents[0])
