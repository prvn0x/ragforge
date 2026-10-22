"""POST /v1/interview/question and POST /v1/interview/grade (interview mode, Phases 3b/3c).

Scope `chat`, like /v1/chat, behind the per-principal quota (`deps.InterviewKey`: 429 with
Retry-After). The session is client-held: the server keeps nothing between calls. The decisions
live in `agent/interview` (pure, eval-callable); this module only maps the wire contract to the
domain, runs the SQL reads, sanitizes every text on the way out (ADR-0011: model output is
sanitized once, here) and signs/checks the `question_token`.

Error mapping: a collection or chunk the caller cannot see -> 404 (never says which); nothing
matching the topic at all -> 409 with Problem.type INTERVIEW_NO_NOTES, everything matching asked
already -> 409 INTERVIEW_EXHAUSTED (clients branch on the type, and may then OFFER source=general;
the server never switches); a model failure -> the provider's own Problem (502/503, as on
/v1/chat), and a reply still unusable after one repair -> 502; a malformed or old-version
question_token -> 400, a forged/expired/foreign one, or one for another level or source -> 403.

Level and source (Phase 3c): /question resolves `level=None` to the configured default and ALWAYS
sets `level` and `source` on its response; the token binds both. /grade uses the request's echoed
level (None: the default), checked by the token. source=general reads no note (`service.py`),
returns no chunk ids and no citations, and its grade is always "unverified" with no flagged
claims; its texts are sanitized with an EMPTY link allow-list (there is no page to link to).

Eval seams (ADR-0017 §1; eval scope + RAGFORGE_EVAL_API, `deps.require_eval_seam`):
- /question `chunk_ids` targets those seed notes instead of selection (403 for any other key,
  422 with source=general, 404 unless each lies in `collection_ids`); the token is a normal v3
  token over the notes actually used.
- /grade: an eval-scope NOTES grade needs no token and its level is free (the frozen D4 items
  have no token and are graded at all five levels). That is today's optional-token path, made an
  explicit exemption (`token_rule`) so it survives when enforcement lands. Every other key keeps
  the v3 binding unchanged; a token that IS sent is always checked; general mode still requires
  its token (the contract's validator), eval scope or not.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    Citation,
    ClaimReason,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewLevel,
    InterviewQuestion,
    InterviewQuestionRequest,
    InterviewSource,
    Problem,
    UnsupportedClaim,
    VerificationStatus,
)
from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.interview.model import (
    Difficulty,
    DraftGrade,
    DraftQuestion,
    GradeRequest,
    InterviewSettings,
    Kind,
    Level,
    Misconception,
    QuestionRequest,
    Source,
)
from ragforge_core.agent.interview.phrases import PhraseCache
from ragforge_core.agent.interview.service import (
    InterviewDeps,
    InterviewExhausted,
    InterviewNoNotes,
    grade,
    next_question,
    targeted_question,
)
from ragforge_core.agent.interview.tokens import QuestionBinding
from ragforge_core.apps.api.chat_stream import citation_from_hit
from ragforge_core.apps.api.deps import (
    ChatKey,
    ChatModels,
    Embedders,
    EvalApiEnabled,
    InterviewConfig,
    InterviewKey,
    QuestionTokenSigner,
    Rerank,
    Retrieval,
    SanitizerFactory,
    Sanitizers,
    TxManager,
    eval_seam_allowed,
    require_eval_seam,
)
from ragforge_core.apps.api.errors import PROBLEM_JSON
from ragforge_core.apps.api.routes.sources import require_collections
from ragforge_core.apps.api.sql import fetch_all
from ragforge_core.domain.errors import RagforgeError, ValidationFailed
from ragforge_core.domain.models import Principal, RankedChunk
from ragforge_core.domain.ports import Tx
from ragforge_core.observability import metrics
from ragforge_core.services.auth.principal import AuthenticatedKey

router = APIRouter(prefix="/v1")

# Shorter than this a chunk is usually a heading-only fragment (selection.MIN_SEED_CHARS).
_SUBSTANTIVE_CHARS: Final = 80


# ----------------------------------------------------------------- SQL reads (InterviewNotes)
# Every read applies the visibility filter retrieval applies (agent/interview/ports.py): RLS
# (tenant), the requested collections, the collection's ACTIVE embedding space, live pages.
# Composed only from these constants; every user-supplied value is a bind parameter.
_COLUMNS: Final = """
    c.id, c.document_id, c.collection_id, c.anchor, d.title, c.heading_path,
    c.display_text, c.display_sha256, d.external_url, c.ordinal
"""
_VISIBLE: Final = """
      FROM rf.chunks c
      JOIN rf.documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
                         AND d.deleted_at IS NULL
      JOIN rf.collections col ON col.id = c.collection_id AND col.tenant_id = c.tenant_id
      JOIN rf.embedding_spaces s ON s.id = col.active_space_id AND s.status = 'active'
     WHERE c.collection_id = ANY (CAST(:collections AS uuid[]))
"""
# Title and heading path, with the collection's own text-search configuration (the one its
# keyword index uses, migration 0003), so stemming matches it; the topic is a bound parameter
# given to plainto_tsquery, which reads it as plain words: never a pattern or query syntax.
_TOPIC_SQL: Final = f"""
    SELECT {_COLUMNS} {_VISIBLE}
       AND NOT (c.id = ANY (CAST(:exclude AS uuid[])))
       AND (to_tsvector(col.fts_config, d.title)
            || to_tsvector(col.fts_config, array_to_string(c.heading_path, ' ')))
           @@ plainto_tsquery(col.fts_config, :topic)
     ORDER BY random()
     LIMIT :limit
"""
# The lexical gate on search hits (ports.InterviewNotes.mentioning): every topic word in the
# chunk's text, its page title or its heading path. Read with `/-_.` as spaces on BOTH sides
# (the topic is normalised the same way, service.gate_terms), so "event-driven" meets "event
# driven" and "actuator" meets "/actuator/health" (final review P1). Only the <= rerank_top
# candidate ids are read, so vectorising their text here is cheap.
_MENTIONING_SQL: Final = f"""
    SELECT c.id {_VISIBLE}
       AND c.id = ANY (CAST(:ids AS uuid[]))
       AND (to_tsvector(col.fts_config, translate(
                d.title || ' ' || array_to_string(c.heading_path, ' ') || ' ' || c.display_text,
                '/-_.', '    '))
            @@ plainto_tsquery(col.fts_config, :topic)
            -- only stop words left ("IT", "to the"): no lexeme to require, the reranker decides
            -- (parenthesised: the OR must never escape the visibility filter above)
            OR numnode(plainto_tsquery(col.fts_config, :topic)) = 0)
"""
_SAMPLE_SQL: Final = f"""
    WITH used AS (
        SELECT DISTINCT document_id FROM rf.chunks WHERE id = ANY (CAST(:exclude AS uuid[]))
    ), one_per_page AS (
        SELECT DISTINCT ON (c.document_id) {_COLUMNS} {_VISIBLE}
           AND NOT (c.id = ANY (CAST(:exclude AS uuid[])))
         ORDER BY c.document_id, length(c.display_text) < :substantive, random()
    )
    SELECT * FROM one_per_page p
     ORDER BY EXISTS (SELECT 1 FROM used u WHERE u.document_id = p.document_id), random()
     LIMIT :limit
"""  # noqa: S608
_BY_ID_SQL: Final = f"""
    SELECT {_COLUMNS} {_VISIBLE} AND c.id = ANY (CAST(:ids AS uuid[]))
"""
_NEIGHBOURS_SQL: Final = f"""
    SELECT {_COLUMNS} {_VISIBLE}
       AND c.document_id = (SELECT document_id FROM rf.chunks WHERE id = :id)
       AND abs(c.ordinal - (SELECT ordinal FROM rf.chunks WHERE id = :id)) <= :radius
     ORDER BY c.ordinal
"""  # noqa: S608
# Acronym topics (agent/interview/acronyms.py): the distinct titles and heading segments of the
# visible chunks, matched in Python. No user text at all reaches this query; LIMIT bounds what
# is returned and ORDER BY makes the truncation deterministic (the same phrases every time). It
# still reads every visible chunk, so the service caches the result briefly (`PhraseCache`).
_PHRASES_SQL: Final = f"""
    SELECT phrase FROM (
        SELECT d.title AS phrase {_VISIBLE}
        UNION
        SELECT unnest(c.heading_path) AS phrase {_VISIBLE}
    ) p
     WHERE phrase <> ''
     ORDER BY phrase
     LIMIT :limit
"""  # noqa: S608
_DOCUMENTS_SQL: Final = (
    "SELECT DISTINCT document_id FROM rf.chunks WHERE id = ANY (CAST(:ids AS uuid[]))"
)


def _chunk(row: RowMapping) -> RankedChunk:
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


class SqlInterviewNotes:
    """`agent.interview.ports.InterviewNotes` on Postgres. Stateless; one per request is free."""

    async def require_collections(self, tx: Tx, collection_ids: Sequence[UUID]) -> None:
        await require_collections(connection(tx), list(dict.fromkeys(collection_ids)))

    async def topic_chunks(
        self,
        tx: Tx,
        collection_ids: Sequence[UUID],
        topic: str,
        exclude: Collection[UUID],
        limit: int,
    ) -> list[RankedChunk]:
        params = {"topic": topic, "exclude": list(exclude), "limit": limit}
        return await _read(tx, _TOPIC_SQL, collection_ids, params)

    async def mentioning(
        self, tx: Tx, collection_ids: Sequence[UUID], chunk_ids: Collection[UUID], topic: str
    ) -> frozenset[UUID]:
        if not chunk_ids:
            return frozenset()
        params = {"collections": list(collection_ids), "ids": list(chunk_ids), "topic": topic}
        rows = await fetch_all(connection(tx), _MENTIONING_SQL, params)
        return frozenset(row["id"] for row in rows)

    async def topic_phrases(self, tx: Tx, collection_ids: Sequence[UUID], limit: int) -> list[str]:
        params = {"collections": list(collection_ids), "limit": limit}
        rows = await fetch_all(connection(tx), _PHRASES_SQL, params)
        return [str(r["phrase"]) for r in rows]

    async def sample_chunks(
        self, tx: Tx, collection_ids: Sequence[UUID], exclude: Collection[UUID], limit: int
    ) -> list[RankedChunk]:
        params = {"exclude": list(exclude), "limit": limit, "substantive": _SUBSTANTIVE_CHARS}
        return await _read(tx, _SAMPLE_SQL, collection_ids, params)

    async def documents_of(self, tx: Tx, chunk_ids: Collection[UUID]) -> frozenset[UUID]:
        rows = await fetch_all(connection(tx), _DOCUMENTS_SQL, {"ids": list(chunk_ids)})
        return frozenset(r["document_id"] for r in rows)

    async def chunks_by_id(
        self, tx: Tx, chunk_ids: Sequence[UUID], collection_ids: Sequence[UUID]
    ) -> list[RankedChunk]:
        return await _read(tx, _BY_ID_SQL, collection_ids, {"ids": list(chunk_ids)})

    async def neighbours(
        self, tx: Tx, chunk_id: UUID, collection_ids: Sequence[UUID], radius: int
    ) -> list[RankedChunk]:
        params = {"id": chunk_id, "radius": radius}
        return await _read(tx, _NEIGHBOURS_SQL, collection_ids, params)


async def _read(
    tx: Tx, sql: str, collection_ids: Sequence[UUID], params: dict[str, Any]
) -> list[RankedChunk]:
    conn: AsyncConnection = connection(tx)
    rows = await fetch_all(conn, sql, {"collections": list(collection_ids), **params})
    return [_chunk(r) for r in rows]


def get_interview_notes() -> SqlInterviewNotes:
    return SqlInterviewNotes()


Notes = Annotated[SqlInterviewNotes, Depends(get_interview_notes)]


# One per process: the phrase read scans every visible chunk, and a session repeats its topic.
# Keyed by tenant and collections, so it never answers across tenants (`phrases.py`).
_PHRASES: Final = PhraseCache()


def get_interview_deps(
    txm: TxManager,
    notes: Notes,
    retriever: Retrieval,
    embedders: Embedders,
    reranker: Rerank,
    chat: ChatModels,
    settings: InterviewConfig,
) -> InterviewDeps:
    return InterviewDeps(
        tx=txm,
        notes=notes,
        retriever=retriever,
        embedders=embedders,
        reranker=reranker,
        chat=chat,
        settings=settings,
        phrases=_PHRASES,
    )


Interview = Annotated[InterviewDeps, Depends(get_interview_deps)]


def get_interview_rng() -> random.Random:
    """The seed sampler's randomness (tests and evals override it with a seeded one)."""
    return random.Random()  # noqa: S311 - choosing a note, not a secret


Rng = Annotated[random.Random, Depends(get_interview_rng)]


# ----------------------------------------------------------------- output mapping
class _Clean:
    """Sanitizes each output text ONCE, with a fresh sanitizer per text (markdown escaping is
    not idempotent) and the notes' own URLs as the link allow-list."""

    def __init__(self, new: SanitizerFactory, notes: Sequence[RankedChunk]) -> None:
        self._new = new
        self._urls = [n.external_url for n in notes if n.external_url]

    def __call__(self, text: str) -> str:
        sanitizer = self._new()
        if self._urls:
            sanitizer.allow(self._urls)
        return (sanitizer.feed(text) + sanitizer.finish()).strip()


def _citations(notes: Sequence[RankedChunk], inferred: Collection[int] = ()) -> list[Citation]:
    return [
        citation_from_hit(note, label=n, inferred=n in inferred)
        for n, note in enumerate(notes, start=1)
    ]


def _binding(
    key: AuthenticatedKey,
    collection_ids: Sequence[UUID],
    chunk_ids: Sequence[UUID],
    question: str,
    expected_points: Sequence[str],
    kind: str,
    difficulty: str,
    level: Level,
    source: Source,
) -> QuestionBinding:
    return QuestionBinding(
        tenant_id=key.principal.tenant_id,
        user_id=key.principal.user_id,
        collection_ids=tuple(collection_ids),
        chunk_ids=tuple(chunk_ids),
        question=question,
        expected_points=tuple(expected_points),
        kind=kind,
        difficulty=difficulty,
        level=level.value,
        source=source.value,
    )


def _level(requested: InterviewLevel | None, settings: InterviewSettings) -> Level:
    """The request's level, or the configured default: resolved ONCE, before anything is signed
    or prompted, so the token, the prompts and the response all name the same bar."""
    return Level(requested.value) if requested is not None else settings.target_level


class TokenRule(StrEnum):
    """Whether /grade needs a question_token (module docstring)."""

    required = "required"  # general mode: the token is the only binding (contract validator)
    optional = "optional"  # notes mode today: checked when sent; enforcement lands later
    exempt = "exempt"  # eval-scope notes grade: frozen items have no token; level is free


# Flipped when token enforcement lands for notes mode; the eval exemption below survives it.
ENFORCE_NOTES_TOKEN: Final = False


def token_rule(
    principal: Principal, source: Source, eval_enabled: bool, *, enforce: bool = ENFORCE_NOTES_TOKEN
) -> TokenRule:
    """The eval scope is checked FIRST (ADR-0017 §1, r3 P1-e): it exempts notes mode only."""
    if source is Source.general:
        return TokenRule.required
    if eval_seam_allowed(principal, eval_enabled):
        return TokenRule.exempt
    return TokenRule.required if enforce else TokenRule.optional


def _exhausted(request: Request, exc: InterviewExhausted) -> JSONResponse:
    """The 409 with the contract's Problem.type (NO_NOTES or EXHAUSTED), so clients branch on
    the type, not the status (the shared handler always writes about:blank)."""
    rid = request.scope.get("state", {}).get("request_id")
    body = Problem(
        type=INTERVIEW_NO_NOTES if isinstance(exc, InterviewNoNotes) else INTERVIEW_EXHAUSTED,
        title=exc.title,
        status=exc.status,
        detail=str(exc),
        instance=request.url.path,
        request_id=rid,
    )
    return JSONResponse(
        body.model_dump(exclude_none=True),
        status_code=exc.status,
        media_type=PROBLEM_JSON,
        headers={"x-request-id": rid} if rid else None,
    )


def _question_out(
    draft: DraftQuestion,
    clean: _Clean,
    key: AuthenticatedKey,
    tokens: QuestionTokenSigner,
    collection_ids: Sequence[UUID],
) -> InterviewQuestion:
    question = clean(draft.question)
    points = [clean(p) for p in draft.expected_points]
    chunk_ids = [n.chunk_id for n in draft.notes]
    kind, difficulty = draft.kind.value, draft.difficulty.value
    binding = _binding(
        key, collection_ids, chunk_ids, question, points, kind, difficulty,
        draft.level, draft.source,
    )  # fmt: skip
    return InterviewQuestion(
        question=question,
        kind=draft.kind.value,
        topic=clean(draft.topic),
        difficulty=draft.difficulty.value,
        chunk_ids=chunk_ids,
        expected_points=points,
        citations=_citations(draft.notes),
        question_token=tokens.issue(binding),
        level=InterviewLevel(draft.level.value),
        source=InterviewSource(draft.source.value),
    )


def _misconception(item: Misconception, source: Source) -> str:
    """What the user said, then what is actually so, and from WHERE: the notes (verified, with
    its label) or general knowledge (the model's own, unverified; never "your notes")."""
    where = "your notes" if source is Source.notes else "general knowledge"
    return f"{item.said} ({where}: {item.correction})"


def _grade_out(draft: DraftGrade, clean: _Clean) -> InterviewGrade:
    answer = draft.answer
    return InterviewGrade(
        score=draft.score,
        summary=clean(draft.summary),
        right=[clean(r) for r in draft.right],
        missed=[clean(m) for m in draft.missed],
        misconceptions=[clean(_misconception(m, draft.source)) for m in draft.misconceptions],
        model_answer=clean(answer.text),
        citations=_citations(draft.notes, inferred=answer.inferred),
        verification=VerificationStatus(answer.status.value),
        unsupported_claims=[
            UnsupportedClaim(text=clean(c.text), reason=ClaimReason(c.reason))
            for c in answer.claims
        ],
        level=InterviewLevel(draft.level.value),
        source=InterviewSource(draft.source.value),
    )


# ----------------------------------------------------------------- metrics
@dataclass(slots=True)
class _Call:
    """The outcome of one interview call; a route sets it for answers it returns (409s)."""

    outcome: str = "error"


@contextmanager
def _measured(endpoint: str, source: Source) -> Iterator[_Call]:
    """`ragforge_interview_requests_total{endpoint, source, outcome}` (ADR-0018) for the route
    body: "ok", the two 409 kinds, "rejected" for any other 4xx problem (a bad token, a chunk
    the caller cannot see), "error" for everything else (model failures included)."""
    call = _Call()
    try:
        yield call
        if call.outcome == "error":
            call.outcome = "ok"
    except RagforgeError as exc:
        call.outcome = "rejected" if 400 <= exc.status < 500 else "error"
        raise
    finally:
        metrics.INTERVIEW_REQUESTS.labels(
            endpoint,
            metrics.bounded(source.value, metrics.INTERVIEW_SOURCES),
            metrics.bounded(call.outcome, metrics.INTERVIEW_OUTCOMES),
        ).inc()


def _exhausted_outcome(exc: InterviewExhausted) -> str:
    return "no_notes" if isinstance(exc, InterviewNoNotes) else "exhausted"


# ----------------------------------------------------------------- routes
async def question_seams(
    body: InterviewQuestionRequest, key: ChatKey, eval_enabled: EvalApiEnabled
) -> None:
    """The eval seam of /v1/interview/question (`chunk_ids`), checked as a dependency declared
    BEFORE `InterviewKey`, so a refused call spends no interview quota (the `eval_quota` rule;
    review round 1, P2). `key` is the same cached ChatKey InterviewKey resolves."""
    if body.chunk_ids:
        require_eval_seam(key.principal, eval_enabled)
        if body.source.value == Source.general.value:
            raise ValidationFailed("chunk_ids: only for source=notes (general reads no note)")


@router.post(
    "/interview/question",
    tags=["interview"],
    response_model=InterviewQuestion,
    responses={
        409: {
            "model": Problem,
            "description": "type INTERVIEW_NO_NOTES: nothing matches the topic at all; "
            "type INTERVIEW_EXHAUSTED: every matching note was already asked",
        }
    },
)
async def interview_question(
    body: InterviewQuestionRequest,
    request: Request,
    _seams: Annotated[None, Depends(question_seams)],  # before the quota: see question_seams
    key: InterviewKey,
    interview: Interview,
    sanitizers: Sanitizers,
    tokens: QuestionTokenSigner,
    rng: Rng,
    settings: InterviewConfig,
) -> InterviewQuestion | JSONResponse:
    source = Source(body.source.value)
    ask = QuestionRequest(
        collection_ids=tuple(dict.fromkeys(body.collection_ids)),
        topic=body.topic,
        kinds=tuple(dict.fromkeys(Kind(k.value) for k in body.kinds)),
        difficulty=Difficulty(body.difficulty.value),
        exclude_chunk_ids=frozenset(body.exclude_chunk_ids),
        level=_level(body.level, settings),
        source=source,
        # notes mode ignores them (contract); never handed to the notes path at all
        exclude_questions=tuple(body.exclude_questions) if source is Source.general else (),
    )
    with _measured("question", source) as call:
        try:
            if body.chunk_ids:
                draft = await targeted_question(
                    interview, key.principal.tenant_id, ask, body.chunk_ids
                )
            else:
                draft = await next_question(interview, key.principal.tenant_id, ask, rng)
        except InterviewExhausted as exc:
            call.outcome = _exhausted_outcome(exc)
            return _exhausted(request, exc)
        clean = _Clean(sanitizers, draft.notes)
        return _question_out(draft, clean, key, tokens, ask.collection_ids)


@router.post("/interview/grade", tags=["interview"])
async def interview_grade(
    body: InterviewGradeRequest,
    key: InterviewKey,
    interview: Interview,
    sanitizers: Sanitizers,
    tokens: QuestionTokenSigner,
    settings: InterviewConfig,
    eval_enabled: EvalApiEnabled,
) -> InterviewGrade:
    """source=notes: the notes are re-read through RLS from `chunk_ids`. Either source: the
    client's question, expected points and answer are untrusted DATA for the grader, never
    instructions or evidence. Whether a token is needed is `token_rule`'s decision: optional
    for notes (enforced later), exempt for an eval-scope notes grade, required for general (the
    contract's validator too): there it is the only binding of the question and key."""
    level = _level(body.level, settings)
    source = Source(body.source.value)
    with _measured("grade", source):
        return await _grade(body, key, interview, sanitizers, tokens, eval_enabled, level, source)


async def _grade(
    body: InterviewGradeRequest,
    key: AuthenticatedKey,
    interview: InterviewDeps,
    sanitizers: SanitizerFactory,
    tokens: QuestionTokenSigner,
    eval_enabled: bool,
    level: Level,
    source: Source,
) -> InterviewGrade:
    rule = token_rule(key.principal, source, eval_enabled)
    if body.question_token is None and rule is TokenRule.required:
        raise ValidationFailed("question_token: required")
    if body.question_token is not None:
        binding = _binding(
            key,
            body.collection_ids,
            body.chunk_ids,
            body.question,
            body.expected_points,
            body.kind.value,
            body.difficulty.value,
            level,
            source,
        )
        tokens.verify(body.question_token, binding)
    graded = GradeRequest(
        collection_ids=tuple(dict.fromkeys(body.collection_ids)),
        question=body.question,
        kind=Kind(body.kind.value),
        difficulty=Difficulty(body.difficulty.value),
        chunk_ids=tuple(body.chunk_ids),
        expected_points=tuple(body.expected_points),
        answer=body.answer,
        level=level,
        source=source,
    )
    draft = await grade(interview, key.principal.tenant_id, graded)
    return _grade_out(draft, _Clean(sanitizers, draft.notes))
