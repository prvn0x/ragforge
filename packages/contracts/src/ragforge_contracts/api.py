"""Request/response bodies for /v1. See docs/DESIGN.md section 9.

Evolution rule (expand/contract, ADR-0002): REQUESTS forbid unknown fields (catch client bugs);
RESPONSES ignore unknown fields, so a server may add fields without breaking older SDK/MCP/UI.
Enum-valued RESPONSE fields are typed `Enum | str` for the same reason (new values must not crash
older clients); request fields stay strict enums.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    SecretStr,
    model_validator,
)

# Known values parse to the enum; unknown values from a newer server pass through as str.
OPEN = Field(union_mode="left_to_right")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Page[T](_Response):
    """Cursor pagination. `next_cursor` is opaque; absent means last page."""

    items: list[T]
    next_cursor: str | None = None


class Accepted(_Response):
    """202 response for asynchronous work (uploads, syncs, reindex)."""

    job_id: UUID


# ----------------------------------------------------------------- errors (RFC 9457)
class Problem(_Response):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    request_id: str | None = None


# ----------------------------------------------------------------- health
class Readiness(_Response):
    ready: bool
    schema_version: int | None
    required_schema: int
    code_version: int
    min_code_version: int | None
    reason: str | None = None


# ----------------------------------------------------------------- collections
class CollectionCreate(_Request):
    name: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class Collection(_Response):
    id: UUID
    name: str
    active_space_id: int
    created_at: datetime


# ----------------------------------------------------------------- connectors
class ConnectorKind(StrEnum):
    notion = "notion"


class ConnectorStatus(StrEnum):
    active = "active"
    paused = "paused"
    error = "error"
    deleting = "deleting"


class ConnectorCreate(_Request):
    kind: ConnectorKind
    target_collection_id: UUID
    display_name: Annotated[str, Field(min_length=1, max_length=200)]
    token: SecretStr  # never echoed, logged or dumped in clear


class Connector(_Response):
    id: UUID
    kind: Annotated[ConnectorKind | str, OPEN]
    target_collection_id: UUID
    display_name: str
    status: Annotated[ConnectorStatus | str, OPEN]
    last_error: str | None
    created_at: datetime


# ----------------------------------------------------------------- documents
class Document(_Response):
    id: UUID
    collection_id: UUID
    title: str
    external_url: str | None
    external_updated_at: datetime | None
    fetched_at: datetime | None


class Exclusion(_Response):
    id: UUID
    source_kind: str
    external_id: str
    reason: str | None
    created_at: datetime


# ----------------------------------------------------------------- jobs
class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    dead = "dead"
    superseded = "superseded"


class Job(_Response):
    id: UUID
    kind: str
    status: Annotated[JobStatus | str, OPEN]
    attempts: int
    last_error: str | None
    enqueued_at: datetime
    progress: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------- retrieval
class QueryRequest(_Request):
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    k: Annotated[int, Field(ge=1, le=50)] = 8
    # ADR-0017 E-R: also return every retrieval stage (requires the `eval` scope)
    stages: bool = False


class Citation(_Response):
    anchor: str
    document_id: UUID
    title: str
    heading_path: list[str]
    url: HttpUrl | None
    snippet: str
    # True when the SERVER attributed this source to a sentence the model left uncited; such
    # citations passed the verifier but clients must label them differently (never presented as
    # the model's own provenance). Additive field: older clients ignore it.
    inferred: bool = False
    # The n of this source's "[S<n>]" label (chat only; None on /query). For a model citation it
    # appears in the answer text; for an inferred one (`inferred=True`) it does NOT: the server
    # matched the source to an uncited sentence, so clients must mark it (ask.py shows "S<n>*").
    # `anchor` is the chunk's source anchor (e.g. a Notion block id), never the answer's label.
    # Additive field: older clients ignore it.
    label: PositiveInt | None = None


class QueryHit(_Response):
    citation: Citation
    score: float
    rerank_score: float | None


class StageHit(_Response):
    """One candidate of one retrieval stage. Ranks are 1-based; ties broken by chunk_id."""

    chunk_id: UUID
    document_id: UUID
    rank: PositiveInt
    score: float


class RetrievalStages(_Response):
    keyword: list[StageHit]
    vector: list[StageHit]
    fused: list[StageHit]
    reranked: list[StageHit]


class QueryResponse(_Response):
    hits: list[QueryHit]
    trace_id: str | None = None
    stages: RetrievalStages | None = None  # only when requested with the eval scope


# ----------------------------------------------------------------- chat
class ChatRequest(_Request):
    conversation_id: UUID | None = None
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    message: Annotated[str, Field(min_length=1, max_length=8000)]


class ApprovalDecision(StrEnum):
    approve = "approve"
    reject = "reject"


class ApprovalResume(_Request):
    decisions: dict[str, ApprovalDecision]  # tool_call_id -> decision


# ----------------------------------------------------------------- job progress (SSE job events)
class JobEvent(_Response):
    job_id: UUID
    status: Annotated[JobStatus | str, OPEN]
    progress: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None


# ----------------------------------------------------------------- audit
class AuditEntry(_Response):
    id: int
    actor: str
    action: str
    target: str | None
    details: dict[str, Any]
    created_at: datetime


# ----------------------------------------------------------------- tenants, users, api keys
class TenantKind(StrEnum):
    personal = "personal"
    team = "team"


class TenantCreate(_Request):
    """Admin-only provisioning (definer function provision_tenant, ADR-0013)."""

    name: Annotated[str, Field(min_length=1, max_length=200)]
    kind: TenantKind = TenantKind.personal
    owner_display_name: Annotated[str, Field(min_length=1, max_length=200)]
    team_visibility_consent: bool = False


class Tenant(_Response):
    id: UUID
    name: str
    kind: Annotated[TenantKind | str, OPEN]
    created_at: datetime


class ApiScope(StrEnum):
    """Mirrors domain.models.Scope (a test asserts they stay identical)."""

    read = "read"
    write = "write"
    chat = "chat"
    approvals_write = "approvals:write"  # never granted to MCP keys
    admin = "admin"
    # ADR-0017: the eval surface. Only on the dedicated eval key of an eval tenant (with chat +
    # read); never combined with an MCP key; the whole surface sits behind RAGFORGE_EVAL_API.
    eval = "eval"


class ApiKeyCreate(_Request):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    scopes: Annotated[list[ApiScope], Field(min_length=1)]


class ApiKey(_Response):
    id: UUID
    name: str
    prefix: str
    scopes: list[str]  # str (not ApiScope) so older clients tolerate new scopes
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKey):
    """Returned exactly once at creation; the secret is never retrievable again."""

    secret: str


class TenantProvisioned(_Response):
    tenant: Tenant
    user_id: UUID
    api_key: ApiKeyCreated


# ----------------------------------------------------------------- flagged claims
class VerificationStatus(StrEnum):
    verified = "verified"
    partial = "partial"
    unsupported = "unsupported"
    unverified = "unverified"  # model below the verifier SLO (ADR-0012)


class ClaimReason(StrEnum):
    """Why a claim is flagged. Clients show `not_found` prominently and `inexact_code` muted."""

    not_found = "not_found"  # the verifier or coverage found no support in the notes
    inexact_code = "inexact_code"  # only code/formatting could not be matched exactly


class UnsupportedClaim(_Response):
    text: str  # sanitized claim text, as in VerificationEvent.unsupported_claims
    # Open: a newer server may add reasons; clients treat an unknown reason as not_found.
    reason: Annotated[ClaimReason | str, OPEN]


# ----------------------------------------------------------------- interview mode (Phase 3b)
class InterviewKind(StrEnum):
    """What a question tests (owner decision: all four, Senior SDE-1 bar)."""

    concept = "concept"  # explain a concept and why it matters
    scenario = "scenario"  # diagnose a failure / production issue the notes describe
    design = "design"  # system design / trade-offs
    code = "code"  # code / LLD (needs a code block in the source notes)


class InterviewDifficulty(StrEnum):
    """A fine adjustment WITHIN the level (Phase 3c): the level is the bar."""

    easier = "easier"
    standard = "standard"
    harder = "harder"


class InterviewLevel(StrEnum):
    """The bar a session is asked and graded at (Phase 3c, owner decision 2026-10-19). Fixed for
    a session; changing it starts a new one, so a session's average means one bar."""

    sde_1 = "sde_1"
    sde_2 = "sde_2"
    senior_sde_1 = "senior_sde_1"  # the default
    senior_sde_2 = "senior_sde_2"
    staff = "staff"


class InterviewSource(StrEnum):
    """Where a question comes from. `general` is the opt-in fallback when the notes have nothing
    (left) on a topic: model general knowledge, NOT from the notes, never verified or cited
    (same rule as ADR-0016 extras). Never chosen by the server on its own."""

    notes = "notes"
    general = "general"


class InterviewQuestionRequest(_Request):
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    # "kafka" -> questions from ALL Kafka pages; None -> mixed across every page
    topic: Annotated[str | None, Field(max_length=200)] = None
    kinds: Annotated[list[InterviewKind], Field(min_length=1)] = Field(
        default_factory=lambda: list(InterviewKind)
    )
    difficulty: InterviewDifficulty = InterviewDifficulty.standard
    # chunks already used this session: never ask from them again (no repeats)
    exclude_chunk_ids: Annotated[list[UUID], Field(max_length=500)] = Field(default_factory=list)
    level: InterviewLevel | None = None  # None: the server's configured default
    source: InterviewSource = InterviewSource.notes
    # source=general only: questions already asked this session on the topic (sanitized text as
    # received), so the model does not repeat itself. Ignored for source=notes.
    exclude_questions: Annotated[
        list[Annotated[str, Field(max_length=1000)]], Field(max_length=20)
    ] = Field(default_factory=list)
    # ADR-0017 E-I: target these seed notes (eval scope only; each within collection_ids)
    chunk_ids: Annotated[list[UUID], Field(max_length=10)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _general_needs_a_topic(self) -> InterviewQuestionRequest:
        if self.source is InterviewSource.general and not (self.topic or "").strip():
            raise ValueError("topic: required for source=general")
        return self


# Problem.type of the 409s for source=notes (clients branch on the TYPE, never the status alone):
# every matching note was already asked ...
INTERVIEW_EXHAUSTED = "urn:ragforge:problem:interview-exhausted"
# ... or no note matches the topic at all (nothing was ever asked). Both are where a client may
# OFFER the general-knowledge fallback; it never switches by itself.
INTERVIEW_NO_NOTES = "urn:ragforge:problem:interview-no-notes"


class InterviewQuestion(_Response):
    question: str  # sanitized CommonMark
    # May differ from the requested kinds when the material does not fit (e.g. no code block).
    kind: Annotated[InterviewKind | str, OPEN]
    topic: str  # what the question is about, for the session summary ("Kafka ISR")
    difficulty: Annotated[InterviewDifficulty | str, OPEN]
    # notes: the notes the question is grounded in; general: always empty (send back to grade)
    chunk_ids: list[UUID]
    # what an answer at `level` must cover (send back to grade). notes: each checked against the
    # notes (may be EMPTY when none passed: the grade then judges against the notes alone);
    # general: the model's own key, unverified.
    expected_points: list[str]
    citations: list[Citation]  # the source pages, labelled 1..n in order (S1, S2, ...)
    # Opaque server-signed binding of this question (chunk ids, expected points, kind,
    # difficulty, level, source, principal). Send it back to grade; optional today, enforced later.
    question_token: str | None = None
    level: Annotated[InterviewLevel | str, OPEN] = InterviewLevel.senior_sde_1
    source: Annotated[InterviewSource | str, OPEN] = InterviewSource.notes


class InterviewGradeRequest(_Request):
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    kind: InterviewKind
    difficulty: InterviewDifficulty = InterviewDifficulty.standard
    # notes: 1..10 ids; general: none (checked below)
    chunk_ids: Annotated[list[UUID], Field(max_length=10)]
    expected_points: Annotated[list[Annotated[str, Field(max_length=500)]], Field(max_length=20)]
    answer: Annotated[str, Field(min_length=1, max_length=8000)]
    question_token: Annotated[str | None, Field(max_length=2000)] = None
    level: InterviewLevel | None = None  # echo the question's level; None: the server default
    source: InterviewSource = InterviewSource.notes

    @model_validator(mode="after")
    def _chunks_match_source(self) -> InterviewGradeRequest:
        if self.source is InterviewSource.notes and not self.chunk_ids:
            raise ValueError("chunk_ids: at least one note is required for source=notes")
        if self.source is InterviewSource.general and self.chunk_ids:
            raise ValueError("chunk_ids: must be empty for source=general")
        # general mode has no notes to re-read: the signed token is the ONLY binding between
        # the server's question/key and this grade, so it is required from day one
        if self.source is InterviewSource.general and not self.question_token:
            raise ValueError("question_token: required for source=general")
        return self


class InterviewGrade(_Response):
    score: Annotated[int, Field(ge=0, le=10)]  # at `level`
    summary: str  # one line: the overall judgement
    right: list[str]  # what the answer got right
    missed: list[str]  # expected points it did not cover
    # notes: statements that contradict the notes; general: contradict the model's key
    misconceptions: list[str]
    # notes: sanitized CommonMark with [S<n>] labels into `citations`; general: no labels
    model_answer: str
    citations: list[Citation]  # always empty for source=general
    # verification of model_answer against the source notes (same rules as chat answers);
    # always "unverified" for source=general (clients show the general-knowledge label)
    verification: Annotated[VerificationStatus | str, OPEN]
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    level: Annotated[InterviewLevel | str, OPEN] = InterviewLevel.senior_sde_1
    source: Annotated[InterviewSource | str, OPEN] = InterviewSource.notes


# ----------------------------------------------------------------- eval surface (ADR-0017 §1)
# Every endpoint here requires the `eval` scope and RAGFORGE_EVAL_API; all reads are RLS-confined
# to the caller's tenant (the eval tenant). Stable identity (ADR-0017 principle 6): datasets never
# store database ids; they store page/chunk KEYS, resolved to ids at run time.
class PageKey(_Response):
    """notion: key = the Notion page id; file: key = metadata.filename (unique per collection)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: Annotated[str, Field(min_length=1, max_length=16)]
    key: Annotated[str, Field(min_length=1, max_length=512)]


class ChunkKey(_Response):
    """`occurrence`: rank among chunks with an equal (page, heading_path, display_sha256), in
    ordinal order (two identical chunks under one heading must not collide)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: PageKey
    heading_path: Annotated[list[Annotated[str, Field(max_length=500)]], Field(max_length=20)]
    display_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    occurrence: Annotated[int, Field(ge=0)] = 0


class ShownSource(_Response):
    """What the model was SHOWN for one answer: ids and digests only, never text."""

    label: PositiveInt  # the S<label> the model saw
    chunk_id: UUID
    document_id: UUID
    display_sha256: str  # hex


class EvalVerifyRequest(_Request):
    answer: Annotated[str, Field(min_length=1, max_length=20000)]
    # ordered: shown_chunk_ids[i] is S<i+1>; each must lie in collection_ids
    shown_chunk_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]


class CoverageUnitKind(StrEnum):
    residue = "residue"
    uncited = "uncited"
    code_unbacked = "code_unbacked"
    invalid_citation = "invalid_citation"  # a label naming nothing shown (ADR-0015 rule 1)


class EvalSpan(_Response):
    text: str  # raw, as in the answer (eval scope: no sanitising, the caller owns the text)
    start: Annotated[int, Field(ge=0)]  # [start, end) in `answer`
    end: Annotated[int, Field(ge=0)]


class EvalCoverageUnit(EvalSpan):
    kind: Annotated[CoverageUnitKind | str, OPEN]


class EvalVerifierClaim(_Response):
    text: str
    # None when the server could not place the claim in the answer (ADR-0015 rule 4a)
    start: int | None = None
    end: int | None = None


class EvalFlaggedClaim(EvalSpan):
    reason: Annotated[ClaimReason | str, OPEN]


class EvalVerifyResponse(_Response):
    """The full ADR-0015 pipeline result PLUS the per-layer evidence before the server merges it
    (ADR-0017 §3 E-V computes caught_by in the client from this)."""

    status: Annotated[VerificationStatus | str, OPEN]
    claims: list[EvalFlaggedClaim]  # the merged flags, as a chat client would see them
    coverage_units: list[EvalCoverageUnit]
    verifier_claims: list[EvalVerifierClaim]
    verifier_ran: bool  # False on a short-circuit (rule 0, a pure refusal, no provenance)
    verifier_verdict: Annotated[VerificationStatus | str | None, OPEN] = None
    verifier_conclusive: bool | None = None
    verifier_model: str | None = None  # what actually ran, not what is configured
    verifier_prompt: str | None = None  # name@version


class EvalChunk(_Response):
    chunk_id: UUID
    document_id: UUID
    page_key: PageKey
    heading_path: list[str]
    ordinal: int
    display_text: str  # raw; the client checks display_sha256 against its run before use
    display_sha256: str  # hex
    # the stable key (occurrence computed server-side, the same rule as /v1/eval/resolve), so a
    # run can record what was shown by KEY and datasets built from runs survive re-chunks
    chunk_key: ChunkKey | None = None


class EvalChunksResponse(_Response):
    chunks: list[EvalChunk]  # requested ids not visible to the caller are simply absent


class EvalResolveRequest(_Request):
    collection_ids: Annotated[list[UUID], Field(min_length=1, max_length=20)]
    page_keys: Annotated[list[PageKey], Field(max_length=500)] = Field(default_factory=list)
    chunk_keys: Annotated[list[ChunkKey], Field(max_length=500)] = Field(default_factory=list)


class ResolveStatus(StrEnum):
    ok = "ok"
    stale = "stale"  # the page exists but no chunk has this (heading_path, display_sha256)
    missing = "missing"  # no such page
    ambiguous = "ambiguous"  # the page key matches several live documents


class ResolvedPage(_Response):
    key: PageKey
    status: Annotated[ResolveStatus | str, OPEN]
    document_id: UUID | None = None


class ResolvedChunk(_Response):
    key: ChunkKey
    status: Annotated[ResolveStatus | str, OPEN]
    chunk_id: UUID | None = None
    # stale only: chunks under the same (page, heading_path), for the owner to re-approve;
    # NEVER applied automatically
    candidates: list[UUID] = Field(default_factory=list)


class EvalResolveResponse(_Response):
    pages: list[ResolvedPage]
    chunks: list[ResolvedChunk]


class EvalSnapshotRequest(_Request):
    """Copy a collection of the CALLER's tenant (admin key; the source tenant comes from the
    session, never from the body) into the caller's paired eval tenant (ADR-0017 §1, r4)."""

    source_collection_id: UUID


class EvalSnapshot(_Response):
    collection_id: UUID  # the snapshot, in the eval tenant
    eval_tenant_id: UUID
    job_id: UUID
    content_digest: str | None = None  # set when the copy job finished
    index_digest: str | None = None


class EvalReembedRequest(_Request):
    space_id: PositiveInt  # the new embedding space to re-embed the snapshot into
