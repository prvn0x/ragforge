"""Domain value objects shared across modules. Immutable; no persistence concerns.

These are the hand-off types between Phase 1 modules:
  C (connectors) produces SourceDocument -> B (rag) produces Chunk/RankedChunk ->
  D (agent) produces Answer -> E (api) streams it through the sanitizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal
from uuid import UUID

SYSTEM_TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")


# ----------------------------------------------------------------- tenancy
@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. tenant_id/user_id always come from here, never from an LLM."""

    tenant_id: UUID
    user_id: UUID
    scopes: frozenset[str]

    def has(self, scope: str) -> bool:
        return scope in self.scopes


class Scope(StrEnum):
    read = "read"
    write = "write"
    chat = "chat"
    approvals_write = "approvals:write"  # never granted to MCP keys (DESIGN section 10)
    admin = "admin"
    eval = "eval"  # ADR-0017: eval tenant's eval key only; never with an MCP key


# ----------------------------------------------------------------- jobs (module A)
class JobPriority(IntEnum):
    """Lower runs first. Strict class order + starvation guard (ADR-0010)."""

    interactive = 0
    webhook = 10
    incremental = 20
    maintenance = 30
    backfill = 40


@dataclass(frozen=True, slots=True)
class JobSpec:
    kind: str
    priority: JobPriority
    payload: dict[str, Any]
    dedupe_key: str  # payload must be a pure function of the dedupe key
    connector_id: UUID | None = None
    run_after: datetime | None = None


@dataclass(frozen=True, slots=True)
class JobLease:
    job_id: UUID
    tenant_id: UUID
    kind: str
    payload: dict[str, Any]
    lease_version: int
    attempts: int


# ----------------------------------------------------------------- sources (module C)
class SourceKind(StrEnum):
    notion = "notion"
    file = "file"


@dataclass(frozen=True, slots=True)
class SourceItemRef:
    source_kind: SourceKind
    external_id: str
    last_edited_time: datetime | None


@dataclass(frozen=True, slots=True)
class NormalizedBlock:
    """One normalized source block. `anchor` is the deep-link target (e.g. Notion block id)."""

    anchor: str
    kind: str  # heading_1..3 | paragraph | list | code | table | quote | callout | equation | media
    text: str  # Markdown for this block
    heading_level: int | None = None
    language: str | None = None  # for code blocks
    never_split: bool = False  # code blocks and table row-groups


@dataclass(frozen=True, slots=True)
class SourceDocument:
    ref: SourceItemRef
    title: str
    external_url: str | None
    blocks: tuple[NormalizedBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    links_to: tuple[str, ...] = ()  # external ids this document links to
    synced_source_ids: tuple[str, ...] = ()
    fetched_at: datetime | None = None  # Notion Date header of first response, minus 60 s


# ----------------------------------------------------------------- chunks (module B)
@dataclass(frozen=True, slots=True)
class Chunk:
    anchor: str
    ordinal: int
    heading_path: tuple[str, ...]
    display_text: str
    embedded_text: str  # "{title} › {heading_path}\n\n{text}"
    display_sha256: bytes  # raw 32-byte digest (DB: bytea, length 32), never hex
    embed_sha256: bytes
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk_id: UUID
    document_id: UUID
    collection_id: UUID
    anchor: str
    title: str
    heading_path: tuple[str, ...]
    display_text: str
    display_sha256: bytes
    external_url: str | None
    fused_score: float  # RRF
    rerank_score: float | None = None


# ----------------------------------------------------------------- answers (module D)
@dataclass(frozen=True, slots=True)
class CitedAnswer:
    text: str
    cited_anchors: tuple[str, ...]


class Verification(StrEnum):
    pending = "pending"
    verified = "verified"
    partial = "partial"
    unsupported = "unsupported"
    unverified = "unverified"


@dataclass(frozen=True, slots=True)
class ChatTurnUsage:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


# ----------------------------------------------------------------- chat model I/O (module D)
class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A complete tool call emitted by the model (arguments fully parsed).

    `id` MUST be globally unique: the ChatModel adapter mints it (uuid4 hex) and never passes
    through provider ids such as "call_0", which can repeat across turns and would collide with
    an old, already-approved row in pending_approvals (unique per conversation + tool_call_id).
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages that requested tools
    tool_call_id: str | None = None  # tool result messages


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    destructive: bool = False  # destructive tools always go through HITL approval


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class UsageReport:
    usage: ChatTurnUsage


ChatStreamEvent = TextDelta | ToolCall | UsageReport


# ----------------------------------------------------------------- index persistence (C -> B seam)
@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """What sync (module C) hands to the index writer (module B) for one source document."""

    id: UUID  # domain.ids.document_id(...)
    collection_id: UUID
    connector_id: UUID | None
    upload_id: UUID | None
    source_kind: SourceKind
    external_id: str
    external_url: str | None
    title: str
    external_updated_at: datetime | None
    fetched_at: datetime  # Notion Date header of first response, minus 60 s (monotonic guard)
    page_sha256: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    links_to: tuple[str, ...] = ()
    synced_source_ids: tuple[str, ...] = ()


class UpsertOutcome(StrEnum):
    inserted = "inserted"
    updated = "updated"  # content changed: chunks/embeddings must be replaced
    unchanged = "unchanged"  # same page_sha256: fetched_at bumped, chunks untouched
    stale = "stale"  # an equal-or-newer fetch already committed: caller does nothing


# ----------------------------------------------------------------- conversations (D <-> E seam)
@dataclass(frozen=True, slots=True)
class PendingApproval:
    id: UUID
    conversation_id: UUID
    run_id: UUID
    tool_call: ToolCall
    status: str  # pending | approved | rejected | expired


class ApprovalChoice(StrEnum):
    approve = "approve"
    reject = "reject"


# Turn events: what the agent runner (D) yields; the API (E) sanitizes text and maps to SSE.
# Ordering contract: TurnSources precedes any TurnText/TurnReplace that may link to those hits
# (D emits it after every retrieval, including tool-driven retrieval mid-turn).
@dataclass(frozen=True, slots=True)
class TurnSources:
    """Retrieved hits that the answer may cite. E feeds their external_urls into the sanitizer's
    link allow-list. NOT shown to clients as citations (that is TurnCitation)."""

    hits: tuple[RankedChunk, ...]


@dataclass(frozen=True, slots=True)
class TurnText:
    text: str  # RAW model text; E passes it through OutputSanitizer before emitting


@dataclass(frozen=True, slots=True)
class TurnCitation:
    hit: RankedChunk
    inferred: bool = False  # server-attributed (verified) rather than cited by the model
    label: int | None = None  # n of the answer's "[S<n>]" for this source (>= 1)

    def __post_init__(self) -> None:
        if self.label is not None and self.label < 1:
            raise ValueError("citation label must be >= 1")


@dataclass(frozen=True, slots=True)
class TurnToolCall:
    call: ToolCall


@dataclass(frozen=True, slots=True)
class TurnInterrupt:
    run_id: UUID
    pending: tuple[ToolCall, ...]


ClaimReasonCode = Literal["not_found", "inexact_code"]


@dataclass(frozen=True, slots=True)
class TurnVerification:
    status: Verification
    # Verbatim RAW claim text the ground check rejected. Text (not offsets) so the seam survives
    # sanitization; E sanitizes each claim like any other text before emitting it.
    unsupported_claims: tuple[str, ...] = ()
    # Why each claim above is flagged, index-aligned ("not_found" | "inexact_code"). Empty when
    # the producer predates reasons; consumers then treat every claim as "not_found".
    claim_reasons: tuple[ClaimReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if self.claim_reasons and len(self.claim_reasons) != len(self.unsupported_claims):
            raise ValueError("claim_reasons must be empty or index-aligned with unsupported_claims")


@dataclass(frozen=True, slots=True)
class TurnReplace:
    text: str  # RAW; sanitized by E


@dataclass(frozen=True, slots=True)
class TurnExtra:
    """ADR-0016: model general knowledge shown APART from the verified answer, labelled "Not from
    your notes". Never verified, never cited, never part of the committed answer."""

    kind: str  # "incident" | "code"
    text: str  # RAW prose (an incident line, or the line after the code); E renders it literally
    code: str = ""  # a code extra's code, verbatim (E fences it)
    language: str = ""  # a code extra's language tag ([A-Za-z0-9_+.#-]{0,20})


@dataclass(frozen=True, slots=True)
class TurnDone:
    conversation_id: UUID
    usage: tuple[ChatTurnUsage, ...]
    prompt_version: str


TurnEvent = (
    TurnSources
    | TurnText
    | TurnCitation
    | TurnToolCall
    | TurnInterrupt
    | TurnVerification
    | TurnReplace
    | TurnExtra
    | TurnDone
)


# ----------------------------------------------------------------- secrets (E -> C seam)
@dataclass(frozen=True, slots=True)
class SealedSecret:
    """Envelope encryption output: ciphertext under a per-secret DEK, DEK wrapped by the KEK."""

    ciphertext: bytes
    wrapped_dek: bytes
