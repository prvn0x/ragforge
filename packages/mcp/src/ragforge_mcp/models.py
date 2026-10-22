"""Typed tool outputs. The MCP SDK publishes each model as the tool's `outputSchema` and returns
it as `structuredContent` (plus a JSON text copy for clients that only read text).

Field ORDER is part of the contract: in `AskResult` the extras come after the citations and the
verification verdict (ADR-0016), so an agent reading the JSON top to bottom meets what is from the
notes, and how well it is supported, before anything that is not.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

NOT_FROM_NOTES_LABEL = "Not from your notes: general knowledge, NOT verified against your notes"


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


class CitationOut(_Out):
    """One source the answer cites. `label` matches the "[S<n>]" marker in the answer text."""

    label: str | None = Field(
        description='"S<n>" as cited in the answer text; null when the server sent no label'
    )
    title: str
    heading_path: list[str]
    url: str | None
    inferred: bool = Field(
        description="True: the SERVER matched this source to a sentence the model left uncited "
        "(then checked it); the answer text does not show this label"
    )
    document_uri: str = Field(description="ragforge://documents/{id} resource for this source")


class ExtraOut(_Out):
    """ADR-0016 extra: general knowledge from the model, NOT the user's notes, never verified."""

    not_from_notes: Literal[True] = True
    label: str = NOT_FROM_NOTES_LABEL
    kind: str = Field(description='"incident" (a common failure pattern) or "code"')
    text: str = Field(
        description="LITERAL CommonMark: every punctuation character is backslash-escaped (or "
        "it is one fenced code block), so no emphasis, heading or link can render. Show it as "
        "Markdown, or drop the backslashes for plain text; never restyle it as part of the answer"
    )


class FlaggedClaimOut(_Out):
    """One flagged sentence and WHY it was flagged (owner decision, 2026-10-18)."""

    text: str
    reason: Literal["not_found", "inexact_code"] = Field(
        description='"not_found": the notes do not support it (worth reading); "inexact_code": '
        "code shortened or reformatted from the notes; surrounding lines may be missing, check it"
    )


class NeedsApproval(_Out):
    """A destructive action is waiting for a HUMAN. This server cannot approve it, by design."""

    run_id: UUID
    ui_url: str = Field(
        description="The RAGForge UI. The user asks for the action there and approves it there: "
        "this pending run itself cannot be loaded or approved"
    )
    pending_tools: list[str]


class AskResult(_Out):
    status: Literal["answered", "needs_approval"]
    answer: str = Field(
        description="CommonMark from the user's notes, [S<n>] = citation label. Some characters "
        "(backslash, [ ] < > & and backticks) stay backslash-escaped ON PURPOSE so the text can "
        "never render as a link, image or HTML: keep the escapes if you render it. Code is "
        "verbatim. Empty when needs_approval"
    )
    citations: list[CitationOut]
    verification: str | None = Field(
        description="verified | partial | unsupported | unverified (null: no verdict was sent)"
    )
    unsupported_claims: list[str] = Field(
        description="Sentences of the answer the verifier could NOT support from the notes"
    )
    flagged_claims: list[FlaggedClaimOut] = Field(
        default_factory=list,
        description="The same sentences with WHY each is flagged. Tell the user about "
        "`not_found` ones; `inexact_code`: code shortened from the notes, lines around it "
        "(a guard, a warning, an option) may be missing; mention it more quietly",
    )
    extras: list[ExtraOut] = Field(
        description="General knowledge, NOT from the notes; present it separately and labelled"
    )
    conversation_id: UUID | None = Field(
        description="Pass back as conversation_id to ask a follow-up in the same conversation"
    )
    needs_approval: NeedsApproval | None = None
    warning: str | None = None


class SearchHit(_Out):
    rank: int
    title: str
    heading_path: list[str]
    url: str | None
    snippet: str = Field(description="Raw passage text from the user's notes (not an answer)")
    rerank_score: float | None
    document_uri: str


class SearchResult(_Out):
    query: str
    hits: list[SearchHit]


class CollectionOut(_Out):
    id: UUID
    name: str
    in_scope: bool = Field(description="True: ask and search read this collection")


class ConnectorOut(_Out):
    id: UUID
    kind: str
    display_name: str
    status: str = Field(description="active | paused | error | deleting")
    last_error: str | None
    collection_id: UUID
    collection_name: str | None


class SourcesResult(_Out):
    collections: list[CollectionOut]
    connectors: list[ConnectorOut]
    config_warning: str | None = Field(
        default=None,
        description="Why ask/search would fail with this server's configuration (an unknown "
        "--collection, or no collection at all); null when the configuration is fine",
    )


class SyncStarted(_Out):
    connector_id: UUID
    display_name: str
    job_id: UUID


class SyncSkipped(_Out):
    connector_id: UUID
    display_name: str
    reason: str


class SyncResult(_Out):
    started: list[SyncStarted]
    skipped: list[SyncSkipped]
