"""Server-sent event contract for POST /v1/chat. See docs/DESIGN.md section 8 and ADR-0011.

Wire format: `event: <type>\\ndata: <json>\\n\\n`. The JSON body is the model below.
Clients MUST render `token` text as plain text until a `verification` event arrives.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ragforge_contracts.api import (
    Citation,
    JobEvent,
    Problem,
    ShownSource,
    UnsupportedClaim,
)
from ragforge_contracts.api import (
    VerificationStatus as VerificationStatus,  # re-exported (it used to live here)
)


class _Event(BaseModel):
    # Responses ignore unknown fields so the server can add fields without breaking clients.
    model_config = ConfigDict(extra="ignore", frozen=True)


class TokenEvent(_Event):
    type: Literal["token"] = "token"
    text: str  # already sanitized server-side


class CitationEvent(_Event):
    type: Literal["citation"] = "citation"
    citation: Citation


class ToolCallEvent(_Event):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool: str
    args: dict[str, Any]


class InterruptEvent(_Event):
    """Destructive tools are pending human approval (UI only; MCP keys cannot approve)."""

    type: Literal["interrupt"] = "interrupt"
    run_id: UUID
    conversation_id: UUID | None = None  # set when the turn created a new conversation
    pending: list[ToolCallEvent]


class VerificationEvent(_Event):
    type: Literal["verification"] = "verification"
    # `| str`: a newer server may add statuses; older clients must not crash on them.
    status: Annotated[VerificationStatus | str, Field(union_mode="left_to_right")]
    # Sanitized text of the claims the ground check rejected (clients highlight by matching).
    unsupported_claims: list[str] = Field(default_factory=list)
    # The same claims WITH why each is flagged (additive; older clients use the list above).
    # `not_found`: the verifier or coverage found no support in the notes (worth reading);
    # `inexact_code`: code shortened/reformatted from a cited note (muted, but check it).
    claims: list[UnsupportedClaim] = Field(default_factory=list)


class AnswerReplaceEvent(_Event):
    type: Literal["answer.replace"] = "answer.replace"
    text: str


class ExtraEvent(_Event):
    """ADR-0016: general-knowledge content (a production incident, or code on request) that is
    NOT from the user's notes. Clients must show it apart from the answer, under the label
    "Not from your notes", and never as verified. Additive: older clients skip unknown types."""

    type: Literal["extra"] = "extra"
    kind: Annotated[Literal["incident", "code"] | str, Field(union_mode="left_to_right")]
    # CommonMark, safe to render: literal prose (every ASCII punctuation escaped, URLs defanged)
    # and, for code, one fenced block. Render it as Markdown (shown raw it has backslashes).
    text: str


class Usage(_Event):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    conversation_id: UUID
    usage: list[Usage]
    prompt_version: str
    trace_id: str | None = None
    # ADR-0017: one prompt set per surface ({"chat": "answer@11,...", "verifier": ...}) so an
    # interview prompt change never looks like a chat change. `prompt_version` stays (compat).
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    # ADR-0017: what the model was SHOWN; only with ?trace=shown and the eval scope
    shown: list[ShownSource] | None = None


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    problem: Problem
    conversation_id: UUID | None = None  # set when the turn created a new conversation


ChatEvent = Annotated[
    TokenEvent
    | CitationEvent
    | ToolCallEvent
    | InterruptEvent
    | VerificationEvent
    | AnswerReplaceEvent
    | ExtraEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]

chat_event_adapter: TypeAdapter[ChatEvent] = TypeAdapter(ChatEvent)


def encode_sse(event: ChatEvent) -> bytes:
    """Encode one event in SSE wire format."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n".encode()


KNOWN_EVENT_TYPES = frozenset(
    {
        "token",
        "citation",
        "tool_call",
        "interrupt",
        "verification",
        "answer.replace",
        "extra",
        "done",
        "error",
    }
)


def decode_sse_data(data: str) -> ChatEvent | None:
    """Decode the `data:` payload of one SSE event.

    Returns None for event types this client version does not know (forward compatibility:
    a newer server may add event types). Malformed known events still raise.
    """
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("SSE data must be a JSON object")
    event_type = payload.get("type")
    if event_type not in KNOWN_EVENT_TYPES:
        return None
    return chat_event_adapter.validate_json(data)


# ----------------------------------------------------------------- GET /v1/jobs/{id}/events
def encode_job_event(event: JobEvent) -> bytes:
    """Job progress stream: one `event: job` per state change, data = JobEvent JSON."""
    return f"event: job\ndata: {event.model_dump_json()}\n\n".encode()


def decode_job_event(data: str) -> JobEvent:
    return JobEvent.model_validate_json(data)
