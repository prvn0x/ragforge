"""Conversation state, built by folding SSE events into turns. Pure: no Streamlit here.

The page keeps one `Conversation` in Streamlit's session state. Each user message (or each
approval decision) is a `Turn`; `record` applies one chat event to the turn being streamed.
Keeping this a plain reducer is what lets the tests prove the behaviour event by event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from ragforge_contracts.api import ApprovalDecision, Citation, UnsupportedClaim
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    ExtraEvent,
    InterruptEvent,
    TokenEvent,
    ToolCallEvent,
    VerificationEvent,
)
from ragforge_ui.errors import UiError, from_problem

SHOW_THE_CODE = "show the code"


@dataclass(frozen=True, slots=True)
class Extra:
    """ADR-0016 general knowledge. NOT from the notes; kept apart from the answer."""

    kind: str
    text: str  # CommonMark from the server (literal prose; code as one fenced block)


@dataclass(frozen=True, slots=True)
class PendingCall:
    tool_call_id: str
    tool: str
    args: dict[str, Any]


@dataclass(slots=True)
class Interrupt:
    """Destructive tool calls waiting for the owner. Resolved once, then never shown as open."""

    run_id: UUID
    pending: list[PendingCall]
    decisions: dict[str, ApprovalDecision] | None = None
    expired: bool = False  # the owner moved on; the server expires it on the next turn
    # Titles of the documents the calls name (str(UUID) -> title), looked up once so the owner
    # sees WHICH page a call removes, not a bare UUID. None: not looked up yet.
    titles: dict[str, str] | None = None

    @property
    def is_open(self) -> bool:
        return self.decisions is None and not self.expired


@dataclass(slots=True)
class Turn:
    prompt: str
    kind: Literal["question", "resume"] = "question"
    answer: str = ""
    revised: bool = False  # the server replaced the answer after verification
    verdict_seen: bool = False
    status: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    # the same claims with WHY each is flagged (empty from an older server: see `split_claims`)
    claims: list[UnsupportedClaim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    extras: list[Extra] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    interrupt: Interrupt | None = None
    error: UiError | None = None
    first_token_s: float | None = None
    total_s: float | None = None
    events: int = 0  # SSE events received; 0 means the server never accepted the request
    # Set when the stream ended (normally or with an error). A turn left unfinished was cut
    # off: Streamlit stopped the script (a rerun or a closed tab) while it was streaming.
    finished: bool = False

    @property
    def has_code(self) -> bool:
        return (
            "```" in self.answer
            or "~~~" in self.answer
            or any(e.kind == "code" for e in self.extras)
        )


@dataclass(slots=True)
class Conversation:
    id: UUID | None = None
    turns: list[Turn] = field(default_factory=list)


def start_turn(
    conversation: Conversation, prompt: str, kind: Literal["question", "resume"] = "question"
) -> Turn:
    """Append a turn. Expiring an open approval waits for `settle_question`: only a question the
    server actually received expires it."""
    turn = Turn(prompt=prompt, kind=kind)
    conversation.turns.append(turn)
    return turn


def reached_server(turn: Turn) -> bool:
    """Did the server take this request? Yes once its stream produced any event (even an `error`
    event: the request was accepted) or ended cleanly. A request that failed before its first
    event (network error, 401, 409, 503) or was cut off before one did not, as far as the UI can
    tell. The one rule behind both `settle_question` and `settle_resume`."""
    return turn.events > 0 or (turn.finished and turn.error is None)


def settle_question(conversation: Conversation, question: Turn) -> None:
    """Expire the approvals still open once a new question reached the server: the server marks
    those pending calls `expired` when a new turn starts (DESIGN section 8), so the buttons must
    go. A question that never got there leaves them in place: the calls are still pending."""
    if not reached_server(question):
        return
    for turn in conversation.turns:
        if turn is not question and turn.interrupt is not None and turn.interrupt.is_open:
            turn.interrupt.expired = True


def record(conversation: Conversation, turn: Turn, event: ChatEvent, elapsed_s: float) -> None:
    """Apply one chat event to `turn` (and the conversation id, once the server sends it)."""
    turn.events += 1
    match event:
        case TokenEvent(text=text):
            if turn.first_token_s is None:
                turn.first_token_s = elapsed_s
            turn.answer += text
        case AnswerReplaceEvent(text=text):
            turn.answer = text
            turn.revised = True
        case CitationEvent(citation=citation):
            _add_citation(turn, citation)
        case ToolCallEvent(tool=tool):
            turn.tools.append(tool)
        case VerificationEvent(status=status, unsupported_claims=bare, claims=claims):
            turn.verdict_seen = True
            turn.status = str(status)
            turn.unsupported_claims = list(bare)
            turn.claims = list(claims)
        case ExtraEvent(kind=kind, text=text):
            turn.extras.append(Extra(kind=str(kind), text=text))
        case InterruptEvent(run_id=run_id, pending=pending, conversation_id=conversation_id):
            turn.interrupt = Interrupt(
                run_id=run_id,
                pending=[PendingCall(p.tool_call_id, p.tool, dict(p.args)) for p in pending],
            )
            conversation.id = conversation_id or conversation.id
        case DoneEvent(conversation_id=conversation_id):
            conversation.id = conversation_id
        case ErrorEvent(problem=problem, conversation_id=conversation_id):
            turn.error = from_problem(problem)
            conversation.id = conversation_id or conversation.id


def _add_citation(turn: Turn, citation: Citation) -> None:
    """One entry per answer label (the server may send a label again for another sentence);
    without labels (older server) one per anchor."""
    key = _citation_key(citation)
    if all(_citation_key(c) != key for c in turn.citations):
        turn.citations.append(citation)


def _citation_key(citation: Citation) -> str:
    return f"S{citation.label}" if citation.label is not None else citation.anchor


def decide(interrupt: Interrupt, approved: set[str]) -> dict[str, ApprovalDecision]:
    """A decision for EVERY pending call: approve only the ids listed, reject the rest.

    Never defaults to approve: a destructive call the owner didn't explicitly pick is rejected,
    and an id that isn't pending can't sneak into the request."""
    return {
        call.tool_call_id: (
            ApprovalDecision.approve if call.tool_call_id in approved else ApprovalDecision.reject
        )
        for call in interrupt.pending
    }


def open_interrupt(conversation: Conversation) -> Interrupt | None:
    """The interrupt the owner can still act on: the latest turn's, if open.

    Turns the server never took (`reached_server`), questions or resumes, are skipped: the calls
    are still pending on the server, so the owner must still be able to decide them."""
    for turn in reversed(conversation.turns):
        if turn.interrupt is not None:
            return turn.interrupt if turn.interrupt.is_open else None
        if reached_server(turn):
            return None
    return None


def settle_resume(
    interrupt: Interrupt, decisions: dict[str, ApprovalDecision], resume_turn: Turn
) -> None:
    """Mark the interrupt decided once the server has taken the decisions (`reached_server`).
    A resume that failed before its first event leaves the interrupt open, so Approve/Reject
    stay on screen for a retry."""
    if reached_server(resume_turn):
        interrupt.decisions = decisions


def resume_prompt(interrupt: Interrupt, decisions: dict[str, ApprovalDecision]) -> str:
    """What the resume turn shows in place of a user message."""
    names = {call.tool_call_id: call.tool for call in interrupt.pending}
    approved = [names[i] for i, d in decisions.items() if d is ApprovalDecision.approve]
    rejected = [names[i] for i, d in decisions.items() if d is ApprovalDecision.reject]
    parts = []
    if approved:
        parts.append("Approved: " + ", ".join(approved))
    if rejected:
        parts.append("Rejected: " + ", ".join(rejected))
    return " · ".join(parts) or "No decision"


def offers_code(turn: Turn) -> bool:
    """Offer "Show the code" after a finished, cited answer that has no code yet (ADR-0016:
    the offer is the client's, not model text, so it adds nothing unverified to the answer)."""
    return (
        turn.finished
        and turn.error is None
        and turn.interrupt is None
        and bool(turn.citations)
        and not turn.has_code
    )


# ------------------------------------------------------------------ approval links
def foreign_run(conversation: Conversation, raw_run_id: str | None) -> UUID | None:
    """The run named by a `?run_id=` link when this session does NOT hold it, else None.

    MCP clients send the owner to `<ui>/?run_id=<uuid>` (they cannot approve themselves), and
    the UI puts its own open run in the URL so a reload lands there too. The API has no read
    endpoint for pending calls yet, so such a run can't be shown or decided here; the page says
    so instead of silently ignoring the link. A malformed id is ignored (None)."""
    if not raw_run_id:
        return None
    try:
        run_id = UUID(raw_run_id)
    except ValueError:
        return None
    held = {t.interrupt.run_id for t in conversation.turns if t.interrupt is not None}
    return None if run_id in held else run_id


# ------------------------------------------------------------------ document titles
DOCUMENT_ARG = "document_id"


def document_id_arg(args: dict[str, Any]) -> str | None:
    """The canonical (lower-case, hyphenated) document id a tool call names, if any."""
    value = args.get(DOCUMENT_ARG)
    if not isinstance(value, str):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def cited_titles(conversation: Conversation) -> dict[str, str]:
    """Titles the conversation has already seen in citations: free, no API call."""
    return {str(c.document_id): c.title for turn in conversation.turns for c in turn.citations}


def missing_titles(interrupt: Interrupt, known: dict[str, str]) -> bool:
    """True if a pending call names a document whose title isn't in `known`."""
    wanted = {d for call in interrupt.pending if (d := document_id_arg(call.args)) is not None}
    return bool(wanted - known.keys())


# ------------------------------------------------------------------ sync jobs
_IN_FLIGHT = frozenset({"queued", "running"})


def in_flight(job_status: str) -> bool:
    """A job that is still queued or running. An unknown (newer) status counts as finished, so a
    status this UI doesn't know can never lock "Sync now" for good."""
    return job_status in _IN_FLIGHT
