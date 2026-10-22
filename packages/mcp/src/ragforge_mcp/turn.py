"""Fold one chat turn's SSE events into an `AskResult` for an AI agent.

An agent reads the whole result at once, so unlike `make ask` nothing is streamed: tokens are
collected, an `answer.replace` overwrites them, and the text is converted once at the end.

Failure rule (DESIGN section 8, "Disconnect"; ADR-0016): an `error` BEFORE the `verification`
verdict means the draft was never committed, so the partial text is discarded and the call fails
(the agent asks again). An `error` AFTER the verdict can only come from the extras step that runs
after `finalize`: the answer is committed and is returned, with a warning.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from uuid import UUID

import anyio
import httpx

from ragforge_client.ask import clean, nothing_to_check, split_claims
from ragforge_contracts.api import Citation
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    ExtraEvent,
    InterruptEvent,
    TokenEvent,
    VerificationEvent,
)
from ragforge_mcp.models import AskResult, CitationOut, ExtraOut, FlaggedClaimOut, NeedsApproval


class TurnFailed(Exception):
    """The turn produced no committed answer. The message is safe to show the agent."""


_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
# Escapes that stay: the ones ADR-0011 invariant 1 relies on to make model text inert. Without an
# unescaped `[` or `<` no link, image, autolink or raw HTML can form, `&` keeps entities literal,
# and a kept `\\` cannot pair with the next char (a literal backslash followed by an escaped `[`
# must not lose a backslash and leave a live `[`). A kept `` \` `` cannot open a code span the
# server never wrote.
_KEEP_ESCAPED = frozenset("\\[]<>&`")
# A citation label "\[S1\]" is unescaped (readability), unless the next char would make it link
# syntax: "[S1](", "[S1][" or the reference definition "[S1]:" (escaped or not, since that char
# may be unescaped itself). A lone "[S1]" is a shortcut reference only if a definition exists,
# and none can: every other "[" stays escaped.
_LABEL = re.compile(r"\\\[S(\d{1,4})\\\](?!\\?[(\[:])")
# CommonMark fences (review r2 P1): an opener of 3+ backticks (no backtick in its info string) or
# 3+ tildes; a closer of the SAME char, at least as long, nothing but spaces after it; no closer
# = the block runs to the end of the text. Indentation is accepted at any depth, not only up to
# 3 spaces: the two possible mistakes are not symmetric. Code read as prose silently loses
# backslashes (a corrupted regex or path); prose read as code only keeps a few cosmetic ones.
_FENCE_OPEN = re.compile(r"[ \t]*(`{3,}(?=[^`\n]*$)|~{3,})")
_BACKTICKS = re.compile(r"`+")


def readable(text: str) -> str:
    """Server CommonMark -> CommonMark an agent can read, still INERT when rendered.

    The server escapes punctuation so model output (which can carry prompt injections from notes)
    never renders as HTML, a link or an image. Those backslashes are noise for an agent
    (`max\\_poll`, `\\[S1\\]`), but a blanket unescape would re-activate exactly what the server
    disarmed (review r1: `\\<img onerror>` became a live tag). So only cosmetic escapes are
    removed: the output differs from the input only by backslashes before characters outside
    `_KEEP_ESCAPED`, plus `[S<n>]` labels that cannot form a link. Fenced blocks and code spans
    are code (backslashes literal) and are kept byte for byte. Nothing is added (no closing `**`
    for an unbalanced one). `clean` drops control and bidi characters."""
    out = [part if is_code else _unescape_prose(part) for part, is_code in split_fences(text)]
    return clean("".join(out))


def split_fences(text: str) -> list[tuple[str, bool]]:
    """`text` cut into (part, is_fenced_code) pieces that join back to `text` exactly.

    Line based, as CommonMark fences are: see `_FENCE_OPEN` for the rules. An unclosed fence
    (a truncated generation; the sanitizer's `finish` does not close it) is code to the end."""
    parts: list[tuple[str, bool]] = []
    lines = text.splitlines(keepends=True)
    prose_start = offset = i = 0
    while i < len(lines):
        opener = _FENCE_OPEN.match(lines[i])
        if opener is None:
            offset += len(lines[i])
            i += 1
            continue
        fence = opener.group(1)
        # A CLOSER is indented 0-3 spaces only (CommonMark): "    ```" inside the code is code,
        # and ending the block there would read the rest of the code as prose (review r3 P1).
        closer = re.compile(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*\r?\n?")
        end = i + 1
        while end < len(lines) and not closer.fullmatch(lines[end]):
            end += 1
        end = min(end + 1, len(lines))  # include the closer line, if there is one
        block = "".join(lines[i:end])
        parts += [(text[prose_start:offset], False), (block, True)]
        offset += len(block)
        prose_start, i = offset, end
    parts.append((text[prose_start:], False))
    return [(part, is_code) for part, is_code in parts if part]


def _unescape_prose(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "`":
            i = _copy_code_span(text, i, out)
        elif char == "\\" and i + 1 < n:
            label = _LABEL.match(text, i)
            if label:
                out.append(f"[S{label.group(1)}]")
                i = label.end()
                continue
            escaped = text[i + 1]
            cosmetic = escaped in _ASCII_PUNCT and escaped not in _KEEP_ESCAPED
            out.append(escaped if cosmetic else text[i : i + 2])
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _copy_code_span(text: str, start: int, out: list[str]) -> int:
    """Copy a code span (a backtick run up to the next run of the same length) verbatim and
    return the index after it; an unclosed run is literal backticks (CommonMark)."""
    opener = _BACKTICKS.match(text, start)
    assert opener is not None  # called on a backtick
    run = len(opener.group(0))
    closer = re.compile(rf"(?<!`)`{{{run}}}(?!`)").search(text, opener.end())
    end = closer.end() if closer else opener.end()
    out.append(text[start:end])
    return end


def document_uri(document_id: UUID) -> str:
    return f"ragforge://documents/{document_id}"


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(
        label=f"S{citation.label}" if citation.label is not None else None,
        title=clean(citation.title),
        heading_path=[clean(h) for h in citation.heading_path],
        url=str(citation.url) if citation.url else None,
        inferred=citation.inferred,
        document_uri=document_uri(citation.document_id),
    )


def approval_url(ui_url: str, run_id: UUID) -> str:
    return f"{ui_url.rstrip('/')}/?run_id={run_id}"


Redact = Callable[[str], str]

# Integrated review P1: the UI cannot load a run raised over MCP (no `GET /v1/approvals/{id}`
# yet), so the text must not tell the agent to wait for an approval that can never come.
NEEDS_APPROVAL_WARNING = (
    "A destructive action needs a human, and cannot be approved from this conversation: this "
    "server has no approve tool, and the UI cannot load this pending run. Tell the user to open "
    "ui_url, ask for the action in the RAGForge UI and approve it there. Your next question in "
    "this conversation cancels this pending action (it would not run from here anyway)."
)


# An unverified turn that cited nothing (e.g. the refusal "Your notes don't cover this.") had no
# notes content to check; the verifier being off is not the reason (same rule as the UI badge).
NOTHING_TO_CHECK_WARNING = "Nothing to check: no answer came from the user's notes."


@dataclass
class _Turn:
    redact: Redact
    conversation_id: UUID | None = None
    tokens: list[str] = field(default_factory=list)
    citations: dict[str, Citation] = field(default_factory=dict)
    verification: str | None = None
    unsupported: list[str] = field(default_factory=list)
    flagged: list[FlaggedClaimOut] = field(default_factory=list)
    extras: list[ExtraOut] = field(default_factory=list)
    interrupt: InterruptEvent | None = None
    error: str | None = None
    done: bool = False

    def apply(self, event: ChatEvent) -> None:
        match event:
            case TokenEvent(text=text):
                self.tokens.append(text)
            case AnswerReplaceEvent(text=text):
                self.tokens = [text]
            case CitationEvent(citation=c):
                # one entry per answer label (the server may repeat a label per anchor)
                key = f"S{c.label}" if c.label is not None else f"{c.document_id}#{c.anchor}"
                self.citations.setdefault(key, c)
            case VerificationEvent(status=status, unsupported_claims=bare, claims=claims):
                self.verification = str(status)
                self.unsupported = [readable(c) for c in bare]
                self.flagged = flagged_out(split_claims(claims, bare).items())
            case ExtraEvent(kind=kind, text=text):
                # ADR-0016: an extra is LITERAL CommonMark (every punctuation escaped, or one
                # fenced code block). Passed on as is: unescaping it would let "**In simple
                # terms:**" render as the answer's own section markup.
                self.extras.append(ExtraOut(kind=clean(str(kind)), text=clean(text)))
            case InterruptEvent(conversation_id=conversation_id):
                self.interrupt = event
                self.conversation_id = conversation_id or self.conversation_id
            case DoneEvent(conversation_id=conversation_id):
                self.conversation_id = conversation_id
                self.done = True
            case ErrorEvent(problem=problem, conversation_id=conversation_id):
                self.conversation_id = conversation_id or self.conversation_id
                # redacted where it is stored: it reaches the agent as a ToolError OR a warning
                message = f"{problem.title}: {problem.detail or ''}".rstrip(": ")
                self.error = self.redact(clean(message))
            case _:  # ToolCallEvent: the agent needs the outcome, not the internal tool call
                pass


async def collect_turn(
    events: AsyncIterable[ChatEvent],
    *,
    ui_url: str,
    redact: Redact,
    conversation_id: UUID | None = None,
    deadline_s: float | None = None,
) -> AskResult:
    """`conversation_id` is the caller's: the server sends it back only on `done` (and on an
    interrupt of a NEW conversation), so it is the fallback for a follow-up that stops early."""
    turn = _Turn(redact=redact, conversation_id=conversation_id)
    timed_out = False
    try:
        with anyio.move_on_after(deadline_s if deadline_s is not None else math.inf) as limit:
            async for event in events:
                turn.apply(event)
        timed_out = limit.cancelled_caught
    except (httpx.TransportError, ValueError):
        # After the verdict the answer is committed: a dropped connection or a malformed frame
        # during the extras must not turn a saved answer into a tool error (reviews P2). Before
        # it, nothing was saved and the error stands.
        if turn.verification is None:
            raise
        turn.error = turn.error or "the stream broke after the answer was verified"
    finally:
        await _close(events)
    if timed_out:
        if turn.verification is None:
            raise TimeoutError  # nothing saved: the caller reports the time limit
        turn.error = turn.error or "the time limit hit after the answer was verified"
    if turn.interrupt is not None:
        return _needs_approval(turn, turn.interrupt, ui_url)
    if turn.error is not None and turn.verification is None:
        retry = f" (conversation_id {turn.conversation_id})" if turn.conversation_id else ""
        raise TurnFailed(
            f"The answer failed before it was verified: {turn.error}. Nothing was saved; "
            f"ask again{retry}."
        )
    if not turn.done and turn.verification is None:
        raise TurnFailed("The answer stream ended early, before any verdict. Ask again.")
    return AskResult(
        status="answered",
        answer=readable("".join(turn.tokens)),
        citations=[citation_out(c) for c in turn.citations.values()],
        verification=turn.verification,
        unsupported_claims=turn.unsupported,
        flagged_claims=turn.flagged,
        extras=turn.extras,
        conversation_id=turn.conversation_id,
        warning=_warning(turn),
    )


def flagged_out(items: list[tuple[str, bool]]) -> list[FlaggedClaimOut]:
    """Reasons for the agent: an older server sends none (every claim `not_found`) and a reason
    this version doesn't know is `not_found` too (`split_claims`), so agents see two values."""
    return [
        FlaggedClaimOut(text=readable(text), reason="inexact_code" if inexact else "not_found")
        for text, inexact in items
    ]


async def _close(events: AsyncIterable[ChatEvent]) -> None:
    """Close the SDK's chat generator (and with it the HTTP response) now, not when the GC gets
    to it. Shielded: after a timeout the surrounding scope is cancelled, and an unshielded await
    would be cancelled before the response is released. The SDK types `chat()` as an
    AsyncIterator, hence the duck-typed lookup. (The SDK's own generators do not yet close their
    inner ones on `aclose`: handoff MCP.md, SDK gap 4.)"""
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        with anyio.CancelScope(shield=True):
            await aclose()


def _needs_approval(turn: _Turn, interrupt: InterruptEvent, ui_url: str) -> AskResult:
    """The answer so far is an unverified draft (the run stopped mid-turn): it is NOT returned."""
    return AskResult(
        status="needs_approval",
        answer="",
        citations=[],
        verification=None,
        unsupported_claims=[],
        extras=[],
        conversation_id=turn.conversation_id,
        needs_approval=NeedsApproval(
            run_id=interrupt.run_id,
            ui_url=approval_url(ui_url, interrupt.run_id),
            pending_tools=[clean(p.tool) for p in interrupt.pending],
        ),
        warning=NEEDS_APPROVAL_WARNING,
    )


def _warning(turn: _Turn) -> str | None:
    """Every applicable warning, joined: a failed later step must not hide a partial verdict."""
    parts = [w for w in (_step_warning(turn), _verdict_warning(turn)) if w]
    return " ".join(parts) or None


def _step_warning(turn: _Turn) -> str | None:
    if turn.error is not None:
        return f"The answer is complete, but a later step failed: {turn.error}."
    return None


def _verdict_warning(turn: _Turn) -> str | None:
    if turn.verification is None:
        return "No verification verdict was sent: treat the answer as unverified."
    if nothing_to_check(
        turn.verification,
        cited=bool(turn.citations),
        flagged=bool(turn.unsupported or turn.flagged),
    ):
        # a refusal / uncited answer: "verifier off" would be false (live failure, 2026-10-19)
        return NOTHING_TO_CHECK_WARNING
    if turn.verification == "unverified":
        return "The answer was not checked against the notes (verifier off for this model)."
    if turn.verification != "verified":
        return (
            f"Verification is {turn.verification}: parts of the answer are not backed by the "
            "user's notes (see unsupported_claims)."
        )
    return None
