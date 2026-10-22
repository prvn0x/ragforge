"""`make ask`: interactive terminal chat over the SDK.

The API key is read from a file (`--key-file`, `$RAGFORGE_KEY_FILE`, or the single
`~/.config/ragforge/*.key`) and is never printed, logged or echoed. Model text is untrusted
(it can quote Notion content verbatim), so terminal control characters are stripped before
anything reaches the screen: an ESC sequence in a page must not be able to drive the terminal.

Commands inside the prompt: `/new` starts a new conversation, `/quit` (or Ctrl-D) exits.
Interview mode (`ragforge_client.interview`): "interview me [on X]", "quiz me", "ask me interview
questions", `/interview [X]` start it (X may be a list: "Kafka, Redis" alternates them); the
next input is graded, the session commands (repeat, skip, harder, easier, next topic, stop) steer
it, and a normal question PAUSES it and goes to Q&A: `/answer` then grades that very message (it
may have been a hedged answer), `/resume` returns to the question, `/stop` prints the summary.
Unlike the UI, the terminal fetches the next question right after printing a grade (the grade
stays on screen above it; the UI waits for "Next question" so its grade card can be read).

Phase 3c: a level in the phrase ("... at staff level", "... as a senior SDE 2", `--level staff`)
sets the session's fixed level (another level starts a new session, after this one's summary).
When the notes have nothing (left) on a topic, an offer line is printed; `/general [topic]` (or
"y" when no question is pending) takes it. General-knowledge questions and grades are printed in
the extras box under "General knowledge · not verified", never like notes content.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import stat
import sys
import time
import unicodedata
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO
from uuid import UUID

import httpx

from ragforge_client.client import ApiError, RagforgeClient
from ragforge_client.interview import (
    GENERAL_GRADE_LABEL,
    GENERAL_LABEL,
    GENERAL_NOTE,
    INTERVIEW_TIMEOUT_S,
    NO_KEY_NOTE,
    WAITING_NOTICE,
    WEAK_SCORE,
    Do,
    InterviewSession,
    NoQuestion,
    Offer,
    Route,
    Start,
    Summary,
    detect_start,
    fallback_reason,
    general_grade,
    is_general,
    level_name,
    question_line,
    route,
    route_suspended,
    summary_headline,
    unknown_level_notice,
)
from ragforge_contracts.api import (
    Citation,
    ClaimReason,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    UnsupportedClaim,
)
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

DEFAULT_KEY_DIR = Path("~/.config/ragforge")
DEFAULT_URL = "http://localhost:8000"

# C0 controls except \t and \n, DEL, C1 controls (0x80-0x9f, e.g. CSI 0x9b), and the Unicode
# bidirectional controls, which reorder what the terminal shows (U+202E reverses the text).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


class TerminalText:
    """Undo the server's Markdown backslash escapes for a TERMINAL, streamed in pieces.

    The server escapes ASCII punctuation in answer text ("\\[S1\\]") so a Markdown renderer can't
    be tricked into links or formatting. A browser renders that back to "[S1]"; a terminal shows
    the raw backslashes. This is display only: `clean` still strips control characters after it.
    Inside a code span (between backticks) backslashes are literal and kept. A backslash at the
    end of a piece is held until the next piece, since an escape may be split across tokens."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_code = False
        self._bold = False

    def feed(self, text: str) -> str:
        # server text may not carry our own markers (only `**` produces them)
        text = text.replace(BOLD_ON, "").replace(BOLD_OFF, "")
        text, self._pending = self._pending + text, ""
        out: list[str] = []
        i = 0
        while i < len(text):
            char = text[i]
            if char == "`":
                self._in_code = not self._in_code
            elif char == "*" and not self._in_code:
                # "**" toggles bold (answer@9's section names); a lone "*" is kept as is
                if i + 1 == len(text):
                    self._pending = char
                    break
                if text[i + 1] == "*":
                    self._bold = not self._bold
                    out.append(BOLD_ON if self._bold else BOLD_OFF)
                    i += 2
                    continue
            elif char == "\\" and not self._in_code:
                if i + 1 == len(text):
                    self._pending = char
                    break
                if text[i + 1] in _ASCII_PUNCT:
                    out.append(text[i + 1])
                    i += 2
                    continue
            out.append(char)
            i += 1
        return "".join(out)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        if self._bold:  # never leave the terminal bold
            self._bold = False
            text += BOLD_OFF
        return text


# Private-use markers for bold: they survive `clean` (not control characters) and `style` turns
# them into ANSI only on a terminal. Model text cannot inject them: `clean` runs on the server
# text BEFORE `style`, and `style` only ever writes the two fixed SGR sequences.
BOLD_ON, BOLD_OFF = "\ue000", "\ue001"


def style(text: str, *, tty: bool) -> str:
    on, off = ("\x1b[1m", "\x1b[0m") if tty else ("", "")
    return text.replace(BOLD_ON, on).replace(BOLD_OFF, off)


# The server fences a code extra with a run one longer than any inside it (`sanitize.fenced`).
_FENCED = re.compile(r"^(`{3,})[^\n`]*\n.*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)


def unescape_outside_fences(text: str) -> str:
    """`unescape_once` on prose only. A fenced block is code: the server kept it verbatim, and
    backslashes, backticks and `**` inside it are literal (Phase 2c review r3: `f(**kw)` showed
    as `f(kw)` and `r'\\.'` as `r'.'`, silently wrong code to copy)."""
    out: list[str] = []
    last = 0
    for block in _FENCED.finditer(text):
        out += [unescape_once(text[last : block.start()]), block.group(0)]
        last = block.end()
    out.append(unescape_once(text[last:]))
    return "".join(out)


def unescape_once(text: str) -> str:
    """`TerminalText` for a whole text (a replacement answer, a highlighted claim)."""
    reader = TerminalText()
    return reader.feed(text) + reader.flush()


def plain_line(text: str) -> str:
    """Server CommonMark shown as ONE line of plain text (an interview topic): escapes undone,
    bold markers and control characters dropped, whitespace collapsed. The server escapes a
    topic like "Map<K,V> & [ISR]" to `Map\\<K,V> \\& \\[ISR\\]`; printing (or re-escaping) that
    as is shows the backslashes (round-1 review, P2)."""
    return " ".join(style(clean(unescape_once(text)), tty=False).split())


class KeyFileError(Exception):
    pass


def clean(text: str) -> str:
    """Strip terminal and bidi control characters from untrusted text (keeps tabs, newlines)."""
    return _CONTROL.sub("", text)


def resolve_key_file(explicit: str | None, env: dict[str, str], key_dir: Path) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if env.get("RAGFORGE_KEY_FILE"):
        return Path(env["RAGFORGE_KEY_FILE"]).expanduser()
    directory = key_dir.expanduser()
    candidates = sorted(directory.glob("*.key")) if directory.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyFileError(
            f"no API key found: put one in {directory}/<name>.key (mode 600) or pass --key-file"
        )
    names = ", ".join(c.name for c in candidates)
    raise KeyFileError(f"several keys in {directory} ({names}): choose one with --key-file")


def read_key(path: Path) -> str:
    """Return the key; refuses a file other users can read (the key grants tenant access)."""
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise KeyFileError(f"cannot read key file {path}: {exc.strerror}") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KeyFileError(f"key file {path} is readable by others: run chmod 600 {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise KeyFileError(f"key file {path} is empty")
    return key


@dataclass(frozen=True, slots=True)
class FlaggedClaims:
    """Flagged claims split by WHY (owner decision, 2026-10-18): frequent false alarms on code
    the notes do contain trained the owner to ignore every warning, so the two are shown apart."""

    not_found: list[str] = field(default_factory=list)  # no support in the notes: worth reading
    inexact: list[str] = field(default_factory=list)  # only code/format didn't match exactly

    def __bool__(self) -> bool:
        return bool(self.not_found or self.inexact)

    def items(self) -> list[tuple[str, bool]]:
        """(text, inexact) pairs, the claims worth reading first."""
        return [(c, False) for c in self.not_found] + [(c, True) for c in self.inexact]


def split_claims(claims: Sequence[UnsupportedClaim], unsupported: Sequence[str]) -> FlaggedClaims:
    """`claims` (with reasons) when the server sent them; every bare `unsupported` claim WITHOUT
    a reason in `claims` is `not_found`. An older server sends no reasons at all, and a producer
    that truncated or de-duplicated `claims` must not hide a warning: only an explicit
    `inexact_code` mutes a claim. An unknown reason (newer server) is `not_found` too."""
    reasoned = {c.text for c in claims}
    bare = [text for text in unsupported if text not in reasoned]
    return FlaggedClaims(
        not_found=[c.text for c in claims if c.reason != ClaimReason.inexact_code] + bare,
        inexact=[c.text for c in claims if c.reason == ClaimReason.inexact_code],
    )


NOT_FOUND_HEADING = "Not found in your notes:"
# Live failure (2026-10-19): the refusal "Your notes don't cover this." was labelled "Not checked:
# the model in use is below the verifier's quality bar", which is only true of an answer the
# verifier skipped. An unverified answer that cites NOTHING has no notes content to check.
NOTHING_TO_CHECK = "Nothing to check: no answer came from your notes."


def nothing_to_check(status: str | None, *, cited: bool, flagged: bool) -> bool:
    """`unverified`, no citation AND no flagged claim: a pure refusal, with no notes content to
    check, so "not checked" would blame a verifier that had nothing to do.

    `flagged` matters because the server also sends `unverified` with no citations for a
    substantive answer under an untrusted verifier (its inferred citations are dropped and every
    sentence is flagged `not_found`). That answer DID make claims; there "below the verifier's
    quality bar" is the true reason, and "no answer came from your notes" would be false. A pure
    refusal is the one case the server marks with no flagged claims."""
    return status == "unverified" and not cited and not flagged


def verdict_text(status: str | None, *, cited: bool, flagged: bool) -> str:
    """The footer's verdict word(s) for the terminal."""
    if nothing_to_check(status, cited=cited, flagged=flagged):
        return NOTHING_TO_CHECK.rstrip(".").lower()
    return status or "no verdict"


INEXACT_HEADING = "Couldn't match exactly (code shortened from your notes, check it):"


@dataclass
class TurnResult:
    conversation_id: UUID | None = None
    status: str | None = None
    citations: list[Citation] = field(default_factory=list)
    first_token_s: float | None = None
    total_s: float = 0.0
    error: str | None = None
    extras: list[tuple[str, str]] = field(default_factory=list)  # (kind, display text)
    code_shown: bool = False
    flagged: FlaggedClaims = field(default_factory=FlaggedClaims)


async def render_turn(
    events: AsyncIterable[ChatEvent], out: TextIO, *, now: Callable[[], float] = time.monotonic
) -> TurnResult:
    """Stream one turn to `out`; returns what the caller needs for the next turn and the footer."""
    started = now()
    result = TurnResult()
    seen: set[str] = set()  # one footer line per answer label (or per anchor without labels)
    reader = TerminalText()
    streamed: list[str] = []  # the whole answer: a fence may be split across tokens
    tty = bool(getattr(out, "isatty", lambda: False)())

    def show(text: str) -> str:
        return style(clean(text), tty=tty)

    async for event in events:
        if not isinstance(event, TokenEvent):
            out.write(show(reader.flush()))  # a held trailing backslash is just a backslash
        match event:
            case TokenEvent(text=text):
                streamed.append(text)
                if result.first_token_s is None:
                    result.first_token_s = now() - started
                out.write(show(reader.feed(text)))
                out.flush()
            case AnswerReplaceEvent(text=text):
                result.code_shown = "```" in text or "~~~" in text
                # The server rewrote the answer after verification: show the final text whole.
                out.write("\n\n[answer revised after verification]\n" + show(unescape_once(text)))
            case CitationEvent(citation=citation):
                key = f"S{citation.label}" if citation.label is not None else citation.anchor
                if key not in seen:
                    seen.add(key)
                    result.citations.append(citation)
            case ToolCallEvent(tool=tool):
                out.write(f"\n[tool: {clean(tool)}]\n")
            case VerificationEvent(status=status, unsupported_claims=bare, claims=claims):
                # listed in the footer, split by reason (`flag_lines`)
                result.status = str(status)
                result.flagged = split_claims(claims, bare)
            case InterruptEvent(pending=pending):
                tools = ", ".join(clean(p.tool) for p in pending)
                out.write(f"\n[needs approval in the UI: {tools}; skipped here]\n")
            case ExtraEvent(kind=kind, text=text):
                # ADR-0016: held back and printed AFTER the footer, in a closed block, so the
                # citations and the verdict never appear beneath it (Phase 2c review, P1)
                result.extras.append((str(kind), clean(unescape_outside_fences(text))))
            case DoneEvent(conversation_id=conversation_id):
                result.conversation_id = conversation_id
            case ErrorEvent(problem=problem, conversation_id=conversation_id):
                result.conversation_id = conversation_id or result.conversation_id
                result.error = clean(f"{problem.title}: {problem.detail or ''}".rstrip(": "))
    out.write(show(reader.flush()))
    answer = "".join(streamed)
    result.code_shown = result.code_shown or "```" in answer or "~~~" in answer
    result.total_s = now() - started
    return result


def _columns(char: str) -> int:
    """Terminal columns of one character: 2 for wide/full-width (CJK, most emoji), 0 for
    combining marks and format characters, else 1."""
    if unicodedata.combining(char) or unicodedata.category(char) == "Cf":
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _wrap(line: str, width: int) -> list[str]:
    """Hard-wrap at `width` DISPLAY columns (tabs expanded first) so a terminal's soft wrap
    never starts a line without the "│ " prefix (final review: counting characters let 14 CJK
    characters push a fake verdict onto an unprefixed visual line). Characters are kept exactly:
    code wraps visually, it is not reflowed."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for char in line.expandtabs(4):
        cols = _columns(char)
        if used + cols > width and current:
            pieces.append("".join(current))
            current, used = [], 0
        current.append(char)
        used += cols
    pieces.append("".join(current))
    return pieces


def citation_lines(citations: Sequence[Citation]) -> list[str]:
    """The answer text cites "[S3]": the footer shows the same label (`Citation.label`), never a
    renumbering that would make [S3] read as the third footer line, nor `anchor`, which is the
    chunk's source anchor (a Notion block id). A server without labels gets a plain index."""
    lines: list[str] = []
    for i, c in enumerate(citations, start=1):
        tag = f"S{c.label}" if c.label is not None else str(i)
        if c.inferred:
            tag += "*"  # the answer text never shows an inferred label: say so (round 4, P2)
        where = " › ".join([c.title, *c.heading_path])
        url = f"  {c.url}" if c.url else ""
        # The server, not the model, attached this page: it must never look like a model citation.
        how = " (inferred)" if c.inferred else ""
        lines.append(clean(f"  [{tag}] {where}{how}{url}"))
    if any(c.inferred for c in citations):
        lines.append("  * matched by the server to an uncited sentence, then checked")
    return lines


_RED, _DIM, _RESET = "\x1b[31m", "\x1b[2m", "\x1b[0m"


def flag_lines(flagged: FlaggedClaims, *, tty: bool) -> list[str]:
    """`not_found` claims under a red heading; `inexact` ones dim, after them. Off a terminal
    the headings alone tell them apart. Claim text is server text: unescaped for the terminal,
    control characters stripped BEFORE the colour codes (it can't inject its own)."""

    def paint(text: str, colour: str) -> str:
        return f"{colour}{text}{_RESET}" if tty else text

    def claim(text: str) -> str:
        return style(clean(unescape_once(text)), tty=False)

    lines: list[str] = []
    if flagged.not_found:
        lines.append(paint(f"  {NOT_FOUND_HEADING}", _RED))
        lines += [paint(f"    ! {claim(c)}", _RED) for c in flagged.not_found]
    if flagged.inexact:
        lines.append(paint(f"  {INEXACT_HEADING}", _DIM))
        lines += [paint(f"    ~ {claim(c)}", _DIM) for c in flagged.inexact]
    return lines


def format_footer(result: TurnResult, width: int | None = None, *, tty: bool = False) -> str:
    width = _width(width)
    lines = ["", *citation_lines(result.citations), *flag_lines(result.flagged, tty=tty)]
    if result.error:
        lines.append(f"  error: {result.error}")
    elif result.citations and not result.code_shown and all(k != "code" for k, _ in result.extras):
        lines.append('  → ask "show the code" for an example')
    ttft = f"{result.first_token_s:.1f}s" if result.first_token_s is not None else "-"
    verdict = verdict_text(
        result.status, cited=bool(result.citations), flagged=bool(result.flagged)
    )
    lines.append(f"  ({verdict} · first token {ttft} · total {result.total_s:.1f}s)")
    for kind, text in result.extras:
        title = _EXTRA_TITLES.get(kind, "general knowledge")
        lines += ["", *boxed(f"── Not from your notes: {title} ──", text, width)]
    return "\n".join(lines) + "\n"


def boxed(title: str, text: str, width: int, end: str | None = None) -> list[str]:
    """The ADR-0016 extras look: a title rule, every line prefixed "│ ", an end rule. Every
    line is prefixed (and hard-wrapped by display width), so nothing inside can pass for the
    footer, a citation, a verdict or the end rule, whatever the model wrote (review r2, P1).
    `text` is already cleaned display text."""
    body = [
        f"│ {piece}"
        for line in style(text, tty=False).splitlines() or [""]
        for piece in _wrap(line, width)
    ]
    return [title, *body, end or _EXTRA_END]


def _width(width: int | None) -> int:
    return max(20, (width or shutil.get_terminal_size((100, 24)).columns) - 2)


_EXTRA_TITLES = {"incident": "a common production failure pattern", "code": "a code example"}
_EXTRA_END = "── end (general knowledge, not verified) ──"


async def pick_collections(rf: RagforgeClient, wanted: Sequence[str]) -> list[UUID]:
    found = (await rf.list_collections(limit=100)).items
    if not wanted:
        return [c.id for c in found][:20]
    by_name = {c.name: c.id for c in found} | {str(c.id): c.id for c in found}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise SystemExit(f"unknown collection(s): {', '.join(missing)}")
    return [by_name[w] for w in wanted]


class InterviewApi(Protocol):
    """The two SDK calls interview mode needs (`RagforgeClient` has them; tests fake them)."""

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion: ...

    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade: ...


_HINT = "  (your answer · repeat · skip · harder · easier · next topic · stop · /answer <text>)\n"
PAUSED_NOTE = (
    "(interview paused for that question: /answer grades it as your answer, /resume goes back "
    "to the question, /stop ends it with the summary)\n"
)
_GENERAL_END = "── end (general knowledge, not verified) ──"


def _display(text: str, *, tty: bool) -> str:
    """Server CommonMark for the terminal: escapes undone outside fences, controls stripped."""
    return style(clean(unescape_outside_fences(text)), tty=tty)


def _item(text: str) -> str:
    return style(clean(unescape_once(text)), tty=False)


def format_question(
    question: InterviewQuestion, number: int, *, tty: bool = False, width: int | None = None
) -> str:
    """The question card: number, kind, level (plus the difficulty when not standard), topic.
    Its sources are NOT shown: page titles before answering are a hint. A general-knowledge
    question is printed inside the extras box under its label, so it never reads like notes
    content. An unknown kind or difficulty (newer server) is shown as its plain text."""
    topic = plain_line(question.topic)
    kind = clean(str(question.kind))
    head = f"\n── Question {number} · {kind} · {clean(question_line(question))} · {topic} ──"
    if not is_general(question):
        return f"{head}\n{_display(question.question, tty=tty)}\n{_HINT}"
    text = f"{GENERAL_NOTE}\n\n{_display(question.question, tty=False)}"
    box = boxed(f"── {GENERAL_LABEL} ──", text, _width(width), _GENERAL_END)
    return "\n".join([head, *box]) + f"\n{_HINT}"


def _items(title: str, mark: str, values: Sequence[str]) -> list[str]:
    shown = [_item(v) for v in values]
    return [f"  {title}", *(f"    {mark} {v}" for v in shown)] if shown else []


def format_grade(
    grade: InterviewGrade,
    question: InterviewQuestion | None = None,
    *,
    tty: bool = False,
    width: int | None = None,
) -> str:
    """Branches on the SOURCE first (design review): a general-knowledge grade is printed in
    the extras box under its label and never shows a verdict, citations or flags; a notes grade
    shows the score, the three lists, the answer key, then the model answer with its sources,
    verdict and split flags. The summary and "Right" are the grading model's judgement;
    "Missed" and "Misconceptions" are checked against the notes by the server (design review):
    the labels say which is which. `question`, when given, shows its answer key (after the
    answer: no longer a hint)."""
    if general_grade(grade, question):
        return _format_general_grade(grade, question, width=width)
    summary = _item(grade.summary)
    lines = [
        "",
        f"── Score {grade.score}/10 ── {summary} (the grader's judgement)",
        *_items("Right (the grader's judgement):", "+", grade.right),
        *_items("Missed (checked against your notes):", "-", grade.missed),
        *_items("Misconceptions (checked against your notes):", "x", grade.misconceptions),
        *_answer_key(question),
        "",
        # neutral: how well the notes back it is the verdict line below, not the heading
        "Model answer:",
        _display(grade.model_answer, tty=tty),
        *citation_lines(grade.citations),
        *flag_lines(split_claims(grade.unsupported_claims, []), tty=tty),
        f"  ({clean(_grade_verdict(grade))})",
    ]
    return "\n".join(lines) + "\n"


def _grade_verdict(grade: InterviewGrade) -> str:
    return verdict_text(
        str(grade.verification),
        cited=bool(grade.citations),
        flagged=bool(grade.unsupported_claims),
    )


def _answer_key(question: InterviewQuestion | None) -> list[str]:
    """Where the key is shown: its points, or the note that the server kept none (Phase 3c: a
    notes question may come with no checked points; the grade judged against the notes)."""
    if question is None:
        return []
    if not question.expected_points:
        return [f"  {NO_KEY_NOTE}."]
    return _items("Answer key (checked against your notes):", "*", question.expected_points)


def _format_general_grade(
    grade: InterviewGrade, question: InterviewQuestion | None, *, width: int | None
) -> str:
    def block(title: str, mark: str, values: Sequence[str]) -> list[str]:
        return [title, *(f"  {mark} {_item(v)}" for v in values)] if values else []

    key = question.expected_points if question is not None else []
    body = [
        f"Score {grade.score}/10: {_item(grade.summary)} (the grader's judgement)",
        *block("Right:", "+", grade.right),
        *block("Missed (against the model's own key):", "-", grade.missed),
        *block("Misconceptions (against the model's own key):", "x", grade.misconceptions),
        *block("The model's answer key (not verified):", "*", key),
        "",
        "Model answer (general knowledge, not verified):",
        _display(grade.model_answer, tty=False),
    ]
    box = boxed(f"── {GENERAL_GRADE_LABEL} ──", "\n".join(body), _width(width), _GENERAL_END)
    return "\n".join(["", *box]) + "\n"


def format_summary(summary: Summary) -> str:
    """The headline ("Staff · Kafka · avg 6.5"), the counts, then notes-based and general-
    knowledge results APART: weak notes topics list the pages to revisit, weak general topics
    say to add notes on them."""
    lines = [
        "",
        "── Interview summary ──",
        clean(f"  {summary_headline(summary)}"),
        f"  {summary.asked} asked · {summary.answered} answered · {summary.skipped} skipped",
    ]
    if summary.weak:
        lines.append(f"  Weak topics from your notes ({WEAK_SCORE}/10 or less), pages to revisit:")
        for weak in summary.weak:
            lines.append(clean(f"    - {plain_line(weak.topic)} ({weak.kind}, {weak.score}/10)"))
            lines += [clean(f"        {p.title}  {p.url or ''}".rstrip()) for p in weak.pages]
    elif summary.notes_answered:
        lines.append("  No weak topics from your notes: every graded answer scored above 6/10.")
    if summary.general_answered:
        lines.append(f"  {GENERAL_LABEL}: {summary.general_answered} graded, not from your notes")
        lines += [
            clean(f"    - {w.topic} ({w.kind}, {w.score}/10): consider adding notes on {w.topic}")
            for w in summary.weak_general
        ]
    if summary.exhausted:
        lines.append(clean(f"  Fully covered (no new notes left): {', '.join(summary.exhausted)}"))
    return "\n".join(lines) + "\n"


def offer_line(offer: Offer, *, armed: bool) -> str:
    """The terminal's one-step offer (never an automatic switch). "y" works only while `armed`
    (the session waits on its offers: `InterviewSession.offering`), else "/general <topic>"."""
    how = '"/general" or "y"' if armed else f'"/general {offer.topic}"'
    return clean(
        f'  Offer: interview you on "{offer.topic}" from general knowledge instead (not from '
        f"your notes, not verified). Type {how} to take it.\n"
    )


def interview_failure(exc: Exception) -> str:
    """One friendly line for a failed interview call; the session is kept whatever failed.
    The fallback 409s are handled before this (`fallback_reason`, by Problem type, never by
    status).

    A read timeout is NOT "cannot reach": the server got the request and is still working (and
    will charge the interview budget when it finishes), so the advice is to wait, not to restart
    the stack. A connect timeout is a real "cannot reach"."""
    if isinstance(exc, ApiError):
        if exc.status == 429:
            return "Rate limited right now: wait a few seconds, then try again."
        if exc.status == 401:
            return "The API rejected this key: check the key file."
        if exc.status == 404:
            return "This API has no interview mode (it may be older than this client)."
        problem = exc.problem
        detail = f" ({problem.detail})" if problem.detail else ""
        return clean(f"The API said {problem.status} {problem.title}{detail}.")
    if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
        return (
            f"The server took longer than {INTERVIEW_TIMEOUT_S:.0f} s (it may still be working): "
            "wait a moment, then try again."
        )
    return f"Cannot reach the API ({type(exc).__name__}): is the stack running (make up)?"


async def ask_next(
    api: InterviewApi, session: InterviewSession, collections: list[UUID], out: TextIO
) -> bool:
    """Fetch and print the next question. A topic the notes have nothing (left) on is dropped,
    its general-knowledge offer printed, and the next topic tried (`InterviewSession.exhausted`).
    False: nothing is left to ask and nothing to offer (summary printed). While only offers are
    left, nothing is fetched: the offers wait for the owner's "/general"."""
    tty = bool(getattr(out, "isatty", lambda: False)())
    if session.waiting:
        out.write(f"\n  {WAITING_NOTICE}\n")
        return True
    for _ in range(len(session.topics) + 1):  # bounded: each pass drops a topic or returns
        try:
            question = await api.interview_question(session.question_request(collections))
        except ApiError as exc:
            if fallback_reason(exc.problem) is None:
                out.write(f'\n  {interview_failure(exc)} Say "next" to try again.\n')
                return True
            outcome = session.exhausted(exc.problem)
            out.write(clean(f"\n  {outcome.notice}\n"))
            if outcome.offer is not None:
                armed = session.waiting  # nothing will be pending after this: "y" is safe
                out.write(offer_line(outcome.offer, armed=armed))
            if outcome.ended:
                out.write(format_summary(session.summary()))
                return False
            if not session.topics:  # not ended, so offers are open: they wait for the owner
                return True
            continue
        except (httpx.HTTPError, ValueError) as exc:
            out.write(f'\n  {interview_failure(exc)} Say "next" to try again.\n')
            return True
        session.asked(question)
        out.write(format_question(question, session.number, tty=tty))
        return True
    return True  # pragma: no cover - the loop always returns


async def interview_turn(
    api: InterviewApi, session: InterviewSession, step: Route, collections: list[UUID], out: TextIO
) -> Do:
    """Apply one routed message to the session and do what it asks. Returns `Do.end` when the
    interview ended (summary printed), `Do.chat` when it was suspended (the caller then sends
    the message to Q&A) and `Do.restart` when a start phrase named another level: this
    session's summary is printed and the caller begins a new session from the same `Start`."""
    tty = bool(getattr(out, "isatty", lambda: False)())
    action = session.apply(step)
    if action.notice:
        out.write(clean(f"\n({action.notice})\n"))
    match action.do:
        case Do.end | Do.restart:
            out.write(format_summary(session.summary()))
        case Do.chat:
            out.write(PAUSED_NOTE)
        case Do.show if session.current is not None:
            out.write(format_question(session.current, session.number, tty=tty))
        case Do.ask:
            if not await ask_next(api, session, collections, out):
                return Do.end
        case Do.grade:
            if await _grade(api, session, action.text, collections, out, tty=tty):
                return Do.grade if await ask_next(api, session, collections, out) else Do.end
    return action.do


async def _grade(
    api: InterviewApi,
    session: InterviewSession,
    answer: str,
    collections: list[UUID],
    out: TextIO,
    *,
    tty: bool,
) -> bool:
    """Grade and print. False: nothing was graded (the question stays pending, say why)."""
    question = session.current
    try:
        request = session.grade_request(collections, answer)
    except NoQuestion:
        out.write('\n  No question is pending: say "next" for one.\n')
        return False
    except ValueError as exc:
        out.write(f"\n  Not sent: {exc}.\n")
        return False
    where = (
        "with the model's own key" if question and is_general(question) else "against your notes"
    )
    out.write(f"\n(grading {where}…)\n")
    try:
        grade = await api.interview_grade(request)
    except (ApiError, httpx.HTTPError, ValueError) as exc:
        session.grade_failed(answer)
        out.write(f"\n  {interview_failure(exc)} Your answer was kept: ")
        out.write('/answer sends it again, or "skip".\n')
        return False
    session.graded(answer, grade)
    out.write(format_grade(grade, question, tty=tty))
    return True


def begin_interview(start: Start, out: TextIO) -> InterviewSession:
    """A new session from a start phrase, saying its level (and an unknown `--level`)."""
    session = InterviewSession.begin(start)
    notice = unknown_level_notice(start, session.level)
    out.write(
        f"\n(interview mode · level {level_name(session.level)}, fixed for this interview: "
        '"stop" ends it with a summary)\n'
    )
    if notice:
        out.write(clean(f"({notice})\n"))
    return session


class ChatApi(InterviewApi, Protocol):
    """What the REPL needs: chat plus the interview calls (`RagforgeClient`; tests fake it)."""

    def chat(
        self, message: str, collection_ids: Sequence[UUID], *, conversation_id: UUID | None = None
    ) -> AsyncIterator[ChatEvent]: ...


async def repl(
    rf: ChatApi,
    collections: list[UUID],
    out: TextIO,
    *,
    read: Callable[[str], str] = input,
    interview_api: InterviewApi | None = None,
) -> None:
    """The prompt loop. `interview_api` serves the interview calls (default `rf`): `main_async`
    passes a client with the longer `INTERVIEW_TIMEOUT_S`."""
    api: InterviewApi = interview_api or rf
    conversation: UUID | None = None
    interview: InterviewSession | None = None
    while True:
        active = interview is not None and not interview.paused
        try:
            # Blocking on purpose: nothing else runs between turns, and an executor thread
            # stuck in input() would hang interpreter shutdown after Ctrl-C.
            line = read("\nanswer › " if active else "\nyou › ")
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            return
        message = line.strip()
        if not message:
            continue
        if message in ("/quit", "/exit"):
            return
        if message == "/new":
            conversation = None
            out.write("(new conversation)\n")
            continue
        step = _interview_route(interview, message)
        if step is not None:
            if interview is not None:
                done = await interview_turn(api, interview, step, collections, out)
                if done is Do.end:
                    interview = None
                if done is Do.chat:
                    conversation = await _chat_turn(rf, collections, message, conversation, out)
                if done is not Do.restart:
                    continue
            # a start phrase outside the mode, or one naming another level (`Do.restart`)
            assert isinstance(step, Start)  # `_interview_route` starts only on a phrase
            interview = begin_interview(step, out)
            if not await ask_next(api, interview, collections, out):
                interview = None
            continue
        elif interview is not None:
            interview.chatted()
        conversation = await _chat_turn(rf, collections, message, conversation, out)


def _interview_route(interview: InterviewSession | None, message: str) -> Route | None:
    """The interview's reading of a REPL line; None = an ordinary Q&A message."""
    if interview is None:
        return detect_start(message)
    if interview.paused:
        return route_suspended(message)
    return route(message, pending=interview.current is not None, offering=interview.offering)


async def _chat_turn(
    rf: ChatApi,
    collections: list[UUID],
    message: str,
    conversation: UUID | None,
    out: TextIO,
) -> UUID | None:
    """One Q&A turn; returns the conversation to continue."""
    out.write("\nragforge › ")
    try:
        result = await render_turn(rf.chat(message, collections, conversation_id=conversation), out)
    except ApiError as exc:
        out.write(clean(f"\n  error: {exc}\n"))
        return conversation
    tty = bool(getattr(out, "isatty", lambda: False)())
    out.write(format_footer(result, tty=tty))
    return result.conversation_id or conversation


async def main_async(args: argparse.Namespace, out: TextIO) -> int:
    try:
        key = read_key(resolve_key_file(args.key_file, dict(os.environ), DEFAULT_KEY_DIR))
    except KeyFileError as exc:
        print(f"ragforge ask: {exc}", file=sys.stderr)
        return 2
    async with (
        RagforgeClient(args.url, key) as rf,
        interview_client(args.url, key) as interview_api,
    ):
        try:
            collections = await pick_collections(rf, args.collection)
        except ApiError as exc:
            print(f"ragforge ask: {exc}", file=sys.stderr)
            return 1
        if not collections:
            print("ragforge ask: this key sees no collections", file=sys.stderr)
            return 1
        out.write(
            f"RAGForge · {args.url} · {len(collections)} collection(s) · /new, /quit, "
            "/interview [topic]\n"
        )
        await repl(rf, collections, out, interview_api=interview_api)
    return 0


def interview_client(
    url: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> RagforgeClient:
    """The client for the interview calls: the SDK's 30 s read timeout would cut off a healthy
    grade (`INTERVIEW_TIMEOUT_S`); the UI uses the same timeout."""
    return RagforgeClient(url, key, timeout=INTERVIEW_TIMEOUT_S, transport=transport)


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragforge-ask", description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("RAGFORGE_API_URL", DEFAULT_URL))
    parser.add_argument("--key-file", help="file holding the API key (never printed)")
    parser.add_argument(
        "--collection", action="append", default=[], help="name or id; repeatable (default: all)"
    )
    args = parser.parse_args(argv)
    try:
        code = asyncio.run(main_async(args, sys.stdout))
    except KeyboardInterrupt:  # Ctrl-C mid-answer: asyncio cancels the turn, then lands here
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    run()
