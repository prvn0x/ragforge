"""`ragforge-ask`: key-file lookup and permissions, control-char stripping, turn rendering."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from ragforge_client.ask import (
    INEXACT_HEADING,
    NOT_FOUND_HEADING,
    NOTHING_TO_CHECK,
    ChatApi,
    FlaggedClaims,
    KeyFileError,
    TurnResult,
    clean,
    flag_lines,
    format_footer,
    format_grade,
    format_question,
    interview_client,
    interview_failure,
    nothing_to_check,
    read_key,
    render_turn,
    repl,
    resolve_key_file,
    split_claims,
)
from ragforge_client.client import ApiError, RagforgeClient
from ragforge_client.interview import INTERVIEW_TIMEOUT_S
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    Citation,
    ClaimReason,
    InterviewDifficulty,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    Problem,
    UnsupportedClaim,
)
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    ExtraEvent,
    TokenEvent,
    VerificationEvent,
)


def _key(path: Path, value: str = "rf_secret", mode: int = 0o600) -> Path:
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


def test_resolve_prefers_flag_then_env_then_single_file(tmp_path: Path) -> None:
    only = _key(tmp_path / "eval.key")
    assert resolve_key_file("/x/flag.key", {"RAGFORGE_KEY_FILE": "/x/env"}, tmp_path) == Path(
        "/x/flag.key"
    )
    assert resolve_key_file(None, {"RAGFORGE_KEY_FILE": "/x/env"}, tmp_path) == Path("/x/env")
    assert resolve_key_file(None, {}, tmp_path) == only


def test_resolve_refuses_to_guess(tmp_path: Path) -> None:
    with pytest.raises(KeyFileError, match="no API key"):
        resolve_key_file(None, {}, tmp_path / "missing")
    _key(tmp_path / "a.key")
    _key(tmp_path / "b.key")
    with pytest.raises(KeyFileError, match=r"several keys .*a\.key, b\.key"):
        resolve_key_file(None, {}, tmp_path)


def test_read_key_rejects_shared_or_empty_files_without_leaking_the_key(tmp_path: Path) -> None:
    assert read_key(_key(tmp_path / "ok.key")) == "rf_secret"
    with pytest.raises(KeyFileError, match="chmod 600") as shared:
        read_key(_key(tmp_path / "shared.key", mode=0o644))
    assert "rf_secret" not in str(shared.value)
    with pytest.raises(KeyFileError, match="empty"):
        read_key(_key(tmp_path / "empty.key", value="  "))
    with pytest.raises(KeyFileError, match="cannot read"):
        read_key(tmp_path / "nope.key")


def test_clean_strips_terminal_escapes_but_keeps_layout() -> None:
    hostile = "ok\x1b]52;c;cGF3bmVk\x07 \x1b[2Jdone\x9b31m\r\n\tnext\x00"
    assert clean(hostile) == "ok]52;c;cGF3bmVk [2Jdone31m\n\tnext"


def _ticks(*values: float) -> Iterator[float]:
    return iter(values)


async def _stream(*events: ChatEvent) -> AsyncIterator[ChatEvent]:
    for event in events:
        yield event


def _citation(label: int | None, title: str = "Kafka") -> Citation:
    """A realistic citation: `anchor` is a Notion block id, never the answer's label."""
    return Citation(
        anchor=f"block-{uuid4()}",
        label=label,
        document_id=uuid4(),
        title=title,
        heading_path=["Delivery", "Exactly once"],
        url=None,
        snippet="...",
    )


async def test_render_turn_streams_text_and_collects_footer_data() -> None:
    conversation = uuid4()
    out = io.StringIO()
    clock = _ticks(0.0, 1.5, 4.0)
    result = await render_turn(
        _stream(
            TokenEvent(text="Idempotent \x1b[31mproducer"),
            TokenEvent(text=" [S1]."),
            CitationEvent(citation=_citation(1)),
            CitationEvent(citation=_citation(1)),  # the same label is listed once
            VerificationEvent(status="partial", unsupported_claims=["made up"]),
            DoneEvent(conversation_id=conversation, usage=[], prompt_version="v1"),
        ),
        out,
        now=lambda: next(clock),
    )
    assert out.getvalue() == "Idempotent [31mproducer [S1]."  # flags go to the footer
    assert result.conversation_id == conversation
    assert result.status == "partial"
    assert [c.label for c in result.citations] == [1]
    assert result.first_token_s == 1.5
    assert result.total_s == 4.0
    footer = format_footer(result)
    assert "[S1] Kafka › Delivery › Exactly once" in footer
    assert "partial · first token 1.5s · total 4.0s" in footer
    assert "  Not found in your notes:\n    ! made up\n" in footer  # an old server: no reasons


async def test_render_turn_shows_replacement_and_keeps_conversation_on_error() -> None:
    conversation = uuid4()
    out = io.StringIO()
    result = await render_turn(
        _stream(
            TokenEvent(text="draft"),
            AnswerReplaceEvent(text="final"),
            ErrorEvent(
                problem=Problem(title="Upstream timeout", status=504, detail=None),
                conversation_id=conversation,
            ),
        ),
        out,
    )
    assert out.getvalue().endswith("[answer revised after verification]\nfinal")
    assert result.conversation_id == conversation
    assert result.error == "Upstream timeout"
    assert "error: Upstream timeout" in format_footer(result)
    assert "no verdict" in format_footer(result)


def test_clean_strips_bidi_controls() -> None:
    """Review P2: U+202E in a model answer would show reversed text in the terminal."""
    hostile = "port ‮2909‬ and ⁦x⁩ ‏y؜"
    assert clean(hostile) == "port 2909 and x y"


def test_footer_marks_server_inferred_citations() -> None:
    """Review P1: an inferred page must never look like one the model cited itself."""
    inferred = _citation(2, title="Redis").model_copy(update={"inferred": True})
    result = TurnResult(citations=[_citation(1), inferred], status="verified")
    footer = format_footer(result)
    assert "[S1] Kafka › Delivery › Exactly once\n" in footer
    assert "[S2*] Redis › Delivery › Exactly once (inferred)" in footer
    assert "* matched by the server to an uncited sentence" in footer


def test_footer_never_shows_the_block_anchor_and_falls_back_to_an_index() -> None:
    """Review round 3, P1: the footer printed `anchor` (a Notion block id) next to an answer
    that cites "[S3]"; it must show the answer's own label."""
    cited = _citation(3)
    footer = format_footer(TurnResult(citations=[cited]))
    assert "[S3] Kafka" in footer
    assert cited.anchor not in footer
    unlabelled = format_footer(TurnResult(citations=[_citation(None), _citation(None)]))
    assert "[1] Kafka" in unlabelled
    assert "[2] Kafka" in unlabelled


def test_citation_label_must_be_positive() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _citation(0)


def test_terminal_shows_labels_without_the_servers_markdown_escapes() -> None:
    """Owner report: answers showed "\\[S1\\]" in the terminal."""
    from ragforge_client.ask import TerminalText, unescape_once

    assert unescape_once("Kafka stores records \\[S1\\]. You matched\\!") == (
        "Kafka stores records [S1]. You matched!"
    )
    assert unescape_once("Use `a\\[0\\]` literally \\[S2\\].") == "Use `a\\[0\\]` literally [S2]."
    assert unescape_once("a \\n b") == "a \\n b"  # not punctuation: kept
    reader = TerminalText()  # an escape split across streamed pieces
    pieces = [reader.feed("records \\"), reader.feed("[S1\\"), reader.feed("]."), reader.flush()]
    assert "".join(pieces) == "records [S1]."


async def test_render_turn_unescapes_streamed_tokens() -> None:
    out = io.StringIO()
    clock = _ticks(0.0, 1.0, 2.0)
    await render_turn(
        _stream(
            TokenEvent(text="Idempotent producer \\"),
            TokenEvent(text="[S1\\]."),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        ),
        out,
        now=lambda: next(clock),
    )
    assert out.getvalue() == "Idempotent producer [S1]."


async def test_an_extra_is_printed_after_the_footer_in_a_closed_block() -> None:
    """Phase 2c review, P1: the citations and verdict printed BELOW the extra made it look
    verified. The extra is held back and shown last, closed by an explicit end rule."""
    out = io.StringIO()
    clock = _ticks(0.0, 1.0, 2.0)
    result = await render_turn(
        _stream(
            TokenEvent(text="Kafka keeps messages in partitions \\[S1\\]."),
            CitationEvent(citation=_citation(1)),
            ExtraEvent(kind="incident", text="Common failure pattern: loops\\!"),
            VerificationEvent(status="verified", unsupported_claims=[]),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        ),
        out,
        now=lambda: next(clock),
    )
    assert "Not from your notes" not in out.getvalue()  # nothing printed inline
    footer = format_footer(result)
    lines = footer.splitlines()
    status = next(i for i, ln in enumerate(lines) if ln.strip().startswith("(verified"))
    extra = lines.index("── Not from your notes: a common production failure pattern ──")
    assert status < extra  # the verdict comes BEFORE the extra
    assert lines[extra + 1] == "│ Common failure pattern: loops!"
    assert lines[extra + 2] == "── end (general knowledge, not verified) ──"
    assert footer.rstrip().endswith("not verified) ──")  # nothing after the closed block
    unknown = format_footer(TurnResult(extras=[("diagram", "x")]))
    assert "Not from your notes: general knowledge" in unknown


def test_the_code_offer_is_skipped_once_code_was_shown() -> None:
    cited = [_citation(1)]
    assert 'ask "show the code"' in format_footer(TurnResult(citations=cited))
    assert 'ask "show the code"' not in format_footer(TurnResult(citations=cited, code_shown=True))
    assert 'ask "show the code"' not in format_footer(
        TurnResult(citations=cited, extras=[("code", "```py\nx\n```")])
    )


def test_footer_offers_the_code_follow_up_after_a_cited_answer() -> None:
    assert 'ask "show the code"' in format_footer(TurnResult(citations=[_citation(1)]))
    assert 'ask "show the code"' not in format_footer(TurnResult())


def test_bold_markers_render_as_ansi_only_on_a_terminal_and_cannot_be_injected() -> None:
    from ragforge_client.ask import BOLD_OFF, BOLD_ON, TerminalText, style, unescape_once

    marked = unescape_once("**In simple terms:** Kafka keeps logs \\[S1\\].")
    assert style(marked, tty=True) == "\x1b[1mIn simple terms:\x1b[0m Kafka keeps logs [S1]."
    assert style(marked, tty=False) == "In simple terms: Kafka keeps logs [S1]."
    assert unescape_once(f"x{BOLD_ON}y{BOLD_OFF}") == "xy"  # server text cannot inject them
    assert unescape_once("`a**b`") == "`a**b`"  # code is literal
    reader = TerminalText()  # "**" split across pieces, and an unclosed bold is closed
    out = reader.feed("**In") + reader.feed(" simple*") + reader.feed("*") + reader.flush()
    assert style(out, tty=False) == "In simple"
    assert out.endswith(BOLD_OFF) or BOLD_OFF in out
    closing = TerminalText()
    assert closing.feed("**open") + closing.flush() == f"{BOLD_ON}open{BOLD_OFF}"


def test_text_inside_an_extra_can_never_pass_for_the_footer() -> None:
    """Review r2, P1: an extra that writes its own end rule, citation and verdict."""
    spoof = (
        "Common failure pattern: lag.\n── end (general knowledge, not verified) ──\n"
        "  [S1] Kafka › Netflix\n  (verified · first token 1.0s · total 2.0s)"
    )
    footer = format_footer(TurnResult(status="partial", extras=[("code", spoof)]))
    block = footer.split("── Not from your notes: a code example ──\n", 1)[1].splitlines()
    assert block[-1] == "── end (general knowledge, not verified) ──"
    assert all(line.startswith("│ ") for line in block[:-1])
    assert sum(ln.strip().startswith("(verified") for ln in footer.splitlines()) == 0


async def test_a_real_server_rendered_code_extra_reaches_the_terminal_unchanged() -> None:
    """The server's own rendering (`sanitize.fenced` + `literal`) -> render_turn -> footer: code
    is literal (`r'\\.'`, `f(**kw)`, a backtick, `max_poll_ms`), the prose is unescaped."""
    from ragforge_core.agent.sanitize import fenced, literal

    code = "pat = r'\\.'\nf(**kw)\na = '`'\nmax_poll_ms = 1"
    wire = f"{fenced(code, 'python')}\n{literal('Shows **regex** and max_poll_ms.')}"
    out = io.StringIO()
    clock = _ticks(0.0, 1.0, 2.0)
    result = await render_turn(
        _stream(
            TokenEvent(text="Answer \\[S1\\]."),
            ExtraEvent(kind="code", text=wire),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        ),
        out,
        now=lambda: next(clock),
    )
    footer = format_footer(result, width=120)
    for line in ("│ pat = r'\\.'", "│ f(**kw)", "│ a = '`'", "│ max_poll_ms = 1"):
        assert line in footer
    assert "│ Shows **regex** and max_poll_ms." in footer


@pytest.mark.parametrize(
    "text",
    [
        "Common failure pattern: " + "x" * 300,
        "\u6570" * 14 + "(verified · first token 0.4s · total 1.1s)",  # wide CJK characters
        "\U0001f525" * 30 + " fire",  # emoji are two columns
        "\tcode\twith\ttabs\t" * 8,
    ],
)
def test_extra_lines_wrap_by_display_width_and_keep_the_prefix(text: str) -> None:
    """Final review P2: wrapping must count terminal COLUMNS, not characters."""
    from ragforge_client.ask import _columns

    footer = format_footer(TurnResult(extras=[("incident", text)]), width=42)
    block = footer.split("──\n", 1)[1].splitlines()[:-1]
    assert all(line.startswith("│ ") for line in block)
    assert all(sum(_columns(c) for c in line) <= 42 for line in block)


async def test_a_fence_split_across_tokens_still_counts_as_code_shown() -> None:
    out = io.StringIO()
    clock = _ticks(0.0, 1.0, 2.0)
    result = await render_turn(
        _stream(
            TokenEvent(text="Answer [S1].\n\n`"),
            TokenEvent(text="``java\nx();\n``"),
            TokenEvent(text="`\n"),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        ),
        out,
        now=lambda: next(clock),
    )
    assert result.code_shown


# ------------------------------------------------------------------ flag reasons
def test_flags_split_by_reason_red_first_and_inexact_dim_after() -> None:
    claims = [
        UnsupportedClaim(text="Kafka uses \\*Raft\\*.", reason=ClaimReason.not_found),
        UnsupportedClaim(text="`createMemento()`", reason=ClaimReason.inexact_code),
        UnsupportedClaim(text="A newer reason", reason="contradicted"),  # unknown: not_found
    ]
    flagged = split_claims(claims, [c.text for c in claims])  # index-aligned, as sent today
    assert flagged.not_found == ["Kafka uses \\*Raft\\*.", "A newer reason"]
    assert flagged.inexact == ["`createMemento()`"]
    plain = flag_lines(flagged, tty=False)
    assert plain == [
        f"  {NOT_FOUND_HEADING}",
        "    ! Kafka uses *Raft*.",
        "    ! A newer reason",
        f"  {INEXACT_HEADING}",
        "    ~ `createMemento()`",
    ]
    painted = flag_lines(flagged, tty=True)
    assert painted[0].startswith("\x1b[31m")
    assert painted[3].startswith("\x1b[2m")


def test_an_old_server_without_reasons_shows_every_claim_as_not_found() -> None:
    flagged = split_claims([], ["a", "b"])
    assert (flagged.not_found, flagged.inexact) == (["a", "b"], [])
    assert not split_claims([], [])


def test_a_bare_claim_missing_from_claims_is_never_hidden() -> None:
    """A producer that truncates or de-duplicates `claims` must not mute a warning."""
    claims = [UnsupportedClaim(text="`acks=all`", reason=ClaimReason.inexact_code)]
    flagged = split_claims(claims, ["`acks=all`", "Kafka uses Raft."])
    assert flagged.not_found == ["Kafka uses Raft."]
    assert flagged.inexact == ["`acks=all`"]


def test_claim_text_cannot_inject_terminal_colours() -> None:
    flagged = split_claims([], ["x\x1b[0m\x1b[32mfine"])
    assert flag_lines(flagged, tty=True)[1] == "\x1b[31m    ! x[0m[32mfine\x1b[0m"


async def test_render_turn_keeps_the_reasons_for_the_footer() -> None:
    result = await render_turn(
        _stream(
            TokenEvent(text="Answer [S1]."),
            VerificationEvent(
                status="partial",
                unsupported_claims=["a", "b"],
                claims=[
                    UnsupportedClaim(text="a", reason="not_found"),
                    UnsupportedClaim(text="b", reason="inexact_code"),
                ],
            ),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        ),
        io.StringIO(),
    )
    footer = format_footer(result)
    assert footer.index(NOT_FOUND_HEADING) < footer.index(INEXACT_HEADING)
    assert "    ! a\n" in footer
    assert "    ~ b\n" in footer


# ------------------------------------------------------------------ interview mode in the REPL
def _iv_question(
    topic: str = "Kafka ISR", kind: str = "concept", token: str | None = None
) -> InterviewQuestion:
    return InterviewQuestion(
        question="Why does the leader wait for the ISR\\?",
        kind=kind,
        topic=topic,
        difficulty="standard",
        chunk_ids=[uuid4()],
        expected_points=["durability"],
        citations=[_citation(1, "Kafka replication")],
        question_token=token,
    )


def _iv_grade(score: int = 5) -> InterviewGrade:
    return InterviewGrade(
        score=score,
        summary="Partly right\\.",
        right=["names the ISR"],
        missed=["min.insync.replicas"],
        misconceptions=["acks=1 is durable"],
        model_answer="The leader waits for the ISR \\[S1\\].",
        citations=[_citation(1, "Kafka replication")],
        verification="partial",
        unsupported_claims=[
            UnsupportedClaim(text="It is fast.", reason="not_found"),
            UnsupportedClaim(text="`acks=all`", reason="inexact_code"),
        ],
    )


def _exhausted(topic: str) -> ApiError:
    return ApiError(
        Problem(type=INTERVIEW_EXHAUSTED, title="Interview exhausted", status=409, detail=topic)
    )


@dataclass
class FakeApi:
    """Scripted SDK: each call pops its next reply (an exception is raised). Records requests."""

    questions: list[InterviewQuestion | Exception] = field(default_factory=list)
    grades: list[InterviewGrade | Exception] = field(default_factory=list)
    question_requests: list[InterviewQuestionRequest] = field(default_factory=list)
    grade_requests: list[InterviewGradeRequest] = field(default_factory=list)
    chats: list[str] = field(default_factory=list)

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        self.question_requests.append(request)
        reply = self.questions.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        self.grade_requests.append(request)
        reply = self.grades.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat(
        self, message: str, collection_ids: Sequence[UUID], *, conversation_id: UUID | None = None
    ) -> AsyncIterator[ChatEvent]:
        self.chats.append(message)
        return _stream(
            TokenEvent(text="A Q&A answer."),
            DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v1"),
        )


async def _session(api: ChatApi, *lines: str) -> str:
    """Run the REPL over `lines` (then Ctrl-D) and return everything it printed."""
    pending = list(lines)

    def read(_prompt: str) -> str:
        if not pending:
            raise EOFError
        return pending.pop(0)

    out = io.StringIO()
    await repl(api, [uuid4()], out, read=read)
    return out.getvalue()


async def test_a_phrase_starts_the_interview_grades_the_next_input_and_stop_summarises() -> None:
    api = FakeApi(
        questions=[_iv_question(token="tok"), _iv_question(topic="Kafka acks")],
        grades=[_iv_grade(5)],
    )
    printed = await _session(api, "interview me on Kafka", "the leader waits", "stop")
    assert api.chats == []  # nothing went to Q&A
    assert [r.topic for r in api.question_requests] == ["Kafka", "Kafka"]
    assert api.grade_requests[0].answer == "the leader waits"
    assert api.grade_requests[0].question_token == "tok"
    assert "── Question 1 · concept · Senior SDE-1 · Kafka ISR ──" in printed
    assert "Why does the leader wait for the ISR?" in printed  # unescaped for the terminal
    assert "── Score 5/10 ── Partly right. (the grader's judgement)" in printed
    assert "Missed (checked against your notes):\n    - min.insync.replicas" in printed
    assert "Misconceptions (checked against your notes):\n    x acks=1 is durable" in printed
    assert f"  {NOT_FOUND_HEADING}\n    ! It is fast." in printed
    assert f"  {INEXACT_HEADING}\n    ~ `acks=all`" in printed
    assert "── Question 2 · concept · Senior SDE-1 · Kafka acks ──" in printed
    assert "── Interview summary ──" in printed
    assert "  Senior SDE-1 · Kafka · avg 5.0\n  2 asked · 1 answered · 1 skipped" in printed
    assert "Kafka ISR (concept, 5/10)" in printed
    assert "Kafka replication" in printed


async def test_the_question_card_does_not_reveal_its_sources() -> None:
    api = FakeApi(questions=[_iv_question()])
    printed = await _session(api, "/interview")
    assert "Kafka replication" not in printed  # a page title before answering is a hint


async def test_a_normal_question_suspends_and_answer_grades_it_after_all() -> None:
    hedged = "Is it because the ISR shrinks?"
    api = FakeApi(questions=[_iv_question(), _iv_question()], grades=[_iv_grade(7)])
    printed = await _session(api, "quiz me", hedged, "/answer")
    assert api.chats == [hedged]  # it went to Q&A first ...
    assert "interview paused" in printed
    assert [r.answer for r in api.grade_requests] == [hedged]  # ... then was graded


async def test_normal_q_and_a_is_unchanged_outside_the_mode() -> None:
    api = FakeApi()
    printed = await _session(api, "what is ISR?", "stop")
    assert api.chats == ["what is ISR?", "stop"]  # "stop" is just a message here
    assert api.question_requests == []
    assert "A Q&A answer." in printed


async def test_topics_alternate_and_an_exhausted_one_is_dropped_by_problem_type() -> None:
    api = FakeApi(
        questions=[_iv_question(topic="Kafka"), _exhausted("Redis"), _iv_question(topic="Kafka")],
    )
    printed = await _session(api, "interview me on Kafka, Redis", "skip")
    assert [r.topic for r in api.question_requests] == ["Kafka", "Redis", "Kafka"]
    assert 'All your "Redis" notes are asked. Continuing with Kafka.' in printed
    # offered, never switched: a question is pending, so only "/general Redis" takes it
    assert 'Type "/general Redis" to take it.' in printed
    assert all(r.source == "notes" for r in api.question_requests)


async def test_when_every_topic_is_exhausted_the_fallback_is_offered_and_stop_summarises() -> None:
    api = FakeApi(questions=[_exhausted("Kafka")])
    printed = await _session(api, "interview me on Kafka", "next", "stop")
    assert 'All your "Kafka" notes are asked.' in printed
    assert 'Type "/general" or "y" to take it.' in printed  # nothing pending: "y" is armed
    assert len(api.question_requests) == 1  # "next" asks nothing while only the offer is left
    assert "take a general-knowledge offer" in printed
    assert "── Interview summary ──" in printed
    assert "Fully covered (no new notes left): Kafka" in printed


async def test_a_plain_409_or_404_is_not_exhaustion_and_keeps_the_session() -> None:
    conflict = ApiError(Problem(title="Conflict", status=409))
    missing = ApiError(Problem(title="Not Found", status=404))
    api = FakeApi(questions=[conflict, missing, _iv_question()])
    printed = await _session(api, "interview me on Kafka", "next", "next")
    assert 'The API said 409 Conflict. Say "next" to try again.' in printed
    assert "no interview mode" in printed
    assert "── Question 1 ·" in printed
    assert "── Interview summary" not in printed


async def test_grading_failures_are_friendly_and_keep_the_question() -> None:
    api = FakeApi(
        questions=[_iv_question()],
        grades=[
            ApiError(Problem(title="Too Many Requests", status=429)),
            httpx.ConnectError("refused"),
            _iv_grade(9),
        ],
    )
    api.questions.append(_iv_question())
    printed = await _session(api, "quiz me", "acks=all", "/answer", "/answer acks=all")
    assert "Rate limited right now" in printed
    assert "Cannot reach the API (ConnectError)" in printed
    assert printed.count("Your answer was kept: /answer sends it again") == 2
    assert [r.answer for r in api.grade_requests] == ["acks=all"] * 3
    assert "── Score 9/10" in printed


async def test_harder_fetches_a_harder_question_and_repeat_reprints() -> None:
    api = FakeApi(questions=[_iv_question(), _iv_question(topic="Raft")])
    printed = await _session(api, "quiz me", "repeat", "harder")
    assert printed.count("── Question 1 · concept · Senior SDE-1 · Kafka ISR ──") == 2
    assert api.question_requests[-1].difficulty is InterviewDifficulty.harder
    assert "(difficulty: harder (within Senior SDE-1))" in printed


async def test_an_unknown_kind_or_difficulty_is_shown_as_plain_text() -> None:
    unknown = _iv_question(kind="puzzle").model_copy(update={"difficulty": "expert"})
    api = FakeApi(questions=[unknown])
    printed = await _session(api, "quiz me")
    assert "── Question 1 · puzzle · Senior SDE-1 · expert · Kafka ISR ──" in printed


async def test_a_bare_answer_after_a_failed_grade_of_a_paused_message_regrades_it() -> None:
    """Round-1 review: the hedged message that suspended the session is cleared when graded;
    a failure must still let a bare `/answer` resend it, as the notice promises."""
    hedged = "Is it because the ISR shrinks?"
    api = FakeApi(
        questions=[_iv_question(), _iv_question()],
        grades=[httpx.ReadError("dropped"), _iv_grade(7)],
    )
    printed = await _session(api, "quiz me", hedged, "/answer", "/answer")
    assert [r.answer for r in api.grade_requests] == [hedged, hedged]
    assert "── Score 7/10" in printed


async def test_skipping_forgets_an_unsent_answer() -> None:
    api = FakeApi(
        questions=[_iv_question(), _iv_question(topic="Raft")],
        grades=[httpx.ReadError("dropped")],
    )
    printed = await _session(api, "quiz me", "acks=all", "skip", "/answer")
    assert len(api.grade_requests) == 1  # the bare /answer only re-shows the new question
    assert printed.count("── Question 2 · ") == 2


async def test_a_slow_grade_uses_the_interview_timeout_and_says_wait_not_unreachable() -> None:
    """Fake transport: the interview client sends the 120 s read timeout (not the SDK's 30 s),
    and a read timeout is reported as a slow server, not as "cannot reach the API"."""
    seen: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        if request.url.path.endswith("/question"):
            return httpx.Response(200, content=_iv_question().model_dump_json())
        raise httpx.ReadTimeout("slow grader", request=request)

    async with interview_client(
        "http://api.test", "rf_secret", transport=httpx.MockTransport(handler)
    ) as client:
        api = _ChatOver(client)
        printed = await _session(api, "quiz me", "acks=all")
    assert [t["read"] for t in seen] == [INTERVIEW_TIMEOUT_S, INTERVIEW_TIMEOUT_S]
    assert INTERVIEW_TIMEOUT_S > 30
    assert "took longer than 120 s (it may still be working)" in printed
    assert "Cannot reach" not in printed
    assert "rf_secret" not in printed


def test_a_connect_timeout_is_still_cannot_reach() -> None:
    assert interview_failure(httpx.ConnectTimeout("x")).startswith("Cannot reach the API")


@dataclass
class _ChatOver:
    """The real SDK's interview calls under the REPL (chat is unused in that test)."""

    client: RagforgeClient

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        return await self.client.interview_question(request)

    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        return await self.client.interview_grade(request)

    def chat(
        self, message: str, collection_ids: Sequence[UUID], *, conversation_id: UUID | None = None
    ) -> AsyncIterator[ChatEvent]:
        raise AssertionError("no Q&A in this test")


async def test_a_server_escaped_topic_is_printed_without_backslashes() -> None:
    escaped = "Map\\<K,V> \\& \\[ISR\\]"  # how the server sends "Map<K,V> & [ISR]"
    api = FakeApi(questions=[_iv_question(topic=escaped), _iv_question()], grades=[_iv_grade(2)])
    printed = await _session(api, "quiz me", "no idea", "stop")
    assert "── Question 1 · concept · Senior SDE-1 · Map<K,V> & [ISR] ──" in printed
    assert "    - Map<K,V> & [ISR] (concept, 2/10)" in printed
    assert "\\" not in printed


async def test_the_model_answer_heading_does_not_claim_provenance() -> None:
    unsupported = _iv_grade(5).model_copy(update={"verification": "unsupported"})
    api = FakeApi(questions=[_iv_question(), _iv_question()], grades=[unsupported])
    printed = await _session(api, "quiz me", "acks=all")
    assert "Model answer:\n" in printed
    assert "from your notes):" not in printed
    assert "  (unsupported)" in printed


# ================================================================== Phase 3c (IV3-UI)
def _no_notes(topic: str) -> ApiError:
    return ApiError(Problem(type=INTERVIEW_NO_NOTES, title="No notes", status=409, detail=topic))


def _general_q(topic: str = "Redis eviction", token: str = "gtok") -> InterviewQuestion:  # noqa: S107
    return InterviewQuestion(
        question="How does Redis evict keys\\?",
        kind="concept",
        topic=topic,
        difficulty="standard",
        chunk_ids=[],
        expected_points=["maxmemory policies"],
        citations=[],
        question_token=token,
        level="staff",
        source="general",
    )


def _general_g(score: int = 4) -> InterviewGrade:
    return InterviewGrade(
        score=score,
        summary="Thin\\.",
        right=["names LRU"],
        missed=["LFU"],
        misconceptions=["evicts at random by default"],
        model_answer="Redis evicts by the maxmemory policy.",
        citations=[],
        verification="unverified",
        level="staff",
        source="general",
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (TurnResult(status="unverified"), "(nothing to check: no answer came from your notes ·"),
        (
            TurnResult(status="unverified", citations=[_citation(1, "Kafka")]),
            "(unverified · first token",
        ),
        (TurnResult(status="verified", citations=[_citation(1, "Kafka")]), "(verified ·"),
        # untrusted verifier: an uncited substantive answer the server flagged made claims
        (
            TurnResult(status="unverified", flagged=FlaggedClaims(not_found=["Kafka is Raft."])),
            "(unverified · first token",
        ),
        (TurnResult(status="partial"), "(partial ·"),  # only `unverified` is reworded
        (TurnResult(), "(no verdict ·"),
    ],
)
def test_footer_verdict_says_nothing_to_check_only_for_an_uncited_unverified_turn(
    result: TurnResult, expected: str
) -> None:
    assert expected in format_footer(result, 80)


def test_nothing_to_check_rule() -> None:
    assert nothing_to_check("unverified", cited=False, flagged=False)
    assert not nothing_to_check("unverified", cited=True, flagged=False)
    assert not nothing_to_check("unverified", cited=False, flagged=True)  # claims were made
    assert not nothing_to_check("unsupported", cited=False, flagged=False)
    assert not nothing_to_check(None, cited=False, flagged=False)
    assert NOTHING_TO_CHECK == "Nothing to check: no answer came from your notes."


async def test_a_level_phrase_sets_the_session_level_on_both_requests() -> None:
    api = FakeApi(
        questions=[_iv_question(token="tok").model_copy(update={"level": "staff"}), _iv_question()],
        grades=[_iv_grade(6)],
    )
    printed = await _session(api, "interview me on Kafka at staff level", "acks", "stop")
    assert api.question_requests[0].level == "staff"
    assert api.grade_requests[0].level == "staff"
    assert "level Staff, fixed for this interview" in printed
    assert "── Question 1 · concept · Staff · Kafka ISR ──" in printed
    assert "  Staff · Kafka · avg 6.0" in printed


async def test_another_level_mid_session_summarises_and_starts_a_new_session() -> None:
    api = FakeApi(questions=[_iv_question(), _iv_question()])
    printed = await _session(api, "quiz me on Kafka", "/interview kafka --level sde2")
    assert [r.level for r in api.question_requests] == ["senior_sde_1", "sde_2"]
    assert "Level changed to SDE-2: that starts a new interview" in printed
    assert printed.index("── Interview summary ──") < printed.index("level SDE-2")
    assert printed.count("── Question 1 ·") == 2  # numbering restarts: a new session


async def test_an_unknown_level_is_reported() -> None:
    api = FakeApi(questions=[_iv_question()])
    printed = await _session(api, "/interview kafka --level wizard")
    assert 'Unknown level "wizard": using Senior SDE-1' in printed
    assert api.question_requests[0].topic == "kafka"


async def test_a_natural_language_unknown_level_is_reported_and_not_searched() -> None:
    api = FakeApi(questions=[_iv_question()])
    printed = await _session(api, "interview me on Kafka at principal level please")
    assert 'Unknown level "principal level": using Senior SDE-1' in printed
    assert api.question_requests[0].topic == "Kafka"


@pytest.mark.parametrize(
    ("problem", "notice"),
    [
        (_no_notes("Redis"), 'Your notes have nothing on "Redis".'),
        (_exhausted("Redis"), 'All your "Redis" notes are asked.'),
    ],
)
async def test_the_fallback_is_offered_on_both_409s_and_y_takes_it(
    problem: ApiError, notice: str
) -> None:
    api = FakeApi(questions=[problem, _general_q(), _general_q()], grades=[_general_g(4)])
    printed = await _session(api, "interview me on Redis at staff level", "y", "LRU", "stop")
    assert notice in printed
    assert 'Type "/general" or "y" to take it.' in printed
    first, second = api.question_requests[:2]
    assert first.source == "notes"  # never switched by itself
    assert (second.source, second.topic, second.level) == ("general", "Redis", "staff")
    grade = api.grade_requests[0]
    assert (grade.source, grade.chunk_ids, grade.question_token) == ("general", [], "gtok")
    # the general question card and grade are boxed under the label, never like notes
    assert "── General knowledge · not verified ──\n│ Written by the model" in printed
    assert "│ How does Redis evict keys?" in printed
    assert "── General knowledge · not verified · graded by the model against its own" in printed
    assert "│ Misconceptions (against the model's own key):" in printed
    assert "(unverified)" not in printed  # no verdict line for a general grade
    assert "  General knowledge · not verified: 1 graded, not from your notes" in printed
    assert "consider adding notes on Redis" in printed


async def test_declining_the_offer_never_switches() -> None:
    api = FakeApi(questions=[_no_notes("Redis")])
    printed = await _session(api, "interview me on Redis", "next", "what is ISR?")
    assert len(api.question_requests) == 1  # "next" asked nothing; no general request
    assert api.chats == ["what is ISR?"]
    assert "take a general-knowledge offer" in printed


async def test_no_offer_for_a_mixed_session() -> None:
    api = FakeApi(questions=[_no_notes("")])
    printed = await _session(api, "quiz me", "y")
    assert "Offer:" not in printed
    assert "── Interview summary ──" in printed
    assert api.chats == ["y"]  # the interview ended: "y" is just a message


async def test_y_is_an_answer_while_a_question_is_pending() -> None:
    api = FakeApi(
        questions=[_iv_question(topic="Kafka"), _no_notes("Redis"), _iv_question(), _iv_question()],
        grades=[_iv_grade(5)],
    )
    printed = await _session(api, "interview me on Kafka, Redis", "skip", "y")
    assert 'Type "/general Redis" to take it.' in printed
    assert [r.answer for r in api.grade_requests] == ["y"]
    assert all(r.source == "notes" for r in api.question_requests)


def test_notes_grade_shows_the_answer_key_or_says_there_is_none() -> None:
    question = _iv_question()
    with_key = format_grade(_iv_grade(5), question)
    assert "Answer key (checked against your notes):\n    * durability" in with_key
    empty = question.model_copy(update={"expected_points": []})
    without = format_grade(_iv_grade(5), empty)
    assert "No checked answer key for this question; graded against your notes." in without
    assert "Answer key (checked" not in without


def test_an_uncited_unverified_notes_grade_says_nothing_to_check() -> None:
    bare = _iv_grade(2).model_copy(
        update={"citations": [], "verification": "unverified", "unsupported_claims": []}
    )
    assert "(nothing to check: no answer came from your notes)" in format_grade(bare)
    cited = _iv_grade(2).model_copy(update={"verification": "unverified"})
    assert "  (unverified)" in format_grade(cited)
    flagged = bare.model_copy(
        update={"unsupported_claims": [UnsupportedClaim(text="ISR is 5.", reason="not_found")]}
    )
    printed = format_grade(flagged)
    assert "  (unverified)" in printed
    assert "nothing to check" not in printed


def test_a_notes_question_card_has_no_general_label() -> None:
    card = format_question(_iv_question(), 1)
    assert "General knowledge" not in card
    assert "│" not in card
    general = format_question(_general_q(), 2, width=60)
    assert "── Question 2 · concept · Staff · Redis eviction ──" in general
    assert "── General knowledge · not verified ──" in general


def test_a_grade_for_a_general_question_is_boxed_even_if_labelled_notes() -> None:
    printed = format_grade(_iv_grade(5), _general_q())
    assert "── General knowledge · not verified · graded by the model" in printed
    assert "(partial)" not in printed
    assert "[S1]" not in printed.replace("│", "").split("Model answer")[0]
