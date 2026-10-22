"""`ask`: answer shape for an agent, extras kept apart, interrupts, and mid-stream failures."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from mcp_support import (
    API,
    COLL_A,
    KEY,
    UI,
    BrokenStream,
    cite,
    connected,
    make_service,
    mock_api,
    sse,
    structured,
    text_of,
)

from ragforge_contracts.api import Problem, UnsupportedClaim
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
    encode_sse,
)
from ragforge_core.agent.sanitize import MarkdownStreamSanitizer, literal
from ragforge_mcp.errors import Redactor
from ragforge_mcp.models import NOT_FROM_NOTES_LABEL
from ragforge_mcp.service import RagforgeService
from ragforge_mcp.turn import collect_turn, readable, split_fences

CONV = UUID("00000000-0000-4000-8000-00000000c0c0")
DONE = DoneEvent(conversation_id=CONV, usage=[], prompt_version="answer@11")
CODE = "```python\nconsumer.poll(timeout_ms=100)  # f(**kw), r'\\.'\n```"


def test_readable_undoes_cosmetic_escapes_but_keeps_code_bold_and_strips_controls() -> None:
    text = "**In simple terms:** a rebalance \\[S1\\] uses max\\_poll\\_interval\\_ms.\u202e\x1b[2J"
    assert readable(text) == "**In simple terms:** a rebalance [S1] uses max_poll_interval_ms.[2J"
    # inline code and fenced code keep their backslashes and `**` byte for byte
    assert readable("see `a\\_b` and\n" + CODE + "\n") == "see `a\\_b` and\n" + CODE + "\n"


def _served(model_text: str) -> str:
    """What the API really sends for this model output (the ADR-0011 sanitizer, in one feed)."""
    sanitizer = MarkdownStreamSanitizer()
    return sanitizer.feed(model_text) + sanitizer.finish()


def _live(markdown: str, needle: str) -> bool:
    """`needle` occurs in `markdown` outside code spans with its first char NOT escaped (an even
    number of backslashes before it), i.e. a renderer would parse it as syntax."""
    prose = re.sub(r"`[^`\n]*`", "", markdown)
    for match in re.finditer(re.escape(needle), prose):
        backslashes = len(prose[: match.start()]) - len(prose[: match.start()].rstrip("\\"))
        if backslashes % 2 == 0:
            return True
    return False


@pytest.mark.parametrize(
    ("model_text", "needle"),
    [
        ("Hi <img src=x onerror=alert(document.cookie)>", "<img"),
        ('Go <a href="javascript:fetch(1)">x</a>', "<a "),
        ("pre-escaped \\[y\\](javascript:alert(1))", "[y]"),
        ("an entity &lt;script&gt; stays text", "&lt;"),
        ("a citation glued to a link \\[S1\\](javascript:x)", "[S1]("),
        ("a citation as a reference definition\n\\[S1\\]: http://evil.example", "[S1]:"),
        ("a backslash then a bracket \\\\[z](/x)", "[z]"),
        ("an image ![i](/x.png) and a link [t](/y)", "]("),
    ],
)
def test_readable_never_reactivates_what_the_server_made_inert(
    model_text: str, needle: str
) -> None:
    """Review r1 P1: a blanket unescape turned the sanitizer's inert text back into live HTML
    and links. Feed the REAL sanitizer's output and check nothing live comes out."""
    served = _served(model_text)
    assert not _live(served, needle)  # precondition: the server made it inert
    out = readable(served)
    assert not _live(out, needle), out
    for opener in "[<&":  # no link, HTML or entity opener became live anywhere
        assert not _live(out.replace("[S1]", "").replace("[S2]", ""), opener), out


REGEX_LINE = "re.compile(r'a\\.b\\_c')  # f(**kw)"


@pytest.mark.parametrize(
    ("block", "code"),
    [
        (f"```python\n{REGEX_LINE}\n```", f"```python\n{REGEX_LINE}\n```"),
        (f"~~~python\n{REGEX_LINE}\n~~~", f"~~~python\n{REGEX_LINE}\n~~~"),
        (f"```\n{REGEX_LINE}\n````", f"```\n{REGEX_LINE}\n````"),  # longer closer
        (f"````\n```\n{REGEX_LINE}\n````", f"````\n```\n{REGEX_LINE}\n````"),  # short inner
        (f"~~~\n```\n{REGEX_LINE}\n~~~", f"~~~\n```\n{REGEX_LINE}\n~~~"),  # other char
        (f"```python\n{REGEX_LINE}\npath = 'C:\\\\x_y'", None),  # unclosed: code to the end
        # review r3 P1: a 4-space-indented ``` inside the code is code, not the closer
        (
            f'```python\ndoc = """\n    ```\n"""\n{REGEX_LINE}\n```',
            f'```python\ndoc = """\n    ```\n"""\n{REGEX_LINE}\n```',
        ),
    ],
)
def test_readable_keeps_every_commonmark_fence_verbatim(block: str, code: str | None) -> None:
    """Review r2 P1: only a backtick fence closed by an equal run was recognised; a `~~~` fence,
    a longer closer or an unclosed fence was unescaped as prose and `\\.` lost its backslash."""
    served = _served(f"Use this:\n{block}\nthen max\\_poll.")
    fenced = served[served.index(block[:3]) :]
    out = readable(served)
    assert REGEX_LINE in served  # precondition: the server passes the code through as is
    if code is not None:
        assert code in out, out
        assert out.endswith("then max_poll."), out  # prose after the block is still cleaned
    else:
        assert out.endswith(fenced), out  # everything from the opener on is code


def test_split_fences_rejoins_exactly_and_keeps_inline_triple_backticks_as_prose() -> None:
    text = "a ```not a fence``` b\n```\nx\n```\nc\n~~~\ny"
    parts = split_fences(text)
    assert "".join(p for p, _ in parts) == text
    assert [is_code for _, is_code in parts] == [False, True, False, True]
    assert parts[1][0] == "```\nx\n```\n"


def test_readable_keeps_labels_and_identifiers_readable_and_adds_nothing() -> None:
    served = _served("A rebalance [S1] uses max_poll_ms (see [S2]).")
    assert readable(served) == "A rebalance [S1] uses max_poll_ms (see [S2])."
    assert readable("a ** b") == "a ** b"  # an unbalanced ** gets no closing ** appended
    assert readable("``a\\_`b`` and \\_") == "``a\\_`b`` and _"  # double-backtick span


async def test_extras_stay_literal_even_with_answer_like_markup() -> None:
    """Review r1 P1: an extra must never render as the answer's own "**In simple terms:**"
    section, a heading or a link."""
    served = literal("**In simple terms:** see [docs](https://x.com) # heading <b>x</b>")
    events: list[ChatEvent] = [
        TokenEvent(text="ok"),
        VerificationEvent(status="verified"),
        ExtraEvent(kind="incident", text=served),
        DONE,
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            out = structured(await client.call_tool("ask", {"question": "q"}))
    text = out["extras"][0]["text"]
    assert text == served
    for markup in ("**", "[docs]", "](", "<b>", " # "):
        assert markup not in text


async def test_ask_returns_the_answer_citations_verdict_then_labelled_extras() -> None:
    events: list[ChatEvent] = [
        TokenEvent(text="**In simple terms:** a consumer is kicked \\[S1\\]"),
        TokenEvent(text=" when max\\_poll is exceeded \\[S2\\]."),
        CitationEvent(citation=cite(1)),
        CitationEvent(citation=cite(1)),  # same label twice (two anchors): listed once
        CitationEvent(citation=cite(2, "Redis", inferred=True)),
        ToolCallEvent(tool_call_id="t1", tool="search_notes", args={}),
        VerificationEvent(status="verified"),
        ExtraEvent(kind="incident", text="Common failure pattern: a slow handler\\! loops\\."),
        ExtraEvent(kind="code", text=CODE),
        DONE,
    ]
    with mock_api() as api:
        chat = api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            out = structured(await client.call_tool("ask", {"question": "what is a rebalance?"}))
    assert out["status"] == "answered"
    assert out["answer"] == (
        "**In simple terms:** a consumer is kicked [S1] when max_poll is exceeded [S2]."
    )
    assert [(c["label"], c["title"], c["inferred"]) for c in out["citations"]] == [
        ("S1", "Kafka", False),
        ("S2", "Redis", True),
    ]
    assert out["citations"][0]["heading_path"] == ["Consumers", "Rebalance"]
    assert out["citations"][0]["document_uri"].startswith("ragforge://documents/")
    assert out["verification"] == "verified"
    assert out["warning"] is None
    assert out["conversation_id"] == str(CONV)
    # extras: marked, labelled, never inside the answer, and AFTER citations and the verdict
    assert [e["kind"] for e in out["extras"]] == ["incident", "code"]
    assert all(e["not_from_notes"] is True for e in out["extras"])
    assert all(e["label"] == NOT_FROM_NOTES_LABEL for e in out["extras"])
    # literal CommonMark passed on as is (ADR-0016): never unescaped into live formatting
    assert out["extras"][0]["text"] == "Common failure pattern: a slow handler\\! loops\\."
    assert out["extras"][1]["text"] == CODE  # code copied verbatim
    assert "failure pattern" not in out["answer"]
    keys = list(out)
    assert keys.index("citations") < keys.index("verification") < keys.index("extras")
    request = json.loads(chat.calls.last.request.content)
    assert request == {
        "conversation_id": None,
        "collection_ids": [str(COLL_A)],
        "message": "what is a rebalance?",
    }


async def test_follow_up_sends_the_conversation_id() -> None:
    with mock_api() as api:
        chat = api.post("/v1/chat").mock(
            return_value=sse([TokenEvent(text="ok"), VerificationEvent(status="verified"), DONE])
        )
        async with connected() as client:
            args = {"question": "and then?", "conversation_id": str(CONV)}
            structured(await client.call_tool("ask", args))
    assert json.loads(chat.calls.last.request.content)["conversation_id"] == str(CONV)


async def test_replacement_answer_wins_and_partial_verdicts_warn() -> None:
    events: list[ChatEvent] = [
        TokenEvent(text="first draft \\[S9\\]"),
        VerificationEvent(status="partial", unsupported_claims=["It uses \\*magic\\*."]),
        AnswerReplaceEvent(text="final answer \\[S1\\]"),
        DONE,
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            out = structured(await client.call_tool("ask", {"question": "q"}))
    assert out["answer"] == "final answer [S1]"
    assert out["verification"] == "partial"
    assert out["unsupported_claims"] == ["It uses *magic*."]
    assert "not backed" in out["warning"]


async def test_interrupt_returns_needs_approval_and_never_approves() -> None:
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={"q": "x"})]
    events: list[ChatEvent] = [
        TokenEvent(text="I will forget that \\[S1\\]"),
        InterruptEvent(run_id=run_id, conversation_id=CONV, pending=pending),
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        approvals = api.post(url__regex=r"/v1/approvals/.*").mock(return_value=httpx.Response(200))
        async with connected() as client:
            out = structured(await client.call_tool("ask", {"question": "forget my address"}))
            tools = {t.name for t in (await client.list_tools()).tools}
    assert out["status"] == "needs_approval"
    assert out["answer"] == ""  # the unverified draft is not passed on
    assert out["needs_approval"] == {
        "run_id": str(run_id),
        "ui_url": f"{UI}/?run_id={run_id}",
        "pending_tools": ["forget_fact"],
    }
    assert out["conversation_id"] == str(CONV)
    assert "cancels this pending action" in out["warning"]
    # integrated review P1: never promise an approval the UI cannot load
    assert "cannot be approved from this conversation" in out["warning"]
    assert not approvals.called
    assert tools == {"ask", "search", "list_sources", "sync_now"}
    assert not any("approv" in name or "resume" in name for name in tools)


async def test_interrupt_in_a_follow_up_keeps_the_callers_conversation_id() -> None:
    """The server sets InterruptEvent.conversation_id only for a NEW conversation."""
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    events: list[ChatEvent] = [InterruptEvent(run_id=run_id, conversation_id=None, pending=pending)]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            args = {"question": "forget it", "conversation_id": str(CONV)}
            out = structured(await client.call_tool("ask", args))
    assert out["status"] == "needs_approval"
    assert out["conversation_id"] == str(CONV)


async def test_error_before_the_verdict_fails_and_drops_the_draft() -> None:
    events: list[ChatEvent] = [
        TokenEvent(text="a half-written draft"),
        ErrorEvent(problem=Problem(title="Model unavailable", status=503), conversation_id=CONV),
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "q"})
    assert result.is_error
    message = text_of(result)
    assert "Model unavailable" in message
    assert str(CONV) in message
    assert "half-written" not in message


async def test_error_after_the_verdict_keeps_the_committed_answer() -> None:
    events: list[ChatEvent] = [
        TokenEvent(text="kept answer"),
        VerificationEvent(status="verified"),
        ErrorEvent(problem=Problem(title="Lease lost", status=409), conversation_id=CONV),
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            out = structured(await client.call_tool("ask", {"question": "q"}))
    assert out["answer"] == "kept answer"
    assert "Lease lost" in out["warning"]
    assert out["conversation_id"] == str(CONV)


async def test_stream_that_ends_without_a_verdict_is_an_error() -> None:
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse([TokenEvent(text="cut off")]))
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "q"})
    assert result.is_error
    assert "ended early" in text_of(result)


async def test_connection_dropped_mid_answer_is_a_clear_error() -> None:
    head = encode_sse(TokenEvent(text="partial"))
    response = httpx.Response(
        200, stream=BrokenStream(head), headers={"content-type": "text/event-stream"}
    )
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=response)
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "q"})
    assert result.is_error
    assert f"connection to the RAGForge API at {API}" in text_of(result)
    assert "ReadError" in text_of(result)


async def test_unverified_and_missing_verdicts_are_flagged() -> None:
    with mock_api() as api:
        api.post("/v1/chat").mock(
            side_effect=[
                # cited: the verifier really skipped notes content (an uncited one says
                # "nothing to check": test_iv3_mcp_verdict.py)
                sse(
                    [
                        TokenEvent(text="a"),
                        CitationEvent(citation=cite(1)),
                        VerificationEvent(status="unverified"),
                        DONE,
                    ]
                ),
                sse([TokenEvent(text="b"), DONE]),
            ]
        )
        async with connected() as client:
            unverified = structured(await client.call_tool("ask", {"question": "q"}))
            no_verdict = structured(await client.call_tool("ask", {"question": "q"}))
    assert "not checked" in unverified["warning"]
    assert no_verdict["verification"] is None
    assert "unverified" in no_verdict["warning"]


async def test_ask_rejects_an_empty_question_without_calling_the_api() -> None:
    with mock_api() as api:
        chat = api.post("/v1/chat")
        async with connected() as client:
            result = await client.call_tool("ask", {"question": ""})
    assert result.is_error
    assert not chat.called


class StallingStream(httpx.AsyncByteStream):
    """Headers and one token arrive, then the server goes silent; records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield encode_sse(TokenEvent(text="partial"))
        await anyio.sleep(30)

    async def aclose(self) -> None:
        self.closed = True


async def test_a_stalled_stream_times_out_and_the_response_is_closed() -> None:
    stream = StallingStream()
    response = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    service, rf = make_service()
    service = RagforgeService(rf, api_url=API, ui_url=UI, redact=service._redact, ask_timeout_s=0.2)
    with mock_api() as api, anyio.fail_after(5):
        api.post("/v1/chat").mock(return_value=response)
        async with rf:
            with pytest.raises(Exception, match=r"longer than 0\.2 s") as err:
                await service.ask("q")
            assert stream.closed
    assert type(err.value).__name__ == "ToolError"
    assert KEY not in str(err.value)


async def test_collect_turn_redacts_a_post_verdict_error_where_it_is_stored() -> None:
    """The warning is redacted by `collect_turn` itself, not only by the final output pass."""

    async def events() -> AsyncIterator[ChatEvent]:
        yield TokenEvent(text="ok")
        yield VerificationEvent(status="verified")
        yield ErrorEvent(problem=Problem(title=f"echo {KEY}", status=500))

    result = await collect_turn(events(), ui_url=UI, redact=Redactor(KEY))
    assert result.warning is not None
    assert KEY not in result.warning
    assert "[redacted]" in result.warning


async def test_the_event_generator_is_closed_when_the_consumer_side_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure outside the generator (here: a bug while folding an event) leaves it suspended
    at a `yield`; `collect_turn` closes it at once instead of leaving it to the GC. (The SDK's
    `chat()` does not yet close ITS inner generators on aclose: handoff MCP.md, SDK gap 4.)"""
    closed: list[bool] = []

    async def events() -> AsyncIterator[ChatEvent]:
        try:
            yield TokenEvent(text="x")
            await anyio.sleep(30)
        finally:
            closed.append(True)

    def broken_apply(self: object, event: ChatEvent) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr("ragforge_mcp.turn._Turn.apply", broken_apply)
    with pytest.raises(RuntimeError, match="bug"):
        await collect_turn(events(), ui_url=UI, redact=Redactor(KEY))
    assert closed == [True]  # checked before any other await: no GC finalizer ran yet


async def test_a_connection_drop_after_the_verdict_keeps_the_saved_answer() -> None:
    """Integrated review P2: after `verification` the answer is committed; a drop during the
    extras must return it (with a warning), not turn it into a tool error."""
    head = b"".join(
        encode_sse(e)
        for e in (
            TokenEvent(text="Kafka keeps logs [S1]."),
            CitationEvent(citation=cite(1)),
            VerificationEvent(status="partial", unsupported_claims=["x"]),
        )
    )
    response = httpx.Response(
        200, stream=BrokenStream(head), headers={"content-type": "text/event-stream"}
    )
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=response)
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "q"})
    assert not result.is_error
    out = structured(result)
    assert "Kafka keeps logs" in out["answer"]
    # both warnings, not just the later-step one (review P2: one hid the other)
    assert "stream broke after the answer was verified" in out["warning"]
    assert "Verification is partial" in out["warning"]


def test_display_url_drops_credentials_query_and_fragment() -> None:
    from ragforge_mcp.errors import display_url

    assert display_url("http://u:p@api.test:8000/x?token=abc#frag") == "http://api.test:8000/x"


async def _stalls_after(*events: ChatEvent) -> AsyncIterator[ChatEvent]:
    for event in events:
        yield event
    await anyio.sleep_forever()
    yield DONE  # pragma: no cover


async def _breaks_after(*events: ChatEvent) -> AsyncIterator[ChatEvent]:
    for event in events:
        yield event
    if events:  # always true here; keeps this an async generator without dead code
        raise ValueError("malformed SSE frame")


async def test_the_time_limit_after_the_verdict_returns_the_saved_answer() -> None:
    """Re-review P2: the stall the time limit exists for, after `verification`, must not turn
    a committed answer into an error (retrying would duplicate the turn)."""
    from ragforge_mcp.turn import collect_turn

    out = await collect_turn(
        _stalls_after(
            TokenEvent(text="Kafka keeps logs [S1]."),
            VerificationEvent(status="verified", unsupported_claims=[]),
        ),
        ui_url="http://ui.test",
        redact=lambda s: s,
        deadline_s=0.05,
    )
    assert "Kafka keeps logs" in out.answer
    assert out.warning is not None
    assert "time limit" in out.warning


async def test_the_time_limit_before_the_verdict_is_still_an_error() -> None:
    from ragforge_mcp.turn import collect_turn

    with pytest.raises(TimeoutError):
        await collect_turn(
            _stalls_after(TokenEvent(text="draft")),
            ui_url="http://ui.test",
            redact=lambda s: s,
            deadline_s=0.05,
        )


async def test_a_malformed_frame_after_the_verdict_keeps_the_saved_answer() -> None:
    from ragforge_mcp.turn import collect_turn

    out = await collect_turn(
        _breaks_after(
            TokenEvent(text="Kafka keeps logs [S1]."),
            VerificationEvent(status="verified", unsupported_claims=[]),
        ),
        ui_url="http://ui.test",
        redact=lambda s: s,
    )
    assert "Kafka keeps logs" in out.answer
    assert out.warning is not None


# ------------------------------------------------------------------ flag reasons (IV-UI)
async def test_flagged_claims_carry_reasons_and_unknown_or_missing_reasons_are_not_found() -> None:
    async def events(verification: VerificationEvent) -> AsyncIterator[ChatEvent]:
        yield TokenEvent(text="Answer \\[S1\\].")
        yield verification
        yield DONE

    with_reasons = VerificationEvent(
        status="partial",
        unsupported_claims=["a", "`b()`", "c"],
        claims=[
            UnsupportedClaim(text="a", reason="not_found"),
            UnsupportedClaim(text="`b()`", reason="inexact_code"),
            UnsupportedClaim(text="c", reason="contradicted"),
        ],
    )
    out = await collect_turn(events(with_reasons), ui_url="http://ui", redact=lambda s: s)
    assert out.unsupported_claims == ["a", "`b()`", "c"]  # kept for older agents
    assert [(f.text, f.reason) for f in out.flagged_claims] == [
        ("a", "not_found"),
        ("c", "not_found"),
        ("`b()`", "inexact_code"),
    ]
    old = VerificationEvent(status="partial", unsupported_claims=["x"])
    out = await collect_turn(events(old), ui_url="http://ui", redact=lambda s: s)
    assert [(f.text, f.reason) for f in out.flagged_claims] == [("x", "not_found")]
