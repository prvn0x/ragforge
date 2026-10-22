"""POST /v1/chat and /v1/approvals/{run_id}: TurnEvent -> SSE mapping and output sanitization.

The fake sanitizer marks everything it emits (`<s>..</s>` from feed, `<f>..</f>` from finish),
so any unmarked text in a text-bearing event would be raw model output reaching the client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from structlog.testing import capture_logs

from module_e2.e2_support import Fakes, TenantWorld, auth, make_hit
from ragforge_contracts.api import ClaimReason, UnsupportedClaim
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
    decode_sse_data,
)
from ragforge_core.domain.errors import LeaseLost, TurnInProgress
from ragforge_core.domain.models import (
    ApprovalChoice,
    ChatTurnUsage,
    ToolCall,
    TurnCitation,
    TurnDone,
    TurnExtra,
    TurnInterrupt,
    TurnReplace,
    TurnSources,
    TurnText,
    TurnToolCall,
    TurnVerification,
    Verification,
)

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]
U1, U2 = "https://notion.so/page-1", "https://notion.so/page-2"


def _events(body: str) -> list[ChatEvent]:
    out: list[ChatEvent] = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        event = decode_sse_data(lines["data"])
        assert event is not None
        assert event.type == lines["event"]
        out.append(event)
    return out


async def _chat(
    http: httpx.AsyncClient, w: TenantWorld, conversation: UUID | None = None
) -> httpx.Response:
    body: dict[str, Any] = {"collection_ids": [str(w.collection)], "message": "question?"}
    if conversation is not None:
        body["conversation_id"] = str(conversation)
    return await http.post("/v1/chat", json=body, headers=auth(w.key))


def _text_payloads(events: list[ChatEvent]) -> list[str]:
    texts: list[str] = []
    for e in events:
        if isinstance(e, TokenEvent | AnswerReplaceEvent):
            texts.append(e.text)
        elif isinstance(e, VerificationEvent):
            texts.extend(e.unsupported_claims)
    return texts


async def test_full_turn_mapping_order_and_sanitization(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    hit1, hit2 = make_hit(w.collection, url=U1, n=1), make_hit(w.collection, url=U2, n=2)
    call = ToolCall(id="call-9", name="search_notes", arguments={"q": "x"})
    usage = ChatTurnUsage(provider="fake", model="m", prompt_tokens=3, completion_tokens=4)
    fakes.agent.script = [
        TurnSources(hits=(hit1,)),
        TurnText("Hello "),
        TurnText("see [link"),  # the fake holds from "[" until finish()
        TurnText(f"]({U1}) end"),
        TurnCitation(hit=hit1, label=3),
        TurnToolCall(call=call),
        TurnSources(hits=(hit2,)),  # mid-turn retrieval
        TurnText(" more"),
        TurnVerification(status=Verification.unsupported, unsupported_claims=("bad claim",)),
        TurnReplace(text="replacement"),
        TurnDone(conversation_id=uuid4(), usage=(usage,), prompt_version="p1"),
    ]
    r = await _chat(http, w)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)

    assert [e.type for e in events] == [
        "token", "token", "citation", "tool_call", "token", "verification", "answer.replace",
        "done",
    ]  # fmt: skip
    assert [e.text for e in events if isinstance(e, TokenEvent)] == [
        "<s>Hello </s>", "<s>see </s>", f"<f>[link]({U1}) end more</f>",
    ]  # fmt: skip
    # Every text-bearing event went through a sanitizer: nothing raw reaches the client.
    assert all(t.startswith(("<s>", "<f>")) for t in _text_payloads(events))
    verification = next(e for e in events if isinstance(e, VerificationEvent))
    assert (verification.status, verification.unsupported_claims) == (
        "unsupported",
        ["<s>bad claim</s>"],
    )
    # A producer without reasons: every claim is "not_found", the prominent flag.
    assert verification.claims == [UnsupportedClaim(text="<s>bad claim</s>", reason="not_found")]
    assert next(e for e in events if isinstance(e, AnswerReplaceEvent)).text == "<s>replacement</s>"
    citation = next(e for e in events if isinstance(e, CitationEvent)).citation
    assert (citation.anchor, citation.label, citation.document_id, str(citation.url)) == (
        "block-1",
        3,  # the answer's "[S3]", carried separately from the source anchor
        hit1.document_id,
        U1,
    )
    tool = next(e for e in events if isinstance(e, ToolCallEvent))
    assert (tool.tool_call_id, tool.tool, tool.args) == (
        "call-9",
        "search_notes",
        {"q": "<s>x</s>"},
    )

    # allow() ran for every TurnSources BEFORE any later text was fed to that sanitizer.
    log = fakes.sanitizers.log
    answer = [(op, arg) for ident, op, arg in log if ident == 0]
    assert answer.index(("allow", (U2,))) < answer.index(("feed", " more"))
    assert answer[0] == ("allow", (U1,))
    # The answer used ONE sanitizer; each later text got its own with the allow-list replayed.
    assert sum(1 for ident, op, _ in log if ident == 0 and op == "feed") == 4
    # instance 1 sanitized the tool-call arg (before the second TurnSources), the rest came later.
    assert fakes.sanitizers.instances[1].allowed == [U1]
    assert all(s.allowed == [U1, U2] for s in fakes.sanitizers.instances[2:])

    # New conversation: created by the API for this user and reported on done.
    [turn] = fakes.agent.turns
    conversation = turn.conversation_id
    assert conversation is not None
    assert (turn.collection_ids, turn.message) == ([w.collection], "question?")
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.conversation_id == conversation
    assert done.usage[0].completion_tokens == 4
    assert superuser.execute(
        "SELECT tenant_id, user_id FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone() == (w.tenant, w.user)


async def test_verification_claims_carry_reasons_with_the_same_sanitized_text(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    """FLAG: `claims[i]` is `unsupported_claims[i]` (sanitized ONCE, same text) with its reason;
    the old list is unchanged for old clients, and a verified answer has no claims."""
    w = world()
    fakes.agent.script = [
        TurnText("answer"),
        TurnVerification(
            status=Verification.partial,
            unsupported_claims=("red claim", "muted `code` claim"),
            claim_reasons=("not_found", "inexact_code"),
        ),
        TurnDone(conversation_id=uuid4(), usage=(), prompt_version="p1"),
    ]
    events = _events((await _chat(http, w)).text)
    verification = next(e for e in events if isinstance(e, VerificationEvent))
    assert verification.status == "partial"
    assert verification.unsupported_claims == ["<s>red claim</s>", "<s>muted `code` claim</s>"]
    assert verification.claims == [
        UnsupportedClaim(text="<s>red claim</s>", reason=ClaimReason.not_found),
        UnsupportedClaim(text="<s>muted `code` claim</s>", reason=ClaimReason.inexact_code),
    ]
    assert [c.text for c in verification.claims] == verification.unsupported_claims

    fakes.agent.script = [
        TurnText("answer"),
        TurnVerification(status=Verification.verified),
        TurnDone(conversation_id=uuid4(), usage=(), prompt_version="p1"),
    ]
    events = _events((await _chat(http, w)).text)
    verified = next(e for e in events if isinstance(e, VerificationEvent))
    assert (verified.status, verified.unsupported_claims, verified.claims) == ("verified", [], [])


@pytest.mark.parametrize("existing", [False, True])
async def test_interrupt_flushes_and_carries_new_conversation_id(
    http: httpx.AsyncClient, fakes: Fakes, world: World, existing: bool
) -> None:
    w = world()
    run = uuid4()
    call = ToolCall(id="c1", name="remove_from_index", arguments={"id": "p"})
    fakes.agent.script = [TurnText("partial ["), TurnInterrupt(run_id=run, pending=(call,))]
    r = await _chat(http, w, w.conversation if existing else None)
    events = _events(r.text)
    assert [e.type for e in events] == ["token", "token", "interrupt"]
    assert [e.text for e in events if isinstance(e, TokenEvent)] == ["<s>partial </s>", "<f>[</f>"]
    interrupt = events[-1]
    assert isinstance(interrupt, InterruptEvent)
    assert interrupt.run_id == run
    assert [p.tool_call_id for p in interrupt.pending] == ["c1"]
    assert interrupt.pending[0].args == {"id": "<s>p</s>"}
    passed = fakes.agent.turns[0].conversation_id
    if existing:
        assert passed == w.conversation
        assert interrupt.conversation_id is None
    else:
        assert interrupt.conversation_id == passed
        assert passed != w.conversation


async def test_mid_stream_error_is_an_error_event_without_internals(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.agent.script = [TurnText("par")]
    fakes.agent.fail = RuntimeError("connection to db password=hunter2 failed")
    with capture_logs() as logs:
        r = await _chat(http, w)
    events = _events(r.text)
    assert [e.type for e in events] == ["token", "error"]
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert (error.problem.status, error.problem.title, error.problem.detail) == (
        500,
        "Internal error",
        None,
    )
    assert error.conversation_id == fakes.agent.turns[0].conversation_id
    assert "hunter2" not in r.text
    assert fakes.agent.closed == 1
    failed = [e for e in logs if e["event"] == "chat_turn_failed"]
    assert failed == [
        {
            "event": "chat_turn_failed",
            "log_level": "warning",
            "error": "RuntimeError",
            "status": 500,
            "title": "Internal error",
        }
    ], "the type is logged, never the internal message"


async def test_domain_error_mid_stream_keeps_its_status(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.agent.script = [TurnText("x")]  # fails AFTER streaming began
    fakes.agent.fail = LeaseLost("run lease lost")
    with capture_logs() as logs:
        events = _events((await _chat(http, w, w.conversation)).text)
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert (error.problem.status, error.conversation_id) == (409, None)
    failed = [e for e in logs if e["event"] == "chat_turn_failed"]
    assert [(e["error"], e["status"]) for e in failed] == [("LeaseLost", 409)]
    assert "detail" not in failed[0], "adapter detail may carry upstream text: never logged"


async def test_stream_without_done_is_reported(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.agent.script = [TurnText("x")]
    events = _events((await _chat(http, w)).text)
    assert [e.type for e in events] == ["token", "error"]


async def test_turn_in_progress_is_409_for_existing_conversation(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.agent.fail_before_first = TurnInProgress()
    r = await _chat(http, w, w.conversation)
    assert r.status_code == 409
    assert r.headers["content-type"].startswith("application/problem+json")
    assert fakes.agent.closed == 1


async def test_immediate_failure_on_new_conversation_still_reports_its_id(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    fakes.agent.fail_before_first = TurnInProgress()
    r = await _chat(http, w)
    assert r.status_code == 200
    [error] = _events(r.text)
    assert isinstance(error, ErrorEvent)
    assert error.problem.status == 409
    assert error.conversation_id == fakes.agent.turns[0].conversation_id


async def test_approvals_resume_streams_sanitized_events(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    fakes.agent.script = [
        TurnText("done it"),
        TurnDone(conversation_id=w.conversation, usage=(), prompt_version="p1"),
    ]
    r = await http.post(
        f"/v1/approvals/{w.run}", json={"decisions": {"call-1": "reject"}}, headers=auth(w.key)
    )
    assert r.status_code == 200
    events = _events(r.text)
    assert [e.type for e in events] == ["token", "done"]
    assert isinstance(events[0], TokenEvent)
    assert events[0].text == "<s>done it</s>"
    [resume] = fakes.agent.resumes
    assert (resume.run_id, resume.decisions) == (w.run, {"call-1": ApprovalChoice.reject})
    assert resume.principal.tenant_id == w.tenant
    assert superuser.execute(
        "SELECT action, target FROM rf.audit_log WHERE tenant_id = %s", (w.tenant,)
    ).fetchall() == [("approvals.submit", str(w.run))]
    r = await http.post(f"/v1/approvals/{w.run}", json={"decisions": {}}, headers=auth(w.key))
    assert r.status_code == 422


async def test_tool_call_args_are_sanitized_at_every_depth(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    """Tool args are model output (prompt injection can put links there): every string leaf goes
    through its own sanitizer with the allow-list replayed; non-strings and keys pass as is."""
    w = world()
    hit = make_hit(w.collection, url=U1)
    args = {"q": "see https://evil.example/p", "n": 3, "flag": True, "none": None,
            "nested": {"urls": ["a", {"deep": "b"}]}}  # fmt: skip
    call = ToolCall(id="t1", name="search_notes", arguments=args)
    pending = ToolCall(id="t2", name="remove_from_index", arguments={"ids": ["p1"]})
    fakes.agent.script = [
        TurnSources(hits=(hit,)),
        TurnToolCall(call=call),
        TurnInterrupt(run_id=uuid4(), pending=(pending,)),
    ]
    events = _events((await _chat(http, w)).text)
    tool = next(e for e in events if isinstance(e, ToolCallEvent))
    assert tool.args == {
        "q": "<s>see https://evil.example/p</s>", "n": 3, "flag": True, "none": None,
        "nested": {"urls": ["<s>a</s>", {"deep": "<s>b</s>"}]},
    }  # fmt: skip
    interrupt = events[-1]
    assert isinstance(interrupt, InterruptEvent)
    assert interrupt.pending[0].args == {"ids": ["<s>p1</s>"]}
    leaf_sanitizers = fakes.sanitizers.instances[1:]
    assert len(leaf_sanitizers) >= 4
    assert all(s.allowed == [U1] for s in leaf_sanitizers)


async def test_done_reports_the_api_created_conversation_id(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    """A misbehaving agent that reports another id on TurnDone cannot redirect the client."""
    w = world()
    fakes.agent.echo_conversation_id = False
    rogue = uuid4()
    fakes.agent.script = [TurnDone(conversation_id=rogue, usage=(), prompt_version="p1")]
    done = _events((await _chat(http, w)).text)[-1]
    assert isinstance(done, DoneEvent)
    assert done.conversation_id == fakes.agent.turns[0].conversation_id
    assert done.conversation_id != rogue


async def test_an_extra_is_its_own_event_rendered_literally(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    """ADR-0016: extras are never formatted. Prose is literal (Markdown escaped, URLs defanged)
    and code is one safely fenced block; the answer's sanitizer (and its allowed source URLs)
    is never used for them."""
    w = world()
    hit1 = make_hit(w.collection, url=U1, n=1)
    usage = ChatTurnUsage(provider="fake", model="m", prompt_tokens=1, completion_tokens=1)
    fakes.agent.script = [
        TurnSources(hits=(hit1,)),
        TurnText("Answer [S1]."),
        TurnVerification(status=Verification.verified),
        TurnExtra(kind="incident", text=f"**bold** see {U1}"),
        TurnExtra(kind="code", text="Shows it.", code="x = 1", language="py"),
        TurnDone(conversation_id=uuid4(), usage=(usage,), prompt_version="p1"),
    ]
    events = _events((await _chat(http, w)).text)
    assert [e.type for e in events][-3:] == ["extra", "extra", "done"]
    incident, code = events[-3], events[-2]
    assert isinstance(incident, ExtraEvent)
    assert isinstance(code, ExtraEvent)
    assert incident.text == "\\*\\*bold\\*\\* see https\\[\\:\\]\\/\\/notion\\.so\\/page\\-1"
    assert code.text == "```py\nx = 1\n```\nShows it\\."
    assert not any(
        arg == f"**bold** see {U1}" for _, op, arg in fakes.sanitizers.log if op == "feed"
    )
