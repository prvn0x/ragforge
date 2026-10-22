"""Chat SSE metrics (ADR-0018 R2-1): TTFT, turn duration, streams in flight, turns by outcome,
and the verification mix / flagged reasons of ANSWERED turns, all per `surface`.

Driven through the real `stream_turn` behind the real middleware, with scripted TurnEvents
(the agent's seam), so the outcome classification is proven on what the agent emits."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from module_obs.obs_metrics_support import Delta, ScriptedChat
from ragforge_core.adapters.llm.chat_ollama import ChatModelError
from ragforge_core.agent.coverage import refusal_only
from ragforge_core.agent.sanitize import MarkdownStreamSanitizer
from ragforge_core.apps.api import chat_stream
from ragforge_core.apps.api.chat_stream import ABSTAIN_TEXT, stream_turn
from ragforge_core.apps.api.middleware import MetricsMiddleware
from ragforge_core.domain.models import (
    ClaimReasonCode,
    TurnDone,
    TurnEvent,
    TurnInterrupt,
    TurnText,
    TurnVerification,
    Verification,
)
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.prompts import render

CHAT = (
    "ragforge_chat_time_to_first_token_seconds",
    "ragforge_chat_turn_duration_seconds",
    "ragforge_chat_turns_total",
    "ragforge_chat_streams_in_flight",
    "ragforge_verification_total",
    "ragforge_flagged_claims_total",
)
REFUSAL = "Your notes don't cover this."


def _done() -> TurnDone:
    return TurnDone(conversation_id=uuid4(), usage=(), prompt_version="answer@1")


def _verified(status: Verification = Verification.verified, *claims: str) -> TurnVerification:
    reasons: tuple[ClaimReasonCode, ...] = tuple("not_found" for _ in claims)
    return TurnVerification(status, tuple(claims), reasons)


class _Surfaces:
    """Stands in for the tenant lookup (its DB path is tested in test_obs_metrics_db)."""

    def __init__(self, found: str) -> None:
        self.found = found

    async def resolve(self, tx: TransactionManager, tenant_id: UUID) -> str:
        return self.found


def _app(script: Sequence[TurnEvent | BaseException], tenant: str | None = None) -> FastAPI:
    app = FastAPI()
    app.state.container = SimpleNamespace(chat=ScriptedChat(), tx=None)

    @app.post("/v1/chat")
    async def chat(request: Request) -> StreamingResponse:
        if tenant is not None:  # what deps.authenticate binds for every authenticated request
            structlog.contextvars.bind_contextvars(tenant_id=tenant)

        async def events() -> AsyncIterator[TurnEvent]:
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return await stream_turn(
            request, events(), MarkdownStreamSanitizer, new_conversation_id=uuid4()
        )

    return app


async def _post(app: FastAPI) -> str:
    transport = httpx.ASGITransport(app=MetricsMiddleware(app))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return (await client.post("/v1/chat")).text


@pytest.fixture(autouse=True)
def _clean_log_context() -> Iterator[None]:
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


def _labels(outcome: str, surface: str = "chat") -> dict[str, str]:
    return {"surface": surface, "outcome": outcome}


# ---------------------------------------------------------------------- outcomes
async def test_an_answered_turn_records_every_chat_metric() -> None:
    delta = Delta(CHAT)
    body = await _post(_app([TurnText("Fact one holds [S1]."), _verified(), _done()]))
    assert "event: done" in body
    assert delta.of("ragforge_chat_turns_total", **_labels("answered")) == 1
    assert delta.of("ragforge_verification_total", surface="chat", status="verified") == 1
    timing = {"surface": "chat", "provider": "ollama"}
    assert delta.of("ragforge_chat_time_to_first_token_seconds_count", **timing) == 1
    assert delta.of("ragforge_chat_turn_duration_seconds_count", **timing) == 1
    assert delta.of("ragforge_chat_streams_in_flight") == 0  # opened and closed


async def test_flagged_claims_are_counted_by_reason() -> None:
    claims = ("Claim A.", "Claim B.")
    flagged = TurnVerification(Verification.partial, claims, ("not_found", "inexact_code"))
    delta = Delta(CHAT)
    await _post(_app([TurnText("Claim A. Claim B."), flagged, _done()]))
    assert delta.of("ragforge_verification_total", surface="chat", status="partial") == 1
    assert delta.of("ragforge_flagged_claims_total", surface="chat", reason="not_found") == 1
    assert delta.of("ragforge_flagged_claims_total", surface="chat", reason="inexact_code") == 1


async def test_an_abstain_is_not_a_verification_verdict() -> None:
    assert render("abstain", closest="\n\n- [Page](https://x)").startswith(ABSTAIN_TEXT)
    text = render("abstain", closest="\n\nThe closest pages I found:\n- Page 2")
    delta = Delta(CHAT)
    await _post(_app([TurnText(text), _verified(), _done()]))
    assert delta.of("ragforge_chat_turns_total", **_labels("abstained")) == 1
    assert delta.labels_of("ragforge_verification_total") == []


async def test_a_refusal_is_not_a_verification_verdict() -> None:
    assert refusal_only(REFUSAL)
    delta = Delta(CHAT)
    await _post(_app([TurnText(REFUSAL), _verified(Verification.unverified), _done()]))
    assert delta.of("ragforge_chat_turns_total", **_labels("refused")) == 1
    assert delta.labels_of("ragforge_verification_total") == []


async def test_a_stream_that_fails_is_an_error_turn() -> None:
    delta = Delta(CHAT)
    body = await _post(_app([TurnText("Partial"), ChatModelError("model provider timed out")]))
    assert "event: error" in body
    assert delta.of("ragforge_chat_turns_total", **_labels("error")) == 1
    assert delta.of("ragforge_chat_streams_in_flight") == 0


async def test_a_turn_paused_for_approval_is_interrupted() -> None:
    delta = Delta(CHAT)
    await _post(_app([TurnInterrupt(uuid4(), ())]))
    assert delta.of("ragforge_chat_turns_total", **_labels("interrupted")) == 1
    assert delta.labels_of("ragforge_chat_turn_duration_seconds_count") == []


async def test_a_stream_closed_by_the_client_is_interrupted_and_leaves_no_stream_open() -> None:
    app = _app([TurnText("Fact one."), _verified(), _done()])
    scope = {"type": "http", "app": app, "path": "/v1/chat", "headers": [], "state": {}}
    request = Request(scope)

    async def events() -> AsyncIterator[TurnEvent]:
        yield TurnText("Fact one holds")
        yield TurnText(" [S1].")

    delta = Delta(CHAT)
    response = await stream_turn(
        request, events(), MarkdownStreamSanitizer, new_conversation_id=uuid4()
    )
    body = response.body_iterator
    await anext(body)  # type: ignore[call-overload]
    assert delta.of("ragforge_chat_streams_in_flight") == 1
    await body.aclose()  # type: ignore[attr-defined]
    assert delta.of("ragforge_chat_streams_in_flight") == 0
    # driven without the middleware, so the surface is the ContextVar default
    assert delta.of("ragforge_chat_turns_total", **_labels("interrupted", "other")) == 1


# ---------------------------------------------------------------------- surface
@pytest.mark.parametrize("found", ["eval", "demo"])
async def test_eval_and_demo_traffic_never_moves_the_owner_series(
    found: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_stream, "_SURFACES", _Surfaces(found))
    delta = Delta(CHAT)
    await _post(_app([TurnText("Fact [S1]."), _verified(), _done()], tenant=str(uuid4())))
    assert delta.of("ragforge_chat_turns_total", **_labels("answered", found)) == 1
    assert delta.of("ragforge_verification_total", surface=found, status="verified") == 1
    assert all(
        labels["surface"] == found for labels in delta.labels_of("ragforge_chat_turns_total")
    )


async def test_without_an_authenticated_tenant_the_route_surface_is_kept() -> None:
    delta = Delta(CHAT)
    await _post(_app([TurnText("Fact [S1]."), _verified(), _done()]))
    assert delta.of("ragforge_chat_turns_total", **_labels("answered", "chat")) == 1
