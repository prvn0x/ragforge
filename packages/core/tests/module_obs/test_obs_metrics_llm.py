"""InstrumentedChatModel (ADR-0018 R2-3): outcomes from the adapters' own error classes, tokens
from the usage reports, the prompt label from `for_prompt`, and the port contract kept."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from module_obs.obs_metrics_support import Delta, ScriptedChat
from ragforge_core.adapters.llm.chat_gemini import (
    GeminiChatModel,
    ModelConfigurationError,
    ModelRequestRejected,
    OutputLimitReached,
    QuotaExhausted,
    ResponseBlocked,
    StreamInterrupted,
    UnsupportedSchema,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OllamaChatModel, OutputTruncated
from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited
from ragforge_core.domain.models import ChatMessage, ChatTurnUsage, Role, TextDelta, UsageReport
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.catalog import LLM_OUTCOMES, UNKNOWN_PROMPT
from ragforge_core.observability.llm import InstrumentedChatModel, llm_outcome
from ragforge_core.observability.prompt_context import for_prompt
from ragforge_core.prompts import get_prompt

LLM = (
    "ragforge_llm_requests_total",
    "ragforge_llm_request_duration_seconds",
    "ragforge_llm_tokens_total",
)
ANSWER = get_prompt("answer").tag
MESSAGES = [ChatMessage(role=Role.user, content="q")]
USAGE = ChatTurnUsage("ollama", "llama3.2:3b", 1, 1)


async def _drain(chat: ChatModel) -> list[object]:
    return [e async for e in chat.stream("sys", MESSAGES)]


def _labels(outcome: str, prompt: str = ANSWER) -> dict[str, str]:
    return {"provider": "ollama", "model": "llama3.2:3b", "prompt": prompt, "outcome": outcome}


# ---------------------------------------------------------------------- happy paths
async def test_a_streamed_call_records_outcome_duration_and_both_token_directions() -> None:
    inner = ScriptedChat(prompt_tokens=11, completion_tokens=4)
    delta = Delta(LLM)
    with for_prompt(ANSWER):
        events = await _drain(InstrumentedChatModel(inner))
    assert [type(e) for e in events] == [TextDelta, UsageReport]  # passed through unchanged
    assert delta.of("ragforge_llm_requests_total", **_labels("ok")) == 1
    assert delta.of("ragforge_llm_request_duration_seconds_count", **_labels("ok")) == 1
    tokens = {"provider": "ollama", "model": "llama3.2:3b", "prompt": ANSWER}
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="prompt") == 11
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="completion") == 4


async def test_complete_json_is_measured_and_returns_the_inner_result() -> None:
    inner = ScriptedChat(payload={"verdict": "supported"}, prompt_tokens=7, completion_tokens=3)
    tag = get_prompt("ground_check").tag
    delta = Delta(LLM)
    with for_prompt(tag):
        payload, usage = await InstrumentedChatModel(inner).complete_json("s", MESSAGES, {})
    assert payload == {"verdict": "supported"}
    assert usage is inner.usage
    assert delta.of("ragforge_llm_requests_total", **_labels("ok", tag)) == 1
    tokens = {"provider": "ollama", "model": "llama3.2:3b", "prompt": tag}
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="completion") == 3


async def test_the_prompt_is_read_when_the_stream_is_created() -> None:
    """A stream created inside `for_prompt` and drained after it still carries its prompt."""
    chat = InstrumentedChatModel(ScriptedChat())
    delta = Delta(LLM)
    with for_prompt(ANSWER):
        stream = chat.stream("s", MESSAGES)
    _ = [e async for e in stream]
    assert delta.of("ragforge_llm_requests_total", **_labels("ok")) == 1


async def test_without_for_prompt_the_call_is_visibly_unknown() -> None:
    delta = Delta(LLM)
    await _drain(InstrumentedChatModel(ScriptedChat()))
    assert delta.of("ragforge_llm_requests_total", **_labels("ok", UNKNOWN_PROMPT)) == 1


async def test_an_unrecognised_prompt_name_never_becomes_a_label() -> None:
    delta = Delta(LLM)
    with for_prompt("tenant-6f1c2a3b-question-text"):
        await _drain(InstrumentedChatModel(ScriptedChat()))
    assert delta.of("ragforge_llm_requests_total", **_labels("ok", "other")) == 1


async def test_the_wrapper_keeps_the_port_identity() -> None:
    chat = InstrumentedChatModel(ScriptedChat())
    assert isinstance(chat, ChatModel)
    assert (chat.provider, chat.model) == ("ollama", "llama3.2:3b")


# ---------------------------------------------------------------------- cancellation
async def test_closing_before_the_usage_report_is_interrupted_and_closes_the_inner() -> None:
    inner = ScriptedChat()
    delta = Delta(LLM)
    with for_prompt(ANSWER):
        stream = InstrumentedChatModel(inner).stream("s", MESSAGES)
        assert isinstance(await anext(stream), TextDelta)
        await stream.aclose()  # type: ignore[attr-defined]
    assert inner.closed == 1  # the port contract: the upstream request is stopped
    assert delta.of("ragforge_llm_requests_total", **_labels("interrupted")) == 1


async def test_closing_after_the_usage_report_counts_as_ok() -> None:
    delta = Delta(LLM)
    with for_prompt(ANSWER):
        stream = InstrumentedChatModel(ScriptedChat()).stream("s", MESSAGES)
        await anext(stream)
        assert isinstance(await anext(stream), UsageReport)
        await stream.aclose()  # type: ignore[attr-defined]
    assert delta.of("ragforge_llm_requests_total", **_labels("ok")) == 1


# ---------------------------------------------------------------------- classification
@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (UpstreamRateLimited(3.0), "rate_limited"),
        (QuotaExhausted("daily"), "quota_exhausted"),
        (ModelConfigurationError("bad key"), "config"),
        (ResponseBlocked("prompt", "safety"), "blocked"),
        (StreamInterrupted("transient"), "interrupted"),
        (OutputTruncated(USAGE), "output_limit"),
        (OutputLimitReached(USAGE), "output_limit"),
        (ChatModelError("model provider timed out"), "timeout"),
        (ChatModelError("model provider unreachable"), "server"),
        (ChatModelError("model provider returned HTTP 503: x"), "server"),
        (ChatModelError("model provider returned HTTP 404: x"), "other"),
        (ChatModelError("malformed record from model provider"), "other"),
        (ModelRequestRejected("model provider rejected the request: 400"), "other"),
        (UnsupportedSchema("bad schema"), "other"),
        (ValueError("not json"), "other"),
    ],
)
async def test_failures_map_to_the_adapters_categories(error: Exception, outcome: str) -> None:
    assert outcome in LLM_OUTCOMES
    assert llm_outcome(error) == outcome
    delta = Delta(LLM)
    with for_prompt(ANSWER), pytest.raises(type(error)):
        await _drain(InstrumentedChatModel(ScriptedChat(error=error)))
    with for_prompt(ANSWER), pytest.raises(type(error)):
        await InstrumentedChatModel(ScriptedChat(error=error)).complete_json("s", MESSAGES, {})
    assert delta.of("ragforge_llm_requests_total", **_labels(outcome)) == 2


async def test_a_truncated_answer_still_counts_the_tokens_it_billed() -> None:
    billed = ChatTurnUsage("ollama", "llama3.2:3b", 900, 120)
    delta = Delta(LLM)
    with for_prompt(ANSWER), pytest.raises(OutputTruncated):
        await _drain(InstrumentedChatModel(ScriptedChat(error=OutputTruncated(billed))))
    tokens = {"provider": "ollama", "model": "llama3.2:3b", "prompt": ANSWER}
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="prompt") == 900


# ---------------------------------------------------------------------- the REAL adapters
# The base ChatModelError categories are told apart by the adapters' fixed messages, so they
# are pinned here against the adapters themselves: a reworded message fails this, not prod.
Handler = Callable[[httpx.Request], httpx.Response]


def _ollama(handler: Handler) -> OllamaChatModel:
    client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return OllamaChatModel(client, model="llama3.2:3b", num_ctx=8192, keep_alive="30m", think=None)


def _gemini(handler: Handler) -> GeminiChatModel:
    client = httpx.AsyncClient(
        base_url="http://gemini.test", transport=httpx.MockTransport(handler)
    )
    return GeminiChatModel(
        client, api_key=SecretStr("synthetic-test-key"), model="gemini-x", thinking_budget=-1
    )


def _raise(exc: type[httpx.TransportError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("synthetic", request=request)

    return handler


def _status(code: int, body: object = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json=body if body is not None else {"error": "x"})

    return handler


def _ollama_ok(request: httpx.Request) -> httpx.Response:
    records = [
        {"message": {"role": "assistant", "content": "hi"}, "done": False},
        {"done": True, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3},
    ]
    return httpx.Response(200, text="\n".join(json.dumps(r) for r in records))


async def _outcome_of(chat: ChatModel) -> str:
    delta = Delta(("ragforge_llm_requests_total",))
    wrapped = InstrumentedChatModel(chat)
    with for_prompt(ANSWER), pytest.raises(RagforgeError):  # every adapter error is one
        await _drain(wrapped)
    (labels,) = delta.labels_of("ragforge_llm_requests_total")
    return labels["outcome"]


async def test_real_ollama_ok_counts_its_usage_fields() -> None:
    delta = Delta(LLM)
    with for_prompt(ANSWER):
        await _drain(InstrumentedChatModel(_ollama(_ollama_ok)))
    tokens = {"provider": "ollama", "model": "llama3.2:3b", "prompt": ANSWER}
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="prompt") == 12
    assert delta.of("ragforge_llm_tokens_total", **tokens, direction="completion") == 3
    assert delta.of("ragforge_llm_requests_total", **_labels("ok")) == 1


@pytest.mark.parametrize(
    ("handler", "outcome"),
    [
        (_raise(httpx.ReadTimeout), "timeout"),
        (_raise(httpx.ConnectError), "server"),
        (_status(500), "server"),
        (_status(429), "rate_limited"),
        (_status(404), "other"),
    ],
)
async def test_real_ollama_failures(handler: Handler, outcome: str) -> None:
    assert await _outcome_of(_ollama(handler)) == outcome


_DAILY = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
            }
        ],
    }
}


@pytest.mark.parametrize(
    ("handler", "outcome"),
    [
        (_raise(httpx.ReadTimeout), "timeout"),
        (_raise(httpx.ConnectError), "server"),
        (_status(500, {"error": {"code": 500, "status": "INTERNAL"}}), "server"),
        (_status(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}), "rate_limited"),
        (_status(429, _DAILY), "quota_exhausted"),
        (_status(403, {"error": {"code": 403, "status": "PERMISSION_DENIED"}}), "config"),
    ],
)
async def test_real_gemini_failures(handler: Handler, outcome: str) -> None:
    assert await _outcome_of(_gemini(handler)) == outcome
