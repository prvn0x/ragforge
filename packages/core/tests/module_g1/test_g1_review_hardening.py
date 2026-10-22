"""Regression tests for the round-1 review: the thinking fallback cannot be triggered (or made
sticky) by echoed user text, replayed tool turns always carry a signature, the key is checked for
header-safe characters before any request, retry delays are bounded, and dict-like schemas fail
at conversion time instead of as a permanent 400 at request time."""

from __future__ import annotations

import math
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from module_g1.g1_support import (
    BASE,
    CANARY_KEY,
    MODEL,
    STREAM_URL,
    USER,
    answered,
    body,
    call_chunk,
    collect,
    finish_chunk,
    google_error,
    ok,
    text_chunk,
)
from ragforge_core.adapters.llm.chat_gemini import (
    MAX_RETRY_AFTER_S,
    SKIP_SIGNATURE,
    GeminiChatModel,
    ModelConfigurationError,
    ModelRequestRejected,
    UnsupportedSchema,
    to_gemini_schema,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError
from ragforge_core.config import LLMProvider, Settings
from ragforge_core.domain.errors import UpstreamRateLimited
from ragforge_core.domain.models import ChatMessage, Role, ToolCall, ToolSpec

SEARCH = ToolSpec(
    "search_notes",
    "Search the notes",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


def _settings(key: str) -> Settings:
    """Never reads the developer's .env (it holds live keys)."""
    return Settings(_env_file=None, llm_provider=LLMProvider.gemini, gemini_api_key=SecretStr(key))


def _invalid(message: str, details: list[dict[str, Any]] | None = None) -> httpx.Response:
    return httpx.Response(400, json=google_error(400, "INVALID_ARGUMENT", message, details))


def _sends_thinking(route: respx.Route, call: int) -> bool:
    return "thinkingConfig" in body(route, call)["generationConfig"]


# ---------------------------------------------------------------------- thinking fallback
@respx.mock
async def test_echoed_user_text_mentioning_think_is_one_call_and_keeps_the_budget(
    model: GeminiChatModel,
) -> None:
    """The reviewer's reproduction: 'I think so' echoed in an unrelated 400."""
    user = (ChatMessage(Role.user, "I think so"),)
    route = respx.post(STREAM_URL).mock(
        side_effect=[_invalid("Invalid value at 'contents': I think so"), answered()]
    )
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", user))
    assert route.call_count == 1
    await collect(model.stream("", USER))
    assert _sends_thinking(route, 1)  # the configured budget survives


@respx.mock
async def test_echoed_phrase_that_matches_still_keeps_the_budget_when_the_retry_fails(
    model: GeminiChatModel,
) -> None:
    """Even a user who types the exact phrase cannot flip the setting: the retry carries the
    same text, is rejected the same way, and only an ACCEPTED retry drops the budget."""
    echoed = _invalid("Invalid argument near: 'what is a thinking budget?'")
    route = respx.post(STREAM_URL).mock(side_effect=[echoed, echoed, answered()])
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))
    assert route.call_count == 2
    await collect(model.stream("", USER))
    assert _sends_thinking(route, 2)


@respx.mock
async def test_failed_fallback_of_any_kind_keeps_the_budget(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            _invalid("Budget 0 is invalid. This model only works in thinking mode."),
            httpx.Response(500, json=google_error(500, "INTERNAL")),
            answered(),
        ]
    )
    with pytest.raises(ChatModelError):
        await collect(model.stream("", USER))
    await collect(model.stream("", USER))
    assert _sends_thinking(route, 2)


@respx.mock
async def test_bad_request_field_violation_on_thinking_config_is_recognised(
    model: GeminiChatModel,
) -> None:
    violation = {
        "@type": "type.googleapis.com/google.rpc.BadRequest",
        "fieldViolations": [{"field": "generation_config.thinking_config.thinking_budget"}],
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            _invalid("opaque", [violation]),
            ok([text_chunk("{}"), finish_chunk()]),
            ok([text_chunk("{}"), finish_chunk()]),
        ]
    )
    schema: dict[str, object] = {"type": "object", "properties": {"a": {"type": "string"}}}
    await model.complete_json("", USER, schema)
    await model.complete_json("", USER, schema)
    assert [_sends_thinking(route, i) for i in range(3)] == [True, False, False]


@respx.mock
async def test_unrelated_field_violation_is_not_a_thinking_rejection(
    model: GeminiChatModel,
) -> None:
    violation = {
        "@type": "type.googleapis.com/google.rpc.BadRequest",
        "fieldViolations": [{"field": "contents[0].parts[0].text"}],
    }
    route = respx.post(STREAM_URL).mock(return_value=_invalid("opaque", [violation]))
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))
    assert route.call_count == 1


# ---------------------------------------------------------------------- thought signatures
def _replay(*turns: tuple[ToolCall, ...]) -> list[ChatMessage]:
    history = [*USER]
    for calls in turns:
        history.append(ChatMessage(Role.assistant, "", tool_calls=calls))
        history.extend(ChatMessage(Role.tool, "r", tool_call_id=c.id) for c in calls)
        history.append(ChatMessage(Role.user, "and then?"))
    return history


@respx.mock
async def test_resume_on_a_fresh_instance_sends_the_placeholder_on_each_turns_first_call(
    client: httpx.AsyncClient,
) -> None:
    """A tool approval resumed after a restart, on another replica, or after LRU eviction: no
    signature is cached, so each model turn's first call carries Google's placeholder (and only
    the first: parallel calls carry one signature)."""
    fresh = GeminiChatModel(client, api_key=SecretStr(CANARY_KEY), model=MODEL)
    route = respx.post(STREAM_URL).mock(return_value=answered())
    first = (ToolCall("a", "search_notes", {"query": "1"}), ToolCall("b", "search_notes", {}))
    second = (ToolCall("c", "search_notes", {"query": "2"}),)
    await collect(fresh.stream("", _replay(first, second), [SEARCH]))
    model_turns = [c["parts"] for c in body(route)["contents"] if c["role"] == "model"]
    assert [[p.get("thoughtSignature") for p in parts] for parts in model_turns] == [
        [SKIP_SIGNATURE, None],
        [SKIP_SIGNATURE],
    ]


@respx.mock
async def test_cached_signature_wins_over_the_placeholder(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            ok([call_chunk("search_notes", {"query": "q"}, signature="c2ln"), finish_chunk()]),
            answered(),
        ]
    )
    calls = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    await collect(model.stream("", _replay(tuple(calls)), [SEARCH]))
    assert body(route)["contents"][1]["parts"][0]["thoughtSignature"] == "c2ln"


@respx.mock
async def test_text_only_model_turns_get_no_signature(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    history = [*USER, ChatMessage(Role.assistant, "hi"), ChatMessage(Role.user, "again")]
    await collect(model.stream("", history))
    assert body(route)["contents"][1]["parts"] == [{"text": "hi"}]


# ---------------------------------------------------------------------- API key characters
@pytest.mark.parametrize(
    "key",
    [
        "AIza“CANARY-smart-quote-0123456789",  # pasted from a doc
        "AIzaCANARY inner-space-0123456789",
        "AIzaCANARY\tinner-tab-0123456789",
        "AIzaCANARYé-latin1-0123456789",
    ],
)
def test_non_header_safe_key_fails_before_any_request_without_echoing_it(
    client: httpx.AsyncClient, key: str
) -> None:
    with pytest.raises(ModelConfigurationError) as direct:
        GeminiChatModel(client, api_key=SecretStr(key), model=MODEL)
    with pytest.raises(ModelConfigurationError) as via_settings:  # the bootstrap path
        GeminiChatModel.from_settings(_settings(key), client)
    for caught in (direct, via_settings):
        assert str(caught.value) == "model provider API key has invalid characters"
        rendered = repr(caught.value) + repr(caught.value.args) + repr(caught.value.__context__)
        assert "CANARY" not in rendered


def test_settings_accept_a_normal_key_and_strip_surrounding_newlines(
    client: httpx.AsyncClient,
) -> None:
    settings = _settings(CANARY_KEY)
    assert GeminiChatModel.from_settings(settings, client).model == settings.gemini_model
    GeminiChatModel(client, api_key=SecretStr(f"{CANARY_KEY}\n"), model=MODEL)


# ---------------------------------------------------------------------- bounded retry delays
@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("inf", MAX_RETRY_AFTER_S),
        ("1e400", MAX_RETRY_AFTER_S),
        ("86400", MAX_RETRY_AFTER_S),
        ("-5", 0.0),
        ("nan", None),
        ("12.5", 12.5),
    ],
)
@respx.mock
async def test_retry_after_header_is_finite_and_clamped(
    model: GeminiChatModel, header: str, expected: float | None
) -> None:
    respx.post(STREAM_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": header}, json={})
    )
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", USER))
    assert caught.value.retry_after_s == expected


@pytest.mark.parametrize(
    ("delay", "expected"), [("1e400s", MAX_RETRY_AFTER_S), ("nans", None), ("infs", 300.0)]
)
@respx.mock
async def test_retry_info_delay_is_finite_and_clamped(
    model: GeminiChatModel, delay: str, expected: float | None
) -> None:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]
    respx.post(STREAM_URL).mock(
        return_value=httpx.Response(429, json=google_error(429, "RESOURCE_EXHAUSTED", "x", details))
    )
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", USER))
    value = caught.value.retry_after_s
    assert value == expected
    assert value is None or math.isfinite(value)


# ---------------------------------------------------------------------- dict-like schemas
@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"tags": {"type": "object", "additionalProperties": {"type": "string"}}},
        },
        {"type": "object", "properties": {"meta": {"type": "object", "properties": {}}}},
        {"type": "array", "items": {"type": "object"}},
        {"type": "object"},
    ],
)
def test_object_without_properties_is_rejected_at_conversion(schema: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedSchema, match="properties"):
        to_gemini_schema(schema)


@respx.mock
async def test_complete_json_with_a_map_schema_fails_before_sending(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    with pytest.raises(UnsupportedSchema):
        await model.complete_json("", USER, {"type": "object", "additionalProperties": True})
    assert route.call_count == 0


@pytest.mark.parametrize("parameters", [{"type": "object"}, {}, {"properties": {}}])
@respx.mock
async def test_argumentless_tool_is_still_declared_without_parameters(
    model: GeminiChatModel, parameters: dict[str, object]
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    await collect(model.stream("", USER, [ToolSpec("list_sources", "List", parameters)]))
    assert "parameters" not in body(route)["tools"][0]["functionDeclarations"][0]


# ---------------------------------------------------------------------- error contract
def test_no_ad_hoc_retryable_flag_on_gemini_errors() -> None:
    """Retry semantics are carried by the class (see the module docstring), not by an
    attribute nothing reads and the shared ChatModelError base does not define."""
    for cls in (ModelConfigurationError, ModelRequestRejected, ChatModelError):
        assert not hasattr(cls, "retryable")


def test_base_url_fixture_is_the_mock() -> None:  # guards against a test hitting the real API
    assert BASE.startswith("https://gemini.test")
