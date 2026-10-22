"""HTTP error mapping and the key/prompt secrecy guarantees of the Gemini adapter."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from module_g1.g1_support import (
    CANARY_KEY,
    MODEL,
    STREAM_URL,
    answered,
    body,
    collect,
    finish_chunk,
    google_error,
    ok,
    text_chunk,
)
from ragforge_core.adapters.llm.chat_gemini import (
    API_KEY_HEADER,
    GeminiChatModel,
    ModelConfigurationError,
    ModelRequestRejected,
    StreamInterrupted,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError
from ragforge_core.domain.errors import UpstreamRateLimited, ValidationFailed
from ragforge_core.domain.models import ChatMessage, Role

PROMPT_CANARY = "PROMPT-CANARY-hunter2-private-note"
SECRET_USER = (ChatMessage(Role.user, PROMPT_CANARY),)


# ---------------------------------------------------------------------- rate limits
@pytest.mark.parametrize("status", [429, 503])
@respx.mock
async def test_rate_limited_honours_retry_after_seconds(
    model: GeminiChatModel, status: int
) -> None:
    respx.post(STREAM_URL).mock(
        return_value=httpx.Response(
            status, headers={"retry-after": "7"}, json=google_error(status, "RESOURCE_EXHAUSTED")
        )
    )
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", SECRET_USER))
    assert caught.value.retry_after_s == 7.0


@respx.mock
async def test_retry_after_http_date_is_honoured(model: GeminiChatModel) -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
    respx.post(STREAM_URL).mock(return_value=httpx.Response(429, headers={"retry-after": when}))
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", SECRET_USER))
    assert caught.value.retry_after_s is not None
    assert 100 < caught.value.retry_after_s <= 120


@respx.mock
async def test_retry_info_in_body_is_used_without_header(model: GeminiChatModel) -> None:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]
    respx.post(STREAM_URL).mock(
        return_value=httpx.Response(429, json=google_error(429, "RESOURCE_EXHAUSTED", "q", details))
    )
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", SECRET_USER))
    assert caught.value.retry_after_s == 17.0


@respx.mock
async def test_rate_limited_without_any_hint(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=httpx.Response(503, text="<html>busy</html>"))
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", SECRET_USER))
    assert caught.value.retry_after_s is None


@respx.mock
async def test_rate_limit_reported_mid_stream(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([google_error(429, "RESOURCE_EXHAUSTED")]))
    with pytest.raises(UpstreamRateLimited):  # nothing emitted yet: safe to retry
        await collect(model.stream("", SECRET_USER))
    respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk("a"), google_error(429, "RESOURCE_EXHAUSTED")])
    )
    with pytest.raises(StreamInterrupted) as caught:  # a retry would duplicate "a"
        await collect(model.stream("", SECRET_USER))
    assert caught.value.cause == "rate_limited"
    assert not isinstance(caught.value, UpstreamRateLimited)


# ---------------------------------------------------------------------- permanent failures
@respx.mock
async def test_invalid_request_is_a_permanent_validation_failure(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(
        return_value=httpx.Response(400, json=google_error(400, "INVALID_ARGUMENT", PROMPT_CANARY))
    )
    with pytest.raises(ModelRequestRejected) as caught:
        await collect(model.stream("", SECRET_USER))
    assert isinstance(caught.value, ValidationFailed)
    assert isinstance(caught.value, ChatModelError)
    assert str(caught.value) == "model provider rejected the request: INVALID_ARGUMENT"


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, google_error(401, "UNAUTHENTICATED")),
        (403, google_error(403, "PERMISSION_DENIED")),
        (404, google_error(404, "NOT_FOUND")),
        # Gemini answers a bad key with 400 + ErrorInfo.reason API_KEY_INVALID, not 401.
        (
            400,
            google_error(
                400,
                "INVALID_ARGUMENT",
                "API key not valid",
                [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                    }
                ],
            ),
        ),
        (400, google_error(400, "FAILED_PRECONDITION", "User location is not supported")),
    ],
)
@respx.mock
async def test_configuration_problems_are_not_retryable(
    model: GeminiChatModel, status: int, error: dict[str, Any]
) -> None:
    respx.post(STREAM_URL).mock(return_value=httpx.Response(status, json=error))
    with pytest.raises(ModelConfigurationError) as caught:
        await collect(model.stream("", SECRET_USER))
    assert not isinstance(caught.value, UpstreamRateLimited)


@respx.mock
async def test_thinking_mention_without_thinking_sent_is_not_retried(
    client: httpx.AsyncClient,
) -> None:
    no_thinking = GeminiChatModel(
        client, api_key=SecretStr(CANARY_KEY), model=MODEL, thinking_budget=None
    )
    route = respx.post(STREAM_URL).mock(
        return_value=httpx.Response(400, json=google_error(400, "INVALID_ARGUMENT", "thinking"))
    )
    with pytest.raises(ModelRequestRejected):
        await collect(no_thinking.stream("", SECRET_USER))
    assert route.call_count == 1
    assert "thinkingConfig" not in body(route)["generationConfig"]


@pytest.mark.parametrize("status", [500, 502, 504])
@respx.mock
async def test_server_errors_are_generic_provider_errors(
    model: GeminiChatModel, status: int
) -> None:
    respx.post(STREAM_URL).mock(return_value=httpx.Response(status, text=PROMPT_CANARY))
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", SECRET_USER))
    assert str(caught.value) == f"model provider returned HTTP {status}: HTTP {status}"


@respx.mock
async def test_redirect_is_never_followed(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        return_value=httpx.Response(307, headers={"location": "https://evil.test/steal"})
    )
    evil = respx.post("https://evil.test/steal").mock(return_value=answered())
    with pytest.raises(ChatModelError, match="HTTP 307"):
        await collect(model.stream("", SECRET_USER))
    assert route.call_count == 1
    assert not evil.called


# ---------------------------------------------------------------------- secrecy
@respx.mock
async def test_key_only_in_header_never_in_url(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    await collect(model.stream("", SECRET_USER))
    request = route.calls.last.request
    assert request.headers[API_KEY_HEADER] == CANARY_KEY
    assert CANARY_KEY not in str(request.url)
    assert "key" not in request.url.params
    assert CANARY_KEY not in request.content.decode()
    assert "authorization" not in request.headers


ERROR_RESPONSES = [
    httpx.Response(
        400, json=google_error(400, "INVALID_ARGUMENT", f"{PROMPT_CANARY} {CANARY_KEY}")
    ),
    httpx.Response(403, json=google_error(403, "PERMISSION_DENIED", f"key {CANARY_KEY}")),
    httpx.Response(500, text=f"{PROMPT_CANARY} {CANARY_KEY}"),
    httpx.Response(429, json=google_error(429, "RESOURCE_EXHAUSTED", PROMPT_CANARY)),
    ok([google_error(500, "INTERNAL", f"{PROMPT_CANARY} {CANARY_KEY}")]),
]


@pytest.mark.parametrize("response", ERROR_RESPONSES)
@respx.mock
async def test_key_and_prompt_never_in_exceptions_or_logs(
    model: GeminiChatModel, response: httpx.Response, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(STREAM_URL).mock(return_value=response)
    caplog.set_level(logging.DEBUG)  # httpx/httpcore included: they log every request
    with pytest.raises((ChatModelError, UpstreamRateLimited)) as caught:
        await collect(model.stream("system " + PROMPT_CANARY, SECRET_USER))
    rendered = [str(caught.value), repr(caught.value), *map(str, caught.value.args)]
    cause = caught.value.__cause__ or caught.value.__context__
    if cause is not None:
        rendered.append(repr(cause))
    rendered.append(caplog.text)
    for text in rendered:
        assert CANARY_KEY not in text
        assert PROMPT_CANARY not in text


@respx.mock
async def test_successful_call_logs_nothing_secret(
    model: GeminiChatModel, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk(PROMPT_CANARY), finish_chunk()]))
    caplog.set_level(logging.DEBUG)
    await collect(model.stream(PROMPT_CANARY, SECRET_USER))
    assert CANARY_KEY not in caplog.text
    assert PROMPT_CANARY not in caplog.text


def test_repr_hides_the_key(model: GeminiChatModel) -> None:
    assert CANARY_KEY not in repr(model)
    assert CANARY_KEY not in str(vars(model) if hasattr(model, "__dict__") else "")


@respx.mock
async def test_huge_error_body_is_read_bounded(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=httpx.Response(500, content=b"x" * 5_000_000))
    with pytest.raises(ChatModelError, match="HTTP 500"):
        await collect(model.stream("", SECRET_USER))
