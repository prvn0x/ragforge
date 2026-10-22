"""NotionClient: version pinning, global rate limiting, Retry-After/backoff, pagination, and the
guarantee that the integration token never appears in logs, exceptions or reprs."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Iterator

import httpx
import pytest
import respx
import structlog

from module_c1.c1_support import (
    BASE,
    TOKEN,
    FakeLimiter,
    FakePenalizingLimiter,
    FakeSleep,
    error_body,
    load,
    paged_handler,
)
from ragforge_core.adapters.connectors.notion import (
    NOTION_VERSION,
    NotionApiError,
    NotionClient,
    NotionConnector,
    NotionNotFound,
    NotionProtocolError,
    NotionTransportError,
)
from ragforge_core.adapters.connectors.ratelimit import (
    PostgresRateLimiter,
    integration_key,
    limiter_key,
)
from ragforge_core.domain.errors import UpstreamRateLimited

PAGE = "59833787-2cf9-4fdf-8782-e53db20768a5"
DATE = {"Date": "Fri, 16 Oct 2026 12:00:00 GMT"}
KEY = limiter_key(TOKEN)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_mocked=True) as mock:
        yield mock


def _client(limiter: FakeLimiter, sleep: FakeSleep | None = None, **kw: float) -> NotionClient:
    return NotionClient(TOKEN, limiter, sleep=sleep or FakeSleep(), **kw)  # type: ignore[arg-type]


def _ok() -> httpx.Response:
    return httpx.Response(200, json=load("handbook_page.json"), headers=DATE)


def _err(status: int, **headers: str) -> httpx.Response:
    return httpx.Response(status, json=error_body(status), headers={**DATE, **headers})


async def test_pins_notion_version_sends_bearer_and_rate_limits_every_attempt(
    router: respx.MockRouter,
) -> None:
    sent: list[dict[str, str]] = []  # captured at send time: the client strips it afterwards
    replies = iter([_err(503), _ok()])

    def server(request: httpx.Request) -> httpx.Response:
        sent.append(dict(request.headers))
        return next(replies)

    router.get(f"/pages/{PAGE}").mock(side_effect=server)
    limiter = FakeLimiter()
    response = await _client(limiter).request("GET", f"/pages/{PAGE}")
    assert response.data["id"] == PAGE
    assert len(sent) == 2
    for headers in sent:
        assert headers["notion-version"] == NOTION_VERSION == "2025-09-03"
        assert headers["authorization"] == f"Bearer {TOKEN}"
    assert limiter.calls == [(KEY, 1), (KEY, 1)]  # one token per HTTP attempt


async def test_get_pagination_uses_query_cursor(router: respx.MockRouter) -> None:
    pages = load("handbook_blocks.json")[PAGE]
    route = router.get(f"/blocks/{PAGE}/children").mock(side_effect=paged_handler(pages))
    items = [b async for b in _client(FakeLimiter()).paginate("GET", f"/blocks/{PAGE}/children")]
    assert len(items) == len(pages[0]["results"]) + len(pages[1]["results"])
    first, second = (c.request.url.params for c in route.calls)
    assert first["page_size"] == "100"
    assert "start_cursor" not in first
    assert second["start_cursor"] == pages[0]["next_cursor"]


async def test_post_pagination_uses_body_cursor(router: respx.MockRouter) -> None:
    pages = load("list_all.json")["data_source_query"]
    route = router.post("/data_sources/x/query").mock(side_effect=paged_handler(pages))
    body = {"filter": {"property": "Done", "checkbox": {"equals": False}}}
    items = [
        r async for r in _client(FakeLimiter()).paginate("POST", "/data_sources/x/query", body=body)
    ]
    assert [i["id"] for i in items] == [
        "d5000000-0000-4000-8000-000000000001",
        "d5000000-0000-4000-8000-000000000002",
    ]
    sent = [json.loads(c.request.content) for c in route.calls]
    assert sent[0] == {**body, "page_size": 100}
    assert sent[1] == {**body, "page_size": 100, "start_cursor": "ds-cursor-2"}


async def test_repeated_cursor_is_a_protocol_error_not_an_infinite_loop(
    router: respx.MockRouter,
) -> None:
    loop = {"object": "list", "results": [], "has_more": True, "next_cursor": "same"}
    router.get("/blocks/x/children").mock(return_value=httpx.Response(200, json=loop))
    with pytest.raises(NotionProtocolError, match="cursor repeated"):
        [b async for b in _client(FakeLimiter()).paginate("GET", "/blocks/x/children")]


async def test_429_applies_retry_after_to_the_global_bucket_then_retries(
    router: respx.MockRouter,
) -> None:
    router.get(f"/pages/{PAGE}").mock(side_effect=[_err(429, **{"Retry-After": "2"}), _ok()])
    limiter, sleep = FakeLimiter(), FakeSleep()
    await _client(limiter, sleep).request("GET", f"/pages/{PAGE}")
    # attempt, drain 2 s worth of tokens (3/s) for EVERY worker of this integration, attempt
    assert limiter.calls == [(KEY, 1), (KEY, 6), (KEY, 1)]
    assert sleep.delays == []  # the limiter does the waiting, globally


async def test_429_retry_after_as_http_date(router: respx.MockRouter) -> None:
    retry_at = {"Retry-After": "Fri, 16 Oct 2026 12:00:05 GMT"}
    router.get(f"/pages/{PAGE}").mock(side_effect=[_err(429, **retry_at), _ok()])
    limiter = FakeLimiter()
    await _client(limiter).request("GET", f"/pages/{PAGE}")
    assert limiter.calls[1] == (KEY, 15)


async def test_long_retry_after_goes_back_to_the_queue(router: respx.MockRouter) -> None:
    route = router.get(f"/pages/{PAGE}").mock(return_value=_err(429, **{"Retry-After": "120"}))
    with pytest.raises(UpstreamRateLimited) as info:
        await _client(FakeLimiter()).request("GET", f"/pages/{PAGE}")
    assert info.value.retry_after_s == 120
    assert route.call_count == 1  # no worker slot pinned for two minutes


async def test_429_with_a_penalizing_limiter_charges_the_bucket_without_blocking(
    router: respx.MockRouter,
) -> None:
    router.get(f"/pages/{PAGE}").mock(side_effect=[_err(429, **{"Retry-After": "2"}), _ok()])
    limiter, sleep = FakePenalizingLimiter(), FakeSleep()
    await _client(limiter, sleep).request("GET", f"/pages/{PAGE}")
    assert limiter.penalties == [(KEY, 2.0)]
    assert limiter.calls == [(KEY, 1), (KEY, 1)]  # the retry's acquire pays the wait
    assert sleep.delays == []


async def test_long_retry_after_still_penalizes_every_worker_before_giving_up(
    router: respx.MockRouter,
) -> None:
    route = router.get(f"/pages/{PAGE}").mock(return_value=_err(429, **{"Retry-After": "120"}))
    limiter = FakePenalizingLimiter()
    with pytest.raises(UpstreamRateLimited) as info:
        await _client(limiter).request("GET", f"/pages/{PAGE}")
    assert info.value.retry_after_s == 120
    assert route.call_count == 1
    assert limiter.penalties == [(KEY, 120.0)]  # the shared bucket now waits ~2 minutes


async def test_persistent_429_exhausts_attempts(router: respx.MockRouter) -> None:
    route = router.get(f"/pages/{PAGE}").mock(return_value=_err(429, **{"Retry-After": "1"}))
    with pytest.raises(UpstreamRateLimited):
        await _client(FakeLimiter(), max_attempts=3).request("GET", f"/pages/{PAGE}")
    assert route.call_count == 3


async def test_5xx_backs_off_honoring_retry_after(router: respx.MockRouter) -> None:
    router.get(f"/pages/{PAGE}").mock(
        side_effect=[_err(502), _err(503, **{"Retry-After": "3"}), _err(500), _ok()]
    )
    sleep = FakeSleep()
    await _client(FakeLimiter(), sleep, backoff_base_s=1.0).request("GET", f"/pages/{PAGE}")
    assert len(sleep.delays) == 3
    assert 0.5 <= sleep.delays[0] <= 1.0  # attempt 1: base * [0.5, 1)
    assert sleep.delays[1] == 3.0  # Retry-After wins
    assert 2.0 <= sleep.delays[2] <= 4.0  # attempt 3: 4 * [0.5, 1)


async def test_5xx_exhaustion_raises_the_last_api_error(router: respx.MockRouter) -> None:
    route = router.get(f"/pages/{PAGE}").mock(return_value=_err(500))
    sleep = FakeSleep()
    with pytest.raises(NotionApiError) as info:
        await _client(FakeLimiter(), sleep, max_attempts=4).request("GET", f"/pages/{PAGE}")
    assert (info.value.http_status, info.value.code) == (500, "internal_server_error")
    assert route.call_count == 4
    assert len(sleep.delays) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_errors_are_not_retried(router: respx.MockRouter, status: int) -> None:
    route = router.get(f"/pages/{PAGE}").mock(return_value=_err(status))
    with pytest.raises(NotionApiError) as info:
        await _client(FakeLimiter()).request("GET", f"/pages/{PAGE}")
    assert route.call_count == 1
    assert info.value.http_status == status
    assert isinstance(info.value, NotionNotFound) == (status == 404)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, text="<html><body>404 Not Found</body></html>"),  # edge / proxy
        httpx.Response(404, json={"object": "error", "status": 404, "code": "invalid_request_url"}),
        httpx.Response(404, json=["not", "an", "error", "object"]),
    ],
)
async def test_only_notions_object_not_found_counts_as_not_found(
    router: respx.MockRouter, response: httpx.Response
) -> None:
    router.get(f"/pages/{PAGE}").mock(return_value=response)
    with pytest.raises(NotionApiError) as info:
        await _client(FakeLimiter()).request("GET", f"/pages/{PAGE}")
    assert info.value.http_status == 404
    assert not isinstance(info.value, NotionNotFound)


async def test_transport_errors_are_retried_then_wrapped(router: respx.MockRouter) -> None:
    route = router.get(f"/pages/{PAGE}").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(NotionTransportError) as info:
        await _client(FakeLimiter(), max_attempts=2).request("GET", f"/pages/{PAGE}")
    assert route.call_count == 2
    # The httpx exception holds the request (and its Authorization header): not reachable
    # through either link of the exception chain.
    assert info.value.__cause__ is None
    assert info.value.__context__ is None
    _assert_no_token_in_frames(info.value)


async def test_non_json_success_is_a_protocol_error(router: respx.MockRouter) -> None:
    router.get(f"/pages/{PAGE}").mock(return_value=httpx.Response(200, text="<html>"))
    with pytest.raises(NotionProtocolError):
        await _client(FakeLimiter()).request("GET", f"/pages/{PAGE}")


def test_token_is_required() -> None:
    with pytest.raises(ValueError, match="token"):
        NotionClient("  ", FakeLimiter())


# ---------------------------------------------------------------------- secret handling
async def test_token_never_appears_in_logs_exceptions_or_reprs(
    router: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    echoing_proxy = {"object": "error", "code": "bad_gateway", "message": f"upstream said {TOKEN}"}
    router.get(f"/pages/{PAGE}").mock(
        side_effect=[
            _err(429, **{"Retry-After": "1"}),
            httpx.ConnectError(f"connect failed Authorization: Bearer {TOKEN}"),
            _err(503),
            httpx.Response(502, json=echoing_proxy),
        ]
    )
    client = _client(FakeLimiter(), max_attempts=4)
    errors: list[BaseException] = []
    with structlog.testing.capture_logs() as events:
        try:
            await client.request("GET", f"/pages/{PAGE}")
        except NotionApiError as exc:
            errors.append(exc)
        router.get(f"/pages/{PAGE}").mock(side_effect=httpx.ReadTimeout(f"t {TOKEN}"))
        try:
            await _client(FakeLimiter(), max_attempts=1).request("GET", f"/pages/{PAGE}")
        except NotionTransportError as exc:
            errors.append(exc)
    assert len(errors) == 2
    assert "[REDACTED]" in str(errors[0])
    surfaces = [
        repr(client),
        repr(NotionConnector(client)),
        repr(PostgresRateLimiter(engine=None)),  # type: ignore[arg-type]
        repr(events),
        caplog.text,
        *(str(e) for e in errors),
        *(repr(e) for e in errors),
        *(repr(e.args) for e in errors),
        *("".join(traceback.format_exception(e)) for e in errors),
    ]
    for surface in surfaces:
        assert TOKEN not in surface
    assert events, "retries should have been logged"


def test_the_database_only_ever_sees_the_token_digest() -> None:
    digest = integration_key(TOKEN)
    assert len(digest) == 32
    assert isinstance(digest, bytes)
    assert TOKEN.encode() not in digest


def _assert_no_token_in_frames(exc: BaseException) -> None:
    """What an error reporter capturing locals would see: every frame of the traceback, and
    one level into each local (e.g. a `response` local's request headers)."""
    tb = exc.__traceback__
    assert tb is not None
    while tb is not None:
        for name, value in tb.tb_frame.f_locals.items():
            assert TOKEN not in repr(value), f"frame local {name!r} holds the token"
            headers = getattr(getattr(value, "request", None), "headers", None)
            if headers is not None:
                assert TOKEN not in repr(dict(headers)), f"{name!r}.request.headers"
        tb = tb.tb_next


async def test_api_errors_leave_no_token_in_traceback_locals(router: respx.MockRouter) -> None:
    """A NotionApiError is raised while `response` is a local of `request()`: its request must
    no longer carry the Authorization header."""
    router.get(f"/pages/{PAGE}").mock(return_value=_err(400))
    with pytest.raises(NotionApiError) as info:
        await _client(FakeLimiter()).request("GET", f"/pages/{PAGE}")
    assert info.value.__context__ is None
    _assert_no_token_in_frames(info.value)


async def test_successful_responses_no_longer_carry_the_authorization_header(
    router: respx.MockRouter,
) -> None:
    router.get(f"/pages/{PAGE}").mock(return_value=_ok())
    client = _client(FakeLimiter())
    response = await client._send("GET", f"/pages/{PAGE}", None, None)
    assert "authorization" not in response.request.headers
    assert TOKEN not in repr(response.request.headers)


async def test_the_limiter_port_never_sees_the_token(router: respx.MockRouter) -> None:
    """The generic RateLimiter port (and every double/wrapper of it) gets a digest key."""
    plain, penalizing = FakeLimiter(), FakePenalizingLimiter()
    for limiter in (plain, penalizing):
        router.get(f"/pages/{PAGE}").mock(side_effect=[_err(429, **{"Retry-After": "1"}), _ok()])
        await _client(limiter).request("GET", f"/pages/{PAGE}")
    recorded = [*plain.calls, *penalizing.calls, *penalizing.penalties]
    assert recorded
    assert all(key == KEY and TOKEN not in key for key, _ in recorded)
    # Same bucket row as hashing the token directly (ADR-0007: integration_key = sha256(token)).
    assert integration_key(KEY) == integration_key(TOKEN)
