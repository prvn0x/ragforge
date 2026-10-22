"""HTTP metrics middleware (ADR-0018 R2-5) and the "/metrics is not on :8000" rule (R2-6).

The real `create_app` is used without its lifespan (no database): the routes exercised here
answer before touching any port (401 without a key, 404, 413 before routing)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from module_obs.obs_metrics_support import Delta
from ragforge_core.apps.api.app import create_app
from ragforge_core.apps.api.middleware import (
    MAX_BODY_BYTES,
    STARTED_AT,
    MetricsMiddleware,
)
from ragforge_core.config import Settings
from ragforge_core.observability.prompt_context import surface

HTTP = ("ragforge_http_requests_total", "ragforge_http_request_duration_seconds")
COUNT = "ragforge_http_requests_total"
SECONDS = "ragforge_http_request_duration_seconds_count"


@pytest.fixture
def app() -> FastAPI:
    application = create_app(Settings(_env_file=None))
    application.state.container = SimpleNamespace(tx=object())  # never reached: 401 first
    return application


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------- labels
async def test_the_route_label_is_the_matched_template_never_the_path(
    http: httpx.AsyncClient,
) -> None:
    delta = Delta(HTTP)
    r = await http.get("/v1/jobs/7a3d5c1e-0000-4000-8000-000000000001")
    assert r.status_code == 401
    assert delta.of(COUNT, route="/v1/jobs/{job_id}", status_class="4xx") == 1
    assert delta.of(SECONDS, route="/v1/jobs/{job_id}") == 1
    assert all("7a3d5c1e" not in str(k) for k in delta.changes())


async def test_an_unrouted_request_is_unmatched(http: httpx.AsyncClient) -> None:
    delta = Delta(HTTP)
    for path in ("/nope", "/v1/tenant-6f1c/secret-path", "/metrics"):
        assert (await http.get(path)).status_code == 404
    assert delta.of(COUNT, route="unmatched", status_class="4xx") == 3
    assert {labels["route"] for labels in delta.labels_of(COUNT)} == {"unmatched"}


async def test_a_body_refused_before_routing_is_counted_as_unmatched_413(
    http: httpx.AsyncClient,
) -> None:
    """Metrics is the OUTERMOST app middleware: BodyLimit's 413 never reaches a route."""
    delta = Delta(HTTP)
    r = await http.post(
        "/v1/chat",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert delta.of(COUNT, route="unmatched", status_class="4xx") == 1


async def test_health_probes_are_not_measured(http: httpx.AsyncClient) -> None:
    delta = Delta(HTTP)
    assert (await http.get("/healthz")).status_code == 200
    app = MetricsMiddleware(_stream_app())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/readyz")).status_code == 404  # unrouted, and still unmeasured
    assert delta.changes() == {}


# ---------------------------------------------------------------------- /metrics is internal
async def test_metrics_are_not_served_by_the_api_app(http: httpx.AsyncClient) -> None:
    assert (await http.get("/metrics")).status_code == 404
    schema = (await http.get("/openapi.json")).json()
    assert not [p for p in schema["paths"] if "metrics" in p]


def test_metrics_middleware_is_the_outermost_app_middleware(app: FastAPI) -> None:
    outermost: object = app.user_middleware[0].cls  # last added = first in the list
    assert outermost is MetricsMiddleware


# ---------------------------------------------------------------------- streaming + failures
def _stream_app(*, fail: bool = False) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/stream/{n}")
    async def stream(n: int) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"data: 1\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    @app.get("/v1/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    return app


async def test_sse_is_counted_but_kept_out_of_the_duration_histogram() -> None:
    app = MetricsMiddleware(_stream_app())
    delta = Delta(HTTP)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/v1/stream/3")).status_code == 200
    assert delta.of(COUNT, route="/v1/stream/{n}", status_class="2xx") == 1
    assert delta.of(SECONDS, route="/v1/stream/{n}") == 0


async def test_an_exception_is_a_5xx() -> None:
    app = MetricsMiddleware(_stream_app())
    delta = Delta(HTTP)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        await client.get("/v1/boom")
    assert delta.of(COUNT, route="/v1/boom", status_class="5xx") == 1


async def _drive(app: MetricsMiddleware, path: str, send: Send, receive: Receive) -> None:
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        # ASGI 2.4: Starlette streams without a disconnect listener and maps OSError on send
        # to ClientDisconnect, which is what uvicorn does when the socket is gone.
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }
    await app(scope, receive, send)


async def test_a_client_abort_mid_stream_is_disconnected_not_an_error() -> None:
    """The ASGI server raises OSError on send once the socket is gone (Starlette turns it into
    ClientDisconnect): the request is `disconnected`, never 2xx or 5xx."""
    sent: list[Message] = []

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and sent:
            raise OSError("client went away")
        sent.append(message)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    delta = Delta(HTTP)
    with pytest.raises(ClientDisconnect):
        await _drive(MetricsMiddleware(_stream_app()), "/v1/stream/1", send, receive)
    assert delta.of(COUNT, route="/v1/stream/{n}", status_class="disconnected") == 1


async def test_a_disconnect_seen_before_the_response_completed_is_disconnected() -> None:
    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await receive()  # the client is gone: the app stops without a final body

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        return None

    delta = Delta(HTTP)
    await _drive(MetricsMiddleware(inner), "/v1/anything", send, receive)
    assert delta.of(COUNT, route="unmatched", status_class="disconnected") == 1


async def test_the_request_start_and_surface_are_set_for_the_app_and_reset_after() -> None:
    seen: dict[str, object] = {}

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen["started"] = scope["state"][STARTED_AT]
        seen["surface"] = surface.get()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    async def send(message: Message) -> None:
        return None

    before = surface.get()
    await _drive(MetricsMiddleware(inner), "/v1/interview/question", send, receive)
    assert isinstance(seen["started"], float)
    assert seen["surface"] == "interview"
    assert surface.get() == before
