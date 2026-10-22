"""The streamable HTTP transport: every request needs the bearer token (review r2), the token
comes from a file or is generated, and the real SDK app works behind the gate. In-process ASGI
only (httpx.ASGITransport): no port is opened."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from mcp_support import KEY, make_service

from ragforge_mcp.config import ConfigError, load_config
from ragforge_mcp.http_auth import ASGIApp, BearerAuth, Message, Receive, Scope, Send, new_token
from ragforge_mcp.server import build_server, http_app, serve_http

TOKEN = "t0ken-for-tests-" + "z" * 24
BASE = "http://127.0.0.1:8765"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {"accept": "application/json, text/event-stream"}


async def _echo(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"reached the app"})


def _client(app: ASGIApp) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)


@pytest.mark.parametrize(
    "authorization",
    [None, "", f"Bearer {TOKEN}x", f"bearer {TOKEN}", TOKEN, f"Bearer {KEY}", "Basic abc"],
)
async def test_requests_without_the_exact_token_get_401_and_never_reach_the_app(
    authorization: str | None,
) -> None:
    headers = {"authorization": authorization} if authorization is not None else {}
    async with _client(BearerAuth(_echo, TOKEN)) as client:
        response = await client.post("/mcp", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert "reached the app" not in response.text
    assert TOKEN not in response.text


async def test_the_right_token_reaches_the_app_and_reprs_hide_it() -> None:
    async with _client(BearerAuth(_echo, TOKEN)) as client:
        response = await client.get("/anything", headers={"authorization": f"Bearer {TOKEN}"})
    assert response.text == "reached the app"
    assert TOKEN not in repr(BearerAuth(_echo, TOKEN))
    assert len(new_token()) >= 40
    assert new_token() != new_token()


async def test_lifespan_passes_through_and_websockets_are_refused() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def receive() -> Message:
        return {}

    gate = BearerAuth(app, TOKEN)
    await gate({"type": "lifespan"}, receive, send)
    await gate({"type": "websocket", "headers": []}, receive, send)
    assert seen == ["lifespan"]
    assert sent == [{"type": "websocket.close", "code": 1008}]


@asynccontextmanager
async def _running(app: ASGIApp) -> AsyncIterator[None]:
    """Drive the ASGI lifespan by hand (httpx.ASGITransport does not): the SDK's session
    manager starts its task group there."""
    to_app_send, to_app = anyio.create_memory_object_stream[Message](4)
    from_app_send, from_app = anyio.create_memory_object_stream[Message](4)

    async def receive() -> Message:
        return await to_app.receive()

    async def run_lifespan() -> None:
        await app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, from_app_send.send)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_lifespan)
        await to_app_send.send({"type": "lifespan.startup"})
        assert (await from_app.receive())["type"] == "lifespan.startup.complete"
        try:
            yield
        finally:
            await to_app_send.send({"type": "lifespan.shutdown"})
            assert (await from_app.receive())["type"] == "lifespan.shutdown.complete"


async def test_the_real_sdk_app_serves_mcp_behind_the_gate(tmp_path: Path) -> None:
    config = load_config(["--http"], {"RAGFORGE_API_KEY": KEY}, tmp_path)
    service, rf = make_service()
    app = http_app(build_server(service), config, TOKEN)
    async with rf, _running(app), _client(app) as client:
        refused = await client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        headers = MCP_HEADERS | {"authorization": f"Bearer {TOKEN}"}
        accepted = await client.post("/mcp", json=INITIALIZE, headers=headers)
    assert refused.status_code == 401
    assert accepted.status_code == 200, accepted.text
    assert '"serverInfo"' in accepted.text
    assert "ragforge" in accepted.text
    assert KEY not in accepted.text


def _write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text + "\n")
    path.chmod(mode)
    return path


def test_http_token_comes_from_a_private_file_or_is_generated(tmp_path: Path) -> None:
    env: dict[str, Any] = {"RAGFORGE_API_KEY": KEY}
    token_file = _write(tmp_path / "mcp.token", TOKEN)
    from_flag = load_config(["--http", "--http-token-file", str(token_file)], env, tmp_path)
    assert from_flag.http_token is not None
    assert from_flag.http_token.get_secret_value() == TOKEN
    assert TOKEN not in repr(from_flag)
    from_env = load_config(["--http"], env | {"RAGFORGE_MCP_TOKEN_FILE": str(token_file)})
    assert from_env.http_token is not None
    assert load_config(["--http"], env, tmp_path).http_token is None  # generated at startup
    shared = _write(tmp_path / "shared.token", TOKEN, mode=0o644)
    with pytest.raises(ConfigError, match=r"token file .* readable by others"):
        load_config(["--http", "--http-token-file", str(shared)], env, tmp_path)
    short = _write(tmp_path / "short.token", "abc")
    with pytest.raises(ConfigError, match="shorter than 20"):
        load_config(["--http", "--http-token-file", str(short)], env, tmp_path)


async def test_serve_http_prints_a_generated_token_on_stderr_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    served: list[object] = []

    async def fake_serve(self: uvicorn.Server, sockets: object = None) -> None:
        served.append(self.config.app)

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    service, rf = make_service()
    async with rf:
        generated = load_config(["--http"], {"RAGFORGE_API_KEY": KEY}, tmp_path)
        await serve_http(build_server(service), generated)
        token_file = _write(tmp_path / "mcp.token", TOKEN)
        stable = load_config(
            ["--http", "--http-token-file", str(token_file)], {"RAGFORGE_API_KEY": KEY}, tmp_path
        )
        await serve_http(build_server(service), stable)
    out, err = capsys.readouterr()
    assert out == ""  # never on stdout
    assert err.count("HTTP bearer token for this run") == 1  # only when generated
    assert TOKEN not in err
    assert KEY not in err
    assert all(isinstance(app, BearerAuth) for app in served)
    assert len(served) == 2
