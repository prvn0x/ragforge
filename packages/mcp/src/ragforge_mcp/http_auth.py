"""Bearer-token gate for the streamable HTTP transport (review r2).

The process holds a RAGForge API key, so an unauthenticated HTTP endpoint lends that key to any
local process or other user who can reach the port (DNS-rebinding protection only stops
browsers). Every HTTP request must therefore carry `Authorization: Bearer <token>`, where the
token is either read from a file (stable across restarts, for an MCP host config) or generated at
startup. It is NOT the API key: leaking it exposes this endpoint, never the key itself.

Pure ASGI (no Starlette import) so it wraps whatever app the MCP SDK builds; `lifespan` passes
through untouched, because the SDK's session manager starts there.
"""

from __future__ import annotations

import hmac
import json
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

TOKEN_BYTES = 32  # 256 bits, the same strength as an API key secret


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


class BearerAuth:
    """Rejects every non-lifespan request without the right bearer token with a 401."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode()

    def __repr__(self) -> str:
        return "BearerAuth(<token>)"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or self._authorized(scope):
            await self._app(scope, receive, send)
            return
        await _unauthorized(scope, send)

    def _authorized(self, scope: Scope) -> bool:
        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        values = [value for name, value in headers if name.lower() == b"authorization"]
        # constant time: the comparison must not leak how much of a guess was right
        return len(values) == 1 and hmac.compare_digest(values[0], self._expected)


async def _unauthorized(scope: Scope, send: Send) -> None:
    if scope["type"] != "http":  # a websocket (unused by MCP): refuse the handshake
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps(
        {"error": "unauthorized", "detail": "send Authorization: Bearer <token>"}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="ragforge-mcp"'),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
