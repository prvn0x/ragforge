"""`make mcp-key`: mint the Claude Code MCP key with your owner key and store it 0600.

The owner key is found with the same rule as `make ask` (`--owner-key-file`, else
`$RAGFORGE_KEY_FILE`, else the single `~/.config/ragforge/*.key`). The new key, named
`mcp:claude-code` with scopes read, chat and write, goes to `~/.config/ragforge/mcp/mcp.key`.

It lives in its OWN folder on purpose: `make ask`/`make ui` pick the single `*.key` directly in
`~/.config/ragforge/` and refuse to guess between two, so a second key there would break them.
The name contains `mcp`, so the API marks it an MCP key that can never hold `approvals:write`
or `admin` (`services/auth/scopes.py`). Only the key prefix and scopes are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TextIO

import httpx
from pydantic import ValidationError

from ragforge_client.ask import DEFAULT_KEY_DIR, KeyFileError, read_key, resolve_key_file
from ragforge_client.client import ApiError, RagforgeClient
from ragforge_contracts.api import ApiKeyCreated, ApiScope

KEY_NAME: Final = "mcp:claude-code"
# `write` is only for the MCP `sync_now` tool; read + chat cover ask/search (docs/handoff/MCP.md).
KEY_SCOPES: Final = (ApiScope.read, ApiScope.chat, ApiScope.write)
DEFAULT_URL: Final = "http://localhost:8000"
DEFAULT_OUT: Final = DEFAULT_KEY_DIR / "mcp" / "mcp.key"
# Read in this order. `RAGFORGE_URL` is the name the plan uses; `ragforge-ask` and
# `connect_notion.py` read `RAGFORGE_API_URL`, so one exported setting works for all of them.
URL_ENV_VARS: Final = ("RAGFORGE_URL", "RAGFORGE_API_URL")
INTERRUPTED: Final = 130  # the shell's exit code for SIGINT


class McpKeyError(Exception):
    """A failure the user can act on. Its message never holds a key."""


def api_url(env: dict[str, str]) -> str:
    """The first URL variable that is set, else the local stack."""
    return next((env[name] for name in URL_ENV_VARS if env.get(name)), DEFAULT_URL)


def check_out_path(out: Path, key_dir: Path) -> None:
    """Refuse an `--out` directly in the discovery folder.

    `make ask`/`make ui` pick the single `*.key` there and refuse two, so an MCP key written
    there would break them: the exact failure the `mcp/` subfolder exists to avoid.
    """
    if out.parent.resolve() == key_dir.expanduser().resolve():
        raise McpKeyError(
            f"--out must not be directly in {key_dir}: make ask/ui would then see two keys; "
            f"use the default {DEFAULT_OUT}"
        )


def reserve_key_file(path: Path) -> int:
    """Create the key file empty (0600) BEFORE minting.

    O_EXCL makes "refuse if it exists" and "create it" one step, and minting second means a key
    is never created on the server when it would have nowhere to go.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise McpKeyError(
            f"{path} already exists; keep using it, or revoke that key and move the file away"
        ) from None


async def mint(
    url: str, owner_key: str, transport: httpx.AsyncBaseTransport | None
) -> ApiKeyCreated:
    """One API call with the typed SDK; connection and API errors become plain messages."""
    try:
        async with RagforgeClient(url, owner_key, transport=transport) as rf:
            return await rf.create_api_key(KEY_NAME, KEY_SCOPES)
    except ApiError as exc:
        raise McpKeyError(f"the API refused: {exc}") from None
    except httpx.TransportError as exc:
        raise McpKeyError(
            f"cannot reach {url} ({type(exc).__name__}); is the stack up? make up"
        ) from None
    except ValidationError:
        # pydantic's message quotes the input, and a 201 body holds the secret: never show it.
        raise McpKeyError(
            "the API answered with a body this client does not understand; if a key was "
            "minted, revoke it (DELETE /v1/api-keys/<id>)"
        ) from None


def save(handle: TextIO, created: ApiKeyCreated) -> None:
    """Write the secret. If that fails the server still holds a live key, so say which one."""
    try:
        handle.write(created.secret + "\n")
        handle.flush()
    except (OSError, KeyboardInterrupt) as exc:
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else exc.strerror
        raise McpKeyError(
            f"key {created.prefix} was minted but not saved ({reason}); "
            f"revoke it (DELETE /v1/api-keys/{created.id})"
        ) from None


def create_mcp_key(
    owner_key_file: str | None,
    out: Path,
    url: str,
    env: dict[str, str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> ApiKeyCreated:
    """Find and read the owner key, reserve `out`, mint, write; remove `out` on any failure."""
    try:
        owner_key = read_key(resolve_key_file(owner_key_file, env, DEFAULT_KEY_DIR))
    except KeyFileError as exc:
        # ask's message names its own flag; this tool's flag is --owner-key-file.
        text = str(exc).replace("--key-file", "--owner-key-file")
        hint = " (create one with make tenant)" if "no API key found" in text else ""
        raise McpKeyError(f"owner key: {text}{hint}") from None
    check_out_path(out, DEFAULT_KEY_DIR)
    fd = reserve_key_file(out)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            created = asyncio.run(mint(url, owner_key, transport))
            save(handle, created)
    except BaseException:  # includes Ctrl-C: never leave an empty or partial key file behind
        out.unlink(missing_ok=True)
        raise
    return created


def main(
    argv: Sequence[str] | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> int:
    parser = argparse.ArgumentParser(prog="make mcp-key", description=__doc__.splitlines()[0])
    parser.add_argument("--owner-key-file", help="owner key file (default: the make ask rule)")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help=f"where to write (default {DEFAULT_OUT})"
    )
    args = parser.parse_args(argv)
    env = dict(os.environ)
    out = Path(args.out).expanduser()
    os.umask(0o077)
    try:
        created = create_mcp_key(args.owner_key_file, out, api_url(env), env, transport)
    except McpKeyError as exc:
        print(f"make mcp-key: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("make mcp-key: interrupted; no key file was written", file=sys.stderr)
        return INTERRUPTED
    scopes = ", ".join(created.scopes)
    print(f"wrote {out} (prefix {created.prefix}, scopes {scopes}; secret not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
