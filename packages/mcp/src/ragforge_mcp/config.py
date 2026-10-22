"""Command line and environment configuration for `ragforge-mcp`.

Key lookup, first match wins:
  1. `--key-file PATH`;
  2. `RAGFORGE_API_KEY` (for MCP hosts that can only pass env vars; a key file is preferred,
     because hosts store env vars in their own config file);
  3. `RAGFORGE_KEY_FILE`, then the single `~/.config/ragforge/*.key`: the same rules as
     `make ask` (`ragforge_client.ask.resolve_key_file` / `read_key`, which refuse a key file
     that group or other can read).

The key is held as a `SecretStr`, so a repr, a log line or a traceback of `Config` shows
`**********`, never the key.

The directory fallback (4) is usually the OWNER's key (approvals:write, key manager), not an MCP
key: it still works, as the spec asks, but `key_warning` says so on stderr at startup.

`--http` requires a bearer token on every request (`http_auth`): read from `--http-token-file` /
`RAGFORGE_MCP_TOKEN_FILE` (same file rules as the key), else generated at startup and printed on
stderr. It binds to loopback only unless `--allow-remote` is given: the transport is plain HTTP,
so off the machine the token travels in clear text.
"""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from ragforge_client.ask import DEFAULT_KEY_DIR, KeyFileError, read_key, resolve_key_file

DEFAULT_URL = "http://localhost:8000"
DEFAULT_UI_URL = "http://localhost:8501"
MAX_COLLECTIONS = 20  # ChatRequest/QueryRequest accept at most 20 collection ids
# Real keys are ~50 chars ("<scheme>_<prefix>_<256-bit secret>"). A short value is a typo or a
# placeholder, and redacting it would rewrite ordinary words in every tool output (review r2).
MIN_KEY_LENGTH = 20


class ConfigError(Exception):
    """Bad configuration. Messages never contain the key."""


@dataclass(frozen=True)
class Config:
    api_url: str
    ui_url: str
    api_key: SecretStr
    collections: tuple[str, ...]  # names or ids; empty = every collection the key sees (max 20)
    http: bool
    host: str
    port: int
    key_warning: str | None = None  # printed on stderr by `main`; never contains the key
    http_token: SecretStr | None = None  # from --http-token-file; None = generate one (--http)


HUMAN_KEY_WARNING = (
    "using the key found in {dir} (no --key-file, RAGFORGE_API_KEY or RAGFORGE_KEY_FILE). That is "
    "usually the owner's key, which can approve actions and manage keys; MCP keys never should. "
    "Create an MCP key (its name contains 'mcp') and pass --key-file."
)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # a hostname other than localhost may resolve anywhere


def build_parser(env: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge-mcp",
        description="RAGForge MCP server: ask and search your notes from an AI agent.",
    )
    parser.add_argument(
        "--url",
        default=env.get("RAGFORGE_URL") or env.get("RAGFORGE_API_URL") or DEFAULT_URL,
        help=f"RAGForge API base URL (env RAGFORGE_URL; default {DEFAULT_URL})",
    )
    parser.add_argument(
        "--ui-url",
        default=env.get("RAGFORGE_UI_URL") or DEFAULT_UI_URL,
        help="UI base URL where a human approves pending actions (env RAGFORGE_UI_URL)",
    )
    parser.add_argument("--key-file", help="file holding the API key (mode 600; never printed)")
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        help="collection name or id to read; repeatable (default: all the key sees, max 20)",
    )
    parser.add_argument(
        "--http", action="store_true", help="serve streamable HTTP instead of stdio"
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (with --http)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (with --http; loopback only unless --allow-remote)",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback --host. Plain HTTP: the bearer token is sent unencrypted",
    )
    parser.add_argument(
        "--http-token-file",
        default=env.get("RAGFORGE_MCP_TOKEN_FILE"),
        help="file holding the bearer token HTTP clients must send (mode 600; env "
        "RAGFORGE_MCP_TOKEN_FILE). Default: a new token per start, printed on stderr",
    )
    return parser


def load_key(
    key_file: str | None, env: Mapping[str, str], key_dir: Path
) -> tuple[SecretStr, str | None]:
    """(the key, a warning when it came from the default directory). Raises `KeyFileError`
    (from `read_key`), whose message never contains the key."""
    if key_file:
        return SecretStr(read_key(Path(key_file).expanduser())), None
    inline = env.get("RAGFORGE_API_KEY", "").strip()
    if inline:
        return SecretStr(inline), None
    from_dir = not env.get("RAGFORGE_KEY_FILE")
    key = SecretStr(read_key(resolve_key_file(None, dict(env), key_dir)))
    return key, HUMAN_KEY_WARNING.format(dir=key_dir) if from_dir else None


def load_config(
    argv: Sequence[str] | None, env: Mapping[str, str], key_dir: Path = DEFAULT_KEY_DIR
) -> Config:
    args = build_parser(env).parse_args(argv)
    if len(args.collection) > MAX_COLLECTIONS:
        raise ConfigError(f"at most {MAX_COLLECTIONS} --collection values")
    if args.http and not is_loopback(args.host) and not args.allow_remote:
        raise ConfigError(
            f"--host {args.host} is not a loopback address. The endpoint is plain HTTP, so its "
            "bearer token would cross the network unencrypted: pass --allow-remote to accept that."
        )
    try:
        key, key_warning = load_key(args.key_file, env, key_dir)
    except (OSError, ValueError) as exc:
        # A directory, no permission, not UTF-8: only the type (a decode error quotes key bytes)
        raise ConfigError(
            f"cannot read the API key file ({type(exc).__name__}); point --key-file at a "
            "readable UTF-8 file holding only the key"
        ) from None
    except KeyFileError as exc:
        raise ConfigError(str(exc)) from None
    if len(key.get_secret_value()) < MIN_KEY_LENGTH:
        raise ConfigError(
            f"the API key is shorter than {MIN_KEY_LENGTH} characters, so it is not a RAGForge "
            "key (a placeholder or a truncated copy?). Create one with the SDK create_api_key."
        )
    http_token = _load_http_token(args.http_token_file) if args.http else None
    return Config(
        api_url=args.url,
        ui_url=args.ui_url,
        api_key=key,
        collections=tuple(dict.fromkeys(args.collection)),
        http=args.http,
        host=args.host,
        port=args.port,
        key_warning=key_warning,
        http_token=http_token,
    )


def _load_http_token(path: str | None) -> SecretStr | None:
    if not path:
        return None
    try:
        token = read_key(Path(path).expanduser())
    except KeyFileError as exc:
        raise ConfigError(str(exc).replace("key file", "token file")) from None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read the HTTP token file ({type(exc).__name__})") from None
    if len(token) < MIN_KEY_LENGTH:
        raise ConfigError(f"the HTTP token is shorter than {MIN_KEY_LENGTH} characters")
    return SecretStr(token)
