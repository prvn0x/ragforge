"""Where the UI finds the API and its key.

The key file rules are the terminal client's (`ragforge_client.ask`), so one key file works for
both: `--key-file` > `$RAGFORGE_KEY_FILE` > the single `~/.config/ragforge/*.key`, and a file
other users can read is refused. Pass flags after `--`:
`streamlit run app.py --server.address 127.0.0.1 -- --key-file ~/.config/ragforge/ui.key`
(the app refuses to load a key when it is reachable from the network: `exposure_refused`).

The key is held as a `SecretStr`, so a repr, a log line or Streamlit's exception view shows
`**********`, never the value.
"""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import SecretStr

from ragforge_client.ask import DEFAULT_KEY_DIR, read_key, resolve_key_file

DEFAULT_URL = "http://localhost:8000"
# Opt-in for a UI reachable beyond this machine, for when something else already guards it
# (a container whose host-side port is bound to 127.0.0.1, an authenticating reverse proxy).
ALLOW_REMOTE_ENV = "RAGFORGE_UI_ALLOW_REMOTE"


@dataclass(frozen=True, slots=True)
class UiConfig:
    api_url: str
    api_key: SecretStr


def load_config(
    argv: Sequence[str], env: Mapping[str, str], key_dir: Path = DEFAULT_KEY_DIR
) -> UiConfig:
    """Read the config. Raises `ragforge_client.ask.KeyFileError` (its message names the file,
    never the key) when no usable key file is found."""
    parser = argparse.ArgumentParser(prog="ragforge-ui", add_help=False)
    parser.add_argument("--url")
    parser.add_argument("--key-file")
    # parse_known_args: Streamlit (and pytest under AppTest) leave their own flags in argv.
    args, _ = parser.parse_known_args(list(argv))
    url = args.url or env.get("RAGFORGE_URL") or env.get("RAGFORGE_API_URL") or DEFAULT_URL
    key = read_key(resolve_key_file(args.key_file, dict(env), key_dir))
    return UiConfig(api_url=url.rstrip("/"), api_key=SecretStr(key))


def display_url(url: str) -> str:
    """The URL without user info, query or fragment: safe to show on the page.

    A URL like `http://user:token@host` would otherwise put a credential on the screen."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal: urlsplit dropped its brackets
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def is_loopback(address: str | None) -> bool:
    """True when Streamlit's `server.address` keeps the server on this machine.

    Unset (Streamlit's default) means EVERY interface, so it is not loopback."""
    host = (address or "").strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # empty, a hostname, or garbage: can't prove it's local
        return False


def exposure_refused(address: str | None, env: Mapping[str, str]) -> bool:
    """Refuse to load the key when the page is reachable from the network.

    Why fail closed in code: the page has no login of its own and acts with the owner's key,
    which holds `approvals:write`. Bound to every interface, anyone on the same Wi-Fi could read
    the notes and approve destructive actions. A Makefile flag alone is one forgotten argument
    away from that, so the app checks its own bind address."""
    return not is_loopback(address) and env.get(ALLOW_REMOTE_ENV) != "1"


def host_refused(host_header: str | None, env: Mapping[str, str]) -> bool:
    """Refuse a request whose Host is not this machine (DNS rebinding, review r3 P1).

    A loopback bind does not stop a malicious page the owner visits from rebinding its own
    domain to 127.0.0.1:8501: the browser then sends `Host: evil.example` and Streamlit
    (allowedHosts empty by default) would serve it a session that loads the owner's key. So the
    app checks the Host of the session itself. No Host at all (no browser request, e.g. tests)
    is not an attack a browser can mount. `RAGFORGE_UI_ALLOW_REMOTE=1` opts out, exactly as for
    the bind address."""
    if host_header is None or env.get(ALLOW_REMOTE_ENV) == "1":
        return False
    host = host_header.strip()
    if host.startswith("["):  # [::1]:8501
        host = host[1 : host.find("]")] if "]" in host else host
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return not is_loopback(host)


def origin_refused(origin: str | None, host: str | None, env: Mapping[str, str]) -> bool:
    """Refuse a session opened by ANOTHER web page on this machine (integrated review P2).

    Streamlit accepts any `localhost`/`127.0.0.1` origin on ANY port, so a local dev server or
    a compromised local app could drive this page with the owner's key. A browser always sends
    `Origin` on the WebSocket; it must be this page's own scheme-less host:port. No Origin (no
    browser request, e.g. tests) is not an attack a page can mount; the remote opt-in skips it."""
    if origin is None or host is None or env.get(ALLOW_REMOTE_ENV) == "1":
        return False
    netloc = urlsplit(origin.strip()).netloc
    return netloc.lower() != host.strip().lower()
