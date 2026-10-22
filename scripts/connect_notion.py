"""`make connect-notion`: create a Notion connector through the local API.

The RAGForge API key comes from a file, found with the same rule as `make ask`
(`ragforge_client.ask.resolve_key_file`: `--key-file`, else `$RAGFORGE_KEY_FILE`, else the single
`~/.config/ragforge/*.key`), so after `make tenant` nothing has to be pasted. Only when no key
file is found at all does it ask for the key with getpass (hidden input). The Notion integration
token is always read with getpass. Neither ever lands in shell history, `ps` output, logs or a
chat transcript.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragforge_client.ask import DEFAULT_KEY_DIR, KeyFileError, read_key, resolve_key_file

API = os.environ.get("RAGFORGE_API_URL", "http://localhost:8000")
DEFAULT_COLLECTION = "notes"  # lowercase, digits, "-", "_" only (CollectionCreate.name)


def call(method: str, path: str, key: str, body: dict[str, object] | None = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(  # noqa: S310 - fixed local http(s) API URL
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        problem = exc.read().decode(errors="replace")[:500]
        sys.exit(f"{method} {path} failed: HTTP {exc.code} {problem}")
    except urllib.error.URLError as exc:
        sys.exit(f"{method} {path} failed: {exc.reason} (is the stack up? make up)")


def find_key_file(explicit: str | None, env: Mapping[str, str], key_dir: Path) -> Path | None:
    """The `make ask` rule, or None when nothing at all points at a key.

    None (and so the hidden prompt) only when there is no flag, no `$RAGFORGE_KEY_FILE` and no
    `*.key` in `key_dir`. A named file that is missing, or several keys, stays an error: falling
    back to a prompt there would hide a typo or silently pick a different tenant.
    """
    # This guard mirrors the three sources resolve_key_file checks, in its order, only to tell
    # "nothing points at a key" apart from a real error. ask.py has no distinct exception for that
    # case (handoff W0-KEYS asks for one). If resolve_key_file gains a source, add it here too.
    directory = key_dir.expanduser()
    if not explicit and not env.get("RAGFORGE_KEY_FILE") and not any(directory.glob("*.key")):
        return None
    return resolve_key_file(explicit, dict(env), key_dir)


def load_key(explicit: str | None, env: Mapping[str, str], key_dir: Path) -> str:
    """The API key from its file, else from a hidden prompt; exits with a message on a bad file."""
    try:
        path = find_key_file(explicit, env, key_dir)
        if path is not None:
            return read_key(path)
    except KeyFileError as exc:
        sys.exit(f"RAGForge key: {exc}")
    key = getpass.getpass("RAGForge API key (hidden; no key file found): ").strip()
    if not key:
        sys.exit("empty key")
    return key


def pick_collection(items: list[dict[str, object]]) -> dict[str, object]:
    for i, c in enumerate(items, 1):
        print(f"  {i}. {c['name']}  ({c['id']})")
    choice = input(f"Target collection [1-{len(items)}]: ").strip() or "1"
    if not choice.isdigit() or not 1 <= int(choice) <= len(items):
        sys.exit("invalid choice")
    return items[int(choice) - 1]


def create_first_collection(key: str) -> dict[str, object]:
    """A new tenant has no collection yet (`make tenant` creates none), so offer to make one.

    Without this the quickstart (`make tenant`, then `make connect-notion`) would stop here, and
    the only way on would be a hand-written request carrying the key.
    """
    print("No collections yet: creating one for the Notion pages.")
    name = input(f"Collection name [{DEFAULT_COLLECTION}]: ").strip() or DEFAULT_COLLECTION
    created = call("POST", "/v1/collections", key, {"name": name})
    if not isinstance(created, dict):
        sys.exit("POST /v1/collections returned no collection")
    print(f"Created collection {created['name']} ({created['id']}).")
    return created


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="make connect-notion", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--key-file", help="file holding the RAGForge API key (never printed)")
    args = parser.parse_args(argv)
    key = load_key(args.key_file, os.environ, DEFAULT_KEY_DIR)
    collections = call("GET", "/v1/collections", key)
    items = collections.get("items", []) if isinstance(collections, dict) else []
    collection = pick_collection(items) if items else create_first_collection(key)
    name = input("Display name [Notion]: ").strip() or "Notion"
    token = getpass.getpass("Notion integration token (hidden): ").strip()
    if not token:
        sys.exit("empty token")
    connector = call(
        "POST",
        "/v1/connectors",
        key,
        {
            "kind": "notion",
            "target_collection_id": collection["id"],
            "display_name": name,
            "token": token,
        },
    )
    assert isinstance(connector, dict)
    print(f"Connector {connector['id']} created ({connector['status']}).")
    print("The worker schedules its first sync within ~30 s. Watch: make logs SERVICE=worker")


if __name__ == "__main__":
    main()
