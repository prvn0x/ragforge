"""`make tenant`: create your tenant and store its owner key in a 0600 file, never on screen.

No API key exists before the first tenant, so the only way in is the operator CLI
(`ragforge-admin create-tenant`) inside the running api container. That CLI prints the key once
on stdout. This script keeps that stdout to itself, writes only the `api_key=` value to
`~/.config/ragforge/owner.key` (folder 0700, file 0600), and prints the tenant id and the path.

It refuses when any `*.key` is already in that folder: `make ask`, `make ui` and the MCP server
pick the single `*.key` there and refuse to guess between two
(`ragforge_client.ask.resolve_key_file`), so a second key would break them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TextIO

from ragforge_client.ask import DEFAULT_KEY_DIR

KEY_FILE_NAME: Final = "owner.key"
# admin.py's stderr reminder. Dropped because the key went to a file, so "store it" would mislead.
STORE_NOTICE: Final = "Store this key now"
STACK_HINT: Final = "is the stack up? make up"
# Not `ragforge-admin create-key`: it prints the key on the terminal, and it cannot grant
# `approvals:write`, so its key could not mint the MCP key (make mcp-key would get a 403).
RECOVERY_HINT: Final = (
    "On a dev stack, start clean: make nuke && make up && make tenant (this deletes all data)"
)
INTERRUPTED: Final = 130  # the shell's exit code for SIGINT


class TenantError(Exception):
    """A failure the user can act on. Its message never holds the key."""


def existing_keys(key_dir: Path) -> list[Path]:
    """The `*.key` files the single-key discovery rule would see (not subfolders)."""
    return sorted(key_dir.glob("*.key")) if key_dir.is_dir() else []


def reserve_key_file(path: Path) -> int:
    """Create the key file empty (0600) BEFORE the tenant exists.

    O_EXCL makes "check it is absent" and "create it" one step, so two runs cannot both pass,
    and a tenant is never created whose key would have nowhere to go.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)  # an older, looser folder is tightened: it holds tenant keys
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def admin_command(compose: str, name: str, owner: str) -> list[str]:
    """The same call as the `eval-demo` target, as an argv list (no shell, no quoting bugs)."""
    return [
        *shlex.split(compose),
        *("exec", "-T", "api", "ragforge-admin", "create-tenant"),
        *("--name", name, "--owner", owner),
    ]


def parse_fields(stdout: str) -> dict[str, str]:
    """admin.py's stdout is `key=value` lines (tenant_id, user_id, scopes, api_key)."""
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            fields[name.strip()] = value.strip()
    return fields


def relay_stderr(stderr: str, secret: str | None, out: TextIO) -> None:
    """Show admin.py's (and compose's) errors, minus the "store this key" reminder.

    A line holding the secret is dropped too. admin.py never prints it there; this keeps a
    future change to admin.py from leaking it through this script.
    """
    for line in stderr.splitlines():
        if line.startswith(STORE_NOTICE) or (secret and secret in line):
            continue
        print(line, file=out)


def provision(command: Sequence[str], handle: TextIO, err: TextIO) -> str:
    """Run the admin CLI, write the key to `handle`, return the tenant id."""
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError as exc:
        raise TenantError(f"cannot run {command[0]}: {exc.strerror} ({STACK_HINT})") from None
    fields = parse_fields(proc.stdout)
    secret = fields.get("api_key") or None
    relay_stderr(proc.stderr, secret, err)
    if proc.returncode != 0:
        raise TenantError(f"ragforge-admin failed with exit code {proc.returncode} ({STACK_HINT})")
    if secret is None:
        raise TenantError(f"ragforge-admin printed no api_key ({STACK_HINT})")
    tenant_id = fields.get("tenant_id", "")
    try:
        handle.write(secret + "\n")
        handle.flush()
    except OSError as exc:
        raise TenantError(
            f"tenant {tenant_id} was created but its key could not be saved ({exc.strerror}). "
            f"{RECOVERY_HINT}"
        ) from None
    return tenant_id


def create_tenant(name: str, owner: str, compose: str, key_dir: Path, err: TextIO) -> Path:
    """Refuse on any existing key, reserve the file, provision, and remove the file on failure."""
    found = existing_keys(key_dir)
    if found:
        raise TenantError(
            f"already have a key at {found[0]}; make ask/ui use the single *.key in {key_dir}, "
            "so a second one would break them. Keep using it, or move it away first"
        )
    path = key_dir / KEY_FILE_NAME
    try:
        fd = reserve_key_file(path)
    except FileExistsError:
        raise TenantError(f"already have a key at {path}") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            tenant_id = provision(admin_command(compose, name, owner), handle, err)
    except BaseException:  # includes Ctrl-C: never leave an empty or partial key file behind
        path.unlink(missing_ok=True)
        raise
    print(f"tenant_id={tenant_id}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make tenant", description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="personal", help="tenant name (default: personal)")
    parser.add_argument("--owner", required=True, help="owner display name")
    parser.add_argument("--compose", default="docker compose", help="compose command prefix")
    args = parser.parse_args(argv)
    os.umask(0o077)  # every folder and file created from here on is private to this user
    try:
        path = create_tenant(
            args.name, args.owner, args.compose, DEFAULT_KEY_DIR.expanduser(), sys.stderr
        )
    except TenantError as exc:
        print(f"make tenant: {exc}; no key file was written", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # The tenant may or may not exist now; the reserved file is already removed.
        print("make tenant: interrupted; no key file was written", file=sys.stderr)
        return INTERRUPTED
    print(f"owner key written to {path} (mode 600, not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
