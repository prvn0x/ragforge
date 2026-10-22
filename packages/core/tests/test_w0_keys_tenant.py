"""W0-KEYS: `make tenant` (scripts/create_tenant.py) stores the owner key safely and never shows it.

The script runs as `make tenant` runs it: a subprocess with a temp HOME. `--compose` points at a
fake compose program that plays `docker compose exec -T api ragforge-admin create-tenant`, so
the test covers argv, the stdout parsing, the stderr filter, exit codes and file modes. No
container, no live stack.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO: Final = Path(__file__).resolve().parents[3]
SCRIPT: Final = REPO / "scripts/create_tenant.py"
# Built at runtime so this file never holds a token-shaped literal (see the gitleaks rule test).
SECRET: Final = "CANARY" + "-owner-key-" + "4b1e9d"
TENANT_ID: Final = "00000000-0000-4000-8000-00000000000a"
STORE_NOTICE: Final = "Store this key now: it cannot be shown again."

# Plays compose + admin.py. Records its argv and whether the key file already existed (the
# script reserves it before provisioning), then behaves as FAKE_MODE says.
FAKE_COMPOSE: Final = f"""
import json, os, sys
key = os.path.join(os.environ["HOME"], ".config/ragforge/owner.key")
with open(os.environ["FAKE_RECORD"], "w") as f:
    umask = os.umask(0)  # the child inherits the script's umask; record it
    json.dump({{"argv": sys.argv[1:], "reserved": os.path.exists(key), "umask": umask}}, f)
mode = os.environ["FAKE_MODE"]
if mode == "down":
    print('service "api" is not running', file=sys.stderr)
    sys.exit(1)
if mode == "admin-error":
    print("error: conflict: a tenant with that name exists", file=sys.stderr)
    sys.exit(2)
print("tenant_id={TENANT_ID}")
print("user_id=00000000-0000-4000-8000-00000000000b")
print("scopes=approvals:write,chat,read,write")
if mode != "no-key":
    print("api_key={SECRET}")
if mode == "exit-after-key":
    sys.exit(1)
if mode == "leak":
    print("debug: {SECRET}", file=sys.stderr)
print("warning: a compose warning the user should see", file=sys.stderr)
print("{STORE_NOTICE}", file=sys.stderr)
"""


@dataclass(frozen=True)
class Run:
    code: int
    out: str
    err: str
    record: dict[str, object] | None


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def key_dir(home: Path) -> Path:
    return home / ".config/ragforge"


def run_tenant(home: Path, mode: str, *extra: str) -> Run:
    fake = home.parent / "fake_compose.py"
    fake.write_text(FAKE_COMPOSE)
    record = home.parent / "record.json"
    record.unlink(missing_ok=True)
    compose = f"{sys.executable} {fake}"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT), "--owner", "Ada", *extra, "--compose", compose],
        capture_output=True,
        text=True,
        check=False,
        env={
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            "FAKE_MODE": mode,
            "FAKE_RECORD": str(record),
        },
    )
    loaded = json.loads(record.read_text()) if record.exists() else None
    return Run(proc.returncode, proc.stdout, proc.stderr, loaded)


def test_success_writes_only_the_key_with_private_modes(home: Path) -> None:
    run = run_tenant(home, "ok", "--name", "acme-demo")

    assert run.code == 0, run.err
    path = key_dir(home) / "owner.key"
    assert path.read_text() == SECRET + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_dir(home).stat().st_mode) == 0o700
    assert f"tenant_id={TENANT_ID}" in run.out
    assert str(path) in run.out


def test_the_secret_never_reaches_stdout_or_stderr(home: Path) -> None:
    run = run_tenant(home, "leak")

    assert run.code == 0, run.err
    assert SECRET not in run.out
    assert SECRET not in run.err
    assert "user_id=" not in run.out
    assert "scopes=" not in run.out


def test_stderr_is_shown_except_the_store_this_key_line(home: Path) -> None:
    run = run_tenant(home, "ok")

    assert "a compose warning the user should see" in run.err
    assert "Store this key now" not in run.err


def test_runs_the_admin_cli_in_the_api_container(home: Path) -> None:
    run = run_tenant(home, "ok", "--name", "acme-demo")

    assert run.record is not None
    assert run.record["argv"] == [
        *("exec", "-T", "api", "ragforge-admin", "create-tenant"),
        *("--name", "acme-demo", "--owner", "Ada"),
    ]


@pytest.mark.parametrize("mode", ["down", "admin-error", "no-key", "exit-after-key"])
def test_failure_removes_the_reserved_file_and_says_why(home: Path, mode: str) -> None:
    run = run_tenant(home, mode)

    assert run.code == 1
    assert run.record is not None
    assert run.record["reserved"] is True
    assert not (key_dir(home) / "owner.key").exists()
    assert "is the stack up? make up" in run.err
    assert "no key file was written" in run.err
    if mode == "down":
        assert 'service "api" is not running' in run.err  # compose's own error is not hidden


def test_missing_compose_binary_is_a_clear_error(home: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT), "--owner", "Ada", "--compose", "no-such-compose-x"],
        capture_output=True,
        text=True,
        check=False,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )

    assert proc.returncode == 1
    assert "make up" in proc.stderr
    assert not (key_dir(home) / "owner.key").exists()


@pytest.mark.parametrize("existing", ["owner.key", "work.key"])
def test_refuses_when_any_key_is_already_there(home: Path, existing: str) -> None:
    key_dir(home).mkdir(parents=True, mode=0o700)
    (key_dir(home) / existing).write_text("previous\n")

    run = run_tenant(home, "ok")

    assert run.code == 1
    assert "already have a key at" in run.err
    assert run.record is None  # refused before any tenant was created
    assert (key_dir(home) / existing).read_text() == "previous\n"
    assert sorted(p.name for p in key_dir(home).iterdir()) == [existing]


def test_keys_in_subfolders_do_not_block(home: Path) -> None:
    """The MCP key and the eval-demo key live in subfolders that discovery does not glob."""
    (key_dir(home) / "mcp").mkdir(parents=True)
    (key_dir(home) / "mcp/mcp.key").write_text("mcp\n")

    run = run_tenant(home, "ok")

    assert run.code == 0, run.err
    assert (key_dir(home) / "owner.key").exists()


# ------------------------------------------------------- in-process: the reservation itself
def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("w0_create_tenant", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reserve_is_0600_even_under_a_loose_umask(tmp_path: Path) -> None:
    """The file mode does not lean on the script's umask alone (defence in depth)."""
    previous = os.umask(0o022)
    try:
        os.close(load_script().reserve_key_file(tmp_path / "cfg/owner.key"))
    finally:
        os.umask(previous)

    assert stat.S_IMODE((tmp_path / "cfg/owner.key").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "cfg").stat().st_mode) == 0o700


def test_reserve_never_reuses_an_existing_file(tmp_path: Path) -> None:
    """O_EXCL: a run racing past the *.key check still cannot overwrite a key."""
    path = tmp_path / "owner.key"
    path.write_text("previous\n")

    with pytest.raises(FileExistsError):
        load_script().reserve_key_file(path)
    assert path.read_text() == "previous\n"


def test_admin_runs_under_umask_077(home: Path) -> None:
    """Defence in depth: anything created during the run is private, not only the key file."""
    run = run_tenant(home, "ok")

    assert run.code == 0, run.err
    assert run.record is not None
    assert run.record["umask"] == 0o077


def test_an_existing_loose_config_folder_is_tightened(home: Path) -> None:
    key_dir(home).mkdir(parents=True)
    key_dir(home).chmod(0o755)

    run = run_tenant(home, "ok")

    assert run.code == 0, run.err
    assert stat.S_IMODE(key_dir(home).stat().st_mode) == 0o700


def test_ctrl_c_gives_the_standard_message_and_exit_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()
    monkeypatch.setenv("HOME", str(tmp_path))

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "run", interrupted)
    previous = os.umask(0o022)
    try:
        code = module.main(["--owner", "Ada"])
    finally:
        os.umask(previous)

    assert code == 130
    assert "interrupted; no key file was written" in capsys.readouterr().err
    assert not (tmp_path / ".config/ragforge/owner.key").exists()
