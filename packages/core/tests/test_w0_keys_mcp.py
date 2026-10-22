"""W0-KEYS: `make mcp-key` (scripts/create_mcp_key.py) and the `make mcp` / `mcp-install` targets.

The script runs in-process against the real typed SDK with an httpx MockTransport standing in for
the API, so the request it sends (name, scopes, which owner key) is what is asserted. The
Makefile targets are checked with `make -n` (prints the recipe, runs nothing). Temp HOME only.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shlex
import shutil
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import IO, Any, Final

import httpx
import pytest

REPO: Final = Path(__file__).resolve().parents[3]
# Built at runtime so this file never holds a token-shaped literal (see the gitleaks rule test).
MCP_SECRET: Final = "CANARY" + "-mcp-key-" + "e07c21"
OWNER: Final = {name: "CANARY-owner-" + name for name in ("flag", "env", "dir")}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "w0_create_mcp_key", REPO / "scripts/create_mcp_key.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


class FakeApi:
    """Records each request; answers like POST /v1/api-keys (or with `status` on failure)."""

    def __init__(
        self, status: int = 201, error: Exception | None = None, malformed: bool = False
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status
        self.error = error
        self.malformed = malformed
        self.umask_seen: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.umask_seen = os.umask(0o077)  # read the umask in force while the key is minted
        os.umask(self.umask_seen)
        if self.malformed:  # a 201 the SDK cannot parse, still holding the secret
            return httpx.Response(201, json={"secret": MCP_SECRET, "prefix": 7})
        if self.error is not None:
            raise self.error
        if self.status != 201:
            problem = {"title": "Forbidden", "status": self.status, "detail": "no key manager"}
            return httpx.Response(self.status, json=problem)
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "00000000-0000-4000-8000-0000000000c1",
                "name": body["name"],
                "prefix": "rf_0000demo",
                "scopes": body["scopes"],
                "created_at": "2026-10-22T00:00:00Z",
                "last_used_at": None,
                "revoked_at": None,
                "secret": MCP_SECRET,
            },
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def bearer(self) -> str:
        return self.requests[-1].headers["authorization"].removeprefix("Bearer ")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAGFORGE_KEY_FILE", raising=False)
    monkeypatch.delenv("RAGFORGE_URL", raising=False)
    monkeypatch.delenv("RAGFORGE_API_URL", raising=False)
    previous = os.umask(0o022)  # the script sets 077; restore so other tests see the default
    yield tmp_path
    os.umask(previous)


def write_key(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(value + "\n")
    path.chmod(0o600)
    return path


def mcp_key(home: Path) -> Path:
    return home / ".config/ragforge/mcp/mcp.key"


def test_mints_the_mcp_key_and_stores_it_privately(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    api = FakeApi()

    assert script.main([], transport=api.transport) == 0

    request = api.requests[-1]
    assert (request.method, request.url.path) == ("POST", "/v1/api-keys")
    assert json.loads(request.content) == {
        "name": "mcp:claude-code",
        "scopes": ["read", "chat", "write"],
    }
    assert api.bearer() == OWNER["dir"]
    assert mcp_key(home).read_text() == MCP_SECRET + "\n"
    assert stat.S_IMODE(mcp_key(home).stat().st_mode) == 0o600
    assert stat.S_IMODE(mcp_key(home).parent.stat().st_mode) == 0o700
    out, err = capsys.readouterr()
    assert MCP_SECRET not in out + err
    assert "rf_0000demo" in out
    assert "read, chat, write" in out


def test_owner_key_discovery_order_flag_then_env_then_dir(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    env_file = write_key(home / "elsewhere/env.key", OWNER["env"])
    flag_file = write_key(home / "elsewhere/flag.key", OWNER["flag"])
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(env_file))

    api = FakeApi()
    assert script.main(["--owner-key-file", str(flag_file)], transport=api.transport) == 0
    assert api.bearer() == OWNER["flag"]

    mcp_key(home).unlink()
    assert script.main([], transport=api.transport) == 0
    assert api.bearer() == OWNER["env"]

    mcp_key(home).unlink()
    monkeypatch.delenv("RAGFORGE_KEY_FILE")
    assert script.main([], transport=api.transport) == 0
    assert api.bearer() == OWNER["dir"]


def test_refuses_when_the_mcp_key_exists_and_mints_nothing(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    write_key(mcp_key(home), "previous")
    api = FakeApi()

    assert script.main([], transport=api.transport) == 1

    assert api.requests == []  # refused before any key was minted on the server
    assert mcp_key(home).read_text() == "previous\n"
    assert "already exists" in capsys.readouterr().err


def test_no_owner_key_points_at_make_tenant(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi()

    assert script.main([], transport=api.transport) == 1

    assert api.requests == []
    assert "make tenant" in capsys.readouterr().err
    assert not mcp_key(home).exists()


@pytest.mark.parametrize(
    ("api", "message"),
    [
        (FakeApi(status=403), "the API refused"),
        (FakeApi(error=httpx.ConnectError("refused")), "is the stack up? make up"),
    ],
)
def test_failure_removes_the_reserved_file(
    home: Path, capsys: pytest.CaptureFixture[str], api: FakeApi, message: str
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])

    assert script.main([], transport=api.transport) == 1

    assert api.requests  # the file was reserved, then the call failed
    assert not mcp_key(home).exists()
    assert message in capsys.readouterr().err


def test_ragforge_url_overrides_the_default(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    monkeypatch.setenv("RAGFORGE_URL", "http://ragforge.test:9001")
    api = FakeApi()

    assert script.main([], transport=api.transport) == 0

    assert (api.requests[-1].url.host, api.requests[-1].url.port) == ("ragforge.test", 9001)


def test_default_url_is_localhost_8000(home: Path) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    api = FakeApi()

    assert script.main([], transport=api.transport) == 0

    assert str(api.requests[-1].url).startswith("http://localhost:8000/")


def test_the_key_is_minted_under_umask_077(home: Path) -> None:
    """Defence in depth: anything created during the run is private, not only the key file."""
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    api = FakeApi()

    assert script.main([], transport=api.transport) == 0

    assert api.umask_seen == 0o077


def test_reserve_is_0600_even_under_a_loose_umask(tmp_path: Path) -> None:
    """The file mode does not lean on main's umask alone."""
    previous = os.umask(0o022)
    try:
        os.close(script.reserve_key_file(tmp_path / "cfg/mcp.key"))
    finally:
        os.umask(previous)

    assert stat.S_IMODE((tmp_path / "cfg/mcp.key").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "cfg").stat().st_mode) == 0o700


def test_an_existing_loose_mcp_folder_is_tightened(home: Path) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    mcp_key(home).parent.mkdir(parents=True)
    mcp_key(home).parent.chmod(0o755)

    assert script.main([], transport=FakeApi().transport) == 0

    assert stat.S_IMODE(mcp_key(home).parent.stat().st_mode) == 0o700


def test_refuses_an_out_path_in_the_discovery_folder(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    owner = write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    api = FakeApi()

    assert script.main(["--out", "~/.config/ragforge/mcp.key"], transport=api.transport) == 1

    assert api.requests == []  # nothing minted
    assert sorted(owner.parent.glob("*.key")) == [owner]  # make ask/ui still see one key
    assert "make ask/ui would then see two keys" in capsys.readouterr().err


def test_ragforge_api_url_is_accepted_and_ragforge_url_wins(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    monkeypatch.setenv("RAGFORGE_API_URL", "http://api-url.test:9002")
    api = FakeApi()

    assert script.main([], transport=api.transport) == 0
    assert api.requests[-1].url.host == "api-url.test"

    mcp_key(home).unlink()
    monkeypatch.setenv("RAGFORGE_URL", "http://url.test:9001")
    assert script.main([], transport=api.transport) == 0
    assert api.requests[-1].url.host == "url.test"


class FullDisk:
    """A key file whose write fails like a full disk; closing still closes the real file."""

    def __init__(self, real: IO[str]) -> None:
        self.real = real

    def __enter__(self) -> FullDisk:
        return self

    def __exit__(self, *exc: object) -> None:
        self.real.close()

    def write(self, _text: str) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self) -> None:
        self.real.flush()


def test_minted_but_not_saved_names_the_key_to_revoke(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])
    real_fdopen = os.fdopen

    def fdopen(fd: int, *args: Any, **kwargs: Any) -> FullDisk:
        return FullDisk(real_fdopen(fd, *args, **kwargs))

    monkeypatch.setattr(script.os, "fdopen", fdopen)

    assert script.main([], transport=FakeApi().transport) == 1

    out, err = capsys.readouterr()
    assert "key rf_0000demo was minted but not saved (No space left on device)" in err
    assert "DELETE /v1/api-keys/00000000-0000-4000-8000-0000000000c1" in err
    assert MCP_SECRET not in out + err
    assert not mcp_key(home).exists()


def test_a_malformed_reply_never_echoes_the_secret(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])

    assert script.main([], transport=FakeApi(malformed=True).transport) == 1

    out, err = capsys.readouterr()
    assert MCP_SECRET not in out + err
    assert "does not understand" in err
    assert not mcp_key(home).exists()


def test_ctrl_c_gives_the_standard_message_and_exit_130(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(home / ".config/ragforge/owner.key", OWNER["dir"])

    def interrupted(coro: Any) -> None:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(script.asyncio, "run", interrupted)

    assert script.main([], transport=FakeApi().transport) == 130

    assert "interrupted; no key file was written" in capsys.readouterr().err
    assert not mcp_key(home).exists()


# ------------------------------------------------------------------ Makefile (make -n)
def dry_run(home: Path, *args: str) -> str:
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is not installed")
    proc = subprocess.run(  # noqa: S603
        [make, "--no-print-directory", "-C", str(REPO), "-n", *args],
        capture_output=True,
        text=True,
        check=True,
        env={"HOME": str(home), "PATH": os.environ.get("PATH", ""), "USER": "ada-test"},
    )
    return proc.stdout


def test_make_mcp_defaults_to_the_mcp_key(home: Path) -> None:
    assert dry_run(home, "mcp").split() == [
        *("uv", "run", "ragforge-mcp", "--key-file"),
        f'"{home}/.config/ragforge/mcp/mcp.key"',
    ]


def test_make_mcp_args_replace_the_default(home: Path) -> None:
    out = dry_run(home, "mcp", "ARGS=--key-file /elsewhere/x.key")

    assert out.split() == ["uv", "run", "ragforge-mcp", "--key-file", "/elsewhere/x.key"]


def test_make_mcp_install_uses_the_new_key_and_the_guide(home: Path) -> None:
    out = dry_run(home, "mcp-install")

    assert "make mcp-key" in out
    assert "docs/GUIDE.md" in out
    assert "docs/handoff/MCP.md" not in out


def test_make_mcp_install_prints_a_command_that_survives_spaces(tmp_path: Path) -> None:
    """Runs the target for real (it only echoes) with paths holding a space, then splits the
    printed line the way a shell would: both paths must come back whole."""
    home = tmp_path / "home with space"
    home.mkdir()
    checkout = str(tmp_path / "check out" / "ragforge")
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is not installed")
    proc = subprocess.run(  # noqa: S603
        # CURDIR is overridden too: the real checkout path may have no space to break on.
        [make, "--no-print-directory", "-s", "-C", str(REPO), "mcp-install", f"CURDIR={checkout}"],
        capture_output=True,
        text=True,
        check=True,
        env={"HOME": str(home), "PATH": os.environ.get("PATH", "")},
    )
    line = next(x for x in proc.stdout.splitlines() if x.startswith("claude mcp add"))
    argv = shlex.split(line)

    assert argv[argv.index("--directory") + 1] == checkout
    assert argv[argv.index("--key-file") + 1] == f"{home}/.config/ragforge/mcp/mcp.key"


def test_make_key_targets_run_the_scripts(home: Path) -> None:
    assert dry_run(home, "mcp-key").split() == ["uv", "run", "python", "scripts/create_mcp_key.py"]
    assert dry_run(home, "connect-notion").split() == [
        *("uv", "run", "python", "scripts/connect_notion.py"),
    ]
    assert dry_run(home, "tenant").split() == [
        *("uv", "run", "python", "scripts/create_tenant.py"),
        *("--name", '"personal"', "--owner", '"ada-test"', "--compose", '"docker', 'compose"'),
    ]
