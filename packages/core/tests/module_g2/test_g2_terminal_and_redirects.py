"""Module G2: the two guarantees the scripted-console tests cannot prove on their own.

1. The key reaches only its intended host. urllib's default redirect handler copies every request
   header, x-goog-api-key included, to wherever a 30x points (another host, or plain http). The
   default opener refuses redirects; a real local server pair shows the key is never re-sent.
2. Hidden input is really hidden. The real script runs under a pseudo-terminal with the real
   TerminalConsole and the real getpass, so the terminal's own echo (or getpass's echoing
   fallback) would show up in the captured output. Only the HTTP opener is replaced, by a local
   fake: no test reaches Google. The tests also cover doctor's per-line rendering of `--check`.
"""

from __future__ import annotations

import http.server
import os
import pty
import select
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from g2_support import FAKE_KEY, REPO_ROOT, SCRIPT, load_setup_provider

sp = load_setup_provider()

_LEAK_MARKER = b"leakcanary"  # a distinctive slice of FAKE_KEY: catches partial echoes too
_PTY_TIMEOUT_S = 20.0


# ------------------------------------------------------------------------- redirect refusal
class _Recorder(http.server.BaseHTTPRequestHandler):
    """Records each request's api-key header; redirects when the server has a `location`."""

    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _RecordingServer)
        server.keys_seen.append(self.headers.get("x-goog-api-key"))
        if server.location is not None:
            self.send_response(server.status)
            self.send_header("Location", server.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b'{"models": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None  # keep request lines (and their query strings) out of the test output


class _RecordingServer(http.server.HTTPServer):
    def __init__(self, status: int = 302, location: str | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _Recorder)
        self.status = status
        self.location = location
        self.keys_seen: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def serve() -> Iterator[list[_RecordingServer]]:
    """Start servers on demand (loopback only) and always shut them down."""
    started: list[_RecordingServer] = []
    yield started
    for server in started:
        server.shutdown()  # returns once serve_forever has exited
        server.server_close()


def _start(started: list[_RecordingServer], server: _RecordingServer) -> _RecordingServer:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    started.append(server)
    return server


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_refused_and_the_key_never_reaches_the_new_host(
    status: int, serve: list[_RecordingServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = _start(serve, _RecordingServer())
    origin = _start(serve, _RecordingServer(status, location=f"{elsewhere.url}/steal"))
    monkeypatch.setattr(sp, "GEMINI_MODELS_URL", f"{origin.url}/v1beta/models")

    with pytest.raises(sp.SetupError, match="redirect") as caught:
        sp.list_gemini_models(FAKE_KEY)  # the real default opener, not a fake

    assert origin.keys_seen == [FAKE_KEY]  # sent once, to the intended host
    assert elsewhere.keys_seen == []  # the redirect target never saw a request at all
    assert FAKE_KEY not in str(caught.value)


def test_the_same_default_opener_still_talks_to_a_server_that_does_not_redirect(
    serve: list[_RecordingServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the test above: the refusal is specific to 30x, not a broken opener."""
    origin = _start(serve, _RecordingServer(location=None))
    monkeypatch.setattr(sp, "GEMINI_MODELS_URL", f"{origin.url}/v1beta/models")

    assert sp.list_gemini_models(FAKE_KEY) == []
    assert origin.keys_seen == [FAKE_KEY]


def test_doctor_and_setup_default_to_the_redirect_refusing_opener() -> None:
    import inspect

    for function in (sp.list_gemini_models, sp.main):
        default = inspect.signature(function).parameters["opener"].default
        assert default is sp._open_without_redirects
    handlers = sp._open_without_redirects.__self__.handlers
    redirect_handlers = [
        h for h in handlers if isinstance(h, sp.urllib.request.HTTPRedirectHandler)
    ]
    assert [type(h) for h in redirect_handlers] == [sp._RefuseRedirects]


# ------------------------------------------------------------------------ real terminal (pty)
# Imports the real script and runs its real main() with the real TerminalConsole (input/getpass on
# the pty). Only the opener is a local fake; it records the header it was given to a file so the
# test can prove the key was sent, without the key ever being printed. Must stay Python 3.9-safe.
_DRIVER = """\
import importlib.util, json, sys
from pathlib import Path

script, record = sys.argv[1], Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("setup_provider_under_pty", script)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class _Response:
    def __init__(self, body):
        self._body = body
    def read(self, *_):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return None

def fake_open(request, timeout=None):
    assert request.full_url.startswith(mod.GEMINI_MODELS_URL + "?")
    record.write_text(request.get_header("X-goog-api-key") or "")
    models = [
        {"name": "models/" + n, "displayName": n, "supportedGenerationMethods": ["generateContent"]}
        for n in ("gemini-2.5-pro", mod.DEFAULT_GEMINI_MODEL)
    ]
    return _Response(json.dumps({"models": models}).encode())

sys.exit(mod.main(sys.argv[3:], opener=fake_open))
"""


_CLAIM_TTY_AND_EXEC = (
    "import fcntl, os, sys, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
    "os.execv(sys.argv[1], sys.argv[1:])"
)


class _Pty:
    """A child process on a real pseudo-terminal, driven expect-style."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        # subprocess's fork+exec is thread-safe (pty.fork is not in a threaded pytest). The
        # launcher starts a new session and claims the pty as its controlling terminal, so the
        # script's getpass opens /dev/tty exactly as it would in a user's shell.
        self.fd, slave = pty.openpty()
        try:
            self.proc = subprocess.Popen(  # noqa: S603 - fixed interpreter, test driver
                [argv[0], "-I", "-c", _CLAIM_TTY_AND_EXEC, *argv],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                env=env,
            )
        finally:
            os.close(slave)
        self.output = b""

    def expect(self, needle: str) -> None:
        """Read until `needle` is on screen. Typing only after the prompt matters: getpass turns
        echo off before printing its prompt, so a key typed earlier would be echoed."""
        deadline = time.monotonic() + _PTY_TIMEOUT_S
        while needle.encode() not in self.output:
            if not self._read_some(deadline):
                raise AssertionError(f"never saw {needle!r}; screen: {self._screen()!r}")

    def send(self, text: str) -> None:
        os.write(self.fd, text.encode())

    def finish(self) -> int:
        deadline = time.monotonic() + _PTY_TIMEOUT_S
        while self._read_some(deadline):
            pass
        code = self.proc.wait(timeout=_PTY_TIMEOUT_S)
        os.close(self.fd)
        return code

    def _read_some(self, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        ready, _, _ = select.select([self.fd], [], [], remaining)
        if not ready:
            return False
        try:
            chunk = os.read(self.fd, 4096)
        except OSError:  # EIO: the child closed the terminal (it exited)
            return False
        self.output += chunk
        return bool(chunk)

    def _screen(self) -> str:
        """For assertion messages: the screen with any key-shaped text masked."""
        return self.output.replace(FAKE_KEY.encode(), b"<KEY>").decode(errors="replace")[-600:]


def _interpreters() -> list[str]:
    """The test interpreter, plus the macOS system python3 `make setup` may run on (3.9)."""
    found = [sys.executable]
    system = "/usr/bin/python3"
    if os.path.exists(system) and os.path.realpath(system) != os.path.realpath(sys.executable):
        found.append(system)
    return found


def _spawn(python: str, tmp_path: Path, *args: str) -> tuple[_Pty, Path]:
    driver = tmp_path / "pty_driver.py"
    driver.write_text(_DRIVER)
    record = tmp_path / "header-seen"
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAGFORGE_")}
    env.update(TERM="dumb", PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="always")
    argv = [python, "-I", str(driver), str(SCRIPT), str(record), *args]
    return _Pty(argv, env), record


@pytest.mark.parametrize("python", _interpreters())
def test_real_terminal_gemini_setup_never_shows_the_key(python: str, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# kept\nRAGFORGE_EMBED_MODEL=bge-m3\n")
    child, record = _spawn(python, tmp_path, "--env-file", str(env_file))

    child.expect("Do you have an API key for a hosted model provider? [y/N]: ")
    child.send("y\n")
    child.expect("Provider [1, Enter=1]: ")
    child.send("1\n")
    child.expect("Send questions and note snippets to Google? [y/N]: ")
    child.send("y\n")
    child.expect("Gemini API key, input hidden: ")
    child.send(FAKE_KEY + "\n")
    child.expect("Model [1-2, Enter=2]: ")
    child.send("\n")
    code = child.finish()

    assert code == 0, child._screen()
    # Control: visible answers ARE echoed, so this capture would have caught an echoed key.
    assert b"Provider [1, Enter=1]: 1" in child.output
    assert _LEAK_MARKER not in child.output
    assert b"GetPassWarning" not in child.output
    assert b"may be echoed" not in child.output
    assert record.read_text() == FAKE_KEY  # the key did travel, in the header
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    text = env_file.read_text()
    assert text.startswith("# kept\nRAGFORGE_EMBED_MODEL=bge-m3\n")
    managed = {k: v for k, v in sp.read_env_values(text).items() if k in sp.MANAGED_VARS}
    assert managed == {
        sp.PROVIDER_VAR: "gemini",
        sp.KEY_VAR: FAKE_KEY,
        sp.MODEL_VAR: sp.DEFAULT_GEMINI_MODEL,
        sp.BUDGET_VAR: "0",
    }


def test_real_terminal_if_unset_asks_on_a_first_run(tmp_path: Path) -> None:
    """`make up`'s gate under a real terminal: it asks, and "no" saves local."""
    env_file = tmp_path / ".env"
    env_file.write_text("RAGFORGE_EMBED_MODEL=bge-m3\n")
    child, record = _spawn(sys.executable, tmp_path, "--if-unset", "--env-file", str(env_file))

    child.expect("Do you have an API key for a hosted model provider? [y/N]: ")
    child.send("\n")
    assert child.finish() == 0, child._screen()

    assert sp.read_env_values(env_file.read_text())[sp.PROVIDER_VAR] == "ollama"
    assert not record.exists()  # local choice: no request was made


def test_real_terminal_if_unset_is_silent_once_a_provider_is_saved(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RAGFORGE_LLM_PROVIDER=ollama\n")
    before = env_file.read_bytes()
    child, _ = _spawn(sys.executable, tmp_path, "--if-unset", "--env-file", str(env_file))

    assert child.finish() == 0
    assert child.output == b""
    assert env_file.read_bytes() == before


def test_make_up_runs_exactly_the_gate_these_tests_drive() -> None:
    """The pty tests drive `setup_provider.py --if-unset`; pin that `make up` runs that command
    with the terminal inherited (no redirection of stdin), before compose starts anything."""
    recipe = subprocess.run(  # noqa: S603 - fixed make binary, dry run only
        [shutil.which("make") or "make", "-n", "-C", str(REPO_ROOT), "up"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    gate = [i for i, line in enumerate(recipe) if "setup_provider.py --if-unset" in line]
    compose = [i for i, line in enumerate(recipe) if " up --build" in line]
    assert len(gate) == 1
    assert compose
    assert gate[0] < compose[0]
    assert "<" not in recipe[gate[0]]
    assert "|" not in recipe[gate[0]]


# ------------------------------------------------------------------- doctor's line rendering
_BASH = shutil.which("bash") or "/bin/bash"


def _doctor_render(output: str, rc: int) -> tuple[list[str], int]:
    """Source doctor.sh (which then only defines its helpers) and render one --check result."""
    script = f'source "{REPO_ROOT}/scripts/doctor.sh"; report_provider "$1" "$2"; echo "fail=$fail"'
    result = subprocess.run(  # noqa: S603 - fixed bash and repo script
        [_BASH, "-c", script, "doctor-test", output, str(rc)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    lines = result.stdout.splitlines()
    return lines[:-1], int(lines[-1].removeprefix("fail="))


def test_doctor_shows_the_shell_override_note_as_its_own_warning_line() -> None:
    note = "Note: shell-exported RAGFORGE_LLM_PROVIDER override .env (as in compose)"
    lines, fail = _doctor_render(f"{note}\nLLM provider: local Ollama", 0)

    assert len(lines) == 2
    assert "!" in lines[0]
    assert lines[0].endswith("shell-exported RAGFORGE_LLM_PROVIDER override .env (as in compose)")
    assert "Note:" not in lines[0]
    assert "✓" in lines[1]
    assert lines[1].endswith("LLM provider: local Ollama")
    assert fail == 0


def test_doctor_marks_every_verdict_line_and_blocks_on_rc_1_but_not_on_the_note() -> None:
    lines, fail = _doctor_render("Note: shell-exported X override .env\nLLM provider: bad", 1)

    assert "!" in lines[0]
    assert "✗" not in lines[0]
    assert "✗" in lines[1]
    assert fail == 1


@pytest.mark.parametrize(("rc", "marker", "fail"), [(2, "!", 0), (1, "✗", 1), (0, "✓", 0)])
def test_doctor_never_hides_a_verdict_even_when_the_probe_printed_none(
    rc: int, marker: str, fail: int
) -> None:
    lines, failed = _doctor_render("", rc)

    assert len(lines) == 1
    assert marker in lines[0]
    assert failed == fail


def test_doctor_runs_its_checks_only_when_executed_not_when_sourced() -> None:
    lines, fail = _doctor_render("LLM provider: local Ollama", 0)

    assert not any("RAGForge doctor" in line for line in lines)
    assert fail == 0
