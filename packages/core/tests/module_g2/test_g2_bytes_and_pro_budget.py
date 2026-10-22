"""Review round 4 guarantees, proved on real files and the real script.

- Unmanaged .env lines survive a write byte for byte, CRLF and lone CR included. The earlier
  merge tests only fed strings to merge_env, which hid that read_text() had already rewritten the
  line endings before the merge ever saw them.
- A .env that is not UTF-8 is a clear one-line error, never a traceback: `make up` still starts on
  local Ollama, doctor blocks with a readable reason, and the file is never rewritten.
- A Pro model is never saved with thinking off (budget 0), and doctor blocks that combination,
  because Gemini rejects it on every question even though the api starts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs
from g2_support import (
    FAKE_KEY,
    REPO_ROOT,
    SCRIPT,
    FakeOpener,
    ScriptedConsole,
    load_setup_provider,
    models_page,
)

sp = load_setup_provider()

# Every kind of ending a hand-edited or Windows-touched .env can carry, plus a quoted value and a
# value holding a lone CR (compose keeps it inside the value; checked with `docker compose config`).
UNMANAGED = (
    b"# written on Windows\r\n"
    b"RAGFORGE_KEK=c2VjcmV0\r\n"
    b"ODD=x\ry\r\n"
    b'QUOTED="a b # not a comment"\r\n'
    b"LF_ONLY=1\n"
    b"\r\n"
)
MANAGED_CRLF = b"RAGFORGE_LLM_PROVIDER=ollama\r\n"
MODELS = models_page("gemini-2.5-pro", "gemini-2.5-flash", sp.DEFAULT_GEMINI_MODEL)


def _run(env: Path, console: ScriptedConsole, opener: FakeOpener, *flags: str) -> int:
    code: int = sp.main(["--env-file", str(env), *flags], console=console, opener=opener)
    return code


# ---------------------------------------------------------------------- byte-for-byte on disk
def test_update_env_keeps_crlf_and_lone_cr_bytes_of_unmanaged_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(UNMANAGED + MANAGED_CRLF)

    sp.update_env(env, {"RAGFORGE_LLM_PROVIDER": "gemini"})

    # The replaced line keeps its own CRLF; nothing else moves by a single byte.
    assert env.read_bytes() == UNMANAGED + b"RAGFORGE_LLM_PROVIDER=gemini\r\n"


def test_interactive_setup_keeps_unmanaged_bytes_exactly(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(UNMANAGED + MANAGED_CRLF)
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    written = env.read_bytes()
    assert written.startswith(UNMANAGED + b"RAGFORGE_LLM_PROVIDER=gemini\r\n")
    appended = written[len(UNMANAGED) + len(b"RAGFORGE_LLM_PROVIDER=gemini\r\n") :]
    assert b"\r" not in appended  # our own new lines are plain LF
    values = sp.read_env_values(written.decode())
    assert values["RAGFORGE_GEMINI_API_KEY"] == FAKE_KEY


def test_a_lone_cr_stays_inside_the_value_as_in_compose(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(UNMANAGED)

    values = sp.read_env_values(sp._read_env_text(env))

    assert values["ODD"] == "x\ry"  # compose: {'ODD': 'x\ry'}; never a stray line "y"
    assert "y" not in values
    assert values["RAGFORGE_KEK"] == "c2VjcmV0"  # CRLF is not part of the value
    assert values["QUOTED"] == "a b # not a comment"


# ------------------------------------------------------------------------- non-UTF-8 .env
_NOT_UTF8 = b"RAGFORGE_KEK=ok\nLATIN1_NOTE=caf\xe9\nRAGFORGE_LLM_PROVIDER=ollama\n"


@pytest.fixture
def latin1_env(tmp_path: Path) -> Path:
    env = tmp_path / ".env"
    env.write_bytes(_NOT_UTF8)
    env.chmod(0o600)
    return env


def test_if_unset_with_a_non_utf8_env_falls_back_to_local(
    latin1_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(latin1_env, ScriptedConsole(), FakeOpener(), "--if-unset") == 0

    err = capsys.readouterr().err
    assert ".env is not valid UTF-8 (line 2)" in err
    assert "Starting on local Ollama" in err
    assert latin1_env.read_bytes() == _NOT_UTF8


def test_check_with_a_non_utf8_env_blocks_with_one_clear_line(
    latin1_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opener = FakeOpener()

    assert _run(latin1_env, ScriptedConsole(tty=False), opener, "--check") == 1

    err = capsys.readouterr().err
    assert err.strip() == (
        "setup: .env is not valid UTF-8 (line 2); fix that line, then run make setup."
    )
    assert opener.requests == []


def test_setup_never_rewrites_a_file_it_cannot_decode(
    latin1_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    console = ScriptedConsole(answers=["n"])

    assert _run(latin1_env, console, FakeOpener()) == 1

    assert latin1_env.read_bytes() == _NOT_UTF8
    assert "not valid UTF-8" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "code"), [(["--check"], 1), (["--if-unset"], 0)])
def test_real_script_reports_non_utf8_without_a_traceback_or_the_bytes(
    latin1_env: Path, flags: list[str], code: int
) -> None:
    # The real interpreter and entry point, as doctor (`2>&1`) and `make up` run it.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAGFORGE_")}
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-I", str(SCRIPT), "--env-file", str(latin1_env), *flags],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert proc.returncode == code
    assert b"Traceback" not in proc.stderr
    assert b"not valid UTF-8 (line 2)" in proc.stderr
    assert b"caf" not in proc.stdout + proc.stderr  # no file content is echoed


# --------------------------------------------------------------------- Pro model, budget 0
def test_picking_a_pro_model_saves_a_dynamic_budget_not_zero(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("RAGFORGE_KEK=c2VjcmV0\n")
    console = ScriptedConsole(answers=["y", "", "y", "1"], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_GEMINI_MODEL"] == "gemini-2.5-pro"
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == sp.PRO_DEFAULT_BUDGET == "-1"
    assert any("cannot turn thinking off" in e for e in console.events)


def test_a_hand_set_zero_budget_is_lifted_for_a_pro_model(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("RAGFORGE_GEMINI_THINKING_BUDGET=0\n")
    console = ScriptedConsole(answers=["y", "", "y", "1"], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    assert sp.read_env_values(env.read_text())["RAGFORGE_GEMINI_THINKING_BUDGET"] == "-1"


def test_a_flash_model_keeps_thinking_off(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("")
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_GEMINI_MODEL"] == sp.DEFAULT_GEMINI_MODEL
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == "0"


_PRO = f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-pro\nRAGFORGE_GEMINI_THINKING_BUDGET=0\n", 1),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-pro\n", 1),  # unset budget = compose default 0
        ("RAGFORGE_GEMINI_MODEL=models/gemini-2.5-pro\nRAGFORGE_GEMINI_THINKING_BUDGET=00\n", 1),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-pro\nRAGFORGE_GEMINI_THINKING_BUDGET=-1\n", 0),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-pro\nRAGFORGE_GEMINI_THINKING_BUDGET=256\n", 0),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\nRAGFORGE_GEMINI_THINKING_BUDGET=0\n", 0),
    ],
)
def test_check_blocks_a_pro_model_with_thinking_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: str, code: int
) -> None:
    env = tmp_path / ".env"
    env.write_text(_PRO + extra)
    opener = FakeOpener([] if code else [MODELS])

    assert _run(env, ScriptedConsole(tty=False), opener, "--check") == code

    out = capsys.readouterr().out
    if code:
        assert "Pro model, which cannot turn thinking off" in out
        assert opener.requests == []  # decided locally; nothing is sent to Google
    else:
        assert "API reachable" in out
    assert FAKE_KEY not in out


def test_compose_budget_default_is_what_check_assumes_when_unset() -> None:
    loaded = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text())
    value = loaded["services"]["api"]["environment"]["RAGFORGE_GEMINI_THINKING_BUDGET"]

    assert value == f"${{RAGFORGE_GEMINI_THINKING_BUDGET:-{sp.COMPOSE_DEFAULT_BUDGET}}}"


@pytest.mark.parametrize("choice", ["2", "3"])
def test_leaving_a_pro_model_resets_setups_own_dynamic_budget(tmp_path: Path, choice: str) -> None:
    """Review round 3, P2: -1 was setup's substitute for the Pro model; kept for a Flash model
    it would silently turn dynamic thinking on."""
    env = tmp_path / ".env"
    env.write_text("RAGFORGE_GEMINI_MODEL=gemini-2.5-pro\nRAGFORGE_GEMINI_THINKING_BUDGET=-1\n")
    console = ScriptedConsole(answers=["y", "", "y", choice], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    values = sp.read_env_values(env.read_text())
    assert not sp.is_pro_model(values["RAGFORGE_GEMINI_MODEL"])
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == "0"


def test_a_hand_set_budget_survives_a_flash_to_flash_switch(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\nRAGFORGE_GEMINI_THINKING_BUDGET=-1\n")
    console = ScriptedConsole(answers=["y", "", "y", "3"], secrets=[FAKE_KEY])

    assert _run(env, console, FakeOpener([MODELS])) == 0

    assert sp.read_env_values(env.read_text())["RAGFORGE_GEMINI_THINKING_BUDGET"] == "-1"
