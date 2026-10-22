"""The interactive flow end to end, with a scripted console and a fake HTTP opener.

Guarantees proved here: the key is read hidden, sent only in a header, never shown in any output;
a rejected or malformed key writes nothing; local is the default and the non-TTY path is silent and
keyless; re-runs rotate the key or switch provider without touching anything else."""

from __future__ import annotations

import http.client
import stat
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from g2_support import (
    FAKE_KEY,
    OTHER_KEY,
    SCRIPT,
    FakeOpener,
    ScriptedConsole,
    http_error,
    load_setup_provider,
    models_page,
)

sp = load_setup_provider()

BASE = "# secrets from make env\nRAGFORGE_KEK=c2VjcmV0\nRAGFORGE_AGENT_MAX_TOOL_ROUNDS=0\n"
MODELS = models_page("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-image")


@pytest.fixture
def env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(BASE)
    path.chmod(0o600)
    return path


def run(env: Path, console: ScriptedConsole, opener: FakeOpener, *flags: str) -> int:
    code: int = sp.main(["--env-file", str(env), *flags], console=console, opener=opener)
    return code


def assert_no_key_leak(capsys: pytest.CaptureFixture[str], console: ScriptedConsole) -> None:
    out, err = capsys.readouterr()
    for key in (FAKE_KEY, OTHER_KEY):
        assert key not in out
        assert key not in err
        assert all(key not in event for event in console.events)


def test_gemini_happy_path_writes_all_four_values_and_never_shows_the_key(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])
    opener = FakeOpener([MODELS])

    assert run(env, console, opener) == 0

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_LLM_PROVIDER"] == "gemini"
    assert values["RAGFORGE_GEMINI_API_KEY"] == FAKE_KEY
    assert values["RAGFORGE_GEMINI_MODEL"] == "gemini-2.5-flash"  # the suggested Flash default
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == "0"
    assert env.read_text().startswith(BASE)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert_no_key_leak(capsys, console)


def test_key_travels_only_in_the_header_over_https(env: Path) -> None:
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])
    opener = FakeOpener([MODELS])

    run(env, console, opener)

    (request,) = opener.requests
    assert request.full_url.startswith("https://generativelanguage.googleapis.com/v1beta/models")
    assert FAKE_KEY not in request.full_url
    assert request.get_header("X-goog-api-key") == FAKE_KEY
    assert request.get_method() == "GET"


def test_privacy_notice_comes_before_the_key_prompt(env: Path) -> None:
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])

    run(env, console, FakeOpener([MODELS]))

    notice = next(i for i, e in enumerate(console.events) if "sent to Google" in e)
    key_prompt = next(i for i, e in enumerate(console.events) if e.startswith("secret:"))
    assert notice < key_prompt
    assert "Free tier" in console.events[notice]
    assert "Paid tier" in console.events[notice]


def test_declining_the_privacy_notice_stays_local_without_asking_for_a_key(env: Path) -> None:
    console = ScriptedConsole(answers=["y", "", "n"])
    opener = FakeOpener()

    assert run(env, console, opener) == 0

    assert opener.requests == []
    assert env.read_text() == BASE + sp.merge_env("", {"RAGFORGE_LLM_PROVIDER": "ollama"})


def test_rejected_key_writes_nothing_and_says_why(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = env.stat()
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[FAKE_KEY] * 3)
    # Plant the key in Google's free-text message: only the UPPER_SNAKE reason may be shown.
    opener = FakeOpener([http_error(400, echo=FAKE_KEY) for _ in range(3)])

    assert run(env, console, opener) == 1

    assert env.read_text() == BASE
    assert env.stat().st_ino == before.st_ino  # not even rewritten with the same content
    out, err = capsys.readouterr()
    assert "rejected this API key (HTTP 400 API_KEY_INVALID)" in out
    assert ".env was not changed" in err
    assert FAKE_KEY not in out + err


@pytest.mark.parametrize(
    "bad_key",
    ["short", f"{FAKE_KEY}\nRAGFORGE_KEK=attacker", f"{FAKE_KEY} #comment", f"$({FAKE_KEY})"],
)
def test_malformed_key_is_never_sent_or_written(env: Path, bad_key: str) -> None:
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[bad_key] * 3)
    opener = FakeOpener()

    assert run(env, console, opener) == 1

    assert opener.requests == []
    assert env.read_text() == BASE


def test_unreachable_api_writes_nothing(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[FAKE_KEY])
    opener = FakeOpener([urllib.error.URLError(OSError("no route"))])

    assert run(env, console, opener) == 1

    assert env.read_text() == BASE
    assert "Could not reach generativelanguage.googleapis.com" in capsys.readouterr().err


def test_rerun_rotates_the_key_in_place(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(env, ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY]), FakeOpener([MODELS]))
    first = env.read_text()

    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[OTHER_KEY])
    opener = FakeOpener([MODELS])
    assert run(env, console, opener) == 0

    assert env.read_text() == first.replace(FAKE_KEY, OTHER_KEY)
    assert opener.requests[0].get_header("X-goog-api-key") == OTHER_KEY
    assert_no_key_leak(capsys, console)


def test_enter_keeps_the_saved_key_and_the_saved_model(env: Path) -> None:
    env.write_text(
        BASE + f"RAGFORGE_GEMINI_API_KEY={FAKE_KEY}\nRAGFORGE_GEMINI_MODEL=gemini-2.5-pro\n"
        "RAGFORGE_GEMINI_THINKING_BUDGET=256\n"
    )
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[""])
    opener = FakeOpener([MODELS])

    assert run(env, console, opener) == 0

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_GEMINI_API_KEY"] == FAKE_KEY
    assert values["RAGFORGE_GEMINI_MODEL"] == "gemini-2.5-pro"  # current choice is the default
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == "256"  # a hand-set budget is kept
    assert opener.requests[0].get_header("X-goog-api-key") == FAKE_KEY


def test_choosing_local_only_flips_the_provider(env: Path) -> None:
    gemini = BASE + sp.merge_env(
        "",
        {
            "RAGFORGE_LLM_PROVIDER": "gemini",
            "RAGFORGE_GEMINI_API_KEY": FAKE_KEY,
            "RAGFORGE_GEMINI_MODEL": "gemini-2.5-flash",
        },
    )
    env.write_text(gemini)
    opener = FakeOpener()

    assert run(env, ScriptedConsole(answers=["n"]), opener) == 0

    assert env.read_text() == gemini.replace(
        "RAGFORGE_LLM_PROVIDER=gemini", "RAGFORGE_LLM_PROVIDER=ollama"
    )
    assert opener.requests == []


def test_model_can_be_picked_by_number(env: Path) -> None:
    console = ScriptedConsole(answers=["y", "", "y", "1"], secrets=[FAKE_KEY])

    run(env, console, FakeOpener([MODELS]))

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_GEMINI_MODEL"] == "gemini-2.5-pro"
    assert values["RAGFORGE_GEMINI_THINKING_BUDGET"] == "-1"  # Pro cannot run with thinking off
    assert any("Pro models cannot turn thinking off" in e for e in console.events)


def test_cancel_writes_nothing(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class Cancelling(ScriptedConsole):
        def secret(self, prompt: str) -> str:
            raise KeyboardInterrupt

    assert run(env, Cancelling(answers=["y", "", "y"]), FakeOpener()) == 130

    assert env.read_text() == BASE
    assert "cancelled" in capsys.readouterr().err


# ------------------------------------------------------------------ non-interactive + make up
def test_non_tty_asks_nothing_and_writes_nothing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    console = ScriptedConsole(tty=False)

    assert run(env, console, FakeOpener()) == 0

    assert not env.exists()
    assert not any(e.startswith(("ask:", "secret:")) for e in console.events)


def test_if_unset_is_silent_once_a_provider_is_saved(env: Path) -> None:
    env.write_text(BASE + "RAGFORGE_LLM_PROVIDER=gemini\n")
    console = ScriptedConsole()

    assert run(env, console, FakeOpener(), "--if-unset") == 0

    assert console.events == []


def test_if_unset_asks_on_first_run_in_a_terminal(env: Path) -> None:
    console = ScriptedConsole(answers=["n"])

    assert run(env, console, FakeOpener(), "--if-unset") == 0

    assert sp.read_env_values(env.read_text())["RAGFORGE_LLM_PROVIDER"] == "ollama"


def test_if_unset_without_a_terminal_stays_keyless_for_real(tmp_path: Path) -> None:
    # The real script and the real stdin check, exactly as `make up` runs it from CI.
    env = tmp_path / ".env"
    env.write_text(BASE)

    result = subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT), "--if-unset", "--env-file", str(env)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert env.read_text() == BASE
    assert result.stdout == result.stderr == ""  # `make up` in CI stays quiet


def test_setup_without_a_terminal_explains_and_writes_nothing_for_real(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(BASE)

    result = subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT), "--env-file", str(env)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert env.read_text() == BASE
    assert "keeping the default (local Ollama)" in result.stdout


# ------------------------------------------------------------------------------ model listing
def test_listing_follows_pages_and_keeps_only_safe_chat_models() -> None:
    first = models_page("gemini-2.5-flash") | {"nextPageToken": "page-2"}
    second = {
        "models": [
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {
                "name": "models/evil\nRAGFORGE_KEK=x",
                "supportedGenerationMethods": ["generateContent"],
            },
            {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
            "garbage",
        ]
    }
    opener = FakeOpener([first, second])

    models = sp.list_gemini_models(FAKE_KEY, opener)

    assert [m.name for m in models] == ["gemini-2.5-flash", "gemini-2.0-flash"]
    assert "pageToken=page-2" in opener.requests[1].full_url
    assert all(r.get_header("X-goog-api-key") == FAKE_KEY for r in opener.requests)
    assert all(FAKE_KEY not in r.full_url for r in opener.requests)


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["gemini-2.5-pro", "gemini-2.5-flash"], "gemini-2.5-flash"),
        (["gemini-3.0-pro", "gemini-3.0-flash-image", "gemini-3.0-flash"], "gemini-3.0-flash"),
        (["gemini-9-flash-preview", "gemini-9-pro"], "gemini-9-flash-preview"),
        ([], None),
    ],
)
def test_suggestion_prefers_a_stable_text_flash_model(
    names: list[str], expected: str | None
) -> None:
    models = [sp.GeminiModel(name=n, display_name=n) for n in names]

    assert sp.suggest_model(models) == expected


@pytest.mark.parametrize(("code", "invalid"), [(400, True), (401, True), (403, True), (500, False)])
def test_http_errors_map_to_key_or_service_problems(code: int, invalid: bool) -> None:
    with pytest.raises(sp.SetupError) as caught:
        sp.list_gemini_models(FAKE_KEY, FakeOpener([http_error(code, echo=FAKE_KEY)]))

    assert isinstance(caught.value, sp.InvalidKeyError) is invalid
    assert FAKE_KEY not in str(caught.value)


# --------------------------------------------------------------------------- --check (doctor)
@pytest.mark.parametrize(
    ("extra", "outcomes", "code", "phrase"),
    [
        ("", [], 0, "local Ollama"),
        ("RAGFORGE_LLM_PROVIDER=ollama\n", [], 0, "local Ollama"),
        ("RAGFORGE_LLM_PROVIDER=openai\n", [], 1, "not supported"),
        ("RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_MODEL=gemini-2.5-flash\n", [], 1, "empty"),
        (
            f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"
            "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\n",
            [MODELS],
            0,
            "API reachable",
        ),
        (
            f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"
            "RAGFORGE_GEMINI_MODEL=gemini-1.0-gone\n",
            [MODELS],
            1,
            "not available",
        ),
        (
            f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"
            "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\n",
            [http_error(400, echo=FAKE_KEY)],
            1,
            "rejected",
        ),
        (
            f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"
            "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\n",
            [TimeoutError()],
            2,
            "Could not reach",
        ),
    ],
)
def test_check_reports_the_provider_without_the_key(
    env: Path,
    capsys: pytest.CaptureFixture[str],
    extra: str,
    outcomes: list[object],
    code: int,
    phrase: str,
) -> None:
    env.write_text(BASE + extra)
    console = ScriptedConsole(tty=False)

    assert run(env, console, FakeOpener(list(outcomes)), "--check") == code

    assert phrase in capsys.readouterr().out
    assert env.read_text() == BASE + extra  # the check never writes
    assert_no_key_leak(capsys, console)


# ------------------------------------------------------------------ review round 1 guarantees
def test_terminal_console_reads_the_key_with_getpass_never_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scripted console proves the flow calls console.secret; this proves the REAL console's
    # secret() is hidden input. If it fell back to input(), the key would echo into scrollback.
    hidden: list[str] = []

    def fake_getpass(prompt: str = "") -> str:
        hidden.append(prompt)
        return f"  {FAKE_KEY}  "

    def forbidden_input(prompt: object = "") -> str:
        raise AssertionError("the key must never be read with input()")

    monkeypatch.setattr(sp.getpass, "getpass", fake_getpass)
    monkeypatch.setattr("builtins.input", forbidden_input)

    assert sp.TerminalConsole().secret("Gemini API key, input hidden: ") == FAKE_KEY
    assert hidden == ["Gemini API key, input hidden: "]


@pytest.mark.parametrize(
    "failure",
    [
        http.client.IncompleteRead(b"{"),
        http.client.BadStatusLine("x"),
        http.client.LineTooLong("h"),
    ],
)
def test_http_protocol_errors_are_user_facing_not_tracebacks(
    env: Path, capsys: pytest.CaptureFixture[str], failure: Exception
) -> None:
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[FAKE_KEY])

    assert run(env, console, FakeOpener([failure])) == 1

    err = capsys.readouterr().err
    assert "Could not reach generativelanguage.googleapis.com" in err
    assert "Traceback" not in err
    assert env.read_text() == BASE


def test_check_maps_a_protocol_error_to_a_warning(env: Path) -> None:
    env.write_text(BASE + f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n")
    opener = FakeOpener([http.client.IncompleteRead(b"")])

    assert run(env, ScriptedConsole(tty=False), opener, "--check") == 2


def test_edits_made_to_env_during_the_prompts_survive_the_write(env: Path) -> None:
    # The user can take minutes on the notice and key; .env must be re-read right before merging.
    class EditsMidFlow(ScriptedConsole):
        def secret(self, prompt: str) -> str:
            with env.open("a") as handle:
                handle.write("RAGFORGE_AGENT_MAX_TOOL_ROUNDS_EXPERIMENT=3\n")
            return super().secret(prompt)

    console = EditsMidFlow(answers=["y", "", "y", ""], secrets=[FAKE_KEY])

    assert run(env, console, FakeOpener([MODELS])) == 0

    text = env.read_text()
    assert text.startswith(BASE + "RAGFORGE_AGENT_MAX_TOOL_ROUNDS_EXPERIMENT=3\n")
    assert sp.read_env_values(text)["RAGFORGE_LLM_PROVIDER"] == "gemini"


@pytest.mark.parametrize("bad_budget", ["-5", "99999", "32769", "1e3"])
def test_out_of_range_budget_is_reset_so_the_saved_config_starts(
    env: Path, bad_budget: str
) -> None:
    env.write_text(BASE + f"RAGFORGE_GEMINI_THINKING_BUDGET={bad_budget}\n")
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[FAKE_KEY])

    assert run(env, console, FakeOpener([MODELS])) == 0

    assert sp.read_env_values(env.read_text())["RAGFORGE_GEMINI_THINKING_BUDGET"] == "0"
    assert any("resetting to 0" in e for e in console.events)


_GEMINI = f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n"


@pytest.mark.parametrize(
    ("extra", "phrase"),
    [
        (
            "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\nRAGFORGE_GEMINI_THINKING_BUDGET=99999\n",
            "-1 to",
        ),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\nRAGFORGE_GEMINI_THINKING_BUDGET=-2\n", "-1 to"),
        ("RAGFORGE_GEMINI_MODEL=gemini-2.5-flash\nRAGFORGE_GEMINI_THINKING_BUDGET=abc\n", "-1 to"),
        ("RAGFORGE_GEMINI_MODEL=.hidden\n", "not a valid model id"),
    ],
)
def test_check_blocks_values_settings_would_reject_without_calling_google(
    env: Path, capsys: pytest.CaptureFixture[str], extra: str, phrase: str
) -> None:
    env.write_text(BASE + _GEMINI + extra)
    console = ScriptedConsole(tty=False)
    opener = FakeOpener()

    assert run(env, console, opener, "--check") == 1

    assert phrase in capsys.readouterr().out
    assert opener.requests == []
    assert_no_key_leak(capsys, console)


def test_check_validates_the_compose_default_when_no_model_is_saved(env: Path) -> None:
    env.write_text(BASE + _GEMINI)
    ok = FakeOpener([models_page(sp.DEFAULT_GEMINI_MODEL)])
    missing = FakeOpener([MODELS])

    assert run(env, ScriptedConsole(tty=False), ok, "--check") == 0
    assert run(env, ScriptedConsole(tty=False), missing, "--check") == 1


def test_check_accepts_a_models_prefixed_id_like_settings_does(env: Path) -> None:
    env.write_text(BASE + _GEMINI + "RAGFORGE_GEMINI_MODEL=models/gemini-2.5-flash\n")

    assert run(env, ScriptedConsole(tty=False), FakeOpener([MODELS]), "--check") == 0


# ------------------------------------------------------------------ review round 2 guarantees
def run_with_shell(
    env: Path, console: ScriptedConsole, opener: FakeOpener, shell: dict[str, str], *flags: str
) -> int:
    code: int = sp.main(
        ["--env-file", str(env), *flags], console=console, opener=opener, environ=shell
    )
    return code


def test_check_accepts_inline_comments_and_export_exactly_as_compose_does(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The .env.example lines, uncommented, plus a personal note on the key: compose hands the api
    # "0" and the bare key, so doctor must judge (and send) the same values.
    env.write_text(
        BASE
        + "RAGFORGE_LLM_PROVIDER=gemini   # or ollama\n"
        + f'export RAGFORGE_GEMINI_API_KEY="{FAKE_KEY}"  # personal\n'
        + "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash  # fast\n"
        + "RAGFORGE_GEMINI_THINKING_BUDGET=0   # -1..32768; 0 = thinking off\n"
    )
    console = ScriptedConsole(tty=False)
    opener = FakeOpener([MODELS])

    assert run(env, console, opener, "--check") == 0

    (request,) = opener.requests
    assert request.get_header("X-goog-api-key") == FAKE_KEY
    assert_no_key_leak(capsys, console)


def test_check_blocks_a_key_compose_would_read_as_a_comment_without_sending_it(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_text(BASE + "RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY=   # get one\n")
    opener = FakeOpener()

    assert run(env, ScriptedConsole(tty=False), opener, "--check") == 1

    assert opener.requests == []
    assert "does not look like an API key" in capsys.readouterr().out


def test_enter_keeps_a_saved_key_that_carries_an_inline_comment(env: Path) -> None:
    env.write_text(BASE + f"RAGFORGE_GEMINI_API_KEY={FAKE_KEY}  # personal\n")
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[""])
    opener = FakeOpener([MODELS])

    assert run(env, console, opener) == 0

    assert opener.requests[0].get_header("X-goog-api-key") == FAKE_KEY
    text = env.read_text()
    assert f"RAGFORGE_GEMINI_API_KEY={FAKE_KEY}  # personal\n" in text  # the note is kept
    assert sp.read_env_values(text)["RAGFORGE_LLM_PROVIDER"] == "gemini"


def test_check_gives_shell_exports_priority_over_env_like_compose(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env.write_text(BASE + f"RAGFORGE_LLM_PROVIDER=ollama\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n")
    shell = {"RAGFORGE_LLM_PROVIDER": "gemini", "RAGFORGE_GEMINI_API_KEY": OTHER_KEY}
    console = ScriptedConsole(tty=False)
    opener = FakeOpener([models_page(sp.DEFAULT_GEMINI_MODEL)])

    assert run_with_shell(env, console, opener, shell, "--check") == 0

    assert opener.requests[0].get_header("X-goog-api-key") == OTHER_KEY  # the shell's key
    out = capsys.readouterr().out
    assert "shell-exported RAGFORGE_LLM_PROVIDER, RAGFORGE_GEMINI_API_KEY override .env" in out
    assert "gemini" in out
    assert OTHER_KEY not in out
    assert FAKE_KEY not in out


def test_an_empty_shell_export_still_overrides_env(env: Path) -> None:
    # compose treats an exported-but-empty variable as set: `${RAGFORGE_GEMINI_API_KEY}` is "".
    env.write_text(BASE + f"RAGFORGE_LLM_PROVIDER=gemini\nRAGFORGE_GEMINI_API_KEY={FAKE_KEY}\n")
    shell = {"RAGFORGE_GEMINI_API_KEY": ""}
    opener = FakeOpener()

    assert run_with_shell(env, ScriptedConsole(tty=False), opener, shell, "--check") == 1

    assert opener.requests == []


def test_if_unset_is_silent_when_the_shell_already_exports_a_provider(env: Path) -> None:
    console = ScriptedConsole()

    shell = {"RAGFORGE_LLM_PROVIDER": "ollama"}
    assert run_with_shell(env, console, FakeOpener(), shell, "--if-unset") == 0

    assert console.events == []
    assert env.read_text() == BASE


def test_setup_warns_when_a_shell_export_would_override_the_saved_choice(env: Path) -> None:
    console = ScriptedConsole(answers=["n"])
    shell = {"RAGFORGE_LLM_PROVIDER": "gemini"}

    assert run_with_shell(env, console, FakeOpener(), shell) == 0

    assert any("your shell exports RAGFORGE_LLM_PROVIDER" in e for e in console.events)


@pytest.mark.parametrize(
    ("outcomes", "phrase"),
    [
        ([http_error(400, echo=FAKE_KEY) for _ in range(3)], "No valid Gemini API key"),
        ([TimeoutError()], "Could not reach"),
    ],
)
def test_if_unset_failure_falls_back_to_local_so_make_up_continues(
    env: Path, capsys: pytest.CaptureFixture[str], outcomes: list[object], phrase: str
) -> None:
    # The owner's flow ends with "otherwise local": a first-run setup that cannot finish must not
    # stop `make up`. Nothing is written, so the next `make up` asks again.
    before = env.stat()
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[FAKE_KEY] * 3)

    assert run(env, console, FakeOpener(list(outcomes)), "--if-unset") == 0

    assert env.read_text() == BASE
    assert env.stat().st_ino == before.st_ino
    err = capsys.readouterr().err
    assert phrase in err
    assert "Starting on local Ollama" in err
    assert_no_key_leak(capsys, console)


def test_same_failure_in_make_setup_still_fails(env: Path) -> None:
    # Only the make-up gate falls back: an explicit `make setup` reports failure with rc 1.
    console = ScriptedConsole(answers=["y", "", "y"], secrets=[FAKE_KEY])

    assert run(env, console, FakeOpener([TimeoutError()])) == 1


def test_ctrl_c_during_first_run_setup_still_aborts_make_up(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Interrupted(ScriptedConsole):
        def secret(self, prompt: str) -> str:
            raise KeyboardInterrupt

    console = Interrupted(answers=["y", "", "y"])

    assert run(env, console, FakeOpener(), "--if-unset") == 130

    assert env.read_text() == BASE
    assert "cancelled" in capsys.readouterr().err


# Google's newer AI Studio key form "AQ.<token>" contains a '.'; the first real setup run
# rejected a valid key because the charset assumed the classic "AIza..." form (letters, digits,
# '-' and '_'). Synthetic value: never a real key.
DOTTED_KEY = "AQ.Ab8TESTONLY_leakcanary_DOTTED-key_0123456789abcdef"


def test_newer_dotted_key_format_is_accepted_and_written_safely(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    console = ScriptedConsole(answers=["y", "", "y", ""], secrets=[DOTTED_KEY])
    opener = FakeOpener([MODELS])

    assert run(env, console, opener) == 0

    values = sp.read_env_values(env.read_text())
    assert values["RAGFORGE_GEMINI_API_KEY"] == DOTTED_KEY
    assert DOTTED_KEY not in "".join(capsys.readouterr())
