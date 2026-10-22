"""Failure messages and configuration: friendly, and never carrying the key."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from ui_support import ALL_INTERFACES

from ragforge_client.ask import KeyFileError
from ragforge_client.client import ApiError
from ragforge_contracts.api import Problem
from ragforge_ui.config import (
    ALLOW_REMOTE_ENV,
    DEFAULT_URL,
    display_url,
    exposure_refused,
    is_loopback,
    load_config,
)
from ragforge_ui.errors import describe_failure, from_problem

URL = "http://localhost:8000"


def _api_error(status: int, detail: str | None = None) -> ApiError:
    return ApiError(Problem(title="t", status=status, detail=detail, request_id="req-1"))


@pytest.mark.parametrize(
    ("status", "title"),
    [
        (401, "The API rejected this key"),
        (403, "This key is not allowed to do that"),
        (409, "The previous answer is still finishing"),
        (429, "Too many requests right now"),
        (503, "The API is not ready"),
    ],
)
def test_known_statuses_get_plain_language(status: int, title: str) -> None:
    error = describe_failure(_api_error(status), URL)
    assert error.title == title
    assert "(request id req-1)" in error.detail


def test_401_points_at_the_key_file() -> None:
    assert "key file" in describe_failure(_api_error(401), URL).detail


def test_unknown_status_uses_the_problem_and_strips_control_characters() -> None:
    error = from_problem(Problem(title="Teapot\x1b[2J", status=418, detail="short‮and stout"))
    assert error.title == "Teapot[2J"
    assert error.detail.startswith("shortand stout")


def test_api_down_names_the_url_and_the_fix() -> None:
    error = describe_failure(httpx.ConnectError("refused"), URL)
    assert error.title == "Cannot reach the RAGForge API"
    assert URL in error.detail
    assert "make up" in error.detail


def test_dropped_stream_and_timeouts() -> None:
    dropped = describe_failure(httpx.RemoteProtocolError("peer closed"), URL)
    assert dropped.title == "The connection dropped"
    assert describe_failure(httpx.ReadTimeout("slow"), URL).title.startswith("The API took")


def test_unreadable_response() -> None:
    assert "could not read" in describe_failure(ValueError("bad json"), URL).title


def test_unknown_exception_shows_its_type_never_its_message() -> None:
    error = describe_failure(RuntimeError("Bearer rf_live_secret leaked"), URL)
    assert "RuntimeError" in error.detail
    assert "rf_live_secret" not in error.title + error.detail


# ------------------------------------------------------------------ config
def _key(path: Path, mode: int = 0o600) -> Path:
    path.write_text("rf_live_secret\n")
    path.chmod(mode)
    return path


def test_config_reads_the_key_file_and_hides_the_value(tmp_path: Path) -> None:
    key = _key(tmp_path / "ui.key")
    config = load_config(["--key-file", str(key)], {}, tmp_path / "none")
    assert config.api_key.get_secret_value() == "rf_live_secret"
    assert config.api_url == DEFAULT_URL
    assert "rf_live_secret" not in repr(config)
    assert "rf_live_secret" not in str(config)


def test_config_url_precedence_and_unknown_flags_are_ignored(tmp_path: Path) -> None:
    _key(tmp_path / "only.key")
    env = {"RAGFORGE_URL": "http://a:1/", "RAGFORGE_API_URL": "http://b:2"}
    assert load_config(["-q", "--other"], env, tmp_path).api_url == "http://a:1"
    assert load_config([], {"RAGFORGE_API_URL": "http://b:2"}, tmp_path).api_url == "http://b:2"
    assert load_config(["--url", "http://c:3"], env, tmp_path).api_url == "http://c:3"


def test_config_uses_env_key_file_and_refuses_a_shared_one(tmp_path: Path) -> None:
    shared = _key(tmp_path / "shared.key", mode=0o640)
    with pytest.raises(KeyFileError, match="chmod 600") as info:
        load_config([], {"RAGFORGE_KEY_FILE": str(shared)}, tmp_path / "none")
    assert "rf_live_secret" not in str(info.value)


def test_config_refuses_to_guess_between_keys(tmp_path: Path) -> None:
    _key(tmp_path / "a.key")
    _key(tmp_path / "b.key")
    with pytest.raises(KeyFileError, match="several keys"):
        load_config([], {}, tmp_path)


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("http://user:tok@host:8000/x?k=v#f", "http://host:8000/x"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_display_url_drops_credentials_and_query(url: str, shown: str) -> None:
    assert display_url(url) == shown


@pytest.mark.parametrize(
    ("address", "local"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LOCALHOST", True),
        (None, False),  # Streamlit's default: every interface
        ("", False),
        (ALL_INTERFACES, False),
        ("::", False),
        ("192.168.1.20", False),
        ("my-laptop.local", False),  # a hostname may resolve to a LAN address
    ],
)
def test_is_loopback(address: str | None, local: bool) -> None:
    assert is_loopback(address) is local


def test_exposure_is_refused_unless_explicitly_allowed() -> None:
    assert exposure_refused(None, {})
    assert exposure_refused(ALL_INTERFACES, {ALLOW_REMOTE_ENV: "true"})  # only the exact opt-in
    assert not exposure_refused(ALL_INTERFACES, {ALLOW_REMOTE_ENV: "1"})
    assert not exposure_refused("127.0.0.1", {})


@pytest.mark.parametrize(
    ("host", "refused"),
    [
        ("localhost:8501", False),
        ("127.0.0.1:8501", False),
        ("[::1]:8501", False),
        ("localhost", False),
        (None, False),  # no browser request (tests): not something a page can mount
        ("evil.example:8501", True),  # DNS rebinding: the attacker's own name
        ("evil.example", True),
        ("192.168.1.20:8501", True),
        ("localhost.evil.example:8501", True),
    ],
)
def test_a_session_through_a_foreign_host_name_is_refused(host: str | None, refused: bool) -> None:
    """Review r3 P1: a loopback bind does not stop DNS rebinding; the Host of the session must
    be this machine, unless the explicit remote opt-in is set."""
    from ragforge_ui.config import host_refused

    assert host_refused(host, {}) is refused
    assert host_refused(host, {ALLOW_REMOTE_ENV: "1"}) is False


@pytest.mark.parametrize(
    ("origin", "host", "refused"),
    [
        ("http://localhost:8501", "localhost:8501", False),
        ("http://127.0.0.1:8501", "127.0.0.1:8501", False),
        (None, "localhost:8501", False),  # no browser request (tests)
        ("http://localhost:3000", "localhost:8501", True),  # another local web page
        ("http://evil.example", "localhost:8501", True),
        ("null", "localhost:8501", True),  # sandboxed iframe / file:// page
    ],
)
def test_a_session_opened_by_another_page_is_refused(
    origin: str | None, host: str, refused: bool
) -> None:
    """Integrated review P2: Streamlit accepts any localhost origin on any port."""
    from ragforge_ui.config import origin_refused

    assert origin_refused(origin, host, {}) is refused
    assert origin_refused(origin, host, {ALLOW_REMOTE_ENV: "1"}) is False
