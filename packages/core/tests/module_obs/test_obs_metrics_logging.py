"""Log redaction (ADR-0018 R2-8): every token shape is masked in what `configure_logging`
actually prints, nested values and rendered tracebacks included. Synthetic tokens only."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from ragforge_core.observability.logging import REDACTED, configure_logging, get_logger

# Synthetic, shape-correct values (never real tokens).
RF_PREFIX = "0123456789abcdef"
RF_KEY = f"rf_{RF_PREFIX}_" + "A" * 43
NOTION = "ntn_" + "b" * 46
NOTION_LEGACY = "secret_" + "C" * 43
GEMINI = "AIza" + "D" * 35
BEARER = "Bearer abc.def-ghi_jkl"
BASIC_B64 = "c3ludGhldGljOm5vdC1hLXBhc3N3b3Jk"  # base64("synthetic:not-a-password")
DSN_PASSWORD = "synthetic-pw@with:odd%2Fchars"  # a raw `/` is not valid in userinfo
DSN = f"postgresql+psycopg://rf_api:{DSN_PASSWORD}@db:5432/ragforge"
STDLIB_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "flashrank.Ranker", "httpx",
                  "httpcore")  # fmt: skip


@pytest.fixture(autouse=True)
def _restore_structlog() -> Iterator[None]:
    """structlog's config AND the stdlib loggers configure_logging touches are put back."""
    saved = {
        name: (list(lg.handlers), lg.level, lg.propagate)
        for name in STDLIB_LOGGERS
        for lg in [logging.getLogger(name)]
    }
    yield
    structlog.reset_defaults()
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


def _logged(capsys: pytest.CaptureFixture[str], **fields: object) -> dict[str, object]:
    configure_logging("INFO", json=True)
    get_logger("test").info("event", **fields)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    out: dict[str, object] = json.loads(line)
    return out


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (RF_KEY, f"rf_{RF_PREFIX}_{REDACTED}"),  # the public prefix (audit actor) survives
        (NOTION, REDACTED),
        (NOTION_LEGACY, REDACTED),
        (GEMINI, REDACTED),
        (BEARER, f"Bearer {REDACTED}"),
        (f"authorization: bearer {NOTION}", f"authorization: bearer {REDACTED}"),
        (f"Authorization: Basic {BASIC_B64}", f"Authorization: Basic {REDACTED}"),
        (
            f"connect to {DSN} failed",
            f"connect to postgresql+psycopg://rf_api:{REDACTED}@db:5432/ragforge failed",
        ),
        ("redis://:pw-synthetic@cache:6379/0", f"redis://:{REDACTED}@cache:6379/0"),
        # a raw `/`, `?` or `#` in an operator-supplied DSN (ours are percent-encoded)
        ("postgresql://u:ab/cd?e#f@db/x and more", f"postgresql://u:{REDACTED}@db/x and more"),
        ("url=postgresql://rf_api:Xy+/Z9==@db", f"url=postgresql://rf_api:{REDACTED}@db"),
    ],
)
def test_each_token_shape_is_masked(
    capsys: pytest.CaptureFixture[str], raw: str, expected: str
) -> None:
    assert _logged(capsys, detail=raw)["detail"] == expected


def test_secrets_inside_free_text_nested_values_and_the_event_are_masked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _logged(
        capsys,
        detail=f"upstream said: bad key {GEMINI} (retry)",
        nested={"headers": {"x": BEARER}, "list": [NOTION, 3]},
        pair=(RF_KEY, "fine"),
    )
    text = json.dumps(out)
    for secret in (GEMINI, NOTION, RF_KEY, "abc.def-ghi_jkl"):
        assert secret not in text
    assert out["detail"] == f"upstream said: bad key {REDACTED} (retry)"
    assert out["nested"] == {"headers": {"x": f"Bearer {REDACTED}"}, "list": [REDACTED, 3]}


def test_a_rendered_traceback_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    try:
        raise RuntimeError(f"connect failed with token {NOTION}")
    except RuntimeError:
        get_logger("test").exception("boom")
    out = capsys.readouterr().out
    assert "connect failed with token" in out
    assert NOTION not in out


def test_ordinary_values_are_left_alone(capsys: pytest.CaptureFixture[str]) -> None:
    ordinary = {
        "prefix": RF_PREFIX,
        "word": "secret_scan",  # too short to be a token
        "actor": f"key:{RF_PREFIX}",
        "count": 42,
        "model": "gemini-3.5-flash-lite",
        "phrase": "basic configuration",  # not base64-shaped
        "url": "https://api.notion.com/v1/pages/a@b",  # an @ in a path is not userinfo
        "dsn": "postgresql://db:5432/ragforge",  # no userinfo at all
        "user_only": "https://owner@example.com/",  # a user name without a password
    }
    out = _logged(capsys, **ordinary)
    assert {k: out[k] for k in ordinary} == ordinary


def test_the_pretty_console_renderer_is_masked_too(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=False)
    get_logger("test").info("event", detail=GEMINI)
    out = capsys.readouterr().out
    assert GEMINI not in out
    assert REDACTED in out


def test_values_that_only_become_text_while_rendering_are_masked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exception object, a set or bytes is stringified by the RENDERER, after every
    processor ran: the renderer pass masks the final line."""
    configure_logging("INFO", json=True)
    get_logger("test").info(
        "event",
        err=ValueError(f"upstream rejected {NOTION}"),
        tokens={GEMINI},
        raw=NOTION_LEGACY.encode(),
        **{f"key_{RF_KEY}": "x"},
    )
    out = capsys.readouterr().out
    for secret in (NOTION, GEMINI, NOTION_LEGACY, RF_KEY):
        assert secret not in out
    assert "upstream rejected" in out
    assert json.loads(out.strip().splitlines()[-1])["event"] == "event"  # still valid JSON


def test_a_dict_key_carrying_a_token_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    out = _logged(capsys, headers={NOTION: "value"})
    assert out["headers"] == {REDACTED: "value"}


@pytest.mark.parametrize("raw", [f"NOTION_TOKEN_{NOTION}", f"client_{NOTION_LEGACY}", f"x{RF_KEY}"])
def test_a_token_glued_to_a_word_is_still_masked(
    capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    detail = str(_logged(capsys, detail=raw)["detail"])
    assert NOTION not in detail
    assert NOTION_LEGACY not in detail
    assert RF_KEY not in detail
    assert REDACTED in detail


def test_the_pretty_console_masks_a_non_string_value(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=False)
    get_logger("test").info("event", err=RuntimeError(NOTION))
    out = capsys.readouterr().out
    assert NOTION not in out
    assert REDACTED in out


# ---------------------------------------------------------------------- stdlib logging
def _stdlib_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    assert NOTION not in captured.err  # nothing may bypass the handler to stderr either
    return [json.loads(line) for line in captured.out.strip().splitlines()]


def test_a_stdlib_record_is_rendered_as_json_and_masked(capsys: pytest.CaptureFixture[str]) -> None:
    """uvicorn, httpx, SQLAlchemy and LangGraph log with `logging`, never through structlog."""
    configure_logging("INFO", json=True)
    logging.getLogger("uvicorn.error").warning(
        "upstream %s failed via %s for %s", NOTION, DSN, RF_KEY
    )
    (line,) = _stdlib_lines(capsys)
    text = json.dumps(line)
    for secret in (NOTION, DSN_PASSWORD, RF_KEY):
        assert secret not in text
    assert line["event"] == (
        f"upstream {REDACTED} failed via postgresql+psycopg://rf_api:{REDACTED}@db:5432/ragforge "
        f"for rf_{RF_PREFIX}_{REDACTED}"
    )
    assert (line["level"], line["logger"]) == ("warning", "uvicorn.error")


def test_a_stdlib_traceback_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    try:
        raise ConnectionError(f"could not connect to {DSN}")
    except ConnectionError:
        logging.getLogger("sqlalchemy.pool").exception("checkout failed")
    (line,) = _stdlib_lines(capsys)
    assert "could not connect to" in str(line["exception"])
    assert DSN_PASSWORD not in json.dumps(line)


def test_a_library_basicconfig_after_configure_logging_adds_no_raw_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """flashrank's Ranker calls logging.basicConfig(level=INFO) when the API loads the model.
    pytest's own capture handlers would make basicConfig a no-op anyway, so the root starts
    bare here, as it does in the container (the fixture puts pytest's handlers back)."""
    logging.getLogger().handlers.clear()
    configure_logging("INFO", json=True)
    before = list(logging.getLogger().handlers)
    logging.basicConfig(level=logging.INFO)
    assert logging.getLogger().handlers == before
    logging.getLogger("flashrank.Ranker").info("loaded with %s", f"Basic {BASIC_B64}")
    (line,) = _stdlib_lines(capsys)
    assert line["event"] == f"loaded with Basic {REDACTED}"


def test_a_raw_handler_attached_before_configure_logging_is_replaced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logging.getLogger().handlers.clear()  # bare, as in the container (see the test above)
    logging.basicConfig(level=logging.INFO)  # a library that configured logging first
    assert [type(h) for h in logging.getLogger().handlers] == [logging.StreamHandler]
    configure_logging("INFO", json=True)
    raw = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(raw) == 1  # ours alone: basicConfig's stderr handler is gone
    logging.getLogger("langgraph").warning("token %s", NOTION)
    (line,) = _stdlib_lines(capsys)
    assert line["event"] == f"token {REDACTED}"


def test_configuring_twice_installs_one_handler(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    configure_logging("INFO", json=False)
    configure_logging("INFO", json=True)
    logging.getLogger("x").warning("once")
    assert len(_stdlib_lines(capsys)) == 1


def test_stdlib_records_honour_the_level_and_httpx_info_is_dropped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("WARNING", json=True)
    logging.getLogger("x").info("below the level")
    configure_logging("INFO", json=True)
    logging.getLogger("httpx").info("HTTP Request: GET https://api.notion.com/v1/x 200")
    logging.getLogger("httpx").warning("kept")
    assert [line["event"] for line in _stdlib_lines(capsys)] == ["kept"]


def test_the_pretty_console_masks_stdlib_records_too(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=False)
    logging.getLogger("uvicorn.error").error("bad %s", GEMINI)
    out = capsys.readouterr().out
    assert GEMINI not in out
    assert REDACTED in out


def test_uvicorns_own_default_handlers_are_taken_back(capsys: pytest.CaptureFixture[str]) -> None:
    """Any uvicorn.Config built without log_config=None (the MCP HTTP server, a test) applies
    uvicorn's LOGGING_CONFIG: raw stream handlers on `uvicorn`/`uvicorn.access` and
    propagate=False, so their records would never reach the redacting root handler."""
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(logging.StreamHandler())  # stderr, unredacted
    uvicorn_logger.propagate = False
    configure_logging("INFO", json=True)
    logging.getLogger("uvicorn.error").warning("token %s", NOTION)
    (line,) = _stdlib_lines(capsys)
    assert line["event"] == f"token {REDACTED}"
