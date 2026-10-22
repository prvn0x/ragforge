"""structlog setup: JSON in containers, pretty in a TTY. request_id/tenant_id/trace_id are bound
per request via contextvars. Never log secrets or (by default) prompt/chunk text.

Redaction (ADR-0018 R2-8) is the LAST line of defence, not the policy: code must still never
log a secret. It exists because the failure that leaks one is always an accident (an upstream
error message echoing a header, an exception repr carrying a URL with a key, a debug line
added in a hurry), and logs travel further than the process (`docker logs`, a pasted bug
report). It masks every token SHAPE this system handles, whatever the field name, on the
FINAL rendered line (`_RedactingRenderer` wraps the renderer). Masking the output rather than
the event dict is what makes it complete: keys, nested values, the formatted traceback and the
repr of a non-string value (an exception object, a set, bytes) that the renderer stringifies
after every processor has run are all just text by then.

Standard-library logging goes through the SAME renderer. Libraries log with `logging`, not
structlog (uvicorn, httpx, SQLAlchemy, LangGraph, flashrank), and flashrank's Ranker even calls
`logging.basicConfig(level=INFO)`, which would attach a plain stderr handler that prints their
records unredacted. `configure_logging` therefore installs ONE root handler whose
`structlog.stdlib.ProcessorFormatter` renders every stdlib record as the same JSON (or console)
line through `_RedactingRenderer`. A later `basicConfig` is then a no-op: it only acts on a root
logger without handlers. The API turns uvicorn's access log off (apps/api/main.py): RAGForge's
own middleware and metrics cover requests, and an access line carries the raw query string.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any, Final

import structlog

REDACTED: Final = "[REDACTED]"

# (pattern, replacement). Shapes, not values: the process never knows every live token.
# The prefixed shapes are deliberately NOT word-anchored: `_` is a word character, so `\b` would
# miss `NOTION_TOKEN_ntn_...` or `client_secret_...`, and in the pretty console the value follows
# an ANSI colour code ending in `m`. Over-masking a log line costs nothing; a miss leaks a token.
_SECRET_SHAPES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # RAGForge API keys: the 16-hex prefix is public (it is the audit actor), the secret is not.
    (re.compile(r"(rf_[0-9a-f]{16}_)[A-Za-z0-9_-]{16,}"), rf"\g<1>{REDACTED}"),
    # Notion integration tokens: current `ntn_...` and legacy `secret_...`.
    (re.compile(r"ntn_[A-Za-z0-9]{16,}"), REDACTED),
    (re.compile(r"secret_[A-Za-z0-9]{16,}"), REDACTED),
    # Google API keys (the Gemini key).
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), REDACTED),
    # Any bearer credential in an Authorization-shaped string.
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), rf"\g<1>{REDACTED}"),
    # HTTP Basic credentials (base64 of user:password). The value must look like base64 (a
    # digit, `+`, `/`, `=` or an inner capital), so "basic configuration" stays readable.
    (
        re.compile(
            r"((?i:basic)\s+)"
            r"(?=[A-Za-z0-9+/]*[0-9+/=]|[A-Za-z0-9+/]+[A-Z])[A-Za-z0-9+/]{8,}={0,2}"
        ),
        rf"\g<1>{REDACTED}",
    ),
    # The password in a URL's userinfo (a DSN in a driver error: postgresql://user:pw@host/db).
    # The user name stays (it names the role); everything up to the LAST `@` that is followed by
    # a host is the password, so a raw (not percent-encoded) `@`, `:`, `/`, `?` or `#` in it is
    # still masked whole. An `@` later in a path (`host:8080/a@b`) over-masks: harmless.
    (
        re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^:/?#@\s]*:)\S*@(?=[^\s@/?#]*(?:[/?#\s]|$))"),
        rf"\g<1>{REDACTED}@",
    ),
)

# A handler of this class is the one configure_logging installs (found again on a re-configure).
_ROOT_HANDLER_NAME: Final = "ragforge-redacting"
# Libraries whose INFO lines are one per outbound HTTP call (with the URL): our own metrics and
# events cover those calls, so only their warnings are kept.
_CHATTY_LIBRARIES: Final = ("httpx", "httpcore")
# uvicorn's default LOGGING_CONFIG (applied by any uvicorn.Config built WITHOUT log_config=None)
# gives these loggers their own raw stream handlers and propagate=False, which would bypass the
# root handler. configure_logging takes them back: no own handlers, propagate to the root.
_UVICORN_LOGGERS: Final = ("uvicorn", "uvicorn.error", "uvicorn.access")


def redact_text(text: str) -> str:
    """`text` with every known token shape masked."""
    for pattern, replacement in _SECRET_SHAPES:
        text = pattern.sub(replacement, text)
    return text


class _RedactingRenderer:
    """Wraps the final renderer and masks its output line, so a value that only becomes text
    while rendering (JSONRenderer's repr fallback, the console's repr) is masked as well."""

    def __init__(self, inner: structlog.types.Processor) -> None:
        self._inner = inner

    def __call__(
        self, logger: object, method: str, event_dict: MutableMapping[str, Any]
    ) -> str | bytes:
        rendered = self._inner(logger, method, event_dict)
        if isinstance(rendered, str):
            return redact_text(rendered)
        if isinstance(rendered, bytes | bytearray):
            return redact_text(bytes(rendered).decode("utf-8", "replace")).encode("utf-8")
        raise TypeError(f"a final log renderer must return text, not {type(rendered).__name__}")


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """structlog AND the stdlib root logger: one format, one redaction, stdout. Idempotent."""
    numeric_level: int = logging.getLevelName(level.upper())
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*processors, _RedactingRenderer(renderer)],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configure_stdlib(numeric_level, [structlog.stdlib.add_logger_name, *processors], renderer)


def _configure_stdlib(
    level: int, pre_chain: list[structlog.types.Processor], renderer: structlog.types.Processor
) -> None:
    """Route every stdlib record through the redacting renderer (see the module docstring).

    The root logger's stream handlers are replaced: ours from an earlier call, and a plain
    `logging.StreamHandler` a library's `basicConfig` attached before us (it prints unredacted).
    Other handlers (pytest's capture handlers are StreamHandler SUBCLASSES) are left alone."""
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_ROOT_HANDLER_NAME)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _RedactingRenderer(renderer),
            ],
        )
    )
    root = logging.getLogger()
    for old in list(root.handlers):
        if old.get_name() == _ROOT_HANDLER_NAME or type(old) is logging.StreamHandler:
            root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)
    for name in _CHATTY_LIBRARIES:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    for name in _UVICORN_LOGGERS:
        own = logging.getLogger(name)
        for old in list(own.handlers):
            own.removeHandler(old)
        own.propagate = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
