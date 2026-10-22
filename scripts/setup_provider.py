"""`make setup`: choose the chat model provider (hosted Gemini or local Ollama) and save it to .env.

Flow (the owner's ask): first "do you have an API key?"; if yes, which provider; then validate and
use it; otherwise stay local. Only the chat model moves: embeddings always stay on local Ollama.

Why it is built this way:
- Stdlib only (like connect_notion.py) so it runs with the system python3 before `uv sync`.
- The key is read with getpass (hidden), sent only in the `x-goog-api-key` HEADER (never the URL,
  which ends up in proxy logs and tracebacks), and never printed, logged or put in an exception.
- .env holds live secrets: it is rewritten atomically (temp file in the same directory, fsync,
  rename) with mode 600, and every line this script does not manage is preserved byte for byte.
- Values written to .env are charset-validated, so neither a pasted key nor a model name returned
  by the remote API can inject extra lines into .env.
- Without a terminal (CI, `make up < /dev/null`) nothing is asked and nothing is written: the
  default provider is local Ollama, so clone-and-run stays keyless.

- .env is read the way docker compose reads it (`export`, quotes, ` # inline comments`), and a
  variable exported in the shell wins over .env as it does in compose, so --check judges exactly
  the config api and worker will run with.

Modes: (none) interactive setup · `--if-unset` interactive only when RAGFORGE_LLM_PROVIDER is absent
(from .env and the shell) and stdin is a TTY, used by `make up`: if that first-run setup fails (bad
key, Google unreachable) it says so and exits 0 so the stack starts on local Ollama; Ctrl-C still
aborts · `--check` non-interactive health check (doctor).
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROVIDER_VAR = "RAGFORGE_LLM_PROVIDER"
KEY_VAR = "RAGFORGE_GEMINI_API_KEY"
MODEL_VAR = "RAGFORGE_GEMINI_MODEL"
BUDGET_VAR = "RAGFORGE_GEMINI_THINKING_BUDGET"
MANAGED_VARS = (PROVIDER_VAR, KEY_VAR, MODEL_VAR, BUDGET_VAR)

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
HTTP_TIMEOUT_S = 10.0
MAX_PAGES = 10  # the model list is a few dozen entries; a bound stops a looping pageToken
MAX_ATTEMPTS = 3

# AI Studio keys are URL-safe tokens: the classic "AIza..." form and the newer "AQ.<token>"
# form, which contains a '.'. The format is NOT ours to define (Google validates the key when
# setup lists models); this charset exists only to keep the .env write safe (no '#', quotes,
# '$', whitespace or newline) and to catch a truncated or mangled paste.
_KEY_RE = re.compile(r"[A-Za-z0-9._\-]{20,512}")
# The same rules as ragforge_core.config.Settings (gemini_model pattern, gemini_thinking_budget
# bounds), so anything setup writes or --check passes is a config the api and worker will accept
# at startup. A G2 test pins these against Settings so the two cannot drift apart.
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]{0,127}")
_SETTINGS_MODEL_RE = re.compile(r"(models/)?" + _MODEL_RE.pattern)
BUDGET_MIN, BUDGET_MAX = -1, 32_768
_REASON_RE = re.compile(r"[A-Z_]{1,64}")
# Flash-class first: fast and cheap. Budget 0 is written for every non-Pro model; a model that
# rejects it (gemini-3.5-flash-lite does, with a bare 400) is handled at runtime: the adapter
# retries once without thinkingConfig and logs `gemini_thinking_config_dropped`. The first
# entry is Settings' gemini_model default (one source of truth, pinned by a test, mirrored in
# compose.yaml and .env.example); the others are fallbacks for keys that cannot use it.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# What a Pro model gets instead of 0: -1 lets the model size its own thinking, the one value
# every Pro generation accepts (a fixed floor such as 128 is version-specific).
PRO_DEFAULT_BUDGET = "-1"
COMPOSE_DEFAULT_BUDGET = "0"  # compose.yaml's `${RAGFORGE_GEMINI_THINKING_BUDGET:-0}`
_PREFERRED_MODELS = (DEFAULT_GEMINI_MODEL, "gemini-2.5-flash", "gemini-flash-latest")
# Our own temp files only (so a sweep never touches anything else); see _sweep_stale_temps.
_TMP_PREFIX, _TMP_SUFFIX = ".env.setup-", ".tmp"
_STALE_TMP_AGE_S = 600.0
_NOT_CHAT = ("image", "tts", "audio", "live", "embedding", "vision", "learnlm")

PRIVACY_NOTICE = """\
Before you continue: what Gemini means for your notes
  * Each question you ask, and the note snippets retrieved to answer it, are sent to Google.
  * Free tier: Google may use that data to improve its products (and humans may review it).
  * Paid tier (billing enabled on the key's project): Google does not use it for that.
  * Embeddings and the search index stay on this machine; only the chat model moves.
  Switch back any time with `make setup` -> local Ollama.
"""


class SetupError(Exception):
    """A user-facing failure. Messages are written for the user and never contain the key."""


class InvalidKeyError(SetupError):
    """Google rejected the key (as opposed to the API being unreachable)."""


class Console(Protocol):
    """The terminal, injectable so tests can drive the flow without a TTY."""

    def isatty(self) -> bool: ...
    def ask(self, prompt: str) -> str: ...
    def secret(self, prompt: str) -> str: ...
    def say(self, message: str) -> None: ...


class TerminalConsole:
    """The real terminal: visible input via input(), hidden input via getpass."""

    def isatty(self) -> bool:
        return sys.stdin.isatty()

    def ask(self, prompt: str) -> str:
        return input(prompt).strip()

    def secret(self, prompt: str) -> str:
        return getpass.getpass(prompt).strip()

    def say(self, message: str) -> None:
        print(message)


@dataclass(frozen=True)
class GeminiModel:
    """A model id without the `models/` prefix, as RAGFORGE_GEMINI_MODEL expects it."""

    name: str
    display_name: str


Opener = Callable[..., Any]  # urllib.request.urlopen's shape; injectable for tests


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a 30x. urllib's redirect copies every request header (x-goog-api-key included)
    onto the new request, whatever its host or scheme, so following one could hand the key to
    another server or send it over plain http. The models endpoint never redirects, so a 30x is
    refused outright and the key goes only to the host it was meant for."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise SetupError(
            f"Gemini API answered with a redirect (HTTP {code}); refusing to resend the key."
        )


# Default handlers (proxy, https, error mapping) with the redirect handler swapped for the refusal.
_open_without_redirects: Opener = urllib.request.build_opener(_RefuseRedirects()).open


# ----------------------------------------------------------------------------------- .env I/O
# `[export ]NAME[ ]=` with optional leading blanks: what docker compose (compose-go) accepts. The
# api and worker get their values from compose's interpolation of .env, so compose's reading of a
# line is the one that counts; read_env_values and merge_env must agree with it or doctor would
# judge a different config from the one the containers run with.
_ASSIGN_RE = re.compile(r"[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=")
# Every value this script writes fits this charset: no newline (a second assignment), blank or '#'
# (a comment), quote, backslash or '$' (interpolation), so it reads back exactly as written.
_SAFE_VALUE_RE = re.compile(r"[A-Za-z0-9._:/\-]*")
_DQ_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


@dataclass(frozen=True)
class _Assignment:
    """One `NAME=value` line split so the value can be swapped and everything around it kept.

    line == head + raw_value + tail, where raw_value is the value as written (quotes included).
    """

    name: str
    value: str
    head: str
    raw_value: str
    tail: str
    quote: str  # '"', "'" or "" (unquoted)


def _parse_assignment(line: str) -> _Assignment | None:
    """Read one line the way compose does (verified against `docker compose config`):

    - leading blanks before the value are skipped;
    - a quoted value ends at its closing quote and anything after it is ignored (so a trailing
      `# comment` is too); double quotes understand backslash escapes, single quotes do not;
    - an unquoted value ends at the first ` #` (space then hash) and trailing blanks are trimmed,
      so `KEY=0   # note` is "0" but `KEY=#x` and `KEY=a#b` keep their '#'.

    python-dotenv (Settings outside compose) reads all of these the same way, except a value that
    is only a comment (`KEY= # x`) or a tab before '#'; compose wins there because it feeds the
    containers. Multi-line quoted values are not followed: no managed value can contain a newline.
    """
    match = _ASSIGN_RE.match(line)
    if match is None:
        return None
    rest = line[match.end() :]
    start = match.end() + len(rest) - len(rest.lstrip(" \t"))
    body = line[start:]
    if body[:1] in ("'", '"'):
        quote = body[0]
        value, length = _read_quoted(body, quote)
    else:
        quote = ""
        cut = body.find(" #")
        value = (body if cut < 0 else body[:cut]).rstrip()
        length = len(value)
    return _Assignment(
        name=match.group(1),
        value=value,
        head=line[:start],
        raw_value=body[:length],
        tail=body[length:],
        quote=quote,
    )


def _read_quoted(body: str, quote: str) -> tuple[str, int]:
    """(value, length of the quoted token) for a value starting with `quote`. An unterminated
    quote (a multi-line value) yields the rest of the line, which no managed value can be."""
    chars: list[str] = []
    i = 1
    while i < len(body):
        ch = body[i]
        if ch == quote:
            return "".join(chars), i + 1
        if quote == '"' and ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            chars.append(_DQ_ESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars), len(body)


def _lines(text: str) -> list[str]:
    """Split on '\n' only, keeping endings. str.splitlines would also split on \v, \f, \x1c and
    U+2028, which compose and python-dotenv do not, so a value containing one could be misread
    as a separate assignment."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def read_env_values(text: str) -> dict[str, str]:
    """Parse assignments as compose does; the last one wins, as in compose and python-dotenv."""
    values: dict[str, str] = {}
    for raw_line in _lines(text):
        parsed = _parse_assignment(raw_line.rstrip("\r\n"))
        if parsed is not None:
            values[parsed.name] = parsed.value
    return values


def merge_env(text: str, updates: Mapping[str, str]) -> str:
    """Set each variable in `updates` and leave every other line exactly as it was.

    An existing assignment is replaced in place: only its value changes, so its position, an
    `export ` prefix, the quoting and an inline `# comment` are all kept. Later duplicates of the
    same variable are dropped so the file has one source of truth. Variables not yet present are
    appended after a short comment.
    """
    for var, value in updates.items():
        if not _SAFE_VALUE_RE.fullmatch(value):  # defence in depth: callers validate charsets
            raise SetupError(f"refusing to write an unsafe value for {var}")
    lines = _lines(text)
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        body = line.rstrip("\r\n")
        parsed = _parse_assignment(body)
        if parsed is None or parsed.name not in updates:
            out.append(line)
            continue
        if parsed.name in seen:
            continue
        seen.add(parsed.name)
        ending = line[len(body) :] or "\n"
        out.append(_replace_value(parsed, updates[parsed.name]) + ending)
    missing = [var for var in updates if var not in seen]
    if missing:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append("# Chat model provider (written by `make setup`; embeddings stay local)\n")
        out.extend(f"{var}={updates[var]}\n" for var in missing)
    return "".join(out)


def _replace_value(parsed: _Assignment, value: str) -> str:
    """The line with only its value swapped. A quoted value stays quoted with the same quote, so
    text after the closing quote keeps being ignored; callers validate that `value` holds no
    quote, blank, '#' or backslash, so the result reads back as exactly `value`."""
    quoted = f"{parsed.quote}{value}{parsed.quote}"
    return f"{parsed.head}{quoted}{parsed.tail}"


def update_env(path: Path, updates: Mapping[str, str]) -> None:
    """Read, merge and atomically rewrite .env under one lock, so a concurrent run of this script
    cannot slip a write between our read and our rename (a lost update)."""
    target = Path(os.path.realpath(path))
    with _env_lock(target.parent) as locked:
        _write_locked(target, merge_env(_read_env_text(target), updates), locked=locked)


def write_env_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` so a crash leaves either the old or the new file, never half."""
    target = Path(os.path.realpath(path))
    with _env_lock(target.parent) as locked:
        _write_locked(target, text, locked=locked)


@contextlib.contextmanager
def _env_lock(directory: Path) -> Iterator[bool]:
    """An exclusive flock on the .env directory (no lock file is created) held for the whole write.

    Yields whether the lock is held. Holding it means no other run of this script is mid-write,
    so every leftover temp file is from a write that died and can be swept at once. Where flock is
    unavailable (no fcntl, or a filesystem that refuses it) the write still happens, unlocked.
    """
    try:
        import fcntl
    except ImportError:  # not POSIX; the scripts target macOS and Linux
        yield False
        return
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        yield False
        return
    try:
        try:
            fcntl.flock(dir_fd, fcntl.LOCK_EX)
            locked = True
        except OSError:
            locked = False
        yield locked
    finally:
        os.close(dir_fd)  # closing the descriptor releases the flock


def _write_locked(target: Path, text: str, *, locked: bool) -> None:
    """The temp file lives in the same directory (rename is atomic only within one filesystem) and
    is created 0600 by mkstemp, so the secret is never readable by others, not even briefly. The
    target is already symlink-resolved, so a symlinked .env keeps its link and its target changes.
    """
    _sweep_stale_temps(target.parent, any_age=locked)
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX, dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def _sweep_stale_temps(directory: Path, *, any_age: bool) -> None:
    """Delete key-bearing temp files left by a write that was SIGKILLed or lost power between
    mkstemp and rename (the BaseException cleanup cannot run then). They are 0600 but would ride
    along in a docker build context or a later `git add`. Only our own prefix is touched. With the
    directory lock held no other run can be mid-write, so every leftover goes; without it, only
    files older than any write could take, so a concurrent run's temp file is never removed."""
    cutoff = time.time() - _STALE_TMP_AGE_S
    for leftover in directory.glob(f"{_TMP_PREFIX}*{_TMP_SUFFIX}"):
        try:
            if leftover.is_file() and (any_age or leftover.stat().st_mtime < cutoff):
                leftover.unlink()
        except OSError:  # best effort: a sweep failure must not block saving the new settings
            continue


def _fsync_dir(directory: Path) -> None:
    """Persist the rename itself (best effort: not every platform allows opening a directory)."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _read_env_text(path: Path) -> str:
    """.env as its exact characters. Decoding bytes ourselves (not read_text) turns off universal
    newlines, which would rewrite every CRLF as LF and split a value holding a lone CR into two
    lines; compose keeps a lone CR inside the value (checked with `docker compose config`), so
    only '\n' may end a line here too, and unmanaged lines are written back byte for byte.

    Not UTF-8 -> SetupError naming only the line number (never the bytes, which may be a secret),
    so `make up` falls back to local and doctor prints one clear line instead of a traceback.
    Nothing is written then: rewriting a file we cannot decode would corrupt it.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data.count(b"\n", 0, exc.start) + 1
        raise SetupError(
            f"{path.name} is not valid UTF-8 (line {line}); fix that line, then run make setup."
        ) from None


# ------------------------------------------------------------------------------- Gemini API
def _error_reason(exc: urllib.error.HTTPError) -> str:
    """Google's machine-readable reason (e.g. API_KEY_INVALID). Only an UPPER_SNAKE token is
    returned, so nothing free-form from the response can reach the terminal."""
    try:
        body = json.loads(exc.read(4096) or b"{}")
    except (ValueError, OSError, AttributeError):  # no/garbled body: the code alone will do
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return ""
    candidates: list[object] = [
        d.get("reason") for d in error.get("details") or [] if isinstance(d, dict)
    ]
    candidates.append(error.get("status"))
    for value in candidates:
        if isinstance(value, str) and _REASON_RE.fullmatch(value):
            return value
    return ""


def _fetch_page(key: str, page_token: str | None, opener: Opener) -> dict[str, Any]:
    query = {"pageSize": "1000"}
    if page_token:
        query["pageToken"] = page_token
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        f"{GEMINI_MODELS_URL}?{urllib.parse.urlencode(query)}",
        headers={"x-goog-api-key": key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(request, timeout=HTTP_TIMEOUT_S) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        reason = _error_reason(exc)
        detail = f"HTTP {exc.code}{f' {reason}' if reason else ''}"
        if exc.code in (400, 401, 403):
            raise InvalidKeyError(f"Google rejected this API key ({detail}).") from None
        if exc.code == 429:
            raise SetupError(f"Google rate-limited the request ({detail}); try again.") from None
        raise SetupError(f"Gemini API returned an unexpected error ({detail}).") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        # HTTPException (IncompleteRead, BadStatusLine, ...) is not an OSError; without it a
        # truncated response would surface as a raw traceback instead of a user-facing message.
        cause = getattr(exc, "reason", exc)
        raise SetupError(
            f"Could not reach generativelanguage.googleapis.com ({type(cause).__name__})."
        ) from None
    except ValueError:
        raise SetupError("Gemini API returned a response that is not JSON.") from None
    if not isinstance(payload, dict):
        raise SetupError("Gemini API returned an unexpected response shape.")
    return payload


def list_gemini_models(key: str, opener: Opener = _open_without_redirects) -> list[GeminiModel]:
    """Validate `key` by listing models; return those that can answer chat (generateContent).

    Listing is free and has no side effects, so it is the cheapest proof that the key works.
    """
    models: list[GeminiModel] = []
    token: str | None = None
    for _ in range(MAX_PAGES):
        page = _fetch_page(key, token, opener)
        for raw in page.get("models") or []:
            model = _parse_model(raw)
            if model is not None:
                models.append(model)
        next_token = page.get("nextPageToken")
        token = next_token if isinstance(next_token, str) and next_token else None
        if token is None:
            break
    return models


def _parse_model(raw: object) -> GeminiModel | None:
    """Keep chat-capable models whose id is safe to write to .env; drop everything else."""
    if not isinstance(raw, dict):
        return None
    methods = raw.get("supportedGenerationMethods") or []
    name = str(raw.get("name") or "").removeprefix("models/")
    if "generateContent" not in methods or not _MODEL_RE.fullmatch(name):
        return None
    display = str(raw.get("displayName") or name)
    display = "".join(ch for ch in display if ch.isprintable())[:80]
    return GeminiModel(name=name, display_name=display)


def suggest_model(models: list[GeminiModel]) -> str | None:
    """A fast Flash-class default: a known-good id if offered, else any stable text Flash model."""
    names = [m.name for m in models]
    for preferred in _PREFERRED_MODELS:
        if preferred in names:
            return preferred
    for name in names:
        lowered = name.lower()
        if (
            "flash" in lowered
            and not any(word in lowered for word in _NOT_CHAT)
            and not any(word in lowered for word in ("preview", "exp"))
        ):
            return name
    return names[0] if names else None


def _chat_models(models: list[GeminiModel]) -> list[GeminiModel]:
    """Hide image/audio/embedding variants: they list generateContent but cannot answer text."""
    chat = [m for m in models if not any(w in m.name.lower() for w in _NOT_CHAT)]
    return chat or models


# ----------------------------------------------------------------------------- interaction
def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


def _choose_index(console: Console, prompt: str, count: int, default: int) -> int:
    for _ in range(MAX_ATTEMPTS):
        answer = console.ask(prompt) or str(default)
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer)
        console.say(f"  Please enter a number from 1 to {count}.")
    raise SetupError("No valid choice made; nothing was changed.")


def _read_valid_key(
    console: Console, existing: str | None, opener: Opener
) -> tuple[str, list[GeminiModel]]:
    """Ask for the key (hidden) until Google accepts it. Enter keeps a saved key (rotation is just
    typing a new one). Raises SetupError, so a rejected key never reaches .env."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        hint = " (Enter keeps the saved key)" if existing else ""
        key = console.secret(f"Gemini API key, input hidden{hint}: ") or (existing or "")
        if not key:
            console.say("  No key entered. Get one at https://aistudio.google.com/apikey")
            continue
        if not _KEY_RE.fullmatch(key):
            console.say(
                "  That does not look like a Gemini API key (letters, digits, '.', '-' and '_')."
            )
            continue
        console.say("  Checking the key with Google...")
        try:
            return key, list_gemini_models(key, opener)
        except InvalidKeyError as exc:
            left = MAX_ATTEMPTS - attempt
            console.say(f"  {exc} {f'{left} attempt(s) left.' if left else ''}".rstrip())
    raise InvalidKeyError("No valid Gemini API key entered; .env was not changed.")


def _choose_model(console: Console, models: list[GeminiModel], current: str | None) -> str:
    chat = _chat_models(models)
    if not chat:
        raise SetupError("This key cannot use any chat model (none supports generateContent).")
    names = [m.name for m in chat]
    default_name = current if current in names else suggest_model(chat)
    default = names.index(default_name) + 1 if default_name in names else 1
    console.say("Models available to this key:")
    for i, model in enumerate(chat, 1):
        tag = "  <- suggested (fast)" if i == default else ""
        console.say(f"  {i:>2}. {model.name}  ({model.display_name}){tag}")
    prompt = f"Model [1-{len(chat)}, Enter={default}]: "
    return names[_choose_index(console, prompt, len(chat), default) - 1]


def is_pro_model(model: str) -> bool:
    """Pro-class Gemini models reject thinkingBudget=0 at generateContent time (thinking cannot be
    turned off), so a Pro model with budget 0 starts fine and then fails every question."""
    return "-pro" in model.lower()


def valid_budget(value: str) -> bool:
    """Settings bounds (-1 = dynamic, 0 = off, up to 32768). Checked here because an out-of-range
    value passes compose but makes api and worker fail pydantic validation at startup."""
    return bool(re.fullmatch(r"-?\d{1,5}", value)) and BUDGET_MIN <= int(value) <= BUDGET_MAX


def _thinking_budget(console: Console, existing: str | None) -> str:
    """Keep a deliberate budget the user set by hand; default 0 (thinking off: lowest latency).
    An out-of-range budget is reset (and said so) rather than kept, so the saved config starts."""
    if not existing:
        return "0"
    if valid_budget(existing):
        return existing
    console.say(
        f"  {BUDGET_VAR}={existing[:12]!r} is outside {BUDGET_MIN}..{BUDGET_MAX}; resetting to 0."
    )
    return "0"


def setup_gemini(console: Console, values: Mapping[str, str], opener: Opener) -> dict[str, str]:
    """Return the .env updates for Gemini, or for local if the user declines the privacy notice."""
    console.say("")
    console.say(PRIVACY_NOTICE)
    if not _yes(console.ask("Send questions and note snippets to Google? [y/N]: ")):
        console.say("Staying on local Ollama.")
        return {PROVIDER_VAR: "ollama"}
    key, models = _read_valid_key(console, values.get(KEY_VAR) or None, opener)
    console.say("  Key accepted.")
    model = _choose_model(console, models, values.get(MODEL_VAR) or None)
    budget = _thinking_budget(console, values.get(BUDGET_VAR))
    previous = values.get(MODEL_VAR) or ""
    if budget == PRO_DEFAULT_BUDGET and is_pro_model(previous) and not is_pro_model(model):
        # -1 was setup's own substitute for the old Pro model, not a choice the user made: a
        # Flash model would silently think dynamically (slower, costlier). Review round 3, P2.
        budget = "0"
        console.say(f"  Switched away from a Pro model; {BUDGET_VAR} back to 0 (thinking off).")
    if is_pro_model(model) and budget == "0":
        budget = PRO_DEFAULT_BUDGET
        console.say(
            f"  Pro models cannot turn thinking off; using {BUDGET_VAR}={budget} (dynamic)."
            " Answers will be slower than with a Flash model."
        )
    return {PROVIDER_VAR: "gemini", KEY_VAR: key, MODEL_VAR: model, BUDGET_VAR: budget}


def run_interactive(
    env_path: Path, console: Console, opener: Opener, environ: Mapping[str, str]
) -> int:
    """The menu. Every path either writes one consistent set of values or writes nothing."""
    if not console.isatty():
        console.say(
            "setup: stdin is not a terminal; keeping the default (local Ollama). "
            "Run `make setup` in a terminal to use a hosted model."
        )
        return 0
    values = read_env_values(_read_env_text(env_path))
    console.say(f"Chat model provider (now: {values.get(PROVIDER_VAR) or 'ollama (default)'})")
    if _yes(console.ask("Do you have an API key for a hosted model provider? [y/N]: ")):
        console.say("Which provider?\n  1. Google Gemini")
        _choose_index(console, "Provider [1, Enter=1]: ", 1, 1)
        updates = setup_gemini(console, values, opener)
    else:
        updates = {PROVIDER_VAR: "ollama"}
    # Re-read now, not before the prompts: the user may spend minutes reading the notice and
    # pasting a key, and any edit made to .env meanwhile (an experiment knob, `make env`) must
    # survive. update_env holds a lock from this read to the rename.
    update_env(env_path, updates)
    _report(console, env_path, updates, environ)
    return 0


def _shell_overrides(environ: Mapping[str, str]) -> list[str]:
    """Managed variables exported in the shell. Compose interpolates the shell's value before
    .env's, so these, not .env, decide what api and worker get (values are never shown)."""
    return [var for var in MANAGED_VARS if var in environ]


def _report(
    console: Console, env_path: Path, updates: Mapping[str, str], environ: Mapping[str, str]
) -> None:
    provider = updates[PROVIDER_VAR]
    console.say("")
    if provider == "gemini":
        console.say(f"Saved to {env_path} (mode 600): provider=gemini, model={updates[MODEL_VAR]}.")
        console.say("The API key is saved but not shown. It reaches only the api and worker.")
    else:
        console.say(f"Saved to {env_path} (mode 600): provider=ollama (local, nothing leaves).")
    overrides = _shell_overrides(environ)
    if overrides:
        console.say(
            f"Note: your shell exports {', '.join(overrides)}; compose uses the shell's value over"
            " .env, so unset it for this choice to apply."
        )
    console.say("Apply it to a running stack with: make up")


def needs_setup(env_path: Path, console: Console, environ: Mapping[str, str]) -> bool:
    """`make up` asks only on a first run in a terminal; CI and re-runs stay silent. A provider
    exported in the shell also counts as chosen: compose would use it over anything we write."""
    values = read_env_values(_read_env_text(env_path))
    chosen = PROVIDER_VAR in values or PROVIDER_VAR in environ
    return not chosen and console.isatty()


def effective_values(env_path: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The managed values compose will hand to api and worker: .env, overlaid by any of them
    exported in the shell (compose's precedence: shell environment first, then .env)."""
    values = read_env_values(_read_env_text(env_path))
    return {
        var: environ.get(var, values[var])
        for var in MANAGED_VARS
        if var in values or var in environ
    }


def run_check(env_path: Path, console: Console, opener: Opener, environ: Mapping[str, str]) -> int:
    """Doctor's probe: 0 healthy, 1 blocking (bad config/key), 2 warning (API unreachable)."""
    overrides = _shell_overrides(environ)
    if overrides:
        console.say(f"Note: shell-exported {', '.join(overrides)} override .env (as in compose)")
    values = effective_values(env_path, environ)
    provider = values.get(PROVIDER_VAR) or "ollama"
    if provider == "ollama":
        console.say("LLM provider: local Ollama")
        return 0
    if provider != "gemini":
        console.say(f"LLM provider {provider!r} is not supported here (use ollama or gemini)")
        return 1
    key, model = values.get(KEY_VAR, ""), values.get(MODEL_VAR, "")
    if not key:
        console.say(f"LLM provider: gemini, but {KEY_VAR} is empty (run make setup)")
        return 1
    if not _KEY_RE.fullmatch(key):
        # Checked before any request: a mangled value (a comment or quote compose kept as part
        # of the key) is never sent to Google, and the user learns why without seeing it.
        console.say(
            f"LLM provider: gemini, but {KEY_VAR} does not look like an API key"
            " (stray quote, space or comment in .env?); run make setup"
        )
        return 1
    problem = _config_problem(model, values.get(BUDGET_VAR, ""))
    if problem:
        console.say(f"LLM provider: gemini, but {problem}; the api would not work (make setup)")
        return 1
    model = model or DEFAULT_GEMINI_MODEL  # compose's `:-` default: what the api will really use
    try:
        names = {m.name for m in list_gemini_models(key, opener)}
    except InvalidKeyError as exc:
        console.say(f"LLM provider: gemini, {exc} Run make setup.")
        return 1
    except SetupError as exc:
        console.say(f"LLM provider: gemini ({model}); {exc}")
        return 2
    if model.removeprefix("models/") not in names:
        console.say(f"LLM provider: gemini, key OK, but model {model!r} is not available to it")
        return 1
    console.say(f"LLM provider: gemini ({model}), key set, API reachable")
    return 0


def _config_problem(model: str, budget: str) -> str:
    """Why the api would not start or could not answer with these values, or "" if they work: a
    value Settings rejects at startup, or a Pro model with thinking off, which Settings accepts but
    Gemini rejects on every question. An unset model or budget means the compose default."""
    if model and not _SETTINGS_MODEL_RE.fullmatch(model):
        return f"{MODEL_VAR} is not a valid model id"
    if budget and not valid_budget(budget):
        return f"{BUDGET_VAR} must be an integer from {BUDGET_MIN} to {BUDGET_MAX}"
    effective_model = model or DEFAULT_GEMINI_MODEL
    effective_budget = budget or COMPOSE_DEFAULT_BUDGET
    if is_pro_model(effective_model) and int(effective_budget) == 0:
        return (
            f"{BUDGET_VAR}=0 with a Pro model, which cannot turn thinking off"
            f" (use {PRO_DEFAULT_BUDGET} or a positive budget)"
        )
    return ""


def main(
    argv: list[str] | None = None,
    console: Console | None = None,
    opener: Opener = _open_without_redirects,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Choose the chat model provider.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--if-unset", action="store_true", help="only if no provider is saved yet")
    mode.add_argument("--check", action="store_true", help="non-interactive health check")
    args = parser.parse_args(argv)
    con = console if console is not None else TerminalConsole()
    env = os.environ if environ is None else environ
    try:
        if args.check:
            return run_check(args.env_file, con, opener, env)
        if args.if_unset and not needs_setup(args.env_file, con, env):
            return 0
        return run_interactive(args.env_file, con, opener, env)
    except SetupError as exc:
        if args.if_unset:
            # First run from `make up`: "otherwise local". A rejected key or an offline Google
            # must not stop the stack; nothing was written, so the next `make up` asks again.
            print(
                f"setup: {exc} Starting on local Ollama for now; run `make setup` to retry.",
                file=sys.stderr,
            )
            return 0
        print(f"setup: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nsetup: cancelled; .env was not changed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
