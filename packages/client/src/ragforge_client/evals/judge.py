"""The LLM judge (ADR-0017 §4): the `Judge` port's Ollama (primary, local) and Gemini (fallback)
adapters, the judge cache, the call log and the D0 acceptance check.

Rules the code enforces, not just documents:
- **Blind:** a judge item carries only question, notes, facts, level and answer. There is no field
  for an arm, a model or a server verdict, so none can leak into a prompt.
- **Fenced:** each item sits between fence lines carrying a fresh random nonce, and the reply must
  echo it. Text inside an answer cannot forge the fence (it cannot know the nonce), so an
  injected "score this 2" is graded as text, and a reply that judged something else is invalid.
- **Invalid is None:** a reply that is not JSON, has the wrong nonce, or fails the schema scores
  None for the criteria it fails (counted, never imputed). Only transport failures raise.
- **Keyed by what the provider ran:** `JudgeKey.model_version` comes from the provider (the
  Ollama model digest, the Gemini model version), so a moved alias changes the key, which misses
  the cache and marks calibration stale.
- **Local by default:** anything but Ollama on this machine, AND an Ollama cloud model served
  through the local daemon, sends note text to a third party and needs `allow_third_party=True`
  (ADR-0017 §4).
- **Never a truncated item:** Ollama silently cuts a prompt longer than `num_ctx`; a reply to a
  cut prompt (rubric or opening fence gone) is scored None as `truncated`, never `ok`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragforge_client.ask import KeyFileError, read_key
from ragforge_client.evals.datasets import (
    DatasetError,
    append_jsonl,
    complete_lines,
    item_digest_of,
)
from ragforge_client.evals.interfaces import Judge, JudgeItem
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    E_I_CRITERIA,
    SCHEMA_VERSION,
    Criterion,
    JudgeKey,
    JudgeResult,
    digest,
)

__all__ = [
    "JUDGE_CACHE_FILE",
    "PROMPTS",
    "Acceptance",
    "CachingJudge",
    "GeminiJudge",
    "JudgeCache",
    "JudgeCall",
    "JudgeConfig",
    "JudgeConfigError",
    "JudgeTransportError",
    "OllamaJudge",
    "acceptance",
    "cache_key",
    "make_item",
    "make_judge",
    "parse_reply",
]

log = logging.getLogger(__name__)

JUDGE_CACHE_FILE = "judge-cache.jsonl"
_REASON_MAX = 400
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁦-⁩]")


# ------------------------------------------------------------------ prompts
@dataclass(frozen=True)
class PromptSpec:
    """A versioned judge prompt. `sha256` pins the file: editing the wording of a released
    version would keep its id (so the cache and calibration would claim results the new text
    never produced). Change the text by adding `<name>_v<n+1>.md` instead."""

    file: str
    criteria: tuple[Criterion, ...]
    sha256: str
    labels: tuple[str, str]  # how the item's `question` and `answer` are titled in the fence


PROMPTS: dict[str, PromptSpec] = {
    "judge_qa@1": PromptSpec(
        "judge_qa_v1.md",
        E_G_CRITERIA,
        "7c08054b4083e3babb17506f9a38ce301504d39d8aaa37ddfa9b6983906f19c5",
        ("QUESTION", "ANSWER"),
    ),
    "judge_question@1": PromptSpec(
        "judge_question_v1.md",
        E_I_CRITERIA,
        "d177469448675b8d423c8c436728b2f4b0f0646253d20925979c48e461bb9b42",
        ("TOPIC", "GENERATED QUESTION"),
    ),
}


def load_prompt(name: str) -> str:
    spec = PROMPTS.get(name)
    if spec is None:
        raise JudgeConfigError(f"unknown judge prompt {name!r} (known: {', '.join(PROMPTS)})")
    data = resources.files("ragforge_client.evals.prompts").joinpath(spec.file).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.sha256:
        raise JudgeConfigError(f"{spec.file} differs from {name}: add a new version instead")
    return data.decode()


# ------------------------------------------------------------------ items, fences, replies
def make_item(
    criteria: Sequence[Criterion],
    question: str,
    answer: str,
    notes: Sequence[str] = (),
    facts: Sequence[str] = (),
    level: str | None = None,
) -> JudgeItem:
    """A JudgeItem whose digest is computed from exactly what the judge is shown (the same
    function the label queue uses, so owner labels join judge results).

    `completeness` is dropped when there are no reference facts: its rubric grades against them,
    and a score against nothing is noise that would flow into calibration. The item is still
    judged on its other criteria; completeness is simply unscored (no result), not imputed."""
    if not facts:
        criteria = [c for c in criteria if c is not Criterion.completeness]
    return JudgeItem(
        item_digest=item_digest_of(question, answer, notes, facts, level),
        criteria=tuple(criteria),
        question=question,
        answer=answer,
        notes=tuple(notes),
        facts=tuple(facts),
        level=level,
    )


def _new_nonce(item: JudgeItem) -> str:
    """16 random hex chars, re-drawn in the (2^-64) case that the item text already holds it."""
    text = "\n".join((item.question, item.answer, *item.notes, *item.facts, item.level or ""))
    while True:
        nonce = secrets.token_hex(8)
        if nonce not in text:
            return nonce


def render_item(item: JudgeItem, spec: PromptSpec, nonce: str) -> str:
    q_label, a_label = spec.labels
    parts = [f"<<<ITEM {nonce}>>>", f"{q_label}:", item.question, ""]
    if item.notes:
        parts.append("NOTES:")
        for i, note in enumerate(item.notes, start=1):
            parts += [f"[N{i}]", note, ""]
    if item.facts:
        parts.append("REFERENCE FACTS:")
        parts += [f"- {f}" for f in item.facts]
        parts.append("")
    if item.level is not None:
        parts += ["LEVEL:", item.level, ""]
    parts += [f"{a_label}:", item.answer, f"<<<END {nonce}>>>", ""]
    parts.append("Criteria to grade: " + ", ".join(c.value for c in item.criteria))
    parts.append("The nonce of this item is the text after ITEM on the fence lines.")
    return "\n".join(parts)


def reply_schema(criteria: Sequence[Criterion]) -> dict[str, Any]:
    """The JSON schema of a reply: the nonce, then per criterion a reason BEFORE the score (the
    model writes its reasoning first, so the score follows from it rather than the other way
    round). The nonce is a plain string, not a `const`: a const would make constrained decoding
    echo it for free and the check would prove nothing."""
    per = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1, 2]},
        },
        "required": ["reason", "score"],
    }
    return {
        "type": "object",
        "properties": {"nonce": {"type": "string"}, **{c.value: per for c in criteria}},
        "required": ["nonce", *(c.value for c in criteria)],
    }


class _CriterionReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: str
    score: Annotated[int, Field(strict=True, ge=0, le=2)]


Outcome = Literal["ok", "partial", "invalid_json", "wrong_nonce", "invalid_schema", "truncated"]


@dataclass(frozen=True)
class ParsedReply:
    outcome: Outcome
    scores: dict[Criterion, tuple[int | None, str]]


def _clean_reason(text: str) -> str:
    return _CONTROL.sub("", " ".join(text.split()))[:_REASON_MAX]


def void_reply(outcome: Outcome, why: str, criteria: Sequence[Criterion]) -> ParsedReply:
    """Every criterion None: the reply as a whole cannot be trusted."""
    return ParsedReply(outcome, {c: (None, f"invalid reply: {why}") for c in criteria})


def parse_reply(text: str, nonce: str, criteria: Sequence[Criterion]) -> ParsedReply:
    """Validate one reply. Wrong nonce or non-JSON voids every criterion; a criterion whose own
    object fails the schema is None while the valid ones stand (`partial`)."""

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return void_reply("invalid_json", "not JSON", criteria)
    if not isinstance(data, dict):
        return void_reply("invalid_schema", "not a JSON object", criteria)
    if data.get("nonce") != nonce:
        return void_reply("wrong_nonce", "the nonce was not echoed", criteria)
    scores: dict[Criterion, tuple[int | None, str]] = {}
    for c in criteria:
        try:
            got = _CriterionReply.model_validate(data.get(c.value))
        except ValidationError:
            scores[c] = (None, f"invalid reply: {c.value} failed the schema")
            continue
        scores[c] = (got.score, _clean_reason(got.reason))
    valid = sum(s is not None for s, _ in scores.values())
    if valid == len(criteria):
        return ParsedReply("ok", scores)
    return ParsedReply("partial" if valid else "invalid_schema", scores)


def cache_key(
    judge: JudgeKey, item_digest: str, criterion: Criterion, batch_key: str | None = None
) -> str:
    """`JudgeResult.cache_key`. The criterion is part of it because there is one result per
    (item, criterion); the batch key only when batched (deferred, r2 P2), so an unbatched key is
    exactly the one the golden fixtures use."""
    value: dict[str, Any] = {
        "judge": judge.model_dump(mode="json"),
        "item": item_digest,
        "crit": criterion.value,
    }
    if batch_key is not None:
        value["batch"] = batch_key
    return digest(value)


# ------------------------------------------------------------------ errors and config
class JudgeConfigError(ValueError):
    """A judge that must not or cannot run: no opt-in, unknown prompt, model not available."""


class JudgeTransportError(RuntimeError):
    """The provider could not be reached or refused the call. `transient` (429/5xx/network) means
    re-run before counting (ADR-0017 §3 E-A error classes). Never carries the key or note text."""

    def __init__(self, provider: str, message: str, *, status: int | None, transient: bool):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.transient = transient


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})
DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


@dataclass(frozen=True)
class JudgeConfig:
    """How to build a judge. `allow_third_party` is the explicit opt-in that any provider other
    than the local Ollama needs, because the notes go to it with every call."""

    provider: Literal["ollama", "gemini"] = "ollama"
    model: str = "qwen3:8b"
    prompt: str = "judge_qa@1"
    base_url: str | None = None
    num_ctx: int = 8192
    keep_alive: str = "2m"  # short: the judge must not pin 5+ GB next to the chat model
    think: bool = False  # qwen3: thinking off (latency; the reason field is the reasoning)
    timeout_s: float = 120.0
    key_file: Path | None = None  # Gemini: the API key file (mode 600; never printed)
    allow_third_party: bool = False

    @property
    def url(self) -> str:
        return (self.base_url or DEFAULT_URLS[self.provider]).rstrip("/")

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama" and urlsplit(self.url).hostname in _LOCAL_HOSTS


# ------------------------------------------------------------------ call log + acceptance
@dataclass(frozen=True)
class JudgeCall:
    """One provider call, in the structured shape the server logs its own model calls in. The
    judge calls skip the server, so the harness logs them itself (ADR-0017 §4). No text."""

    provider: str
    model_version: str
    prompt: str
    latency_s: float
    item_digest: str
    outcome: Outcome

    def log_line(self) -> str:
        return json.dumps(
            {
                "event": "judge_call",
                "provider": self.provider,
                "model_version": self.model_version,
                "prompt": self.prompt,
                "latency_ms": round(self.latency_s * 1000),
                "item_digest": self.item_digest,
                "outcome": self.outcome,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class Acceptance:
    """ADR-0017 §4's D0 acceptance of a judge candidate. Swap growth (≤ 2 GB) is not measured
    here: it needs the host sampler the integrator runs alongside (evalrun's resource sampler)."""

    calls: int
    valid_rate: float | None
    latency_p50_s: float | None
    latency_max_s: float | None
    passed: bool
    reasons: tuple[str, ...]


def acceptance(
    calls: Sequence[JudgeCall], *, max_latency_s: float = 60.0, min_valid_rate: float = 0.99
) -> Acceptance:
    """Per-item latency (every call ≤ `max_latency_s`) and the valid-JSON rate (a reply that
    parsed with the right nonce, even if one criterion failed, is valid JSON; `partial` counts as
    valid here and its None criteria are counted by the metrics). A `truncated` call is NOT
    valid: a judge whose prompt does not fit its context fails acceptance, it is not excused."""
    if not calls:
        return Acceptance(0, None, None, None, False, ("no calls",))
    latencies = [c.latency_s for c in calls]
    valid = sum(c.outcome in ("ok", "partial") for c in calls) / len(calls)
    worst = max(latencies)
    reasons: list[str] = []
    if worst > max_latency_s:
        reasons.append(f"slowest item {worst:.1f}s > {max_latency_s:.0f}s")
    if valid < min_valid_rate:
        reasons.append(f"valid JSON {valid:.1%} < {min_valid_rate:.0%}")
    return Acceptance(
        len(calls), valid, statistics.median(latencies), worst, not reasons, tuple(reasons)
    )


# ------------------------------------------------------------------ the adapters
@dataclass(frozen=True)
class Completion:
    """What a provider returned: the reply text (None: no usable text, a bad reply) and whether
    the provider says it cut the prompt to fit its context."""

    text: str | None
    truncated: bool = False


class _HttpJudge(ABC):
    """Shared scoring loop: one item per call (batching is deferred), fenced, parsed, logged."""

    provider: Literal["ollama", "gemini"]

    def __init__(self, config: JudgeConfig, http: httpx.AsyncClient, model_version: str) -> None:
        self._config = config
        self._http = http
        self._spec = PROMPTS[config.prompt]
        self._system = load_prompt(config.prompt)
        self._key = JudgeKey(
            provider=self.provider,
            model_version=model_version,
            decoding=self._decoding(),
            prompt=config.prompt,
        )
        self._calls: list[JudgeCall] = []

    @property
    def key(self) -> JudgeKey:
        return self._key

    @property
    def calls(self) -> tuple[JudgeCall, ...]:
        return tuple(self._calls)

    @abstractmethod
    def _decoding(self) -> dict[str, float | int | str]: ...

    @abstractmethod
    async def _complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        """The provider's reply. A provider that answered with no usable text is a bad reply
        (scored None). Raises JudgeTransportError for transport failures only."""

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        results: list[JudgeResult] = []
        for item in items:
            results.extend(await self._score_one(item))
        return results

    def _check(self, item: JudgeItem) -> None:
        if not item.criteria or not set(item.criteria) <= set(self._spec.criteria):
            raise ValueError(f"{self._key.prompt} grades {[c.value for c in self._spec.criteria]}")
        if item.item_digest != item_digest_of(
            item.question, item.answer, item.notes, item.facts, item.level
        ):
            # a wrong digest would file this result under another item in the cache
            raise ValueError("item_digest does not match the item: build items with make_item")
        if Criterion.completeness in item.criteria and not item.facts:
            raise ValueError("completeness is graded against reference facts, and there are none")

    async def _score_one(self, item: JudgeItem) -> list[JudgeResult]:
        self._check(item)
        nonce = _new_nonce(item)
        started = time.monotonic()
        reply = await self._complete(
            self._system, render_item(item, self._spec, nonce), reply_schema(item.criteria)
        )
        latency = time.monotonic() - started
        if reply.truncated:
            # the closing fence survives a cut, so the nonce would still match: refuse first
            parsed = void_reply("truncated", "the prompt did not fit num_ctx", item.criteria)
        else:
            parsed = parse_reply(reply.text or "", nonce, item.criteria)
        call = JudgeCall(
            self.provider,
            self._key.model_version,
            self._key.prompt,
            latency,
            item.item_digest,
            parsed.outcome,
        )
        self._calls.append(call)
        log.info("%s", call.log_line())
        now = datetime.now(UTC)
        return [
            JudgeResult(
                cache_key=cache_key(self._key, item.item_digest, c),
                item_digest=item.item_digest,
                criterion=c,
                score=score,
                reason=reason,
                judge=self._key,
                scored_at=now,
            )
            for c, (score, reason) in parsed.scores.items()
        ]

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _raise_for(provider: str, response: httpx.Response) -> None:
    if response.is_success:
        return
    status = response.status_code
    transient = status == 429 or status >= 500
    # the status line only: an error body can echo the request (note text)
    raise JudgeTransportError(provider, f"HTTP {status}", status=status, transient=transient)


async def _request(
    provider: str, http: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    try:
        response = await http.request(method, url, **kwargs)
    except httpx.TimeoutException:
        raise JudgeTransportError(provider, "timed out", status=None, transient=True) from None
    except httpx.TransportError as exc:
        name = type(exc).__name__
        raise JudgeTransportError(provider, name, status=None, transient=True) from None
    except httpx.TooManyRedirects:
        # a misconfigured base URL or proxy: re-running will not help
        raise JudgeTransportError(
            provider, "TooManyRedirects", status=None, transient=False
        ) from None
    except httpx.HTTPError as exc:  # e.g. DecodingError: a body garbled on the way
        name = type(exc).__name__  # the type only: the message can carry the URL
        raise JudgeTransportError(provider, name, status=None, transient=True) from None
    _raise_for(provider, response)
    return response


def _bad_envelope(provider: str) -> JudgeTransportError:
    """The provider's own JSON envelope did not parse or lacks its fields. Unlike a bad MODEL
    reply (deterministic at temperature 0, so cached and counted as None), a garbled envelope is
    a transport accident (a proxy, a cut body): caching it would score the item None forever, so
    it is transient and the run retries it."""
    return JudgeTransportError(provider, "malformed reply envelope", status=None, transient=True)


_REPLY_RESERVE = 512  # tokens: three criteria, each a reason (<= 400 chars) and a score, as JSON
_MIN_CHARS_PER_TOKEN = 3  # English BPE averages ~4; below 3 is rare outside code or CJK
_CLOUD_SUFFIXES = (":cloud", "-cloud")


def _is_cloud_model(name: str, entry: dict[str, Any]) -> bool:
    """An Ollama model that runs on ollama.com: the local daemon only forwards the prompt (and
    the note text in it). `/api/tags` marks these with `remote_host`/`remote_model`; the name
    suffix catches a daemon too old to report them."""
    names = (name, str(entry.get("name", "")), str(entry.get("model", "")))
    return bool(entry.get("remote_host") or entry.get("remote_model")) or any(
        n.removesuffix(":latest").endswith(_CLOUD_SUFFIXES) for n in names
    )


class OllamaJudge(_HttpJudge):
    """The local judge (ADR-0017 §4): Ollama `/api/chat`, `format` = the reply JSON schema,
    temperature 0 with a fixed seed, a fixed `num_ctx`, a short `keep_alive`, thinking off."""

    provider: Literal["ollama", "gemini"] = "ollama"

    def _decoding(self) -> dict[str, float | int | str]:
        return {
            "temperature": 0,
            "num_ctx": self._config.num_ctx,
            "seed": 0,
            "think": "on" if self._config.think else "off",
        }

    @classmethod
    async def open(cls, config: JudgeConfig, http: httpx.AsyncClient) -> OllamaJudge:
        """Probe `/api/tags` for the model's digest: `qwen3:8b` is an alias that a `pull` moves,
        the digest is what actually runs."""
        response = await _request("ollama", http, "GET", "/api/tags")
        try:
            models = response.json()["models"]
            wanted = {config.model, f"{config.model}:latest"}
            entry = next(m for m in models if m.get("name") in wanted or m.get("model") in wanted)
            version = f"{config.model}@{str(entry['digest'])[:12]}"
        except (ValueError, KeyError, TypeError, StopIteration, AttributeError):
            raise JudgeConfigError(
                f"ollama has no model {config.model!r}: run `ollama pull {config.model}`"
            ) from None
        if _is_cloud_model(config.model, entry) and not config.allow_third_party:
            raise JudgeConfigError(
                f"ollama model {config.model!r} is a cloud model: the local daemon would send "
                "note text to ollama.com: opt in explicitly (allow_third_party) or use a local one"
            )
        return cls(config, http, version)

    async def _complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        if self._too_long(system, user):
            return Completion(None, truncated=True)  # never sent: it could only be cut
        body = {
            "model": self._config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self._config.num_ctx},
        }
        response = await _request("ollama", self._http, "POST", "/api/chat", json=body)
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise _bad_envelope("ollama") from None
        if not isinstance(content, str):
            raise _bad_envelope("ollama")
        return Completion(content, truncated=self._saturated(data.get("prompt_eval_count")))

    def _too_long(self, system: str, user: str) -> bool:
        """The first line of the truncation guard, before any call: a prompt longer than
        `_MIN_CHARS_PER_TOKEN` chars per token of the usable window cannot fit. It does not rely
        on `prompt_eval_count`, which Ollama may report without a cached system-prompt prefix (so
        the count check alone could fail open). Errs towards flagging: a false `truncated` is a
        visible failure of acceptance, a missed one is a silently wrong score."""
        budget = (self._config.num_ctx - _REPLY_RESERVE) * _MIN_CHARS_PER_TOKEN
        return len(system) + len(user) > budget

    def _saturated(self, prompt_tokens: object) -> bool:
        """Ollama cuts an over-long prompt to `num_ctx` WITHOUT an error, and reports the tokens
        it kept in `prompt_eval_count`. A prompt within `_REPLY_RESERVE` of the window was cut
        (or leaves no room for the reply). The second line of the guard (`_too_long` is the
        first); a missing count cannot be checked here, the integrator confirms the live daemon
        reports it (handoff)."""
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            return False
        return prompt_tokens >= self._config.num_ctx - _REPLY_RESERVE


class GeminiJudge(_HttpJudge):
    """The fallback judge: Gemini `generateContent`, one item per call (batching deferred, r2
    P2), JSON reply constrained by `responseJsonSchema`. The key travels in the
    `x-goog-api-key` header, never in the URL (URLs end up in logs and tracebacks)."""

    provider: Literal["ollama", "gemini"] = "gemini"

    def __init__(
        self, config: JudgeConfig, http: httpx.AsyncClient, model_version: str, served: str
    ) -> None:
        super().__init__(config, http, model_version)
        self._served = served

    def _decoding(self) -> dict[str, float | int | str]:
        return {"temperature": 0}

    @classmethod
    async def open(cls, config: JudgeConfig, http: httpx.AsyncClient) -> GeminiJudge:
        """Probe `models/<model>` for its `version`, then make one tiny generate call (a fixed
        word, no note text) for the `modelVersion` the API actually SERVES. The key binds both:
        an alias can move to a new served model while the metadata `version` stays the same, and
        a key without the served version would then hit the old cache and keep calibration
        "trusted" for a model it never measured."""
        response = await _request("gemini", http, "GET", f"/models/{config.model}")
        try:
            version = str(response.json()["version"])
        except (ValueError, KeyError, TypeError):
            raise JudgeConfigError(f"gemini returned no version for {config.model!r}") from None
        probe = {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1},
        }
        url = f"/models/{config.model}:generateContent"
        served = _served_version(await _request("gemini", http, "POST", url, json=probe))
        if served is None:
            raise JudgeConfigError(f"gemini did not report the served version of {config.model!r}")
        return cls(config, http, f"{config.model}@{version}/{served}", served)

    async def _complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        url = f"/models/{self._config.model}:generateContent"
        response = await _request("gemini", self._http, "POST", url, json=body)
        try:
            data = response.json()
        except ValueError:
            raise _bad_envelope("gemini") from None
        if not isinstance(data, dict):
            raise _bad_envelope("gemini")
        self._check_served(_served_version(response))
        return _gemini_completion(data)

    def _check_served(self, served: str | None) -> None:
        """Every reply must come from the model the key names: a reply from another one, or one
        that does not say (open() required the field, so its absence is a changed API, not a
        quirk), would be filed under a key (and a calibration) it does not belong to."""
        if served is None:
            raise JudgeTransportError(
                "gemini", "the reply did not report its modelVersion", status=None, transient=False
            )
        if served != self._served:
            raise JudgeTransportError(
                "gemini",
                f"served model {served} is not the judge key's {self._served}; re-run",
                status=None,
                transient=False,
            )


def _gemini_completion(data: dict[str, Any]) -> Completion:
    """A parsed `generateContent` envelope as a Completion. A safety block (a `blockReason`, or a
    candidate the API finished without content) is the provider's decision on THIS input, as
    repeatable at temperature 0 as a bad reply, so it is a bad reply (None, cached, counted).
    Anything else missing is a garbled envelope: transient."""
    candidates = data.get("candidates")
    if not candidates:
        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            return Completion(None)
        raise _bad_envelope("gemini")
    try:
        candidate = candidates[0]
        content = candidate.get("content")
        if content is None and candidate.get("finishReason") not in (None, "STOP"):
            return Completion(None)
        text = "".join(str(p.get("text", "")) for p in content["parts"])
    except (KeyError, TypeError, IndexError, AttributeError):
        raise _bad_envelope("gemini") from None
    return Completion(text)


def _served_version(response: httpx.Response) -> str | None:
    try:
        served = response.json().get("modelVersion")
    except (ValueError, AttributeError):
        return None
    return served if isinstance(served, str) and served else None


def judge_http_client(
    config: JudgeConfig,
    *,
    timeout: httpx.Timeout,
    headers: dict[str, str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """The judge's HTTP client. A LOCAL judge never honours environment proxies (HTTP_PROXY /
    ALL_PROXY are common on corporate laptops): its prompts carry note text, which must not
    leave the machine while the judge reports itself as local (review P1, ADR-0017 privacy)."""
    return httpx.AsyncClient(
        base_url=config.url,
        timeout=timeout,
        headers=headers,
        transport=transport,
        trust_env=not config.is_local,
    )


async def make_judge(
    config: JudgeConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> OllamaJudge | GeminiJudge:
    """Build and probe a judge. Async because the key needs the provider-returned model version
    BEFORE the first score (the run header records it). Use it as `async with`.

    Refuses, before any network call, a non-local provider without `allow_third_party`, an
    unknown prompt, and a Gemini key file that other users can read."""
    if config.prompt not in PROMPTS:
        raise JudgeConfigError(f"unknown judge prompt {config.prompt!r}")
    if not config.is_local and not config.allow_third_party:
        raise JudgeConfigError(
            f"the {config.provider} judge at {urlsplit(config.url).hostname} would send note text "
            "to a third party: opt in explicitly (allow_third_party)"
        )
    timeout = httpx.Timeout(config.timeout_s, connect=10.0)
    headers: dict[str, str] = {}
    if config.provider == "gemini":
        if config.key_file is None:
            raise JudgeConfigError("the gemini judge needs a key file (mode 600)")
        try:
            headers["x-goog-api-key"] = read_key(config.key_file)
        except KeyFileError as exc:
            raise JudgeConfigError(str(exc)) from None
    http = judge_http_client(config, timeout=timeout, headers=headers, transport=transport)
    try:
        if config.provider == "ollama":
            return await OllamaJudge.open(config, http)
        return await GeminiJudge.open(config, http)
    except BaseException:
        await http.aclose()
        raise


# ------------------------------------------------------------------ cache
class JudgeCache:
    """`<dataset dir>/judge-cache.jsonl`: JudgeResults keyed by `cache_key`. A changed JudgeKey
    (provider, served model version, decoding, prompt) or item digest is a different key, so it
    misses: nothing is ever invalidated in place, old lines simply stop matching. Invalid replies
    (score None) are cached too: at temperature 0 a re-ask returns the same reply, and the metric
    must count it, not retry until lucky."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._index: dict[str, JudgeResult] = {}
        self._offset = 0  # bytes of the file already indexed (always at a line end)
        self._lines = 0

    def _refresh(self) -> None:
        """Index what was appended since the last read: this process or another (the Eval page,
        a second run) may have written. Reads only the new bytes, so a run's gets and puts stay
        linear in the cache size. An unfinished last line is left for the next refresh; a file
        that shrank was replaced, and is re-read from the start."""
        if not self.path.exists():
            self._index, self._offset, self._lines = {}, 0, 0
            return
        size = self.path.stat().st_size
        if size < self._offset:
            self._index, self._offset, self._lines = {}, 0, 0
        if size == self._offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            lines, used = complete_lines(handle.read(size - self._offset))
        for line in lines:
            self._lines += 1
            if not line.strip():
                continue
            try:
                result = JudgeResult.model_validate_json(line)
            except ValidationError:
                raise DatasetError(
                    f"{self.path}:{self._lines}: not a v{SCHEMA_VERSION} judge result"
                ) from None
            self._index[result.cache_key] = result
        self._offset += used

    def get(self, key: str) -> JudgeResult | None:
        self._refresh()
        return self._index.get(key)

    def put(self, results: Sequence[JudgeResult]) -> None:
        if not results:
            return
        append_jsonl(self.path, [r.model_dump_json() for r in results])
        self._refresh()

    def __len__(self) -> int:
        self._refresh()
        return len(self._index)


@dataclass
class CachingJudge:
    """A Judge that answers from the cache, and on ANY miss re-asks the item's full criteria set.

    Never only the missing criteria: that changes the rendered prompt and the reply schema, and a
    criterion's score can depend on what it is asked alongside, while `cache_key` has no slot for
    the co-asked set. Re-asking the whole item keeps every result of one item from one prompt
    (the fresh results supersede the cached ones: the newest line wins)."""

    inner: Judge
    cache: JudgeCache
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    @property
    def key(self) -> JudgeKey:
        return self.inner.key

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        out: list[JudgeResult] = []
        for item in items:
            cached = [
                self.cache.get(cache_key(self.key, item.item_digest, c)) for c in item.criteria
            ]
            hits = [r for r in cached if r is not None]
            if len(hits) == len(item.criteria):
                self.hits += len(hits)
                out.extend(hits)
                continue
            self.misses += len(item.criteria)
            got = await self.inner.score([item])
            self.cache.put(got)
            fresh: dict[Criterion, JudgeResult] = {r.criterion: r for r in got}
            out.extend(fresh[c] for c in item.criteria)
        return out
