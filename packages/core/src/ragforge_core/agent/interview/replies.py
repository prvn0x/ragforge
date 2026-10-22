"""The JSON the interview prompts ask for, the one-repair-retry policy around it, and the one
structured log line every interview model call writes.

WHY pydantic models with `strict=True` and `extra="forbid"`: the reply is model output steered
by untrusted text (the notes, and for grading the user's own answer). Strict validation means a
reply is used only in exactly the declared shape: `"score": "10"`, `"score": 10.0`,
`"score": true`, `"score": 11` or an extra `"override": ...` key are all rejected, never coerced.
A prompt injection can at most change the VALUES within the declared bounds, which is the limit
of what any grader can promise. The JSON schema sent to the model is generated from the same
models (`model_json_schema`), so the two can never drift apart.

DESIGN section 8: "JSON-schema output with one repair retry". A reply that is still UNUSABLE
after the retry is a provider failure: 502 with the same title as every other model error. A
reviewer (`Reviewed`) may instead call a reply usable but imperfect ("degraded": a question whose
expected points are not backed by the notes, Phase 3c live failure): it is repaired once like a
malformed one, and if the repair is no better the usable reply is accepted, so a fine question is
never lost to its answer key. Each attempt logs its reason CATEGORY (`reason`), never its text.
Provider errors (timeouts, HTTP errors, rate limits, the adapter's own "not JSON") are NOT
retried here: they propagate as their own Problem, exactly as on /v1/chat.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from ragforge_core.agent.interview.model import CallTags
from ragforge_core.agent.interview.prompts import Prompt
from ragforge_core.domain.errors import RagforgeError
from ragforge_core.domain.models import ChatMessage, ChatTurnUsage, Role
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.logging import get_logger
from ragforge_core.observability.prompt_context import for_prompt

log = get_logger(__name__)

# Caps leave room for the sanitizer's escaping (each escape adds one character) inside the
# contract's limits on what the client sends back to /grade (question <= 2000 chars, each
# expected point <= 500).
MAX_QUESTION_CHARS: Final = 900
MAX_POINT_CHARS: Final = 200
MAX_POINTS: Final = 6
MAX_LINE_CHARS: Final = 400
MAX_MODEL_ANSWER_CHARS: Final = 4000
MAX_ITEMS: Final = 8
_RAW_ECHO_CHARS: Final = 4000  # what the repair turn shows the model of its own bad reply
UNTAGGED: Final = CallTags()


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _text(max_length: int) -> object:
    return StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length)


Point = Annotated[str, _text(MAX_POINT_CHARS)]
_Line = Annotated[str, _text(MAX_LINE_CHARS)]


class QuestionReply(_Reply):
    """The question and its points are judged SEPARATELY (`question.review_question`): the
    prompt asks for 3 to 6 points, but the schema accepts 0 to 6, so a reply with two good
    points is not schema-invalid (Phase 3c, live 502 on Spring Boot notes)."""

    question: Annotated[str, _text(MAX_QUESTION_CHARS)]
    topic: Annotated[str, _text(80)]
    expected_points: Annotated[list[Point], Field(max_length=MAX_POINTS)]


class MisconceptionReply(_Reply):
    said: _Line
    correction: _Line


class GradeReply(_Reply):
    score: Annotated[int, Field(ge=0, le=10)]
    summary: _Line
    right: Annotated[list[_Line], Field(max_length=MAX_ITEMS)]
    missed: Annotated[list[_Line], Field(max_length=MAX_ITEMS)]
    misconceptions: Annotated[list[MisconceptionReply], Field(max_length=MAX_ITEMS)]
    model_answer: Annotated[str, _text(MAX_MODEL_ANSWER_CHARS)]


class ModelReplyInvalid(RagforgeError):
    """The model's structured reply was still unusable after the repair retry. Same status and
    title as the provider adapters' `ChatModelError`, so clients handle every model failure the
    same way (the detail says which)."""

    status = 502
    title = "Model provider error"


@dataclass(frozen=True, slots=True)
class Reviewed[R: _Reply]:
    """One reply as a reviewer judged it.

    - `reply` set, `problem` empty: acceptable as is;
    - `reply` set, `problem` set: usable but DEGRADED (repaired once, accepted if no better);
    - `reply` None: unusable (repaired once, then 502).
    `problem` is our validator's message (shown to the model in the repair turn and logged as
    `detail`: field paths, rules and counts, never a value, so never note text); `reason` its
    category for the log; `metrics` extra counts to log."""

    reply: R | None
    problem: str = ""
    reason: str = ""
    metrics: Mapping[str, int] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if self.reply is None:
            return "malformed"
        return "degraded" if self.problem else "parsed"


@dataclass(frozen=True, slots=True)
class Parsed[R: _Reply]:
    reply: R
    usage: tuple[ChatTurnUsage, ...]  # one per model call, the repair included
    degraded: bool = False  # accepted although the reviewer had a (non-fatal) problem


def strict[R: _Reply](reply: type[R]) -> Callable[[Mapping[str, object]], Reviewed[R]]:
    """The default reviewer: the strict schema, nothing else."""

    def review(payload: Mapping[str, object]) -> Reviewed[R]:
        try:
            return Reviewed(reply.model_validate(payload))
        except ValidationError as exc:
            return Reviewed(None, schema_summary(exc), "schema")

    return review


def log_model_call(
    *,
    prompt: str,
    chat: ChatModel,
    started: float,
    chunk_ids: Collection[UUID],
    outcome: str,
    tags: CallTags = UNTAGGED,
    usage: ChatTurnUsage | None = None,
    attempt: int = 1,
    fields: Mapping[str, object] | None = None,
    **extra: object,
) -> None:
    """THE structured line of every interview model call (evals and cost dashboards read it):
    prompt name@version, provider/model, token usage, latency, the chunk ids the model was given,
    the outcome, and the call's level/kind/difficulty/source. Never text: no notes, questions,
    answers or keys."""
    log.info(
        "interview_model_call",
        prompt=prompt,
        provider=chat.provider,
        model=chat.model,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        latency_ms=round((time.perf_counter() - started) * 1000),
        chunk_ids=sorted(str(c) for c in chunk_ids),
        attempt=attempt,
        outcome=outcome,
        **tags.fields(),
        **(fields or {}),
        **extra,
    )


async def ask_json[R: _Reply](
    chat: ChatModel,
    prompt: Prompt,
    reply: type[R],
    *,
    chunk_ids: Collection[UUID],
    review: Callable[[Mapping[str, object]], Reviewed[R]] | None = None,
    tags: CallTags = UNTAGGED,
    log_extra: Mapping[str, object] | None = None,
) -> Parsed[R]:
    """One structured call with at most one repair retry. `reply` gives the JSON schema sent to
    the model; `review` judges each payload (default: `strict(reply)`). A clean reply is
    returned at once; after the repair, the best USABLE (degraded) reply seen is accepted: the
    one that kept the most (`metrics["points_kept"]`, ties to the later, repaired one), else
    `ModelReplyInvalid`."""
    with for_prompt(prompt.tag):  # the call's metrics label (ADR-0018 R2-3)
        return await _ask_json(chat, prompt, reply, chunk_ids, review, tags, log_extra)


async def _ask_json[R: _Reply](
    chat: ChatModel,
    prompt: Prompt,
    reply: type[R],
    chunk_ids: Collection[UUID],
    review: Callable[[Mapping[str, object]], Reviewed[R]] | None,
    tags: CallTags,
    log_extra: Mapping[str, object] | None,
) -> Parsed[R]:
    judge = review or strict(reply)
    schema = reply.model_json_schema()
    messages: list[ChatMessage] = [ChatMessage(role=Role.user, content=prompt.user)]
    usage: list[ChatTurnUsage] = []
    fallback: R | None = None
    fallback_kept = -1
    extra = dict(log_extra or {})
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            payload, used = await chat.complete_json(prompt.system, messages, schema)
        except ValueError as exc:  # the port's "unparseable JSON" (json errors subclass it)
            log_model_call(
                prompt=prompt.tag, chat=chat, started=started, chunk_ids=chunk_ids,
                outcome="malformed", tags=tags, attempt=attempt, reason="json", fields=extra,
            )  # fmt: skip
            problem, raw = _first_line(str(exc)), ""
        except Exception as exc:
            log_model_call(
                prompt=prompt.tag, chat=chat, started=started, chunk_ids=chunk_ids,
                outcome="error", tags=tags, attempt=attempt, error=type(exc).__name__, fields=extra,
            )  # fmt: skip
            raise
        else:
            usage.append(used)
            verdict = judge(payload)
            log_model_call(
                prompt=prompt.tag, chat=chat, started=started, chunk_ids=chunk_ids,
                outcome=verdict.outcome, tags=tags, usage=used, attempt=attempt,
                reason=verdict.reason or None, detail=verdict.problem or None,
                fields={**verdict.metrics, **extra},
            )  # fmt: skip
            if verdict.reply is not None and not verdict.problem:
                return Parsed(verdict.reply, tuple(usage))
            if verdict.reply is not None and (fallback is None or _kept(verdict) >= fallback_kept):
                fallback, fallback_kept = verdict.reply, _kept(verdict)
            problem = verdict.problem
            raw = json.dumps(payload, ensure_ascii=False)[:_RAW_ECHO_CHARS]
        if attempt == 1:
            messages.extend(_repair(raw, problem))
    if fallback is not None:
        return Parsed(fallback, tuple(usage), degraded=True)
    raise ModelReplyInvalid(f"the model's {reply.__name__} did not match the schema twice")


def _kept(verdict: Reviewed[_Reply]) -> int:
    return verdict.metrics.get("points_kept", 0)


def _repair(raw: str, error: str) -> Sequence[ChatMessage]:
    shown = [ChatMessage(role=Role.assistant, content=raw)] if raw else []
    return [
        *shown,
        ChatMessage(
            role=Role.user,
            content=f"That reply was not acceptable ({error}). "
            "Reply again with JSON matching the schema exactly.",
        ),
    ]


def schema_summary(exc: ValidationError) -> str:
    """Field paths and error kinds only: never the offending values, which may echo notes. A
    path keeps only DECLARED field names and list indexes: the key of an extra (forbidden) field
    is model-written text a note can steer, so it becomes `<extra>`."""
    declared = _declared_fields()
    parts = [
        f"{'.'.join(_loc_part(p, declared) for p in e['loc']) or 'reply'}: {e['type']}"
        for e in exc.errors()[:5]
    ]
    return "; ".join(parts)


def _loc_part(part: object, declared: frozenset[str]) -> str:
    if isinstance(part, int) or part in declared:
        return str(part)
    return "<extra>"


def _declared_fields() -> frozenset[str]:
    """Every field name of every reply model (subclasses included, wherever defined)."""
    names: set[str] = set()
    pending: list[type[_Reply]] = [_Reply]
    while pending:
        model = pending.pop()
        names.update(model.model_fields)
        pending.extend(model.__subclasses__())
    return frozenset(names)


def _first_line(text: str) -> str:
    return text.splitlines()[0][:200] if text.strip() else "invalid JSON"
