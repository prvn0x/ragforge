"""Write one interview question: from chosen notes (the question call, then one ground check of
its expected points), or from general knowledge (one call, nothing to check). No I/O besides the
model.

Eval seams (fake or real chat model, fixed notes, no HTTP, database or sanitizer; RAW drafts):
- `write_question(chat, notes, ...)` (source=notes);
- `write_general_question(chat, topic=..., ...)` (source=general);
- `review_question(notes)`: the pure reply reviewer, callable on a recorded payload.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

from ragforge_core.agent.citations import normalize_citations
from ragforge_core.agent.interview.model import (
    CallTags,
    Difficulty,
    DraftQuestion,
    InterviewSettings,
    Kind,
    Level,
    Source,
)
from ragforge_core.agent.interview.prompts import (
    general_question_prompt,
    question_prompt,
    recent_questions,
)
from ragforge_core.agent.interview.replies import (
    MAX_POINTS,
    Point,
    QuestionReply,
    Reviewed,
    ask_json,
    schema_summary,
)
from ragforge_core.agent.interview.verify import lexically_backed, recover_label, verified_items
from ragforge_core.agent.labels import CITATION_MARKER
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

MIN_QUESTION_CHARS: Final = 10
# Fewer lexically backed expected points than this and the reply is repaired once; if the
# repair is no better, the question is accepted with the points it has (possibly none).
MIN_BACKED_POINTS: Final = 2
# A marker and the spaces before it ("Why X [S1]?" -> "Why X?"). Line breaks are kept: a code
# question carries a fenced block.
_MARKER: Final = re.compile(rf"[ \t]*{CITATION_MARKER.pattern}")
_POINT: Final = TypeAdapter(Point, config={"strict": True})

# Reason categories of `interview_model_call` (`reason`), for evals to count.
SCHEMA: Final = "schema"  # the question or topic broke the schema: unusable
POINTS_SCHEMA: Final = "points_schema"  # only points broke it: salvaged, the rest dropped
QUESTION_TOO_SHORT: Final = "question_too_short"
POINTS_UNBACKED: Final = "points_unbacked"


# Lenient forms the normaliser leaves alone but a reader still takes for a citation: a RANGE or
# LIST of labels ("(s1-s2)", "(S1, T2)") or one with a "source" cue ("(source: S2)"). A bare
# "(S3)" is left to the normaliser's own product judgement: "Simple Storage Service (S3)" and
# "(T2) and (T3) instances" are text, not citations (re-review P1).
_LOOSE_MARKER: Final = re.compile(
    r"[ \t]*\(\s*(?:"
    r"sources?\s*:?\s*[ST]?\d+(?:\s*[,\u2013-]\s*[ST]?\d+)*"
    r"|[ST]\d+(?:\s*[,\u2013-]\s*[ST]?\d+)+"
    r")\s*\)",
    re.IGNORECASE,
)


def strip_labels(text: str) -> str:
    """The text without citation markers: the prompt forbids them in a question, and a marker
    there would point the candidate at where the answer is. General-mode texts are stripped
    too: they come from no note, so a label would be a false citation."""
    return _LOOSE_MARKER.sub("", _MARKER.sub("", normalize_citations(text))).strip()


def _salvage(payload: Mapping[str, object], exc: ValidationError) -> QuestionReply | None:
    """When ONLY `expected_points` broke the schema (a point too long, seven of them, a number
    instead of a string), the question is still good: keep the points that are valid alone (at
    most `MAX_POINTS`). Anything wrong outside the points makes the reply unusable."""
    if any(not e["loc"] or e["loc"][0] != "expected_points" for e in exc.errors()):
        return None
    raw = payload.get("expected_points")
    points: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            points.append(_POINT.validate_python(item))
        except ValidationError:
            continue
    try:
        return QuestionReply.model_validate({**payload, "expected_points": points[:MAX_POINTS]})
    except ValidationError:
        return None


def _parse(payload: Mapping[str, object]) -> tuple[QuestionReply | None, str, str]:
    """(reply, problem, reason) of the schema step alone."""
    try:
        return QuestionReply.model_validate(payload), "", ""
    except ValidationError as exc:
        salvaged = _salvage(payload, exc)
        if salvaged is None:
            return None, schema_summary(exc), SCHEMA
        return salvaged, schema_summary(exc), POINTS_SCHEMA


def review_question(
    notes: Sequence[RankedChunk],
) -> Callable[[Mapping[str, object]], Reviewed[QuestionReply]]:
    """The question and its points are judged SEPARATELY (design review, Phase 3c):
    - an unusable QUESTION (schema-invalid, or too short once labels are removed) makes the
      reply unusable: repaired once, then 502;
    - weak POINTS only degrade it: each point is kept if lexically backed (an unlabelled one
      gets its label back when exactly one note backs it, `verify.recover_label`), the others
      are dropped, and fewer than `MIN_BACKED_POINTS` asks for one repair. A reply whose
      question is fine is never lost to its points.
    The reply returned carries only the kept points; `metrics` logs given/backed/kept."""

    def review(payload: Mapping[str, object]) -> Reviewed[QuestionReply]:
        reply, problem, reason = _parse(payload)
        if reply is None:
            return Reviewed(None, problem, reason)
        if len(strip_labels(reply.question)) < MIN_QUESTION_CHARS:
            why = "question: too short once citation labels are removed"
            return Reviewed(None, why, QUESTION_TOO_SHORT)
        kept: list[str] = []
        backed = 0
        for point in reply.expected_points:  # the model's order is kept
            if lexically_backed(point, notes):
                kept.append(point)
                backed += 1
            elif (recovered := recover_label(point, notes)) is not None:
                kept.append(recovered)
        metrics = {
            "points_given": len(reply.expected_points),
            "points_backed": backed,
            "points_kept": len(kept),
        }
        reply = reply.model_copy(update={"expected_points": kept})
        if len(kept) < MIN_BACKED_POINTS:
            # WHY repair rather than accept at once: a question with no answer key grades worse,
            # and a reply that dropped its labels is the commonest way to lose every point.
            why = (
                f"expected_points: only {len(kept)} end with the [S<n>] label of a note that "
                f"states them; at least {MIN_BACKED_POINTS} must"
            )
            return Reviewed(reply, why, POINTS_UNBACKED, metrics)
        return Reviewed(reply, problem, reason, metrics)

    return review


async def write_question(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    *,
    kind: Kind,
    difficulty: Difficulty,
    topic: str | None,
    settings: InterviewSettings,
    level: Level | None = None,
) -> DraftQuestion:
    """Two model calls: the question (plus at most one repair), then ONE ground check of its
    expected points with the same pass the grade's items get (`verify.verified_items`).

    WHY ground-check here (design decision "grounded grading"): clients may show the points as
    the answer key, i.e. as what the notes say, so an unbacked point must never leave the server.
    A point the verifier rejects is dropped; a verifier error or an untrusted verifier drops all
    of them (fail closed): the question is still asked, and /grade judges against the notes."""
    level = level or settings.target_level
    tags = CallTags(level, kind, difficulty, Source.notes)
    prompt = question_prompt(notes, kind=kind, difficulty=difficulty, topic=topic, level=level)
    parsed = await ask_json(
        chat,
        prompt,
        QuestionReply,
        chunk_ids=[n.chunk_id for n in notes],
        review=review_question(notes),
        tags=tags,
    )
    reply = parsed.reply
    checked = await verified_items(
        chat, notes, reply.expected_points, verifier_trusted=settings.verifier_trusted, tags=tags
    )
    points = tuple(p for p in checked if p is not None)
    log.info(
        "interview_question_written",
        **tags.fields(),
        notes=len(notes),
        degraded=parsed.degraded,
        points_given=len(reply.expected_points),
        points_kept=len(points),
    )
    return DraftQuestion(
        question=strip_labels(reply.question),
        kind=kind,
        topic=reply.topic,
        difficulty=difficulty,
        notes=tuple(notes),
        expected_points=points,
        level=level,
        source=Source.notes,
    )


async def write_general_question(
    chat: ChatModel,
    *,
    topic: str,
    kind: Kind,
    difficulty: Difficulty,
    level: Level,
    asked_before: Sequence[str] = (),
) -> DraftQuestion:
    """ONE model call (plus at most one repair) and nothing else: no notes, no retrieval, no
    verifier (there is nothing to verify against). Every text is label-free (`strip_labels`):
    a marker would claim a note that does not exist. `asked_before` is client DATA, bounded by
    `prompts.recent_questions` before it reaches the prompt."""
    tags = CallTags(level, kind, difficulty, Source.general)
    recent = recent_questions(asked_before)
    prompt = general_question_prompt(
        topic=topic, kind=kind, difficulty=difficulty, level=level, asked_before=recent
    )
    parsed = await ask_json(
        chat,
        prompt,
        QuestionReply,
        chunk_ids=(),
        review=_review_general,
        tags=tags,
        log_extra={"excluded_given": len(asked_before), "excluded_used": len(recent)},
    )
    reply = parsed.reply
    points = tuple(p for p in (strip_labels(p) for p in reply.expected_points) if p)
    log.info(
        "interview_question_written",
        **tags.fields(),
        notes=0,
        degraded=parsed.degraded,
        points_given=len(reply.expected_points),
        points_kept=len(points),
    )
    return DraftQuestion(
        question=strip_labels(reply.question),
        kind=kind,
        topic=strip_labels(reply.topic) or topic,
        difficulty=difficulty,
        notes=(),
        expected_points=points,
        level=level,
        source=Source.general,
    )


def _review_general(payload: Mapping[str, object]) -> Reviewed[QuestionReply]:
    """The schema (points salvaged as in notes mode) and the question-length rule; the points
    are the model's own key, so there is no backing to check."""
    reply, problem, reason = _parse(payload)
    if reply is None:
        return Reviewed(None, problem, reason)
    if len(strip_labels(reply.question)) < MIN_QUESTION_CHARS:
        return Reviewed(None, "question: too short", QUESTION_TOO_SHORT)
    return Reviewed(reply, problem, reason)
