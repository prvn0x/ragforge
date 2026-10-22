"""Grade one answer: against the notes (one grading call, then the verification calls), or from
general knowledge (one judge call, nothing verified).

Eval seams: `grade_answer(chat, notes, request, settings)` and `grade_general(chat, request)` take
a fake or real chat model (and fixed notes), with no HTTP, database or sanitizer.

Notes mode calls `interview_grade@2` (plus at most one
repair), then, CONCURRENTLY, one ground check of the model answer and one of the items (`missed`
points and misconception corrections). The two checks are independent, so running them together
keeps the grade at two model round trips of latency.

What is verified and what is judgement (`model.DraftGrade`): `model_answer`, `missed` and the
misconception corrections claim to state what the NOTES say, so they pass the chat path's
verification (`verify.py`); an item that fails is dropped, never shown (with an untrusted verifier,
all of them are). A misconception's `said` must quote the user's answer (`quoted`). `score`,
`summary` and `right` judge the USER's answer: nothing in the notes can confirm them, so they are
bounded by the schema (`replies.GradeReply`) and labelled as the model's judgement by clients.

General mode (`grade_general`, source=general) calls `interview_general_grade@1` once (plus at
most one repair) and NOTHING else: there are no notes, so `verify` is never called (it would be
meaningless with none) and the answer is "unverified" ALWAYS, with no flagged claims and no
citations. The client-echoed question and key are DATA (bound by the question token in the
route); the prompt treats the key as a hint the model's knowledge may override. Every text is
label-free (`strip_labels`): a marker would claim a note that does not exist.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

from ragforge_core.agent.interview.model import (
    CallTags,
    CheckedAnswer,
    DraftGrade,
    GradeRequest,
    InterviewSettings,
    Misconception,
    Source,
)
from ragforge_core.agent.interview.prompts import general_grade_prompt, grade_prompt
from ragforge_core.agent.interview.question import strip_labels
from ragforge_core.agent.interview.replies import GradeReply, ask_json
from ragforge_core.agent.interview.verify import check_answer, verified_items
from ragforge_core.domain.models import RankedChunk, Verification
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)


async def grade_answer(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    request: GradeRequest,
    settings: InterviewSettings,
) -> DraftGrade:
    tags = _tags(request, Source.notes)
    prompt = grade_prompt(
        notes,
        question=request.question,
        expected_points=request.expected_points,
        answer=request.answer,
        level=request.level,
    )
    chunk_ids = [n.chunk_id for n in notes]
    reply = (await ask_json(chat, prompt, GradeReply, chunk_ids=chunk_ids, tags=tags)).reply
    trusted = settings.verifier_trusted
    # A misconception must quote the user: one whose `said` is not in the answer is dropped
    # before any verifier call (no point checking a correction of something never said).
    said = [(quoted(m.said, request.answer), m.correction) for m in reply.misconceptions]
    wrong = [(q, correction) for q, correction in said if q is not None]
    items = [*reply.missed, *(correction for _, correction in wrong)]
    answer, kept = await asyncio.gather(
        check_answer(chat, notes, reply.model_answer, verifier_trusted=trusted, tags=tags),
        verified_items(chat, notes, items, verifier_trusted=trusted, tags=tags),
    )
    missed = tuple(text for text in kept[: len(reply.missed)] if text is not None)
    misconceptions = tuple(
        Misconception(said=q, correction=text)
        for (q, _), text in zip(wrong, kept[len(reply.missed) :], strict=True)
        if text is not None
    )
    log.info(
        "interview_graded",
        **tags.fields(),
        score=reply.score,
        verification=answer.status.value,
        flagged=len(answer.claims),
        missed_given=len(reply.missed),
        missed_kept=len(missed),
        misconceptions_given=len(reply.misconceptions),
        misconceptions_quoted=len(wrong),
        misconceptions_kept=len(misconceptions),
    )
    return DraftGrade(
        score=reply.score,
        summary=reply.summary,
        right=tuple(reply.right),
        missed=missed,
        misconceptions=misconceptions,
        answer=answer,
        notes=tuple(notes),
        level=request.level,
        source=Source.notes,
    )


def _tags(request: GradeRequest, source: Source) -> CallTags:
    return CallTags(request.level, request.kind, request.difficulty, source)


async def grade_general(chat: ChatModel, request: GradeRequest) -> DraftGrade:
    """One judge call against the client-echoed question and key (see the module docstring)."""
    tags = _tags(request, Source.general)
    prompt = general_grade_prompt(
        question=request.question,
        expected_points=request.expected_points,
        answer=request.answer,
        level=request.level,
    )
    reply = (await ask_json(chat, prompt, GradeReply, chunk_ids=(), tags=tags)).reply
    said = (
        (quoted(m.said, request.answer), strip_labels(m.correction)) for m in reply.misconceptions
    )
    misconceptions = tuple(
        Misconception(said=q, correction=correction) for q, correction in said if q and correction
    )
    log.info(
        "interview_graded",
        **tags.fields(),
        score=reply.score,
        verification=Verification.unverified.value,
        misconceptions_given=len(reply.misconceptions),
        misconceptions_kept=len(misconceptions),
    )
    return DraftGrade(
        score=reply.score,
        summary=strip_labels(reply.summary),
        right=_plain_items(reply.right),
        missed=_plain_items(reply.missed),
        misconceptions=misconceptions,
        answer=CheckedAnswer(strip_labels(reply.model_answer), Verification.unverified, (), (), ()),
        notes=(),
        level=request.level,
        source=Source.general,
    )


def _plain_items(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(text for text in (strip_labels(i) for i in items) if text)


_EDGE: Final = "\"'`\u201c\u201d\u2018\u2019.,;:!? \t\n"


def _plain(text: str) -> str:
    return " ".join(strip_labels(text).split()).strip(_EDGE).casefold()


def quoted(said: str, answer: str) -> str | None:
    """`said` as a one-line quote of the user's answer, or None when the answer does not
    contain it (compared without citation markers, case, outer quotes/punctuation and spacing
    differences).

    WHY: `said` is model-written text rendered next to a VERIFIED correction, and the wire
    merges both into one string, so a client cannot style them apart. Requiring it to be the
    user's own words stops the model from attributing a statement the user never made, and
    removing markers stops it from looking like note content."""
    plain = _plain(said)
    if not plain or plain not in _plain(answer):
        return None
    return " ".join(strip_labels(said).split())
