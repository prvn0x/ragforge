"""Verification of what the grader says the NOTES say, with the chat path's own checks.

Nothing here re-implements a rule: every decision is a call into the public functions the chat
path's `verify_citations` / `ground_check` / `finalize` nodes use (agent/citations.py,
coverage.py, attribution.py, grounding.py), read-only. So "verified" means the same thing on a
model answer as on a chat answer (ADR-0015): the ground check calls it supported AND its residue
is empty. Two entry points:

- `check_answer`: the model answer, exactly the chat pipeline: normalize markers -> rule 1
  (labels name shown notes) -> attribution (inferred citations, code backing) -> one ground
  check on the annotated text -> `settle` -> the untrusted-verifier downgrade of `finalize`.
- `verified_items`: /question's expected points, and the grade's `missed` points and
  misconception corrections. Each must carry
  its own [S<n>] label on every sentence (strict rule 1), back its code, and survive ONE ground
  check over all items together. An item that fails any step is dropped, never shown.

Fail closed: a verifier that errors, or that cannot be read (inconclusive, or claims it cannot
place), never yields "verified": the answer is `unverified` and every item is dropped.

Dotted names (Phase 3c): every check here passes `names = coverage.dotted_names` of the SHOWN
notes' text (the DOT seam), so a setting name the notes contain ("max.in.flight.requests.per.
connection", "spring.datasource.url") is one token, not two sentences. Without it the cut half
carried no label and a correct expected point failed the strict check: the likely root cause of
the live "QuestionReply did not match the schema twice" 502 on Spring Boot notes.

Only for source=notes: general-knowledge mode has no notes and never calls anything here (with
empty notes every call would be meaningless; `question`/`grade` keep the two paths apart).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ragforge_core.agent.attribution import (
    NOT_FOUND,
    annotate,
    attribute,
    inferred_labels,
    no_provenance,
    settle,
)
from ragforge_core.agent.citations import check_citations, normalize_citations
from ragforge_core.agent.coverage import Unit, dotted_names, segment, valid_labels
from ragforge_core.agent.fencing import fence_hits
from ragforge_core.agent.grounding import GroundResult, ground_check
from ragforge_core.agent.interview.model import CallTags, CheckedAnswer, FlaggedClaim
from ragforge_core.agent.interview.replies import UNTAGGED, log_model_call
from ragforge_core.agent.labels import CITATION_MARKER
from ragforge_core.agent.state import HitDict, hit_to_state
from ragforge_core.domain.models import (
    ChatTurnUsage,
    ClaimReasonCode,
    RankedChunk,
    Verification,
)
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.logging import get_logger
from ragforge_core.prompts import get_prompt

log = get_logger(__name__)

_VERIFIER: Final = "ground_check"


@dataclass(frozen=True, slots=True)
class _Notes:
    """The notes as the chat path's checks take them: hit dicts labelled S1..Sn in order."""

    hits: list[HitDict]
    shown: list[int]
    valid: frozenset[str]
    evidence: str  # the fenced <sources> block the verifier reads
    names: frozenset[str]  # dotted names of the shown notes (`coverage.dotted_names`)

    @classmethod
    def of(cls, notes: Sequence[RankedChunk]) -> _Notes:
        hits = [hit_to_state(n) for n in notes]
        shown = list(range(1, len(hits) + 1))
        names = dotted_names(n.display_text for n in notes)
        return cls(hits, shown, valid_labels(shown, 0), fence_hits(hits, shown), names)


def _answer(
    text: str,
    status: Verification,
    flagged: dict[str, ClaimReasonCode] | None = None,
    cited: Sequence[int] = (),
    inferred: Sequence[int] = (),
) -> CheckedAnswer:
    claims = tuple(FlaggedClaim(t, r) for t, r in (flagged or {}).items())
    return CheckedAnswer(text, status, claims, tuple(cited), tuple(inferred))


def _not_found(claims: Sequence[str]) -> dict[str, ClaimReasonCode]:
    return dict.fromkeys(claims, NOT_FOUND)


async def _ground(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    evidence: str,
    text: str,
    also: str = "",
    tags: CallTags = UNTAGGED,
) -> GroundResult | None:
    """One ground check, logged like every interview model call; None when the verifier itself
    failed (the caller fails closed). Cancellation is not swallowed (only `Exception`)."""
    started = time.perf_counter()
    tag = get_prompt(_VERIFIER).tag
    chunk_ids = [n.chunk_id for n in notes]
    try:
        result = await ground_check(chat, evidence, text, also=also)
    except Exception as exc:
        log_model_call(
            prompt=tag, chat=chat, started=started, chunk_ids=chunk_ids,
            outcome="error", tags=tags, error=type(exc).__name__,
        )  # fmt: skip
        return None
    outcome = f"verdict:{result.status.value}" if result.conclusive else "malformed"
    log_model_call(
        prompt=tag, chat=chat, started=started, chunk_ids=chunk_ids, outcome=outcome, tags=tags,
        usage=total_usage(result.usage), attempt=len(result.usage) or 1,
        located=result.located, claims=len(result.unsupported_claims),
    )  # fmt: skip
    return result


def total_usage(usage: Sequence[ChatTurnUsage]) -> ChatTurnUsage | None:
    """The token usage of several calls to one model, summed (None for no call)."""
    if not usage:
        return None
    first = usage[0]
    return ChatTurnUsage(
        first.provider,
        first.model,
        sum(u.prompt_tokens for u in usage),
        sum(u.completion_tokens for u in usage),
    )


async def check_answer(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    raw: str,
    *,
    verifier_trusted: bool,
    tags: CallTags = UNTAGGED,
) -> CheckedAnswer:
    """The model answer, verified exactly like a chat answer (see the module docstring)."""
    known = _Notes.of(notes)
    draft = normalize_citations(raw)
    if not draft.strip():
        return _answer(draft, Verification.unsupported)
    report = check_citations(draft, known.shown, n_tool_results=0, strict=False)
    if not report.ok:  # rule 1: a label naming no note
        return _answer(draft, Verification.unsupported, _not_found(report.offending))
    found = attribute(draft, known.hits, known.shown, known.valid, names=known.names)
    if found.refusal_only:  # makes no claim
        return _answer(draft, Verification.unverified)
    if no_provenance(found, tools_ran=False):
        # Prominent flags for all: nothing is covered, so muting any of them would understate it.
        return _answer(draft, Verification.unsupported, _not_found(found.uncited))
    view = annotate(draft, found.inferred)
    result = await _ground(chat, notes, known.evidence, view.text, also=draft, tags=tags)
    if result is None:
        # Fail closed: nothing was checked. Rule-4a style: show the model's own labels, no
        # server-inferred provenance, and everything attribution already rejected is flagged.
        unchecked = [*found.uncited, *found.code_unbacked, *(i.sentence for i in found.inferred)]
        return _answer(draft, Verification.unverified, _not_found(unchecked), report.cited)
    claims = [c if c in draft else view.to_answer(c) for c in result.unsupported_claims]
    settled = settle(
        draft,
        verdict=result.status,
        conclusive=result.conclusive,
        located=result.located and all(claims),
        verifier_claims=tuple(dict.fromkeys(c for c in claims if c)),
        valid=known.valid,
        attribution=found,
        tools_ran=False,
        names=known.names,
    )
    status = settled.status
    flagged = _flags(settled.unsupported_claims, settled.reasons)
    if status is Verification.verified:
        flagged = {}
    inferred = list(settled.inferred)
    if not verifier_trusted:
        # finalize's rule 7: a verifier below its SLO certifies nothing, so no "verified" badge
        # and no server-inferred provenance; those sentences are flagged instead.
        if status in (Verification.verified, Verification.partial):
            status = Verification.unverified
        flagged.update(_not_found([i.sentence for i in inferred]))
        inferred = []
    return _answer(draft, status, flagged, report.cited, inferred_labels(inferred, report.cited))


def _flags(claims: Sequence[str], reasons: Sequence[ClaimReasonCode]) -> dict[str, ClaimReasonCode]:
    """Claims with their reasons; a producer without reasons (or misaligned) means "not_found"
    for all: a missing reason must never mute a warning (chat_stream does the same)."""
    if len(reasons) != len(claims):
        return _not_found(claims)
    flagged: dict[str, ClaimReasonCode] = {}
    for claim, reason in zip(claims, reasons, strict=True):
        flagged[claim] = NOT_FOUND if flagged.get(claim) == NOT_FOUND else reason
    return flagged


def lexically_backed(item: str, notes: Sequence[RankedChunk]) -> bool:
    """The CPU-only half of `verified_items`, also the cheap reply check of /question: every
    prose unit carries a label naming a shown note (strict rule 1), at least one note is cited,
    and every code span or block is backed by the cited note (attribution's code rule). It does
    NOT prove the note states the item: that is the ground check's job."""
    return _backed(item, _Notes.of(notes))


def _backed(item: str, known: _Notes) -> bool:
    text = normalize_citations(item)
    # rule 1 (every label names a shown note, no stray citation-shaped fragment) ...
    report = check_citations(text, known.shown, n_tool_results=0, strict=False)
    if not report.ok or not report.cited:
        return False
    # ... plus strict mode's "every prose unit has a label", over the SAME units attribution
    # reads, cut with the notes' dotted names. WHY not `check_citations(strict=True)`: it
    # segments without `names` (citations.py is outside the DOT seam), so it would still cut
    # "spring.datasource.url" and reject the point; this is its `needs_label` rule verbatim.
    if any(_needs_label(unit) for unit in segment(text, names=known.names)):
        return False
    found = attribute(text, known.hits, known.shown, known.valid, names=known.names)
    return found.covered and not found.uncited and not found.code_unbacked


def _needs_label(unit: Unit) -> bool:
    return not unit.code and unit.substantive and unit.allowed is None and not unit.labels


def recover_label(item: str, notes: Sequence[RankedChunk]) -> str | None:
    """An UNLABELLED item with the label of the one shown note that lexically backs it, or None.

    The commonest way a model loses an expected point is dropping its label. Attribution can
    propose one (the chat path's inferred citations); it is applied here only when EXACTLY ONE
    note backs every unit of the item (ambiguity is never resolved by guessing), and the result
    must then pass `lexically_backed` like any labelled point. It still faces the ground check
    (`verified_items`), so this recovers wording, never certifies content."""
    text = " ".join(normalize_citations(item).split())
    if not text or CITATION_MARKER.search(text):
        return None
    known = _Notes.of(notes)
    backers = [label for label in known.shown if _only(text, known, label)]
    if len(backers) != 1:
        return None
    body = text.rstrip(".")
    labelled = f"{body} [S{backers[0]}]" + ("." if body != text else "")
    return labelled if _backed(labelled, known) else None


def _only(text: str, known: _Notes, label: int) -> bool:
    """Every substantive unit of `text` is attributed to note `label` alone."""
    found = attribute(text, known.hits, [label], known.valid, names=known.names)
    return bool(found.inferred) and not found.uncited and not found.code_unbacked


async def verified_items(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    items: Sequence[str],
    *,
    verifier_trusted: bool,
    tags: CallTags = UNTAGGED,
) -> tuple[str | None, ...]:
    """Each item as verified (markers normalized, one line), or None when it is dropped: the
    lexical pass, then ONE ground check over the surviving items (one bullet each). An item the
    verifier flags (any claim overlapping it) is dropped; an "unsupported", inconclusive or
    unlocatable verdict, or a verifier error, drops every item (fail closed: the verdict cannot
    be pinned on the items it is about).

    `verifier_trusted=False` drops every item WITHOUT calling the verifier. WHY: these items are
    shown only because they are verified (the wire has no per-item status), and ADR-0015 rule 7
    says a verifier below its SLO certifies nothing; `check_answer` applies the same rule by
    downgrading the model answer to "unverified". The lexical pass alone proves no statement."""
    flat = [" ".join(normalize_citations(item).split()) for item in items]
    if not verifier_trusted:
        if items:
            log.info("interview_items_unchecked", items=len(items), reason="verifier_untrusted")
        return tuple(None for _ in items)
    known = _Notes.of(notes)
    candidates = [text for text in flat if _backed(text, known)]
    if not candidates:
        return tuple(None for _ in items)
    doc, spans = _bullets(candidates)
    result = await _ground(chat, notes, known.evidence, doc, tags=tags)
    if (
        result is None
        or result.status is Verification.unsupported
        or not (result.conclusive and result.located)
    ):
        return tuple(None for _ in items)
    flagged = [s for claim in result.unsupported_claims for s in _occurrences(doc, claim)]
    kept = {
        text
        for text, span in zip(candidates, spans, strict=True)
        if not any(_overlap(span, f) for f in flagged)
    }
    return tuple(text if text in kept else None for text in flat)


def _bullets(items: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """The (single-line) items as one Markdown list, and each item's [start, end) in it."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    at = 0
    for item in items:
        line = f"- {item}"
        spans.append((at + 2, at + len(line)))
        parts.append(line)
        at += len(line) + 1
    return "\n".join(parts), spans


def _occurrences(text: str, claim: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = text.find(claim) if claim else -1
    while start != -1:
        out.append((start, start + len(claim)))
        start = text.find(claim, start + 1)
    return out


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
