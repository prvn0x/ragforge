"""/v1/eval/verify's pipeline (ADR-0017 §1, §3 E-V): the chat path's verification of ONE supplied
answer over supplied shown notes, returning the verdict AND each layer's evidence before merging.

WHY it exists: E-V measures the guarantee as shipped (ADR-0017 principle 2), so it must run the
same checks a chat answer goes through, in the same order, with the same settings:

    normalize markers -> rule 1 (check_citations, strict as configured, dotted names)
    -> attribute -> [short-circuits] -> ONE ground check on the annotated text -> settle

It stops at `settle` on purpose: `finalize`'s rule 7 (an untrusted verifier certifies nothing,
so `verified`/`partial` become `unverified`) is NOT applied. E-V is the measurement that decides
whether a verifier model may be trusted (ADR-0017 §3: badge-flip recall is computed over the
sources whose clean item is `verified` in the same run); applying the downgrade of the model
under test would empty that set, so an untrusted model could never qualify. A client that wants
the chat badge of an untrusted server applies rule 7 itself from `status` (review round 1, P1).

Nothing here re-implements a rule: every decision is a call into the public functions the chat
nodes `verify_citations` / `ground_check` / `finalize` use (citations.py, coverage.py,
attribution.py, grounding.py), read-only, exactly like `interview/verify.check_answer`. What is
added is only OBSERVATION:

- coverage evidence: the deterministic layer's own flags, computed as if the verifier passed
  every inferred citation (attribution's uncited and code-unbacked units, rule-1 offenders, and
  the residue with every candidate kept). WHY not `settle`'s residue: that one also holds units
  that lost their inferred citation because of the VERIFIER (an overlapping claim, a closed
  gate), and counting those as coverage catches would credit the wrong layer in `caught_by`.
- verifier evidence: every claim the verifier returned, from its raw reply (`_Recorder`), each
  placed in the answer or `None` when it cannot be (a paraphrase `parse_verdict` discards, a
  claim that was nothing but an inserted marker): ADR-0015 rule 4a's "unplaced objection".
- every span is [start, end) in the answer AS SUPPLIED. Normalization may rewrite markers
  (`[1]` -> `[S1]`), so positions found in the normalized draft are mapped back through the
  rewrites themselves (`_PositionMap`), never assumed equal. WHY not a text alignment: difflib
  was roughly cubic here (4 s on 3,000 characters, minutes on the 20,000 the contract allows),
  synchronously inside the async route (review round 1, P1); the rewrite log is linear.

Provider errors propagate (the route maps them like /v1/chat: 502/503), so the harness can
classify them as provider-transient instead of reading a fail-closed `unverified` as a verdict.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from ragforge_core.agent.attribution import (
    NOT_FOUND,
    Attribution,
    Settled,
    annotate,
    attribute,
    merge_reasons,
    no_provenance,
    settle,
)
from ragforge_core.agent.citations import (  # the private names: see _normalize_pass
    _LENIENT,
    _MAX_PASSES,
    _RANGE,
    CitationReport,
    _canonical,
    _expand_range,
    check_citations,
    normalize_citations,
)
from ragforge_core.agent.coverage import dotted_names, residue, valid_labels
from ragforge_core.agent.fencing import fence_hits
from ragforge_core.agent.grounding import GroundResult, ground_check
from ragforge_core.agent.state import HitDict, hit_to_state
from ragforge_core.agent.structure import structure
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    ClaimReasonCode,
    RankedChunk,
    ToolSpec,
    Verification,
)
from ragforge_core.domain.ports import ChatModel
from ragforge_core.prompts import get_prompt

VERIFIER_PROMPT: Final = "ground_check"

# Coverage-unit kinds. The first three are the contract's CoverageUnitKind; `invalid_citation`
# (rule 1: a label naming no shown note, a stray citation fragment, or, in strict mode, a prose
# unit with no label) is an additive value of that OPEN enum (docs/handoff/EV-API.md).
RESIDUE: Final = "residue"
UNCITED: Final = "uncited"
CODE_UNBACKED: Final = "code_unbacked"
INVALID_CITATION: Final = "invalid_citation"


@dataclass(frozen=True, slots=True)
class Span:
    """[start, end) in the supplied answer; `text` is answer[start:end]."""

    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CoverageUnit:
    kind: str
    span: Span


@dataclass(frozen=True, slots=True)
class VerifierClaim:
    text: str  # as the verifier wrote it (stripped)
    span: Span | None  # None: the server could not place it (ADR-0015 rule 4a)


@dataclass(frozen=True, slots=True)
class FlaggedSpan:
    span: Span
    reason: ClaimReasonCode


@dataclass(frozen=True, slots=True)
class EvalVerdict:
    status: Verification
    claims: tuple[FlaggedSpan, ...]  # settle's merged flags (a trusted server's chat badge)
    coverage_units: tuple[CoverageUnit, ...]
    verifier_claims: tuple[VerifierClaim, ...] = ()
    verifier_ran: bool = False
    verifier_verdict: Verification | None = None
    verifier_conclusive: bool | None = None
    verifier_model: str | None = None
    verifier_prompt: str | None = None


# ----------------------------------------------------------------- positions
@dataclass(frozen=True, slots=True)
class _Rewrite:
    """One label span a normalization pass changed: [src_start, src_end) of the pass's input
    became [dst_start, dst_end) of its output. Text between rewrites is copied verbatim."""

    src_start: int
    src_end: int
    dst_start: int
    dst_end: int


class _Pass:
    """Maps positions of one pass's OUTPUT back onto its input in O(log n) per position.

    Outside a rewrite a position moves by the length change of the rewrites before it; a span
    edge INSIDE a rewrite widens to the whole rewritten block (a start to its beginning, an end
    to its end), so a mapped span never cuts a marker in half."""

    def __init__(self, rewrites: Sequence[_Rewrite]) -> None:
        self._rewrites = tuple(rewrites)
        self._starts = [r.dst_start for r in self._rewrites]

    def start(self, pos: int) -> int:
        i = bisect_right(self._starts, pos) - 1  # the last rewrite starting at or before pos
        if i < 0:
            return pos
        r = self._rewrites[i]
        return r.src_start if pos < r.dst_end else r.src_end + (pos - r.dst_end)

    def end(self, pos: int) -> int:
        i = bisect_left(self._starts, pos) - 1  # the last rewrite starting strictly before pos
        if i < 0:
            return pos
        r = self._rewrites[i]
        return r.src_end if pos <= r.dst_end else r.src_end + (pos - r.dst_end)


_Edit = tuple[int, int, int, int]  # (src_start, src_end, dst_start, dst_end), segment-relative


def _logged_sub(
    pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], text: str
) -> tuple[str, list[_Edit]]:
    """`pattern.sub(repl, text)`, also returning each match it changed (re.sub replaces exactly
    the non-overlapping matches finditer yields, in order)."""
    out: list[str] = []
    edits: list[_Edit] = []
    pos = at = 0
    for match in pattern.finditer(text):
        new = repl(match)
        out.append(text[pos : match.start()])
        at += match.start() - pos
        if new != match.group(0):
            edits.append((match.start(), match.end(), at, at + len(new)))
        out.append(new)
        at += len(new)
        pos = match.end()
    out.append(text[pos:])
    return "".join(out), edits


def _normalize_pass(text: str) -> tuple[str, tuple[_Pass, _Pass]]:
    """`citations._normalize_once`, logging every marker it rewrote. It rewrites only
    `structure(text).label_spans`, each by `_rewrite` (a range expansion, then the lenient-marker
    rewrite, both plain regex substitutions), so logging the two substitutions is exact; the two
    returned passes map the output back through the intermediate text. `_PositionMap` still
    checks the result against `normalize_citations`, the chat path's own call."""
    out: list[str] = []
    ranges: list[_Rewrite] = []
    markers: list[_Rewrite] = []
    pos = mid = dst = 0
    for start, end in structure(text).label_spans:
        out.append(text[pos:start])
        mid += start - pos
        dst += start - pos
        expanded, first = _logged_sub(_RANGE, _expand_range, text[start:end])
        canonical, second = _logged_sub(_LENIENT, _canonical, expanded)
        ranges.extend(_Rewrite(start + a, start + b, mid + c, mid + d) for a, b, c, d in first)
        markers.extend(_Rewrite(mid + a, mid + b, dst + c, dst + d) for a, b, c, d in second)
        out.append(canonical)
        mid += len(expanded)
        dst += len(canonical)
        pos = end
    out.append(text[pos:])
    return "".join(out), (_Pass(ranges), _Pass(markers))


class NormalizationDrift(RuntimeError):
    """The rewrite log no longer reproduces `normalize_citations`: a server bug (citations.py
    changed how it rewrites). Raised, never papered over, because a span mapped through a wrong
    log would silently credit the wrong layer in `caught_by`."""


class _PositionMap:
    """Maps [start, end) of the normalized draft onto the supplied answer by composing each
    fixpoint pass's rewrite log (`normalize_citations` runs up to `_MAX_PASSES` passes). Linear
    in the answer to build, O(passes x log rewrites) per span. An answer that already uses
    `[S<n>]` has no rewrite and maps to itself."""

    def __init__(self, answer: str, draft: str) -> None:
        passes: list[_Pass] = []
        text = answer
        for _ in range(_MAX_PASSES):
            normalized, steps = _normalize_pass(text)
            if normalized == text:
                break
            passes.extend(steps)
            text = normalized
        if text != draft:
            raise NormalizationDrift("the marker rewrite log does not reproduce the draft")
        self._passes = tuple(reversed(passes))  # draft -> ... -> answer

    def span(self, start: int, end: int) -> tuple[int, int]:
        for step in self._passes:
            start, end = step.start(start), step.end(end)
        return start, max(start, end)


@dataclass(frozen=True, slots=True)
class _Texts:
    """The supplied answer, the normalized draft and the map between them."""

    answer: str
    draft: str
    positions: _PositionMap

    def spans(self, text: str) -> list[Span]:
        """Every occurrence of `text` in the draft, as spans of the answer. Claims and units are
        verbatim substrings of the draft (grounding.parse_verdict, coverage.segment), so an
        occurrence is a positional fact; a text with several occurrences yields several spans."""
        out: list[Span] = []
        at = self.draft.find(text) if text else -1
        while at != -1:
            start, end = self.positions.span(at, at + len(text))
            out.append(Span(self.answer[start:end], start, end))
            at = self.draft.find(text, at + 1)
        return out


# ----------------------------------------------------------------- the verifier's raw reply
class _Recorder:
    """A ChatModel that delegates to the real one and keeps every JSON reply it returned, so the
    claims `parse_verdict` drops (paraphrases) can still be reported, unplaced. Reading replies
    is all it does: the verifier call, its retry policy and its parsing stay `ground_check`'s."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.replies: list[dict[str, object]] = []

    def stream(
        self, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        return self.inner.stream(system, messages, tools)  # unrecorded: the verifier never streams

    async def complete_json(
        self, system: str, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        payload, usage = await self.inner.complete_json(system, messages, schema)
        self.replies.append(payload)
        return payload, usage


def _raw_claims(recorder: _Recorder, result: GroundResult) -> list[str]:
    """The claims of the reply `ground_check` accepted: its LAST reply, since it returns right
    after the first one that parses. None for an inconclusive check (no reply was accepted, so
    no claim of it is the verifier's verdict)."""
    if not result.conclusive or not recorder.replies:
        return []
    claims = recorder.replies[-1].get("unsupported_claims")
    if not isinstance(claims, list):
        return []
    return list(dict.fromkeys(c.strip() for c in claims if isinstance(c, str) and c.strip()))


# ----------------------------------------------------------------- the pipeline
@dataclass(frozen=True, slots=True)
class _Notes:
    """The shown notes as the chat path's checks take them: hits labelled S1..Sn in order."""

    hits: list[HitDict]
    shown: list[int]
    valid: frozenset[str]
    names: frozenset[str]

    @classmethod
    def of(cls, notes: Sequence[RankedChunk]) -> _Notes:
        shown = list(range(1, len(notes) + 1))
        return cls(
            [hit_to_state(n) for n in notes],
            shown,
            valid_labels(shown, 0),
            dotted_names(n.display_text for n in notes),
        )


async def verify_answer(
    chat: ChatModel,
    notes: Sequence[RankedChunk],
    answer: str,
    *,
    strict: bool,
) -> EvalVerdict:
    """The full pipeline on `answer`, whose labels name `notes` in order (S1 = notes[0]).

    `strict` is the server's configured citation mode (settings.agent_strict_citations): a
    supplied answer is a first draft, never a strict regeneration. `status` is `settle`'s, with
    no untrusted-verifier downgrade (module docstring); the verifier's own verdict is in
    `verifier_verdict`. Short-circuits (empty answer, rule 1, a pure refusal, no provenance) run
    no verifier: `verifier_ran` is False and the verifier fields stay None."""
    if not answer.strip():
        return EvalVerdict(Verification.unsupported, (), ())
    known = _Notes.of(notes)
    draft = normalize_citations(answer)
    texts = _Texts(answer, draft, _PositionMap(answer, draft))
    report = check_citations(draft, known.shown, n_tool_results=0, strict=strict, names=known.names)
    found = attribute(draft, known.hits, known.shown, known.valid, names=known.names)
    units = _coverage_units(texts, known, report, found)
    if not report.ok:  # rule 1: unsupported before any verifier call is spent on it
        return _short(Verification.unsupported, texts, report.offending, units)
    if found.refusal_only:  # a refusal makes no claim: nothing to check, nothing flagged
        return EvalVerdict(Verification.unverified, (), ())
    if no_provenance(found, tools_ran=False):
        return _short(Verification.unsupported, texts, found.uncited, units)
    return await _ground(chat, known, texts, found, units)


def _short(
    status: Verification, texts: _Texts, flagged: Iterable[str], units: tuple[CoverageUnit, ...]
) -> EvalVerdict:
    """A short-circuit: every flag "not_found" (nothing was verified, so nothing is muted;
    `nodes.unverified_flags`, `attribution.all_not_found`)."""
    claims, reasons = merge_reasons((c, NOT_FOUND) for c in flagged)
    return EvalVerdict(status, _flagged(texts, claims, reasons), units)


async def _ground(
    chat: ChatModel,
    known: _Notes,
    texts: _Texts,
    found: Attribution,
    units: tuple[CoverageUnit, ...],
) -> EvalVerdict:
    """`nodes.ground_check` + `settle` for one answer with no tool results."""
    draft = texts.draft
    view = annotate(draft, found.inferred)
    recorder = _Recorder(chat)
    evidence = fence_hits(known.hits, known.shown)
    result = await ground_check(recorder, evidence, view.text, also=draft)
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
    pairs: list[tuple[str, ClaimReasonCode]] = []
    if status is not Verification.verified:
        pairs.extend(zip(settled.unsupported_claims, _aligned(settled), strict=True))
    merged, reasons = merge_reasons(pairs)
    return EvalVerdict(
        status=status,
        claims=_flagged(texts, merged, reasons),
        coverage_units=units,
        verifier_claims=_verifier_claims(texts, view.to_answer, _raw_claims(recorder, result)),
        verifier_ran=True,
        verifier_verdict=result.status,
        verifier_conclusive=result.conclusive,
        # what the provider REPORTED; None when no reply came back (every attempt raised), never
        # the configured alias, which may since have moved to another model
        verifier_model=result.usage[-1].model if result.usage else None,
        verifier_prompt=get_prompt(VERIFIER_PROMPT).tag,
    )


def _aligned(settled: Settled) -> tuple[ClaimReasonCode, ...]:
    """`Settled.reasons`, or all "not_found" when they are not index-aligned (a missing reason
    must never mute a warning: `nodes._flagged`, `chat_stream`)."""
    claims, reasons = settled.unsupported_claims, settled.reasons
    return reasons if len(reasons) == len(claims) else (NOT_FOUND,) * len(claims)


def _flagged(
    texts: _Texts, claims: Sequence[str], reasons: Sequence[ClaimReasonCode]
) -> tuple[FlaggedSpan, ...]:
    return tuple(
        FlaggedSpan(span, reason)
        for claim, reason in zip(claims, reasons, strict=True)
        for span in texts.spans(claim)
    )


def _verifier_claims(
    texts: _Texts, to_answer: Callable[[str], str], raw: Sequence[str]
) -> tuple[VerifierClaim, ...]:
    """Each raw claim at every place it occurs, mapped back like `nodes.ground_check` maps it
    (a claim quoting the annotated view loses the inserted markers), or once with no span."""
    out: list[VerifierClaim] = []
    for claim in raw:
        located = claim if claim in texts.draft else to_answer(claim)
        spans = texts.spans(located) if located else []
        if not spans:
            out.append(VerifierClaim(claim, None))
        out.extend(VerifierClaim(claim, span) for span in spans)
    return tuple(out)


def _coverage_units(
    texts: _Texts, known: _Notes, report: CitationReport, found: Attribution
) -> tuple[CoverageUnit, ...]:
    """The deterministic layer's flags (module docstring), each text once under its most
    specific kind, at every place it occurs: rule-1 offenders, code-unbacked units, other
    uncited units, then whatever else the residue holds with EVERY candidate citation kept."""
    if found.refusal_only:
        return ()
    kept = [i.sentence for i in found.inferred]
    left = residue(texts.draft, known.valid, kept, names=known.names)
    kinds: dict[str, str] = {}
    for kind, group in (
        (INVALID_CITATION, report.offending),
        (CODE_UNBACKED, found.code_unbacked),
        (UNCITED, found.uncited),
        (RESIDUE, left),
    ):
        for text in group:
            kinds.setdefault(text, kind)
    return tuple(
        CoverageUnit(kind, span) for text, kind in kinds.items() for span in texts.spans(text)
    )


__all__ = [
    "CODE_UNBACKED",
    "INVALID_CITATION",
    "RESIDUE",
    "UNCITED",
    "VERIFIER_PROMPT",
    "CoverageUnit",
    "EvalVerdict",
    "FlaggedSpan",
    "NormalizationDrift",
    "Span",
    "VerifierClaim",
    "verify_answer",
]
