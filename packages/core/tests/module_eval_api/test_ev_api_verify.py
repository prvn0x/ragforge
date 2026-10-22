"""agent/eval_verify.py on the committed D0 verifier set (evals/demo/verifier.json) and synthetic
answers, with a scripted verifier: the per-layer evidence /v1/eval/verify returns (ADR-0017 §3
E-V computes `caught_by` from it in the client)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from module_eval_api.ev_api_support import (
    ScriptedVerifier,
    d0_item,
    d0_notes,
    d0_verifier_set,
    mutated_text,
    objects,
)
from ragforge_client.evals.schema import EXPECTED_LAYER, Layer, VerifierItem
from ragforge_core.agent.citations import normalize_citations
from ragforge_core.agent.eval_verify import (
    CODE_UNBACKED,
    INVALID_CITATION,
    UNCITED,
    EvalVerdict,
    NormalizationDrift,
    Span,
    _PositionMap,
    verify_answer,
)
from ragforge_core.domain.models import RankedChunk, Verification
from ragforge_core.prompts import get_prompt

CLEAN = [i for i in d0_verifier_set().items if i.mutation is None]
PLANTED = [i for i in d0_verifier_set().items if i.mutation is not None]
# D0's one clean item the SHIPPED coverage layer rejects before any verifier call: an unlabelled
# lead-in ("The runbook guards it ...:") over a code block that is not the page's whole block,
# with "[S1]" alone on its own line. A coverage-caused false reject (ADR-0017 §3 E-V), recorded
# here so a change to it is noticed, not silently absorbed.
COVERAGE_REJECTED_CLEAN = "c-deploy"


async def _verify(item: VerifierItem, chat: ScriptedVerifier | None = None) -> EvalVerdict:
    return await verify_answer(
        chat or ScriptedVerifier(), d0_notes(item), item.answer, strict=False
    )


def _overlaps(span: Span | None, start: int, end: int) -> bool:
    return span is not None and span.start < end and start < span.end


def _spans_consistent(answer: str, verdict: EvalVerdict) -> None:
    """Every reported span is a real [start, end) of the SUPPLIED answer holding its text."""
    spans = [
        *(u.span for u in verdict.coverage_units),
        *(c.span for c in verdict.claims),
        *(c.span for c in verdict.verifier_claims if c.span is not None),
    ]
    for span in spans:
        assert 0 <= span.start <= span.end <= len(answer)
        assert answer[span.start : span.end] == span.text


# ----------------------------------------------------------------- D0 clean items
@pytest.mark.parametrize("item", CLEAN, ids=lambda i: i.id)
async def test_d0_clean_items_with_a_supporting_verifier(item: VerifierItem) -> None:
    chat = ScriptedVerifier()
    verdict = await _verify(item, chat)
    _spans_consistent(item.answer, verdict)
    if item.id == COVERAGE_REJECTED_CLEAN:
        assert verdict.status is Verification.unsupported
        assert not verdict.verifier_ran
        assert not chat.calls
        assert {u.kind for u in verdict.coverage_units} == {UNCITED}
        return
    assert verdict.status is Verification.verified
    assert verdict.coverage_units == ()
    assert verdict.claims == ()
    assert verdict.verifier_ran
    assert verdict.verifier_verdict is Verification.verified
    assert verdict.verifier_conclusive is True
    assert verdict.verifier_claims == ()
    assert len(chat.calls) == 1  # ONE ground check, as on the chat path


async def test_verify_reports_the_model_and_prompt_that_actually_ran() -> None:
    verdict = await _verify(d0_item("c-durability"), ScriptedVerifier(usage_model="m@2026-10"))
    assert verdict.verifier_model == "m@2026-10"  # the provider's reported model, not config
    assert verdict.verifier_prompt == get_prompt("ground_check").tag == "ground_check@3"


async def test_no_reply_reports_no_model_rather_than_the_configured_alias() -> None:
    """Both attempts fail to parse at the adapter (ValueError: no usage): the configured alias
    is NOT what ran, so nothing is claimed (ADR-0017 §1: the model that actually ran)."""
    chat = ScriptedVerifier([ValueError("not JSON"), ValueError("not JSON")])
    verdict = await _verify(d0_item("c-durability"), chat)
    assert len(chat.calls) == 2
    assert verdict.verifier_ran
    assert verdict.verifier_conclusive is False
    assert verdict.verifier_model is None


# ----------------------------------------------------------------- D0 planted items
@pytest.mark.parametrize(
    "item",
    [i for i in PLANTED if i.mutation and EXPECTED_LAYER[i.mutation] is Layer.coverage],
    ids=lambda i: i.id,
)
async def test_coverage_layer_mutations_are_caught_without_the_verifier(
    item: VerifierItem,
) -> None:
    """A verifier that objects to nothing: the catch can only be coverage's own evidence."""
    assert item.span is not None
    verdict = await _verify(item)
    _spans_consistent(item.answer, verdict)
    assert any(_overlaps(u.span, item.span.start, item.span.end) for u in verdict.coverage_units)
    assert verdict.verifier_claims == ()
    assert verdict.status is not Verification.verified


@pytest.mark.parametrize("item", PLANTED, ids=lambda i: i.id)
async def test_a_verifier_objection_is_located_on_the_mutation(item: VerifierItem) -> None:
    """The verifier quotes the mutated text verbatim: its claim's span must land on the planted
    span (span-based catch, ADR-0017 r2 P0-B), and the badge must not stay `verified`."""
    assert item.span is not None
    chat = ScriptedVerifier([objects(mutated_text(item))])
    verdict = await _verify(item, chat)
    _spans_consistent(item.answer, verdict)
    if not verdict.verifier_ran:  # coverage rejected the whole answer before any verifier call
        assert not chat.calls
        assert any(
            _overlaps(u.span, item.span.start, item.span.end) for u in verdict.coverage_units
        )
        return
    located = [c for c in verdict.verifier_claims if c.span is not None]
    assert any(_overlaps(c.span, item.span.start, item.span.end) for c in located)
    assert verdict.status is not Verification.verified


async def test_a_supporting_verifier_misses_verifier_layer_mutations_by_construction() -> None:
    """The control for the test above: a number drift the verifier does not flag passes the
    deterministic layer (its number is not checked against the page by coverage), so a catch
    of it can only ever come from the verifier's evidence."""
    item = d0_item("p-number-2")
    verdict = await _verify(item)
    assert verdict.status is Verification.verified
    assert verdict.coverage_units == ()


# ----------------------------------------------------------------- unplaced claims
async def test_a_paraphrased_objection_is_reported_with_a_null_span() -> None:
    item = d0_item("c-durability")
    chat = ScriptedVerifier([objects("producers do not wait for every replica")])
    verdict = await _verify(item, chat)
    assert [(c.text, c.span) for c in verdict.verifier_claims] == [
        ("producers do not wait for every replica", None)
    ]
    # ADR-0015 rule 4a: an objection nobody can place closes the gate; never "verified"
    assert verdict.status is not Verification.verified
    assert verdict.verifier_verdict is Verification.partial


async def test_placed_and_unplaced_claims_are_reported_side_by_side() -> None:
    item = d0_item("c-durability")
    quoted = "a single broker loss never loses an acknowledged event"
    chat = ScriptedVerifier([objects(quoted, "something about acks")])
    verdict = await _verify(item, chat)
    placed = {c.text: c.span for c in verdict.verifier_claims}
    assert placed["something about acks"] is None
    span = placed[quoted]
    assert span is not None
    assert item.answer[span.start : span.end] == quoted


async def test_an_inconclusive_verifier_reports_no_claims() -> None:
    malformed: dict[str, object] = {"verdict": "maybe"}
    chat = ScriptedVerifier([malformed, malformed])
    verdict = await _verify(d0_item("c-durability"), chat)
    assert len(chat.calls) == 2  # ground_check's one repair retry, unchanged
    assert verdict.verifier_ran
    assert verdict.verifier_conclusive is False
    assert verdict.verifier_claims == ()  # no accepted reply: none of its claims is a verdict
    assert verdict.status is not Verification.verified


async def test_a_verifier_failure_propagates_for_the_harness_to_classify() -> None:
    chat = ScriptedVerifier([RuntimeError("provider down")])
    with pytest.raises(RuntimeError, match="provider down"):
        await _verify(d0_item("c-durability"), chat)


# ----------------------------------------------------------------- short-circuits
async def test_an_empty_answer_is_unsupported_without_a_verifier_call() -> None:
    chat = ScriptedVerifier()
    item = d0_item("c-durability")
    verdict = await verify_answer(chat, d0_notes(item), "  \n", strict=False)
    assert verdict.status is Verification.unsupported
    assert not verdict.verifier_ran
    assert verdict.verifier_verdict is None
    assert verdict.verifier_model is None
    assert not chat.calls


async def test_a_label_naming_no_shown_note_short_circuits_as_coverage() -> None:
    item = d0_item("c-durability")
    answer = "Acme producers set `acks=all` [S1]. Brokers keep three copies [S7]."
    chat = ScriptedVerifier()
    verdict = await verify_answer(chat, d0_notes(item), answer, strict=False)
    assert verdict.status is Verification.unsupported
    assert not verdict.verifier_ran
    assert not chat.calls
    bad = [u for u in verdict.coverage_units if u.kind == INVALID_CITATION]
    assert [u.span.text for u in bad] == ["Brokers keep three copies [S7]."]
    assert [c.span.text for c in verdict.claims] == ["Brokers keep three copies [S7]."]


async def test_strict_mode_as_configured_rejects_an_unlabelled_sentence() -> None:
    item = d0_item("c-durability")
    unlabelled = item.answer.replace(" [S1].", ".", 1)
    lenient = await verify_answer(ScriptedVerifier(), d0_notes(item), unlabelled, strict=False)
    strict = await verify_answer(ScriptedVerifier(), d0_notes(item), unlabelled, strict=True)
    assert lenient.verifier_ran  # attribution infers the label; the verifier checks it
    assert not strict.verifier_ran
    assert strict.status is Verification.unsupported


async def test_a_pure_refusal_is_unverified_and_flags_nothing() -> None:
    item = d0_item("c-durability")
    chat = ScriptedVerifier()
    verdict = await verify_answer(
        chat, d0_notes(item), "Your notes don't cover this.", strict=False
    )
    assert verdict.status is Verification.unverified
    assert not verdict.verifier_ran
    assert verdict.coverage_units == ()
    assert verdict.claims == ()
    assert not chat.calls


async def test_an_answer_with_no_provenance_short_circuits() -> None:
    item = d0_item("c-durability")
    chat = ScriptedVerifier()
    answer = "Penguins migrate across Antarctica every winter to find colder ice."
    verdict = await verify_answer(chat, d0_notes(item), answer, strict=False)
    assert verdict.status is Verification.unsupported
    assert not verdict.verifier_ran
    assert [(u.kind, u.span.text) for u in verdict.coverage_units] == [(UNCITED, answer)]
    assert not chat.calls


# ----------------------------------------------------------------- spans in the SUPPLIED answer
async def test_spans_map_through_marker_normalization() -> None:
    """`[1]` is normalized to `[S1]` before checking; spans still index the supplied text."""
    item = d0_item("p-fabricated-1")
    assert item.span is not None
    lenient = item.answer.replace("[S1]", "[1]")
    verdict = await verify_answer(ScriptedVerifier(), d0_notes(item), lenient, strict=False)
    fabricated = mutated_text(item)
    start = lenient.index(fabricated)
    _spans_consistent(lenient, verdict)
    assert any(
        u.span == Span(fabricated, start, start + len(fabricated)) for u in verdict.coverage_units
    )


async def test_code_unbacked_units_carry_their_kind() -> None:
    item = d0_item("p-typo-1")  # a typo inside inline code: not a whole code element of the page
    verdict = await _verify(item)
    assert item.span is not None
    kinds = {
        u.kind for u in verdict.coverage_units if _overlaps(u.span, item.span.start, item.span.end)
    }
    assert kinds == {CODE_UNBACKED}


_MARKERS = st.sampled_from(["[S1]", "[1]", "[s1]", "(S1)", "[ S1 ]"])


@settings(max_examples=60, deadline=None)
@given(item=st.sampled_from(d0_verifier_set().items), marker=_MARKERS, quote=st.booleans())
async def test_every_span_indexes_the_supplied_answer(
    item: VerifierItem, marker: str, quote: bool
) -> None:
    """Property: whatever marker spelling the answer uses and whatever the verifier quotes,
    each reported span is a real slice of the supplied answer holding exactly its text."""
    answer = item.answer.replace("[S1]", marker)
    words = answer.split()
    claim = " ".join(words[2:6]) if quote and len(words) > 6 else "not in the answer at all"
    verdict = await verify_answer(
        ScriptedVerifier([objects(claim)]), d0_notes(item), answer, strict=False
    )
    _spans_consistent(answer, verdict)


# ----------------------------------------------------------------- spans land where they belong
_CANONICAL = re.compile(r"\[S(\d+)\]")
_SPELLINGS: list[Callable[[str], str]] = [
    lambda n: f"[S{n}]",
    lambda n: f"[{n}]",
    lambda n: f"[s{n}]",
    lambda n: f"(S{n})",
    lambda n: f"[ S{n} ]",
    lambda n: f"[source {n}]",
]


@dataclass(frozen=True)
class _Respelled:
    """`canonical` with each `[S<n>]` respelled, and where each marker went: the test's OWN
    record of the positions, independent of the server's rewrite log."""

    text: str
    edits: tuple[tuple[int, int, int], ...]  # (canonical start, canonical end, length change)

    def moved(self, pos: int) -> int:
        """A canonical span edge, moved by every marker that starts before it (spans never cut
        a marker, so an edge is never inside one)."""
        return pos + sum(d for s, _, d in self.edits if s < pos)


def _respell(canonical: str, spell: Callable[[str], str]) -> _Respelled:
    edits: list[tuple[int, int, int]] = []

    def sub(m: re.Match[str]) -> str:
        new = spell(m[1])
        edits.append((m.start(), m.end(), len(new) - len(m[0])))
        return new

    return _Respelled(_CANONICAL.sub(sub, canonical), tuple(edits))


def _all_spans(verdict: EvalVerdict) -> list[tuple[str, Span | None]]:
    return [
        *((u.kind, u.span) for u in verdict.coverage_units),
        *((c.reason, c.span) for c in verdict.claims),
        *((c.text, c.span) for c in verdict.verifier_claims),
    ]


async def _assert_spans_follow_the_markers(
    canonical: str, respelled: _Respelled, notes: Sequence[RankedChunk], *claims: str
) -> None:
    """The oracle: the respelled answer normalizes to the SAME draft, so the pipeline's verdict
    is the same and every span must be the canonical span moved by the markers before it, and
    hold the canonical text respelled. A map that shifts a span by even one character fails."""
    assert normalize_citations(respelled.text) == normalize_citations(canonical)
    want = await verify_answer(ScriptedVerifier([objects(*claims)]), notes, canonical, strict=False)
    got = await verify_answer(
        ScriptedVerifier([objects(*claims)]), notes, respelled.text, strict=False
    )
    assert want.status is got.status
    pairs = list(zip(_all_spans(want), _all_spans(got), strict=True))
    for (label, w), (same, g) in pairs:
        assert label == same
        if w is None or g is None:
            assert w is g is None
            continue
        assert (g.start, g.end) == (respelled.moved(w.start), respelled.moved(w.end))
        assert respelled.text[g.start : g.end] == g.text


@settings(max_examples=80, deadline=None)
@given(
    item=st.sampled_from(d0_verifier_set().items),
    spell=st.sampled_from(_SPELLINGS),
    quote=st.booleans(),
)
async def test_every_span_lands_on_the_text_it_names(
    item: VerifierItem, spell: Callable[[str], str], quote: bool
) -> None:
    """Property (review round 1, P2): not just a valid slice, the RIGHT slice, for every marker
    spelling normalization accepts (a spelling it does not rewrite in some context is skipped:
    that answer is a different draft, not a mapping case)."""
    respelled = _respell(item.answer, spell)
    assume(normalize_citations(respelled.text) == normalize_citations(item.answer))
    words = item.answer.split()
    claim = " ".join(words[2:6]) if quote and len(words) > 6 else "not in the answer at all"
    await _assert_spans_follow_the_markers(item.answer, respelled, d0_notes(item), claim)


@pytest.mark.parametrize(
    ("marker", "stacked"),
    [
        ("[S1, S2]", "[S1-S2]"),
        ("[S1, S2]", "[S1-2]"),
        ("[S1, S2]", "[S1 to S2]"),
        ("[S1, S2]", "[Sources 1-2]"),
        ("[S1, S2]", "[s1, s2]"),
        ("[S1][S2]", "[s1][S2]"),
        ("[S1][S2]", "[1][2]"),
    ],
)
async def test_stacked_and_range_labels_map_back_whole(marker: str, stacked: str) -> None:
    """Ranges and stacked labels rewrite one source span into a longer or shorter marker run
    (a range in two substitutions: expanded, then canonicalised); a span ending on it must end
    where the supplied marker run ends."""
    item = d0_item("c-idempotency")  # two shown notes: S1 and S2 are both valid
    canonical = item.answer.replace("[S1]", marker, 1)
    first = canonical.index(marker)
    end = first + len(marker)
    respelled = _Respelled(
        canonical[:first] + stacked + canonical[end:],
        ((first, end, len(stacked) - (end - first)),),
    )
    sentence = canonical[:first].rstrip()  # the first sentence, quoted without its marker
    with_marker = canonical[:end]  # and quoted up to the END of the marker run
    got = await verify_answer(
        ScriptedVerifier([objects(with_marker)]), d0_notes(item), respelled.text, strict=False
    )
    assert [(c.span.start, c.span.end) for c in got.verifier_claims if c.span] == [
        (0, first + len(stacked))
    ]
    await _assert_spans_follow_the_markers(
        canonical, respelled, d0_notes(item), sentence, with_marker
    )


async def test_a_long_lenient_answer_maps_in_linear_time() -> None:
    """Review round 1, P1: the old difflib map was roughly cubic (4 s at 3,000 characters) and
    blocked the event loop. 20,000 characters, the contract's maximum, every marker rewritten,
    must stay well inside a request's budget."""
    item = d0_item("c-durability")
    sentence = "Kafka acks all replicas [1]. "
    answer = (sentence * (20_000 // len(sentence)))[:20_000]
    started = time.perf_counter()
    verdict = await verify_answer(ScriptedVerifier(), d0_notes(item), answer, strict=False)
    elapsed = time.perf_counter() - started
    _spans_consistent(answer, verdict)
    draft = normalize_citations(answer)
    positions = _PositionMap(answer, draft)  # and every one of the ~690 markers maps exactly
    at = [m.start() for m in re.finditer(r"\[S1\]", draft)]
    assert [positions.span(a, a + 4) for a in at] == [
        (m.start(), m.end()) for m in re.finditer(r"\[1\]", answer)
    ]
    assert elapsed < 5.0, f"{elapsed:.2f}s for 20,000 characters"


def test_the_rewrite_log_matches_normalization_or_refuses() -> None:
    """The log is checked against `normalize_citations` itself: a drift is a loud server error,
    never a silently mis-mapped span."""
    answer = "Acks wait for replicas [1]. Brokers keep copies [s1]."
    assert _PositionMap(answer, normalize_citations(answer)).span(0, 4) == (0, 4)
    with pytest.raises(NormalizationDrift):
        _PositionMap(answer, normalize_citations(answer) + " ")
