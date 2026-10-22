"""FLAG (b): every flagged claim carries WHY it is flagged (`TurnVerification.claim_reasons`,
index-aligned with `unsupported_claims`): "not_found" (the verifier, attribution's uncited units,
the residue) or "inexact_code" (only the code of a labelled sentence could not be matched).

The reason only changes what the client SHOWS; the status semantics are unchanged (an
"inexact_code" claim still makes the answer `partial`)."""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from module_d1.d1_fakes import (
    COLLECTION,
    Harness,
    collect,
    make_harness,
    make_hit,
    of_type,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import AgentSettings
from ragforge_core.agent.attribution import (
    INEXACT_CODE,
    NOT_FOUND,
    Attribution,
    Settled,
    attribute,
    flag_reasons,
    merge_reasons,
    settle,
    uncited_reasons,
)
from ragforge_core.agent.coverage import valid_labels
from ragforge_core.agent.state import AgentState, HitDict, claim_reasons_from_state, hit_to_state
from ragforge_core.domain.models import RankedChunk, TurnEvent, TurnVerification, Verification

CODE_PAGE = make_hit(
    1,
    title="Kafka producer",
    text="Kafka stores records in partitions. The producer guarantees ordering per partition.\n\n"
    '```java\nprops.put("processing.guarantee", "exactly_once_v2");\n'
    "kafka-topics --create --topic logs --partitions 3\nproducer.flush();\n```",
)
OTHER = make_hit(2, title="Redis", text="Redis Cluster splits the keyspace into 16384 hash slots.")
HITS: list[HitDict] = [hit_to_state(CODE_PAGE), hit_to_state(OTHER)]
VALID = valid_labels([1, 2], 0)

COVERED = "Kafka stores records in partitions [S1]."
EXCERPT = "Create it with `kafka-topics --create --topic logs` from a shell [S1]."  # muted
TYPO = "Set `processing.guarantee=exactly_one_v2` on the client [S1]."  # red: not on the page
VALUE = "The producer calls `producer.flush(true)` before closing [S1]."  # red: "true" not there
FABRICATED = "Kafka was invented by NASA in 1850."  # uncited: red


def _attribute(answer: str) -> Attribution:
    return attribute(answer, HITS, [1, 2], VALID)


def _settle(answer: str, *claims: str, verdict_: Verification | None = None) -> Settled:
    return settle(
        answer,
        verdict=verdict_ or (Verification.partial if claims else Verification.verified),
        conclusive=True,
        located=True,
        verifier_claims=claims,
        valid=VALID,
        attribution=_attribute(answer),
        tools_ran=False,
    )


def _pairs(settled: Settled) -> list[tuple[str, str]]:
    return list(zip(settled.unsupported_claims, settled.reasons, strict=True))


# ---------------------------------------------------------------- reasons by source of the flag
def test_code_that_only_differs_in_form_is_inexact_code_and_still_partial() -> None:
    settled = _settle(f"{COVERED} {EXCERPT}")
    assert settled.status is Verification.partial  # the status semantics do not change
    assert _pairs(settled) == [(EXCERPT, INEXACT_CODE)]


@pytest.mark.parametrize("wrong", [TYPO, VALUE], ids=["identifier-typo", "changed-value"])
def test_code_with_a_name_or_value_the_page_lacks_is_not_found(wrong: str) -> None:
    """The measured false-verified class (`exactly_one_v2` for `exactly_once_v2`) must be RED,
    never the muted flag: a muted typo is a silent wrong answer."""
    settled = _settle(f"{COVERED} {wrong}")
    assert settled.status is Verification.partial
    assert _pairs(settled) == [(wrong, NOT_FOUND)]


def test_uncited_text_and_verifier_claims_are_not_found() -> None:
    answer = f"{COVERED} Each partition is replicated to followers [S1]. {FABRICATED}"
    settled = _settle(answer, "Each partition is replicated to followers")
    assert _pairs(settled) == [
        ("Each partition is replicated to followers", NOT_FOUND),
        (FABRICATED, NOT_FOUND),
    ]


def test_a_verifier_claim_overlapping_an_inexact_unit_turns_it_red() -> None:
    """The verifier quoting PART of the sentence objects to the whole unit: positional overlap,
    not text equality, decides that not_found wins."""
    settled = _settle(f"{COVERED} {EXCERPT}", "from a shell")
    assert _pairs(settled) == [("from a shell", NOT_FOUND), (EXCERPT, NOT_FOUND)]


def test_a_verifier_claim_elsewhere_leaves_the_inexact_unit_muted() -> None:
    settled = _settle(f"{COVERED} {EXCERPT}", "Kafka stores records in partitions")
    assert _pairs(settled) == [
        ("Kafka stores records in partitions", NOT_FOUND),
        (EXCERPT, INEXACT_CODE),
    ]


def test_an_unsupported_verdict_flags_every_claim_not_found() -> None:
    settled = _settle(f"{COVERED} {EXCERPT}", EXCERPT, verdict_=Verification.unsupported)
    assert settled.status is Verification.unsupported
    assert _pairs(settled) == [(EXCERPT, NOT_FOUND)]


def test_a_verified_answer_has_no_claims_and_no_reasons() -> None:
    settled = _settle(COVERED)
    assert (settled.status, settled.unsupported_claims, settled.reasons) == (
        Verification.verified,
        (),
        (),
    )


def test_a_code_unbacked_unit_is_inexact_despite_its_uncited_membership() -> None:
    """attribute() lists every code-unbacked unit in `uncited` too (it is residue); that
    membership must not turn the muted flag red."""
    found = _attribute(f"{COVERED} {EXCERPT}")
    assert EXCERPT in found.uncited
    assert found.code_inexact == (EXCERPT,)
    assert flag_reasons(f"{COVERED} {EXCERPT}", found) == ((EXCERPT,), (INEXACT_CODE,))


def test_a_residue_unit_is_not_found_even_when_its_code_alone_is_inexact() -> None:
    answer = f"{COVERED} {EXCERPT}"
    found = _attribute(answer)
    assert flag_reasons(answer, found, left=(EXCERPT,)) == ((EXCERPT,), (NOT_FOUND,))


def test_uncited_reasons_keep_answer_order() -> None:
    """The no-provenance path (no verifier ran) keeps `uncited` order, with reasons."""
    answer = f"{FABRICATED} {EXCERPT} {TYPO}"
    found = _attribute(answer)
    assert uncited_reasons(found) == (
        (FABRICATED, EXCERPT, TYPO),
        (NOT_FOUND, INEXACT_CODE, NOT_FOUND),
    )


# ---------------------------------------------------------------- dedupe and alignment
def test_merge_keeps_first_position_and_not_found_wins_in_either_order() -> None:
    claims, reasons = merge_reasons(
        [
            ("a", INEXACT_CODE),
            ("b", NOT_FOUND),
            ("a", NOT_FOUND),
            ("c", INEXACT_CODE),
            ("b", INEXACT_CODE),
        ]
    )
    assert (claims, reasons) == (("a", "b", "c"), (NOT_FOUND, NOT_FOUND, INEXACT_CODE))


def test_alignment_survives_duplicate_claims() -> None:
    answer = f"{COVERED} {EXCERPT}"
    settled = _settle(answer, "from a shell", "from a shell", EXCERPT)
    assert len(settled.unsupported_claims) == len(settled.reasons)
    assert len(set(settled.unsupported_claims)) == len(settled.unsupported_claims)
    assert _pairs(settled) == [("from a shell", NOT_FOUND), (EXCERPT, NOT_FOUND)]


# ---------------------------------------------------------------- checkpoints
def test_a_checkpoint_from_before_reasons_reads_every_claim_as_not_found() -> None:
    old: AgentState = {"unsupported_claims": ["a", "b"]}  # no claim_reasons key at all
    assert claim_reasons_from_state(old) == [NOT_FOUND, NOT_FOUND]


@pytest.mark.parametrize(
    ("reasons", "expected"),
    [
        (["inexact_code", "not_found"], ["inexact_code", "not_found"]),
        (["inexact_code"], ["not_found", "not_found"]),  # misaligned: never mute
        (["inexact_code", "surprise"], ["inexact_code", "not_found"]),  # unknown: red
    ],
)
def test_stored_reasons_are_read_fail_toward_the_prominent_flag(
    reasons: list[str], expected: list[str]
) -> None:
    state: AgentState = {"unsupported_claims": ["a", "b"], "claim_reasons": reasons}
    assert claim_reasons_from_state(state) == expected


# ---------------------------------------------------------------- end to end through the runner
def _harness(*hits: RankedChunk, settings: AgentSettings | None = None) -> Harness:
    return make_harness(
        results=[list(hits or (CODE_PAGE, OTHER))],
        settings=settings or AgentSettings(heartbeat_interval_s=3600),
    )


async def _verification(h: Harness) -> TurnVerification:
    events: list[TurnEvent] = await collect(
        h.runner.run_turn(principal(), None, [COLLECTION], "How do I create a topic?")
    )
    (ver,) = of_type(events, TurnVerification)
    return ver


async def test_turn_carries_index_aligned_reasons() -> None:
    h = _harness()
    h.script(say(f"{COVERED} {EXCERPT} {TYPO} {FABRICATED}"), verdict())
    ver = await _verification(h)
    assert ver.status is Verification.partial
    # The claim order is unchanged (verifier claims, residue, then code-unbacked units).
    assert list(zip(ver.unsupported_claims, ver.claim_reasons, strict=True)) == [
        (FABRICATED, NOT_FOUND),
        (EXCERPT, INEXACT_CODE),
        (TYPO, NOT_FOUND),
    ]


async def test_a_verified_turn_emits_no_claims() -> None:
    h = _harness()
    h.script(say(COVERED), verdict())
    ver = await _verification(h)
    assert (ver.status, ver.unsupported_claims, ver.claim_reasons) == (
        Verification.verified,
        (),
        (),
    )


async def test_an_untrusted_verifier_flags_inferred_sentences_not_found() -> None:
    """Rule 7: nothing certified the inferred citation, so the sentence is red."""
    inferred = "Kafka stores records in partitions."
    h = _harness(settings=AgentSettings(heartbeat_interval_s=3600, verifier_trusted=False))
    h.script(say(f"{inferred} {EXCERPT}"), verdict())
    ver = await _verification(h)
    assert ver.status is Verification.unverified
    assert dict(zip(ver.unsupported_claims, ver.claim_reasons, strict=True)) == {
        EXCERPT: INEXACT_CODE,
        inferred: NOT_FOUND,
    }


async def test_an_answer_rejected_before_the_ground_check_regenerates_and_ends_clean() -> None:
    """No provenance at all (nothing covered, nothing inferable): the uncited units (with their
    reasons, `uncited_reasons`) feed the strict regeneration, and the clean regenerated answer
    ends with no claims: no stale reason survives into the emitted verification."""
    h = _harness()
    h.script(say(f"{EXCERPT} {FABRICATED}"), say(COVERED), verdict())
    ver = await _verification(h)
    strict = [c for c in h.chat.calls if c.kind == "stream"][-1]
    assert FABRICATED in str(strict.messages[-1].content)
    assert ver.status is Verification.verified
    assert (ver.unsupported_claims, ver.claim_reasons) == ((), ())


async def test_the_verifier_call_logs_one_eval_line_with_ids_and_no_text() -> None:
    h = _harness()
    h.script(say(f"{COVERED} {EXCERPT}"), verdict("partially_supported", "from a shell"))
    with capture_logs() as logs:
        await _verification(h)
    (line,) = [e for e in logs if e["event"] == "answer_settled"]
    assert line["prompt"].startswith("ground_check@")
    assert line["model"] == "fake-chat"
    assert line["outcome"] == "parsed"
    assert line["chunk_ids"] == [str(CODE_PAGE.chunk_id), str(OTHER.chunk_id)]
    assert line["inexact_code"] == 0  # the verifier's overlapping quote turned it red
    assert isinstance(line["latency_ms"], int)
    assert {"prompt_tokens", "completion_tokens", "status", "verdict"} <= line.keys()
    assert not any("shell" in str(v) or "Kafka" in str(v) for v in line.values())


def test_an_answer_rejected_before_any_verifier_is_never_muted() -> None:
    """Only an excerpt, so nothing is covered (no provenance): attribution alone would call it
    inexact_code, but no verifier passed anything here, so the flag is red (final review P2)."""
    from ragforge_core.agent.attribution import uncited_reasons
    from ragforge_core.agent.nodes import unverified_flags

    found = _attribute(EXCERPT)
    assert "inexact_code" in uncited_reasons(found)[1]  # the muted reading this rule overrides
    claims, reasons = unverified_flags(found)
    assert claims
    assert set(reasons) == {"not_found"}
    assert len(reasons) == len(claims)
