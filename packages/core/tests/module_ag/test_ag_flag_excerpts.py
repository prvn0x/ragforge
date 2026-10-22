"""FLAG (b), the muted flag: "inexact_code" is only for code that is an UNALTERED excerpt of one
code element of the cited page (blind review round 1, P1). A changed value, a reordered or
truncated command, or a name found only somewhere else on the page is "not_found" (red).

Also: an objection the verifier could not place never lets a sentence be muted (round 1, P2),
and a name that appears only in a comment of the page's code is the verifier's to judge."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_d1.d1_fakes import make_hit
from ragforge_core.agent.attribution import (
    _RUN_TOKEN,
    INEXACT_CODE,
    NOT_FOUND,
    Settled,
    attribute,
    settle,
)
from ragforge_core.agent.coverage import valid_labels
from ragforge_core.agent.state import HitDict, hit_to_state
from ragforge_core.domain.models import ClaimReasonCode, Verification

PAGE = (
    "Kafka stores records in partitions. The producer guarantees ordering per partition.\n\n"
    "```bash\nrm -rf /tmp/cache\nkafka-topics --create --topic logs --partitions 3\n```\n\n"
    "```properties\nenable.idempotence=true\nenable.auto.commit=false\n```\n\n"
    "```properties\nacks=1\nretries=all\nmax.poll.interval.ms=300000\n```\n\n"
    "```java\n// never call dropAllTables() in prod\nschema.migrate(version);\n```"
)
HITS: list[HitDict] = [hit_to_state(make_hit(1, title="Kafka ops", text=PAGE))]
VALID = valid_labels([1], 0)
COVERED = "Kafka stores records in partitions [S1]."


def _settle(
    answer: str,
    *claims: str,
    verdict: Verification | None = None,
    conclusive: bool = True,
    located: bool = True,
) -> Settled:
    return settle(
        answer,
        verdict=verdict or (Verification.partial if claims else Verification.verified),
        conclusive=conclusive,
        located=located,
        verifier_claims=claims,
        valid=VALID,
        attribution=attribute(answer, HITS, [1], VALID),
        tools_ran=False,
    )


def _reason(sentence: str) -> ClaimReasonCode:
    """The reason `sentence` (a labelled sentence with an inline code span) is flagged with after
    a lenient verdict. Segmentation may cut it into several units (a "." inside the span ends a
    sentence: coverage `_ends_inside`), so: "not_found" if any of its units is."""
    settled = _settle(f"{COVERED} {sentence}")
    assert settled.status is Verification.partial
    assert settled.reasons
    assert all(claim in sentence for claim in settled.unsupported_claims)
    return NOT_FOUND if NOT_FOUND in settled.reasons else INEXACT_CODE


# ---------------------------------------------------------------- the reviewer's counterexamples
@pytest.mark.parametrize(
    ("code", "why"),
    [
        ("rm -rf /", "a destructive command cut where its path argument starts"),
        ("enable.idempotence=false", "the value of another key on the page: inverted meaning"),
        ("acks=all", "`acks` and `all` are both on the page, never together"),
        ("rm -rf ./cache", "different spacing and path from `/tmp/cache`"),
        ("--partitions 3 --topic logs", "options reordered"),
        ("kafka-topics --create --topic", "a flag cut from its value"),
        ("max.poll", "a cut inside a dotted key"),
        ("retries=", "an assignment cut from its value"),
        ("rf /tmp/cache", "starts inside a word of the source"),
    ],
)
def test_altered_or_truncated_code_is_red(code: str, why: str) -> None:
    assert _reason(f"Run `{code}` on the broker [S1].") == NOT_FOUND, why


@pytest.mark.parametrize(
    "code",
    [
        "kafka-topics --create --topic logs",  # stops at a whole value
        "--topic logs --partitions 3",  # an inner run, ends at the element's end
        "enable.idempotence=true",  # one line of a block, exactly
        "rm  -rf\t/tmp/cache",  # whitespace differs only in amount and kind
    ],
)
def test_an_unaltered_excerpt_is_muted(code: str) -> None:
    assert _reason(f"Run `{code}` on the broker [S1].") == INEXACT_CODE


def test_an_excerpt_block_of_whole_lines_is_muted() -> None:
    """Two lines of a three-line block, cited by a label line: a line break is a gap."""
    settled = _settle(f"{COVERED}\n\n```properties\nacks=1\nretries=all\n```\n[S1]")
    assert settled.status is Verification.partial
    assert settled.reasons == (INEXACT_CODE,)


def test_a_block_that_drops_a_middle_line_is_red() -> None:
    settled = _settle(f"{COVERED}\n\n```properties\nacks=1\nmax.poll.interval.ms=300000\n```\n[S1]")
    assert settled.reasons == (NOT_FOUND,)


# ---------------------------------------------------------------- property: one token mutated
_LINES = [
    "kafka-topics --create --topic logs --partitions 3",
    "enable.idempotence=true",
    "acks=1",
    "max.poll.interval.ms=300000",
]
_PAGE_TOKENS = sorted({t for line in _LINES for t in _RUN_TOKEN.findall(line) if not t.isspace()})


@settings(max_examples=150, deadline=None)
@given(
    line=st.sampled_from(_LINES),
    index=st.integers(min_value=0, max_value=40),
    replacement=st.sampled_from(_PAGE_TOKENS),
)
def test_swapping_one_token_for_another_from_the_page_is_red(
    line: str, index: int, replacement: str
) -> None:
    """Every word of the mutant is still on the page, so a bag of words would mute it; the
    mutant is not a run of any code element, so it must be red."""
    tokens = _RUN_TOKEN.findall(line)
    positions = [i for i, t in enumerate(tokens) if not t.isspace()]
    at = positions[index % len(positions)]
    if tokens[at] == replacement:
        return
    mutant = "".join([*tokens[:at], replacement, *tokens[at + 1 :]])
    if any(mutant in page_line for page_line in PAGE.splitlines()):
        return  # the swap happened to produce another line of the page
    assert _reason(f"Run `{mutant}` on the broker [S1].") == NOT_FOUND


# ---------------------------------------------------------------- unplaceable objections
EXCERPT = "Delete it with `kafka-topics --create --topic logs` [S1]."


def test_an_objection_placed_elsewhere_leaves_the_excerpt_muted() -> None:
    settled = _settle(f"{COVERED} {EXCERPT}", "Kafka stores records in partitions")
    assert settled.reasons == (NOT_FOUND, INEXACT_CODE)


@pytest.mark.parametrize(
    ("kwargs", "claims"),
    [
        ({"located": False}, ("Kafka stores records in partitions",)),
        ({"conclusive": False}, ()),
        ({"verdict": Verification.partial}, ()),  # partial, but names no claim
    ],
    ids=["not-located", "inconclusive", "partial-without-claims"],
)
def test_an_objection_that_cannot_be_placed_never_mutes(
    kwargs: dict[str, object], claims: tuple[str, ...]
) -> None:
    """The muted flag covers the whole sentence, prose included ("Delete" is the error here), so
    an objection nobody can place must leave every flag red."""
    settled = _settle(f"{COVERED} {EXCERPT}", *claims, **kwargs)  # type: ignore[arg-type]
    assert EXCERPT in settled.unsupported_claims
    assert set(settled.reasons) == {NOT_FOUND}


# ---------------------------------------------------------------- a name only in a comment
def test_a_name_only_in_a_source_comment_backs_the_name_not_the_claim() -> None:
    """Documented limit (ADR-0015): the token rule backs that the NAME is on the page; the page
    here only warns against it in a comment. What the sentence says about it is the verifier's:
    with a verdict that catches the planted error the answer is red; a lenient one verifies it."""
    planted = "Call `dropAllTables()` to migrate the schema [S1]."
    answer = f"{COVERED} {planted}"
    caught = _settle(answer, planted)
    assert caught.status is Verification.partial
    assert (caught.unsupported_claims, caught.reasons) == ((planted,), (NOT_FOUND,))
    assert _settle(answer).status is Verification.verified
