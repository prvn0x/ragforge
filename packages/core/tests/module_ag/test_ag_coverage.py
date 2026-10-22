"""ADR-0015's guarantee, stated as coverage: `verified` => the residue is empty.

Residue = the whole answer minus (a) text a valid model label covers (only up to the previous
boundary), (b) units whose inferred citation passed the ground check, (c) a closed allowlist.
The property at the bottom recomputes the residue with the TEST's own oracle, over generated
answers, with a verifier stub that always says "supported" (the worst case for the guarantee).
Since review round 4 the oracle reads the answer with its own markdown-it-py parse (code and
visible labels come from the parser, as the reader's renderer sees them), not with regexes.
"""

from __future__ import annotations

import bisect
import itertools
import re
import textwrap
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt
from markdown_it.token import Token

from module_ag.ag_hits import HITS, SHOWN, VALID
from ragforge_core.agent import coverage as coverage_module
from ragforge_core.agent import structure as structure_module
from ragforge_core.agent.attribution import attribute, no_provenance, settle
from ragforge_core.agent.citations import check_citations, normalize_citations
from ragforge_core.agent.coverage import (
    TERMINATORS,
    Allowed,
    allowed,
    has_rtl,
    refusal_only,
    residue,
    segment,
    source_sentences,
    states_nothing,
    valid_labels,
)
from ragforge_core.agent.structure import LINKIFY, structure
from ragforge_core.domain.models import Verification


@contextmanager
def parser_config(*, linkify: bool) -> Iterator[None]:
    """Run the pipeline with linkify on or off (review of the parser rewrite, round 1: the
    prod and CI both install linkify-it-py now; tests still run with and without it). The
    parse caches are cleared on the way in and out."""

    def clear() -> None:
        structure_module.structure.cache_clear()
        structure_module.bare_html.cache_clear()
        coverage_module.segment.cache_clear()
        coverage_module.visible_runs.cache_clear()

    saved = structure_module._MD
    structure_module._MD = structure_module._parser(linkify=linkify)  # type: ignore[misc]
    clear()
    try:
        yield
    finally:
        structure_module._MD = saved  # type: ignore[misc]
        clear()


# Both parser configurations, for the attack lists and the property (linkify needs the package).
LINKIFY_CONFIGS = [
    pytest.param(False, id="no-linkify"),
    pytest.param(
        True, id="linkify", marks=pytest.mark.skipif(not LINKIFY, reason="no linkify-it-py")
    ),
]


def status_of(
    answer: str, *, tools_ran: bool = False, shown: tuple[int, ...] = (1, 2)
) -> tuple[Verification, tuple[str, ...]]:
    """The production pipeline (nodes.verify_citations + ground_check + settle) with a verifier
    that ALWAYS says supported: the worst case for the guarantee. By default only the two
    prose pages of the previous review's attack script are shown."""
    draft = normalize_citations(answer)
    valid = valid_labels(shown, 1 if tools_ran else 0)
    report = check_citations(draft, shown, n_tool_results=1 if tools_ran else 0, strict=False)
    if not report.ok:
        return Verification.unsupported, report.offending
    found = attribute(draft, HITS, shown, valid)
    if found.refusal_only:
        return Verification.unverified, ()
    if no_provenance(found, tools_ran):
        return Verification.unsupported, found.uncited
    out = settle(
        draft,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=valid,
        attribution=found,
        tools_ran=tools_ran,
    )
    return out.status, out.unsupported_claims


# ---------------------------------------------------------------- segmentation
def _units(answer: str) -> list[tuple[str, tuple[str, ...]]]:
    return [(u.text, u.labels) for u in segment(answer)]


def test_a_label_covers_only_the_clause_it_terminates() -> None:
    assert _units("Kafka has partitions [S1]; it runs on port 9999.") == [
        ("Kafka has partitions [S1]", ("S1",)),
        ("it runs on port 9999.", ()),
    ]


def test_an_em_dash_clause_after_a_label_is_its_own_unit() -> None:
    assert _units("Kafka has partitions [S1] \u2014 and NASA built it.")[1] == (
        "and NASA built it.",
        (),
    )


def test_a_lower_case_sentence_is_a_sentence() -> None:
    assert _units("Kafka has partitions [S1]. it was built by NASA.")[1] == (
        "it was built by NASA.",
        (),
    )


def test_cjk_sentence_ends_split_without_whitespace() -> None:
    assert _units("Kafka \u5206\u533a [S1]\u3002\u5b83\u4f7f\u7528\u7aef\u53e39999\u3002") == [
        ("Kafka \u5206\u533a [S1]\u3002", ("S1",)),
        ("\u5b83\u4f7f\u7528\u7aef\u53e39999\u3002", ()),
    ]


def test_a_bare_label_after_a_sentence_cites_that_sentence() -> None:
    assert _units("Kafka has partitions. [S1] Postgres uses MVCC [S2].") == [
        ("Kafka has partitions. [S1]", ("S1",)),
        ("Postgres uses MVCC [S2].", ("S2",)),
    ]


def test_e_g_does_not_end_a_sentence() -> None:
    assert _units("Use a log, e.g. Kafka, for events [S1].") == [
        ("Use a log, e.g. Kafka, for events [S1].", ("S1",))
    ]


def test_code_blocks_are_single_units_and_a_following_label_cites_them() -> None:
    answer = "Run this:\n```sh\nrm -rf /data\n```\n[S3]\n\n    kafka-topics --list\n\nDone."
    units = segment(answer)
    assert [(u.code, u.labels) for u in units] == [
        (False, ()),
        (True, ("S3",)),
        (True, ()),
        (False, ()),
    ]
    assert units[2].text == "    kafka-topics --list"


def test_markdown_structure_is_not_residue() -> None:
    answer = "## Kafka [S1]\n> - Kafka stores records in partitions [S1].\n1. [x] Done [S1].\n---"
    assert residue(answer, VALID, ()) == []


def test_a_list_numeral_alone_is_an_answer_not_a_marker() -> None:
    assert residue("Kafka stores records in partitions [S1].\n3.", VALID, ()) == ["3."]
    assert residue("- 3", VALID, ()) == ["3"]


def test_emoji_and_punctuation_are_ignored_but_words_next_to_them_are_not() -> None:
    assert residue("\U0001f44d ... !!", VALID, ()) == []
    assert residue("\U0001f389 Great!", VALID, ()) == ["\U0001f389 Great!"]


# Review of the redesign, round 2 (P1): "states nothing" is an allowlist of characters too.
LOOKALIKES = [
    # circled
    "\u24c3\u24d0\u24e2\u24d0 \u24d1\u24e4\u24d8\u24db\u24e3 \u24da\u24d0\u24d5\u24da\u24d0",
    "\U0001f17d\U0001f170\U0001f182\U0001f170 \U0001f171\U0001f184\U0001f178\U0001f17b",  # squared
    "\U0001f1f3\U0001f1e6\U0001f1f8\U0001f1e6",  # regional indicators
    "\u281d\u2801\u280e\u2801 \u2803\u2825\u280a\u2807\u281e",  # braille
    "\U0001f51f\U0001f51f\U0001f51f",  # KEYCAP TEN
    "\u2776\u2777",  # dingbat negative circled digits
    "\u2122",  # TRADE MARK SIGN (NFKC "TM")
    "\u20ac",  # a currency sign states something
    "\u0363",  # COMBINING LATIN SMALL LETTER A
    "\U000e0041",  # TAG LATIN CAPITAL LETTER A
    "\ue000",  # private use: the reader may see anything
]


@pytest.mark.parametrize("text", LOOKALIKES)
def test_letter_lookalikes_state_something(text: str) -> None:
    assert not all(states_nothing(c) for c in text)
    assert residue(f"Kafka stores records in partitions [S1]. {text}.", VALID, ()) != []


@pytest.mark.parametrize(
    "char",
    [
        *(" ", "\u00a0", ".", "\u2014", "|", ">", "`", "~", "\u200d", "\ufe0f", "\u0301"),
        *("\U0001f44d", "\U0001f680", "\U0001f389", "\u2705", "\u274c", "\u2192", "\U0001f3fb"),
    ],
)
def test_punctuation_marks_and_pictographs_state_nothing(char: str) -> None:
    assert states_nothing(char)


def test_compatibility_letters_are_read_as_the_letters_they_are() -> None:
    """Circled letters are NFKC letters: attribution reads "Ⓝⓐⓢⓐ" as the word "nasa", exactly
    like the plain spelling (lexical support, then the verifier). Braille has no NFKC letters,
    so the tokenizer cannot see it and the unit gets no inferred citation (`coverage.opaque`)."""
    circled = "Kafka stores records in partitions \u24c3\u24d0\u24e2\u24d0."
    plain = unicodedata.normalize("NFKC", circled)
    assert status_of(circled)[0] is status_of(plain)[0]
    assert attribute(circled, HITS, SHOWN, VALID).inferred != ()
    braille = "Kafka stores records in partitions \u281d\u2801\u280e\u2801."
    assert attribute(braille, HITS, SHOWN, VALID).inferred == ()


def test_residue_is_verbatim() -> None:
    answer = "Kafka [S1]; it runs on 9999.\n```\nrm -rf /\n```\n\u5b83\u3002 ok"
    for text in residue(answer, VALID, ()):
        assert text in answer


# ---------------------------------------------------------------- segmentation fails closed
# Review of the redesign, round 1 (P1): a missed boundary lets a label cover the sentence before
# it. Every terminator cuts, with or without whitespace after it.
_JOINS = [
    ".",
    "!",
    "?",
    "\u2026",  # ellipsis
    "\u0964",  # Devanagari danda
    "\uff0e",  # full-width full stop
    "\uff61",  # half-width ideographic full stop
    "\u061f",  # Arabic question mark
    ".\u200b",  # zero-width space after the period
    '."',
    "\u3002",
    ";",  # a semicolon ends a clause (any script)
    "\uff1b",
]


@pytest.mark.parametrize("join", _JOINS)
def test_a_terminator_cuts_without_whitespace(join: str) -> None:
    answer = f"NASA built Kafka in 1850{join}Kafka stores records in partitions [S1]."
    units = segment(answer)
    assert units[0].text.startswith("NASA built Kafka in 1850")
    assert units[0].labels == ()
    # An Arabic-script mark may reorder the line on screen: the whole answer is residue then
    # (`has_rtl`), which is stricter, never laxer.
    expected = [answer] if has_rtl(answer) else [units[0].text]
    assert residue(answer, VALID, ()) == expected


def test_every_punctuation_character_named_like_a_sentence_end_is_a_terminator() -> None:
    """The table is checked against the test's own scan of unicodedata, not copied from it."""
    named = ("FULL STOP", "QUESTION MARK", "EXCLAMATION MARK", "DANDA", "ELLIPSIS")
    missing = [
        f"U+{c:04X}"
        for c in range(0x110000)
        if unicodedata.category(chr(c)).startswith("P")
        and any(w in unicodedata.name(chr(c), "") for w in named)
        and chr(c) not in TERMINATORS
    ]
    assert missing == []


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("Kafka 3.5 runs on node.js [S1].", 1),  # digit.digit and lower.lower are tokens
        ("Use a log, e.g. Kafka, i.e. a broker [S1].", 1),
        ("NASA built Kafka.Kafka stores records [S1].", 2),  # upper after the dot
        ("Built in 1850.Kafka stores records [S1].", 2),
        ("Built in 1850.kafka stores records [S1].", 2),  # digit.lower
        ("It is v2!great [S1].", 2),  # "!" and "?" always cut
        ("Kafka.\u200bkafka [S1].", 2),
        # review round 2 (P2): lower.lower is a token only when identifier-shaped
        ("nasa built it in the year eighteen fifty.kafka stores records [S1].", 2),
        ("Edit postgresql.conf and use `os.path.join`, e.g. in main.py [S1].", 1),
        ("Use os.path.join, then [S1].", 2),  # a bare "join" after the dot: cut, fail closed
        ("Kafka runs on node.js.it runs on port 9999 [S1].", 2),  # found by the stress run
        ("Call foo.bar(x) then baz.qux [S1].", 2),  # "qux" is no extension: cut, fail closed
    ],
)
def test_only_a_dot_inside_a_token_is_exempt(text: str, parts: int) -> None:
    assert len(segment(text)) == parts


def test_source_sentences_use_the_same_boundaries() -> None:
    """A source page is segmented exactly like an answer, from its own parse: its code is not
    a sentence, and a span inside its fenced block is part of that block."""
    assert source_sentences("A.B\u3002C") == ["A.", "B\u3002", "C"]
    page = "Run `ls -l` now.\n```\nrm -rf /data. NASA\n```\n> Quoted **bold** text."
    assert source_sentences(page) == ["Run ls -l now.", "Quoted bold text."]


@pytest.mark.parametrize(
    "answer",
    [
        "3. Kafka stores records in partitions [S1].",
        "(7) Kafka stores records in partitions [S1].",
        "100. Kafka stores records in partitions [S1].",
        "> 42) Kafka stores records in partitions [S1].",
        "- (7) Kafka stores records in partitions [S1].",
        "Kafka is a log [S1]. (7) Kafka stores records in partitions [S1].",
        "b) Kafka stores records in partitions [S1].",
    ],
)
def test_a_leading_enumerator_outside_a_numbered_list_is_residue(answer: str) -> None:
    """Review round 1 (P2): "(7)", "3." alone or "100." may BE the answer to "how many"."""
    left = residue(answer, VALID, ())
    assert len(left) == 1
    assert len(left[0]) <= 5  # the enumerator alone; the cited sentence stays covered


@pytest.mark.parametrize(
    "answer",
    [
        "1. Kafka stores records in partitions [S1].",
        "1. Kafka stores records in partitions [S1].\n2. Postgres uses MVCC [S2].",
        "1. Kafka stores records in partitions [S1].\n\n2) Postgres uses MVCC [S2].",
        "> 1. Kafka stores records in partitions [S1].",
    ],
)
def test_the_numerals_of_an_actual_numbered_list_are_structure(answer: str) -> None:
    assert residue(answer, VALID, ()) == []


@pytest.mark.parametrize(
    "answer",
    [
        "NASA built Kafka.\n\n---\n\n[S1]",
        "NASA built Kafka.\n- [S1]",
        "NASA built Kafka.\n\n> [S1]",
        "NASA built Kafka:\n\n[S1]",
        "NASA built Kafka. \U0001f680 [S1]",
        "NASA built Kafka.\n\n[S1]",
        "## NASA built Kafka\n[S1]",
        "```\nrm -rf /data\n```\n\n[S1]",
        "```\nrm -rf /data\n```\n- [S1]",
    ],
)
def test_an_orphan_label_covers_nothing(answer: str) -> None:
    """Review round 1 (P2): a bare label cites the unit before it only on the same line, on the
    next line of the same plain paragraph, or on the line right after a code block."""
    assert residue(answer, VALID, ()) != []


@pytest.mark.parametrize(
    "answer",
    [
        "Kafka stores records in partitions. [S1]",
        "Kafka stores records in partitions [S1]. [S2]",
        "Kafka stores records in partitions.\n[S1]",  # markdown renders one paragraph
        "Kafka stores records in partitions.\nSource: S1",
        "```\nrm -rf /data\n```\n[S1]",
    ],
)
def test_a_bare_label_the_reader_sees_with_its_claim_covers_it(answer: str) -> None:
    assert residue(normalize_citations(answer), VALID, ()) == []


@pytest.mark.parametrize(
    "answer",
    [
        "NASA built Kafka in 1850 `[S1]`.",
        "NASA built Kafka in 1850 <!-- [S1] -->.",
        "NASA built Kafka in 1850 <!-- [S1]",
    ],
)
def test_a_label_in_inline_code_or_an_html_comment_is_not_a_label(answer: str) -> None:
    """Review round 1 (P2): the reader sees code, or nothing, where the label is."""
    assert all(u.labels == () for u in segment(answer))  # "<!" also cuts: more residue, fine
    assert not check_citations(answer, SHOWN, strict=True).ok
    assert check_citations(answer, SHOWN, strict=False).cited == ()


def test_check_citations_and_strict_mode_use_the_coverage_units() -> None:
    """Review round 1 (P2): one definition of a unit. A tail after a label is a unit of its own,
    so strict mode flags it (it used to pass rule 1 and be caught only by the residue)."""
    report = check_citations("X is a log [S1]. it was built by NASA.", SHOWN, strict=True)
    assert report.offending == ("it was built by NASA.",)
    report = check_citations("X is a log [S1] \u2014 and NASA built it.", SHOWN, strict=True)
    assert report.offending == ("and NASA built it.",)
    # allowlisted text and code need no label; a label citing a code block is checked
    ok = "X is a log [S1].\n```\nrm -rf /\n```\nLet me know if you have other questions."
    assert check_citations(ok, SHOWN, strict=True).ok
    assert check_citations("```\nrm -rf /\n```\n[S9]", SHOWN, strict=False).invalid == ("S9",)
    # round 3 (P1): a tag's attribute text is text; a bare tag needs no label
    report = check_citations("X is a log [S1]. <p NASA built Kafka>", SHOWN, strict=True)
    assert report.offending == ("<p NASA built Kafka>",)
    assert check_citations("X is a log [S1].<br/>", SHOWN, strict=True).ok


def test_a_fence_closes_only_with_its_own_character() -> None:
    """CommonMark: a ``` line inside a ~~~ block is code, and the fence definition is the
    parser's, shared by normalisation (a lenient marker inside the block is left as written).
    A fence with an info string that is no language tag is still a fence (the parser says so,
    and so does the reader's renderer): its text is code, never prose, and never covered."""
    answer = "~~~\nrun [1]\n```\nNASA built Kafka [1].\n~~~"
    assert [u.code for u in segment(answer)] == [True]
    assert normalize_citations(answer) == answer
    unknown = "```Kafka\nKafka stores records in partitions [1]."
    assert normalize_citations(unknown) == unknown
    assert [u.code for u in segment(unknown)] == [True]
    assert status_of(unknown)[0] is Verification.unsupported


# ---------------------------------------------------------------- the closed allowlist
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Your notes don't cover this.", Allowed.refusal),
        ("Your notes don\u2019t cover this.", Allowed.refusal),
        ("The notes do not cover that question.", Allowed.refusal),
        ("I couldn't find this in your notes.", Allowed.refusal),
        ("Let me know if you have other questions.", Allowed.pleasantry),
        ("Let me know if you have any follow-up questions!", Allowed.pleasantry),
        ("Feel free to ask if you need anything else.", Allowed.pleasantry),
        ("I hope this helps.", Allowed.pleasantry),
        ("Hope that helps! \U0001f642", Allowed.pleasantry),
        ("Here is what I found:", Allowed.pleasantry),
    ],
)
def test_allowlisted_non_claims(text: str, kind: Allowed) -> None:
    assert allowed(text) is kind


@pytest.mark.parametrize(
    "text",
    [
        "No, it doesn't cover that.",  # a polarity clause answers a yes/no question
        "No, they don't cover that.",
        "Your notes don't cover this; no.",
        "The notes don't cover it; no.",
        "Your notes don't cover this, but yes.",
        "Your notes do not cover partitions, but Kafka has 5 brokers.",
        "Your notes don't cover the retention period; it is 7 days.",
        "Yes, it covers that.",
        "It covers both.",
        "Let me know the port.",
        "Let me know if you have 3 brokers.",
        "Here is the answer:",
        "Sure.",
        "Yes.",
        "Thanks!",
    ],
)
def test_anything_else_is_not_allowlisted(text: str) -> None:
    assert allowed(text) is None


def test_a_labelled_refusal_is_still_a_refusal() -> None:
    """answer_strict up to @4 asked for a refusal labelled with a source: the label does not
    turn "no answer" into a claim that could be badged `verified`."""
    assert refusal_only("Your notes don't cover this [S1].")
    assert status_of("Your notes don't cover this [S1].") == (Verification.unverified, ())


def test_a_pure_refusal_with_pleasantries_is_refusal_only() -> None:
    assert refusal_only("Your notes don't cover this. Let me know if you have other questions.")
    assert not refusal_only("Hope that helps.")  # no refusal: makes no claim AND refuses nothing
    assert not refusal_only("Your notes don't cover this.\n```\nrm -rf /data\n```")
    assert not refusal_only("Your notes don't cover this. Sure.")


# ---------------------------------------------------------------- happy path (no recall loss)
def test_a_fully_cited_answer_with_a_pleasantry_is_verified() -> None:
    answer = (
        "Kafka stores records in partitions [S1]. Postgres uses MVCC [S2].\n\n"
        "Let me know if you have other questions."
    )
    assert status_of(answer) == (Verification.verified, ())


def test_an_a1_style_correct_uncited_answer_gets_inferred_citations_and_is_verified() -> None:
    answer = (
        "Kafka stores records in partitions. Postgres uses MVCC.\n\n- Vacuum reclaims dead tuples."
    )
    draft = normalize_citations(answer)
    found = attribute(draft, HITS, SHOWN, VALID)
    assert [i.label for i in found.inferred] == [1, 2, 2]
    assert status_of(answer) == (Verification.verified, ())


def test_a_verbatim_code_block_is_attributed_and_can_be_verified() -> None:
    answer = (
        "Create a topic with kafka-topics [S3].\n"
        "```\nkafka-topics --create --topic logs --partitions 3\n```"
    )
    assert status_of(answer, shown=SHOWN) == (Verification.verified, ())


# ---------------------------------------------------------------- every attack, as a regression
V, P, U, N = (
    Verification.verified,
    Verification.partial,
    Verification.unsupported,
    Verification.unverified,
)
# (answer, status with an always-"supported" verifier). Ported from the previous review's
# attack script: every case it found reaching `verified` is now `partial`/`unsupported`/
# `unverified`; the correct ones stay `verified`.
ATTACKS: list[tuple[str, Verification]] = [
    ("Kafka stores records in partitions.", V),
    ("Kafka does not store records in partitions.", U),  # was verified (negation)
    ("Kafka never stores records in partitions.", U),  # was verified (negation)
    ("No, it doesn't cover that.", U),  # was verified (refusal + polarity)
    ("No, they don't cover that.", U),  # was verified
    ("Your notes don't cover this.", N),  # was verified; a refusal makes no claim
    ("Your notes don't cover this. Sure.", U),  # was verified
    ("Your notes don't cover this.\n```\nrm -rf /data\n```", U),  # was verified (code)
    ("Kafka stores records in partitions [S1].\n```\nport=9999\n```", P),  # was verified
    ("Kafka stores records in partitions [S1]. it runs on port 9999 and was built by NASA.", P),
    ("Kafka stores records in partitions [S1]; it runs on port 9999 and was built by NASA.", P),
    ("The default port is `9999`. Kafka stores records in partitions [S1].", P),
    ("Kafka stores records in partitions [S1]. The port is `9999`.", P),
    ("Kafka stores records in partitions [S1]\u3002 \u5b83\u4f7f\u7528\u7aef\u53e39999\u3002", P),
    ("Yes.", U),
    ("Sure.", U),
    ("Certainly!", U),
    ("Okay.", U),
    ("Found it.", U),
    ("Overall, sure.", U),
    ("In short: certainly.", U),
    ("Here is the answer:", U),
    ("Yes. Kafka stores records in partitions.", P),  # "Yes." is residue: recall cost, ADR
    ("No. Kafka stores records in partitions.", P),
    ("\u306f\u3044\u3002", U),
    ("\u0414\u0430.", U),
    ("Nein.", U),
    ("Non.", U),
    ("S\u00ed.", U),
    ("\U0001f44d", U),
    ("1. Kafka stores records in partitions.\n2. Postgres uses MVCC.", V),
    ("- Kafka stores records in partitions\n- 3", P),
    ("Kafka has 3 brokers.", V),
    ("Kafka has 3 partitions.", U),
    # number words are numbers (review round 2): the source says "3 brokers", so this is true
    ("The cluster has three brokers.", V),
    ("The cluster has 4 brokers.", U),
    ("The default port is 9092.", V),
    ("The default port is 9093.", U),
    ("Autovacuum runs every 3 seconds by default.", U),
    ("Kafka stores records in partitions [1].", V),
    ("Kafka stores records in partitions [S1]. Postgres uses MVCC (S2).", V),
    ("Kafka stores records in partitions [S9].", U),
    ("Kafka stores records in partitions [S1, S2].", V),
    ("Thanks!", U),
    ("Thank you.", U),
    ("Glad to help.", U),
    ("Hope that helps.", U),
    ("I hope this helps!", U),
    ("Don't.", U),
    ("Doesn't.", U),
    ("Not really.", U),
    ("Not at all.", U),
    ("Neither.", U),
    ("True.", U),
    ("Correct.", U),
    ("Right.", U),
    ("Exactly.", U),
    ("Absolutely.", U),
    ("Of course.", U),
    ("Indeed.", U),
    ("Sure, it does.", U),
    ("Sure, that's right.", U),
    ("Ok, summary: none.", U),
    ("Your notes do not cover partitions, but Kafka has 5 brokers.", U),
    ("Your notes don't cover this, but yes.", U),
    ("Your notes don't cover the retention period; it is 7 days.", U),
    ("The notes don't cover it; no.", U),  # was verified
    (
        "Kafka stores records in partitions [S1]. etcd listens on port 9999 and stores 7 replicas.",
        P,
    ),
    ("Kafka stores records in partitions [S1]. kubectl deletes the cluster every 5 minutes.", P),
    ('Kafka stores records in partitions [S1]. "Postgres" uses 12 brokers.', P),
    ("Kafka stores records in partitions [S1] \u2014 and Postgres drops 12 tables nightly.", P),
    ("Kafka stores records in partitions [S1]. Postgres drops 12 tables nightly.", P),
    ("Kafka has 3 brokers. No.", P),  # was verified (polarity joined to the claim before it)
    ("Kafka stores records in partitions [S1].\n\n    rm -rf /var/lib/kafka  # indented", P),
    ("Kafka stores records in partitions [S1].\n`rm -rf /`", P),
    # found by this redesign's property: text on a fence line rode on a verbatim code block
    ("```\nkafka-topics --list\n``` and NASA built Kafka in 1850.", U),
    ("```Kafka was built by NASA\nkafka-topics --list\n```", U),
    ("Kafka stores records in partitions [S1].\n3. - Postgres has 12 replicas.", P),
    ("It does.", U),
    ("Only once.", U),
    ("Both.", U),
    ("Here is a summary:", U),
]
# Review of the coverage redesign, round 1: every input of its findings. Each reached
# `verified` with the always-supported verifier before the fix.
_TAIL = "Kafka stores records in partitions [S1]."
ATTACKS += [(f"NASA built Kafka in 1850{join}{_TAIL}", P) for join in _JOINS]
ATTACKS += [
    (
        "\u0915\u093e\u092b\u094d\u0915\u093e NASA "
        "\u0928\u0947 \u092c\u0928\u093e\u092f\u093e\u0964" + _TAIL,
        P,
    ),
    # P1: code attribution is token-aligned, with whole numbers and a minimum token count
    ("```\nport is 9\n```", U),
    ("```\nruns every 6\n```", U),
    ("```\nis 3\n```", U),
    ("```\na\nb\nc\n```", U),
    (f"{_TAIL}\n```\nport is 9\n```", P),
    (f"{_TAIL}\n```\nruns every 6\n```", P),
    # round 3 (P2): code quotes only whole source lines; a fragment of a prose line is not one
    ("```\nport is 9092\n```", U),
    # P2: a leading numeral that is not an actual list marker is residue
    (f"3. {_TAIL}", P),
    (f"(7) {_TAIL}", P),
    (f"100. {_TAIL}", P),
    (f"> 42) {_TAIL}", P),
    # a semicolon ends the clause a label covers (found by this round's own probing)
    (f"NASA built Kafka in 1850;{_TAIL}", P),
    (f"NASA built Kafka in 1850\uff1b{_TAIL}", P),
    (f"1. {_TAIL}", V),
    (f"1. {_TAIL}\n2. Postgres uses MVCC [S2].", V),
    # P2: a bare label cites only what the reader sees it next to
    ("NASA built Kafka.\n\n---\n\n[S1]", U),
    ("NASA built Kafka.\n- [S1]", U),
    ("NASA built Kafka.\n\n> [S1]", U),
    ("NASA built Kafka:\n\n[S1]", U),
    ("NASA built Kafka. \U0001f680 [S1]", U),
    ("Kafka stores records in partitions.\n[S1]", V),
    # P2: a label in inline code or an HTML comment is no label
    ("NASA built Kafka in 1850 `[S1]`.", U),
    ("NASA built Kafka in 1850 <!-- [S1] -->.", U),
    # P2: negating verbs and signed numbers withhold an inferred citation
    ("Kafka fails to store records in partitions.", U),
    ("The cluster has -3 brokers.", U),
    ("The cluster has 3.5 brokers.", U),
]
# Review of the coverage redesign, round 2: every input of its findings.
ATTACKS += [(f"{_TAIL} {text}.", P) for text in LOOKALIKES]  # P1: letters outside isalnum
ATTACKS += [
    (f"{_TAIL} \U0001f44d\u2705", V),  # pictographs still state nothing
    ("Kafka stores records in partitions [S1] \U0001f680.", V),
    ("Kafka stores records in partitions \u281d\u2801\u280e\u2801.", U),  # no inference
    ("Your notes don't cover this \u281d\u2801\u280e\u2801.", U),  # not a refusal
    # P2: table cells and HTML breaks are boundaries
    ("| NASA built Kafka in 1850 | Kafka stores records in partitions [S1] |", P),
    ("| Claim | Note |\n|---|---|\n| NASA built Kafka in 1850 | " + _TAIL + " |", P),
    ("Claim | Note\n--- | ---\nNASA built Kafka in 1850 | " + _TAIL, P),
    # a code span is atomic (round 4): the label vouches for code S1 does not hold -> rejected
    ("| `NASA built Kafka | Kafka stores records in partitions` [S1] |", U),
    ("| Kafka stores records in partitions | [S1] |", V),  # a source column cites its row
    ("NASA built Kafka in 1850 | " + _TAIL, P),  # a pipe outside a table still separates
    # a pipe in inline code is code; S1 has no such span, so the label cannot vouch for it
    ("Run `cat x | grep y` to filter [S1].", U),
    ("NASA built Kafka in 1850<br>" + _TAIL, P),
    ("NASA built Kafka in 1850<br/>" + _TAIL, P),
    # an HTML block is opaque (round 4): residue, never inferred, no label inside counts
    ("<p>NASA built Kafka in 1850</p><p>" + _TAIL + "</p>", U),
    (_TAIL + "<br>Postgres uses MVCC [S2].", V),
    ("NASA built Kafka in 1850.<br>[S1]", U),  # a label after a break is an orphan
    # P2: the lower.lower exemption is for identifiers only
    ("nasa built it in the year eighteen fifty.kafka stores records in partitions [S1].", P),
    ("Kafka runs on node.js.it runs on port 9999 [S1].", P),  # found by the round-2 stress run
    # P2: number words and hedges reach the number and negation checks
    ("Kafka stores records in three partitions.", U),
    ("Kafka stores records in zero partitions.", U),
    ("Kafka hardly stores records in partitions.", U),
    ("Kafka rarely stores records in partitions.", U),
    ("Kafka seldom stores records in partitions.", U),
    ("Kafka barely stores records in partitions.", U),
    ("The cluster has four brokers.", U),
    ("The default port is nine.", U),
    # P2: a labelled refusal makes no claim and shows no citation (test_ag_turns)
    ("Your notes don't cover this [S1].", N),
]
# Review of the coverage redesign, round 3: every input of its findings.
_ATTRIBUTED_TAGS = [
    "<p NASA built Kafka in 1850>",
    "<br title='NASA built Kafka in 1850'>",
    '<div class="NASA built Kafka in 1850">',
    "</p NASA built Kafka in 1850>",
    "<h1 x=9999 brokers>",
    "<BR CLASS=nasa-built-kafka-1850>",
    "<pre NASA>",
    "<li value=1850>",
    "<hr NASA built Kafka in 1850>",
]
ATTACKS += [(f"{_TAIL} {tag}", P) for tag in _ATTRIBUTED_TAGS]  # P1: attribute text is text
ATTACKS += [(f"{_TAIL}{tag}", P) for tag in _ATTRIBUTED_TAGS]
# at a line start most of these open an HTML BLOCK (round 4: the parser says so): the whole
# line is opaque, so nothing is covered and the answer is unsupported, not partial
ATTACKS += [
    (f"{tag}{_TAIL}", U if tag.lstrip("<").split()[0].strip("/").lower() != "br" else P)
    for tag in _ATTRIBUTED_TAGS
]
ATTACKS += [(f"{_TAIL}\n{tag}", P) for tag in _ATTRIBUTED_TAGS]
ATTACKS += [
    # P1: a refusal followed by an attributed tag is not a pure refusal
    ("Your notes don't cover this.<p yes it does, 1850>", U),
    ("Your notes don't cover this.<br title='yes'>", U),
    # bare tags still state nothing, whatever their spacing and case
    (f"{_TAIL}<BR />", V),
    # "< br/ >" is not HTML to the parser (a space after "<"): its text is text, so residue
    # (parser rewrite, round 1: only a tag the parser recognises may state nothing)
    (f"{_TAIL}< br/ >", P),
    (f"<p>{_TAIL}</p>", U),  # an HTML block: opaque (round 4, recall cost in the ADR)
    # P2: an escaped pipe is a visual separator too
    ("NASA built Kafka in 1850 \\| " + _TAIL, P),
    # P2: the noun after a number must be one of its neighbours in the source
    ("Autovacuum runs every 60 brokers.", U),
    ("Autovacuum runs every 60 seconds.", V),
    # P2: code quotes only whole source lines (S3: "kafka-topics --create --topic logs ...")
    ("```\nkafka-topics --create --topic logs\n```", U),
    ("```\nkafka-topics --create --topic logs --partitions 3\n```", U),  # S3 is not shown
]


@pytest.mark.parametrize(("answer", "expected"), ATTACKS)
def test_review_attacks(answer: str, expected: Verification) -> None:
    status, highlighted = status_of(answer)
    assert status is expected
    if status is not Verification.verified:
        draft = normalize_citations(answer)
        assert all(h in draft for h in highlighted)  # highlights are verbatim


@pytest.mark.parametrize("answer", ["Kafka has 99 brokers.", "Sure.", "Done, it was removed."])
def test_a_tool_turn_without_coverage_is_never_verified(answer: str) -> None:
    """Review attack: with a tool result, "Sure." was `verified`. Tool turns are exempt from
    rule 2 (their text rests on tool results), never from the residue."""
    assert status_of(answer, tools_ran=True)[0] is Verification.partial


def test_a_cited_tool_reply_is_verified() -> None:
    assert status_of("The page was removed [T1].", tools_ran=True)[0] is Verification.verified


# Review of the coverage redesign, round 4: the structure the guarantee is computed on came from
# regexes (a second Markdown grammar). Every input of its findings, plus the shapes found while
# moving to the CommonMark parse. S3 (the CLI page) is shown for these (`_R4_SHOWN`).
_S3_CODE = "kafka-topics --create --topic logs --partitions 3"
ATTACKS_R4: list[tuple[str, Verification]] = [
    # indented code with a label: the parser says code, so the label is not a label
    ("    rm -rf /var/lib/kafka [S1]", U),
    (f"{_TAIL}\n\n    rm -rf /var/lib/kafka [S1]", P),
    (f"{_TAIL}\n\n    NASA built Kafka in 1850 [S1].", P),
    # fences and indented code nested in blockquotes and list items
    ("> ```\n> rm -rf /data\n> ```\n> [S1]", U),
    (f"{_TAIL}\n\n> ```\n> rm -rf /data\n> ```", P),
    (f"- {_TAIL}\n\n  ```\n  rm -rf /data\n  ```", P),
    (f"1. {_TAIL}\n\n        rm -rf /data", P),
    (f"> {_TAIL}\n>\n>     rm -rf /data", P),
    (f"- {_TAIL}\n  ```\n  NASA built Kafka in 1850.\n  ```\n  [S1]", P),
    # found by this round's stress run: a block owned its whole first line, so the numeral of
    # a list that does not start at 1 rode on a verbatim-attributed fence
    (f"3. ```\n   {_S3_CODE}\n   ```", P),
    (f"1. ```\n   {_S3_CODE}\n   ```", V),
    (f"> 3. ```\n>    {_S3_CODE}\n>    ```", P),
    # found by this round's stress run: linkify swallowed a final ellipsis into the URL, and
    # the sentence end it hid let one label cover the sentence before it
    ("NASA built Kafka in 1850 https://x.example/a\u2026 " + _TAIL, P),
    # an inline span is atomic: a ".", ";" or line break inside it is not a boundary
    ("Run `rm -rf /. NASA built Kafka in 1850` [S1].", U),
    ("Run `rm -rf /data; NASA built Kafka in 1850` [S1].", U),
    ("Run `rm -rf\n/data` [S1].", U),
    ("NASA built Kafka in 1850 `x.\ny` [S1].", U),
    # double-backtick spans (and their one-space padding) are spans
    ("Run ``rm -rf `x` /data`` [S1].", U),
    ("Run `` NASA built Kafka in 1850. `` [S1].", U),
    # a label the reader does not see as a label covers nothing (and is an invalid fragment)
    ("NASA built Kafka in 1850 [docs](https://x.example/[S1]).", U),
    ('NASA built Kafka in 1850 [docs](https://x.example "[S1]").', U),
    ("NASA built Kafka in 1850 <https://x.example/[S1]>.", U),
    ("NASA built Kafka in 1850 https://x.example/[S1].", U),
    ("NASA built Kafka in 1850 www.x.example/[S1].", U),
    ("NASA built Kafka in 1850 ~~[S1]~~.", U),
    ("NASA built Kafka in 1850 ~[S1]~.", U),
    ("NASA built Kafka in 1850 [see [S1]](https://x.example).", U),
    ("NASA built Kafka in 1850 \\[S1\\].", U),
    ("NASA built Kafka in 1850 &#91;S1&#93;.", U),
    ("NASA built Kafka in 1850 <span title='[S1]'>.", U),
    ("NASA built Kafka in 1850 <!-- [S1]", U),
    # text the parser hides from its tokens is residue, never nothing
    ("[S1]: https://x.example\n\nNASA built Kafka in 1850 [S1].", U),
    (f"{_TAIL}\n\n[NASA built Kafka in 1850]: https://x.example", P),
    ("| Claim | Source |\n|---|---|\n| " + _TAIL + " | ok | NASA built Kafka in 1850 |", P),
    # a zero-width character or a look-alike letter cannot hide a negation or carry evidence
    ("Kafka does n\u03bft store records in partitions.", U),
    ("Kafka does n\u200bot store records in partitions.", U),
    ("Kafka does n\u00adot store records in partitions.", U),
    ("Kafka does \u039d\u039f\u03a4 store records in partitions.", U),
    ("Kafka d\u03bfes not store records in partitions.", U),
    ("Kafka stores records in partiti\u03bfns.", U),
    ("Your n\u03bftes don't cover this.", U),
    # recall (no regression): cited, uncited-correct and whole verbatim code, labelled or
    # inferred, at any nesting depth, still reach `verified`
    ("Kafka stores **records** in partitions. Postgres uses *MVCC*.", V),
    ("List topics with ``kafka-topics --list`` [S3].", V),
    ("List topics with `kafka-topics --list` [S3].", V),
    (f"Create a topic [S3]:\n\n```bash\n{_S3_CODE}\n```\n[S3]", V),
    (f"Create a topic [S3]:\n\n```bash\n{_S3_CODE}\n```", V),
    (f"- Create a topic with kafka-topics [S3]:\n\n  ```\n  {_S3_CODE}\n  ```", V),
    (f"> Create a topic with kafka-topics [S3]:\n>\n> ```\n> {_S3_CODE}\n> ```\n> [S3]", V),
    (f"Create a topic with kafka-topics [S3]:\n\n    {_S3_CODE}", V),
    (f"> {_TAIL}\n> Postgres uses MVCC [S2].", V),
    ("- Kafka stores records in partitions.\n- Postgres uses MVCC.", V),
    (f"See the [Kafka docs](https://kafka.apache.org) for details: {_TAIL}", V),
    ("Kafka stores records in partitions\r\n[S1]", V),
    ("Kafka does not delete records [S1]. \U0001f44d", V),
]


@pytest.mark.parametrize("linkify", LINKIFY_CONFIGS)
@pytest.mark.parametrize(("answer", "expected"), ATTACKS_R4)
def test_review_attacks_round_4(answer: str, expected: Verification, linkify: bool) -> None:
    with parser_config(linkify=linkify):
        status, highlighted = status_of(answer, shown=SHOWN)
        assert status is expected
        if status is not Verification.verified:
            draft = normalize_citations(answer)
            assert all(h in draft for h in highlighted)  # highlights are verbatim


# Review of the parser rewrite, round 1: every input of its findings (and probes around them).
_S4_LINES = ("cd /tmp/build", "rm -rf cache")
_S5_CODE = "if ok:\n    deploy()\ncleanup()"
_S5_MIXED = " if ok:\n\t    deploy()\n\tcleanup()"  # removing 1 char each is not a uniform dedent
# every character str.splitlines() treats as a line break, other than "\n" and "\r"
_LINE_BREAKS = ["\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"]
_HIDDEN_NOT = [
    "no\u3164t",  # HANGUL FILLER (Lo, default-ignorable): renders blank
    "no\u115ft",  # HANGUL CHOSEONG FILLER
    "no\u1160t",  # HANGUL JUNGSEONG FILLER
    "no\uffa0t",  # HALFWIDTH HANGUL FILLER
    "no\u034ft",  # COMBINING GRAPHEME JOINER (Mn)
    "no\ufe0ft",  # VARIATION SELECTOR-16 (Mn)
    "no\u180bt",  # MONGOLIAN FREE VARIATION SELECTOR ONE (Mn)
    "no\u17b4t",  # KHMER VOWEL INHERENT AQ (Mn)
    "\u0274\u1d0f\u1d1b",  # small capitals N O T
    "\u0274ot",  # a small capital mixed with ordinary letters
    "n\u1d0f\u1d1b",
]
ATTACKS_R5: list[tuple[str, Verification]] = [
    # P2: code is compared on "\n" only; a Unicode line separator is not a line break
    *(
        (f"```sh\n{_S4_LINES[0]}{sep}{_S4_LINES[1]}\n```{cite}", U)
        for sep in _LINE_BREAKS
        for cite in ("", "\n[S4]")
    ),
    *((f"{_TAIL}\n\n```sh\n{_S4_LINES[0]}{sep}{_S4_LINES[1]}\n```", P) for sep in _LINE_BREAKS),
    # a trailing form feed or line separator is part of the code, not trailing whitespace
    *((f"List topics with `kafka-topics --list{sep}` [S3].", U) for sep in _LINE_BREAKS),
    ("```\nkafka-topics --list\x0c\n```", U),
    # P2: a negation hidden by an invisible filler or by small capitals withholds a candidate
    *((f"Kafka does {word} store records in partitions.", U) for word in _HIDDEN_NOT),
    *((f"{_TAIL} Postgres does {word} use MVCC.", P) for word in _HIDDEN_NOT),
    # P2: a marker inside a linkified URL or an autolink is an invalid fragment (rule 1), with
    # or without linkify
    ("Kafka stores records in partitions http://x.com/[S1] .", U),
    ("Kafka stores records in partitions <http://x/[S1]>.", U),
    ("Kafka stores records in partitions https://x.example/a[S1]b and more.", U),
    (f"{_TAIL} See <https://x.example/[S2]>.", U),
    # strikethrough is the parser's GFM reading: "~x~" and "~~x~~" strike; three tildes do not
    # for GFM but do for markdown-it's default reading, so no label of that part counts
    ("NASA built Kafka in 1850 ~~~[S1]~~~.", U),
    ("NASA built Kafka in 1850 ~[S1]~ and ~~[S2]~~.", U),
    # recall (no regression): the exact two-line block, inferred or labelled; "~" meaning
    # "approximately" (P2: the old whole-paragraph tilde heuristic rejected these labels)
    (f"```sh\n{_S4_LINES[0]}\n{_S4_LINES[1]}\n```", V),
    (f"```sh\n{_S4_LINES[0]}\n{_S4_LINES[1]}\n```\n[S4]", V),
    (f"```sh\n{_S4_LINES[0]}  \n{_S4_LINES[1]}\t\n```", V),  # trailing blanks only
    ("The cluster has ~3 brokers [S1]. Autovacuum runs every ~60 seconds [S2].", V),
    ("Kafka stores records in partitions (~3 per broker, ~60 s apart) on ~3 brokers [S1].", V),
    ("Kafka does not delete records [S1]. Kafka stores records in partitions [S1].", V),
    (f"See <https://kafka.apache.org> and {_TAIL}", V),  # an autolink inside a cited sentence
]


@pytest.mark.parametrize("linkify", LINKIFY_CONFIGS)
@pytest.mark.parametrize(("answer", "expected"), ATTACKS_R5)
def test_review_attacks_parser_round_1(answer: str, expected: Verification, linkify: bool) -> None:
    with parser_config(linkify=linkify):
        status, highlighted = status_of(answer, shown=SHOWN)
        assert status is expected, answer
        if status is not Verification.verified:
            draft = normalize_citations(answer)
            assert all(h in draft for h in highlighted)  # highlights are verbatim


# Blind review round 2 of the parser rewrite.
ATTACKS_R6: list[tuple[str, Verification]] = [
    # P2: the dedent is the common whitespace PREFIX, not a count: mixed tabs and spaces that
    # move cleanup() under the `if` never match S5's block, inferred or labelled
    (f"```python\n{_S5_MIXED}\n```", U),
    (f"```python\n{_S5_MIXED}\n```\n[S5]", U),
    (f"{_TAIL}\n\n```python\n{_S5_MIXED}\n```", P),
    (f"{_TAIL}\n\n    {_S5_MIXED.replace(chr(10), chr(10) + '    ')}", P),
    # P2: a bare label right after HTML (a tag or a comment) never covers it
    (f"{_TAIL} <span title='NASA built Kafka in 1850'> [S1]", P),
    ("Kafka stores records in partitions.<!-- NASA built Kafka in 1850 --> [S1]", P),
    (f"{_TAIL} <img src=x alt='NASA built Kafka'>[S1]", P),
    (f"{_TAIL} <p NASA built Kafka in 1850> [S1]", P),
    # P2: a bare label right after a leading enumerator never covers it (it may BE the answer)
    (f"{_TAIL} 3) [S1]", P),
    (f"{_TAIL} (7) [S1]", P),
    # P2: negation forms the review found (they withhold candidates; the verifier backstops)
    ("Kafka doesnt store records in partitions.", U),
    # "." cuts the sentence here, so pieces are judged alone: never verified
    ("Kafka does n.o.t store records in partitions.", P),
    # "." cuts the sentence here, so pieces are judged alone: never verified
    ("Kafka does n.o.t store records in partitions.", P),
    ("Kafka is incapable of storing records in partitions.", U),
    ("The Kafka cluster has a dozen brokers.", U),
    # recall: S5's block exact or uniformly dedented, labelled or inferred; prose with a word
    # that merely ends in "nt" is not negated; an enumerator with its sentence stays covered
    (f"```python\n{_S5_CODE}\n```", V),
    (f"```python\n{_S5_CODE}\n```\n[S5]", V),
    ("```python\n" + "\n".join(f"\t{ln}" for ln in _S5_CODE.split("\n")) + "\n```", V),
    (f"> ```python\n> {_S5_CODE.replace(chr(10), chr(10) + '> ')}\n> ```", V),
    ("The Kafka client stores records in partitions [S1].", V),
    ("Kafka stores records in partitions [S1].<br>The cluster has 3 brokers [S1].", V),
]


@pytest.mark.parametrize("linkify", LINKIFY_CONFIGS)
@pytest.mark.parametrize(("answer", "expected"), ATTACKS_R6)
def test_review_attacks_parser_round_2(answer: str, expected: Verification, linkify: bool) -> None:
    with parser_config(linkify=linkify):
        status, highlighted = status_of(answer, shown=SHOWN)
        assert status is expected, answer
        if status is not Verification.verified:
            draft = normalize_citations(answer)
            assert all(h in draft for h in highlighted)  # highlights are verbatim


# Blind review round 3 of the parser rewrite: every input of its findings, plus probes.
_TITLED = '(https://x.example "NASA built Kafka in 1850; Kafka does not use partitions")'
# every apostrophe look-alike `coverage.APOSTROPHES` folds, and one it does not (U+02EE)
_APOSTROPHE_LOOKALIKES = "\u00b4\u2032\u2035\u0060\u02b9\u02bb\u02bd\u02c8\u02ca\u02cb\ua78c\uff40"
_APOSTROPHE_LOOKALIKES += "\u0384\u1ffd\u1fef\u055a\u05f3\u201b\uff07\u02ee"
_UNICODE_SPACES = ["\u00a0", "\u202f", "\u2003", "\u3000"]
ATTACKS_R7: list[tuple[str, Verification]] = [
    # P2: a sentence cut inside a link's text cannot let the text, destination and title ride on
    # an inferred unit: every atom of a link is opaque
    *((f"[{cut} Kafka stores records in partitions]{_TITLED}", U) for cut in "!\u2026;."),
    *((f"{_TAIL} [{cut} Kafka stores records in partitions]{_TITLED}", P) for cut in "!\u2026;."),
    ("Kafka stores records in [partitions](https://nasa.example/built-kafka-1850).", U),
    ("[Kafka stores records in partitions](https://x.example). The cluster has 3 brokers.", P),
    # P2: str.strip's Unicode whitespace at the edges of a paragraph, heading or cell (recall;
    # an uncited sentence next to it is still residue)
    *((f"{_TAIL}{ws}", V) for ws in _UNICODE_SPACES),
    *((f"{ws}{_TAIL}", V) for ws in _UNICODE_SPACES),
    *((f"{_TAIL}{ws}\n\nPostgres uses MVCC [S2].{ws}", V) for ws in _UNICODE_SPACES),
    *((f"## Kafka stores records in partitions [S1].{ws}", V) for ws in _UNICODE_SPACES),
    *((f"| {_TAIL}{ws}|{ws}Postgres uses MVCC [S2] |\n|---|---|", V) for ws in _UNICODE_SPACES),
    *((f"{_TAIL}{ws}\n\nNASA built Kafka in 1850.{ws}", P) for ws in _UNICODE_SPACES),
    *((f"| {_TAIL}{ws}|{ws}NASA built Kafka |\n|---|---|", P) for ws in _UNICODE_SPACES),
    # P2: an apostrophe look-alike cannot hide an n't negation (folded, or the word is opaque)
    *((f"Kafka doesn{c}t store records in partitions.", U) for c in _APOSTROPHE_LOOKALIKES),
    *((f"{_TAIL} Postgres doesn{c}t use MVCC.", P) for c in _APOSTROPHE_LOOKALIKES),
    ("Kafka\u00b4s brokers store records in partitions.", V),  # a folded possessive: recall
    ("Your notes don\u00b4t cover this.", Verification.unverified),
    # P2: the numeral of an empty list item is residue (it may BE the answer)
    (f"1. {_TAIL}\n\nHere is what I found:\n\n2.", P),
    (f"{_TAIL}\n\n1)\n2)\n3)", P),
    ("Your notes don't cover this.\n\n1.\n2.\n3.", U),
    (f"{_TAIL}\n\n1. [S1]\n2. [S2]", P),
    (f"{_TAIL}\n\n1. \U0001f44d", P),
    # recall: the numerals of items with content stay structure
    (f"1. {_TAIL}\n2. Postgres uses MVCC [S2].", V),
    (f"1. - {_TAIL}\n2. Postgres uses MVCC [S2].", V),
    (f"1. {_TAIL}\n\n   ```bash\n   {_S3_CODE}\n   ```\n2. Postgres uses MVCC [S2].", V),
]


@pytest.mark.parametrize("linkify", LINKIFY_CONFIGS)
@pytest.mark.parametrize(("answer", "expected"), ATTACKS_R7)
def test_review_attacks_parser_round_3(answer: str, expected: Verification, linkify: bool) -> None:
    with parser_config(linkify=linkify):
        status, highlighted = status_of(answer, shown=SHOWN)
        assert status is expected, answer
        if status is not Verification.verified:
            draft = normalize_citations(answer)
            assert all(h in draft for h in highlighted)  # highlights are verbatim


# ACCEPTED RESIDUALS (ADR-0015 section 4), pinned so that changing one is a deliberate decision:
# a pictograph states nothing, including the polarity/verdict glyphs that flip or grade a cited
# claim, and the negation forms the lexical guard does not read reach `verified` with a verifier
# that always says supported. The real verifier is the only backstop for these.
ACCEPTED_RESIDUALS = [
    *(f"{_TAIL} {glyph}" for glyph in ("\u274c", "\u2717", "\U0001f44e", "\U0001f6ab")),
    "Kafka stores records in partitions [S1] \u2605\u2605\u2605\u2606",
    "Kafka stores records in partitions \u2717",  # inferred
    "Kafka does n o t store records in partitions.",  # letters spaced out
    "The Kafka cluster has III brokers.",  # a Roman numeral is not read as a number
]
# Formerly accepted, now blocked by a later fail-closed rule (round 6: `_foreign_tail` cuts a
# free-standing non-Latin letter, so "Kafka" and "stores records ..." are separate units and the
# uncited one is residue). Pinned so a loosening of that rule is noticed.
FORMERLY_ACCEPTED_NOW_BLOCKED = [
    "Kafka \u4e0d stores records in partitions.",  # a non-English negation (CJK "not")
]


@pytest.mark.parametrize("answer", FORMERLY_ACCEPTED_NOW_BLOCKED)
def test_formerly_accepted_residuals_are_now_blocked(answer: str) -> None:
    assert status_of(answer, shown=SHOWN)[0] is not Verification.verified, answer


@pytest.mark.parametrize("answer", ACCEPTED_RESIDUALS)
def test_accepted_residuals_are_pinned(answer: str) -> None:
    assert status_of(answer, shown=SHOWN)[0] is Verification.verified, answer


def test_every_format_character_is_stripped_with_the_default_ignorables() -> None:
    """`clean` strips Default_Ignorable_Code_Point (a literal Unicode 15.0 table) and every Cf
    character; the Hangul fillers make text opaque (they are letters a tokenizer keeps)."""
    from ragforge_core.agent.coverage import DEFAULT_IGNORABLE, clean, opaque

    for code in range(0x110000):
        char = chr(code)
        if unicodedata.category(char) == "Cf" or char in DEFAULT_IGNORABLE:
            assert clean(f"a{char}b") == "ab", hex(code)
    for filler in "\u115f\u1160\u3164\uffa0":
        assert opaque(f"no{filler}t")
    assert not opaque("Kafka \u2764\ufe0f records")  # a variation selector is not a filler


@pytest.mark.parametrize(
    "answer",
    [
        "NASA built Kafka in 1850 ~~[S1]~~.",
        "NASA built Kafka in 1850 \\[S9\\].",
        "NASA built Kafka in 1850 [see [S1]](https://x.example).",
        "Kafka stores records in partitions [S1]. Also \\[S9\\].",
    ],
)
def test_a_label_the_reader_sees_but_that_is_not_one_is_an_invalid_fragment(answer: str) -> None:
    """Round 4: a marker the reader would take for a citation but that is not accepted as one
    (struck through, escaped, in a link's text) is rejected (rule 1), never decoration riding on
    a covered sentence."""
    report = check_citations(normalize_citations(answer), SHOWN, strict=False)
    assert not report.ok
    assert report.cited == () or "S9" in answer


def test_structure_finds_code_at_any_depth_with_the_parsers_content() -> None:
    answer = (
        "> - ```py\n>   one\n>   ```\n\n"
        "1. x\n\n       two\n\n"
        "Run ``a `b` c`` and `d\ne`.\n\n"
        "| `f \\| g` | h |\n|---|---|\n"
    )
    found = [(c.text, c.block) for c in structure(answer).code]
    assert found == [
        ("one\n", True),
        ("two\n", True),
        ("a `b` c", False),
        ("d e", False),
        ("f | g", False),
    ]
    for code in structure(answer).code:
        assert answer[code.start : code.end].strip()  # every element has a real source span


def test_text_the_parser_hides_is_an_opaque_unit() -> None:
    """A reference definition renders nothing in CommonMark but is shown literally by the
    sanitizer; a cell beyond the header's width is dropped by the parser. Neither may vanish."""
    for answer, hidden in [
        ("[S1]: https://x.example\n\nfoo", "S1]: https://x.example"),
        ("| a |\n|---|\n| b | NASA built Kafka |", "NASA built Kafka"),
        ("3. Kafka", "3"),
    ]:
        assert any(u.opaque and hidden in u.text for u in segment(answer)), answer


# ---------------------------------------------------------------- the property, with an oracle
# The oracle is the TEST's own reading of ADR-0015 section 4, written without structure.py or
# coverage.py. Since review round 4 it reads the answer with its OWN markdown-it-py parse: code
# (blocks at any depth and inline spans) and the labels the reader sees come from the parser's
# tokens, so a regex view of Markdown cannot hide a blind spot shared with the implementation.
# It needs no source offsets: it works on what the reader sees (each inline block's visible
# text) and compares inferred units by their visible text. Where it is unsure it is LENIENT
# (it never flags what the ADR allows), so a failure is always a real bypass.
_ORACLE_MD = MarkdownIt("commonmark", {"html": True, "strikethrough_single_tilde": True}).enable(
    ["table", "strikethrough"]
)
_ORACLE_TERMS = frozenset(
    chr(c)
    for c in range(0x30000)
    if unicodedata.category(chr(c)).startswith("P")
    and any(
        w in unicodedata.name(chr(c), "")
        for w in ("FULL STOP", "QUESTION MARK", "EXCLAMATION MARK", "DANDA", "ELLIPSIS")
    )
)
_ORACLE_TERMS |= frozenset(";\uff1b")  # a semicolon ends the clause a label covers (ADR-0015)
_ORACLE_CLOSERS = "\"'\u201d\u2019)]"
_ORACLE_LABEL = re.compile(r"\[[ST]\d+(?:, ?[ST]\d+)*\]")
_ORACLE_ENUM = re.compile(r"(?:\(\w{1,3}\)|\w{1,3}[.)])(?=\s)")
_ORACLE_BARE_TAG = re.compile(r"</?(?:br|p)\s*/?>", re.IGNORECASE)
_ORACLE_TAG = re.compile(r"</?[A-Za-z][^<>\n]*>")
# The oracle's "states nothing": punctuation, separators, marks and format characters by
# category, the markdown symbols, and the few emoji the generator uses, listed by hand.
_ORACLE_QUIET = frozenset("|>`~+<=^\U0001f44d\U0001f389\U0001f680\u2705\U0001f642")
_LEAD = " \t;,:\u2013\u2014\u00a0"  # what a unit may start with that it does not own
_CODE_MARK = "\x01"  # one visible "character" per inline code span: never cut inside
_ATOM_MARK = "\x02"  # an image or a URL: visible, never a label, never cut inside
_ORACLE_LANGUAGES = {"", "python", "sh", "bash"}


def _says(text: str) -> bool:
    return any(
        not (
            c.isspace()
            or unicodedata.category(c)[0] in "PZM"
            or unicodedata.category(c) == "Cf"
            or c in _ORACLE_QUIET
        )
        for c in text
    )


_ORACLE_ALLOWED = {
    "your notes don't cover this",
    "let me know if you have other questions",
    "i hope this helps",
    "hope that helps",
}


def _phrase(text: str) -> str:
    """The ADR matches the allowlist on WORDS: punctuation and emoji never change a phrase, but
    anything else that is not a word does (braille is no word, and it says something)."""
    if _says(re.sub(r"\w+(?:'\w+)?", " ", text)):
        return "\x00"  # matches no allowlisted phrase
    return " ".join(re.findall(r"\w+(?:'\w+)?", text.lower()))


def _oracle_cuts(text: str) -> list[int]:
    """Where the oracle's sentences end: after every terminator run (plus closing quotes), except
    a lone "." between two digits or two lower-case letters of an identifier, or ending
    "e.g."/"i.e."; after every "|"; around every line break, Unicode line separators included."""
    cuts: list[int] = []
    i = 0
    while i < len(text):
        if text[i] in "|\n\u2028\u2029\x85\x0b\x0c\x1c\x1d\x1e":
            cuts.extend((i, i + 1))
            i += 1
            continue
        if text[i] not in _ORACLE_TERMS:
            i += 1
            continue
        j = i
        while j < len(text) and text[j] in _ORACLE_TERMS:
            j += 1
        prev, nxt = text[i - 1 : i], text[j : j + 1]
        lone_dot = j - i == 1 and text[i] == "."
        after = re.match(r"[a-z]*", text[j:])
        if lone_dot and (
            (prev.isdigit() and nxt.isdigit())
            or (prev.islower() and after is not None and after[0] in ("js", "py", "conf", "g", "e"))
            or (text[max(0, i - 3) : i + 1].lower() in ("e.g.", "i.e."))
        ):
            i = j
            continue
        while j < len(text) and text[j] in _ORACLE_CLOSERS:
            j += 1
        cuts.append(j)
        i = j
    return cuts


def _url_cuts(text: str) -> list[int]:
    """The oracle's cuts, minus those inside a URL-shaped word (linkify makes it one link; the
    oracle is lenient where a renderer may differ)."""
    urls = [
        (w.start(), w.end())
        for w in re.finditer(r"\S+", text)
        if "://" in w.group(0) or "www." in w.group(0)
    ]
    return [c for c in _oracle_cuts(text) if not any(a < c < b for a, b in urls)]


def _norm_code(text: str) -> str:
    """Exact up to trailing spaces/tabs, a uniform dedent and blank edge lines; lines are split
    on "\\n" only (a U+2028 inside a line is part of the line)."""
    lines = [ln.rstrip(" \t") for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return textwrap.dedent("\n".join(lines))


def _source_code() -> set[str]:
    """Every whole code element of the shown pages, by the oracle's own parse."""
    found: set[str] = set()
    for hit in HITS:
        for token in _ORACLE_MD.parse(hit["display_text"]):
            if token.type in ("fence", "code_block"):
                found.add(_norm_code(token.content))
            for child in token.children or []:
                if child.type == "code_inline":
                    found.add(_norm_code(child.content))
    return found


_SOURCE_CODE = _source_code()


@dataclass
class _Visible:
    """One inline block as the oracle reads it."""

    text: str  # visible text, code spans as _CODE_MARK, images/URLs as _ATOM_MARK
    label_ok: list[bool]  # per character: plain text outside links/strikethrough/HTML
    codes: list[str]  # the content of each _CODE_MARK, in order
    boundary: set[int]  # forced cuts (HTML, hard breaks)
    hard: set[int]  # positions of hard breaks and HTML (a bare label after them is an orphan)
    html: list[tuple[int, int]]  # HTML and tag-shaped text: a bare label never cites a piece
    # holding one (blind review round 2, P2: a renderer may hide it)
    links: list[tuple[int, int]]  # a link's text: hides a destination/title, so never inferred


def _visible(children: Sequence[Token]) -> _Visible:
    out = _Visible("", [], [], set(), set(), [], [])
    link = strike = 0
    link_start = 0
    skip = False
    for child in children:
        kind = child.type
        if skip:
            skip = kind != "link_close"
            continue
        if kind == "text":
            out.text += child.content
            out.label_ok += [not link and not strike] * len(child.content)
        elif kind == "text_special":
            out.text += child.content
            out.label_ok += [False] * len(child.content)
        elif kind in ("softbreak", "hardbreak"):
            if kind == "hardbreak":
                out.hard.add(len(out.text))
            out.text += "\n"
            out.label_ok.append(False)
        elif kind == "code_inline" and not child.content.strip():
            # an empty code box ("``` ```") shows nothing: it states nothing (as in production)
            out.text += " "
            out.label_ok.append(False)
        elif kind == "code_inline":
            out.codes.append(child.content)
            out.text += _CODE_MARK
            out.label_ok.append(False)
            if child.content.rstrip()[-1:] in _ORACLE_TERMS:
                # round 8: "`2017.`" ends the sentence it closes, like a terminator after it
                out.boundary.add(len(out.text))
        elif kind == "html_inline":
            out.boundary.update((len(out.text), len(out.text) + len(child.content)))
            out.hard.add(len(out.text))
            out.html.append((len(out.text), len(out.text) + len(child.content)))
            out.text += child.content
            out.label_ok += [False] * len(child.content)
        elif kind == "image" or (kind == "link_open" and child.markup in ("autolink", "linkify")):
            out.text += _ATOM_MARK
            out.label_ok.append(False)
            skip = kind == "link_open"
        elif kind == "link_open":
            link += 1
            link_start = len(out.text)
        elif kind == "link_close":
            link -= 1
            out.links.append((link_start, len(out.text) + 1))  # the closing bracket too
        elif kind in ("s_open", "s_close"):
            strike += 1 if kind == "s_open" else -1
    for tag in _ORACLE_TAG.finditer(out.text):
        out.boundary.update((tag.start(), tag.end()))
        out.html.append((tag.start(), tag.end()))
        out.label_ok[tag.start() : tag.end()] = [False] * (tag.end() - tag.start())
    for word in re.finditer(r"\S+", out.text):  # a URL-shaped word may be linkified
        if "://" in word.group(0) or "www." in word.group(0):
            out.label_ok[word.start() : word.end()] = [False] * (word.end() - word.start())
    return out


def _flat(text: str) -> str:
    """Whitespace collapsed, and the edges a unit may or may not own (leading clause
    punctuation, trailing terminators, closers and pipes) dropped, for comparing visible text."""
    tail = "".join([" |", *_ORACLE_TERMS, _ORACLE_CLOSERS])
    return " ".join(text.split()).lstrip(_LEAD).rstrip(tail)


def _written_out(text: str, codes: Sequence[str]) -> str:
    """`text` with each code mark replaced by its span's content, in order."""
    pieces = text.split(_CODE_MARK)
    return "".join(p + (codes[i] if i < len(pieces) - 1 else "") for i, p in enumerate(pieces))


def _kept_visible(kept: Sequence[str]) -> list[str]:
    """The visible text of every kept (inferred) prose unit, code spans written out."""
    out: list[str] = []
    for text in kept:  # read as inline text: a kept unit is a piece of one paragraph or cell
        for token in _ORACLE_MD.parseInline(text):
            if token.type == "inline":
                seen = _visible(token.children or [])
                out.append(_flat(_written_out(seen.text, seen.codes)))
    return out


# A code span stands alone after whitespace/start (opening quotes or brackets allowed between)
# and before whitespace/end (".,;:!?" and closing quotes or brackets allowed between).
_OPENERS = "\"'([{\u2018\u201c\u00ab"
_CLOSERS = "\"')]}\u2019\u201d\u00bb"


def _code_ok(code: str, *, inline: bool = True) -> bool:
    """Whole code elements of the shown pages only (the prose-quote exception was removed after
    review round 9)."""
    return _norm_code(code) in _SOURCE_CODE


def _bare_labels(seen: _Visible) -> bool:
    """The block is nothing but valid labels the reader sees (a sources column of a table)."""
    labels = [
        m for m in _ORACLE_LABEL.finditer(seen.text) if all(seen.label_ok[m.start() : m.end()])
    ]
    rest = _ORACLE_LABEL.sub(" ", seen.text)
    valid = all(x.strip() in VALID for m in labels for x in m.group(0)[1:-1].split(","))
    return bool(labels) and valid and not _says(rest)


def _oracle_space(text: str, i: int, hard: set[int]) -> bool:
    """A visible space the reader (and a copy) keeps: not a hard break, not a control."""
    c = text[i]
    return i not in hard and (c in " \t\n" or unicodedata.category(c) == "Zs")


def _oracle_span_alone(text: str, at: int, hard: set[int]) -> bool:
    """The answer's code span at `at` stands alone (review round 4, P1): before it the start or a
    space (opening brackets/quotes allowed between), after it ".,;:!?" and closers then a space
    or the end. Touching another span, an invisible character or a hard break joins it."""
    i = at
    while i > 0 and text[i - 1] in _OPENERS:
        i -= 1
    if i > 0 and not _oracle_space(text, i - 1, hard):
        return False
    j = at + 1
    while j < len(text) and (text[j] in ".,;:!?" or text[j] in _CLOSERS):
        j += 1
    return j == len(text) or _oracle_space(text, j, hard)


def _oracle_inline(seen: _Visible, kept: list[str]) -> list[str]:
    """The residue of one inline block."""
    text = seen.text
    labels = [m for m in _ORACLE_LABEL.finditer(text) if all(seen.label_ok[m.start() : m.end()])]
    valid = [m for m in labels if all(x.strip() in VALID for x in m.group(0)[1:-1].split(","))]
    cuts = sorted({0, len(text), *seen.boundary, *_url_cuts(text)})
    code_at = [i for i, c in enumerate(text) if c == _CODE_MARK]
    left: list[str] = []
    # (start, end) of the last piece that says something and is not covered yet
    last: tuple[int, int] | None = None

    def visible(a: int, b: int) -> str:
        written = _written_out(text[a:b], seen.codes[bisect.bisect_left(code_at, a) :])
        return _flat(written)

    def codes_ok(a: int, b: int) -> bool:
        lo, hi = bisect.bisect_left(code_at, a), bisect.bisect_left(code_at, b)
        return all(_code_ok(c) for c in seen.codes[lo:hi]) and all(
            _oracle_span_alone(text, code_at[k], seen.hard) for k in range(lo, hi)
        )

    def settle(a: int, b: int) -> None:
        nonlocal last
        piece = text[a:b]
        if not _says(piece.replace(_ATOM_MARK, "x").replace(_CODE_MARK, "x")):
            return
        if _ORACLE_BARE_TAG.fullmatch(piece.strip()):
            return  # a bare break tag states nothing
        words = _phrase(piece) if _CODE_MARK not in piece and _ATOM_MARK not in piece else ""
        if words in _ORACLE_ALLOWED:
            return
        # Trailing terminators and spaces state nothing: production may cut before them (round 6
        # `_foreign_tail` ends "... \u24d0" before a " !" that the oracle's own cut keeps), so the
        # coverage question is about the piece without them.
        core_end = b
        while core_end > a and (
            text[core_end - 1].isspace() or text[core_end - 1] in _ORACLE_TERMS
        ):
            core_end -= 1
        flat = visible(a, core_end)
        in_link = any(x < core_end and a < y for x, y in seen.links)
        if any(flat and flat in k for k in kept) and codes_ok(a, core_end) and not in_link:
            return
        left.append(piece)
        last = (a, b)

    for a, b in itertools.pairwise(cuts):
        if a == b:
            continue
        lead = a + len(text[a:b]) - len(text[a:b].lstrip(_LEAD))
        enum = _ORACLE_ENUM.match(text, lead, b)
        if enum:  # a leading enumerator is a piece of its own (never covered by a label)
            settle(lead, enum.end())
            last = None
            a = enum.end()
        cursor = a
        inside = [m for m in labels if a <= m.start() and m.end() <= b]
        for m in inside:
            before = text[cursor : m.start()]
            if _says(_ORACLE_LABEL.sub(" ", before).replace(_CODE_MARK, "x")):
                if m in valid and codes_ok(cursor, m.start()):
                    last = None  # covered by the model's label
                else:
                    settle(cursor, m.end())
            elif m in valid:  # a bare label run: cites the piece right before it, if adjacent
                gap = text[last[1] : m.start()] if last else ""
                gap_ok = all(
                    c.isspace() or c in _ORACLE_TERMS or c in _ORACLE_CLOSERS + "|" for c in gap
                )
                hard = seen.hard & set(range(last[1], m.start())) if last else set()
                html = last and any(a < last[1] and last[0] < b for a, b in seen.html)
                if last and gap_ok and gap.count("\n") <= 1 and not hard and not html:
                    left.remove(text[last[0] : last[1]])
                last = None
            cursor = m.end()
        tail = text[cursor:b]
        if inside and not _says(tail.replace(_CODE_MARK, "x").replace(_ATOM_MARK, "x")):
            continue
        settle(cursor, b)
    return left


def _oracle_residue(answer: str, kept: Sequence[str]) -> list[str]:
    """The test's own residue (empty means every visible thing is accounted for)."""
    env: dict[str, object] = {}
    tokens = _ORACLE_MD.parse(answer, env)
    left: list[str] = []
    if env.get("references"):
        left.append("<a reference definition: rendered as text by the sanitizer>")
    kept_prose = _kept_visible(kept)
    kept_code = {
        _norm_code(t.content)
        for k in kept
        for t in _ORACLE_MD.parse(k)
        if t.type in ("fence", "code_block")
    }
    counted = 0  # the last numeral the most recent ordered list rendered
    lists: list[list[int]] = []  # [start, items] per open ordered list
    # per open list item: its numeral ("" for a bullet) and whether it says anything besides
    # labels; an empty item's numeral is residue (blind review round 3, P2)
    open_items: list[list[str | bool]] = []

    def has_content() -> None:
        for item in open_items:
            item[1] = True

    previous_cell: list[str] = []  # the residue the previous cell of this row left
    for i, token in enumerate(tokens):
        kind = token.type
        if kind == "ordered_list_open":
            start = int(token.attrGet("start") or 1)
            if start not in (1, counted + 1):
                left.append(f"{start}.")
            lists.append([start, 0])
        elif kind == "list_item_open":
            ordered = bool(lists) and token.markup in ".)"
            if ordered:
                lists[-1][1] += 1
            open_items.append([f"{token.info}{token.markup}" if ordered else "", False])
        elif kind == "list_item_close":
            numeral, content = open_items.pop()
            if numeral and not content:
                left.append(str(numeral))
        elif kind == "ordered_list_close":
            start, items = lists.pop()
            counted = start + items - 1
        elif kind in ("fence", "code_block"):
            has_content()
            # a run of valid labels starting the paragraph on the very next line cites the block
            nxt = tokens[i + 1 : i + 3]
            run = (
                re.match(rf"(?:\s*{_ORACLE_LABEL.pattern})+", nxt[1].content)
                if len(nxt) == 2 and nxt[0].type == "paragraph_open"
                else None
            )
            labelled = (
                run is not None
                and nxt[0].map is not None
                and token.map is not None
                and nxt[0].map[0] == token.map[1]
                and nxt[0].level == token.level
                and all(
                    x.strip() in VALID
                    for m in _ORACLE_LABEL.finditer(run.group(0))
                    for x in m.group(0)[1:-1].split(",")
                )
            )
            # inferred: a kept unit is (a piece of) this block's source holding all its code
            lines = answer.split("\n")[token.map[0] : token.map[1]] if token.map else []
            source = "\n".join(lines)
            inferred = _norm_code(token.content) in kept_code or any(
                k in source and all(x.strip(" \t") in k for x in token.content.split("\n"))
                for k in kept
            )
            exact = (
                _norm_code(token.content) in _SOURCE_CODE
                and token.info.strip() in _ORACLE_LANGUAGES
            )
            if (token.content.strip() or token.info.strip()) and not (
                (labelled or inferred) and exact
            ):
                left.append(token.content)
        elif kind == "html_block":
            has_content()
            if _says(_ORACLE_BARE_TAG.sub(" ", token.content)):
                left.append(token.content)
        elif kind == "tr_open":
            previous_cell = []
        elif kind == "inline":
            seen = _visible(token.children or [])
            if _says(_ORACLE_LABEL.sub(" ", seen.text).replace(_CODE_MARK, "x")):
                has_content()
            cell = tokens[i - 1].type in ("td_open", "th_open")
            if cell and previous_cell and _bare_labels(seen):
                left.remove(previous_cell[-1])  # a sources column cites the cell before it
            pieces = _oracle_inline(seen, kept_prose)
            left.extend(pieces)
            previous_cell = pieces if cell else []
    return left


_FACTS = [
    "Kafka stores records in partitions",
    "The cluster has 3 brokers",
    "The default port is 9092",
    "Postgres uses MVCC",
    "Vacuum reclaims dead tuples",
    "Autovacuum runs every 60 seconds by default",
    "Kafka does not store records in partitions",
    "Kafka fails to store records in partitions",
    "The default port is 9093",
    "The cluster has -3 brokers",
    "Kafka was invented by NASA in 1850",
    "Kafka has 3 partitions",
    "it runs on port 9999",
    "kafka stores records in partitions",
    "Kafka stores records in three partitions",
    "The cluster has three brokers",
    "Kafka hardly stores records in partitions",
    "nasa built it in eighteen fifty.kafka stores records in partitions",
    "Kafka stores records in partitions \u281d\u2801\u280e\u2801",
    "Kafka stores records in partitions \u24c3\u24d0\u24e2\u24d0",
    "Kafka runs on node.js",
    "It does",
    "Yes",
    "No",
    # round 4: look-alike letters and format characters (a hidden negation, borrowed evidence)
    "Kafka does n\u03bft store records in partitions",
    "Kafka does n\u200bot store records in partitions",
    "Kafka does \u039d\u039f\u03a4 store records in partitions",
    "Kafka stores records in partiti\u03bfns",
    "Kafka stores **records** in partitions",
    # parser rewrite, round 1: invisible fillers and small capitals; "~" as "approximately"
    *(f"Kafka does {word} store records in partitions" for word in _HIDDEN_NOT),
    "The cluster has ~3 brokers",
    "Autovacuum runs every ~60 seconds by default",
    # parser rewrite, round 3: apostrophe look-alikes (folded) and one that is not (opaque)
    *(f"Kafka doesn{c}t store records in partitions" for c in "\u00b4\u2032`\ua78c\u02ee"),
]
_TERMS = [".", ".", "!", "?", "\u2026", "\u0964", "\uff0e", "\uff61", "\u061f", "\u3002", '."']
_CJK = [
    "Kafka \u5c06\u8bb0\u5f55\u5b58\u50a8\u5728\u5206\u533a\u4e2d\u3002",
    "\u5b83\u4f7f\u7528\u7aef\u53e39999\u3002",
    "\u662f\u7684\uff01",
    "\u4e3a\u4ec0\u4e48\uff1f",
    "\u0915\u093e\u092b\u094d\u0915\u093e NASA \u0928\u0947 \u092c\u0928\u093e\u092f\u093e\u0964",
]
_CODE = [
    # S6 prose quotes: whole (recall: "retention.ms" through transparent emphasis) and trimmed
    # at a hidden element or inside a token (never covered)
    "retention.ms",
    "rm -rf /",
    "rm -rf /home",
    "git reset --hard HEAD",
    "git reset --hard HEAD~1",
    "kafka-topics --create --topic logs --partitions 3",
    "kafka-topics --list",
    "rm -rf /data",
    "port=9999",
    "port is 9",
    "port is 9092",
    "runs every 6",
    "a\nb\nc",
    # parser rewrite, round 1: the two-line block of S4, exact (recall) or joined by a character
    # str.splitlines() would split on (never exact)
    "\n".join(_S4_LINES),
    *(sep.join(_S4_LINES) for sep in _LINE_BREAKS),
    # blind review round 2 (P2): S5's block exact, uniformly dedented by a tab (recall), and
    # re-indented with mixed tabs and spaces so cleanup() moves under the `if` (never exact)
    _S5_CODE,
    "\n".join(f"\t{ln}" for ln in _S5_CODE.split("\n")),
    _S5_MIXED,
]
_FENCES = [("```", "```"), ("~~~", "~~~"), ("```python", "```"), ("```sh", "```"), ("~~~", "```")]
_EXTRAS = [
    "Your notes don't cover this.",
    "No, it doesn't cover that.",
    "Your notes don't cover this; no.",
    "Your notes don't cover this, but it is 7 days.",
    "Your notes don't cover this \u281d\u2801\u280e\u2801.",
    "Let me know if you have other questions.",
    "I hope this helps!",
    "Hope that helps.",
    "\U0001f44d",
    "\U0001f389 Great!",
    "3.",
    "- 3",
    "...",
    "---",
    "- [S1]",
    "> [S2]",
    "[S1]",
    *(f"{text}." for text in LOOKALIKES),
    *LOOKALIKES,
    "\u2705",
    "Your notes don't cover this.<p yes it does, 1850>",
    "Your notes don't cover this.<br>",
    "<p NASA built Kafka in 1850>",
]
_TAGS = ["<br>", "<br/>", "<br />", "</p><p>", "<p>"]
# round 3 (P1): tags whose attributes carry letters and digits; their text must be residue
_TAGS += [
    "<p NASA built Kafka in 1850>",
    "<br title='port 9999'>",
    '<div class="NASA">',
    "</p 1850>",
    "<BR CLASS=nasa-1850>",
    "<li value=1850>",
]
_LABELS = ["[S1]", "[S2]", "[S3]", "[S6]", "[S1, S2]", "[1]", "[S9]", "(S2)"]
# round 4: code the shown CLI page (S3) holds whole, for recall, and code it does not
_S3_CODES = [
    "kafka-topics --create --topic logs --partitions 3",
    "kafka-topics --list",
    "\n".join(_S4_LINES),
]


def _contained(text: str, container: str) -> str:
    """`text` inside a blockquote or a list item (every line prefixed or indented)."""
    lines = text.split("\n")
    if container == "> ":
        return "\n".join(f"> {ln}" if ln else ">" for ln in lines)
    pad = " " * len(container)
    return "\n".join([f"{container}{lines[0]}", *(f"{pad}{ln}" if ln else "" for ln in lines[1:])])


@st.composite
def _part(draw: st.DrawFn) -> str:
    fact = draw(st.sampled_from(_FACTS))
    other = draw(st.sampled_from(_FACTS))
    label = draw(st.sampled_from(_LABELS))
    term = draw(st.sampled_from(_TERMS))
    kind = draw(st.integers(min_value=0, max_value=20))
    if kind == 0:
        return f"{fact} {label}{term}"
    if kind == 1:  # trailing text after the last label of a sentence
        other = draw(st.sampled_from([*_FACTS, *LOOKALIKES]))
        sep = draw(
            st.sampled_from(["; ", ";", " \u2014 ", ", ", ". ", " ", ".", "!", "\u2026", "\u3002"])
        )
        return f"{fact} {label}{sep}{other.lower()}{term}"
    if kind == 2:  # a bare label after the sentence: same line, next line, or an orphan
        gap = draw(
            st.sampled_from(
                [
                    *(" ", "", "\n", "\n\n", " \U0001f680 ", "\n- ", "\n\n---\n\n", "\n> "),
                    # blind review round 2 (P2): HTML or an enumerator right before the label
                    *("<!-- NASA built Kafka in 1850 --> ", " <span title='NASA 1850'> "),
                    *(" <img src=x alt='NASA built Kafka'>", " 3) ", " (7) "),
                ]
            )
        )
        return f"{fact}{term}{gap}{label}"
    if kind == 3:  # uncited (attribution may or may not cover it)
        return f"{fact}{term}"
    if kind == 4:  # a list item, an enumerator, a quote
        marker = draw(st.sampled_from(["- ", "1. ", "2. ", "(7) ", "42. ", "3) ", "> ", "b) "]))
        return f"{marker}{fact}{draw(st.sampled_from(['', ' ' + label]))}{term}"
    if kind == 5:
        return draw(st.sampled_from(_CJK))
    if kind == 6:
        return f"{fact} {label}\u3002{draw(st.sampled_from(_CJK))}"
    if kind == 7:  # fenced code, optionally cited by a label line (or an orphan one)
        code = draw(st.sampled_from(_CODE))
        opened, closed = draw(st.sampled_from(_FENCES))
        cite = draw(st.sampled_from(["", f"\n{label}", f"\n\n{label}", f"\n- {label}"]))
        return f"{opened}\n{code}\n{closed}{cite}"
    if kind == 8:  # indented code
        return f"\n    {draw(st.sampled_from(_CODE)).replace(chr(10), chr(10) + '    ')}\n"
    if kind == 9:  # inline code in a cited sentence, or after it
        code = draw(st.sampled_from(_CODE)).replace("\n", " ")
        other = draw(st.sampled_from(_CODE)).replace("\n", " ")
        glue = draw(st.sampled_from(["", "**", "\u200b", "&#8203;", " ", "\\\n"]))
        return draw(
            st.sampled_from(
                [
                    f"Run `{code}` {label}.",
                    f"{fact} {label}. Run `{code}`.",
                    # review round 4: two spans that may touch and read as one command
                    f"Run `{code}`{glue}`{other}` {label}.",
                ]
            )
        )
    if kind == 11:  # two clauses before one label: a semicolon ends the first, a comma does not
        glue = draw(st.sampled_from(["; ", ";", "\uff1b", ", ", " \u2014 ", " | ", " \\| "]))
        return f"{fact}{glue}{other.lower()} {label}{term}"
    if kind == 12:  # a table row, optionally under a header and delimiter row
        cell = st.sampled_from(
            [fact, f"{fact} {label}", label, other, f"`{fact} | {other}` {label}"]
        )
        cells = draw(st.lists(cell, min_size=1, max_size=3))
        header = draw(st.sampled_from(["", "| Claim | Source |\n|---|---|\n", "|---|---|\n"]))
        return header + "| " + " | ".join(cells) + " |"
    if kind == 13:  # an HTML tag between two sentences, one label after it, or after the label
        tag = draw(st.sampled_from(_TAGS))
        return draw(
            st.sampled_from(
                [
                    f"{fact}{tag}{other} {label}{term}",
                    f"{fact}{tag}{other} {label}",
                    f"{fact} {label}{term}{tag}",
                    f"{fact} {label}{tag}",
                ]
            )
        )
    if kind == 10:  # a label the reader cannot see as one (code, comment, link, strike, URL...)
        hidden = [
            f"{fact} `{label}`{term}",
            f"{fact} <!-- {label} -->{term}",
            f"{fact} <!-- {label}{term}",
            f"{fact} [docs](https://x.example/{label}){term}",
            f'{fact} [docs](https://x.example "{label}"){term}',
            f"{fact} <https://x.example/{label}>{term}",
            f"{fact} https://x.example/{label}{term}",
            f"{fact} ~~{label}~~{term}",
            f"{fact} ~{label}~{term}",
            f"{fact} ~~~{label}~~~{term}",
            f"{fact} http://x.com/{label} {term}",
            f"{fact} https://x.example/a{label}b{term}",
            f"{fact} [see {label}](https://x.example){term}",
            f"{fact} \\{label[:-1]}\\]{term}",
            f"{fact} &#91;S1&#93;{term}",
            f"{fact} <span title='{label}'>{term}",
            f"[{label[1:-1]}]: https://x.example\n\n{fact} {label}{term}",
        ]
        return draw(st.sampled_from(hidden))
    if kind == 14:  # code or a cited sentence nested in a blockquote or a list item
        code = draw(st.sampled_from([*_CODE, *_S3_CODES]))
        inner = draw(
            st.sampled_from(
                [
                    f"{fact} {label}{term}",
                    f"```\n{code}\n```",
                    f"```\n{code}\n```\n{label}",
                    f"{fact} {label}{term}\n\n```\n{code}\n```",
                    f"{fact} {label}{term}\n\n    {code}",
                    f"    {code} {label}",
                ]
            )
        )
        return _contained(inner, draw(st.sampled_from(["> ", "- ", "1. ", "3. "])))
    if kind == 15:  # inline spans: multi-backtick, split by ". ", ";" or a line break
        code = draw(st.sampled_from([*_CODE, *_S3_CODES])).replace("\n", " ")
        return draw(
            st.sampled_from(
                [
                    f"Run ``{code}`` {label}.",
                    f"Run `` `{code}` `` {label}.",
                    f"Run `{code}. {other}` {label}.",
                    f"Run `{code};{other}` {label}.",
                    f"Run `{code}\n{other}` {label}.",
                    f"{fact} ``a `b` {other}`` {label}{term}",
                ]
            )
        )
    if kind == 17:  # parser rewrite, round 3: a link (titled or not) with a cut in its text
        cut = draw(st.sampled_from(["", "! ", "\u2026 ", "; ", ". "]))
        title = draw(st.sampled_from(["", f' "{other}"', f" '{other} {label}'"]))
        return draw(
            st.sampled_from(
                [
                    f"[{cut}{fact}](https://x.example{title}){term}",
                    f"{fact} [{cut}{other}](https://x.example{title}) {label}{term}",
                    f"[{fact}{term} {other}](https://x.example{title})",
                ]
            )
        )
    if kind == 18:  # Unicode whitespace that str.strip removes, at a part's edges
        space = draw(st.sampled_from(["\u00a0", "\u202f", "\u2003", "\u3000"]))
        head = draw(st.sampled_from(["", "## ", "| ", "- ", "1. "]))
        body = draw(st.sampled_from([f"{fact} {label}{term}", f"{fact}{term}"]))
        return f"{head}{space}{body}{space}" + (" |\n|---|" if head == "| " else "")
    if kind == 19:  # list items with nothing but a numeral, a label or a pictograph in them
        empty = draw(st.sampled_from(["", " [S1]", " \U0001f44d", " [S9]"]))
        marker = draw(st.sampled_from([".", ")"]))
        count = draw(st.integers(min_value=1, max_value=3))
        items = "\n".join(f"{n}{marker}{empty}" for n in range(2, 2 + count))
        return draw(st.sampled_from([f"1{marker} {fact} {label}{term}\n{items}", items]))
    if kind == 16:  # verbatim code of the shown page, labelled or not (recall)
        code = draw(st.sampled_from(_S3_CODES))
        return draw(
            st.sampled_from(
                [
                    f"```bash\n{code}\n```\n[S3]",
                    f"```\n{code}\n```",
                    f"List topics with `{code}` [S3].",
                    f"    {code}\n",
                    f"\n    {code} [S3]\n",
                ]
            )
        )
    return draw(st.sampled_from(_EXTRAS))


_SEPARATORS = st.sampled_from(["", " ", "\u200b", "\u00a0", "\u3000", "\u202f", "\n", "\n\n"])


_NO_SEP = [""] * 6


@settings(max_examples=1500, deadline=None)
@given(
    parts=st.lists(_part(), min_size=1, max_size=6),
    separators=st.lists(_SEPARATORS, min_size=6, max_size=6),
)
# the shapes of review round 1, always run (hypothesis also finds them at random, in stress runs)
@example(parts=["(7) Kafka stores records in partitions [S1]."], separators=_NO_SEP)
@example(
    parts=["Kafka was invented by NASA in 1850.", "Kafka has 3 partitions [S1]."],
    separators=_NO_SEP,
)
@example(
    parts=["Kafka was invented by NASA in 1850\u0964", "Postgres uses MVCC [S2]."],
    separators=_NO_SEP,
)
@example(parts=["Kafka was invented by NASA in 1850.", "\n\n---\n\n[S1]"], separators=_NO_SEP)
@example(parts=["Kafka was invented by NASA in 1850 `[S1]`."], separators=_NO_SEP)
# and of review round 2
@example(
    parts=["| Kafka was invented by NASA in 1850 | Kafka stores records in partitions [S1] |"],
    separators=_NO_SEP,
)
@example(
    parts=["Kafka was invented by NASA in 1850<br>Kafka stores records in partitions [S1]."],
    separators=_NO_SEP,
)
@example(
    parts=["nasa built it in eighteen fifty.kafka stores records in partitions [S1]."],
    separators=_NO_SEP,
)
@example(
    parts=["Kafka stores records in partitions [S1].", "\u281d\u2801\u280e\u2801."],
    separators=_NO_SEP,
)
@example(
    parts=[
        "Kafka stores records in partitions [S1].",
        "| Kafka stores records in partitions |",
        "Your notes don't cover this.",
    ],
    separators=_NO_SEP,
)
@example(  # found by the round-2 stress run: a real bypass (a chain before the dot exempted it)
    parts=[
        "Kafka stores records in partitions [S1]; kafka runs on node.js.",
        "it runs on port 9999 [S1].",
    ],
    separators=_NO_SEP,
)
@example(  # found by the round-2 stress run: an oracle gap (an allowlisted tail), not a bypass
    parts=[
        "Kafka stores records in partitions<br>Kafka stores records in partitions [S1]",
        "Your notes don't cover this.",
    ],
    separators=_NO_SEP,
)
# and of review round 3 (P1): attribute text after a cited sentence, and a refusal hiding one
@example(
    parts=["Kafka stores records in partitions [S1].<p NASA built Kafka in 1850>"],
    separators=_NO_SEP,
)
@example(parts=["Your notes don't cover this.<p yes it does, 1850>"], separators=_NO_SEP)
@example(  # found by the round-4 stress run: a list numeral rode on a verbatim fence
    parts=["3. ```\n   kafka-topics --create --topic logs --partitions 3\n   ```"],
    separators=_NO_SEP,
)
@example(  # found by the round-4 stress run: a sentence end hidden in a linkified URL
    parts=["NASA built Kafka in 1850 https://x.example/a\u2026 Kafka stores records [S1]."],
    separators=_NO_SEP,
)
@example(parts=["Run `rm -rf /. NASA built Kafka in 1850` [S1]."], separators=_NO_SEP)
@example(parts=["> ```\n> rm -rf /data\n> ```\n> [S1]"], separators=_NO_SEP)
@example(parts=["NASA built Kafka in 1850 ~~[S1]~~."], separators=_NO_SEP)
@example(  # parser rewrite, round 1: code joined by a Unicode line separator
    parts=["```sh\ncd /tmp/build\u2028rm -rf cache\n```\n[S4]"], separators=_NO_SEP
)
@example(parts=["Kafka stores records in partitions http://x.com/[S1] ."], separators=_NO_SEP)
@example(
    parts=["The cluster has ~3 brokers [S1]. Autovacuum runs every ~60 seconds [S2]."],
    separators=_NO_SEP,
)
# and of blind review round 2 (P2): mixed-indent code, a label after HTML or an enumerator
@example(parts=[f"```python\n{_S5_MIXED}\n```\n[S5]"], separators=_NO_SEP)
@example(
    parts=["Kafka stores records in partitions.<!-- NASA built Kafka in 1850 --> [S1]"],
    separators=_NO_SEP,
)
@example(parts=[_TAIL, " 3) [S1]"], separators=_NO_SEP)
# and of blind review round 3: a cut inside a titled link's text; an empty item's numeral
@example(
    parts=['[! Kafka stores records in partitions](https://x.example "NASA built Kafka")'],
    separators=_NO_SEP,
)
@example(parts=[f"1. {_TAIL}\n\nHere is what I found:\n\n2."], separators=_NO_SEP)
@example(parts=[_TAIL, "\n\n1)\n2)\n3)"], separators=_NO_SEP)
# found by the property after Phase 2c: an empty code box between glued fences states nothing
@example(
    parts=[
        "```\nkafka-topics --create --topic logs --partitions 3\n```",
        "```\nkafka-topics --list\n```",
        "```\nkafka-topics --list\n```",
    ],
    separators=["", " ", "", "", "", ""],
)
def test_verified_implies_the_residue_is_empty(parts: list[str], separators: list[str]) -> None:
    """For ANY generated answer, with a verifier that ALWAYS says supported: `verified` implies
    the oracle's residue is empty, and every kept inferred unit is a single sentence or a code
    block (so an implementation cannot "infer" its way over a boundary)."""
    answer = "".join(p + s for p, s in zip(parts, separators, strict=False)).strip()
    draft = normalize_citations(answer)
    report = check_citations(draft, SHOWN, strict=False)
    if not report.ok:
        return
    found = attribute(draft, HITS, SHOWN, VALID)
    for text in residue(draft, VALID, ()):
        assert text in draft  # highlights are always verbatim
    if found.refusal_only:
        assert _oracle_residue(draft, []) == [], draft  # wholly allowlisted, per the oracle
        return
    if no_provenance(found, tools_ran=False):
        return  # rule 2: unsupported
    out = settle(
        draft,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=VALID,
        attribution=found,
        tools_ran=False,
    )
    if out.status is not Verification.verified:
        return
    kept = [i.sentence for i in out.inferred]
    for text in kept:  # one sentence or one code block, read by the oracle's own parse
        if any(t.type in ("fence", "code_block") for t in _ORACLE_MD.parse(text)):
            continue
        for token in _ORACLE_MD.parseInline(text):
            if token.type == "inline":
                seen = _visible(token.children or [])
                assert not any(
                    all(seen.label_ok[m.start() : m.end()])
                    for m in _ORACLE_LABEL.finditer(seen.text)
                ), text
                strip = "".join([" \t|", *_ORACLE_TERMS, _ORACLE_CLOSERS])
                core = seen.text
                # trailing terminators, closers and symbols ("partitions.\u2122") state nothing
                while core and (core[-1] in strip or unicodedata.category(core[-1])[0] == "S"):
                    core = core[:-1]
                assert all(c == 0 or c >= len(core) for c in _url_cuts(seen.text)), text
    assert _oracle_residue(draft, kept) == [], draft


@pytest.mark.parametrize("linkify", LINKIFY_CONFIGS)
def test_verified_implies_the_residue_is_empty_in_both_parser_configs(linkify: bool) -> None:
    """The same property (with its fixed examples) with linkify explicitly on and off: the
    prod and CI both install linkify-it-py (agent extra) since the integrator's packaging fix."""
    with parser_config(linkify=linkify):
        test_verified_implies_the_residue_is_empty()


# ---------------------------------------------------------------- bidi controls (round 3, P1)
_RLO, _PDF, _RLI, _PDI = "\u202e", "\u202c", "\u2067", "\u2069"


@pytest.mark.parametrize(
    "answer",
    [
        f"Kafka does {_RLO}ton{_PDF} store records in partitions.",  # displays "does not"
        f"Kafka {_RLO}t'nseod{_PDF} store records in partitions.",  # displays "doesn't"
        f"The default port is {_RLO}9092{_PDF}.",  # displays 2909
        f"The default port is {_RLO}9092{_PDF} [S1].",  # even under a model label
        f"Kafka stores records in {_RLI}partitions{_PDI} [S1].",  # isolates too
    ],
)
def test_bidi_controls_never_reach_verified(answer: str) -> None:
    """A bidi override changes what the reader SEES, so text holding one is never covered:
    stripping it would check a different string than the one displayed."""
    status, _ = status_of(answer)
    assert status is not Verification.verified


@pytest.mark.parametrize(
    "answer",
    [
        # a control alone in a unit that states nothing still reverses the rest of the paragraph
        "Kafka stores records in partitions [S1]. \u202e. "
        "snoititrap ni sdrocer erots ton seod akfaK [S1].",
        "Kafka stores records in partitions. \u202e! The default port is 9092.",
        "Kafka stores records in partitions [S1]. \u200f\u200f.",
        "Kafka stores records in partitions [S1].\n\n- \u202e",
    ],
)
def test_any_bidi_control_anywhere_blocks_verified(answer: str) -> None:
    status, _ = status_of(answer)
    assert status is not Verification.verified


# ---------------------------------------------------------------- Phase 2b review, P2
@pytest.mark.parametrize(
    "answer",
    [
        "The default port is &#x202E;9092&#x202C; [S1].",
        "The default port is 9092 [S1]. &#x202E;Kafka stores records in partitions [S1].",
        "Kafka stores records in &rlm;partitions [S1].",
        "Kafka stores records in &#8238;partitions [S1].",
    ],
)
def test_bidi_controls_written_as_entities_never_reach_verified(answer: str) -> None:
    """The checker reads entities as characters; a surface that does not escape "&" would show
    the reversal, so a referenced bidi control counts exactly like a raw one."""
    status, _ = status_of(answer)
    assert status is not Verification.verified


@pytest.mark.parametrize("dot", [chr(c) for c in (0xA4F8, 0x00B7, 0x2027, 0x30FB, 0x22C5)])
def test_a_dot_lookalike_ends_the_sentence_a_label_covers(dot: str) -> None:
    """ "1850<U+A4F8> Kafka ..." reads as two sentences: S1 covers only the second."""
    answer = f"NASA built Kafka in 1850{dot} Kafka stores records in partitions [S1]."
    status, claims = status_of(answer)
    assert status is not Verification.verified
    assert any("1850" in c for c in claims)


def test_plain_text_without_entities_or_lookalikes_still_verifies() -> None:
    assert status_of("Kafka stores records in partitions [S1].")[0] is Verification.verified


# Review round 2, P2: the look-alike list was closed and missed these; now a rule.
_ROUND2_DOTS = (0x2E33, 0x2981, 0x2022, 0x02D9, 0x0323, 0x0971, 0x10FB, 0x0E4F, 0x05C3, 0x2E30)


@pytest.mark.parametrize("code", _ROUND2_DOTS)
def test_any_dot_like_character_before_a_space_ends_the_sentence(code: int) -> None:
    answer = f"NASA built Kafka in 1850{chr(code)} Kafka stores records in partitions [S1]."
    status, claims = status_of(answer)
    assert status is not Verification.verified
    assert any("1850" in c for c in claims) or claims == (answer,)


@pytest.mark.parametrize(
    "answer",
    [
        "Kafka stores records in partitions [S1] → consumers read them [S1].",  # arrow
        f"Kafka stores records in partitions [S1]. It has 3 {chr(0xD7)} replicas [S1].",
        "Kafka stores records in partitions [S1].",
    ],
)
def test_arrows_and_operators_do_not_cut(answer: str) -> None:
    units = [u for u in segment(answer) if u.substantive]
    assert all(u.labels for u in units), [u.text for u in units]


@pytest.mark.parametrize(
    "answer",
    [
        "The default ports are א 9092 9093 [S1].",
        "Kafka stores records in partitions [S1]. ب 7 30.",
    ],
)
def test_right_to_left_letters_next_to_digits_make_the_answer_residue(answer: str) -> None:
    """Review round 2, P2: the bidi algorithm may show those digits in another order."""
    assert has_rtl(answer)
    assert residue(answer, VALID, ()) == [answer]
    assert status_of(answer)[0] is not Verification.verified


def test_right_to_left_letters_without_digits_are_residue_too() -> None:
    """Review round 3, P2: first-strong direction swaps whole phrases, digits or not."""
    answer = "\u05d0 Postgres \u05d0 is faster than \u05d0 MySQL \u05d0 for analytical joins.\n[S1]"
    assert has_rtl(answer)
    assert residue(answer, VALID, ()) == [answer]
    assert not has_rtl("Kafka stores records in partitions [S1].")


@pytest.mark.parametrize(
    "between",
    ["\u200b", "\u2060", "\u00ad", "**", "_", "~~", "`", "\u200b**"],
)
def test_a_dot_like_end_is_found_through_invisible_and_emphasis_characters(between: str) -> None:
    """Review round 3, P2: a zero-width character or a closing emphasis between the dot and
    the space hid the end, so S1 covered the 1850 claim."""
    opener = between.replace("\u200b", "").replace("\u2060", "").replace("\u00ad", "")
    answer = (
        f"NASA built Kafka in {opener}1850\u02d9{between} Kafka stores records in partitions [S1]."
    )
    status, _ = status_of(answer)
    assert status is not Verification.verified, answer


@pytest.mark.parametrize(
    ("answer", "verified"),
    [
        ("Set `retention.ms` to 0 for compacted topics [S6].", False),  # prose only: not code
        ("Undo with `git reset --hard HEAD~1` first [S6].", False),  # multi-token (round 7)
        ("Clean with `rm -rf /` nightly [S6].", False),  # ends where code/link starts
        ("Clean with `rm -rf /home` nightly [S6].", False),  # struck on the page
        ("Undo with `git reset --hard HEAD` first [S6].", False),  # trims "HEAD~1"
    ],
)
def test_prose_quotes_against_the_hidden_element_page(answer: str, verified: bool) -> None:
    status, _ = status_of(answer, shown=(1, 2, 3, 4, 5, 6))
    assert (status is Verification.verified) is verified, answer


# ---------------------------------------------------------------- review round 4
_RM_PAGE = "Never run rm -rf / on a host. Clean var/cache daily on the host box."


@pytest.mark.parametrize(
    "glue",
    ["", "**", "​", "⁠", "&#8203;", "&zwnj;", "*", "\\\n"],
)
def test_adjacent_answer_code_spans_never_join_into_backed_code(glue: str) -> None:
    """Review round 4, P1: `rm -rf /` and `var/cache` are each quotable from the page, but
    touching (or joined by an invisible character, emphasis or a hard break) they read as one
    command the page never states."""
    from module_d1.d1_fakes import make_hit
    from ragforge_core.agent.state import hit_to_state

    hits = [hit_to_state(make_hit(1, title="Ops", text=_RM_PAGE))]
    valid = valid_labels([1], 0)
    if glue in ("**", "*"):
        joined = f"`rm -rf /`{glue}`var/cache`{glue}"
    else:
        joined = f"`rm -rf /`{glue}`var/cache`"
    for answer in (
        f"Clean with {joined} daily on the host box [S1].",
        f"Clean with {joined} daily on the host box.",
    ):
        found = attribute(answer, hits, [1], valid)
        out = settle(
            answer,
            verdict=Verification.verified,
            conclusive=True,
            located=True,
            verifier_claims=(),
            valid=valid,
            attribution=found,
            tools_ran=False,
        )
        assert out.status is not Verification.verified, answer


def test_separated_answer_code_spans_are_still_backed() -> None:
    from module_d1.d1_fakes import make_hit
    from ragforge_core.agent.state import hit_to_state

    page = "Clean `var/cache` daily on the `host` box."  # both as code on the page
    hits = [hit_to_state(make_hit(1, title="Ops", text=page))]
    valid = valid_labels([1], 0)
    answer = "Clean `var/cache` daily on the `host` box [S1]."
    found = attribute(answer, hits, [1], valid)
    assert found.code_unbacked == ()


@pytest.mark.parametrize("ref", ["&#x5D0;", "&#1488;", "&#1575;", "&#x10900;"])
def test_rtl_letters_written_as_entities_make_the_answer_residue(ref: str) -> None:
    answer = f"The ports are {ref} 9092 9093 here [S1]."
    assert has_rtl(answer)
    assert status_of(answer)[0] is not Verification.verified


@pytest.mark.parametrize("mark", [0x2757, 0x2753, 0x2755, 0x2762, 0x2234, 0x22EF])
def test_emoji_and_symbol_sentence_marks_end_the_sentence(mark: int) -> None:
    answer = f"NASA built Kafka in 1850{chr(mark)} Kafka stores records in partitions [S1]."
    assert status_of(answer)[0] is not Verification.verified


# ---------------------------------------------------------------- review round 5
@pytest.mark.parametrize(
    ("page", "quoted", "covered"),
    [
        ("Delete with rm /tmp/*.log /var/*.bak weekly.", "rm /tmp/.log /var/.bak", False),
        ("Keep /srv/_data_ intact; wipe /srv/_tmp_ instead.", "/srv/data", False),
        ("Keep /srv/_data_ intact; wipe /srv/_tmp_ instead.", "/srv/tmp", False),
        (
            "find . -name __pycache__ -exec rm -rf {} +",
            "find . -name pycache -exec rm -rf {} +",
            False,
        ),
        ("Set **retention.ms** to 0 for compacted topics.", "retention.ms", False),
        ("Set *retention.ms* to 0.", "retention.ms", False),
    ],
)
def test_a_prose_quote_never_contains_an_emphasis_delimiter(
    page: str, quoted: str, covered: bool
) -> None:
    """Review round 5, P1: Notion text is not Markdown-escaped, so literal "*"/"_" parse as
    emphasis; dropping the delimiters let a quote state a path or glob the page never does."""
    from ragforge_core.agent.attribution import _source, code_matches
    from ragforge_core.agent.structure import structure as parse

    (code,) = parse(f"Use `{quoted}` here.").code
    hit: dict[str, object] = {
        "title": "Ops",
        "heading_path": [],
        "display_text": page,
        "external_url": None,
    }
    assert code_matches(code, _source(1, hit)) is covered  # type: ignore[arg-type]


@pytest.mark.parametrize("mark", [0x01C3, 0x0294, 0x0660, 0x06F0, 0x0ED0, 0x0E2F])
def test_a_letter_or_digit_posing_as_a_terminator_ends_the_sentence(mark: int) -> None:
    """Review round 5, P2: confusables of "!", "?" and "." that are letters or digits."""
    answer = f"NASA built Kafka in 1850{chr(mark)} Kafka stores records in partitions [S1]."
    assert status_of(answer)[0] is not Verification.verified


def test_accented_words_are_not_cut() -> None:
    answer = "Kafka stores records in partitions near the caf\u00e9 [S1]."
    units = [u for u in segment(answer) if u.substantive]
    assert all(u.labels for u in units)


def test_arabic_indic_digits_count_as_right_to_left() -> None:
    answer = "Kafka uses \u0661\u0660 \u0662\u0660 brokers [S1]."
    assert has_rtl(answer)
    assert status_of(answer)[0] is not Verification.verified


# ---------------------------------------------------------------- review round 6
@pytest.mark.parametrize(
    "glue", ["\x1c", "\x0b", "\x85", "\u2028", "<!---->", "\\\n", "**", "\u200b"]
)
def test_a_clean_twin_never_launders_joined_code(glue: str) -> None:
    """Review round 6, P1: `residue` keys survivors by text, so a joined unit whose text also
    occurs once in clean (inferred, kept) form dropped out of the residue."""
    from module_d1.d1_fakes import make_hit
    from ragforge_core.agent.state import hit_to_state

    hits = [hit_to_state(make_hit(1, title="Ops", text=_RM_PAGE))]
    valid = valid_labels([1], 0)
    tail = "**" if glue == "**" else ""
    answer = (
        f"Clean the host box daily with `rm -rf /`{glue}`var/cache`{tail} daily on the host box."
        "\n\nClean the host box daily with `rm -rf /`\n\n`var/cache` daily on the host box."
    )
    draft = normalize_citations(answer)
    found = attribute(draft, hits, [1], valid)
    out = settle(
        draft,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=valid,
        attribution=found,
        tools_ran=False,
    )
    assert out.status is not Verification.verified, glue


def _scan_chars() -> list[str]:
    """Every assigned non-ASCII code point below U+30000 that is not a cased letter (the
    general rule's domain), sampled every 7th to keep the scan fast."""
    out = []
    for cp in range(0x80, 0x30000, 7):
        c = chr(cp)
        cat = unicodedata.category(c)
        if cat in ("Ll", "Lu", "Lt", "Mn", "Me", "Zs", "Zl", "Zp") or cat[0] == "C" or c.isspace():
            continue
        if c == "\ufffc":  # the parser's own placeholder for a URL or an image
            continue
        out.append(c)
    return out


def test_every_non_cased_character_before_a_space_after_ascii_cuts() -> None:
    """Review round 6, P2: test the rule over Unicode data, not a hand list. Stuck to an ASCII
    token every such character cuts; free-standing between spaces only letters and digits do
    (a free-standing symbol, "3 \u00d7 4", an arrow or an emoji, is not a terminator)."""
    from ragforge_core.agent.coverage import sentence_ends

    misses = []
    for c in _scan_chars():
        texts = [f"NASA built Kafka in 1850{c} Kafka stores", f"in (1850){c} Kafka"]
        if unicodedata.category(c) in ("Lo", "Lm", "Nd", "No"):
            texts.append(f"in 1850 {c} Kafka")
        for text in texts:
            if text.rindex(" Kafka") not in set(sentence_ends(text, 0, len(text))):
                misses.append((hex(ord(c)), text[:14]))
                break
    assert misses == [], misses[:10]


def test_cjk_and_thai_words_are_not_cut_inside() -> None:
    from ragforge_core.agent.coverage import sentence_ends

    # Multi-character words keep their characters. A one-character non-Latin word between spaces
    # IS cut (fail-closed trade-off for an English corpus: it only adds residue).
    for text in ("\u6570\u636e \u5e93\u5b58 x", "\u0e01\u0e32 \u0e23\u0e30 x"):
        assert [e for e in sentence_ends(text, 0, len(text)) if e < len(text)] == [], text


# ---------------------------------------------------------------- review round 8
_TWO_SENTENCE_PAGE = "Kafka stores records in partitions. Kafka 1.0 shipped in 2017."


@pytest.mark.parametrize(
    "answer",
    [
        "NASA built Kafka in `2017.` Kafka stores records in partitions [S1].",
        "Redis evicts keys under `noeviction;` Kafka stores records in partitions [S1].",
        "NASA built Kafka in `2017!` Kafka stores records in partitions [S1].",
    ],
)
def test_a_terminator_inside_a_code_span_ends_the_sentence(answer: str) -> None:
    """Review round 8, P1: the span hid the sentence end, so one label covered two sentences."""
    from module_d1.d1_fakes import make_hit
    from ragforge_core.agent.state import hit_to_state

    hits = [hit_to_state(make_hit(1, title="Kafka", text=_TWO_SENTENCE_PAGE))]
    valid = valid_labels([1], 0)
    draft = normalize_citations(answer)
    found = attribute(draft, hits, [1], valid)
    out = settle(
        draft,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=valid,
        attribution=found,
        tools_ran=False,
    )
    assert out.status is not Verification.verified, answer
    assert len([u for u in segment(draft) if u.substantive]) >= 2


@pytest.mark.parametrize(
    ("page", "quoted"),
    [
        ("Call __init__ first.", "init"),
        ("Read __name__ here.", "name"),
        ("Export __all__ there.", "all"),
        ("Use _prod_ only.", "prod"),
    ],
)
def test_an_underscore_delimiter_never_trims_an_identifier(page: str, quoted: str) -> None:
    """Review round 8, P2: Notion's literal "__init__" parsed as strong "init"."""
    from ragforge_core.agent.attribution import _source, code_matches
    from ragforge_core.agent.structure import structure as parse

    (code,) = parse(f"Use `{quoted}` here.").code
    hit: dict[str, object] = {
        "title": "T",
        "heading_path": [],
        "display_text": page,
        "external_url": None,
    }
    assert not code_matches(code, _source(1, hit))  # type: ignore[arg-type]


@pytest.mark.parametrize("space", ["\u2006", "\u2009", "\u200a", "\u202f"])
def test_thin_spaces_do_not_separate_two_code_spans(space: str) -> None:
    from ragforge_core.agent.attribution import joined_code

    assert joined_code(f"Set `noeviction`{space}`as` default.") != frozenset()


# ---------------------------------------------------------------- review round 9
@pytest.mark.parametrize("span", ["1850.)", "1850.'", 'noeviction."', "2017.**", "2017.\u200b"])
def test_a_terminator_before_closers_in_a_code_span_still_ends_the_sentence(span: str) -> None:
    answer = f"NASA built Kafka in `{span}` Kafka stores records in partitions [S1]."
    units = [u for u in segment(normalize_citations(answer)) if u.substantive]
    assert len(units) >= 2, [u.text for u in units]
    assert status_of(answer)[0] is not Verification.verified


@pytest.mark.parametrize("blank", ["\u2800", "\u3164", "\u115f", "\uffa0"])
def test_a_blank_glyph_anywhere_makes_the_answer_residue(blank: str) -> None:
    """Review round 9, P2: a look-alike terminator before a blank glyph hid the end."""
    answer = f"NASA built Kafka in 1850\u2e30{blank}Kafka stores records in partitions [S1]."
    assert residue(answer, VALID, ()) == [answer]
    assert status_of(answer)[0] is not Verification.verified


# ---------------------------------------------------------------- final review
@pytest.mark.parametrize(
    "answer",
    [
        "NASA built Kafka in 1850&#x2E30;&#x2800;Kafka stores records in partitions [S1].",
        "NASA built Kafka in 1850⸰&#x2800;Kafka stores records in partitions [S1].",
        "NASA built Kafka in 1850⸰&#x3164;Kafka stores records in partitions [S1].",
        "NASA built Kafka in 1850⸰&#x115F;Kafka stores records in partitions [S1].",
        "NASA built Kafka in 1850⸰&#xFFA0;Kafka stores records in partitions [S1].",
    ],
)
def test_blank_glyphs_written_as_entities_make_the_answer_residue(answer: str) -> None:
    """Final review, P2: the whole-answer checks now also run on the entity-decoded text."""
    assert residue(answer, VALID, ()) == [answer]
    assert status_of(answer)[0] is not Verification.verified
