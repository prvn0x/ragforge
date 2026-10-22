"""DOT (2026-10-19): source-backed dotted names are not sentence ends.

Live false flags: "... POST request to /actuator/loggers/{logger.name} with a JSON body [S1]."
was cut after "logger." and "Setting max.in.flight.requests.per.connection to 1 ... [S2]." after
"per.", and each first fragment was flagged "Not found in your notes" although the notes say it.
The fix (ADR-0015, "Current state") is SOURCE-BACKED: an interior ASCII dot followed by
[A-Za-z0-9_] is in-token only when the maximal [A-Za-z0-9_.] run around it (leading/trailing
dots stripped) is a dotted name found verbatim in the shown sources (`coverage.dotted_names`).

The invariant these tests pin: the exemption only REMOVES the cuts at interior dots of runs in
`names`; it never merges two sentences a reader sees as two (dot then whitespace or the end),
and every other rule (dot runs, homoglyph dots, bidi) stays ahead of it.
"""

from __future__ import annotations

import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_d1.d1_fakes import make_hit
from ragforge_core.agent.attribution import attribute, settle
from ragforge_core.agent.coverage import (
    dotted_names,
    refusal_only,
    residue,
    segment,
    sentence_ends,
    source_sentences,
    valid_labels,
    visible_runs,
)
from ragforge_core.agent.state import HitDict, hit_to_state
from ragforge_core.domain.models import Verification

# ---------------------------------------------------------------- the two live cases
LOGGERS_PAGE = (
    "Spring Boot Actuator can change a log level at runtime: send a POST request to "
    "`/actuator/loggers/{logger.name}` with a JSON body such as "
    '`{"configuredLevel": "DEBUG"}`.'
)
ORDERING_PAGE = (
    "Kafka producer ordering: setting max.in.flight.requests.per.connection to 1 provides the "
    "strictest ordering, at the cost of throughput."
)
HITS: list[HitDict] = [
    hit_to_state(make_hit(1, title="Spring Boot Actuator", text=LOGGERS_PAGE)),
    hit_to_state(make_hit(2, title="Kafka producer", text=ORDERING_PAGE)),
]
SHOWN = [1, 2]
VALID = valid_labels(SHOWN, 0)
NAMES = dotted_names(h["display_text"] for h in HITS)

LOGGERS = (
    "You can accomplish this by making a POST request to /actuator/loggers/{logger.name} with "
    "a JSON body containing the configuredLevel [S1]."
)
ORDERING = "Setting max.in.flight.requests.per.connection to 1 provides strictest ordering [S2]."
LIVE = f"{LOGGERS} {ORDERING}"


def _texts(answer: str, names: frozenset[str] = frozenset()) -> list[str]:
    return [u.text for u in segment(answer, names=names)]


def _cuts(text: str, names: frozenset[str] = frozenset()) -> list[int]:
    return list(sentence_ends(text, 0, len(text), names=names))


def test_dotted_names_of_the_live_pages() -> None:
    assert frozenset({"logger.name", "max.in.flight.requests.per.connection"}) == NAMES


def test_without_names_both_live_sentences_are_still_cut_as_before() -> None:
    """Default empty `names` is today's behaviour exactly: the reported false flag."""
    assert _texts(LIVE) == [
        "You can accomplish this by making a POST request to /actuator/loggers/{logger.",
        "name} with a JSON body containing the configuredLevel [S1].",
        "Setting max.in.flight.requests.per.",
        "connection to 1 provides strictest ordering [S2].",
    ]
    # ...and the first fragment of each is residue: the false "Not found in your notes"
    left = residue(LIVE, VALID, ())
    assert left == [
        "You can accomplish this by making a POST request to /actuator/loggers/{logger.",
        "Setting max.in.flight.requests.per.",
    ]


def test_with_source_names_each_live_sentence_is_one_unit() -> None:
    assert _texts(LIVE, NAMES) == [LOGGERS, ORDERING]
    assert residue(LIVE, VALID, (), names=NAMES) == []


def test_both_live_sentences_are_covered_and_verified_when_cited() -> None:
    found = attribute(LIVE, HITS, SHOWN, VALID, names=NAMES)
    assert found.covered
    assert (found.uncited, found.code_unbacked, found.inferred) == ((), (), ())
    settled = settle(
        LIVE,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=VALID,
        attribution=found,
        tools_ran=False,
        names=NAMES,
    )
    assert settled.status is Verification.verified
    assert (settled.unsupported_claims, settled.residue) == ((), ())


def test_settle_without_the_same_names_still_fails_closed() -> None:
    """A caller that forgets `names` in `settle` re-cuts the answer: residue, never a silent
    pass (the seam's docstrings say every reader of one answer must pass the same set)."""
    found = attribute(LIVE, HITS, SHOWN, VALID, names=NAMES)
    settled = settle(
        LIVE,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=VALID,
        attribution=found,
        tools_ran=False,
    )
    assert settled.status is Verification.partial


def test_an_uncited_live_sentence_is_still_flagged() -> None:
    """Keeping the name whole changes the unit, not the rule: no label, no coverage."""
    bare = ORDERING.replace(" [S2]", "")
    assert residue(bare, VALID, (), names=NAMES) == [bare]


def test_names_from_a_source_cut_that_source_the_same_way() -> None:
    assert source_sentences(ORDERING_PAGE, names=NAMES) == [ORDERING_PAGE]
    assert len(source_sentences(ORDERING_PAGE)) == 2  # the old per-sentence evidence split


# ---------------------------------------------------------------- dotted_names
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("set a.b.c now", {"a.b.c"}),
        ("end of a.b.", {"a.b"}),  # trailing dot stripped
        ("..a.b..", {"a.b"}),  # leading and trailing dots stripped
        (".bashrc and x.", set()),  # no INTERIOR dot left
        ("{logger.name}", {"logger.name"}),
        ("/dev/null", set()),
        ("Kafka.It", {"Kafka.It"}),  # case kept verbatim
        ("node.js 3.5", {"node.js", "3.5"}),
        ("a\u2024b and a\uff0eb", set()),  # homoglyph dots are not ASCII "."
        ("caf\u00e9.menu", set()),  # a non-ASCII letter bounds the run: ".menu" -> "menu"
        ("x_y.z_1", {"x_y.z_1"}),
    ],
)
def test_dotted_names_takes_maximal_ascii_runs_with_an_interior_dot(
    text: str, expected: set[str]
) -> None:
    assert dotted_names([text]) == frozenset(expected)


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        # review round 2, P2: names come from what the reader sees, not from raw Markdown
        ("set max.in.**flight**.requests now", {"max.in.flight.requests"}),
        ("set max\\_poll.ms now", {"max_poll.ms"}),
        ("send `/actuator/loggers/{logger.name}` now", {"logger.name"}),
        ("```\nfoo.bar=1\n```", {"foo.bar"}),  # a code block is shown text
        ("<div>a.b</div>", set()),  # an HTML block is not
        ("a `x.y`.z b", {"x.y"}),  # a code span bounds the run: never "x.y.z"
        ("a.b<br>c.d", {"a.b", "c.d"}),  # inline HTML bounds the run
    ],
)
def test_dotted_names_reads_the_shown_text_of_a_markdown_page(
    page: str, expected: set[str]
) -> None:
    assert dotted_names([page]) == frozenset(expected)


def test_a_name_split_by_emphasis_in_the_page_matches_the_answer() -> None:
    names = dotted_names(["Set max.in.**flight**.requests.per.connection to 1."])
    answer = "Setting max.in.flight.requests.per.connection to 1 gives ordering [S1]."
    assert len(segment(answer, names=names)) == 1


def test_dotted_names_unions_texts() -> None:
    assert dotted_names(["a.b", "c.d e"]) == frozenset({"a.b", "c.d"})
    assert dotted_names([]) == frozenset()


# ---------------------------------------------------------------- merge attacks
# Every one is cut exactly as without `names`, whatever (wrong) dotted names the sources hold.
ATTACK_NAMES = NAMES | frozenset({"fast.and", "and.it", "nasa.in", "null", "a.b", "js.it"})


@pytest.mark.parametrize(
    "text",
    [
        "Kafka is fast.and.it never loses data",
        "NASA built Kafka in nasa.in.fact it was 1850",
        "Write to /dev/null.it is discarded",
        "Kafka (nasa.it was built in 1850",
        "It is done.it.works now",
        "Kafka runs on node.js.it runs on port 9999",
        "Kafka is fast.And it never loses data",
        "see a.b. Then Kafka stores",
        "e.g. Kafka",
        "Kafka 3.5 shipped. Then 3.6",
        "max.in.flight.requests.per.connection. Then Kafka",  # the name, then a real end
        "Setting Max.in.flight.requests.per.connection to 1",  # case differs: not the name
        "Setting max.in.flight.requests.per.connections to 1",  # one char more: not the name
        "Setting producer.max.in.flight.requests.per.connection to 1",  # a longer run
    ],
)
def test_merge_attacks_are_cut_exactly_as_without_names(text: str) -> None:
    assert _cuts(text, ATTACK_NAMES) == _cuts(text)


# A source typo with the space missing ("Redis is fast.It keeps data") reads as two sentences:
# a capital after a dot is never exempt, even when a page holds the run verbatim, whatever the
# run's dot count (review round 1 and round 2, P1: "2.8.It", "U.S.The", "kafka.streams.It").
TYPO_PAGE = "Redis is fast.It keeps data in memory. Use java.util.List for lists."
TYPO_NAMES = dotted_names([TYPO_PAGE]) | frozenset({"fast.And", "data.The", "Kafka.It"})


def test_dotted_names_of_a_typo_page_include_the_typo() -> None:
    """The collection stays verbatim; the narrowing is in the exemption."""
    assert frozenset({"fast.It", "java.util.List"}) <= TYPO_NAMES


@pytest.mark.parametrize(
    "text",
    [
        "Redis loses nothing on crash, fast.It keeps data in memory",
        "Kafka is fast.And it never loses data",
        "Kafka keeps data.The broker never loses it",
        "Use Kafka.It is durable",
        "Use *fast*.It keeps data",  # emphasis is not visible text: the run is still "fast.It"
    ],
)
def test_a_lone_dot_before_a_capital_is_cut_even_when_a_source_holds_the_run(text: str) -> None:
    assert _cuts(text, TYPO_NAMES) == _cuts(text)
    assert len(_texts(text, TYPO_NAMES)) == 2, text


def test_an_uncited_clause_cannot_ride_on_a_cited_one_through_a_source_typo() -> None:
    """The P1 counterexample end to end: the unbacked first clause is residue, not covered."""
    hits: list[HitDict] = [hit_to_state(make_hit(1, title="Redis", text=TYPO_PAGE))]
    names = dotted_names(h["display_text"] for h in hits)
    answer = "Redis loses nothing on crash, fast.It keeps data in memory [S1]."
    valid = valid_labels([1], 0)
    assert residue(answer, valid, (), names=names) == ["Redis loses nothing on crash, fast."]
    found = attribute(answer, hits, [1], valid, names=names)
    # the page does not back it, so nothing infers it: it stays an uncited (flagged) clause
    assert (found.uncited, found.inferred) == (("Redis loses nothing on crash, fast.",), ())


def test_a_capital_after_a_dot_in_a_multi_dot_name_keeps_its_cut() -> None:
    """Round 2, P1: "java.util.List" is cut before "List" with or without names, exactly as it
    was before DOT; the extra cut only costs a false flag, never a merge."""
    text = "Use java.util.List for lists"
    assert _cuts(text) == [text.index("List")]
    assert _cuts(text, TYPO_NAMES) == _cuts(text)


# Round 2, P1 probes: a capital after a dot in a 2+-dot run is a sentence start a reader sees.
CAPITAL_CASES = [
    (
        "Kafka 2.8.It added KRaft mode.",
        "Kafka was built by NASA in 1850, as 2.8.It added KRaft mode [S1].",
        ("Kafka was built by NASA in 1850, as 2.8.",),
    ),
    (
        "Offices in the U.S.The company grew.",
        "Kafka was built by NASA in the U.S.The company grew [S1].",
        ("Kafka was built by NASA in the U.", "S."),  # the old rules already cut "U."
    ),
    (
        "Use kafka.streams.It keeps state.",
        "Kafka never loses data, use kafka.streams.It keeps state [S1].",
        ("Kafka never loses data, use kafka.streams.",),
    ),
]


@pytest.mark.parametrize(("page", "answer", "unbacked"), CAPITAL_CASES)
def test_a_capital_after_any_backed_dot_still_cuts_end_to_end(
    page: str, answer: str, unbacked: tuple[str, ...]
) -> None:
    """Every dotted run of the answer is backed by the page, yet the unbacked first clause is
    still residue and uncited, exactly as without names."""
    hits: list[HitDict] = [hit_to_state(make_hit(1, title="Page", text=page))]
    names = dotted_names(h["display_text"] for h in hits)
    assert dotted_names([answer]) <= names
    assert _cuts(answer, names) == _cuts(answer)
    valid = valid_labels([1], 0)
    assert residue(answer, valid, (), names=names) == list(unbacked)
    found = attribute(answer, hits, [1], valid, names=names)
    assert found.uncited == unbacked
    # even a verifier that says "verified" cannot carry the unbacked clause through
    settled = settle(
        answer,
        verdict=Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=(),
        valid=valid,
        attribution=found,
        tools_ran=False,
        names=names,
    )
    assert settled.status is Verification.partial
    assert settled.residue == unbacked


def test_fast_and_it_keeps_its_sentence_end() -> None:
    """The example rule the design review rejected ("a dotted name with 2+ dots is a token")
    would merge this; the source-backed rule cuts it unless a page holds "fast.and.it"."""
    text = "Kafka is fast.and.it never loses data"
    assert text.index("it never") in _cuts(text, ATTACK_NAMES)


def test_a_dot_followed_by_whitespace_or_the_end_is_never_exempt() -> None:
    names = frozenset({"a.b"})
    for text in ("see a.b. Then x", "see a.b.\nthen x", "see a.b. then x", "see a.b."):
        dot = text.index("b.") + 1
        assert dot + 1 in _cuts(text, names), text


def test_a_name_followed_by_a_space_still_ends_there_only_by_the_old_rules() -> None:
    """ "a.b c" has its dot interior and followed by "b": exempt by the name; a lone "a. b" (dot
    then a space) is a sentence end whatever `names` holds."""
    assert _cuts("x a. b y", frozenset({"a.b", "a"})) == _cuts("x a. b y") == [4]


# ---------------------------------------------------------------- other rules stay ahead
@pytest.mark.parametrize(
    "dot",
    [
        "\u2024",  # one dot leader (a Sentence_Terminal superset member)
        "\uff0e",  # fullwidth full stop
        "\u00b7",  # middle dot (DOT_LOOKALIKES)
        "\u2027",  # hyphenation point
        "\u0701",  # Syriac supralinear full stop
        "..",  # a dot run is a terminator run, never a token dot
        ".\u200b",  # an invisible character after the dot
    ],
)
def test_homoglyph_and_doubled_dots_still_cut(dot: str) -> None:
    text = f"Setting max.in.flight.requests.per{dot}connection to 1"
    at, after = text.index("per") + 3, text.index("connection")
    assert _cuts(text, NAMES) == _cuts(text)
    # the look-alike (or the dot before the invisible character) still ends the sentence
    assert any(at < cut <= after for cut in _cuts(text, NAMES)), text


def test_bidi_anywhere_still_makes_the_whole_answer_residue() -> None:
    answer = LIVE.replace("ordering", "orde\u202ering")
    assert residue(answer, VALID, (), names=NAMES) == [answer]


def test_rtl_text_still_makes_the_whole_answer_residue() -> None:
    answer = LIVE.replace("ordering", "ordering \u05d0")
    assert residue(answer, VALID, (), names=NAMES) == [answer]


# ---------------------------------------------------------------- emphasis and code
def test_parsed_emphasis_around_a_name_reads_as_the_name() -> None:
    """Parsed emphasis delimiters are not visible text: "**max.in.flight.requests.per**.
    connection" renders as the name, partly bold, with no space, so the reader sees one token.
    The run is read in the VISIBLE text, which is what the reader compares with the page."""
    answer = "Setting **max.in.flight.requests.per**.connection to 1 provides ordering [S2]."
    assert _texts(answer) == [
        "Setting **max.in.flight.requests.per**.",
        "connection to 1 provides ordering [S2].",
    ]
    assert _texts(answer, NAMES) == [answer]


def test_literal_asterisks_bound_the_run_so_the_old_cuts_stay() -> None:
    """ "per**.**connection" is NOT emphasis (CommonMark flanking: "**" between a letter and
    punctuation), so the asterisks are visible text and end the run: the name is not matched,
    and the cuts are exactly as without names (an extra cut is the safe side)."""
    answer = "Setting max.in.flight.requests.per**.**connection to 1 provides ordering [S2]."
    assert _texts(answer, NAMES) == _texts(answer)
    assert len(_texts(answer)) == 3


def test_emphasis_never_hides_a_sentence_end_that_is_not_a_name() -> None:
    answer = "Kafka is fast**.**and.it never loses data [S1]."
    assert _texts(answer, NAMES) == _texts(answer)
    assert len(_texts(answer)) == 3
    answer = "Kafka is **fast**.and.it never loses data [S1]."
    assert _texts(answer, ATTACK_NAMES) == _texts(answer)
    answer = "See logger.name**.** Then Kafka stores records [S1]."
    # the name is whole; the dot after it, then a space, is a sentence end
    assert _texts(answer, NAMES) == ["See logger.name**.", "** Then Kafka stores records [S1]."]


@pytest.mark.parametrize(
    "answer",
    [
        # a code span bounds the run, so the rest of the run is not the name: cut as before
        "Setting `max`.in.flight.requests.per.connection to 1 gives ordering [S2].",
        # a code span right after the dot: not followed by an identifier character
        "Setting max.in.flight.requests.per.`connection` to 1 gives ordering [S2].",
    ],
)
def test_code_spans_adjacent_to_the_dot_keep_the_old_cuts(answer: str) -> None:
    assert _texts(answer, NAMES) == _texts(answer)


def test_a_name_inside_a_code_span_is_atomic_either_way() -> None:
    answer = "Set `max.in.flight.requests.per.connection` to 1 for ordering [S2]."
    assert _texts(answer, NAMES) == _texts(answer) == [answer]


# ---------------------------------------------------------------- the other seam functions
def test_visible_runs_and_refusal_only_accept_names_and_do_not_change() -> None:
    assert visible_runs(LIVE, names=NAMES) == visible_runs(LIVE)
    refusal = "Your notes don't cover this."
    assert refusal_only(refusal, names=NAMES)
    assert refusal_only(refusal)


def test_attribute_infers_an_uncited_live_sentence_as_one_unit() -> None:
    bare = ORDERING.replace(" [S2]", "")
    found = attribute(bare, HITS, SHOWN, VALID, names=NAMES)
    assert [(i.label, i.sentence) for i in found.inferred] == [(2, bare)]
    assert found.uncited == ()


# ---------------------------------------------------------------- properties
_NAME_RUN = re.compile(r"[A-Za-z0-9_.]+")
_IDENT = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
_WORDS = st.sampled_from(
    ["kafka", "Kafka", "it", "It", "and", "js", "a", "b", "x_1", "3", "5", "per", "in", "Then", "e"]
)
_SEPS = st.sampled_from(
    [".", ".", ".", " ", ". ", ".\n", "..", "/", "{", "}", "(", "**", "\u2024", ", ", "_"]
)


@st.composite
def _texts_and_names(draw: st.DrawFn) -> tuple[str, frozenset[str]]:
    pieces = draw(st.lists(st.tuples(_WORDS, _SEPS), min_size=1, max_size=14))
    text = "".join(w + s for w, s in pieces) + draw(_WORDS)
    runs = sorted(dotted_names([text]))
    chosen = draw(st.lists(st.sampled_from(runs), unique=True)) if runs else []
    noise = draw(st.lists(st.sampled_from(["a.b", "kafka.it", "x.y.z", "per.in"]), max_size=2))
    return text, frozenset(chosen) | frozenset(noise)


def _named_dots(text: str, names: frozenset[str]) -> set[int]:
    """The cut positions the exemption may remove, computed independently of coverage.py: the
    end (dot + 1) of every interior ASCII dot followed by an identifier character, in a maximal
    run whose stripped form is in `names`, never a dot before a capital (round 2, P1)."""
    out: set[int] = set()
    for match in _NAME_RUN.finditer(text):
        core = match.group(0).strip(".")
        if core not in names:
            continue
        a = match.start() + match.group(0).index(core)
        for k in range(a + 1, a + len(core) - 1):
            # a dot run ("a..b") is a terminator run: every other rule stays ahead
            interior = text[k] == "." and text[k - 1] in _IDENT and text[k + 1] in _IDENT
            if interior and not "A" <= text[k + 1] <= "Z":
                out.add(k + 1)
    return out


@settings(max_examples=600, deadline=None)
@given(_texts_and_names())
def test_names_remove_only_the_interior_dots_of_named_runs(
    case: tuple[str, frozenset[str]],
) -> None:
    text, names = case
    base = set(_cuts(text))
    with_names = set(_cuts(text, names))
    assert with_names == base - _named_dots(text, names), (text, sorted(names))


@settings(max_examples=400, deadline=None)
@given(_texts_and_names())
def test_cores_not_in_names_keep_every_cut(case: tuple[str, frozenset[str]]) -> None:
    text, _ = case
    absent = frozenset({"zz.top", "q.r.s"}) - dotted_names([text])
    assert _cuts(text, absent) == _cuts(text)


@settings(max_examples=400, deadline=None)
@given(_texts_and_names())
def test_no_dot_then_whitespace_is_ever_merged_even_with_every_name_backed(
    case: tuple[str, frozenset[str]],
) -> None:
    """Worst case: the sources hold EVERY dotted run of the answer. A single dot followed by
    whitespace (". Then", ". lower", ".\\n") still ends a sentence, exactly as without names."""
    text, _ = case
    everything = dotted_names([text])
    with_names = set(_cuts(text, everything))
    for k, char in enumerate(text[:-1]):
        if char == "." and text[k + 1].isspace() and (k + 1) in set(_cuts(text)):
            assert k + 1 in with_names, (text, k)


@settings(max_examples=400, deadline=None)
@given(_texts_and_names())
def test_a_dot_before_a_capital_is_never_exempt_even_with_every_name_backed(
    case: tuple[str, frozenset[str]],
) -> None:
    """Worst case again: in ANY run ("fast.It", "2.8.It", "kafka.streams.Then"), a dot before
    a capital is a sentence end with names exactly when it is one without (round 2, P1)."""
    text, _ = case
    base, with_names = set(_cuts(text)), set(_cuts(text, dotted_names([text])))
    for k, char in enumerate(text[:-1]):
        if char == "." and "A" <= text[k + 1] <= "Z" and k + 1 in base:
            assert k + 1 in with_names, (text, k)


@settings(max_examples=200, deadline=None)
@given(_texts_and_names())
def test_segment_with_names_never_has_more_units(case: tuple[str, frozenset[str]]) -> None:
    """End to end through the parser: removing cuts can only merge pieces, never add any."""
    text, names = case
    assert len(segment(text, names=names)) <= len(segment(text))


# ---------------------------------------------------------------- CPU
def test_the_exemption_stays_linear_on_degenerate_runs() -> None:
    """One run is cached at a time, so a 20,000-character dotted run (every dot a candidate) or
    20,000 characters of short named runs scans each character O(1) times."""
    long_run = ".".join(["a"] * 10_000)
    many = "a.b " * 5_000
    for text, names in (
        (long_run, frozenset({long_run})),
        (long_run, frozenset({"a.a"})),
        (many, frozenset({"a.b"})),
    ):
        started = time.perf_counter()
        _cuts(text, names)
        assert time.perf_counter() - started < 0.2, text[:20]
