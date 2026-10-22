"""FLAG (a): a single identifier in an answer's inline code span is backed by the code of the
S page the sentence cites (owner report 2026-10-18: "explain Memento" flagged `createMemento()`,
which sits inside a larger code block of the cited page).

Every test goes through `attribute` + `settle` with a verifier that says "supported": the answer
is `verified` exactly when the token rule (and nothing else) backs the span. The rule is narrow,
so most tests are negatives: each one is a way an earlier partial-code match leaked."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_d1.d1_fakes import make_hit
from ragforge_core.agent.attribution import (
    Settled,
    attribute,
    identifier_name,
    settle,
)
from ragforge_core.agent.coverage import valid_labels
from ragforge_core.agent.state import HitDict, hit_to_state
from ragforge_core.domain.models import Verification

# A realistic note: prose, then the class in a fenced block (the Notion normalizer's output).
MEMENTO_CODE = """\
public class ServiceConfiguration {
    private static final int MAX_RETRIES = 3;
    private String endpoint;

    public ConfigMemento createMemento() {
        return new ConfigMemento(this.endpoint);
    }

    public void restore(ConfigMemento memento) {
        this.endpoint = memento.getEndpoint();
        props.put("max.poll.interval.ms", 300000);
        retry(MAX_RETRIES, backoff_ms);
        producer.configure("acks=all retries=3");
    }
}"""
MEMENTO_PAGE = (
    "The Memento pattern captures an object's state so it can be restored later. "
    "The undoStack keeps every snapshot.\n\n"
    f"```java\n{MEMENTO_CODE}\n```"
)
OTHER_PAGE = "Kafka stores records in partitions.\n\n```java\nvoid saveSnapshot() {}\n```"
HITS: list[HitDict] = [
    hit_to_state(make_hit(1, title="Memento pattern", text=MEMENTO_PAGE)),
    hit_to_state(make_hit(2, title="Kafka", text=OTHER_PAGE)),
]
VALID = valid_labels([1, 2], 0)
# A sentence the label covers on its own, so an answer with one flagged sentence is `partial`.
COVERED = "The Memento pattern captures an object's state so it can be restored later [S1]. "


def _settled(answer: str, hits: list[HitDict] = HITS, *claims: str) -> Settled:
    """The answer settled against a verifier that objects to exactly `claims`."""
    found = attribute(answer, hits, range(1, len(hits) + 1), valid_labels(range(1, 3), 0))
    return settle(
        answer,
        verdict=Verification.partial if claims else Verification.verified,
        conclusive=True,
        located=True,
        verifier_claims=claims,
        valid=VALID,
        attribution=found,
        tools_ran=False,
    )


def _sentence(span: str, label: str = "S1") -> str:
    return f"The method `{span}` restores the saved endpoint [{label}]."


def _backed(span: str, label: str = "S1") -> bool:
    return _settled(_sentence(span, label)).status is Verification.verified


# ---------------------------------------------------------------- positives
def test_the_reported_memento_answer_is_verified() -> None:
    """The owner's case, reproduced: both names are whole tokens of the cited page's block."""
    answer = (
        "The `ServiceConfiguration` class has a `createMemento()` method that saves its "
        "endpoint [S1]."
    )
    settled = _settled(answer)
    assert settled.status is Verification.verified
    assert (settled.unsupported_claims, settled.reasons, settled.code_unbacked) == ((), (), ())


@pytest.mark.parametrize(
    "span",
    [
        "createMemento()",  # declared with no argument
        "createMemento",  # the bare name of the method
        "ServiceConfiguration",  # camel hump
        "ConfigMemento",
        "MAX_RETRIES",  # "_"
        "backoff_ms",
        "max.poll.interval.ms",  # a whole dotted name inside a string literal
        "memento.getEndpoint()",  # a whole dotted call
        "getEndpoint()",  # a simple name matches any dotted segment
        "this.endpoint",
        "restore()",  # `name()` names a method declared WITH arguments: restore(ConfigMemento m)
    ],
)
def test_a_single_identifier_from_the_cited_code_is_backed(span: str) -> None:
    assert identifier_name(span)
    assert _backed(span)


def test_an_inline_code_span_of_the_cited_page_counts() -> None:
    """The page's author marked the name as code exactly as a block does."""
    page = "Tune it with `session.timeout.ms` before deploying."
    hits = [hit_to_state(make_hit(1, title="Kafka", text=page)), HITS[1]]
    answer = "The `session.timeout.ms` setting is tuned before deploying [S1]."
    assert _settled(answer, hits).status is Verification.verified


def test_an_indented_code_block_of_the_cited_page_counts() -> None:
    page = "Restore it like this:\n\n    snapshot = editor.saveState()\n    editor.load(snapshot)\n"
    hits = [hit_to_state(make_hit(1, title="Editor", text=page)), HITS[1]]
    answer = "The editor exposes `saveState()` to take the snapshot [S1]."
    assert _settled(answer, hits).status is Verification.verified


def test_one_of_several_cited_pages_may_back_the_span() -> None:
    answer = "The `createMemento()` method saves the endpoint state [S2][S1]."
    assert _settled(answer).status is Verification.verified


# ---------------------------------------------------------------- negatives
@pytest.mark.parametrize(
    ("span", "why"),
    [
        ("createMementoX()", "a longer identifier"),
        ("xcreateMemento()", "a longer identifier"),
        ("Memento()", "a suffix of ConfigMemento"),
        ("CreateMemento()", "case-sensitive"),
        ("createmento()", "a deleted character"),
        ("interval.ms", "a dotted sub-path names a different key"),
        ("max.poll", "a dotted prefix names a different key"),
        ("poll.interval", "a dotted infix"),
        ("endpoint.getEndpoint()", "a dotted name the page never writes"),
        ("undoStack", "only in the page's PROSE"),
        ("saveSnapshot()", "only in an UNCITED page's code"),
        ("memento", "no identifier shape: an ordinary word"),
        ("endpoint", "no identifier shape"),
        ("acks=all", "'=' states a value: whole-element only (the block has it, not whole)"),
        ("ConfigMemento createMemento()", "multi-token"),
        ("createMemento(this)", "an argument list"),
        (" createMemento()", "a leading space the parser keeps"),
        ("createMem\u0435nto()", "a Cyrillic look-alike"),
        ("createMem\u200bento()", "an invisible character"),
        ("createMemento\uff08\uff09", "full-width parentheses"),
        ("createMemento( )", "something inside the parentheses"),
    ],
)
def test_anything_but_the_exact_whole_token_is_not_backed(span: str, why: str) -> None:
    assert not _backed(span), why


def test_a_tool_label_never_backs_code() -> None:
    """Only `S` labels name page code; a T label has no page to read."""
    answer = "The method `createMemento()` restores the saved endpoint [T1]."
    found = attribute(answer, HITS, [1, 2], valid_labels([1, 2], 1))
    assert found.code_unbacked == (answer,)


def test_a_bidi_control_inside_the_span_makes_the_answer_residue() -> None:
    settled = _settled(_sentence("createMemento\u202e()"))
    assert settled.status is not Verification.verified
    assert settled.unsupported_claims


def test_a_source_token_next_to_a_non_ascii_or_kebab_character_backs_nothing() -> None:
    """`createMemento` inside `\u00e9createMemento` or `--max_memory` inside a CLI flag may read as
    part of a longer token: fail closed."""
    page = "```\n\u00e9createMemento()\nredis-cli --max_memory 2gb\n```"
    hits = [hit_to_state(make_hit(1, title="Odd", text=page)), HITS[1]]
    for span in ("createMemento()", "max_memory"):
        assert _settled(_sentence(span), hits).status is not Verification.verified, span


def test_source_code_with_a_bidi_control_backs_no_identifier() -> None:
    page = "```\nint createMemento() { return 0; } // \u202e reversed\n```"
    hits = [hit_to_state(make_hit(1, title="Odd", text=page)), HITS[1]]
    assert _settled(_sentence("createMemento()"), hits).status is not Verification.verified


def test_a_joined_span_is_never_backed_even_when_each_half_is_a_token() -> None:
    """The fail-closed `joined_code` rule runs ahead of the token rule."""
    joined = "Call `memento`**`.getEndpoint()`** to read the endpoint [S1]."
    settled = _settled(COVERED + joined)
    assert settled.status is Verification.partial
    assert (settled.unsupported_claims, settled.reasons) == ((joined,), ("not_found",))


def test_an_unlabelled_sentence_never_borrows_the_token_rule() -> None:
    """Inferred citations keep the whole-element rule (`code_matches` is unchanged): only a
    model label states which page the name came from."""
    answer = "The ServiceConfiguration class uses `createMemento()` to capture endpoint state."
    found = attribute(answer, HITS, [1, 2], VALID)
    assert (found.inferred, found.uncited) == ((), (answer,))


def test_whole_commands_still_need_the_whole_element() -> None:
    page = "Create a topic:\n```bash\nkafka-topics --create --topic logs --partitions 3\n```"
    hits = [hit_to_state(make_hit(1, title="CLI", text=page)), HITS[1]]
    excerpt = "Create it with `kafka-topics --create --topic logs` from the shell [S1]."
    settled = _settled(f"Create a topic from the shell [S1]. {excerpt}", hits)
    assert settled.status is Verification.partial
    assert settled.code_unbacked == (excerpt,)


# ---------------------------------------------------------------- the claim about the name
def test_a_planted_error_about_a_backed_name_is_the_verifiers_to_catch() -> None:
    """The token rule backs the NAME only. What the sentence says about it is the ground check's:
    with an objection the claim is flagged (red), without one it is verified, exactly as for the
    prose around it (ADR-0015 section 4.1, verifier-dependent)."""
    planted = "The `createMemento()` method deletes the whole database [S1]."
    caught = _settled(planted, HITS, planted)
    assert caught.status is Verification.partial
    assert (caught.unsupported_claims, caught.reasons) == ((planted,), ("not_found",))
    missed = _settled(planted)
    assert missed.status is Verification.verified  # the documented residual, not the token rule


def test_a_verifier_quote_of_part_of_the_sentence_still_flags_it_red() -> None:
    sentence = "The `createMemento()` method deletes the whole database [S1]."
    caught = _settled(sentence, HITS, "deletes the whole database")
    assert caught.status is Verification.partial
    assert caught.reasons == ("not_found",)


# ---------------------------------------------------------------- properties
_PAGE_WORDS = frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", MEMENTO_CODE))
_ALPHABET = "aeoMXx_.()0 -=$\u0435\u200b"


def _mutations(name: str) -> set[str]:
    """Every one-character deletion, substitution and insertion of `name` over `_ALPHABET`."""
    out = {name[:i] + name[i + 1 :] for i in range(len(name))}
    for i in range(len(name) + 1):
        for c in _ALPHABET:
            out.add(name[:i] + c + name[i:])
            if i < len(name):
                out.add(name[:i] + c + name[i + 1 :])
    return out - {name}


def _really_on_page(span: str) -> bool:
    """An independent oracle: the span's last name is a word of the page's code and the whole
    span is literally in it."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", span)
    return bool(words) and words[-1] in _PAGE_WORDS and span.removesuffix("()") in MEMENTO_CODE


@pytest.mark.parametrize(
    "name", ["createMemento()", "ServiceConfiguration", "MAX_RETRIES", "max.poll.interval.ms"]
)
def test_no_one_character_mutation_of_a_backed_name_is_backed(name: str) -> None:
    """Property (ADR-0015: "property-tested with one-character mutations"): a mutation is backed
    only if it is itself a real token of the page (the independent oracle), never by accident."""
    assert _backed(name)
    leaked = [m for m in sorted(_mutations(name)) if _backed(m) and not _really_on_page(m)]
    assert leaked == []


@settings(max_examples=150, deadline=None)
@given(st.text(alphabet="abcXYZ_.()019 =-\u0435", min_size=1, max_size=14))
def test_a_backed_span_is_always_a_grammar_identifier_on_the_page(span: str) -> None:
    """Whatever the span, backing implies the grammar holds and the name is on the page."""
    if "`" in span or span.strip() != span:
        return
    if _backed(span):
        assert identifier_name(span)
        assert _really_on_page(span)


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ("createMemento()", True),
        ("ConfigurationManager", True),
        ("max.poll.interval.ms", True),
        ("MAX_RETRIES", True),
        ("_private", True),
        ("Memento", False),  # a word
        ("acks", False),
        ("true", False),
        ("null", False),
        ("id", False),
        ("3", False),
        ("acks=all", False),
        ("foo(x)", False),
        ("a..b", False),
        (".b", False),
        ("b.", False),
        ("x()()", False),
        ("caf\u00e9Bar", False),  # non-ASCII: never, even with a camel hump
        ("createMemento ()", False),
    ],
)
def test_the_identifier_grammar(span: str, expected: bool) -> None:
    assert identifier_name(span) is expected
