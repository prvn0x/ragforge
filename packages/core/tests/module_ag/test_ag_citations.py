"""Lenient citation parsing (ADR-0015): small-model markers are normalized to `[S<n>]`, and NO
lenient form can ever make a label that was not shown count as a citation."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt

from ragforge_core.agent.citations import (
    check_citations,
    normalize_citations,
)

SHOWN = (1, 2, 3)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Kafka uses partitions [1].", "Kafka uses partitions [S1]."),
        ("Kafka uses partitions [1, 2].", "Kafka uses partitions [S1, S2]."),
        ("Kafka uses partitions [1][2].", "Kafka uses partitions [S1][S2]."),
        ("Kafka uses partitions (S1).", "Kafka uses partitions [S1]."),
        ("Kafka uses partitions (S1, S2).", "Kafka uses partitions [S1, S2]."),
        ("Kafka uses partitions (S1 and S3).", "Kafka uses partitions [S1, S3]."),
        # a capitalised word before a clause-end "(S<n>)" that is not an acronym of it (round 5)
        ("Joins run in Kafka Streams (S2).", "Joins run in Kafka Streams [S2]."),
        ("Keys are hashed by Redis (S2).", "Keys are hashed by Redis [S2]."),
        ("Kafka uses partitions [S1, S2].", "Kafka uses partitions [S1, S2]."),
        ("Kafka uses partitions [s1; s2].", "Kafka uses partitions [S1, S2]."),
        ("Kafka uses partitions [S 2].", "Kafka uses partitions [S2]."),
        ("Kafka uses partitions [Source 1].", "Kafka uses partitions [S1]."),
        ("Kafka uses partitions [Sources 1, 3].", "Kafka uses partitions [S1, S3]."),
        ("Kafka uses partitions [source: S2].", "Kafka uses partitions [S2]."),
        ("Kafka uses partitions (source 3).", "Kafka uses partitions [S3]."),
        ("Kafka uses partitions.\nSource: S1", "Kafka uses partitions.\n[S1]"),
        ("Kafka uses partitions. Source: S2.", "Kafka uses partitions. [S2]."),
        ("Kafka uses partitions (S1)(S2).", "Kafka uses partitions [S1][S2]."),
        ("The sync worked (T1).", "The sync worked [T1]."),
        # ranges are expanded to the labels they name (validity is decided afterwards)
        ("Kafka uses partitions [S1-S3].", "Kafka uses partitions [S1, S2, S3]."),
        ("Kafka uses partitions [S1-2].", "Kafka uses partitions [S1, S2]."),
        ("Kafka uses partitions [s2 to s3].", "Kafka uses partitions [S2, S3]."),
        ("Kafka uses partitions [Sources 1-3].", "Kafka uses partitions [S1, S2, S3]."),
    ],
)
def test_lenient_forms_normalize_to_canonical_labels(raw: str, canonical: str) -> None:
    assert normalize_citations(raw) == canonical
    assert normalize_citations(canonical) == canonical  # idempotent


@pytest.mark.parametrize(
    "prose",
    [
        "Upload the backup to Amazon S3.",  # an S<n> that is a product name, not a label
        "You will see S3 buckets in the console.",  # a cue word, but not at a clause end
        # AWS S3 after a cue word or in a parenthesis is prose (HLD notes are full of it)
        "For object storage, see S3.",
        "Backups go to cloud storage, cf. S3.",
        "Store blobs in S3 (see S3).",
        "Amazon Simple Storage Service (S3) is cheap.",
        "The data source S3 holds the raw files.",
        "Our backup source: S3 in us-east-1.",
        "Read arr[1] before arr[2].",  # indexing
        "It was released in [2024].",  # a year, not a citation
        "See [1](https://example.com) for details.",  # a Markdown link
        "Use `x[1]` and `(S1)` in code.",  # inline code is not prose
        "```\nvalues = data[1]  # (S1)\n```",  # fenced code is not prose
        "Step (1) comes first.",  # parenthesized numbers are list markers
        "Use [S3 docs](https://aws.amazon.com/s3/) for limits.",  # a link about Amazon S3
        "AWS (S3, EC2 and Lambda) hosts the pipeline.",  # a list of AWS services
        "Logs go to S3 (S3) and to Loki.",  # a product name mid-sentence
        # review round 5: the same names at a clause end (an acronym definition or a vendor)
        "Media is stored in Amazon Simple Storage Service (S3).",
        "Media is stored in Simple Storage Service (S3).",
        "Backups go to AWS (S3).",
        "Logs go to S3 (S3).",
        "Use Amazon S3 (S3), not EBS.",
        # review round 3: a bare-number bracket mid-clause is prose, never a model citation
        "Use list [1, 2] in Python.",
        "The vector [1] holds one element.",
    ],
)
def test_non_citation_text_is_left_byte_identical(prose: str) -> None:
    assert normalize_citations(prose) == prose
    assert check_citations(prose, SHOWN, strict=False).ok  # and it is not a stray fragment


@pytest.mark.parametrize(
    "marker",
    [
        "[S2-S9]",  # a range naming unshown labels: expanded, then rejected label by label
        "[S3-S1]",  # descending: not expanded, a stray fragment
        "[S1-S40]",  # too wide to expand: a stray fragment
        "[S2, S10000]",  # a label too long to be canonical
        "[S1, S2-S9]",
        "[1, S2]",
        "[Ss1]",
        "(S1-S3) mid-sentence and",
        "(S1, S10000) mid-sentence and",
    ],
)
def test_a_malformed_citation_makes_its_sentence_offending(marker: str) -> None:
    """Review round 2, P1: a citation-shaped fragment that is not a valid canonical label is an
    invalid label. Its sentence fails the check, so it can never reach `verified` and the reader
    is never shown a reference to a source that does not exist."""
    sentence = f"Redis Cluster splits the keyspace into 999 hash slots {marker}."
    report = check_citations(normalize_citations(sentence), SHOWN, strict=False)
    assert report.invalid
    assert not report.ok


@pytest.mark.parametrize(
    "marker", ["[7]", "(S7)", "[Source 7]", "[s7]", "(source 7)", "[S1, 7]", "(S2 and S7)"]
)
def test_a_lenient_marker_naming_an_unshown_label_is_rejected_not_accepted(marker: str) -> None:
    answer = normalize_citations(f"Kafka uses partitions {marker}.")
    report = check_citations(answer, SHOWN, strict=False)
    assert "S7" in report.invalid
    assert 7 not in report.cited
    assert report.offending  # the sentence fails: fabricated provenance is never shown


_FORMS = st.sampled_from(
    [
        "[{n}]",
        "[S{n}]",
        "[s{n}]",
        "(S{n})",
        "[Source {n}]",
        "(source {n})",
        "\nSource: S{n}\n",
        "[S{n}, S{m}]",
        "[{n}; {m}]",
        "(S{n} and S{m})",
        "[T{n}]",
        # malformed forms (review round 2): ranges, over-long labels, mixed lists
        "[S{n}-S{m}]",
        "[S{n}-{m}]",
        "[S{n}, S{m}0000]",
        "(S{n}-S{m})",
        "[{n}, S{m}]",
        "[S{n} to S{m}]",
    ]
)
_FILLER = st.text(
    alphabet=st.sampled_from(list("abc S[]();,.:`\n0123456789-~*_>\\<!&#")), max_size=12
)

# The test's OWN definition of a citation-shaped bracket, independent of the implementation's:
# a one-line bracket group made only of label characters that contains an S/T label.
_BRACKET = re.compile(r"\[([^\[\]\n]*)\]")
_LABEL_CHARS = re.compile(r"[\sSTst\d,;:\-]*")
_VALID = re.compile(r"[ST]\d{1,4}(?:\s*,\s*[ST]\d{1,4})*")
# The test's own CommonMark reading (review of the redesign, round 4: the old oracle found code
# with a one-line backtick regex and missed a span across a line break). It finds the text the
# reader sees (text and escapes outside code and HTML) with markdown-it-py directly, not
# through ragforge's structure module.
_MD = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])


def _visible_texts(markdown: str) -> list[str]:
    """Every run of text the reader sees, per inline block (code, HTML, URLs and images split
    runs; a link's destination is not text)."""
    runs: list[str] = []
    for token in _MD.parse(markdown):
        if token.type != "inline":
            continue
        run = ""
        for child in token.children or []:
            if child.type in ("text", "text_special"):
                run += child.content
            elif child.type in ("softbreak", "hardbreak"):
                run += "\n"
            elif child.type in ("code_inline", "html_inline", "image", "link_open", "link_close"):
                runs.append(run)
                run = ""
        runs.append(run)
    return runs


def _validly_cited(group: str, shown: set[int], n_tools: int) -> bool:
    if not _VALID.fullmatch(group):
        return False
    labels = re.findall(r"([ST])(\d+)", group)
    return all(int(n) in shown if k == "S" else 1 <= int(n) <= n_tools for k, n in labels)


@settings(max_examples=400, deadline=None)
@given(
    parts=st.lists(st.tuples(_FILLER, _FORMS, st.integers(0, 99), st.integers(0, 99)), max_size=6),
    shown=st.sets(st.integers(1, 9), max_size=4),
    n_tools=st.integers(0, 2),
)
def test_property_no_text_can_cite_a_label_that_was_not_shown(
    parts: list[tuple[str, str, int, int]], shown: set[int], n_tools: int
) -> None:
    """Whatever the model writes, after normalization every accepted source label was shown,
    and EVERY sentence carrying a citation-shaped bracket is either validly cited or offending
    (review round 2: `cited <= shown` alone holds by construction and missed `[S2-S9]`)."""
    answer = "".join(filler + form.format(n=n, m=m) for filler, form, n, m in parts)
    normalized = normalize_citations(answer)
    report = check_citations(normalized, shown, n_tool_results=n_tools, strict=False)
    assert set(report.cited) <= shown
    if report.invalid:
        assert not report.ok
    # Segmentation-independent (review round 1 of the coverage redesign): every invalid
    # citation-shaped bracket the READER sees (the test's own parse), wherever the units are
    # cut, is offending: the answer is rejected (rule 1). Its text lies in an offending span.
    for run in _visible_texts(normalized):
        for m in _BRACKET.finditer(run):
            group = m.group(1)
            if not (_LABEL_CHARS.fullmatch(group) and re.search(r"[STst]\s?\d", group)):
                continue
            if _validly_cited(group, shown, n_tools):
                continue
            assert not report.ok, (normalized, m.group(0))
    assert normalize_citations(normalized) == normalized
