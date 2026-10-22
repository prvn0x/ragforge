"""Chunker (DESIGN 7.2): example tests for each rule + Hypothesis properties for the guarantees
(full coverage, code never split, table chunks carry their header, size bounds, determinism)."""

from __future__ import annotations

import hashlib
import itertools
import re
from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ragforge_core.domain.models import (
    Chunk,
    NormalizedBlock,
    SourceDocument,
    SourceItemRef,
    SourceKind,
)
from ragforge_core.domain.ports import Chunker
from ragforge_core.rag.chunker import HeadingChunker
from ragforge_core.rag.tokens import count_tokens

CHUNKER = HeadingChunker()
_TABLE_ROW = re.compile(r"^\|.*\|$")


def doc(*blocks: NormalizedBlock, title: str = "Handbook") -> SourceDocument:
    ref = SourceItemRef(SourceKind.notion, "page-1", datetime(2026, 1, 1, tzinfo=UTC))
    return SourceDocument(ref=ref, title=title, external_url=None, blocks=blocks)


def heading(text: str, level: int, anchor: str = "h") -> NormalizedBlock:
    return NormalizedBlock(anchor, f"heading_{level}", f"{'#' * level} {text}", level)


def para(text: str, anchor: str = "p") -> NormalizedBlock:
    return NormalizedBlock(anchor, "paragraph", text)


def code(text: str, anchor: str = "c") -> NormalizedBlock:
    return NormalizedBlock(anchor, "code", f"```python\n{text}\n```", language="python")


def table(
    header: tuple[str, str], rows: list[tuple[str, str]], anchor: str = "t"
) -> NormalizedBlock:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return NormalizedBlock(anchor, "table", "\n".join(lines))


def sentences(n: int, word: str = "alpha") -> str:
    return " ".join(f"{word} sentence number {i} has several plain words." for i in range(n))


def body(chunk: Chunk) -> str:
    """display_text without the overlap prefix copied from the previous chunk."""
    return chunk.display_text[int(chunk.metadata["overlap_chars"]) :]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})[^\n`]*$", re.MULTILINE)


def unfence(text: str) -> str:
    """Drop fence lines: a code block over the hard ceiling is re-fenced per piece (rule 4)."""
    return _FENCE_LINE.sub("", text)


# ---------------------------------------------------------------------- examples per rule
def test_implements_the_port() -> None:
    assert isinstance(CHUNKER, Chunker)


def test_short_document_is_one_chunk_with_contextual_header_in_embedded_text_only() -> None:
    chunks = CHUNKER.chunk(doc(heading("Setup", 1), para("Install it.", "p1")))
    assert len(chunks) == 1
    (c,) = chunks
    assert c.heading_path == ("Setup",)
    assert c.display_text == "# Setup\n\nInstall it."
    assert c.embedded_text == "Handbook › Setup\n\n# Setup\n\nInstall it."
    assert "Handbook" not in c.display_text
    assert c.display_sha256 == hashlib.sha256(c.display_text.encode()).digest()
    assert c.embed_sha256 == hashlib.sha256(c.embedded_text.encode()).digest()
    assert len(c.display_sha256) == 32
    assert isinstance(c.display_sha256, bytes)
    assert c.anchor == "h"
    assert c.ordinal == 0
    assert c.token_count == count_tokens(c.display_text)


def test_heading_path_follows_levels() -> None:
    long = sentences(30)
    chunks = CHUNKER.chunk(
        doc(
            heading("A", 1, "h1"), para(long, "p1"),
            heading("B", 2, "h2"), para(long, "p2"),
            heading("C", 3, "h3"), para(long, "p3"),
            heading("D", 2, "h4"), para(long, "p4"),
        )
    )  # fmt: skip
    paths = [c.heading_path for c in chunks]
    assert ("A",) in paths
    assert ("A", "B") in paths
    assert ("A", "B", "C") in paths
    assert ("A", "D") in paths  # D (level 2) pops C and B


def test_large_sections_never_cross_a_heading() -> None:
    chunks = CHUNKER.chunk(
        doc(heading("One", 2, "h1"), para(sentences(20), "p1"), heading("Two", 2, "h2"),
            para(sentences(20, "beta"), "p2"))
    )  # fmt: skip
    for c in chunks:
        assert not ("alpha" in c.display_text and "beta" in c.display_text)


def test_small_section_merges_forward_across_a_heading() -> None:
    chunks = CHUNKER.chunk(
        doc(heading("Tiny", 2, "h1"), para("Just a note.", "p1"), heading("Next", 2, "h2"),
            para(sentences(5), "p2"))
    )  # fmt: skip
    assert len(chunks) == 1
    assert "Just a note." in chunks[0].display_text
    assert "## Next" in chunks[0].display_text
    assert chunks[0].heading_path == ("Tiny",)


def test_long_section_splits_near_target_with_overlap() -> None:
    chunks = CHUNKER.chunk(doc(heading("Long", 1), para(sentences(120), "p1")))
    assert len(chunks) > 2
    assert all(c.token_count <= CHUNKER.max for c in chunks)
    assert all(c.token_count >= CHUNKER.target * 0.5 for c in chunks[:-1])
    assert chunks[0].metadata["overlap_chars"] == 0
    for prev, cur in itertools.pairwise(chunks):
        overlap = cur.display_text[: int(cur.metadata["overlap_chars"])].strip()
        assert overlap, "every follow-up chunk inside a long section starts with overlap"
        assert prev.display_text.endswith(overlap)
        assert count_tokens(overlap) <= CHUNKER.overlap_tokens


def test_no_overlap_across_sections() -> None:
    chunks = CHUNKER.chunk(
        doc(heading("A", 1, "h1"), para(sentences(10), "p1"), heading("B", 1, "h2"),
            para(sentences(10), "p2"))
    )  # fmt: skip
    first_of_b = next(c for c in chunks if c.heading_path == ("B",))
    assert first_of_b.metadata["overlap_chars"] == 0
    assert first_of_b.display_text.startswith("# B")


def test_code_block_is_never_split_when_over_max_but_under_the_hard_ceiling() -> None:
    big = "\n".join(f"x_{i} = compute({i})  # step" for i in range(80))
    assert CHUNKER.max < count_tokens(code(big).text) <= CHUNKER.hard_max
    chunks = CHUNKER.chunk(doc(heading("Code", 1), para(sentences(10)), code(big, "c1")))
    holders = [c for c in chunks if "x_0 = compute(0)" in c.display_text]
    assert len(holders) == 1
    assert code(big).text in holders[0].display_text
    assert holders[0].anchor == "c1"


def test_code_block_over_the_hard_ceiling_is_split_into_fenced_pieces() -> None:
    """Rule 4's exception (first real Notion sync): the embedder rejects input over its context."""
    big = "\n".join(f"x_{i} = compute({i})  # step" for i in range(300))
    chunks = CHUNKER.chunk(doc(heading("Code", 1), code(big, "c1")))
    pieces = [c for c in chunks if "compute(" in c.display_text]
    assert len(pieces) > 1
    assert all(c.token_count <= CHUNKER.hard_max for c in pieces)
    assert all(c.display_text.startswith("```python\n") for c in pieces)
    assert all(c.display_text.endswith("\n```") for c in pieces)
    assert all(c.anchor == "c1" for c in pieces)


def test_large_table_splits_into_row_groups_with_header_repeated() -> None:
    rows = [(f"key {i} with words", f"value {i} with more words") for i in range(200)]
    chunks = CHUNKER.chunk(doc(heading("Data", 1), table(("Key", "Value"), rows, "t1")))
    table_chunks = [c for c in chunks if "| key " in c.display_text]
    assert len(table_chunks) > 1
    for c in table_chunks:
        start = c.display_text.index("| Key | Value |")
        assert c.display_text[start:].startswith("| Key | Value |\n| --- | --- |")
        assert c.token_count <= CHUNKER.max
    seen = [r for c in table_chunks for r in re.findall(r"\| key (\d+) ", c.display_text)]
    assert sorted(set(map(int, seen))) == list(range(200))


def test_identical_text_on_two_pages_shares_display_sha_but_not_embed_sha() -> None:
    a = CHUNKER.chunk(doc(heading("H", 1), para("Same words."), title="Page A"))[0]
    b = CHUNKER.chunk(doc(heading("H", 1), para("Same words."), title="Page B"))[0]
    assert a.display_sha256 == b.display_sha256
    assert a.embed_sha256 != b.embed_sha256


def test_empty_and_whitespace_documents() -> None:
    assert CHUNKER.chunk(doc()) == []
    assert CHUNKER.chunk(doc(para("   "), para("\n"))) == []


def test_giant_word_without_spaces_is_split_not_dropped() -> None:
    blob = "x" * 5000
    chunks = CHUNKER.chunk(doc(heading("B", 1), para(blob)))
    assert squash("".join(body(c) for c in chunks)).count("x") == 5000
    assert all(c.token_count <= CHUNKER.max for c in chunks)


def test_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValueError, match="require"):
        HeadingChunker(target_tokens=600, max_tokens=512)


# ---------------------------------------------------------------------- properties
_words = st.text(alphabet="abcdefghij", min_size=1, max_size=12)
_sentence = st.lists(_words, min_size=1, max_size=25).map(lambda ws: " ".join(ws) + ".")
_paragraph = st.lists(_sentence, min_size=1, max_size=40).map(" ".join)


@st.composite
def _blocks(draw: st.DrawFn) -> tuple[NormalizedBlock, ...]:
    out: list[NormalizedBlock] = []
    for i in range(draw(st.integers(min_value=1, max_value=14))):
        kind = draw(st.sampled_from(["heading", "para", "para", "para", "code", "table"]))
        anchor = f"b{i}"
        if kind == "heading":
            level = draw(st.integers(min_value=1, max_value=3))
            out.append(heading(f"Head {i}", level, anchor))
        elif kind == "para":
            out.append(para(draw(_paragraph), anchor))
        elif kind == "code":
            n = draw(st.integers(min_value=1, max_value=250))  # uniform: big blocks are common
            out.append(code("\n".join(f"call_{i}(arg_{j})" for j in range(n)), anchor))
        else:
            n = draw(st.integers(min_value=1, max_value=120))
            out.append(
                table((f"K{i}", f"V{i}"), [(f"r{i}x{j}", f"w {j}") for j in range(n)], anchor)
            )
    return tuple(out)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_blocks())
def test_property_full_coverage_in_order(blocks: tuple[NormalizedBlock, ...]) -> None:
    """Every non-whitespace character of the source appears, in order, in the chunks once the
    overlap prefixes are removed; only repeated table headers may add text."""
    chunks = CHUNKER.chunk(doc(*blocks))
    source = squash(unfence("\n".join(b.text for b in blocks)))
    produced = squash(unfence("\n".join(body(c) for c in chunks)))
    if not any(b.kind == "table" for b in blocks):
        assert produced == source
    else:
        it = iter(produced)
        assert all(ch in it for ch in source), "source is not a subsequence of the chunks"


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_blocks())
def test_property_code_blocks_never_split(blocks: tuple[NormalizedBlock, ...]) -> None:
    """Under the hard ceiling a code block is atomic; over it, every piece is its own fenced block
    within the ceiling (rule 4)."""
    chunks = CHUNKER.chunk(doc(*blocks))
    for block in blocks:
        if block.kind != "code":
            continue
        if count_tokens(block.text) <= CHUNKER.hard_max:
            assert any(block.text in c.display_text for c in chunks), "code block was split"
        else:
            opener = block.text.split("\n", 1)[0]
            pieces = [c for c in chunks if c.anchor == block.anchor]
            assert all(c.display_text.startswith(opener) for c in pieces)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_blocks())
def test_property_table_rows_are_preceded_by_their_header(
    blocks: tuple[NormalizedBlock, ...],
) -> None:
    chunks = CHUNKER.chunk(doc(*blocks))
    for c in chunks:
        current_header: str | None = None
        lines = c.display_text.split("\n")
        for i, line in enumerate(lines):
            if not _TABLE_ROW.match(line) or line.startswith("| ---"):
                continue
            if i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
                current_header = line  # a header row (followed by the separator)
                continue
            assert current_header is not None, f"table row without header in chunk {c.ordinal}"
            table_id = re.match(r"\| r(\d+)x", line)
            if table_id:
                assert current_header.startswith(f"| K{table_id.group(1)} |")


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_blocks())
def test_property_bounds_ordinals_and_determinism(blocks: tuple[NormalizedBlock, ...]) -> None:
    chunks = CHUNKER.chunk(doc(*blocks))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= CHUNKER.hard_max  # the embedder's context is a hard limit
        if c.token_count > CHUNKER.max:  # only an atomic code block may exceed the max
            assert any(b.kind == "code" and b.anchor == c.anchor for b in blocks)
        assert c.display_text.strip()
        assert c.anchor in {b.anchor for b in blocks}
    assert CHUNKER.chunk(doc(*blocks)) == chunks


# ---------------------------------------------------------------------- unspaced scripts
def test_token_counter_charges_unspaced_scripts_per_character() -> None:
    """WordPiece emits one token per CJK ideograph (and per kana / Thai character, and per jamo
    of an NFD-decomposed Hangul syllable); a whole run is NOT one word."""
    assert count_tokens("中" * 700) == 700
    assert count_tokens("ひらがなカタカナ") == 8
    assert count_tokens("สวัสดี") == 6
    assert count_tokens("한국") == 6  # NFD: 한 = 3 jamo, 국 = 3 jamo
    assert count_tokens("가") == 2  # NFD: 2 jamo
    assert count_tokens("abc中文def") == 4
    assert count_tokens("中文。") == 3  # full-width punctuation is its own token


def test_unpunctuated_chinese_paragraph_never_exceeds_max() -> None:
    """Review finding: 2160 Chinese characters used to come out as ONE 2004-char chunk counted
    as 392 tokens (really ~2000 model tokens), which the reranker silently truncated."""
    text = "".join(chr(0x4E00 + (i * 37) % 20000) for i in range(2160))
    chunks = CHUNKER.chunk(doc(heading("规则", 1), para(text)))
    assert len(chunks) >= 2160 // CHUNKER.max
    assert all(c.token_count <= CHUNKER.max for c in chunks)
    assert all(len(body(c)) <= CHUNKER.target for c in chunks)
    assert squash("".join(body(c) for c in chunks)) == squash("# 规则" + text)


def test_chinese_sentences_split_at_full_width_stops() -> None:
    sentence = "这是一个测试句子用于检查分块" * 4 + "。"  # 57 tokens, no spaces anywhere
    chunks = CHUNKER.chunk(doc(heading("规则", 1), para(sentence * 30)))
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= CHUNKER.max
        assert body(c).rstrip().endswith("。")  # chunks break between sentences


_cjk = st.text(alphabet=st.characters(min_codepoint=0x4E00, max_codepoint=0x4FFF), min_size=1)
_kana = st.text(alphabet=st.characters(min_codepoint=0x3041, max_codepoint=0x30FA), min_size=1)
_hangul = st.text(alphabet=st.characters(min_codepoint=0xAC00, max_codepoint=0xD7A3), min_size=1)
_thai = st.text(alphabet=st.characters(min_codepoint=0x0E01, max_codepoint=0x0E30), min_size=1)
_intl_word = st.one_of(_cjk, _kana, _hangul, _thai, _words).map(lambda w: w[:60])
_intl_sentence = st.tuples(
    st.lists(_intl_word, min_size=1, max_size=20),
    st.sampled_from(["", " "]),  # CJK text usually has no spaces at all
    st.sampled_from(["\u3002", "\uff01", ".", ""]),
).map(lambda t: t[1].join(t[0]) + t[2])
_intl_paragraph = st.lists(_intl_sentence, min_size=1, max_size=30).map("".join)


@st.composite
def _intl_blocks(draw: st.DrawFn) -> tuple[NormalizedBlock, ...]:
    out: list[NormalizedBlock] = []
    for i in range(draw(st.integers(min_value=1, max_value=8))):
        if draw(st.booleans()):
            out.append(heading(f"見出し {i}", draw(st.integers(min_value=1, max_value=3)), f"h{i}"))
        out.append(para(draw(_intl_paragraph), f"p{i}"))
    return tuple(out)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_intl_blocks())
def test_property_non_latin_text_is_covered_and_bounded(
    blocks: tuple[NormalizedBlock, ...],
) -> None:
    """With no code or tables, every chunk fits the hard max, and its token_count is the real
    count of its text, for CJK, kana, Hangul, Thai and mixed text alike."""
    chunks = CHUNKER.chunk(doc(*blocks))
    assert squash("".join(body(c) for c in chunks)) == squash("".join(b.text for b in blocks))
    for c in chunks:
        assert c.token_count == count_tokens(c.display_text)
        assert c.token_count <= CHUNKER.max
