"""Regression (first real Notion sync): an atomic block larger than the embedding model's context
made a page permanently unindexable (Ollama 400 "input exceeds the model context"). The chunker
now enforces `hard_max_tokens` on every chunk while keeping each piece valid code / a table with
its header."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ragforge_core.domain.models import NormalizedBlock, SourceDocument, SourceItemRef, SourceKind
from ragforge_core.rag.chunker import HeadingChunker
from ragforge_core.rag.tokens import count_tokens

HARD = 1024


def _doc(*blocks: NormalizedBlock) -> SourceDocument:
    ref = SourceItemRef(SourceKind.notion, "page-1", datetime(2026, 1, 1, tzinfo=UTC))
    return SourceDocument(ref=ref, title="Handbook", external_url=None, blocks=blocks)


def _code_block(lines: int, lang: str = "java") -> NormalizedBlock:
    body = "\n".join(f'    System.out.println("line {i} of a long listing");' for i in range(lines))
    return NormalizedBlock("c", "code", f"```{lang}\n{body}\n```", language=lang)


def test_huge_code_block_is_capped_and_every_piece_is_a_valid_fence() -> None:
    chunks = HeadingChunker().chunk(_doc(_code_block(900)))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.display_text) <= HARD
        assert c.display_text.startswith("```java\n")
        assert c.display_text.endswith("\n```")
    # Nothing lost: every source line appears exactly once across the pieces.
    seen = [ln for c in chunks for ln in c.display_text.split("\n") if "System.out" in ln]
    assert len(seen) == 900
    assert len(set(seen)) == 900


def test_code_block_under_the_ceiling_stays_one_atomic_chunk() -> None:
    block = _code_block(35)  # over the 512 max, under the 1024 hard ceiling
    assert 512 < count_tokens(block.text) <= HARD
    chunks = HeadingChunker().chunk(_doc(block))
    assert [c.display_text for c in chunks] == [block.text]


def test_one_enormous_line_is_windowed_under_the_ceiling() -> None:
    blob = " ".join(f"tok{i}" for i in range(4000))  # e.g. minified code / base64
    block = NormalizedBlock("c", "code", f"```\n{blob}\n```", language=None)
    chunks = HeadingChunker().chunk(_doc(block))
    assert all(count_tokens(c.display_text) <= HARD for c in chunks)
    assert " ".join(c.display_text.strip("`\n") for c in chunks).split() == blob.split()


def test_huge_table_row_group_is_capped() -> None:
    wide = " | ".join(["x" * 40] * 60)
    table = "| a | b |\n| --- | --- |\n" + "\n".join(f"| {i} | {wide} |" for i in range(30))
    block = NormalizedBlock("t", "table", table, never_split=True)
    chunks = HeadingChunker().chunk(_doc(block))
    assert all(count_tokens(c.display_text) <= HARD for c in chunks)


def test_ceiling_must_not_be_below_max() -> None:
    with pytest.raises(ValueError, match="hard_max_tokens"):
        HeadingChunker(max_tokens=512, hard_max_tokens=400)


@settings(max_examples=40, deadline=None)
@given(
    lines=st.integers(min_value=1, max_value=1500),
    width=st.integers(min_value=1, max_value=60),
    fence=st.sampled_from(["```", "~~~", "````"]),
)
def test_no_chunk_ever_exceeds_the_ceiling(lines: int, width: int, fence: str) -> None:
    body = "\n".join(" ".join(f"w{j}" for j in range(width)) for _ in range(lines))
    block = NormalizedBlock("c", "code", f"{fence}py\n{body}\n{fence}", language="py")
    for c in HeadingChunker().chunk(_doc(block)):
        assert count_tokens(c.display_text) <= HARD
