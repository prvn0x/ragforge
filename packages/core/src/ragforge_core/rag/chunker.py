"""Heading-first chunker (DESIGN section 7.2).

Rules, in priority order:
  1. A heading always starts a new section; a chunk never spans two sections, EXCEPT that a
     section under `min_section_tokens` (80) is merged forward into the next one(s) while the
     merged result stays within `target_tokens`. Tiny sections ("## Notes" + one line) make poor,
     context-free chunks; merging them is the one sanctioned way to cross a heading.
  2. A section within `target_tokens` (400) is exactly one chunk.
  3. A longer section is packed greedily from sentence/line-sized units up to the target. Each
     chunk after the first starts with ~15% overlap taken from the tail of the previous chunk
     (overlap exists ONLY inside long sections, never across a heading).
  4. Code blocks (and any block flagged `never_split`) are atomic: never split, never used as
     overlap. One larger than `max_tokens` becomes its own over-long chunk rather than being cut,
     because half a code block is worse than a truncated one for both retrieval and citation.
     EXCEPT above `hard_max_tokens` (default 1024): the embedding model has a hard context
     (nomic-embed-text: 2048) and rejects longer input outright, so an unbounded atomic chunk made
     a whole page permanently unindexable. Such a block is split at line boundaries into pieces of
     at most `hard_max_tokens`; a fenced code block's pieces are each re-wrapped in the original
     fence (same language) so every piece is still valid, renderable code.
  5. Tables larger than the target are split into row groups, each starting with the repeated
     header row + separator, so every table chunk is self-describing.

Hashes (raw 32-byte digests, never hex):
  display_sha256 = sha256(display_text)    -> collapse duplicates at retrieval time
  embed_sha256   = sha256(embedded_text)   -> content address of the embedding
where embedded_text = "{title} › {heading_path joined by ' › '}\\n\\n{display_text}". The header
is in embedded_text ONLY, so identical paragraphs on different pages embed differently (intended)
but still collapse by display_sha256.

`Chunk.metadata["overlap_chars"]` is the length of the overlap prefix of display_text (0 for the
first chunk of a section): the UI can de-emphasise it and tests use it to prove full coverage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ragforge_core.domain.models import Chunk, NormalizedBlock, SourceDocument
from ragforge_core.rag.tokens import count_tokens

HEADER_SEPARATOR = " › "
_BLOCK_JOIN = "\n\n"
# A line break, whitespace after sentence-final punctuation, or the (possibly empty) gap after
# CJK full-width sentence punctuation, which is normally followed by no space at all. Captured so
# the separator that preceded each piece can be restored when neighbouring pieces land in the
# same chunk.
_SENTENCE_SPLIT = re.compile(r"(\n+|(?<=[.!?:;])[ \t]+|(?<=[\u3002\uff01\uff1f\uff1b])[ \t]*)")
_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HEADING_MARKS = re.compile(r"^\s*#{1,6}\s+")


@dataclass(frozen=True, slots=True)
class _Unit:
    """The smallest piece the packer moves around: a sentence, a line, a code block, a table
    row group, or a whole block when its section is short."""

    text: str
    anchor: str
    tokens: int
    block: int  # index of the source block (pieces of one block rejoin with `sep`)
    sep: str  # separator before this unit when it follows a unit of the same block
    atomic: bool  # code/table/never_split: never split and never repeated as overlap


@dataclass(slots=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: list[tuple[int, NormalizedBlock]]
    tokens: int


@dataclass(frozen=True, slots=True)
class _Draft:
    heading_path: tuple[str, ...]
    units: tuple[_Unit, ...]
    overlap: int  # how many leading units are overlap copied from the previous chunk


class HeadingChunker:
    """Implements the `Chunker` port. Pure and deterministic: same document, same chunks."""

    def __init__(
        self,
        *,
        target_tokens: int = 400,
        max_tokens: int = 512,
        min_section_tokens: int = 80,
        overlap_ratio: float = 0.15,
        hard_max_tokens: int = 1024,
    ) -> None:
        if not 0 < min_section_tokens <= target_tokens <= max_tokens <= hard_max_tokens:
            raise ValueError(
                "require 0 < min_section_tokens <= target_tokens <= max_tokens <= hard_max_tokens"
            )
        if not 0 <= overlap_ratio < 0.5:
            raise ValueError("overlap_ratio must be in [0, 0.5)")
        self.target = target_tokens
        self.max = max_tokens
        self.min_section = min_section_tokens
        self.overlap_tokens = int(target_tokens * overlap_ratio)
        self.hard_max = hard_max_tokens

    # ------------------------------------------------------------------ public
    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        drafts: list[_Draft] = []
        for section in self._merge_small(self._sections(doc.blocks)):
            drafts.extend(self._pack(section))
        return [self._finish(doc.title, ordinal, d) for ordinal, d in enumerate(drafts)]

    # ------------------------------------------------------------------ sections
    def _sections(self, blocks: tuple[NormalizedBlock, ...]) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []  # (level, heading text)
        current = _Section((), [], 0)
        for index, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue
            level = _heading_level(block)
            if level is not None:
                if current.blocks:
                    sections.append(current)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, _HEADING_MARKS.sub("", text).strip() or text))
                current = _Section(tuple(h for _, h in stack), [], 0)
            current.blocks.append((index, block))
            current.tokens += count_tokens(text)
        if current.blocks:
            sections.append(current)
        return sections

    def _merge_small(self, sections: list[_Section]) -> list[_Section]:
        """Rule 1: only an accumulation still under `min_section` may absorb the next section,
        and only if the result fits the target (so merged sections never need splitting)."""
        merged: list[_Section] = []
        for section in sections:
            last = merged[-1] if merged else None
            if (
                last is not None
                and last.tokens < self.min_section
                and last.tokens + section.tokens <= self.target
            ):
                last.blocks.extend(section.blocks)
                last.tokens += section.tokens
            else:
                merged.append(_Section(section.heading_path, list(section.blocks), section.tokens))
        return merged

    # ------------------------------------------------------------------ packing
    def _pack(self, section: _Section) -> list[_Draft]:
        if section.tokens <= self.target:
            whole = tuple(u for i, b in section.blocks for u in self._whole_block_units(i, b))
            return [_Draft(section.heading_path, whole, 0)]
        units = [u for i, b in section.blocks for u in self._fine_units(i, b)]
        drafts: list[_Draft] = []
        current: list[_Unit] = []
        size = 0  # tokens in `current` (additive: units are split at whitespace only)
        overlap = 0
        for unit in units:
            if len(current) > overlap and size + unit.tokens > self.target:
                drafts.append(_Draft(section.heading_path, tuple(current), overlap))
                current = self._overlap_tail(current[overlap:])
                size = sum(u.tokens for u in current)
                if size + unit.tokens > self.max:
                    current, size = [], 0  # overlap must never push a chunk past the hard max
                overlap = len(current)
            current.append(unit)
            size += unit.tokens
        if len(current) > overlap:
            drafts.append(_Draft(section.heading_path, tuple(current), overlap))
        return drafts

    def _overlap_tail(self, previous: list[_Unit]) -> list[_Unit]:
        """Trailing non-atomic units of the previous chunk worth <= overlap_tokens. If even the
        last unit is too long, take a word suffix of it so long sentences still get context."""
        tail: list[_Unit] = []
        budget = self.overlap_tokens
        for unit in reversed(previous):
            if unit.atomic or unit.tokens > budget:
                if not tail and not unit.atomic:
                    tail.append(self._word_suffix(unit, budget))
                break
            tail.insert(0, unit)
            budget -= unit.tokens
        return [u for u in tail if u.tokens > 0]

    @staticmethod
    def _word_suffix(unit: _Unit, budget: int) -> _Unit:
        words = unit.text.split()
        kept: list[str] = []
        used = 0
        for word in reversed(words):
            cost = count_tokens(word)
            if used + cost > budget:
                break
            kept.insert(0, word)
            used += cost
        return _Unit(" ".join(kept), unit.anchor, used, unit.block, unit.sep, False)

    # ------------------------------------------------------------------ units
    def _whole_block_units(self, index: int, block: NormalizedBlock) -> list[_Unit]:
        text = block.text.strip()
        atomic = _atomic(block)
        pieces = self._cap(text) if atomic else [text]
        return [_Unit(p, block.anchor, count_tokens(p), index, _BLOCK_JOIN, atomic) for p in pieces]

    def _cap(self, text: str) -> list[str]:
        """Split an atomic block over `hard_max` at line boundaries (rule 4). Fenced code keeps its
        fence on every piece; a single line still over the ceiling falls back to word windows."""
        if count_tokens(text) <= self.hard_max:
            return [text]
        fence = _FENCE.match(text)
        if fence:
            opener, body, closer = fence.group("open"), fence.group("body"), fence.group("close")
            budget = self.hard_max - count_tokens(opener) - count_tokens(closer)
            return [f"{opener}\n{p}\n{closer}" for p in self._line_groups(body, budget)]
        return self._line_groups(text, self.hard_max)

    def _line_groups(self, text: str, budget: int) -> list[str]:
        groups: list[str] = []
        lines: list[str] = []
        used = 0
        for line in text.split("\n"):
            cost = count_tokens(line)
            if cost > budget:  # one enormous line (minified code, base64): window it
                if lines:
                    groups.append("\n".join(lines))
                    lines, used = [], 0
                groups.extend(_windows(line, budget))
                continue
            if lines and used + cost > budget:
                groups.append("\n".join(lines))
                lines, used = [], 0
            lines.append(line)
            used += cost
        if lines:
            groups.append("\n".join(lines))
        return [g for g in groups if g.strip()]

    def _fine_units(self, index: int, block: NormalizedBlock) -> list[_Unit]:
        text = block.text.strip()
        if block.kind == "table":
            groups = self._table_groups(text)
            if groups is not None:
                return [
                    _Unit(p, block.anchor, count_tokens(p), index, _BLOCK_JOIN, True)
                    for g in groups
                    for p in self._cap(g)
                ]
        if _atomic(block):
            return self._whole_block_units(index, block)
        units: list[_Unit] = []
        for piece, sep in _split_sentences(text):
            for part in self._fit(piece):
                units.append(_Unit(part, block.anchor, count_tokens(part), index, sep, False))
                sep = " "
        return units

    def _fit(self, piece: str) -> list[str]:
        """Split a sentence/line that alone exceeds the target into word windows (and a single
        whitespace-free run longer than the target, e.g. an unpunctuated CJK paragraph, into
        windows of at most `target` tokens)."""
        if count_tokens(piece) <= self.target:
            return [piece]
        out: list[str] = []
        window: list[str] = []
        used = 0
        for word in piece.split():
            for part in _token_windows(word, self.target):
                cost = count_tokens(part)
                if window and used + cost > self.target:
                    out.append(" ".join(window))
                    window, used = [], 0
                window.append(part)
                used += cost
        if window:
            out.append(" ".join(window))
        return out

    def _table_groups(self, text: str) -> list[str] | None:
        """Row groups of a Markdown table, each prefixed with the header + separator rows.
        None if `text` is not a header/separator Markdown table (then it is split as text)."""
        lines = text.split("\n")
        if len(lines) < 3 or not _TABLE_SEPARATOR_ROW.match(lines[1]):
            return None
        header = f"{lines[0]}\n{lines[1]}"
        header_cost = count_tokens(header)
        groups: list[str] = []
        rows: list[str] = []
        used = header_cost
        for row in lines[2:]:
            cost = count_tokens(row)
            if rows and used + cost > self.target:
                groups.append("\n".join([header, *rows]))
                rows, used = [], header_cost
            rows.append(row)
            used += cost
        if rows:
            groups.append("\n".join([header, *rows]))
        return groups

    # ------------------------------------------------------------------ output
    @staticmethod
    def _finish(title: str, ordinal: int, draft: _Draft) -> Chunk:
        overlap_text = _join(draft.units[: draft.overlap])
        display = _join(draft.units)
        overlap_chars = 0
        if draft.overlap:
            # The overlap prefix plus the separator that joins it to the first new unit.
            overlap_chars = len(overlap_text) + len(_separator(draft.units, draft.overlap))
        header = HEADER_SEPARATOR.join(p for p in (title.strip(), *draft.heading_path) if p)
        embedded = f"{header}\n\n{display}" if header else display
        return Chunk(
            anchor=draft.units[draft.overlap].anchor,
            ordinal=ordinal,
            heading_path=draft.heading_path,
            display_text=display,
            embedded_text=embedded,
            display_sha256=hashlib.sha256(display.encode()).digest(),
            embed_sha256=hashlib.sha256(embedded.encode()).digest(),
            token_count=count_tokens(display),
            metadata={"overlap_chars": overlap_chars},
        )


# ---------------------------------------------------------------------- helpers
def _heading_level(block: NormalizedBlock) -> int | None:
    if block.heading_level is not None:
        return block.heading_level
    if block.kind.startswith("heading"):
        suffix = block.kind.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 1
    return None


def _atomic(block: NormalizedBlock) -> bool:
    return block.never_split or block.kind in ("code", "table")


# A whole fenced code block: opening fence (with optional info string), body, matching close.
_FENCE = re.compile(
    r"\A(?P<open>(?P<f>`{3,}|~{3,})[^\n]*)\n(?P<body>.*)\n(?P<close>(?P=f)[`~]*)[ \t]*\Z",
    re.DOTALL,
)


def _windows(line: str, budget: int) -> list[str]:
    """Word windows of at most `budget` tokens (a whitespace-free run is cut by _token_windows)."""
    out: list[str] = []
    window: list[str] = []
    used = 0
    for word in line.split():
        for part in _token_windows(word, budget):
            cost = count_tokens(part)
            if window and used + cost > budget:
                out.append(" ".join(window))
                window, used = [], 0
            window.append(part)
            used += cost
    if window:
        out.append(" ".join(window))
    return out


def _split_sentences(text: str) -> list[tuple[str, str]]:
    """(piece, separator-before-piece) pairs; the separators are whitespace only, so token
    counts stay additive across pieces."""
    parts = _SENTENCE_SPLIT.split(text)
    out: list[tuple[str, str]] = []
    sep = _BLOCK_JOIN
    for i, part in enumerate(parts):
        if i % 2:
            sep = part
        elif part.strip():
            out.append((part.strip(), sep))
    return out


def _token_windows(word: str, size: int) -> list[str]:
    """Consecutive slices of `word` costing at most `size` tokens each.

    Windows are sized in TOKENS, not characters: a character can cost up to 3 tokens (Hangul),
    so a fixed `size`-character window could be 3x over the limit. Each window starts at `size`
    characters and shrinks proportionally to its overshoot; it terminates because every single
    character costs at most 3 tokens, well under any sane `size`."""
    if count_tokens(word) <= size:
        return [word]
    windows: list[str] = []
    start = 0
    while start < len(word):
        end = min(len(word), start + size)
        cost = count_tokens(word[start:end])
        while cost > size and end - start > 1:
            end = start + max(1, (end - start) * size // cost)
            cost = count_tokens(word[start:end])
        windows.append(word[start:end])
        start = end
    return windows


def _separator(units: tuple[_Unit, ...], i: int) -> str:
    return units[i].sep if units[i].block == units[i - 1].block else _BLOCK_JOIN


def _join(units: tuple[_Unit, ...]) -> str:
    if not units:
        return ""
    parts = [units[0].text]
    for i in range(1, len(units)):
        parts.append(_separator(units, i))
        parts.append(units[i].text)
    return "".join(parts)
