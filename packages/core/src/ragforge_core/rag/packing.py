"""Context packing: the final step of retrieval (DESIGN 2: packed context 2k tokens local).

Each source is rendered as a numbered block so the generator can cite `[n]`; `PackedContext.hits`
maps n (1-based position) back to the RankedChunk and therefore to its anchor and URL. Hits that
do not fit the remaining budget are skipped (not truncated: a truncated passage can lose the very
sentence that supports a claim, which the ground check would then reject).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragforge_core.domain.models import RankedChunk
from ragforge_core.rag.chunker import HEADER_SEPARATOR
from ragforge_core.rag.tokens import count_tokens

DEFAULT_BUDGET_TOKENS = 2000


@dataclass(frozen=True, slots=True)
class PackedContext:
    text: str
    hits: tuple[RankedChunk, ...]  # hits[i] is source [i + 1] in `text`
    tokens: int


def render_source(number: int, hit: RankedChunk) -> str:
    location = HEADER_SEPARATOR.join(p for p in (hit.title, *hit.heading_path) if p)
    return f"[{number}] {location}\n{hit.display_text}"


def pack(
    hits: Sequence[RankedChunk], *, budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> PackedContext:
    """Pack hits in order into at most `budget_tokens` (approximate, see rag.tokens)."""
    parts: list[str] = []
    kept: list[RankedChunk] = []
    used = 0
    for hit in hits:
        block = render_source(len(kept) + 1, hit)
        cost = count_tokens(block)
        if used + cost > budget_tokens:
            continue
        parts.append(block)
        kept.append(hit)
        used += cost
    return PackedContext("\n\n".join(parts), tuple(kept), used)
