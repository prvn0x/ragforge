"""Reference implementation (the executable definition) of the two snapshot digests.

The production digests are computed in SQL by `rf._eval_snapshot_digests` (migration 0012),
inside the same transaction as the copy, a re-embed or an exclusion purge, so a stored digest
always describes committed content. This module is the readable specification that pins that
SQL (a property test compares both on random corpora), the way
`evals/demo/fixtures/reference_metrics.py` pins the metric names.

Encoding: every field is a netstring `"<utf-8 byte length>:<text>,"`, so no separator inside a
title, key or chunk text can make two different inputs collide. Ordering is by code point,
which is byte order on UTF-8 and therefore Postgres' `COLLATE "C"`.

content digest (the gate requires equality)
    tag, then each page sorted by (source_kind, page_key, part) where
    part = source_kind, page_key, chunk count, the display texts in ordinal order.
index digest (recorded; may differ between arms)
    tag, the active space id, then the sorted entries
    (source_kind, page_key, heading count, headings..., display_sha256 hex, occurrence,
    embed_sha256 hex); `occurrence` is the chunk's rank among equal
    (page, heading_path, display_sha256) in ordinal order (ADR-0017 principle 6).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

CONTENT_TAG: Final = "ragforge.eval_snapshot.content.v1"
INDEX_TAG: Final = "ragforge.eval_snapshot.index.v1"


@dataclass(frozen=True, slots=True)
class SnapshotChunk:
    ordinal: int
    heading_path: tuple[str, ...]
    display_text: str
    display_sha256: bytes
    embed_sha256: bytes


@dataclass(frozen=True, slots=True)
class SnapshotPage:
    source_kind: str
    key: str  # page_key(...) of the document
    chunks: tuple[SnapshotChunk, ...]


def netstring(value: str) -> str:
    return f"{len(value.encode())}:{value},"


def page_key(source_kind: str, external_id: str, metadata: Mapping[str, object]) -> str:
    """notion: the page id; file: metadata.filename (the upload id is random per install)."""
    if source_kind == "file":
        filename = metadata.get("filename")
        return filename if isinstance(filename, str) else external_id
    return external_id


def occurrences(chunks: Sequence[SnapshotChunk]) -> list[int]:
    """Occurrence of each chunk (same order as `chunks`), counted in ordinal order."""
    seen: Counter[tuple[tuple[str, ...], bytes]] = Counter()
    rank: dict[int, int] = {}
    for chunk in sorted(chunks, key=lambda c: c.ordinal):
        key = (chunk.heading_path, chunk.display_sha256)
        rank[chunk.ordinal] = seen[key]
        seen[key] += 1
    return [rank[c.ordinal] for c in chunks]


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _page_part(page: SnapshotPage) -> str:
    ordered = sorted(page.chunks, key=lambda c: c.ordinal)
    return (
        netstring(page.source_kind)
        + netstring(page.key)
        + netstring(str(len(ordered)))
        + "".join(netstring(c.display_text) for c in ordered)
    )


def content_digest(pages: Iterable[SnapshotPage]) -> str:
    parts = sorted((p.source_kind, p.key, _page_part(p)) for p in pages)
    return _sha256_hex(netstring(CONTENT_TAG) + "".join(part for _, _, part in parts))


def _index_entries(page: SnapshotPage) -> list[str]:
    entries = []
    for chunk, occurrence in zip(page.chunks, occurrences(page.chunks), strict=True):
        entries.append(
            netstring(page.source_kind)
            + netstring(page.key)
            + netstring(str(len(chunk.heading_path)))
            + "".join(netstring(h) for h in chunk.heading_path)
            + netstring(chunk.display_sha256.hex())
            + netstring(str(occurrence))
            + netstring(chunk.embed_sha256.hex())
        )
    return entries


def index_digest(space_id: int, pages: Iterable[SnapshotPage]) -> str:
    entries = sorted(entry for page in pages for entry in _index_entries(page))
    return _sha256_hex(netstring(INDEX_TAG) + netstring(str(space_id)) + "".join(entries))
