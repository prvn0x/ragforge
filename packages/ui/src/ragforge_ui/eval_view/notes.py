"""The notes a model was SHOWN, fetched on demand and digest-checked (ADR-0017 §1, §7).

Run files hold chunk ids and display digests, never note text. The text is fetched through the
SDK's `eval_chunks` only when the owner asks for it, and is shown ONLY when the chunk's digest
still equals the one the run recorded: otherwise the note changed since the run and showing the
new text next to the old answer would mislead (it is reported, never silently substituted).
The text lives in local variables of one script run; it is never put in session state or cache.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID

from ragforge_client.ask import read_key
from ragforge_client.client import RagforgeClient
from ragforge_client.evals.schema import ShownRef
from ragforge_contracts.api import EvalChunk
from ragforge_ui.config import display_url


class ChunkSource(Protocol):
    """Where note text comes from (the SDK in the app, a fake in tests)."""

    def chunks(self, chunk_ids: Sequence[UUID]) -> list[EvalChunk]:
        """The visible chunks among `chunk_ids`; invisible ids are simply absent."""
        ...


class SdkChunkSource:
    """`ChunkSource` over `RagforgeClient.eval_chunks` with the eval key.

    The key is read from its file on each fetch (hidden-file rules of `read_key`), never held:
    this object may sit in a local variable across a rerun's lifetime, and its repr shows only
    the display-safe URL."""

    def __init__(self, api_url: str, key_file: Path, *, timeout: float = 30.0) -> None:
        self._url = api_url
        self._key_file = key_file
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"SdkChunkSource(api_url={display_url(self._url)!r})"

    @property
    def api_url(self) -> str:
        return display_url(self._url)

    def chunks(self, chunk_ids: Sequence[UUID]) -> list[EvalChunk]:
        ids = list(chunk_ids)
        key = read_key(self._key_file)

        async def go() -> list[EvalChunk]:
            async with RagforgeClient(self._url, key, timeout=self._timeout) as rf:
                return (await rf.eval_chunks(ids)).chunks

        return asyncio.run(go())


class NoteState(StrEnum):
    ok = "ok"
    changed = "changed"  # the chunk's text is not the text the model was shown
    missing = "missing"  # not visible to the eval key (deleted, or another tenant)


@dataclass(frozen=True, slots=True)
class Note:
    label: int  # the S<label> the model saw
    state: NoteState
    where: tuple[str, ...] = ()  # page key, then heading path (only when fetched)
    text: str | None = None  # only when state is ok


def sha256_hex(text: str) -> str:
    """The server's display digest: SHA-256 of the display text's UTF-8 bytes (chunker.py)."""
    return hashlib.sha256(text.encode()).hexdigest()


def shown_notes(refs: Sequence[ShownRef], source: ChunkSource) -> list[Note]:
    """Fetch the shown chunks in one call and keep a text only if BOTH the server's digest and
    our own hash of the text match the run's recorded digest (a server can't vouch for itself)."""
    ids: dict[int, UUID] = {}
    for ref in refs:
        try:
            ids[ref.label] = UUID(ref.chunk_id)
        except ValueError:
            continue  # not an id this server could have issued: reported as missing
    fetched = {c.chunk_id: c for c in source.chunks(list(ids.values()))} if ids else {}
    notes: list[Note] = []
    for ref in sorted(refs, key=lambda r: r.label):
        chunk = fetched.get(ids[ref.label]) if ref.label in ids else None
        if chunk is None:
            notes.append(Note(ref.label, NoteState.missing))
            continue
        where = (f"{chunk.page_key.source_kind}:{chunk.page_key.key}", *chunk.heading_path)
        same = (
            chunk.display_sha256 == ref.display_sha256
            and sha256_hex(chunk.display_text) == ref.display_sha256
        )
        state = NoteState.ok if same else NoteState.changed
        notes.append(Note(ref.label, state, where, chunk.display_text if same else None))
    return notes
