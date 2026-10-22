"""The embed input of a stored chunk, rebuilt for a re-embed (ADR-0017 r4 P2).

The ingest path embeds `embedded_text = "{title} › {heading_path joined by ' › '}\\n\\n{display}"`
(rag/chunker.py `Chunker._finish`; the embedder adds the space's doc prefix itself) and stores
`embed_sha256 = sha256(embedded_text)`. The snapshot stores chunks, not raw source, so a
re-embed must rebuild exactly that text. The rule is mirrored here because services may not
import rag (import-linter layers) and the chunker has no public function for it yet (the
handoff asks the integrator to extract one and wire it in); `checked_embed_input` therefore
re-hashes every rebuilt text against the stored `embed_sha256` BEFORE anything is embedded or
written, so a divergence can never key a vector under the wrong hash: the job fails instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

HEADER_SEPARATOR: Final = " › "  # rag.chunker.HEADER_SEPARATOR (a test keeps them equal)


class EmbedInputMismatch(ValueError):
    """The rebuilt embed input does not hash to the stored embed_sha256."""


def embedded_text(title: str, heading_path: Sequence[str], display_text: str) -> str:
    header = HEADER_SEPARATOR.join(p for p in (title.strip(), *heading_path) if p)
    return f"{header}\n\n{display_text}" if header else display_text


def checked_embed_input(
    title: str, heading_path: Sequence[str], display_text: str, embed_sha256: bytes
) -> str:
    text = embedded_text(title, heading_path, display_text)
    if hashlib.sha256(text.encode()).digest() != embed_sha256:
        raise EmbedInputMismatch(
            "a snapshot chunk's rebuilt embed input does not match its embed_sha256 (the "
            "chunker's header rule changed?): refusing to re-embed; take a new snapshot"
        )
    return text
