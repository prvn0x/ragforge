"""Deterministic identifiers (DESIGN section 9, ADR-0014). FROZEN: every module derives document and
chunk ids through these functions; changing them breaks deep links and restores."""

from __future__ import annotations

from uuid import UUID, uuid5

# Fixed forever. Generated once for RAGForge; never change.
RAGFORGE_NAMESPACE = UUID("6f1c2a9e-3b7d-5c44-9a0e-2f8d1b7c4e53")

_UPLOAD = "upload"


def document_id(
    tenant_id: UUID, collection_id: UUID, connector_id: UUID | None, external_id: str
) -> UUID:
    """Same tuple as the documents uniqueness key: (tenant, collection, connector|'upload', ext)."""
    source = str(connector_id) if connector_id is not None else _UPLOAD
    return uuid5(
        RAGFORGE_NAMESPACE, f"doc\x1f{tenant_id}\x1f{collection_id}\x1f{source}\x1f{external_id}"
    )


def chunk_id(document: UUID, ordinal: int) -> UUID:
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    return uuid5(RAGFORGE_NAMESPACE, f"chunk\x1f{document}\x1f{ordinal}")
