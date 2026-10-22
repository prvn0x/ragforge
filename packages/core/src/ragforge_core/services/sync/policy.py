"""Pure sync rules: job kinds, dedupe keys, the freshness rule and cursor arithmetic.

Kept free of I/O so the rules that protect correctness (ADR-0007, ADR-0010) are unit-testable
in isolation and shared verbatim by sync_connector and reconcile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from ragforge_core.domain.models import SourceDocument

KIND_SYNC_CONNECTOR = "sync_connector"
KIND_SYNC_PAGE = "sync_page"
KIND_RECONCILE = "reconcile_deletes"
KIND_DELETE_CONNECTOR = "delete_connector"
KIND_INGEST_FILE = "ingest_file"
SYNC_KINDS = frozenset(
    {
        KIND_SYNC_CONNECTOR,
        KIND_SYNC_PAGE,
        KIND_RECONCILE,
        KIND_DELETE_CONNECTOR,
        KIND_INGEST_FILE,
    }
)

CURSOR_SCOPE = "changed"


@dataclass(frozen=True, slots=True)
class SyncPolicy:
    """Tunables (defaults per ADR-0007 / DESIGN section 3). Not secrets; safe to log."""

    # The connector owns the listing overlap (ADR-0007; NotionConnector subtracts 5 min). A
    # second overlap here would only re-list pages and spend rate budget (handoff C1, req. 2).
    cursor_overlap: timedelta = timedelta(0)
    poll_interval_s: int = 300
    lookup_batch: int = 500  # ids per exclusion/freshness query
    delete_batch: int = 500  # documents per delete_connector transaction
    embed_batch: int = 64  # texts per embed_documents call
    heartbeat_every: int = 200  # listed items between lease heartbeats
    max_error_chars: int = 500


# ----------------------------------------------------------------- dedupe keys
# The jobs unique index is (tenant_id, dedupe_key), so every key is tenant-scoped by
# construction; the connector id is in the key because Notion page ids are global and the same
# page may be reachable through two connectors of one tenant. Payloads are a pure function of
# the key (ADR-0010): a merge that keeps the old payload loses nothing.
def sync_page_key(connector_id: UUID, external_id: str) -> str:
    return f"{KIND_SYNC_PAGE}:{connector_id}:{external_id}"


def sync_page_payload(connector_id: UUID, external_id: str) -> dict[str, Any]:
    return {"connector_id": str(connector_id), "external_id": external_id}


# ----------------------------------------------------------------- freshness (ADR-0007)
def truncate_to_minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def is_fresh(last_edited_time: datetime | None, fetched_at: datetime | None) -> bool:
    """A page is fresh iff last_edited_time < date_trunc('minute', fetched_at).

    `last_edited_time` is minute-granular, so an edit in the fetch's own minute must count as
    stale (refetched); an unknown timestamp on either side is never fresh.
    """
    if last_edited_time is None or fetched_at is None:
        return False
    return last_edited_time < truncate_to_minute(fetched_at)


def listing_since(cursor: datetime | None, overlap: timedelta) -> datetime | None:
    """Inclusive cursor with an overlap that absorbs minute granularity and listing lag."""
    return None if cursor is None else cursor - overlap


def next_cursor(current: datetime | None, observed: datetime | None) -> datetime | None:
    """Never moves backwards, whatever the listing returned (the overlap re-lists old items)."""
    if observed is None:
        return current
    if current is None:
        return observed
    return max(current, observed)


# ----------------------------------------------------------------- content hash
def page_sha256(doc: SourceDocument) -> bytes:
    """Raw 32-byte digest of everything a re-sync could change except fetch metadata.

    `unchanged` (same hash) lets the index skip chunk replacement, so the hash must cover every
    field the document row or its chunks derive from; fetched_at/last_edited_time are excluded
    because they change without content changing.
    """
    canonical = {
        "title": doc.title,
        "url": doc.external_url,
        "blocks": [
            [b.anchor, b.kind, b.text, b.heading_level, b.language, b.never_split]
            for b in doc.blocks
        ],
        "metadata": doc.metadata,
        "links_to": list(doc.links_to),
        "synced_source_ids": list(doc.synced_source_ids),
    }
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).digest()
