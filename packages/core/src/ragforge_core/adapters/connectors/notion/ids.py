"""Notion id forms.

* external ids (documents.external_id, links_to, synced_source_ids): canonical dashed lowercase
  UUID, the form the API returns, so the same object always has the same key.
* anchors: the 32-hex dashless form, which is exactly what a Notion deep link fragment uses,
  so `deep_link(page_url, anchor)` is plain concatenation for every consumer.
"""

from __future__ import annotations

from uuid import UUID


def canonical_id(raw: str) -> str:
    """Dashed lowercase UUID. Raises ValueError for anything that is not a Notion id."""
    return str(UUID(raw))


def anchor_id(raw: str) -> str:
    """Dashless lowercase 32-hex id (the deep-link fragment form)."""
    return UUID(raw).hex


def deep_link(page_url: str, block_id: str) -> str:
    """`https://www.notion.so/Page-<id>#<blockid-without-dashes>`."""
    base = page_url.split("#", 1)[0]
    return f"{base}#{anchor_id(block_id)}"
