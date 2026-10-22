"""NotionConnector: the SourceConnector port for one Notion integration (ADR-0007, DESIGN Flow A).

* `list_changed(since)`: search sorted by last_edited_time DESC; stops at the first page older
  than `since - list_overlap` (default 5 min, inclusive: `last_edited_time` is minute-granular
  and search is eventually consistent, so the overlap re-lists borderline pages; freshness skips
  them). The overlap is applied HERE, once (module C1 spec). A caller that already subtracts its
  own overlap from the cursor should build the connector with `list_overlap=timedelta(0)`.
* `list_all()`: search (pages) PLUS every data source's query endpoint. Search is not
  guaranteed exhaustive; the data-source endpoints are the authoritative row listing.
* `fetch(ref)`: page, then the recursive block tree (depth <= 10). `fetched_at` = Date header of
  the FIRST response minus 60 s: Notion's clock, taken at fetch start, so an edit made while a
  long recursive fetch runs is never mistaken for fresh.
* `confirm_deleted(id)`: True only for 404 (deleted or unshared), in_trash or archived. Every
  other outcome is NOT evidence: an expired token (401) or a capability change (403) must never
  tombstone a whole workspace.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ragforge_core.adapters.connectors.notion.client import NotionClient
from ragforge_core.adapters.connectors.notion.errors import (
    NotionApiError,
    NotionNotFound,
    NotionProtocolError,
)
from ragforge_core.adapters.connectors.notion.ids import anchor_id, canonical_id
from ragforge_core.adapters.connectors.notion.normalize import (
    CHILD_BEARING_TYPES,
    BlockNode,
    normalize_blocks,
)
from ragforge_core.adapters.connectors.notion.properties import (
    properties_markdown,
    simplify_properties,
    title_of,
    truncated_properties,
)
from ragforge_core.adapters.connectors.notion.richtext import LinkCollector
from ragforge_core.domain.errors import NotFound, ValidationFailed
from ragforge_core.domain.models import NormalizedBlock, SourceDocument, SourceItemRef, SourceKind
from ragforge_core.observability.logging import get_logger

MAX_BLOCK_DEPTH = 10
LIST_OVERLAP = timedelta(minutes=5)
FETCHED_AT_SKEW = timedelta(seconds=60)
# A synced ORIGINAL that answers one of these is unavailable to this integration (deleted, not
# shared, or restricted), not a failure of the page being fetched.
_SYNCED_UNAVAILABLE_STATUS = frozenset({403, 404})

log = get_logger(__name__)


class NotionPageGone(NotFound):
    """The page is in the trash/archived (or 404): the caller should confirm and tombstone."""


@dataclass(slots=True)
class _FetchState:
    truncated: bool = False
    unavailable_synced: int = 0


class NotionConnector:
    """SourceConnector for one integration token (wrapped by `client`)."""

    def __init__(
        self,
        client: NotionClient,
        *,
        max_depth: int = MAX_BLOCK_DEPTH,
        list_overlap: timedelta = LIST_OVERLAP,
    ) -> None:
        if not 1 <= max_depth <= MAX_BLOCK_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_BLOCK_DEPTH}")
        if list_overlap < timedelta(0):
            raise ValueError("list_overlap must not be negative")
        self._client = client
        self._max_depth = max_depth
        self._list_overlap = list_overlap

    def __repr__(self) -> str:
        return f"NotionConnector(max_depth={self._max_depth}, list_overlap={self._list_overlap})"

    # ------------------------------------------------------------------ listing
    async def list_changed(self, since: datetime | None) -> AsyncIterator[SourceItemRef]:
        threshold = since - self._list_overlap if since is not None else None
        body = {
            "filter": {"property": "object", "value": "page"},
            "sort": {"timestamp": "last_edited_time", "direction": "descending"},
        }
        seen: set[str] = set()
        async for item in self._client.paginate("POST", "/search", body=body):
            ref = _page_ref(item)
            if ref is None or ref.external_id in seen:
                continue
            if (
                threshold is not None
                and ref.last_edited_time is not None
                and ref.last_edited_time < threshold
            ):
                return  # sorted descending: everything after this is older still
            seen.add(ref.external_id)
            yield ref

    async def list_all(self) -> AsyncIterator[SourceItemRef]:
        seen: set[str] = set()
        page_filter = {"filter": {"property": "object", "value": "page"}}
        async for item in self._client.paginate("POST", "/search", body=page_filter):
            ref = _page_ref(item)
            if ref is not None and ref.external_id not in seen:
                seen.add(ref.external_id)
                yield ref
        ds_filter = {"filter": {"property": "object", "value": "data_source"}}
        async for source in self._client.paginate("POST", "/search", body=ds_filter):
            if source.get("object") != "data_source" or not isinstance(source.get("id"), str):
                continue
            path = f"/data_sources/{canonical_id(source['id'])}/query"
            async for row in self._client.paginate("POST", path, body={}):
                ref = _page_ref(row)
                if ref is not None and ref.external_id not in seen:
                    seen.add(ref.external_id)
                    yield ref

    # ------------------------------------------------------------------ fetch
    async def fetch(self, ref: SourceItemRef) -> SourceDocument:
        try:
            page_id = canonical_id(ref.external_id)
        except ValueError:
            # Permanent: retrying a malformed id can never succeed. Not NotionPageGone, because
            # that invites a tombstone, and a bad id is not evidence that any page was deleted.
            raise ValidationFailed("external_id is not a Notion page id") from None
        try:
            first = await self._client.request("GET", f"/pages/{page_id}")
        except NotionNotFound:
            raise NotionPageGone(f"Notion page {page_id} not found") from None
        if first.date is None:
            # Without Notion's clock the monotonic fetched_at guard is unsound; retry later.
            raise NotionProtocolError("Notion response had no usable Date header")
        fetched_at = first.date - FETCHED_AT_SKEW
        page = first.data
        if _is_gone(page):
            raise NotionPageGone(f"Notion page {page_id} is in the trash")

        state = _FetchState()
        tree = await self._children(page_id, depth=1, state=state, synced_path=frozenset())
        links = LinkCollector()
        properties = page.get("properties") or {}
        lead: tuple[NormalizedBlock, ...] = ()
        metadata: dict[str, Any] = {"notion": _notion_meta(page)}
        if _is_data_source_row(page):
            values = simplify_properties(properties, links)
            metadata["properties"] = values
            truncated_props = truncated_properties(properties)
            if truncated_props:
                metadata["truncated_properties"] = truncated_props
            text = properties_markdown(values)
            if text:
                lead = (NormalizedBlock(anchor=anchor_id(page_id), kind="paragraph", text=text),)
        normalized = normalize_blocks(tree, links)
        truncated = state.truncated or normalized.truncated
        if normalized.unsupported:
            metadata["unsupported_blocks"] = normalized.unsupported
            log.info("notion.unsupported_blocks", page_id=page_id, counts=normalized.unsupported)
        if truncated:
            metadata["truncated_depth"] = self._max_depth
            log.warning("notion.depth_truncated", page_id=page_id, max_depth=self._max_depth)
        if state.unavailable_synced:
            metadata["unavailable_synced_blocks"] = state.unavailable_synced
        return SourceDocument(
            ref=SourceItemRef(
                SourceKind.notion, page_id, _parse_time(page.get("last_edited_time"))
            ),
            title=title_of(properties) or "Untitled",
            external_url=page.get("url") if isinstance(page.get("url"), str) else None,
            blocks=lead + normalized.blocks,
            metadata=metadata,
            links_to=tuple(i for i in normalized.links_to if i != page_id),
            synced_source_ids=normalized.synced_source_ids,
            fetched_at=fetched_at,
        )

    async def _children(
        self, block_id: str, *, depth: int, state: _FetchState, synced_path: frozenset[str]
    ) -> tuple[BlockNode, ...]:
        nodes = []
        path = f"/blocks/{canonical_id(block_id)}/children"
        async for block in self._client.paginate("GET", path):
            nodes.append(await self._node(block, depth=depth, state=state, synced_path=synced_path))
        return tuple(nodes)

    async def _node(
        self,
        block: Mapping[str, Any],
        *,
        depth: int,
        state: _FetchState,
        synced_path: frozenset[str],
    ) -> BlockNode:
        kind = block.get("type")
        if kind == "synced_block":
            return await self._synced(block, depth=depth, state=state, synced_path=synced_path)
        if not block.get("has_children") or kind not in CHILD_BEARING_TYPES:
            return BlockNode(block)
        if depth >= self._max_depth:
            state.truncated = True
            return BlockNode(block, truncated=True)
        children = await self._children(
            str(block["id"]), depth=depth + 1, state=state, synced_path=synced_path
        )
        return BlockNode(block, children)

    async def _synced(
        self,
        block: Mapping[str, Any],
        *,
        depth: int,
        state: _FetchState,
        synced_path: frozenset[str],
    ) -> BlockNode:
        """Resolve a synced block. A REFERENCE (synced_from set) is replaced by its original's
        children and recorded in synced_source_ids: editing the original does not touch this
        page's last_edited_time, so reconcile must watch the original (ADR-0007)."""
        source = ((block.get("synced_block") or {}).get("synced_from") or {}).get("block_id")
        if not isinstance(source, str):
            if not block.get("has_children"):
                return BlockNode(block)
            if depth >= self._max_depth:
                state.truncated = True
                return BlockNode(block, truncated=True)
            children = await self._children(
                str(block["id"]), depth=depth + 1, state=state, synced_path=synced_path
            )
            return BlockNode(block, children)
        source_id = canonical_id(source)
        if source_id in synced_path:  # A embeds B embeds A: render once, never loop
            return BlockNode(block, synced_source_id=source_id)
        if depth >= self._max_depth:
            state.truncated = True
            return BlockNode(block, synced_source_id=source_id, truncated=True)
        try:
            children = await self._children(
                source_id, depth=depth + 1, state=state, synced_path=synced_path | {source_id}
            )
        except NotionApiError as exc:
            # Original deleted, not shared (404) or restricted (403, `restricted_resource`):
            # its content is unavailable but the page itself is fine. Still recorded so
            # reconcile picks it up if access changes. Anything else (5xx, 401) is a real
            # failure of this fetch and propagates.
            if exc.http_status not in _SYNCED_UNAVAILABLE_STATUS:
                raise
            state.unavailable_synced += 1
            return BlockNode(block, synced_source_id=source_id)
        return BlockNode(block, children, synced_source_id=source_id)

    # ------------------------------------------------------------------ deletes
    async def confirm_deleted(self, external_id: str) -> bool:
        try:
            page_id = canonical_id(external_id)
        except ValueError:
            return False
        try:
            response = await self._client.request("GET", f"/pages/{page_id}")
        except NotionNotFound:
            return True
        except Exception as exc:  # any other failure is not evidence of deletion
            log.warning(
                "notion.confirm_deleted.inconclusive", page_id=page_id, error=type(exc).__name__
            )
            return False
        return _is_gone(response.data)


# ---------------------------------------------------------------------- helpers
def _is_gone(page: Mapping[str, Any]) -> bool:
    return bool(page.get("in_trash") or page.get("archived"))


def _is_data_source_row(page: Mapping[str, Any]) -> bool:
    parent = page.get("parent") or {}
    return parent.get("type") in ("data_source_id", "database_id")


def _page_ref(item: Mapping[str, Any]) -> SourceItemRef | None:
    if item.get("object") != "page" or _is_gone(item) or not isinstance(item.get("id"), str):
        return None
    try:
        page_id = canonical_id(item["id"])
    except ValueError:
        return None
    return SourceItemRef(SourceKind.notion, page_id, _parse_time(item.get("last_edited_time")))


def _parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _notion_meta(page: Mapping[str, Any]) -> dict[str, Any]:
    parent = page.get("parent") or {}
    parent_type = parent.get("type")
    return {
        "parent_type": parent_type,
        "parent_id": parent.get(parent_type) if isinstance(parent_type, str) else None,
        "created_time": page.get("created_time"),
        "last_edited_time": page.get("last_edited_time"),
    }
