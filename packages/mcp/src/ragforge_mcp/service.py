"""What each MCP tool does, as plain async methods over the SDK (ADR-0005: thin API client).

Kept apart from `server.py` so the behaviour is testable without the MCP protocol, and so the
protocol layer stays a list of descriptions. Every API failure leaves through `_guard`, which turns
it into a `ToolError` (or `ResourceError`) the agent can act on, with the key redacted.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from uuid import UUID

import anyio
from mcp.server.mcpserver.exceptions import (
    ResourceError,
    ResourceNotFoundError,
    ToolError,
)

from ragforge_client.ask import clean
from ragforge_client.client import RagforgeClient
from ragforge_contracts.api import Collection, Connector, Document, Page
from ragforge_mcp.config import MAX_COLLECTIONS
from ragforge_mcp.errors import EXPECTED, Redactor, redact_model, to_tool_error
from ragforge_mcp.models import (
    AskResult,
    CollectionOut,
    ConnectorOut,
    SearchHit,
    SearchResult,
    SourcesResult,
    SyncResult,
    SyncSkipped,
    SyncStarted,
)
from ragforge_mcp.turn import TurnFailed, collect_turn, document_uri

PAGE_LIMIT = 200  # the API's maximum page size
MAX_PAGES = 50  # 10k items: DESIGN's per-tenant page budget; stops a runaway cursor loop
# A whole chat turn, end to end. The SDK stream has no read timeout (tokens may pause while the
# model thinks), so without this a server that stalls after the headers would hang the tool call
# until the MCP host gives up. A turn normally takes 5-60 s.
ASK_TIMEOUT_S = 180.0


async def all_items[T](
    fetch: Callable[[str | None], Awaitable[Page[T]]], *, max_pages: int = MAX_PAGES
) -> AsyncIterator[T]:
    """Walk a cursor-paginated list endpoint, at most `max_pages` pages."""
    cursor: str | None = None
    for _ in range(max_pages):
        page = await fetch(cursor)
        for item in page.items:
            yield item
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


class RagforgeService:
    def __init__(
        self,
        client: RagforgeClient,
        *,
        api_url: str,
        ui_url: str,
        redact: Redactor,
        collections: Sequence[str] = (),
        ask_timeout_s: float = ASK_TIMEOUT_S,
    ) -> None:
        self._rf = client
        self._ask_timeout_s = ask_timeout_s
        self._api_url = api_url
        self._ui_url = ui_url
        self._redact = redact
        self._wanted = tuple(collections)
        self._scope: list[UUID] | None = None
        self._scope_lock = anyio.Lock()

    # ------------------------------------------------------------- tools
    async def ask(self, question: str, conversation_id: UUID | None = None) -> AskResult:
        with self._guard():
            scope = await self._collection_scope()
            events = self._rf.chat(question, scope, conversation_id=conversation_id)
            try:
                # The deadline lives inside collect_turn: after the verdict a timeout returns
                # the saved answer with a warning instead of an error (review P2).
                result = await collect_turn(
                    events,
                    ui_url=self._ui_url,
                    redact=self._redact,
                    conversation_id=conversation_id,
                    deadline_s=self._ask_timeout_s,
                )
            except TurnFailed as exc:
                raise ToolError(self._redact(str(exc))) from None
            except TimeoutError:
                raise ToolError(
                    f"The answer took longer than {self._ask_timeout_s:g} s and was abandoned. "
                    "The API may be overloaded; retry shortly (with the same conversation_id "
                    "for a follow-up)."
                ) from None
        return redact_model(result, self._redact)

    async def search(self, query: str, k: int) -> SearchResult:
        with self._guard():
            response = await self._rf.query(query, await self._collection_scope(), k=k)
        hits = [
            SearchHit(
                rank=rank,
                title=clean(hit.citation.title),
                heading_path=[clean(h) for h in hit.citation.heading_path],
                url=str(hit.citation.url) if hit.citation.url else None,
                snippet=clean(hit.citation.snippet),
                rerank_score=hit.rerank_score,
                document_uri=document_uri(hit.citation.document_id),
            )
            for rank, hit in enumerate(response.hits, start=1)
        ]
        return redact_model(SearchResult(query=query, hits=hits), self._redact)

    async def list_sources(self) -> SourcesResult:
        """The diagnostic tool, so it must work exactly when the scope is broken (an unknown
        --collection, an empty tenant): it reports the problem as `config_warning` instead of
        failing like ask/search do."""
        with self._guard():
            collections = await self._collections()
            connectors = await self._connectors()
        scope, problem = self._match(collections)
        names = {c.id: c.name for c in collections}
        result = SourcesResult(
            collections=[
                CollectionOut(id=c.id, name=clean(c.name), in_scope=c.id in scope)
                for c in collections
            ],
            connectors=[_connector_out(c, names.get(c.target_collection_id)) for c in connectors],
            config_warning=problem,
        )
        return redact_model(result, self._redact)

    async def sync_now(self, connector_id: UUID | None = None) -> SyncResult:
        with self._guard():
            if connector_id is not None:
                targets = [await self._rf.get_connector(connector_id)]
            else:
                targets = [c for c in await self._connectors() if str(c.kind) == "notion"]
                if not targets:
                    raise ToolError("This API key sees no Notion connector: nothing to sync.")
        started: list[SyncStarted] = []
        skipped: list[SyncSkipped] = []
        failures: list[Exception] = []
        for connector in targets:
            name = clean(connector.display_name)
            if str(connector.status) == "deleting":
                skipped.append(_skipped(connector, name, "the connector is being deleted"))
                continue
            try:
                accepted = await self._rf.sync_connector(connector.id)
            except EXPECTED as exc:
                failures.append(exc)
                reason = str(to_tool_error(exc, self._api_url, self._redact))
                skipped.append(_skipped(connector, name, reason))
                continue
            started.append(
                SyncStarted(connector_id=connector.id, display_name=name, job_id=accepted.job_id)
            )
        if failures and not started:
            self._fail_if_uniform(failures)
        return redact_model(SyncResult(started=started, skipped=skipped), self._redact)

    def _fail_if_uniform(self, failures: list[Exception]) -> None:
        """Nothing started and every connector failed the same way (a 401, a missing `write`
        scope, the API down): one clear ToolError beats N identical `skipped` entries. Mixed
        failures stay per connector in `skipped` (review r2: one 409 must not hide the rest)."""
        messages = {str(to_tool_error(exc, self._api_url, self._redact)) for exc in failures}
        if len(messages) == 1:
            raise to_tool_error(failures[0], self._api_url, self._redact) from None

    # ------------------------------------------------------------- resource
    async def document(self, document_id: str) -> str:
        """Metadata of one document as JSON, only within the collections ask/search read (the
        --collection scope applies to every way into the notes, review r2). The API has no
        GET /v1/documents/{id} and exposes no chunk text (handoff MCP.md, API gap 2), so this
        walks the document list of each in-scope collection (filtered by the API): about one
        call per collection plus one per 200 documents, the cost of that gap."""
        try:
            wanted = UUID(document_id)
        except ValueError:
            raise ResourceNotFoundError(f"not a document id: {clean(document_id)[:64]}") from None
        with self._guard(resource=True):
            try:
                scope = await self._collection_scope()
            except ToolError as exc:  # a broken --collection config: same message as ask/search
                raise ResourceError(str(exc)) from None
            for collection_id in scope:
                doc = await self._find_document(collection_id, wanted)
                if doc is not None:
                    return self._redact(_document_json(doc))
        # the same answer for "no such document" and "outside the scope": no existence oracle
        raise ResourceNotFoundError(f"no document {wanted} is visible to this server's collections")

    # ------------------------------------------------------------- internals
    @contextmanager
    def _guard(self, *, resource: bool = False) -> Iterator[None]:
        try:
            yield
        except EXPECTED as exc:
            error = to_tool_error(exc, self._api_url, self._redact)
            if resource:
                raise ResourceError(str(error)) from None
            raise error from None

    async def _find_document(self, collection_id: UUID, wanted: UUID) -> Document | None:
        def fetch(cursor: str | None) -> Awaitable[Page[Document]]:
            return self._rf.list_documents(
                collection_id=collection_id, cursor=cursor, limit=PAGE_LIMIT
            )

        async for doc in all_items(fetch):
            if doc.id == wanted:
                return doc
        return None

    async def _collections(self) -> list[Collection]:
        fetch = self._rf.list_collections
        return [c async for c in all_items(lambda c: fetch(cursor=c, limit=PAGE_LIMIT))]

    async def _connectors(self) -> list[Connector]:
        fetch = self._rf.list_connectors
        return [c async for c in all_items(lambda c: fetch(cursor=c, limit=PAGE_LIMIT))]

    async def _collection_scope(self) -> list[UUID]:
        """Collections ask/search read, resolved on first use (not at startup: the server must
        start, and list its tools, even while the API is down). Cached for the process lifetime;
        a collection created later needs a restart, the same as `make ask`."""
        async with self._scope_lock:
            if self._scope is None:
                self._scope = self._resolve(await self._collections())
            return self._scope

    def _resolve(self, visible: list[Collection]) -> list[UUID]:
        scope, problem = self._match(visible)
        if problem is not None:
            raise ToolError(self._redact(problem))
        return scope

    def _match(self, visible: list[Collection]) -> tuple[list[UUID], str | None]:
        """(the collections ask/search read, why that scope is unusable or None). Never raises:
        `list_sources` reports the problem, ask/search fail with it (`_resolve`)."""
        if not visible:
            return [], (
                "This API key sees no collections. Create one and connect Notion first "
                "(make connect-notion)."
            )
        if not self._wanted:
            return [c.id for c in visible][:MAX_COLLECTIONS], None
        by_ref = {c.name: c.id for c in visible} | {str(c.id): c.id for c in visible}
        missing = [w for w in self._wanted if w not in by_ref]
        if missing:
            return [], (
                f"Unknown collection(s) in the server config: {', '.join(missing)}. "
                f"Visible: {', '.join(clean(c.name) for c in visible)}."
            )
        return list(dict.fromkeys(by_ref[w] for w in self._wanted)), None


def _skipped(connector: Connector, name: str, reason: str) -> SyncSkipped:
    return SyncSkipped(connector_id=connector.id, display_name=name, reason=reason)


def _connector_out(c: Connector, collection_name: str | None) -> ConnectorOut:
    return ConnectorOut(
        id=c.id,
        kind=str(c.kind),
        display_name=clean(c.display_name),
        status=str(c.status),
        last_error=clean(c.last_error) if c.last_error else None,
        collection_id=c.target_collection_id,
        collection_name=clean(collection_name) if collection_name else None,
    )


def _document_json(doc: Document) -> str:
    body = {
        "id": str(doc.id),
        "collection_id": str(doc.collection_id),
        "title": clean(doc.title),
        "url": doc.external_url,
        "external_updated_at": _iso(doc.external_updated_at),
        "fetched_at": _iso(doc.fetched_at),
        "chunks": None,
        "note": "The RAGForge API does not expose chunk text yet; use the search tool for "
        "passages of this document.",
    }
    return json.dumps(body, indent=2)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
