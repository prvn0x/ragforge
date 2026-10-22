"""NotionConnector (SourceConnector port): fetched_at from the FIRST Date header, inclusive
5-minute listing overlap, data-source rows as documents, and confirm_deleted semantics."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from module_c1.c1_support import (
    BASE,
    HANDBOOK_ID,
    MENTIONED_PAGE_ID,
    ROW_ID,
    SYNCED_ORIGINAL_ID,
    TOKEN,
    FakeLimiter,
    FakeSleep,
    error_body,
    load,
    paged_handler,
    serve_page,
    serve_search,
)
from ragforge_core.adapters.connectors.notion import (
    NotionApiError,
    NotionClient,
    NotionConnector,
    NotionPageGone,
    NotionProtocolError,
)
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import SourceItemRef, SourceKind
from ragforge_core.domain.ports import SourceConnector

DATE = {"Date": "Fri, 16 Oct 2026 12:00:00 GMT"}
GONE_ID = "a7000000-0000-4000-8000-000000000009"


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_mocked=True) as mock:
        yield mock


def _connector(max_attempts: int = 3) -> NotionConnector:
    client = NotionClient(TOKEN, FakeLimiter(), sleep=FakeSleep(), max_attempts=max_attempts)
    return NotionConnector(client)


def _ref(page_id: str) -> SourceItemRef:
    return SourceItemRef(SourceKind.notion, page_id, None)


def test_implements_the_frozen_port() -> None:
    assert isinstance(_connector(), SourceConnector)


# ---------------------------------------------------------------------- fetch
async def test_fetched_at_is_the_first_responses_date_minus_60s(router: respx.MockRouter) -> None:
    # The page GET answers at 12:00:00, the (later) block listings at 12:07:30: an edit made
    # during the long recursive fetch must still look newer than fetched_at.
    serve_page(router, "handbook_page.json", "handbook_blocks.json")
    doc = await _connector().fetch(_ref(HANDBOOK_ID))
    assert doc.fetched_at == datetime(2026, 10, 16, 11, 59, 0, tzinfo=UTC)


async def test_fetch_accepts_dashless_ids_and_returns_canonical_ref(
    router: respx.MockRouter,
) -> None:
    serve_page(router, "handbook_page.json", "handbook_blocks.json")
    doc = await _connector().fetch(_ref(HANDBOOK_ID.replace("-", "")))
    assert doc.ref.external_id == HANDBOOK_ID
    assert doc.metadata["notion"]["parent_type"] == "workspace"


async def test_missing_date_header_fails_closed(router: respx.MockRouter) -> None:
    router.get(f"/pages/{HANDBOOK_ID}").mock(
        return_value=httpx.Response(200, json=load("handbook_page.json"))
    )
    with pytest.raises(NotionProtocolError, match="Date"):
        await _connector().fetch(_ref(HANDBOOK_ID))


async def test_data_source_row_is_its_own_document_with_properties_in_metadata(
    router: respx.MockRouter,
) -> None:
    serve_page(router, "row_page.json", "row_blocks.json")
    doc = await _connector().fetch(_ref(ROW_ID))
    assert doc.title == "Fix login bug"
    assert doc.metadata["properties"] == {
        "Name": "Fix login bug",
        "Status": "In progress",
        "Tags": ["auth", "p1"],
        "Due": {"start": "2026-10-20", "end": None},
        "Owner": ["Ada Lovelace"],
        "Points": 3,
        "Done": False,
        "Spec": "https://example.com/spec",
        "Notes": "Repro on Safari only",
        "Related": [MENTIONED_PAGE_ID],
        "Ticket": "ENG-42",
        "Estimate": 6,
        "Attachment": ["trace.log"],
        # "Run" (button) is an unknown property type: dropped, not stored raw
    }
    assert doc.metadata["notion"]["parent_type"] == "data_source_id"
    assert doc.links_to == (MENTIONED_PAGE_ID,)  # relations are page links
    lead, body = doc.blocks
    assert lead.anchor == ROW_ID.replace("-", "")
    assert lead.kind == "paragraph"
    assert "**Status:** In progress" in lead.text
    assert "**Tags:** auth, p1" in lead.text
    assert "SIGNEDSECRET" not in lead.text + str(doc.metadata)
    assert body.text == "Repro steps are in the attached log."
    assert "truncated_properties" not in doc.metadata


async def test_properties_cut_short_by_the_page_object_are_recorded(
    router: respx.MockRouter,
) -> None:
    """Notion returns at most 25 relation/people/rich_text items and sets `has_more`: the gap
    must be visible in metadata, not silently look like the complete value."""
    serve_page(router, "row_page.json", "row_blocks.json")
    page = load("row_page.json")
    page["properties"]["Related"]["has_more"] = True
    page["properties"]["Owner"]["has_more"] = True
    router.get(f"/pages/{ROW_ID}").mock(return_value=httpx.Response(200, json=page, headers=DATE))
    doc = await _connector().fetch(_ref(ROW_ID))
    assert doc.metadata["truncated_properties"] == ["Owner", "Related"]
    assert doc.metadata["properties"]["Related"] == [MENTIONED_PAGE_ID]  # partial, still kept


async def test_trashed_or_missing_page_raises_page_gone(router: respx.MockRouter) -> None:
    trashed = {**load("handbook_page.json"), "in_trash": True}
    router.get(f"/pages/{HANDBOOK_ID}").mock(
        return_value=httpx.Response(200, json=trashed, headers=DATE)
    )
    router.get(f"/pages/{GONE_ID}").mock(
        return_value=httpx.Response(404, json=error_body(404), headers=DATE)
    )
    with pytest.raises(NotionPageGone):
        await _connector().fetch(_ref(HANDBOOK_ID))
    with pytest.raises(NotionPageGone):
        await _connector().fetch(_ref(GONE_ID))


@pytest.mark.parametrize(
    "unavailable",
    [
        httpx.Response(404, json=error_body(404)),
        httpx.Response(403, json={"object": "error", "status": 403, "code": "restricted_resource"}),
    ],
)
async def test_unshared_synced_original_is_recorded_but_not_fatal(
    router: respx.MockRouter, unavailable: httpx.Response
) -> None:
    blocks = load("handbook_blocks.json")
    page = load("handbook_page.json")
    router.get(f"/pages/{HANDBOOK_ID}").mock(
        return_value=httpx.Response(200, json=page, headers=DATE)
    )
    for parent, pages in blocks.items():
        if parent.startswith("_"):
            continue
        if parent == SYNCED_ORIGINAL_ID:
            router.get(f"/blocks/{parent}/children").mock(return_value=unavailable)
        else:
            router.get(f"/blocks/{parent}/children").mock(side_effect=paged_handler(pages))
    doc = await _connector().fetch(_ref(HANDBOOK_ID))
    assert doc.synced_source_ids == (SYNCED_ORIGINAL_ID,)
    assert doc.metadata["unavailable_synced_blocks"] == 1
    assert "Shared security policy text." not in [b.text for b in doc.blocks]


# ---------------------------------------------------------------------- listing
async def test_list_changed_uses_inclusive_5_minute_overlap_and_stops_early(
    router: respx.MockRouter,
) -> None:
    route = serve_search(router, load("search_changed.json"))
    since = datetime(2026, 10, 16, 11, 40, tzinfo=UTC)
    refs = [r async for r in _connector().list_changed(since)]
    assert [r.external_id for r in refs] == [
        HANDBOOK_ID,
        ROW_ID,
        # a7..03 is in the trash: skipped (reconcile confirms deletes)
        "a7000000-0000-4000-8000-000000000004",  # 11:35 == since - 5 min: inclusive
        # duplicate handbook on page 2: de-duplicated; 11:34 ends the listing
    ]
    assert refs[0].last_edited_time == datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    assert route.call_count == 2  # page 3 never requested: sorted descending
    body = json.loads(route.calls[0].request.content)
    assert body["sort"] == {"timestamp": "last_edited_time", "direction": "descending"}
    assert body["filter"] == {"property": "object", "value": "page"}


async def test_synced_original_server_errors_still_fail_the_fetch() -> None:
    """Only 403/404 mean "unavailable to us"; a 5xx must not silently drop content."""
    # assert_all_called=False: the fetch aborts, so later block listings are never requested.
    with respx.mock(base_url=BASE, assert_all_mocked=True, assert_all_called=False) as router:
        serve_page(router, "handbook_page.json", "handbook_blocks.json")
        router.get(f"/blocks/{SYNCED_ORIGINAL_ID}/children").mock(
            return_value=httpx.Response(500, json=error_body(500))
        )
        with pytest.raises(NotionApiError) as info:
            await _connector(max_attempts=1).fetch(_ref(HANDBOOK_ID))
    assert info.value.http_status == 500


async def test_list_overlap_is_configurable_so_it_is_applied_exactly_once(
    router: respx.MockRouter,
) -> None:
    """A caller that already subtracted its overlap builds the connector with 0: 11:35 is then
    older than `since` and ends the listing."""
    serve_search(router, load("search_changed.json"))
    client = NotionClient(TOKEN, FakeLimiter(), sleep=FakeSleep())
    connector = NotionConnector(client, list_overlap=timedelta(0))
    since = datetime(2026, 10, 16, 11, 40, tzinfo=UTC)
    refs = [r.external_id async for r in connector.list_changed(since)]
    assert refs == [HANDBOOK_ID, ROW_ID]
    with pytest.raises(ValueError, match="list_overlap"):
        NotionConnector(client, list_overlap=timedelta(seconds=-1))


async def test_list_changed_without_since_lists_everything(router: respx.MockRouter) -> None:
    pages = load("search_changed.json")
    pages[1] = {**pages[1], "has_more": False, "next_cursor": None}
    serve_search(router, pages)
    refs = [r.external_id async for r in _connector().list_changed(None)]
    assert len(refs) == 5
    assert len(set(refs)) == 5


async def test_list_all_adds_data_source_rows_and_deduplicates(router: respx.MockRouter) -> None:
    fx = load("list_all.json")

    def search(request: httpx.Request) -> httpx.Response:
        value = json.loads(request.content)["filter"]["value"]
        key = "search_pages" if value == "page" else "search_data_sources"
        return httpx.Response(200, json=fx[key], headers=DATE)

    router.post("/search").mock(side_effect=search)
    query = router.post("/data_sources/da000000-0000-4000-8000-000000000001/query").mock(
        side_effect=paged_handler(fx["data_source_query"])
    )
    refs = [r.external_id async for r in _connector().list_all()]
    assert refs == [HANDBOOK_ID, ROW_ID, "d5000000-0000-4000-8000-000000000002"]
    assert query.call_count == 2


# ---------------------------------------------------------------------- confirm_deleted
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(404, json=error_body(404)), True),  # deleted or not shared
        (httpx.Response(200, json={**load("handbook_page.json"), "in_trash": True}), True),
        (httpx.Response(200, json={**load("handbook_page.json"), "archived": True}), True),
        (httpx.Response(200, json=load("handbook_page.json")), False),  # alive
        (httpx.Response(401, json=error_body(401)), False),  # revoked token != deleted
        (httpx.Response(403, json=error_body(403)), False),  # capability change != deleted
        (httpx.Response(400, json=error_body(400)), False),
        (httpx.Response(500, json=error_body(500)), False),
        (httpx.Response(429, json=error_body(429), headers={"Retry-After": "600"}), False),
        (httpx.ConnectError("down"), False),
        # A 404 that is not Notion's object_not_found (proxy HTML, wrong route) proves nothing.
        (httpx.Response(404, text="<html><h1>404 Not Found</h1></html>"), False),
        (httpx.Response(404, json={"object": "error", "code": "invalid_request_url"}), False),
    ],
)
async def test_confirm_deleted_only_on_definitive_evidence(
    router: respx.MockRouter, response: httpx.Response | Exception, expected: bool
) -> None:
    route = router.get(f"/pages/{GONE_ID}")
    if isinstance(response, Exception):
        route.mock(side_effect=response)
    else:
        route.mock(return_value=response)
    assert await _connector().confirm_deleted(GONE_ID) is expected


async def test_confirm_deleted_with_a_malformed_id_is_not_evidence(
    router: respx.MockRouter,
) -> None:
    assert await _connector().confirm_deleted("../../users/me") is False
    assert not router.calls


async def test_fetch_with_a_malformed_id_fails_permanently_without_a_request(
    router: respx.MockRouter,
) -> None:
    with pytest.raises(ValidationFailed):
        await _connector().fetch(_ref("not-a-notion-id"))
    assert not router.calls
