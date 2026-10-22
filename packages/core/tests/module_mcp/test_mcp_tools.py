"""`search`, `list_sources`, `sync_now`, the document resource, collection scope and API errors."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from mcp.shared.exceptions import MCPError
from mcp_support import (
    API,
    COLL_A,
    COLL_B,
    KEY,
    cite,
    collection,
    connected,
    connector,
    document,
    everything,
    mock_api,
    page,
    problem,
    structured,
    text_of,
)

from ragforge_client.client import ApiError
from ragforge_contracts.api import Problem
from ragforge_mcp.errors import api_error_message


def _query_response(n: int) -> dict[str, Any]:
    hits = []
    for i in range(n):
        c = cite(None, f"Page {i}").model_dump(mode="json")
        c["snippet"] = f"passage {i}\x1b[31m"
        hits.append({"citation": c, "score": 0.5, "rerank_score": 1.0 - i / 10})
    return {"hits": hits, "trace_id": "t"}


# ----------------------------------------------------------------- search
async def test_search_returns_ranked_passages_and_sends_k() -> None:
    with mock_api() as api:
        query = api.post("/v1/query").mock(
            return_value=httpx.Response(200, json=_query_response(3))
        )
        async with connected() as client:
            out = structured(await client.call_tool("search", {"query": "rebalance", "k": 3}))
    assert [h["rank"] for h in out["hits"]] == [1, 2, 3]
    assert out["hits"][0]["title"] == "Page 0"
    assert out["hits"][0]["snippet"] == "passage 0[31m"  # terminal escapes stripped
    assert out["hits"][1]["rerank_score"] == pytest.approx(0.9)
    sent = json.loads(query.calls.last.request.content)
    assert sent == {"query": "rebalance", "collection_ids": [str(COLL_A)], "k": 3}


@pytest.mark.parametrize("k", [0, 21])
async def test_search_k_is_bounded_before_any_api_call(k: int) -> None:
    with mock_api() as api:
        query = api.post("/v1/query")
        async with connected() as client:
            result = await client.call_tool("search", {"query": "x", "k": k})
    assert result.is_error
    assert not query.called


# ----------------------------------------------------------------- collection scope
async def test_configured_collections_resolve_by_name_or_id_once() -> None:
    visible = [collection(COLL_A, "notes"), collection(COLL_B, "work")]
    with mock_api(visible) as api:
        query = api.post("/v1/query").mock(
            return_value=httpx.Response(200, json=_query_response(1))
        )
        async with connected(collections=["work", str(COLL_A)]) as client:
            structured(await client.call_tool("search", {"query": "x"}))
            structured(await client.call_tool("search", {"query": "y"}))
        assert api["collections"].call_count == 1  # cached after the first use
    sent = json.loads(query.calls.last.request.content)["collection_ids"]
    assert sent == [str(COLL_B), str(COLL_A)]


async def test_unknown_configured_collection_is_a_clear_error() -> None:
    with mock_api() as api:
        query = api.post("/v1/query")
        async with connected(collections=["nope"]) as client:
            result = await client.call_tool("search", {"query": "x"})
    assert result.is_error
    assert "Unknown collection(s) in the server config: nope" in text_of(result)
    assert "Visible: notes" in text_of(result)
    assert not query.called


async def test_key_without_collections_says_what_to_do() -> None:
    with mock_api(collections=[]):
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "x"})
    assert result.is_error
    assert "sees no collections" in text_of(result)


async def test_default_scope_is_capped_at_twenty_collections() -> None:
    many = [collection(uuid4(), f"c{i}") for i in range(25)]
    with mock_api(many) as api:
        query = api.post("/v1/query").mock(
            return_value=httpx.Response(200, json=_query_response(0))
        )
        async with connected() as client:
            structured(await client.call_tool("search", {"query": "x"}))
    sent = json.loads(query.calls.last.request.content)["collection_ids"]
    assert sent == [c["id"] for c in many[:20]]


# ----------------------------------------------------------------- list_sources
async def test_list_sources_walks_pages_and_marks_scope() -> None:
    n1, n2 = uuid4(), uuid4()
    first = page([collection(COLL_A, "notes")], next_cursor="p2")
    second = page([collection(COLL_B, "work")])
    with mock_api() as api:
        api["collections"].mock(
            side_effect=lambda request: httpx.Response(
                200, json=second if request.url.params.get("cursor") == "p2" else first
            )
        )
        api.get("/v1/connectors").mock(
            return_value=httpx.Response(
                200,
                json=page(
                    [connector(n1, "Notion", status="error"), connector(n2, "Old", coll=COLL_B)]
                ),
            )
        )
        async with connected(collections=["work"]) as client:
            out = structured(await client.call_tool("list_sources", {}))
    assert [(c["name"], c["in_scope"]) for c in out["collections"]] == [
        ("notes", False),
        ("work", True),
    ]
    assert [(c["display_name"], c["status"], c["collection_name"]) for c in out["connectors"]] == [
        ("Notion", "error", "notes"),
        ("Old", "active", "work"),
    ]


async def test_list_sources_still_works_when_the_configured_scope_is_broken() -> None:
    """Review r1 P1: the diagnostic tool failed exactly when it was needed. An unknown
    --collection is reported, and the visible collections and connectors are still listed."""
    cid = uuid4()
    with mock_api() as api:
        api.get("/v1/connectors").mock(
            return_value=httpx.Response(200, json=page([connector(cid, "Notion")]))
        )
        async with connected(collections=["typo"]) as client:
            out = structured(await client.call_tool("list_sources", {}))
            search = await client.call_tool("search", {"query": "x"})
    assert [(c["name"], c["in_scope"]) for c in out["collections"]] == [("notes", False)]
    assert [c["display_name"] for c in out["connectors"]] == ["Notion"]
    assert "Unknown collection(s) in the server config: typo" in out["config_warning"]
    assert search.is_error  # ask/search still refuse to run on a broken scope


async def test_list_sources_on_an_empty_tenant_returns_empty_lists_and_says_why() -> None:
    with mock_api(collections=[]) as api:
        api.get("/v1/connectors").mock(return_value=httpx.Response(200, json=page([])))
        async with connected() as client:
            out = structured(await client.call_tool("list_sources", {}))
    assert (out["collections"], out["connectors"]) == ([], [])
    assert "sees no collections" in out["config_warning"]


async def test_list_sources_has_no_warning_when_the_scope_is_fine() -> None:
    with mock_api() as api:
        api.get("/v1/connectors").mock(return_value=httpx.Response(200, json=page([])))
        async with connected() as client:
            out = structured(await client.call_tool("list_sources", {}))
    assert out["config_warning"] is None
    assert out["collections"][0]["in_scope"] is True


# ----------------------------------------------------------------- sync_now
async def test_sync_now_defaults_to_every_notion_connector_and_skips_deleting() -> None:
    a, b, gone = uuid4(), uuid4(), uuid4()
    other_kind = uuid4()
    job_a, job_b = uuid4(), uuid4()
    connectors = [
        connector(a, "Work Notion"),
        connector(b, "Home Notion"),
        connector(gone, "Gone", status="deleting"),
        connector(other_kind, "Drive", kind="gdrive"),
    ]
    with mock_api() as api:
        api.get("/v1/connectors").mock(return_value=httpx.Response(200, json=page(connectors)))
        sync_a = api.post(f"/v1/connectors/{a}/sync").mock(
            return_value=httpx.Response(202, json={"job_id": str(job_a)})
        )
        api.post(f"/v1/connectors/{b}/sync").mock(
            return_value=httpx.Response(202, json={"job_id": str(job_b)})
        )
        sync_gone = api.post(f"/v1/connectors/{gone}/sync")
        sync_other = api.post(f"/v1/connectors/{other_kind}/sync")
        async with connected() as client:
            out = structured(await client.call_tool("sync_now", {}))
    assert [(s["display_name"], s["job_id"]) for s in out["started"]] == [
        ("Work Notion", str(job_a)),
        ("Home Notion", str(job_b)),
    ]
    assert out["skipped"] == [
        {
            "connector_id": str(gone),
            "display_name": "Gone",
            "reason": "the connector is being deleted",
        }
    ]
    assert sync_a.called
    assert not sync_gone.called
    assert not sync_other.called


async def test_sync_now_one_connector() -> None:
    cid, job = uuid4(), uuid4()
    with mock_api() as api:
        api.get(f"/v1/connectors/{cid}").mock(
            return_value=httpx.Response(200, json=connector(cid, "Work"))
        )
        api.post(f"/v1/connectors/{cid}/sync").mock(
            return_value=httpx.Response(202, json={"job_id": str(job)})
        )
        async with connected() as client:
            out = structured(await client.call_tool("sync_now", {"connector_id": str(cid)}))
    assert out["started"] == [
        {"connector_id": str(cid), "display_name": "Work", "job_id": str(job)}
    ]


async def test_sync_now_without_notion_connectors_and_without_write_scope() -> None:
    cid = uuid4()
    with mock_api() as api:
        listing = api.get("/v1/connectors")
        listing.mock(return_value=httpx.Response(200, json=page([])))
        async with connected() as client:
            empty = await client.call_tool("sync_now", {})
        listing.mock(return_value=httpx.Response(200, json=page([connector(cid, "N")])))
        api.post(f"/v1/connectors/{cid}/sync").mock(
            return_value=problem(403, "Forbidden", "this API key lacks the `write` scope")
        )
        async with connected() as client:
            forbidden = await client.call_tool("sync_now", {})
    assert empty.is_error
    assert "no Notion connector" in text_of(empty)
    assert forbidden.is_error
    assert "lacks the `write` scope" in text_of(forbidden)
    assert "sync_now needs the `write` scope" in text_of(forbidden)


async def test_sync_now_keeps_going_past_a_busy_connector_and_returns_started_jobs() -> None:
    """Review r2: a 409 on the first connector aborted the call, hid the job already started
    and never tried the rest."""
    busy, ok, denied = uuid4(), uuid4(), uuid4()
    job = uuid4()
    connectors = [connector(busy, "Busy"), connector(ok, "Ok"), connector(denied, "Denied")]
    with mock_api() as api:
        api.get("/v1/connectors").mock(return_value=httpx.Response(200, json=page(connectors)))
        api.post(f"/v1/connectors/{busy}/sync").mock(
            return_value=problem(409, "Conflict", "a sync is already running")
        )
        api.post(f"/v1/connectors/{ok}/sync").mock(
            return_value=httpx.Response(202, json={"job_id": str(job)})
        )
        api.post(f"/v1/connectors/{denied}/sync").mock(
            return_value=problem(403, "Forbidden", f"not yours {KEY}")
        )
        async with connected() as client:
            result = await client.call_tool("sync_now", {})
    out = structured(result)
    assert out["started"] == [{"connector_id": str(ok), "display_name": "Ok", "job_id": str(job)}]
    reasons = {s["display_name"]: s["reason"] for s in out["skipped"]}
    assert reasons["Busy"].startswith("Conflict (409): a sync is already running")
    assert "(403)" in reasons["Denied"]
    assert KEY not in everything(result)


async def test_sync_now_fails_only_when_nothing_started() -> None:
    a, b = uuid4(), uuid4()
    with mock_api() as api:
        api.get("/v1/connectors").mock(
            return_value=httpx.Response(200, json=page([connector(a, "A"), connector(b, "B")]))
        )
        api.post(f"/v1/connectors/{a}/sync").mock(return_value=problem(409, "Conflict", "busy"))
        sync_b = api.post(f"/v1/connectors/{b}/sync")
        sync_b.mock(return_value=problem(409, "Conflict", "busy"))
        async with connected() as client:
            same = await client.call_tool("sync_now", {})
        sync_b.mock(return_value=problem(503, "Unavailable", "queue down"))
        async with connected() as client:
            mixed = await client.call_tool("sync_now", {})
    assert same.is_error  # one clear error instead of two identical skips
    assert "Conflict (409): busy" in text_of(same)
    out = structured(mixed)  # different reasons: reported per connector
    assert out["started"] == []
    assert [s["display_name"] for s in out["skipped"]] == ["A", "B"]


# ----------------------------------------------------------------- resource
async def test_document_resource_finds_the_document_across_pages() -> None:
    wanted, other = uuid4(), uuid4()
    first = page([document(other, "Other")], next_cursor="p2")
    second = page([document(wanted, "Kafka notes")])
    with mock_api() as api:
        api.get("/v1/documents").mock(
            side_effect=lambda request: httpx.Response(
                200, json=second if request.url.params.get("cursor") == "p2" else first
            )
        )
        async with connected() as client:
            templates = await client.list_resource_templates()
            result = await client.read_resource(f"ragforge://documents/{wanted}")
    assert [t.uri_template for t in templates.resource_templates] == ["ragforge://documents/{id}"]
    content = result.contents[0]
    assert content.mime_type == "application/json"
    body = json.loads(getattr(content, "text", ""))
    assert body["id"] == str(wanted)
    assert body["title"] == "Kafka notes"
    assert body["chunks"] is None  # the API exposes no chunk text: said, not faked
    assert "does not expose chunk text" in body["note"]


async def test_document_resource_reads_only_the_configured_collections() -> None:
    """Review r2: ask/search honoured --collection but the resource returned any document."""
    inside, outside = uuid4(), uuid4()
    collections = [collection(COLL_A, "work"), collection(COLL_B, "private")]
    with mock_api(collections) as api:
        docs = api.get("/v1/documents").mock(
            side_effect=lambda request: httpx.Response(
                200,
                json=page(
                    [document(inside, "Work page", COLL_A)]
                    if request.url.params.get("collection_id") == str(COLL_A)
                    else [document(outside, "Diary", COLL_B)]
                ),
            )
        )
        async with connected(collections=["work"]) as client:
            found = await client.read_resource(f"ragforge://documents/{inside}")
            with pytest.raises(MCPError, match=r"no document .* is visible"):
                await client.read_resource(f"ragforge://documents/{outside}")
        async with connected(collections=["typo"]) as client:
            with pytest.raises(MCPError, match="Unknown collection"):
                await client.read_resource(f"ragforge://documents/{inside}")
    assert json.loads(getattr(found.contents[0], "text", ""))["title"] == "Work page"
    asked = {call.request.url.params.get("collection_id") for call in docs.calls}
    assert asked == {str(COLL_A)}  # the out-of-scope collection was never listed


async def test_document_resource_not_found_bad_id_and_api_errors() -> None:
    with mock_api() as api:
        docs = api.get("/v1/documents")
        docs.mock(return_value=httpx.Response(200, json=page([])))
        async with connected() as client:
            with pytest.raises(MCPError, match=r"no document .* is visible"):
                await client.read_resource(f"ragforge://documents/{uuid4()}")
            with pytest.raises(MCPError, match="not a document id"):
                await client.read_resource("ragforge://documents/not-a-uuid")
            docs.mock(return_value=problem(401, "Unauthorized"))
            with pytest.raises(MCPError, match="rejected the API key"):
                await client.read_resource(f"ragforge://documents/{uuid4()}")


# ----------------------------------------------------------------- error mapping
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (problem(401, "Unauthorized", "invalid API key"), "rejected the API key (401)"),
        (
            problem(403, "Forbidden", "this API key lacks the `read` scope"),
            "lacks the `read` scope",
        ),
        (problem(429, "Too many requests"), "rate limiting this key (429): retry shortly"),
        (problem(503, "Service busy", "try later"), "busy or not ready (503): try later"),
        (problem(409, "Conflict", "turn in progress"), "Conflict (409): turn in progress"),
        (problem(422, "Validation failed", "invalid fields: k"), "Invalid request (422)"),
        (
            problem(500, "Internal error", request_id="req-42"),
            "error 500: Internal error (request id req-42)",
        ),
        (httpx.Response(502, text="<html>bad gateway</html>"), "RAGForge API error 502"),
    ],
)
async def test_api_errors_become_actionable_tool_errors(
    response: httpx.Response, expected: str
) -> None:
    with mock_api() as api:
        api.post("/v1/query").mock(return_value=response)
        async with connected() as client:
            result = await client.call_tool("search", {"query": "x"})
    assert result.is_error
    assert expected in text_of(result)


async def test_unreachable_and_timeout() -> None:
    with mock_api() as api:
        route = api.post("/v1/query")
        route.mock(side_effect=httpx.ConnectError("refused"))
        async with connected() as client:
            down = await client.call_tool("search", {"query": "x"})
            route.mock(side_effect=httpx.ReadTimeout("slow"))
            slow = await client.call_tool("search", {"query": "x"})
    assert f"unreachable at {API}" in text_of(down)
    assert "make up" in text_of(down)
    assert "timed out" in text_of(slow)


def test_retry_after_is_used_when_the_sdk_provides_it() -> None:
    exc = ApiError(Problem(title="Too many requests", status=429))
    exc.retry_after = 7  # type: ignore[attr-defined]  # a newer SDK sets it (handoff, SDK gap 1)
    assert "retry after 7 s" in api_error_message(exc)


async def test_unexpected_bugs_do_not_leak_internal_text() -> None:
    with mock_api() as api:
        api.post("/v1/query").mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))
        async with connected() as client:
            result = await client.call_tool("search", {"query": "x"})
    assert result.is_error
    assert text_of(result) == "Error executing tool search"
