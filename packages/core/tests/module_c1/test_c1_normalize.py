"""DESIGN 7.1 normalization, proven against recorded Notion responses (respx) and pure trees."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from module_c1.c1_support import (
    BASE,
    CHILD_PAGE_ID,
    HANDBOOK_ID,
    LINKED_PAGE_ID,
    MENTIONED_PAGE_ID,
    SYNCED_ORIGINAL_ID,
    TOKEN,
    FakeLimiter,
    bid,
    load,
    serve_page,
)
from ragforge_core.adapters.connectors.notion import NotionClient, NotionConnector, deep_link
from ragforge_core.adapters.connectors.notion.ids import canonical_id
from ragforge_core.adapters.connectors.notion.normalize import (
    TABLE_GROUP_MAX_ROWS,
    BlockNode,
    normalize_blocks,
)
from ragforge_core.adapters.connectors.notion.richtext import (
    LinkCollector,
    notion_page_id_from_href,
    rich_text_md,
)
from ragforge_core.domain.models import NormalizedBlock, SourceDocument, SourceItemRef, SourceKind

HANDBOOK_URL = "https://www.notion.so/Engineering-Handbook-598337872cf94fdf8782e53db20768a5"


def _hex(block_id: str) -> str:
    return block_id.replace("-", "")


@pytest.fixture
async def handbook() -> AsyncIterator[SourceDocument]:
    with respx.mock(base_url=BASE, assert_all_mocked=True) as router:
        serve_page(router, "handbook_page.json", "handbook_blocks.json")
        async with NotionClient(TOKEN, FakeLimiter()) as client:
            ref = SourceItemRef(SourceKind.notion, HANDBOOK_ID, None)
            yield await NotionConnector(client).fetch(ref)


def _by_anchor(doc: SourceDocument) -> dict[str, NormalizedBlock]:
    return {b.anchor: b for b in doc.blocks}


async def test_every_block_type_in_the_design_table_is_normalized(
    handbook: SourceDocument,
) -> None:
    kinds = {b.kind for b in handbook.blocks}
    assert kinds == {
        "heading_1",
        "heading_2",
        "heading_3",
        "paragraph",
        "list",
        "quote",
        "callout",
        "code",
        "equation",
        "table",
        "media",
    }


async def test_exact_markdown_in_document_order(handbook: SourceDocument) -> None:
    got = [(b.anchor, b.text) for b in handbook.blocks]
    assert got == [
        (_hex(bid(1)), "# Engineering Handbook"),
        (
            _hex(bid(2)),
            "We value **clarity** and *speed* `make check` see [the RFC](https://example.com/rfc)"
            " or Onboarding where $a^2 + b^2 = c^2$ (owner @Ada Lovelace)",
        ),
        (_hex(bid(3)), "## Deploys"),
        (_hex(bid(301)), "Deploy on Tuesdays."),  # toggleable heading expanded
        (_hex(bid(4)), "- First bullet"),
        (_hex(bid(401)), "  - Nested bullet"),
        (_hex(bid(5)), "1. Step one"),
        (_hex(bid(6)), "2. Step two"),
        (_hex(bid(7)), "- [x] Write tests"),
        (_hex(bid(8)), "- [ ] Ship it"),
        (_hex(bid(9)), "More details"),  # toggle summary ...
        (_hex(bid(901)), "Hidden text"),  # ... and its children, expanded inline
        (_hex(bid(902)), "Deeper"),  # nested toggle
        (_hex(bid(903)), "Deepest text"),
        (_hex(bid(10)), "> Simplicity is prerequisite for reliability."),
        (_hex(bid(11)), "> 💡 Remember to rotate keys."),
        (_hex(bid(12)), "```python\ndef hello():\n    return 'hi'\n```"),
        (_hex(bid(13)), "$$\nE = mc^2\n$$"),
        (_hex(bid(14)), "### Data"),
        (
            _hex(bid(15)),
            "| Name | Role |\n| --- | --- |\n| Ada | Engineer \\| Lead |\n| Linus |  |",
        ),
        ("5e000000000040008000000000000101", "Shared security policy text."),  # resolved
        (_hex(bid(1701)), "Original synced content."),
        (_hex(bid(20)), "[image] Architecture diagram — arch.png"),  # name from URL path
        (_hex(bid(21)), "[file] spec.pdf"),
        (_hex(bid(22)), "[bookmark] <https://example.com/postmortems>"),
        (_hex(bid(2411)), "Left column"),  # column_list / column are transparent
        (_hex(bid(2421)), "Right column"),
        (_hex(bid(28)), "[video] Deploy demo"),
    ]


async def test_headings_carry_levels_and_code_and_tables_are_never_split(
    handbook: SourceDocument,
) -> None:
    blocks = _by_anchor(handbook)
    assert [blocks[_hex(bid(n))].heading_level for n in (1, 3, 14)] == [1, 2, 3]
    assert all(b.heading_level is None for b in handbook.blocks if not b.kind.startswith("head"))
    code = blocks[_hex(bid(12))]
    assert (code.kind, code.language, code.never_split) == ("code", "python", True)
    assert blocks[_hex(bid(15))].never_split
    assert [b.anchor for b in handbook.blocks if b.never_split] == [_hex(bid(12)), _hex(bid(15))]


async def test_anchors_are_dashless_block_ids_and_deep_links_land_on_the_block(
    handbook: SourceDocument,
) -> None:
    assert handbook.external_url == HANDBOOK_URL
    for block in handbook.blocks:
        assert len(block.anchor) == 32
        assert "-" not in block.anchor
        assert int(block.anchor, 16) >= 0
    table = _by_anchor(handbook)[_hex(bid(15))]
    assert deep_link(HANDBOOK_URL, table.anchor) == (
        f"{HANDBOOK_URL}#b1000000000040008000000000000015"
    )
    assert deep_link(f"{HANDBOOK_URL}#old", bid(15)) == deep_link(HANDBOOK_URL, bid(15))


async def test_links_synced_sources_and_unsupported_counter(handbook: SourceDocument) -> None:
    # page mention (text kept, edge recorded), child_page, link_to_page: in document order
    assert handbook.links_to == (MENTIONED_PAGE_ID, CHILD_PAGE_ID, LINKED_PAGE_ID)
    assert handbook.synced_source_ids == (SYNCED_ORIGINAL_ID,)  # the reference, not the original
    assert handbook.metadata["unsupported_blocks"] == {"unsupported": 1, "meeting_notes": 1}
    assert "truncated_depth" not in handbook.metadata


async def test_no_signed_urls_unsafe_links_or_title_duplication(handbook: SourceDocument) -> None:
    everything = "\n".join(b.text for b in handbook.blocks)
    assert "SIGNEDSECRET" not in everything
    assert "amazonaws" not in everything
    assert "javascript:" not in everything
    assert handbook.title == "Engineering Handbook"
    assert handbook.fetched_at == datetime(2026, 10, 16, 11, 59, tzinfo=UTC)
    assert handbook.ref == SourceItemRef(
        SourceKind.notion, HANDBOOK_ID, datetime(2026, 10, 16, 11, 58, tzinfo=UTC)
    )


# ---------------------------------------------------------------------- pure trees
def _rt(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}, "annotations": {}, "plain_text": text}]


def _block(n: int, kind: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"object": "block", "id": bid(n), "type": kind, kind: body or {}}


def _row(n: int, *cells: str) -> BlockNode:
    return BlockNode(_block(n, "table_row", {"cells": [_rt(c) for c in cells]}))


def test_large_tables_split_into_row_groups_with_the_header_repeated() -> None:
    rows = [_row(1000 + i, f"name {i}", f"value {i}") for i in range(45)]
    table = BlockNode(
        _block(7, "table", {"table_width": 2, "has_column_header": True}),
        children=(_row(999, "Key", "Value"), *rows),
    )
    page = normalize_blocks([table], LinkCollector())
    groups = page.blocks
    assert [len(g.text.splitlines()) - 2 for g in groups] == [TABLE_GROUP_MAX_ROWS, 20, 5]
    for group in groups:
        assert group.text.startswith("| Key | Value |\n| --- | --- |\n")
        assert (group.kind, group.never_split) == ("table", True)
    # Group 1 deep-links to the table; later groups to their own first row (rows 20 and 40).
    assert [g.anchor for g in groups] == [_hex(bid(7)), _hex(bid(1020)), _hex(bid(1040))]
    body = [line for g in groups for line in g.text.splitlines()[2:]]
    assert body == [f"| name {i} | value {i} |" for i in range(45)]  # nothing lost or duplicated


def test_row_groups_are_also_bounded_by_size() -> None:
    wide = "x" * 400
    rows = [_row(1000 + i, wide) for i in range(10)]
    table = BlockNode(_block(7, "table", {"table_width": 1}), children=tuple(rows))
    groups = normalize_blocks([table], LinkCollector()).blocks
    assert len(groups) > 1
    assert all(len(g.text) <= 1500 + 410 for g in groups)
    assert all(g.text.startswith("|  |\n| --- |\n") for g in groups)  # headerless: blank header


def test_numbered_lists_restart_after_an_interruption_and_honor_start_index() -> None:
    nodes = [
        BlockNode(_block(1, "numbered_list_item", {"rich_text": _rt("a")})),
        BlockNode(_block(2, "numbered_list_item", {"rich_text": _rt("b")})),
        BlockNode(_block(3, "paragraph", {"rich_text": _rt("break")})),
        BlockNode(_block(4, "numbered_list_item", {"rich_text": _rt("c"), "list_start_index": 7})),
        BlockNode(_block(5, "numbered_list_item", {"rich_text": _rt("d")})),
    ]
    texts = [b.text for b in normalize_blocks(nodes, LinkCollector()).blocks]
    assert texts == ["1. a", "2. b", "break", "7. c", "8. d"]


def test_code_containing_fences_and_plain_text_language() -> None:
    node = BlockNode(_block(1, "code", {"rich_text": _rt("```\nx\n```"), "language": "plain text"}))
    (block,) = normalize_blocks([node], LinkCollector()).blocks
    assert block.text == "````\n```\nx\n```\n````"
    assert block.language == "plain text"


def test_empty_paragraphs_and_contentless_media_are_dropped() -> None:
    nodes = [
        BlockNode(_block(1, "paragraph", {"rich_text": []})),
        BlockNode(_block(2, "image", {"caption": [], "type": "file", "file": {"url": "s3"}})),
        BlockNode(_block(3, "equation", {"expression": "  "})),
    ]
    assert normalize_blocks(nodes, LinkCollector()).blocks == ()


def test_uncaptioned_media_keeps_a_file_name_from_the_url_path_never_the_query() -> None:
    signed = (
        "https://prod-files-secure.s3.us-west-2.amazonaws.com/w/b/Q3%20plan.png?X-Amz-Signature=S"
    )
    nodes = [
        BlockNode(_block(1, "image", {"caption": [], "type": "file", "file": {"url": signed}})),
        BlockNode(
            _block(
                2,
                "video",
                {
                    "caption": [],
                    "type": "external",
                    "external": {"url": "https://www.youtube.com/watch?v=abc"},
                },
            )
        ),
        BlockNode(
            _block(
                3,
                "pdf",
                {"caption": [], "type": "external", "external": {"url": "https://x.test/a/"}},
            )
        ),
    ]
    blocks = normalize_blocks(nodes, LinkCollector()).blocks
    assert [b.text for b in blocks] == ["[image] Q3 plan.png"]  # no page-like "watch", no "a"
    assert "Signature" not in blocks[0].text
    assert "amazonaws" not in blocks[0].text


def test_link_text_and_hrefs_cannot_break_out_of_the_generated_link() -> None:
    """Best-effort hygiene (the output sanitizer owns URL safety): a crafted href cannot close
    the link and open a second `javascript:` one, and brackets in link text stay literal."""
    item = {
        "type": "text",
        "plain_text": "see [docs]",
        "annotations": {},
        "href": "https://a.test/x) [y](javascript:alert(1)",
    }
    text = rich_text_md([item], LinkCollector())
    assert text == "[see \\[docs\\]](https://a.test/x%29%20[y]%28javascript:alert%281%29)"
    assert "](javascript:" not in text


# ---------------------------------------------------------------------- depth limit
def _toggle_chain(depth: int) -> dict[str, list[dict[str, Any]]]:
    """Page -> t1 -> t2 -> ... -> t{depth}: each toggle has one child toggle."""
    listings: dict[str, list[dict[str, Any]]] = {}
    parent = HANDBOOK_ID
    for level in range(1, depth + 1):
        block = _block(5000 + level, "toggle", {"rich_text": _rt(f"level {level}")})
        block["has_children"] = True
        listings[parent] = [{"object": "list", "results": [block], "has_more": False}]
        parent = bid(5000 + level)
    return listings


@pytest.mark.parametrize("max_depth", [10, 3])
async def test_recursion_stops_at_the_depth_limit(max_depth: int) -> None:
    listings = _toggle_chain(12)
    with respx.mock(base_url=BASE, assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"/pages/{HANDBOOK_ID}").mock(
            return_value=httpx.Response(
                200,
                json=load("handbook_page.json"),
                headers={"Date": "Fri, 16 Oct 2026 12:00:00 GMT"},
            )
        )
        routes = {
            parent: router.get(f"/blocks/{parent}/children").mock(
                return_value=httpx.Response(200, json=pages[0])
            )
            for parent, pages in listings.items()
        }
        async with NotionClient(TOKEN, FakeLimiter()) as client:
            connector = NotionConnector(client, max_depth=max_depth)
            doc = await connector.fetch(SourceItemRef(SourceKind.notion, HANDBOOK_ID, None))
    assert [b.text for b in doc.blocks] == [f"level {i}" for i in range(1, max_depth + 1)]
    fetched = [parent for parent, route in routes.items() if route.called]
    assert len(fetched) == max_depth  # page + t1..t{max_depth-1}; t{max_depth} not expanded
    assert doc.metadata["truncated_depth"] == max_depth


def test_connector_rejects_depth_above_the_design_limit() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        NotionConnector(NotionClient(TOKEN, FakeLimiter()), max_depth=11)


# ---------------------------------------------------------------------- inline Notion links
@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (f"https://www.notion.so/Roadmap-{_hex(LINKED_PAGE_ID)}", LINKED_PAGE_ID),
        (
            f"https://www.notion.so/acme/Roadmap-{_hex(LINKED_PAGE_ID)}#{_hex(bid(3))}",
            LINKED_PAGE_ID,
        ),
        (f"https://acme.notion.site/{_hex(LINKED_PAGE_ID)}?pvs=4", LINKED_PAGE_ID),
        (f"/{_hex(LINKED_PAGE_ID)}", LINKED_PAGE_ID),  # Notion's relative workspace link
        (f"https://notion.so/{LINKED_PAGE_ID}", LINKED_PAGE_ID),
        # a database row opened as a peek: ?p= names the page, the path names the database
        (f"https://www.notion.so/{_hex(bid(1))}?v=abc&p={_hex(LINKED_PAGE_ID)}", LINKED_PAGE_ID),
        (f"https://example.com/Roadmap-{_hex(LINKED_PAGE_ID)}", None),  # not Notion
        (f"https://notion.so.evil.example/{_hex(LINKED_PAGE_ID)}", None),
        (f"javascript://notion.so/{_hex(LINKED_PAGE_ID)}", None),
        ("https://www.notion.so/product", None),
        (f"Roadmap-{_hex(LINKED_PAGE_ID)}", None),  # relative without a leading slash
    ],
)
def test_notion_page_id_from_href(href: str, expected: str | None) -> None:
    got = notion_page_id_from_href(href)
    assert (canonical_id(got) if got else None) == expected


def test_inline_hyperlinks_to_notion_pages_become_links_to_edges() -> None:
    url = f"https://www.notion.so/Roadmap-{_hex(LINKED_PAGE_ID)}"
    items: list[dict[str, Any]] = [
        {"type": "text", "plain_text": "see ", "annotations": {}, "href": None},
        {"type": "text", "plain_text": "the roadmap", "annotations": {}, "href": url},
        {"type": "text", "plain_text": " and ", "annotations": {}, "href": None},
        {"type": "text", "plain_text": "docs", "annotations": {}, "href": "https://example.com/x"},
    ]
    links = LinkCollector()
    text = rich_text_md(items, links)
    assert text == f"see [the roadmap]({url}) and [docs](https://example.com/x)"
    assert links.ids() == (LINKED_PAGE_ID,)


def test_heading_4_is_normalized_not_dropped() -> None:
    """Regression (first real sync): Notion's newer heading_4 blocks were counted as unsupported
    and their text silently lost."""
    node = BlockNode(_block(42, "heading_4", {"rich_text": _rt("Isolation levels")}))
    page = normalize_blocks([node], LinkCollector())
    assert page.unsupported == {}
    [block] = page.blocks
    assert (block.kind, block.heading_level, block.text) == (
        "heading_4",
        4,
        "#### Isolation levels",
    )
