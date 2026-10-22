"""Sandboxed DocumentParser: content sniffing, formats, and enforced limits."""

from __future__ import annotations

import asyncio
import io
import time
import zipfile
from typing import Any

import pytest

from ragforge_core.adapters.connectors.files.parser import (
    MIME_DOCX,
    MIME_MARKDOWN,
    MIME_PDF,
    MIME_TEXT,
    ParseLimits,
    ParseRejected,
    SandboxedDocumentParser,
    sniff_mime,
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx(body: str, *, extra: dict[str, bytes] | None = None) -> bytes:
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def para(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"


# ================================================================= sniffing
def test_sniffing_uses_content_not_extension() -> None:
    assert sniff_mime(b"%PDF-1.7 ...", "notes.md") == MIME_PDF
    assert sniff_mime(b"# Title\n\ntext", "report.pdf") == MIME_MARKDOWN
    assert sniff_mime(b"plain words only", "x.txt") == MIME_TEXT
    assert sniff_mime(b"plain words only", "x.md") == MIME_MARKDOWN
    assert sniff_mime(docx(para("x")), "x.txt") == MIME_DOCX
    assert sniff_mime(b"\x89PNG\r\n\x1a\n....", "x.md") == "image/png"
    assert sniff_mime(b"abc\x00\x01\x02def", "x.md") == "application/octet-stream"


# ================================================================= formats
async def test_markdown_structure() -> None:
    md = (
        "# Guide\n\nIntro paragraph\nstill intro.\n\n## Setup\n\n- one\n- two\n  continued\n\n"
        "```python\nx = 1\n\ny = 2\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> quoted\n"
    )
    doc = await SandboxedDocumentParser().parse("guide.md", md.encode())
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading_1", "paragraph", "heading_2", "list", "code", "table", "quote"]
    assert doc.title == "Guide"
    code = doc.blocks[4]
    assert code.never_split
    assert code.language == "python"
    assert "y = 2" in code.text
    assert doc.blocks[0].heading_level == 1
    assert len({b.anchor for b in doc.blocks}) == len(doc.blocks)
    assert doc.metadata["mime"] == MIME_MARKDOWN


async def test_plain_text_and_title_fallback() -> None:
    doc = await SandboxedDocumentParser().parse("dir/notes.txt", b"first\n\nsecond\r\n\r\nthird")
    assert [b.text for b in doc.blocks] == ["first", "second", "third"]
    assert doc.title == "notes"


async def test_docx_paragraphs_headings_and_tables() -> None:
    body = (
        para("Report", "Title")
        + para("Body text")
        + para("Section", "Heading2")
        + f"<w:tbl><w:tr><w:tc>{para('h1')}</w:tc><w:tc>{para('h2')}</w:tc></w:tr>"
        + f"<w:tr><w:tc>{para('v1')}</w:tc><w:tc>{para('v2')}</w:tc></w:tr></w:tbl>"
    )
    doc = await SandboxedDocumentParser().parse("r.docx", docx(body))
    assert [b.kind for b in doc.blocks] == ["heading_1", "paragraph", "heading_2", "table"]
    assert doc.title == "Report"
    assert "| v1 | v2 |" in doc.blocks[3].text


# ================================================================= limits
async def test_oversize_is_rejected_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_spawn(*a: Any, **k: Any) -> Any:
        raise AssertionError("must not spawn a sandbox for oversize input")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)
    parser = SandboxedDocumentParser(ParseLimits(max_bytes=10))
    with pytest.raises(ParseRejected, match="exceeds"):
        await parser.parse("a.txt", b"x" * 11)


async def test_timeout_kills_the_child_promptly() -> None:
    parser = SandboxedDocumentParser(ParseLimits(timeout_s=0.3, max_blocks=10_000_000))
    big = ("word " * 40 + "\n\n").encode() * 200_000  # ~40 MB of paragraphs
    started = time.monotonic()
    with pytest.raises(ParseRejected, match="exceeded"):
        await parser.parse("big.txt", big)
    assert time.monotonic() - started < 10


async def test_zip_bomb_is_rejected() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = b"<w:document xmlns:w='" + W.encode() + b"'><w:body>" + b" " * (20 * 1024 * 1024)
        zf.writestr("word/document.xml", payload + b"</w:body></w:document>")
    data = buf.getvalue()
    assert len(data) < 100_000  # tiny on disk, 20 MB expanded
    parser = SandboxedDocumentParser(ParseLimits(max_unzipped_bytes=1024 * 1024))
    with pytest.raises(ParseRejected, match="expands"):
        await parser.parse("bomb.docx", data)


async def test_docx_with_dtd_is_rejected() -> None:
    evil = (
        '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
        f'<w:document xmlns:w="{W}"><w:body>{para("&b;")}</w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", evil)
    with pytest.raises(ParseRejected, match="DTD"):
        await SandboxedDocumentParser().parse("evil.docx", buf.getvalue())


async def test_block_cap_is_enforced() -> None:
    parser = SandboxedDocumentParser(ParseLimits(max_blocks=10))
    with pytest.raises(ParseRejected, match="blocks"):
        await parser.parse("many.txt", b"p\n\n" * 50)


async def test_page_cap_is_enforced_for_docx() -> None:
    page_break = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
    parser = SandboxedDocumentParser(ParseLimits(max_pages=3))
    with pytest.raises(ParseRejected, match="pages"):
        await parser.parse("long.docx", docx(page_break * 5))


@pytest.mark.parametrize(
    ("name", "data"),
    [("x.pdf", b"%PDF-1.4 binary"), ("x.png", b"\x89PNG\r\n\x1a\n\x00\x00"), ("e.txt", b"")],
)
async def test_unsupported_or_empty_inputs_are_rejected(name: str, data: bytes) -> None:
    with pytest.raises(ParseRejected):
        await SandboxedDocumentParser().parse(name, data)


async def test_child_runs_isolated_with_an_empty_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen["args"], seen["kwargs"] = args, kwargs
        return await real(*args, **kwargs)

    monkeypatch.setenv("RAGFORGE_KEK", "must-not-leak")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    await SandboxedDocumentParser().parse("a.txt", b"hello")
    assert "-I" in seen["args"]
    assert set(seen["kwargs"]["env"]) == {"PATH", "LC_ALL"}
    assert seen["kwargs"]["start_new_session"] is True
