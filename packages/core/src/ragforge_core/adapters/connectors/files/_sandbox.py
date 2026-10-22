"""Parser sandbox child process. Run as a SCRIPT (`python -I _sandbox.py <mime> <limits-json>`),
never imported by the server: it must not import ragforge_core (keeps the child's attack
surface and startup small) and uses only the standard library.

Protocol: the raw file bytes arrive on stdin; one JSON object leaves on stdout:
  {"ok": true, "title": str|null, "pages": int, "blocks": [{anchor, kind, text, heading_level,
   language, never_split}, ...]}
  {"ok": false, "code": "limit"|"unsupported"|"malformed"|"memory", "message": str}

Limits are applied to THIS process before any input is read: address space (memory), CPU
seconds, no file writes (RLIMIT_FSIZE=0), no core dumps, few file descriptors, no forking.
The parent enforces the wall-clock timeout and the input size cap.
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

try:  # POSIX only; the server runs on Linux/macOS
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

Block = dict[str, Any]

MIME_MARKDOWN = "text/markdown"
MIME_TEXT = "text/plain"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class SandboxError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ================================================================= limits
def _set_limit(name: str, value: int) -> None:
    if resource is None or not hasattr(resource, name):
        return
    which = getattr(resource, name)
    try:
        _soft, hard = resource.getrlimit(which)
        cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(which, (cap, cap if hard == resource.RLIM_INFINITY else hard))
    except (ValueError, OSError):  # not supported on this OS (e.g. RLIMIT_AS on macOS)
        pass


def apply_limits(limits: dict[str, int]) -> None:
    _set_limit("RLIMIT_AS", limits["memory_bytes"])
    _set_limit("RLIMIT_DATA", limits["memory_bytes"])
    _set_limit("RLIMIT_CPU", limits["cpu_s"])
    _set_limit("RLIMIT_FSIZE", 0)
    _set_limit("RLIMIT_CORE", 0)
    _set_limit("RLIMIT_NOFILE", 64)
    _set_limit("RLIMIT_NPROC", 0)


# ================================================================= text decoding
def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            raise SandboxError("malformed", "invalid UTF-16 text") from None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    text = data.decode("latin-1")
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r\f")
    if controls > max(8, len(text) // 100):
        raise SandboxError("unsupported", "binary content is not text")
    return text


# ================================================================= block builder
class Blocks:
    """Collects blocks and enforces the block cap as it goes (fail fast, bounded memory)."""

    def __init__(self, max_blocks: int) -> None:
        self.items: list[Block] = []
        self._max = max_blocks

    def add(
        self,
        kind: str,
        text: str,
        *,
        heading_level: int | None = None,
        language: str | None = None,
        never_split: bool = False,
    ) -> None:
        text = text.strip("\n")
        if not text.strip():
            return
        if len(self.items) >= self._max:
            raise SandboxError("limit", f"document exceeds {self._max} blocks")
        self.items.append(
            {
                "anchor": f"b{len(self.items)}",
                "kind": kind,
                "text": text,
                "heading_level": heading_level,
                "language": language,
                "never_split": never_split,
            }
        )


def heading_kind(level: int) -> str:
    return f"heading_{min(max(level, 1), 3)}"


# ================================================================= markdown / plain text
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*([\w+#.-]*)")
_LIST = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+")


def _group_kind(line: str) -> str:
    if _LIST.match(line):
        return "list"
    if line.lstrip().startswith("|"):
        return "table"
    if line.lstrip().startswith(">"):
        return "quote"
    return "paragraph"


def parse_markdown(text: str, out: Blocks) -> None:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    group: list[str] = []
    group_kind = "paragraph"

    def flush() -> None:
        nonlocal group
        if group:
            out.add(group_kind, "\n".join(group))
            group = []

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker, language = fence.group(1), fence.group(2) or None
            body: list[str] = [line]
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(marker):
                body.append(lines[i])
                i += 1
            body.append(lines[i] if i < len(lines) else marker)  # close unterminated fences
            out.add("code", "\n".join(body), language=language, never_split=True)
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            out.add(heading_kind(level), line.strip(), heading_level=level)
        elif not line.strip():
            flush()
        else:
            kind = _group_kind(line)
            continuation = kind == "paragraph" and group_kind == "list" and line[:1] in " \t"
            if group and kind != group_kind and not continuation:
                flush()
            if not group:
                group_kind = kind
            group.append(line)
        i += 1
    flush()


def parse_plain(text: str, out: Blocks) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for para in re.split(r"\n[ \t]*\n", normalized):
        out.add("paragraph", para)


# ================================================================= docx (stdlib zip + xml)
def _read_member(zf: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Bounded decompression: never trust the declared size (zip bombs lie about it)."""
    with zf.open(name) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise SandboxError("limit", "compressed content expands beyond the size limit")
    return data


def _docx_xml(data: bytes, limits: dict[str, int]) -> ET.Element:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise SandboxError("malformed", "not a valid DOCX (zip) file") from None
    with zf:
        infos = zf.infolist()
        if len(infos) > limits["max_zip_entries"]:
            raise SandboxError("limit", "archive has too many entries")
        if sum(i.file_size for i in infos) > limits["max_unzipped_bytes"]:
            raise SandboxError("limit", "compressed content expands beyond the size limit")
        if "word/document.xml" not in zf.namelist():
            raise SandboxError("unsupported", "zip archive is not a DOCX document")
        try:
            xml = _read_member(zf, "word/document.xml", limits["max_unzipped_bytes"])
        except (zipfile.BadZipFile, EOFError):
            raise SandboxError("malformed", "corrupt DOCX archive") from None
    # OOXML never carries a DTD; refusing one blocks entity-expansion attacks outright.
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise SandboxError("malformed", "DOCX XML must not contain a DTD")
    try:
        return ET.fromstring(xml)  # noqa: S314 - DTD rejected above; memory-capped process
    except ET.ParseError:
        raise SandboxError("malformed", "corrupt DOCX XML") from None


def _run_text(el: ET.Element) -> str:
    parts: list[str] = []
    for node in el.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _paragraph(p: ET.Element, out: Blocks) -> int:
    """Emit one w:p; returns the number of explicit page breaks it contains."""
    style_el = p.find(f"{_W}pPr/{_W}pStyle")
    style = (style_el.get(f"{_W}val") or "") if style_el is not None else ""
    text = _run_text(p)
    breaks = sum(1 for br in p.iter(f"{_W}br") if br.get(f"{_W}type") == "page")
    match = re.fullmatch(r"(?i)heading\s*([1-9])", style)
    if style.lower() == "title" or match:
        level = int(match.group(1)) if match else 1
        out.add(heading_kind(level), f"{'#' * min(level, 6)} {text.strip()}", heading_level=level)
    elif p.find(f"{_W}pPr/{_W}numPr") is not None:
        out.add("list", f"- {text.strip()}")
    else:
        out.add("paragraph", text)
    return breaks


def _table(tbl: ET.Element, out: Blocks) -> None:
    rows: list[str] = []
    for i, tr in enumerate(tbl.iter(f"{_W}tr")):
        cells = [_run_text(tc).replace("|", "\\|").replace("\n", " ") for tc in tr.iter(f"{_W}tc")]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "---|" * len(cells))
    out.add("table", "\n".join(rows))


def parse_docx(data: bytes, limits: dict[str, int], out: Blocks) -> int:
    body = _docx_xml(data, limits).find(f"{_W}body")
    pages = 1
    if body is None:
        return pages
    for child in body:
        if child.tag == f"{_W}p":
            pages += _paragraph(child, out)
        elif child.tag == f"{_W}tbl":
            _table(child, out)
        if pages > limits["max_pages"]:
            raise SandboxError("limit", f"document exceeds {limits['max_pages']} pages")
    return pages


# ================================================================= entry point
def _title(blocks: list[Block]) -> str | None:
    for block in blocks:
        if str(block["kind"]).startswith("heading_"):
            return str(block["text"]).lstrip("#").strip() or None
    return None


def run(mime: str, data: bytes, limits: dict[str, int]) -> dict[str, Any]:
    out = Blocks(limits["max_blocks"])
    pages = 1
    if mime == MIME_DOCX:
        pages = parse_docx(data, limits, out)
    elif mime == MIME_MARKDOWN:
        parse_markdown(decode_text(data), out)
    elif mime == MIME_TEXT:
        parse_plain(decode_text(data), out)
    else:
        raise SandboxError("unsupported", f"unsupported file type: {mime}")
    return {"ok": True, "title": _title(out.items), "pages": pages, "blocks": out.items}


def main() -> int:
    mime, limits = sys.argv[1], json.loads(sys.argv[2])
    apply_limits(limits)
    try:
        data = sys.stdin.buffer.read(limits["max_bytes"] + 1)
        if len(data) > limits["max_bytes"]:
            raise SandboxError("limit", "file exceeds the size limit")
        result = run(mime, data, limits)
    except SandboxError as exc:
        result = {"ok": False, "code": exc.code, "message": exc.message}
    except MemoryError:
        result = {"ok": False, "code": "memory", "message": "parser exceeded its memory limit"}
    except RecursionError:
        result = {"ok": False, "code": "malformed", "message": "document nesting is too deep"}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=True).encode("ascii"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
