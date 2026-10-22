"""Sandboxed DocumentParser (module C2): user files are parsed in a short-lived child process.

Why a subprocess: parsers are the classic memory-corruption / resource-exhaustion surface. The
child (`_sandbox.py`) runs with an address-space and CPU rlimit, no file writes, no forking,
an EMPTY environment (no DB passwords, no KEK) and `python -I` (no user site, no PYTHON* env,
no cwd on sys.path). The parent enforces a wall-clock timeout by killing the child's whole
process group, caps input size before spawning anything, and bounds concurrency.

The file type is decided by content sniffing, never by the extension: a `.md` that is really a
PDF is treated as a PDF. The extension only breaks the tie between plain text and Markdown,
which are the same family and both safe.

Supported: text/markdown, text/plain, DOCX (stdlib zip + XML, DTDs refused, bounded
decompression). PDF needs a parser library that is not installed yet (see docs/handoff/C2.md);
PDFs are rejected as unsupported until then.

Memory limits use RLIMIT_AS, enforced on Linux (production containers); macOS ignores it, so
there the CPU limit and the parent's timeout are the backstop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import NormalizedBlock, SourceDocument, SourceItemRef, SourceKind

MIME_MARKDOWN = "text/markdown"
MIME_TEXT = "text/plain"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_PDF = "application/pdf"
MIME_ZIP = "application/zip"
MIME_BINARY = "application/octet-stream"
SUPPORTED = frozenset({MIME_MARKDOWN, MIME_TEXT, MIME_DOCX})

_SANDBOX = Path(__file__).with_name("_sandbox.py")
_MARKDOWN_EXT = (".md", ".markdown", ".mdown", ".mkd")
_BLOCK_KINDS = frozenset(
    {"heading_1", "heading_2", "heading_3", "paragraph", "list", "code", "table", "quote"}
)

# Magic numbers of common binary formats we refuse. Pure-ASCII magics ("MZ", "BZh") are left
# out on purpose: real text may start with them, and those binaries fail the text test anyway.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", MIME_PDF),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x1f\x8b", "application/gzip"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "application/vnd.rar"),
    (b"\x7fELF", "application/x-executable"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"),
)


class ParseRejected(ValidationFailed):
    """The file cannot be ingested (type, size, limits, malformed). Retrying cannot help, so
    the sync service fails the job permanently."""

    title = "File rejected"


@dataclass(frozen=True, slots=True)
class ParseLimits:
    max_bytes: int = 50 * 1024 * 1024
    timeout_s: float = 60.0
    memory_bytes: int = 1024 * 1024 * 1024
    max_pages: int = 2_000
    max_blocks: int = 50_000
    max_unzipped_bytes: int = 200 * 1024 * 1024
    max_zip_entries: int = 2_000
    max_concurrency: int = 2

    def for_child(self) -> dict[str, int]:
        values = asdict(self)
        del values["timeout_s"], values["max_concurrency"]
        values["cpu_s"] = int(self.timeout_s) + 1
        return {k: int(v) for k, v in values.items()}


# ================================================================= sniffing
def sniff_mime(data: bytes, filename: str = "") -> str:
    """Content-based type detection. The filename only picks markdown vs plain for text."""
    head = data[:8192]
    if head.startswith(b"PK\x03\x04"):
        # DOCX keeps its member names uncompressed in local headers; the child verifies.
        return MIME_DOCX if b"word/" in data[:65536] else MIME_ZIP
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    if b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return MIME_BINARY
    if not _looks_like_text(head):
        return MIME_BINARY
    if filename.lower().endswith(_MARKDOWN_EXT) or _markdown_signals(head):
        return MIME_MARKDOWN
    return MIME_TEXT


def _looks_like_text(head: bytes) -> bool:
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    controls = sum(1 for b in head if b < 32 and b not in b"\t\n\r\f")
    return controls <= max(8, len(head) // 100)


def _markdown_signals(head: bytes) -> bool:
    text = head.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    return any(
        line.startswith(("# ", "## ", "### ", "```", "~~~", "> ", "|")) for line in lines
    ) or ("](" in text and "[" in text)


# ================================================================= the parser
class SandboxedDocumentParser:
    """DocumentParser port implementation. One instance per process; thread-unsafe by design
    (asyncio only)."""

    def __init__(self, limits: ParseLimits | None = None) -> None:
        self._limits = limits or ParseLimits()
        self._slots = asyncio.Semaphore(max(1, self._limits.max_concurrency))

    async def parse(self, filename: str, data: bytes) -> SourceDocument:
        if len(data) > self._limits.max_bytes:
            raise ParseRejected(f"file exceeds {self._limits.max_bytes} bytes")
        if not data:
            raise ParseRejected("file is empty")
        mime = sniff_mime(data, filename)
        if mime not in SUPPORTED:
            raise ParseRejected(f"unsupported file type: {mime}")
        async with self._slots:
            result = await self._run_child(mime, data)
        return _to_document(filename, mime, result)

    async def _run_child(self, mime: str, data: bytes) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(_SANDBOX),
            mime,
            json.dumps(self._limits.for_child()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={"PATH": os.defpath, "LC_ALL": "C.UTF-8"},
            cwd="/",
            start_new_session=True,  # own process group: the timeout kills everything
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(data), self._limits.timeout_s)
        except TimeoutError:
            await _kill(proc)
            raise ParseRejected(f"parsing exceeded {self._limits.timeout_s:g}s") from None
        except BaseException:
            await _kill(proc)
            raise
        return _decode_result(proc.returncode, out)


async def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        await proc.wait()


def _decode_result(returncode: int | None, out: bytes) -> dict[str, Any]:
    if returncode != 0 or not out:
        # Killed by RLIMIT_CPU/AS (signal) or crashed: same outcome for the user.
        raise ParseRejected("parser terminated abnormally (resource limit or malformed file)")
    try:
        result = json.loads(out)
    except ValueError:
        raise ParseRejected("parser produced invalid output") from None
    if not isinstance(result, dict):
        raise ParseRejected("parser produced invalid output")
    if not result.get("ok"):
        raise ParseRejected(str(result.get("message") or "file rejected")[:300])
    return result


def _to_document(filename: str, mime: str, result: dict[str, Any]) -> SourceDocument:
    """Re-validate the child's output: the child processes hostile input, so its output is
    untrusted too (types, kinds and lengths are checked before they reach the index)."""
    raw_blocks = result.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ParseRejected("parser produced invalid output")
    blocks = tuple(_block(b) for b in raw_blocks)
    title = result.get("title")
    stem = Path(filename.replace("\\", "/")).stem or "Untitled"
    return SourceDocument(
        ref=SourceItemRef(SourceKind.file, filename, None),
        title=(title if isinstance(title, str) and title.strip() else stem)[:300],
        external_url=None,
        blocks=blocks,
        metadata={"mime": mime, "pages": int(result.get("pages") or 1)},
    )


def _block(raw: object) -> NormalizedBlock:
    if not isinstance(raw, dict):
        raise ParseRejected("parser produced invalid output")
    kind, text, anchor = raw.get("kind"), raw.get("text"), raw.get("anchor")
    level, language = raw.get("heading_level"), raw.get("language")
    if (
        kind not in _BLOCK_KINDS
        or not isinstance(text, str)
        or not isinstance(anchor, str)
        or not (level is None or isinstance(level, int))
        or not (language is None or isinstance(language, str))
    ):
        raise ParseRejected("parser produced invalid output")
    return NormalizedBlock(
        anchor=anchor,
        kind=str(kind),
        text=text,
        heading_level=level,
        language=language,
        never_split=bool(raw.get("never_split")),
    )
