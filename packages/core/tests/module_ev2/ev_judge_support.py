"""Shared helpers for the EV-JUDGE tests: fake providers (respx side effects that behave like a
judge model), demo-data access and synthetic v2 run rows. Only evals/demo/** and synthetic data."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import httpx

from ragforge_client.evals.schema import (
    AnswerOutput,
    DatasetRef,
    EvalKind,
    RunHeader,
    RunRow,
    ShownRef,
    Split,
    Surface,
)
from ragforge_contracts.api import ChunkKey, EvalChunk, EvalChunksResponse

ROOT = Path(__file__).resolve().parents[4]
DEMO = ROOT / "evals" / "demo"
FIXTURES = DEMO / "fixtures"
T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
NS = UUID("6f1c1d1e-0000-4000-8000-000000000000")

OLLAMA = "http://localhost:11434"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_DIGEST = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9"

_FENCE = re.compile(r"^<<<ITEM ([0-9a-f]{16})>>>$", re.MULTILINE)


def fence_nonce(user_text: str) -> str:
    """The nonce on the REAL opening fence (the first line of the rendered item)."""
    match = _FENCE.search(user_text)
    assert match is not None, "no fence in the prompt"
    return match.group(1)


def criteria_of(user_text: str) -> list[str]:
    line = next(ln for ln in user_text.splitlines() if ln.startswith("Criteria to grade: "))
    return line.removeprefix("Criteria to grade: ").split(", ")


Reply = Callable[[str], str]  # user prompt -> the model's reply text


def good_reply(score: int = 2) -> Reply:
    def reply(user: str) -> str:
        crits = {c: {"reason": "supported by N1", "score": score} for c in criteria_of(user)}
        return json.dumps({"nonce": fence_nonce(user), **crits})

    return reply


@dataclass
class FakeOllama:
    """`/api/tags` + `/api/chat`, replying through `reply` to the user message it receives."""

    reply: Reply
    model: str = "qwen3:8b"
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def tags(self, request: httpx.Request) -> httpx.Response:
        models = [{"name": self.model, "model": self.model, "digest": OLLAMA_DIGEST}]
        return httpx.Response(200, json={"models": models})

    def chat(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        user = body["messages"][1]["content"]
        message = {"role": "assistant", "content": self.reply(user)}
        return httpx.Response(200, json={"model": self.model, "message": message, "done": True})


@dataclass
class FakeGemini:
    reply: Reply
    model: str = "gemini-2.5-flash"
    served: list[str] = field(default_factory=lambda: ["gemini-2.5-flash-001"])
    requests: list[httpx.Request] = field(default_factory=list)

    def get_model(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"name": f"models/{self.model}", "version": "001"})

    def generate(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        user = body["contents"][0]["parts"][0]["text"]
        served = self.served[min(len(self.requests), len(self.served)) - 1]
        candidate = {"content": {"parts": [{"text": self.reply(user)}]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate], "modelVersion": served})


# ------------------------------------------------------------------ demo chunks + synthetic runs
@dataclass(frozen=True)
class DemoChunk:
    key: ChunkKey
    text: str
    chunk_id: UUID
    document_id: UUID


def demo_chunks() -> list[DemoChunk]:
    out: list[DemoChunk] = []
    for i, entry in enumerate(json.loads((DEMO / "chunks.json").read_text())):
        key = ChunkKey.model_validate(entry["key"])
        out.append(
            DemoChunk(
                key,
                entry["text"],
                uuid5(NS, f"chunk-{i}"),
                uuid5(NS, f"doc-{key.page.key}"),
            )
        )
    return out


def chunk_for(chunks: Sequence[DemoChunk], key: ChunkKey) -> DemoChunk:
    return next(c for c in chunks if c.key == key)


@dataclass
class FakeChunkSource:
    """`eval_chunks` over the demo chunks. `tamper` replaces a chunk's text (a note edited since
    the run); with `lie`, the fake keeps reporting the OLD digest for the new text."""

    chunks: Sequence[DemoChunk]
    tamper: dict[UUID, str] = field(default_factory=dict)
    lie: bool = False
    calls: list[list[UUID]] = field(default_factory=list)

    def _chunk(self, c: DemoChunk) -> EvalChunk:
        text = self.tamper.get(c.chunk_id, c.text)
        sha = c.key.display_sha256
        if c.chunk_id in self.tamper and not self.lie:
            sha = hashlib.sha256(text.encode()).hexdigest()
        return EvalChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            page_key=c.key.page,
            heading_path=list(c.key.heading_path),
            ordinal=0,
            display_text=text,
            display_sha256=sha,
        )

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse:
        self.calls.append(list(chunk_ids))
        wanted = set(chunk_ids)
        return EvalChunksResponse(
            chunks=[self._chunk(c) for c in self.chunks if c.chunk_id in wanted]
        )


def answer_row(
    item_id: str, answer: str, shown: Sequence[DemoChunk], *, status: str = "verified"
) -> RunRow:
    return RunRow(
        item_id=item_id,
        input_digest="0" * 64,
        shown=[
            ShownRef(
                label=i,
                chunk_id=str(c.chunk_id),
                display_sha256=c.key.display_sha256,
                chunk_key=c.key,
            )
            for i, c in enumerate(shown, start=1)
        ],
        output=AnswerOutput(
            answer=answer,
            status=status,
            cited_pages=[f"file:{c.key.page.key}" for c in shown],
            abstained=False,
        ),
    )


def header(run_id: str = "r1") -> RunHeader:
    return RunHeader(
        run_id=run_id,
        eval_kind=EvalKind.generator,
        surface=Surface.chat,
        arm="test",
        started_at=T0,
        dataset=DatasetRef(name="synthetic", digest="0" * 64, splits=[Split.dev]),
    )
