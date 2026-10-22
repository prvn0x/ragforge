"""Shared fixtures-as-functions for module EV tests: a tiny question set and run-file writers.

Question ids and text are synthetic; nothing here comes from the owner's private set.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ragforge_client.evalrun import ABSTAIN_PREFIX, Outcome
from ragforge_contracts.api import Citation
from ragforge_contracts.sse import ChatEvent

QUESTIONS = """
[[q]]
id = "dev-eos"
split = "dev"
question = "Idempotent producer vs exactly-once?"
pages = ["Kafka Delivery"]
facts = [
  { fact = "retries", match = ["retries"] },
  { fact = "transactions", match = ["transaction"] },
  { fact = "read committed", match = ["read_committed"] },
]

[[q]]
id = "dev-lru"
split = "dev"
question = "How does an LRU cache evict?"
pages = ["Caching"]
facts = [
  { fact = "least recently used", match = ["least recently"] },
  { fact = "hash map + list", match = ["linked list"] },
]

[[q]]
id = "secret-heldout-raft"
split = "heldout"
question = "SECRET heldout question text"
pages = ["Consensus"]
facts = [
  { fact = "log compaction", match = ["snapshot"] },
]

[[q]]
id = "dev-off"
split = "dev"
question = "Sourdough hydration?"
answerable = false
"""

GOOD_EOS = "Retries are deduplicated; EOS adds a transaction with read_committed consumers."
GOOD_LRU = "Evicts the least recently used entry using a hash map plus a linked list."
GOOD_RAFT = "A snapshot replaces the compacted log prefix."


def write_questions(tmp_path: Path) -> Path:
    path = tmp_path / "questions.toml"
    path.write_text(QUESTIONS, encoding="utf-8")
    return path


def write_run(
    path: Path,
    outcomes: Sequence[Outcome],
    *,
    label: str = "arm",
    split: str = "all",
    notes: str = "",
    footer: dict[str, Any] | None = None,
    questions: dict[str, str] | None = None,
) -> Path:
    """`questions`: the header asked set (id -> question_digest); None = legacy header."""
    header: dict[str, Any] = {"label": label, "split": split, "started": "t", "url": "u"}
    header["notes"] = notes
    if questions is not None:
        header["questions"] = questions
    rows: list[dict[str, Any]] = [{"header": header}, *(asdict(o) for o in outcomes)]
    if footer is not None:
        rows.append({"footer": footer})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def outcome(
    qid: str,
    final: str,
    *,
    model: Sequence[str] = (),
    inferred: Sequence[str] = (),
    total_s: float = 1.0,
    error: str | None = None,
) -> Outcome:
    return Outcome(
        id=qid,
        draft=final,
        final=final,
        cited_titles=[*model, *inferred],
        inferred_titles=list(inferred),
        models=["m:1b"],
        providers=["ollama"],
        first_token_s=0.5,
        total_s=total_s,
        error=error,
    )


def perfect_run_outcomes(total_s: float = 1.0) -> list[Outcome]:
    return [
        outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"], total_s=total_s),
        outcome("dev-lru", GOOD_LRU, model=["Caching"], total_s=total_s),
        outcome("secret-heldout-raft", GOOD_RAFT, model=["Consensus"], total_s=total_s),
        outcome("dev-off", f"{ABSTAIN_PREFIX}\n- closest: none", total_s=total_s),
    ]


def citation(title: str, *, inferred: bool = False) -> Citation:
    return Citation(
        anchor="S1",
        document_id=uuid4(),
        title=title,
        heading_path=[],
        url=None,
        snippet="",
        inferred=inferred,
    )


class FakeChat:
    """Replays a fixed event list; records what was asked (never a real server)."""

    def __init__(self, events: Sequence[ChatEvent]) -> None:
        self.events = list(events)
        self.asked: list[str] = []

    async def chat(self, message: str, collection_ids: Sequence[UUID]) -> AsyncIterator[ChatEvent]:
        self.asked.append(message)
        for event in self.events:
            yield event
