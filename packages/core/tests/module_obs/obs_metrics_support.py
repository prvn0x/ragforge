"""Module OBS-METRICS test support: reading the process-wide metrics registry as DELTAS.

The metrics are module globals (defined once at import, ADR-0018 R2-6), so a test never assumes
a value starts at 0: it snapshots the samples it cares about, runs the code path, and asserts on
the difference. `Delta` does that for every sample of one or more metric families at once.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field

from prometheus_client.samples import Sample

from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    TextDelta,
    ToolSpec,
    UsageReport,
)
from ragforge_core.observability.metrics import REGISTRY

SampleKey = tuple[str, tuple[tuple[str, str], ...]]


def samples(names: Iterable[str] = ()) -> dict[SampleKey, float]:
    """Every sample in the registry (or of the given families), keyed by (name, labels)."""
    wanted = set(names)
    out: dict[SampleKey, float] = {}
    for family in REGISTRY.collect():
        if wanted and family.name not in wanted:
            continue
        for s in family.samples:
            out[_key(s)] = s.value
    return out


def _key(s: Sample) -> SampleKey:
    return (s.name, tuple(sorted(s.labels.items())))


@dataclass
class Delta:
    """`Delta(families)` snapshots; `.changes()` returns {(sample, labels): increase} for every
    sample that moved since (new series included), ignoring `_created` timestamps."""

    families: Sequence[str]
    before: dict[SampleKey, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.before = samples(_family_names(self.families))

    def changes(self) -> dict[SampleKey, float]:
        now = samples(_family_names(self.families))
        return {
            key: value - self.before.get(key, 0.0)
            for key, value in now.items()
            if not key[0].endswith("_created") and value != self.before.get(key, 0.0)
        }

    def of(self, sample: str, **labels: str) -> float:
        """The increase of exactly one sample (0.0 when it did not move)."""
        return self.changes().get((sample, tuple(sorted(labels.items()))), 0.0)

    def labels_of(self, sample: str) -> list[dict[str, str]]:
        """The label sets of every series of `sample` that moved."""
        return [dict(labels) for (name, labels) in self.changes() if name == sample]


def _family_names(names: Sequence[str]) -> list[str]:
    """prometheus_client names a counter family without its `_total` suffix."""
    return [n.removesuffix("_total") for n in names]


# ----------------------------------------------------------------- a minimal scripted model
class ScriptedChat:
    """A ChatModel that streams `text` then a usage report (or raises `error` first)."""

    provider = "ollama"
    model = "llama3.2:3b"

    def __init__(
        self,
        text: str = "hello",
        *,
        error: BaseException | None = None,
        prompt_tokens: int = 11,
        completion_tokens: int = 4,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.usage = ChatTurnUsage(self.provider, self.model, prompt_tokens, completion_tokens)
        self.payload = payload or {"ok": True}
        self.closed = 0

    def stream(
        self, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[ChatStreamEvent]:
        try:
            if self.error is not None:
                raise self.error
            yield TextDelta(self.text)
            yield UsageReport(self.usage)
        finally:
            self.closed += 1

    async def complete_json(
        self, system: str, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        if self.error is not None:
            raise self.error
        return dict(self.payload), self.usage
