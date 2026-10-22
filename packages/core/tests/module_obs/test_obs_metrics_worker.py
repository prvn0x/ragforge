"""Worker-side and adapter metrics (ADR-0018): job outcomes and durations through the ONE
measured queue, Notion request statuses, embedding calls, and the DB pool gauges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest

from module_obs.obs_metrics_support import Delta
from ragforge_core.adapters.connectors.notion.client import NotionClient
from ragforge_core.adapters.connectors.notion.errors import NotionTransportError
from ragforge_core.adapters.llm.embedder_ollama import (
    EmbeddingFailed,
    OllamaEmbedder,
)
from ragforge_core.apps.worker import main as worker_main
from ragforge_core.apps.worker.job_metrics import MAX_ATTEMPTS, ObservedJobQueue
from ragforge_core.bootstrap import WorkerContainer
from ragforge_core.config import Settings
from ragforge_core.db.engine import make_engine
from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited
from ragforge_core.domain.models import JobLease, JobSpec
from ragforge_core.domain.ports import Tx
from ragforge_core.observability import metrics
from ragforge_core.services.sync.deps import SyncDeps

JOBS = ("ragforge_jobs_total", "ragforge_job_duration_seconds")


# ---------------------------------------------------------------------- job outcomes
class FakeQueue:
    def __init__(self, leases: Sequence[JobLease]) -> None:
        self.leases = list(leases)
        self.calls: list[str] = []

    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        self.calls.append("enqueue")
        return uuid4()

    async def claim(self, worker_id: str, n: int, *, prefer_low: bool = False) -> list[JobLease]:
        self.calls.append(f"claim:{prefer_low}")
        out, self.leases = self.leases[:n], self.leases[n:]
        return out

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        self.calls.append("assert_lease")

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        self.calls.append("report_progress")

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        self.calls.append("complete")

    async def heartbeat(self, lease: JobLease) -> bool:
        self.calls.append("heartbeat")
        return True

    async def release(self, lease: JobLease) -> None:
        self.calls.append("release")

    async def defer(self, lease: JobLease, delay_s: int) -> None:
        self.calls.append("defer")

    async def fail(
        self, lease: JobLease, error: str, retry_after_s: float | None, *, permanent: bool = False
    ) -> None:
        self.calls.append(f"fail:{permanent}")


def _lease(kind: str = "sync_page", attempts: int = 1) -> JobLease:
    return JobLease(uuid4(), uuid4(), kind, {}, lease_version=1, attempts=attempts)


async def _claimed(*leases: JobLease) -> tuple[ObservedJobQueue, FakeQueue]:
    inner = FakeQueue(leases)
    queue = ObservedJobQueue(inner)
    assert list(await queue.claim("w", len(leases), prefer_low=True)) == list(leases)
    assert inner.calls == ["claim:True"]  # keyword passed through
    return queue, inner


async def test_a_completed_job_is_done_with_its_duration_since_the_claim() -> None:
    lease = _lease()
    queue, inner = await _claimed(lease)
    delta = Delta(JOBS)
    await queue.complete(cast(Tx, None), lease)
    assert inner.calls[-1] == "complete"
    assert delta.of("ragforge_jobs_total", kind="sync_page", outcome="done") == 1
    assert delta.of("ragforge_job_duration_seconds_count", kind="sync_page") == 1


@pytest.mark.parametrize(
    ("attempts", "permanent", "outcome"),
    [(1, False, "failed"), (MAX_ATTEMPTS, False, "dead"), (1, True, "dead")],
)
async def test_a_failed_job_is_failed_or_dead(attempts: int, permanent: bool, outcome: str) -> None:
    lease = _lease(attempts=attempts)
    queue, inner = await _claimed(lease)
    delta = Delta(JOBS)
    await queue.fail(lease, "boom", None, permanent=permanent)
    assert inner.calls[-1] == f"fail:{permanent}"
    assert delta.of("ragforge_jobs_total", kind="sync_page", outcome=outcome) == 1


@pytest.mark.parametrize("method", ["release", "defer"])
async def test_a_job_given_back_is_requeued(method: str) -> None:
    lease = _lease("sync_connector")
    queue, _ = await _claimed(lease)
    delta = Delta(JOBS)
    if method == "release":
        await queue.release(lease)
    else:
        await queue.defer(lease, 60)
    assert delta.of("ragforge_jobs_total", kind="sync_connector", outcome="requeued") == 1


async def test_pass_through_calls_record_nothing() -> None:
    lease = _lease()
    queue, inner = await _claimed(lease)
    delta = Delta(JOBS)
    tx = cast(Tx, None)
    await queue.assert_lease(tx, lease)
    await queue.report_progress(tx, lease, {})
    assert await queue.heartbeat(lease) is True
    await queue.enqueue(tx, JobSpec("k", 20, {}, "k"))  # type: ignore[arg-type]
    assert inner.calls[1:] == ["assert_lease", "report_progress", "heartbeat", "enqueue"]
    assert delta.changes() == {}


async def test_a_job_kind_is_never_free_text() -> None:
    lease = _lease("Sync Page; DROP")
    queue, _ = await _claimed(lease)
    delta = Delta(JOBS)
    await queue.complete(cast(Tx, None), lease)
    assert delta.of("ragforge_jobs_total", kind="other", outcome="done") == 1


def test_the_worker_measures_the_queue_its_handlers_report_through() -> None:
    """`observe_jobs` puts the measured queue where the self-reporting sync handlers find it;
    connector claims and upkeep keep the plain adapter."""
    plain = FakeQueue([])
    sync = SyncDeps(*[cast(Any, None)] * 10)
    container = WorkerContainer(
        settings=Settings(_env_file=None),
        engine=cast(Any, None),
        tx=cast(Any, None),
        secrets=cast(Any, None),
        jobs=cast(Any, plain),
        sync=sync,
    )
    observed = worker_main.observe_jobs(container)
    assert container.sync.queue is observed
    assert cast(object, container.jobs) is plain


# ---------------------------------------------------------------------- notion
class _Limiter:
    async def acquire(self, key: str, tokens: int = 1) -> None:
        return None


async def _no_sleep(_: float) -> None:
    return None


def _notion(handler: Any, *, attempts: int = 2) -> NotionClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return NotionClient(
        "ntn_synthetic_token_for_tests_only_000", _Limiter(), http=http, sleep=_no_sleep,
        max_attempts=attempts,
    )  # fmt: skip


async def test_every_notion_attempt_is_counted_by_status() -> None:
    replies = iter(
        [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={})]
    )
    delta = Delta(("ragforge_notion_requests_total",))
    await _notion(lambda request: next(replies)).request("GET", "/users/me")
    assert delta.of("ragforge_notion_requests_total", status="429") == 1
    assert delta.of("ragforge_notion_requests_total", status="2xx") == 1


@pytest.mark.parametrize(("code", "status"), [(500, "5xx"), (404, "4xx")])
async def test_notion_errors_are_counted(code: int, status: str) -> None:
    delta = Delta(("ragforge_notion_requests_total",))
    with pytest.raises(RagforgeError):
        await _notion(lambda request: httpx.Response(code, json={})).request("GET", "/x")
    assert delta.of("ragforge_notion_requests_total", status=status) >= 1


async def test_a_notion_transport_failure_is_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic", request=request)

    delta = Delta(("ragforge_notion_requests_total",))
    with pytest.raises(NotionTransportError):
        await _notion(handler, attempts=1).request("GET", "/x")
    assert delta.of("ragforge_notion_requests_total", status="network") == 1


# ---------------------------------------------------------------------- embeddings
def _embedder(handler: Any) -> OllamaEmbedder:
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaEmbedder(http, model="nomic-embed-text", dim=2)


def _labels(outcome: str) -> dict[str, str]:
    return {"process": metrics.process(), "outcome": outcome}


async def test_an_embedding_call_is_measured() -> None:
    delta = Delta(("ragforge_embed_requests_total", "ragforge_embed_request_duration_seconds"))
    embedder = _embedder(lambda r: httpx.Response(200, json={"embeddings": [[0.6, 0.8]]}))
    assert await embedder.embed_query("q") == [0.6, 0.8]
    assert delta.of("ragforge_embed_requests_total", **_labels("ok")) == 1
    count = "ragforge_embed_request_duration_seconds_count"
    assert delta.of(count, process=metrics.process()) == 1


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("synthetic", request=request)


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("synthetic", request=request)


@pytest.mark.parametrize(
    ("handler", "error", "outcome"),
    [
        (_timeout, EmbeddingFailed, "timeout"),
        (_refused, EmbeddingFailed, "server"),
        (lambda r: httpx.Response(500, json={"error": "x"}), EmbeddingFailed, "server"),
        (lambda r: httpx.Response(503), UpstreamRateLimited, "server"),
        (lambda r: httpx.Response(404, json={"error": "not found"}), EmbeddingFailed, "other"),
        (lambda r: httpx.Response(200, json={"embeddings": [[1.0]]}), EmbeddingFailed, "other"),
    ],
)
async def test_embedding_failures_are_classified(
    handler: Any, error: type[Exception], outcome: str
) -> None:
    delta = Delta(("ragforge_embed_requests_total",))
    with pytest.raises(error):
        await _embedder(handler).embed_query("q")
    assert delta.of("ragforge_embed_requests_total", **_labels(outcome)) == 1


async def test_every_provider_request_is_one_measured_call() -> None:
    """A rejected batch is retried per input (the embedder isolates the bad one): the 400 and
    the retry are two calls, each with its own outcome."""
    replies = iter(
        [
            httpx.Response(400, json={"error": "input exceeds context"}),
            httpx.Response(200, json={"embeddings": [[0.6, 0.8]]}),
        ]
    )
    delta = Delta(("ragforge_embed_requests_total",))
    await _embedder(lambda r: next(replies)).embed_documents(["a page"])
    assert delta.of("ragforge_embed_requests_total", **_labels("other")) == 1
    assert delta.of("ragforge_embed_requests_total", **_labels("ok")) == 1


# ---------------------------------------------------------------------- pool gauges
async def test_pool_gauges_read_the_engine_pool_at_scrape_time() -> None:
    settings = Settings(_env_file=None, db_pool_size=7)
    engine = make_engine(settings, "api")
    try:
        labels = {"process": metrics.process(), "pool": "api"}
        assert metrics.sample("ragforge_db_pool_size", labels) == 7
        assert metrics.sample("ragforge_db_pool_checked_out", labels) == 0
    finally:
        await engine.dispose()
