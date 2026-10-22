"""The exported metrics ARE the locked catalogue (ADR-0018 R2-1), and no label can carry an
unbounded value (the cardinality rules of §4).

`test_every_series_in_the_registry_has_bounded_labels` checks EVERY series that exists when it
runs, after first pushing hostile values (ids, paths, free text) through every public recorder,
so it guards both the code paths of this module's other tests and the guards themselves."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.metrics import MetricWrapperBase

from module_obs.obs_metrics_support import ScriptedChat, samples
from ragforge_core.apps.api.app import create_app
from ragforge_core.apps.api.routes import ROUTERS
from ragforge_core.config import Settings
from ragforge_core.observability import catalog, metrics
from ragforge_core.observability.db_collector import DB_GAUGES, DbGaugeCollector
from ragforge_core.observability.llm import InstrumentedChatModel
from ragforge_core.observability.prompt_context import for_prompt
from ragforge_core.observability.surface import (
    DEMO_TENANT_NAME,
    surface_of_path,
    surface_of_tenant,
)

KINDS: dict[type[MetricWrapperBase], str] = {
    Counter: "counter",
    Gauge: "gauge",
    Histogram: "histogram",
}
IN_PROCESS: dict[str, MetricWrapperBase] = {
    name: value
    for name, value in vars(metrics).items()
    if isinstance(value, MetricWrapperBase) and name.isupper()
}


def _name(m: MetricWrapperBase) -> str:
    """The exported name (prometheus_client stores a counter's without `_total`)."""
    base: str = m._name
    return f"{base}_total" if isinstance(m, Counter) else base


def _exported() -> dict[str, MetricWrapperBase]:
    return {_name(m): m for m in IN_PROCESS.values() if _name(m) != metrics.COLLECT_ERRORS_NAME}


# ---------------------------------------------------------------------- the catalogue
def test_in_process_metrics_plus_the_db_collector_are_exactly_the_catalogue() -> None:
    in_process = set(_exported())
    assert in_process.isdisjoint(DB_GAUGES)
    assert in_process | set(DB_GAUGES) == {m.name for m in catalog.CATALOG}


@pytest.mark.parametrize("spec", catalog.CATALOG, ids=lambda m: m.name)
def test_each_metric_has_the_catalogue_type_labels_buckets_and_help(spec: catalog.Metric) -> None:
    if spec.name in DB_GAUGES:
        (family,) = [
            f for f in DbGaugeCollector(engine=_no_engine()).describe() if f.name == spec.name
        ]
        assert (family.type, family.documentation) == ("gauge", spec.help)
        return
    m = _exported()[spec.name]
    assert KINDS[type(m)] == spec.kind
    assert tuple(m._labelnames) == spec.labels
    assert m._documentation == spec.help
    if spec.kind == "histogram":
        assert isinstance(m, Histogram)
        assert tuple(m._upper_bounds[:-1]) == spec.buckets
        assert m._upper_bounds[-1] == float("inf")


def test_the_only_metric_beyond_the_catalogue_is_the_collector_error_counter() -> None:
    names = {f.name for f in metrics.REGISTRY.collect()}
    expected = {m.name.removesuffix("_total") for m in catalog.CATALOG if m.name not in DB_GAUGES}
    assert names - expected == {metrics.COLLECT_ERRORS_NAME.removesuffix("_total")}


def _no_engine() -> Any:
    """describe() must not touch the database: an engine that would fail proves it."""
    return object()


# ---------------------------------------------------------------------- label guards
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("answer@999", "other"),  # a real-looking tag that this build does not have
        ("7c9e6679-7425-40de-944b-e07fc1f90ae7", "other"),
        ("how do I reset my password?", "other"),
    ],
)
def test_unknown_prompt_names_become_other(value: str, expected: str) -> None:
    assert metrics.prompt_label(value) == expected


def test_every_template_of_this_build_is_a_known_prompt_label() -> None:
    from ragforge_core.prompts import PROMPT_VERSION

    for tag in PROMPT_VERSION.split(","):
        assert metrics.prompt_label(tag) == tag
    assert metrics.prompt_label(catalog.UNKNOWN_PROMPT) == catalog.UNKNOWN_PROMPT


@pytest.mark.parametrize(
    ("fn", "value", "expected"),
    [
        (metrics.route_label, "/v1/jobs/{job_id}", "/v1/jobs/{job_id}"),
        (metrics.route_label, "/v1/what ever?x=1", "other"),
        (metrics.ident_label, "sync_page", "sync_page"),
        (metrics.ident_label, "Sync Page", "other"),
        (metrics.model_label, "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
        (metrics.model_label, "models/llama3.2:3b", "models/llama3.2:3b"),
        (metrics.model_label, "a model with spaces", "other"),
        (metrics.model_label, "x" * 81, "other"),
    ],
)
def test_shape_guards(fn: Any, value: str, expected: str) -> None:
    assert fn(value) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [(None, "network"), (200, "2xx"), (204, "2xx"), (429, "429"), (404, "4xx"), (503, "5xx")],
)
def test_notion_status_classes(code: int | None, expected: str) -> None:
    assert metrics.notion_status(code) == expected
    assert expected in catalog.NOTION_STATUS


def test_status_classes_stay_inside_the_catalogue() -> None:
    assert [metrics.status_class(c) for c in (200, 302, 404, 500)] == ["2xx", "3xx", "4xx", "5xx"]
    assert metrics.status_class(99) == "other"


def test_the_process_label_is_bounded() -> None:
    before = metrics.process()
    try:
        metrics.set_process("tenant-42")
        assert metrics.process() == "other"
        metrics.set_process("worker")
        assert metrics.process() == "worker"
    finally:
        metrics.set_process(before)


# ---------------------------------------------------------------------- cardinality, globally
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# The templates of the real app, plus the ad-hoc test apps' own (test_obs_metrics_http), since
# the registry is process-wide. Any other value (a raw path, an id) is a cardinality leak.
_TEST_ROUTES = {"/v1/stream/{n}", "/v1/boom"}
# The app's own routes PLUS every /v1 router (they are attached at startup, so a bare
# create_app() lists only 7 of them: in a full run, other modules' requests reach the rest).
_ROUTES = (
    {getattr(r, "path", "") for r in create_app(Settings(_env_file=None)).routes}
    | {getattr(route, "path", "") for router in ROUTERS for route in router.routes}
    | {
        metrics.UNMATCHED_ROUTE,
        metrics.OTHER,
        *_TEST_ROUTES,
    }
)
_ALLOWED: dict[str, set[str]] = {
    "status_class": set(catalog.STATUS_CLASSES) | {"other"},
    "surface": set(catalog.SURFACES),
    "provider": set(metrics.PROVIDERS),
    "prompt": set(metrics.PROMPT_TAGS) | {"other"},
    "direction": set(metrics.DIRECTIONS),
    "process": set(metrics.PROCESSES),
    "stage": set(catalog.RETRIEVAL_STAGES) | {"other"},
    "reason": set(catalog.CLAIM_REASONS) | {"other"},
    "priority": set(metrics.PRIORITIES) | {"other"},
    "endpoint": set(metrics.INTERVIEW_ENDPOINTS),
    "source": set(metrics.INTERVIEW_SOURCES) | set(metrics.COLLECT_SOURCES) | {"other"},
}
# `outcome` and `status` mean different things per metric.
_OUTCOMES: dict[str, set[str]] = {
    "ragforge_chat_turns": set(catalog.TURN_OUTCOMES),
    "ragforge_llm_requests": set(catalog.LLM_OUTCOMES),
    "ragforge_llm_request_duration_seconds": set(catalog.LLM_OUTCOMES),
    "ragforge_embed_requests": set(catalog.EMBED_OUTCOMES),
    "ragforge_jobs": set(catalog.JOB_OUTCOMES),
    "ragforge_interview_requests": set(metrics.INTERVIEW_OUTCOMES),
}
_STATUSES: dict[str, set[str]] = {
    "ragforge_verification": set(catalog.VERIFICATION_STATUSES) | {"other"},
    "ragforge_notion_requests": set(catalog.NOTION_STATUS),
}
_SHAPED = {"model", "kind", "pool", "connector_kind", "version", "schema_version", "le", "route"}


async def _push_hostile_values() -> None:
    hostile = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    metrics.observe_embed(f"tenant {hostile}", 0.01)
    metrics.count_notion(999)
    with metrics.retrieval_stage(hostile):
        pass
    chat = InstrumentedChatModel(ScriptedChat())
    chat._model = metrics.model_label(f"model {hostile}")
    with for_prompt(f"question about {hostile}"):
        _ = [e async for e in chat.stream("s", [])]
    app = create_app(Settings(_env_file=None))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get(f"/v1/{hostile}/documents")


async def test_every_series_in_the_registry_has_bounded_labels() -> None:
    await _push_hostile_values()
    problems: list[str] = []
    for (name, labels), _value in samples().items():
        family = re.sub(r"_(total|count|sum|bucket|created)$", "", name)
        for label, value in labels:
            if _UUID.search(value):
                problems.append(f"{name}{{{label}={value}}}: an id as a label")
            elif label == "outcome":
                if value not in _OUTCOMES.get(family, set()):
                    problems.append(f"{name}{{outcome={value}}}")
            elif label == "status":
                if value not in _STATUSES.get(family, set()):
                    problems.append(f"{name}{{status={value}}}")
            elif label == "route":
                if value not in _ROUTES:
                    problems.append(f"{name}{{route={value}}}")
            elif label in _ALLOWED:
                if value not in _ALLOWED[label]:
                    problems.append(f"{name}{{{label}={value}}}")
            elif label not in _SHAPED:
                problems.append(f"{name}: unexpected label {label}")
    assert problems == [], "\n".join(problems)


# ---------------------------------------------------------------------- surface rules
def test_the_demo_tenant_name_is_the_one_make_eval_demo_creates() -> None:
    makefile = (Path(__file__).parents[4] / "Makefile").read_text(encoding="utf-8")
    assert f"create-tenant --name {DEMO_TENANT_NAME} " in makefile


@pytest.mark.parametrize(
    ("kind", "name", "expected"),
    [
        ("eval", "anything", "eval"),
        ("personal", DEMO_TENANT_NAME, "demo"),
        ("personal", "my notes", "chat"),
        ("team", "x", "chat"),
    ],
)
def test_the_surface_of_a_tenant(kind: str, name: str, expected: str) -> None:
    assert surface_of_tenant(kind, name) == expected
    assert expected in catalog.SURFACES


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/v1/interview/grade", "interview"), ("/v1/eval/verify", "eval"), ("/v1/chat", "chat")],
)
def test_the_surface_of_a_route(path: str, expected: str) -> None:
    assert surface_of_path(path) == expected


def test_bounded_series_exist_at_zero_from_import() -> None:
    """A restarted process exports every all-bounded series at 0, so the first event after the
    restart is an increase() Prometheus can see (OBS-STACK's alert windows rely on it)."""
    have = samples(("ragforge_chat_turns", "ragforge_verification", "ragforge_notion_requests"))
    for surface in catalog.SURFACES:
        for outcome in catalog.TURN_OUTCOMES:
            key = ("ragforge_chat_turns_total", (("outcome", outcome), ("surface", surface)))
            assert key in have, key
        for status in catalog.VERIFICATION_STATUSES:
            key = ("ragforge_verification_total", (("status", status), ("surface", surface)))
            assert key in have, key
    for status in catalog.NOTION_STATUS:
        assert ("ragforge_notion_requests_total", (("status", status),)) in have
