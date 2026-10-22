"""`/metrics` on the internal port (ADR-0018 R2-6): a real listener serves the catalogue, the
single-process guard, and the entrypoints' wiring (with synthetic settings: never `.env`)."""

from __future__ import annotations

import socket
import urllib.request
from typing import Any

import pytest
import uvicorn
from sqlalchemy import create_engine

from ragforge_core import CODE_REQUIRED_SCHEMA
from ragforge_core.apps.api import main as api_main
from ragforge_core.apps.worker import main as worker_main
from ragforge_core.config import Settings
from ragforge_core.observability import catalog, metrics
from ragforge_core.observability.db_collector import DB_GAUGES, DbGaugeCollector
from ragforge_core.observability.server import (
    MULTIPROC_ENV,
    MultiProcessUnsupported,
    ensure_single_process,
    serve_metrics,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        assert r.status == 200
        return str(r.read().decode())


def test_the_listener_serves_the_catalogue_and_build_info() -> None:
    metrics.set_build_info()
    server = serve_metrics(_free_port(), "127.0.0.1")
    assert server is not None
    try:
        body = _scrape(server.port)
    finally:
        server.close()
    for spec in catalog.CATALOG:
        if spec.name not in DB_GAUGES:
            assert f"# TYPE {spec.name} {spec.kind}" in body, spec.name
    version = metrics.package_version()
    assert (
        f'ragforge_build_info{{process="{metrics.process()}",schema_version="'
        f'{CODE_REQUIRED_SCHEMA}",version="{version}"}} 1.0'
    ) in body


def test_a_failing_db_read_leaves_the_gauges_absent_and_the_scrape_ok() -> None:
    """The collector is registered like production does; the DB is unreachable: the scrape is
    still 200, the in-process metrics are there, the DB gauges are absent, the error counted."""
    dead = create_engine(
        "postgresql+psycopg://rf_metrics:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    collector = DbGaugeCollector(dead)
    metrics.REGISTRY.register(collector)
    server = serve_metrics(_free_port(), "127.0.0.1")
    assert server is not None
    before = metrics.sample("ragforge_metrics_collect_errors_total", {"source": "queue_stats"})
    try:
        body = _scrape(server.port)
    finally:
        server.close()
        metrics.REGISTRY.unregister(collector)
        collector.close()
    assert "ragforge_http_requests_total" in body
    assert not [g for g in DB_GAUGES if f"\n{g}" in body]
    after = metrics.sample("ragforge_metrics_collect_errors_total", {"source": "queue_stats"})
    assert after == before + 1


def test_port_zero_disables_the_listener() -> None:
    assert serve_metrics(0, "127.0.0.1") is None


# ---------------------------------------------------------------------- one process
@pytest.mark.parametrize("workers", [0, 2, 4])
def test_more_than_one_server_process_is_refused(workers: int) -> None:
    with pytest.raises(MultiProcessUnsupported):
        ensure_single_process(workers)


def test_multiprocess_mode_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MULTIPROC_ENV, "/tmp/prom")  # noqa: S108 - never created
    with pytest.raises(MultiProcessUnsupported):
        ensure_single_process(1)


def test_the_api_entrypoint_runs_one_uvicorn_process_and_its_own_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, api_metrics_port=0)  # synthetic: never reads .env
    seen: dict[str, Any] = {}
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    monkeypatch.setattr(
        api_main, "serve_metrics", lambda port, addr: seen.update(listener=(port, addr))
    )
    before = metrics.process()
    try:
        api_main.run()
        assert metrics.process() == "api"
    finally:
        metrics.set_process(before)
    assert seen["workers"] == api_main.API_WORKERS == 1
    assert seen["listener"] == (0, "127.0.0.1")
    assert seen["port"] == 8000  # the app port: the listener is another one
    # uvicorn's records go through the root redacting handler (log_config=None), and its access
    # log, which prints raw query strings unredacted by any processor of ours, is off.
    assert seen["log_config"] is None
    assert seen["access_log"] is False
    assert not [p for p in seen["app"].routes if "metrics" in getattr(p, "path", "")]
    # the collector was unregistered again on the way out
    assert not [c for c in metrics.REGISTRY._collector_to_names if isinstance(c, DbGaugeCollector)]


def test_the_worker_listener_uses_its_own_port(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_main, "serve_metrics", lambda port, addr: seen.update(port=port, addr=addr)
    )
    before = metrics.process()
    try:
        worker_main.start_metrics(Settings(_env_file=None, metrics_addr="0.0.0.0"))  # noqa: S104
        assert metrics.process() == "worker"
    finally:
        metrics.set_process(before)
    assert seen == {"port": catalog.WORKER_METRICS_PORT, "addr": "0.0.0.0"}  # noqa: S104


def test_settings_default_to_the_catalogue_ports_on_loopback() -> None:
    s = Settings(_env_file=None)
    assert (s.api_metrics_port, s.worker_metrics_port) == (
        catalog.API_METRICS_PORT,
        catalog.WORKER_METRICS_PORT,
    )
    assert s.metrics_addr == "127.0.0.1"
