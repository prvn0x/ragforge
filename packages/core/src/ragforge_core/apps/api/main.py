"""`ragforge-api` entrypoint."""

from __future__ import annotations

from typing import Final

import uvicorn

from ragforge_core.apps.api.app import create_app
from ragforge_core.config import get_settings
from ragforge_core.observability import metrics
from ragforge_core.observability.db_collector import DbGaugeCollector
from ragforge_core.observability.logging import configure_logging
from ragforge_core.observability.server import ensure_single_process, serve_metrics

# ADR-0018 R2-6: ONE process (metrics live in process memory). Passed to uvicorn explicitly,
# so WEB_CONCURRENCY in the environment cannot fork extra workers behind the guard's back.
API_WORKERS: Final = 1
# uvicorn's own logging: `log_config=None` leaves the root handler from configure_logging in
# charge (JSON + redaction, observability/logging.py), so uvicorn.error lines are rendered and
# masked like ours. The access log is OFF: the HTTP middleware's metrics and our request events
# cover requests, and an access line prints the raw query string (ADR-0018 final review P1-2).
UVICORN_OPTIONS: Final = {
    "host": "0.0.0.0",  # noqa: S104 - container-internal; compose controls exposure
    "port": 8000,
    "workers": API_WORKERS,
    "log_config": None,
    "access_log": False,
    "proxy_headers": True,
    "timeout_graceful_shutdown": 25,
}


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    ensure_single_process(API_WORKERS)
    metrics.set_process("api")  # before create_app: the engine's pool gauges carry it
    metrics.set_build_info()
    collector = DbGaugeCollector.from_settings(settings)
    metrics.REGISTRY.register(collector)
    server = serve_metrics(settings.api_metrics_port, settings.metrics_addr)
    try:
        uvicorn.run(create_app(settings), **UVICORN_OPTIONS)  # type: ignore[arg-type]
    finally:
        if server is not None:
            server.close()
        metrics.REGISTRY.unregister(collector)
        collector.close()
