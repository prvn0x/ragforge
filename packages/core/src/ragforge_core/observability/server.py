"""`/metrics` on an internal port (ADR-0018 R2-6), never on the application.

Why a separate listener (prometheus_client's own WSGI server in a daemon thread) and not a
FastAPI route: the app's port 8000 is published on the host, and a route there would appear in
the OpenAPI schema and sit behind the app's middleware. The metrics port has NO `ports:` entry
in compose; only Prometheus on the compose network reaches it. Binding 0.0.0.0 is therefore
required inside the container (the compose service sets RAGFORGE_METRICS_ADDR); the Settings
default is loopback, so a process started by hand on a laptop does not expose it on the LAN.

Process model: prometheus_client keeps metrics in process memory. The API is ONE uvicorn
process; several worker processes would each serve a different, partial view on the same
port (or fail to bind). Multiprocess mode needs PROMETHEUS_MULTIPROC_DIR on a writable volume,
which the read-only root filesystem does not have, so it is refused instead of half-working.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Final
from wsgiref.simple_server import WSGIServer

from prometheus_client import start_http_server

from ragforge_core.observability import metrics
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

MULTIPROC_ENV: Final = "PROMETHEUS_MULTIPROC_DIR"


class MultiProcessUnsupported(RuntimeError):
    """More than one server process was requested; in-process metrics cannot serve that."""


def ensure_single_process(workers: int) -> None:
    """R2-6 guard: refuse `workers > 1` (and a multiprocess directory) at startup."""
    if workers != 1:
        raise MultiProcessUnsupported(
            f"the API must run as ONE process (workers={workers}): metrics are in-process; "
            "scale with replicas instead (Prometheus scrapes each)"
        )
    if os.environ.get(MULTIPROC_ENV):
        raise MultiProcessUnsupported(
            f"{MULTIPROC_ENV} is set but prometheus multiprocess mode is not supported"
        )


@dataclass(slots=True)
class MetricsServer:
    server: WSGIServer
    thread: threading.Thread

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def serve_metrics(port: int, addr: str) -> MetricsServer | None:
    """Start the listener for `metrics.REGISTRY`; `port=0` disables it (tests, local runs)."""
    if port == 0:
        return None
    server, thread = start_http_server(port, addr=addr, registry=metrics.REGISTRY)
    log.info("metrics_listening", addr=addr, port=port, process=metrics.process())
    return MetricsServer(server, thread)
