"""FastAPI application factory: health, readiness and the /v1 API (DESIGN section 9)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragforge_contracts.api import Readiness
from ragforge_core.apps.api.errors import install_error_handlers
from ragforge_core.apps.api.jobfeed import PgJobFeed
from ragforge_core.apps.api.middleware import (
    BodyLimitMiddleware,
    MetricsMiddleware,
    RequestIdMiddleware,
)
from ragforge_core.apps.api.routes import ROUTERS
from ragforge_core.bootstrap import ApiContainer, build_api
from ragforge_core.config import Settings, get_settings
from ragforge_core.db.readiness import check_readiness


def create_app(settings: Settings | None = None, container: ApiContainer | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Every resource is registered the moment it exists, so a failure while starting a later
        # one (or while closing an earlier one) still closes the rest, in reverse order.
        async with AsyncExitStack() as resources:
            built = container or await build_api(settings)
            resources.push_async_callback(built.aclose)
            app.state.container = built
            # One LISTEN connection per process for every job-progress SSE client (jobfeed.py).
            # The container's settings, not the factory argument: a test (or a caller) that
            # passes a container built from other settings must not LISTEN on another database.
            feed = PgJobFeed(built.settings.dsn("api", driver="postgresql"))
            resources.push_async_callback(feed.aclose)
            feed.start()
            app.state.job_feed = feed
            yield

    app = FastAPI(
        title="RAGForge",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    install_error_handlers(app)

    # Order: the last added runs first, so RequestId wraps BodyLimit and its 413 carries the id.
    # Metrics is added LAST, so it is outermost: it times and counts everything, 413s included
    # (ADR-0018 R2-5). /metrics itself is NOT served here: it has its own internal port
    # (observability/server.py), so GET :8000/metrics stays a 404 outside the OpenAPI schema.
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(MetricsMiddleware)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=Readiness, include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        readiness = await check_readiness(request.app.state.container.engine)
        return JSONResponse(readiness.model_dump(), status_code=200 if readiness.ready else 503)

    for router in ROUTERS:
        app.include_router(router)

    return app
