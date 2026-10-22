"""/v1 routers (DESIGN section 9)."""

from fastapi import APIRouter

from ragforge_core.apps.api.routes import (
    ask,
    content,
    eval,
    eval_snapshots,
    identity,
    interview,
    jobs,
    sources,
)

ROUTERS: tuple[APIRouter, ...] = (
    identity.router,
    sources.router,
    content.router,
    jobs.router,
    ask.router,
    interview.router,
    eval.router,
    eval_snapshots.router,
)

__all__ = ["ROUTERS"]
