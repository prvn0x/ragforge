"""`ragforge-db-bootstrap`: one-shot, superuser/admin only. Creates roles, schemas, extensions."""

from __future__ import annotations

from ragforge_core.config import get_settings
from ragforge_core.db.bootstrap import bootstrap_database
from ragforge_core.observability.logging import configure_logging, get_logger

log = get_logger(__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    bootstrap_database(settings)
    log.info("db_bootstrap_done", database=settings.db_name)
