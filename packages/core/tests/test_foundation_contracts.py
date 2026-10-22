"""Tests for the fixes from the Phase 0 code review: frozen ids, transaction helpers over the
real SQLAlchemy pool, default privileges, composite FKs, migration checksums, error headers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from psycopg import errors
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from pgtest import IdentityReranker
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.db import migrate as migrate_mod
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.domain.errors import Unauthorized, UpstreamRateLimited
from ragforge_core.domain.ids import RAGFORGE_NAMESPACE, chunk_id, document_id
from ragforge_core.domain.ports import TransactionManager

Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]


# ----------------------------------------------------------------- ids (unit)
def test_ids_are_deterministic_and_distinct() -> None:
    t, c, k = UUID(int=1), UUID(int=2), UUID(int=3)
    assert document_id(t, c, k, "page") == document_id(t, c, k, "page")
    assert document_id(t, c, None, "sha") != document_id(t, c, k, "sha")
    assert document_id(t, c, k, "a") != document_id(t, UUID(int=9), k, "a")
    d = document_id(t, c, k, "p")
    assert chunk_id(d, 0) != chunk_id(d, 1)
    with pytest.raises(ValueError, match="ordinal"):
        chunk_id(d, -1)


def test_ids_golden_values_are_frozen() -> None:
    """Changing the derivation silently breaks every stored id, deep link and restore."""
    assert UUID("6f1c2a9e-3b7d-5c44-9a0e-2f8d1b7c4e53") == RAGFORGE_NAMESPACE
    d1 = document_id(UUID(int=1), UUID(int=2), UUID(int=3), "page")
    assert d1 == UUID("40076072-1ee8-58b8-b7df-946090d8c138")
    assert document_id(UUID(int=1), UUID(int=2), None, "sha") == UUID(
        "0585ccd0-a433-5e9d-bacf-70ed54538183"
    )
    assert chunk_id(d1, 7) == UUID("39cdb54d-9bd0-5bf3-9b68-587dadcfe0f4")


# ----------------------------------------------------------------- transactions (integration)
@pytest.mark.integration
async def test_transaction_manager_scopes_and_resets_over_the_pool(
    settings: Settings, make_tenant: MakeTenant
) -> None:
    tenant_a, _, coll_a = make_tenant()
    tenant_b, _, _ = make_tenant()
    engine: AsyncEngine = make_engine(settings.model_copy(update={"db_pool_size": 1}), "api")
    tm = SqlTransactionManager(engine)
    assert isinstance(tm, TransactionManager)
    try:
        async with tm.tenant(tenant_a) as tx:
            ids = {r[0] for r in (await tx.execute(text("SELECT id FROM rf.collections"))).all()}
            assert ids == {coll_a}
        # pool_size=1: the next transaction reuses the same physical connection.
        async with tm.system() as tx:
            n = (await tx.execute(text("SELECT count(*) FROM rf.collections"))).scalar_one()
            assert n == 0  # no tenant carried over: fail closed
        async with tm.tenant(tenant_b) as tx:
            ids = {r[0] for r in (await tx.execute(text("SELECT id FROM rf.collections"))).all()}
            assert coll_a not in ids
    finally:
        await engine.dispose()


@pytest.mark.integration
def test_no_rf_function_is_executable_by_public(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT p.proname, p.proacl::text
        FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'rf'
        """
    ).fetchall()
    assert rows
    # NULL acl means "default" which includes PUBLIC EXECUTE; '=X/' is an explicit PUBLIC grant.
    public = [name for name, acl in rows if acl is None or "{=X/" in str(acl) or ",=X/" in str(acl)]
    assert public == []


@pytest.mark.integration
def test_jobs_cannot_reference_another_tenants_connector(
    superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant_a, _, _ = make_tenant()
    tenant_b, _, coll_b = make_tenant()
    connector_b = uuid4()
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'b')",
        (connector_b, tenant_b, coll_b),
    )
    with pytest.raises(errors.ForeignKeyViolation):
        superuser.execute(
            "INSERT INTO rf.jobs (tenant_id, connector_id, kind, priority, dedupe_key) "
            "VALUES (%s, %s, 'sync', 20, 'k')",
            (tenant_a, connector_b),
        )


@pytest.mark.integration
def test_edited_applied_migration_is_detected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = migrate_mod.migration_checksums()
    tampered = {**real, "0002": "0" * 64}
    monkeypatch.setattr(migrate_mod, "migration_checksums", lambda: tampered)
    with pytest.raises(migrate_mod.MigrationEdited, match="0002"):
        migrate_mod.upgrade_to_head(settings)


def test_checksums_cover_every_migration_file() -> None:
    versions = Path(migrate_mod.__file__).parent / "migrations" / "versions"
    files = {p.name[:4] for p in versions.glob("[0-9][0-9][0-9][0-9]_*.py")}
    assert set(migrate_mod.migration_checksums()) == files


# ----------------------------------------------------------------- API error headers
@pytest.fixture
async def boom_app(settings: Settings) -> FastAPI:
    app = create_app(settings, await build_api(settings, reranker=IdentityReranker()))

    @app.get("/_boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    @app.get("/_limited")
    async def limited() -> None:
        raise UpstreamRateLimited(retry_after_s=2.3)

    @app.get("/_unauth")
    async def unauth() -> None:
        raise Unauthorized("missing key")

    return app


@pytest.mark.integration
async def test_error_responses_keep_request_id_and_headers(boom_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=boom_app, raise_app_exceptions=False)
    async with (
        boom_app.router.lifespan_context(boom_app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as http,
    ):
        r = await http.get("/_boom", headers={"x-request-id": "rid-500"})
        assert r.status_code == 500
        assert r.headers["x-request-id"] == "rid-500"
        assert "secret internal detail" not in r.text

        r = await http.get("/_limited")
        assert (r.status_code, r.headers["retry-after"]) == (503, "3")

        r = await http.get("/_unauth")
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Bearer")

        r = await http.post("/healthz")
        assert r.status_code == 405
        assert "GET" in r.headers["allow"]
