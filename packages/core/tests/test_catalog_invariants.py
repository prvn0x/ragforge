"""Catalog-wide invariants (ADR-0013). They inspect EVERYTHING in schema rf, so every definer
function, view and grant any Phase 1 module adds is checked automatically."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import psycopg
import pytest

from pgtest import connect
from ragforge_contracts.api import ApiScope
from ragforge_core.config import Settings
from ragforge_core.domain.models import Scope

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]

REQUIRED_DEFINER_PATH = "search_path=pg_catalog, rf, ext, pg_temp"
VIEWS_ALLOWED_TO_BYPASS_RLS = {"queue_stats"}  # aggregates only, no tenant identifiers


def test_every_security_definer_function_pins_search_path(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT p.proname, coalesce(array_to_string(p.proconfig, ';'), '')
        FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'rf' AND p.prosecdef
        """
    ).fetchall()
    bad = [name for name, config in rows if REQUIRED_DEFINER_PATH not in str(config).split(";")]
    assert bad == [], f"definer functions without `{REQUIRED_DEFINER_PATH}`: {bad}"


def test_every_view_is_security_invoker(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT c.relname, coalesce(array_to_string(c.reloptions, ','), '')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rf' AND c.relkind IN ('v', 'm')
        """
    ).fetchall()
    bad = [
        name
        for name, opts in rows
        if name not in VIEWS_ALLOWED_TO_BYPASS_RLS and "security_invoker=true" not in str(opts)
    ]
    assert bad == [], f"views that would bypass RLS as owner rf_system: {bad}"


def test_no_table_or_view_granted_to_public(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT c.relname, c.relacl::text
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rf' AND c.relkind IN ('r', 'p', 'v', 'm', 'S')
        """
    ).fetchall()
    bad = [name for name, acl in rows if acl and ("{=" in str(acl) or ",=" in str(acl))]
    assert bad == []


def test_app_roles_cannot_touch_definer_only_tables(superuser: Conn) -> None:
    for table in ("notion_rate", "connector_inflight", "migration_checksums", "jobs"):
        for role in ("rf_api", "rf_worker", "rf_metrics"):
            (can_write,) = superuser.execute(  # type: ignore[misc]
                "SELECT has_table_privilege(%s, %s, 'INSERT,UPDATE,DELETE')",
                (role, f"rf.{table}"),
            ).fetchone()
            assert can_write is False, f"{role} can write rf.{table}"


def test_api_can_insert_identity_tables_within_its_tenant(
    settings: Settings, make_tenant: Callable[..., tuple[UUID, UUID, UUID]]
) -> None:
    tenant, _, _ = make_tenant()
    with connect(settings, "api") as conn:
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
        conn.execute(
            "INSERT INTO rf.audit_log (tenant_id, actor, action) VALUES (%s, 'test', 'x')",
            (tenant,),
        )
        conn.execute(
            "INSERT INTO rf.usage_events (tenant_id, kind, provider, model) "
            "VALUES (%s, 'chat', 'p', 'm')",
            (tenant,),
        )


def test_tenant_delete_cascades_through_connectors(
    superuser: Conn, make_tenant: Callable[..., tuple[UUID, UUID, UUID]]
) -> None:
    tenant, _, coll = make_tenant()
    superuser.execute(
        "INSERT INTO rf.connectors (tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, 'notion', %s, 'c')",
        (tenant, coll),
    )
    superuser.execute("DELETE FROM rf.tenants WHERE id = %s", (tenant,))
    assert superuser.execute(
        "SELECT count(*) FROM rf.connectors WHERE tenant_id = %s", (tenant,)
    ).fetchone() == (0,)


def test_scope_enums_stay_in_sync() -> None:
    assert {s.value for s in Scope} == {s.value for s in ApiScope}
