"""Integration tests for roles, ownership, RLS and schema invariants (ADR-0002/0013/0014).

These are the guarantees every Phase 1 module builds on. If one fails, stop and fix it first.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import errors

from pgtest import connect
from ragforge_core import CODE_REQUIRED_SCHEMA
from ragforge_core.config import Settings
from ragforge_core.db.bootstrap import bootstrap_database
from ragforge_core.db.migrate import upgrade_to_head, verify_schema

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]

LOGIN_ROLES = ("rf_migrator", "rf_api", "rf_worker", "rf_metrics", "rf_backup")


def test_schema_at_required_version(superuser: Conn) -> None:
    row = superuser.execute(
        "SELECT schema_version, min_code_version FROM rf.schema_meta"
    ).fetchone()
    assert row == (CODE_REQUIRED_SCHEMA, 0)


def test_bootstrap_and_migrate_are_idempotent(settings: Settings, superuser: Conn) -> None:
    bootstrap_database(settings)
    upgrade_to_head(settings)
    row = superuser.execute("SELECT schema_version FROM rf.schema_meta").fetchone()
    assert row == (CODE_REQUIRED_SCHEMA,)


def test_every_table_is_classified(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT c.relname, obj_description(c.oid, 'pg_class')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rf' AND c.relkind IN ('r', 'p')
        """
    ).fetchall()
    unclassified = [name for name, comment in rows if not str(comment or "").startswith("class: ")]
    assert rows
    assert unclassified == []


def test_every_object_owned_by_rf_system(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT c.relname, pg_get_userbyid(c.relowner)
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rf'
        UNION ALL
        SELECT p.proname, pg_get_userbyid(p.proowner)
        FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'rf'
        """
    ).fetchall()
    assert rows
    assert {owner for _, owner in rows} == {"rf_system"}


def test_tenant_tables_have_rls_enabled_not_forced(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
               (SELECT count(*) FROM pg_policy p WHERE p.polrelid = c.oid
                  AND p.polname = 'tenant_isolation')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'tenant_id' AND NOT a.attisdropped
        WHERE n.nspname = 'rf' AND c.relkind IN ('r', 'p') AND NOT c.relispartition
        """
    ).fetchall()
    assert rows
    bad = [r for r in rows if not (r[1] is True and r[2] is False and r[3] == 1)]
    assert bad == []


def test_login_roles_have_no_dangerous_attributes(superuser: Conn) -> None:
    rows = superuser.execute(
        """
        SELECT rolname, rolsuper, rolbypassrls, rolcreaterole, rolcreatedb
        FROM pg_roles WHERE rolname = ANY(%s) OR rolname = 'rf_system'
        """,
        (list(LOGIN_ROLES),),
    ).fetchall()
    assert len(rows) == len(LOGIN_ROLES) + 1
    assert all(not any(flags) for _, *flags in rows)


def test_rf_system_cannot_log_in(superuser: Conn) -> None:
    assert superuser.execute(
        "SELECT rolcanlogin FROM pg_roles WHERE rolname = 'rf_system'"
    ).fetchone() == (False,)


def test_app_role_cannot_create_temp_tables(settings: Settings) -> None:
    with connect(settings, "api") as conn, pytest.raises(errors.InsufficientPrivilege):
        conn.execute("CREATE TEMP TABLE shadow (x int)")


def test_rls_fails_closed_without_tenant(settings: Settings, make_tenant: MakeTenant) -> None:
    make_tenant()
    with connect(settings, "api") as conn:
        assert conn.execute("SELECT count(*) FROM rf.collections").fetchone() == (0,)
        conn.execute("SELECT set_config('app.tenant_id', '', true)")
        assert conn.execute("SELECT count(*) FROM rf.collections").fetchone() == (0,)


def test_rls_isolates_tenants(settings: Settings, make_tenant: MakeTenant) -> None:
    tenant_a, _, coll_a = make_tenant()
    tenant_b, _, coll_b = make_tenant()
    with connect(settings, "api") as conn:
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant_a),))
        seen = {r[0] for r in conn.execute("SELECT id FROM rf.collections").fetchall()}
        assert seen == {coll_a}
        # Writing a row for another tenant violates the WITH CHECK clause.
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO rf.collections (tenant_id, name, active_space_id) VALUES (%s, 'x', 1)",
                (tenant_b,),
            )
    assert coll_b != coll_a


def test_tenant_setting_is_transaction_local(settings: Settings, make_tenant: MakeTenant) -> None:
    tenant, _, _ = make_tenant()
    with connect(settings, "api") as conn:
        with conn.transaction():
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
            assert conn.execute("SELECT count(*) FROM rf.collections").fetchone() == (1,)
        # Same pooled connection, next transaction: no tenant carried over.
        assert conn.execute("SELECT count(*) FROM rf.collections").fetchone() == (0,)


def test_cross_tenant_foreign_keys_rejected(superuser: Conn, make_tenant: MakeTenant) -> None:
    tenant_a, _, _ = make_tenant()
    _, _, coll_b = make_tenant()
    with pytest.raises(errors.ForeignKeyViolation):
        superuser.execute(
            "INSERT INTO rf.connectors (tenant_id, kind, target_collection_id, display_name) "
            "VALUES (%s, 'notion', %s, 'x')",
            (tenant_a, coll_b),
        )


def test_app_roles_cannot_write_jobs_directly(settings: Settings, make_tenant: MakeTenant) -> None:
    tenant, _, _ = make_tenant()
    for role in ("api", "worker"):
        with connect(settings, role) as conn:
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
            with pytest.raises(errors.InsufficientPrivilege):
                conn.execute(
                    "INSERT INTO rf.jobs (tenant_id, kind, priority, dedupe_key) "
                    "VALUES (%s, 'x', 0, 'k')",
                    (tenant,),
                )


def test_documents_unique_treats_null_connector_as_equal(
    superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    upload = uuid4()
    superuser.execute(
        "INSERT INTO rf.uploads "
        "(id, tenant_id, collection_id, blob_key, filename, sha256, size_bytes) "
        "VALUES (%s, %s, %s, %s, 'a.pdf', %s, 1)",
        (upload, tenant, coll, f"k-{upload}", b"\x00" * 32),
    )
    insert = (
        "INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'file', 'same-sha', 't')"
    )
    superuser.execute(insert, (uuid4(), tenant, coll, upload))
    with pytest.raises(errors.UniqueViolation):
        superuser.execute(insert, (uuid4(), tenant, coll, upload))


def test_backup_role_reads_all_tenants(settings: Settings, make_tenant: MakeTenant) -> None:
    make_tenant()
    make_tenant()
    with connect(settings, "backup") as conn:
        row = conn.execute("SELECT count(DISTINCT tenant_id) FROM rf.collections").fetchone()
        assert row is not None
        assert int(str(row[0])) >= 2
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute("DELETE FROM rf.collections")


def test_migrator_does_not_inherit_owner_privileges(settings: Settings) -> None:
    # Without SET ROLE the migrator has no table privileges (INHERIT FALSE).
    with (
        connect(settings, "migrator") as conn,
        pytest.raises(errors.InsufficientPrivilege),
    ):
        conn.execute("SELECT count(*) FROM rf.tenants")


def test_migrate_entrypoint_verification_passes(settings: Settings) -> None:
    readiness = verify_schema(settings)
    assert readiness.ready, readiness.reason
