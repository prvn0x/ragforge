"""Helpers every migration uses. Keep migrations as reviewed raw SQL (pgvector/RLS DDL is not
something autogenerate gets right)."""

from __future__ import annotations

from alembic import op


def run_sql(statement: str) -> None:
    """Execute raw SQL (multiple statements allowed; no bind parameters)."""
    op.get_bind().exec_driver_sql(statement)


def bump_schema_version(*, min_code_version: int | None = None) -> None:
    """Every migration bumps schema_version; only contract migrations raise min_code_version,
    and only to <= the previous release's CODE_VERSION (ADR-0002)."""
    run_sql("UPDATE rf.schema_meta SET schema_version = schema_version + 1")
    if min_code_version is not None:
        run_sql(
            "UPDATE rf.schema_meta SET min_code_version = "  # noqa: S608 - int() only
            f"GREATEST(min_code_version, {int(min_code_version)})"
        )


def classify(table: str, data_class: str) -> None:
    """Record the table's data class (ADR-0014). A test fails for unclassified tables."""
    assert data_class in ("authoritative", "derived", "system")
    run_sql(f"COMMENT ON TABLE rf.{table} IS 'class: {data_class}'")


def tenant_table(table: str) -> None:
    """Enable RLS with the standard tenant policy + backup read policy (ADR-0013).

    RLS is ENABLED, not FORCED: the NOLOGIN owner rf_system (and therefore the named SECURITY
    DEFINER functions it owns) sees all tenants; every login role is filtered.
    """
    run_sql(
        f"""
        ALTER TABLE rf.{table} ENABLE ROW LEVEL SECURITY;
        CREATE POLICY tenant_isolation ON rf.{table}
            USING (tenant_id = rf.current_tenant())
            WITH CHECK (tenant_id = rf.current_tenant());
        CREATE POLICY backup_read ON rf.{table} FOR SELECT TO rf_backup USING (true);
        """
    )
