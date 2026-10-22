"""ADR-0018 R2-2: `rf.connector_freshness`, the sync-freshness input of the metrics collector.

OWNED BY OBS-METRICS. Additive (expand only).

Per connector KIND, the age of the STALEST enabled connector's last successful sync, so one
connector that stopped syncing is visible however many others are fresh (`min(last_success)`):

    connector_kind | connectors | oldest_success_at | oldest_success_age_seconds

Which source is authoritative: a finished `sync_connector` job in state `done`, in `rf.jobs`
or (after five minutes) `rf.jobs_history`. That is the only record written on EVERY successful
poll: the handler completes the listing job even when nothing changed, whereas the sync cursor
(`rf.sync_cursors.updated_at`) only moves when a listing returns an item, so a quiet workspace
would read as stale. The job is matched to its connector by `payload->>'connector_id'` AND the
job's own tenant (listing jobs leave `jobs.connector_id` unset, apps/worker/upkeep.py), so a
foreign or malformed payload can never make another tenant's connector look fresh.

Only a REAL listing counts: the job must carry the `listed` progress marker, which only the
listing path of handle_sync_connector writes (services/sync/handlers.py). The handler also
completes the job as `done` WITHOUT listing when the connector is not active (`_complete_noop`,
progress `{"skipped": ...}`), and POST /v1/connectors/{id}/sync accepts an `error` connector, so
without the marker a "sync now" click on a broken connector would reset its age to ~0 and mask
the stale-sync alert while nothing synced. A positive marker (not "NOT skipped") also fails safe
for any future no-op path: an unknown completion can only make a connector look staler.

"Enabled" is status `active` or `error`: an `error` connector is the one most in need of an
alert; `paused` and `deleting` ones are excluded. A connector with no successful sync on record
(new, or older than the 30-day history retention) counts from its `created_at`, which can only
make it look staler, never fresher.

Shape (ADR-0013): aggregates only, no tenant ids. Instead of a second view that bypasses RLS
(`queue_stats` stays the only one, and test_catalog_invariants keeps saying so), the
cross-tenant read is a named SECURITY DEFINER function with the pinned search_path, EXECUTE
for rf_metrics only, and the view over it is `security_invoker`: rf_metrics can read the view
and nothing else (not rf.jobs, not rf.connectors). Two partial expression indexes keep the
per-scrape read to an index scan of finished listing jobs, not the whole job history.

No percent signs anywhere below: run_sql goes through psycopg's paramstyle.

Revision ID: 0013
Revises: 0012
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_INDEXES = """
-- Small tables at this scale (rf.jobs holds only unfinished and recently finished rows), so
-- plain CREATE INDEX: the brief SHARE lock is cheaper than a CONCURRENTLY build per partition.
CREATE INDEX jobs_sync_done ON rf.jobs (tenant_id, (payload->>'connector_id'), finished_at)
    WHERE kind = 'sync_connector' AND status = 'done' AND progress ? 'listed';
CREATE INDEX jobs_history_sync_done
    ON rf.jobs_history (tenant_id, (payload->>'connector_id'), finished_at)
    WHERE kind = 'sync_connector' AND status = 'done' AND progress ? 'listed';
"""

_FRESHNESS = """
CREATE FUNCTION rf.connector_freshness_rows()
    RETURNS TABLE (
        connector_kind             text,
        connectors                 bigint,
        oldest_success_at          timestamptz,
        oldest_success_age_seconds double precision
    )
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
    AS $$
        WITH finished AS (
            SELECT j.tenant_id, j.payload->>'connector_id' AS connector_id, j.finished_at
              FROM rf.jobs j
             WHERE j.kind = 'sync_connector' AND j.status = 'done' AND j.progress ? 'listed'
            UNION ALL
            SELECT h.tenant_id, h.payload->>'connector_id', h.finished_at
              FROM rf.jobs_history h
             WHERE h.kind = 'sync_connector' AND h.status = 'done' AND h.progress ? 'listed'
        ), last_success AS (
            SELECT c.kind, coalesce(max(f.finished_at), c.created_at) AS at
              FROM rf.connectors c
              LEFT JOIN finished f
                ON f.tenant_id = c.tenant_id AND f.connector_id = c.id::text
             WHERE c.status IN ('active', 'error')
             GROUP BY c.id, c.kind, c.created_at
        )
        SELECT s.kind,
               count(*)::bigint,
               min(s.at),
               greatest(extract(epoch FROM now() - min(s.at)), 0)::double precision
          FROM last_success s
         GROUP BY s.kind
    $$;
COMMENT ON FUNCTION rf.connector_freshness_rows() IS
    'ADR-0018: aggregates only (no tenant ids); EXECUTE for rf_metrics only';

CREATE VIEW rf.connector_freshness WITH (security_invoker = true) AS
    SELECT connector_kind, connectors, oldest_success_at, oldest_success_age_seconds
      FROM rf.connector_freshness_rows();
COMMENT ON VIEW rf.connector_freshness IS
    'ADR-0018: per connector kind, the age of the stalest enabled connector''s last successful sync';

REVOKE ALL ON FUNCTION rf.connector_freshness_rows() FROM PUBLIC;
REVOKE ALL ON rf.connector_freshness FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.connector_freshness_rows() TO rf_metrics;
GRANT SELECT ON rf.connector_freshness TO rf_metrics;
"""


def upgrade() -> None:
    run_sql(_INDEXES)
    run_sql(_FRESHNESS)
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
