"""ADR-0018 final review P1-3: `rf.dead_jobs_last_day()`, the input of `ragforge_jobs_dead`.

OWNED BY OBS-METRICS. Additive (expand only).

Why: the gauge used to count `status = 'dead'` rows of `rf.queue_stats`, which reads `rf.jobs`
only. `rf.move_finished_jobs` (the worker's upkeep, every five minutes) archives finished rows,
dead ones included, to `rf.jobs_history` once they are five minutes old. So a dead job vanished
from the gauge about five minutes after it died: the board's dead-jobs panel went back to 0 and
the RagforgeDeadJobsRising rule saw a "falling" count, although nothing had been fixed.

What: the number of jobs that went dead in the last 24 hours, wherever the row lives now
(`rf.jobs` or `rf.jobs_history`). A row is in exactly one of the two: the archive moves it
(DELETE ... RETURNING into the history) in one statement. The window is fixed and has no
parameter: one day is long enough for the owner to see a death the next time the board is open,
short enough that the gauge returns to 0 by itself once the cause is fixed, and well inside the
history's 30-day retention. A requeued dead job leaves the count because its status changes.

Shape (ADR-0013, the same pattern as 0013's connector_freshness_rows): a named SECURITY DEFINER
function owned by rf_system with the pinned search_path, returning ONE aggregate number (no
tenant ids, no job ids), REVOKE ALL FROM PUBLIC and EXECUTE for rf_metrics only. rf_metrics still
cannot read rf.jobs or rf.jobs_history themselves. A partial index on the history keeps the
per-scrape read to the dead rows of the current partitions (the history is range-partitioned by
`finished_at`, so the time bound also prunes older months); `rf.jobs` is small and already has
`jobs_finished`.

No percent signs anywhere below: run_sql goes through psycopg's paramstyle.

Revision ID: 0014
Revises: 0013
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

_INDEX = """
-- Plain CREATE INDEX (it cascades to every partition): the history is small at this scale.
CREATE INDEX jobs_history_dead ON rf.jobs_history (finished_at) WHERE status = 'dead';
"""

_DEAD_JOBS = """
CREATE FUNCTION rf.dead_jobs_last_day()
    RETURNS bigint
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
    AS $$
        SELECT (SELECT count(*) FROM rf.jobs j
                 WHERE j.status = 'dead' AND j.finished_at >= now() - interval '24 hours')
             + (SELECT count(*) FROM rf.jobs_history h
                 WHERE h.status = 'dead' AND h.finished_at >= now() - interval '24 hours')
    $$;
COMMENT ON FUNCTION rf.dead_jobs_last_day() IS
    'ADR-0018: jobs that went dead in the last 24 h (rf.jobs + rf.jobs_history); one aggregate, '
    'no tenant ids; EXECUTE for rf_metrics only';

REVOKE ALL ON FUNCTION rf.dead_jobs_last_day() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.dead_jobs_last_day() TO rf_metrics;
"""


def upgrade() -> None:
    run_sql(_INDEX)
    run_sql(_DEAD_JOBS)
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
