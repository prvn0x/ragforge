"""Phase 2a review fixes: connector scheduling that does not duplicate, reset backoff or starve.

`rf.list_due_connectors` (0002) only READ due connectors and left `next_poll_at` to the sync job,
which advances it on success only. With a 30 s scheduler tick that caused (blind ops review):
- concurrent twin listings: a listing running longer than one tick was re-enqueued (dedupe only
  covers queued jobs), each twin re-listing from the same cursor;
- backoff reset: the scheduler's enqueue merged into a requeued failure and lowered its
  `run_after` to now(), so 8 attempts burned in minutes, then a fresh job, forever;
- starvation: failing connectors never advanced `next_poll_at` and filled every LIMIT-100 tick.

Two separate mechanisms (round 2 split them: one timer doing both jobs meant a lost enqueue
after a claim cost the whole backoff, up to a day):
- **Claim lease (twin guard).** `rf.claim_due_connectors(limit, lease_s)` returns due active
  connectors that have NO queued or running `sync_connector:<id>` job and moves each one's
  `next_poll_at` forward by a SHORT, FIXED lease (the poll interval), as one statement under
  row locks with SKIP LOCKED. Claimed rows go to the back of the order (no starvation); a queued
  retry is never touched (its backoff stands). If the worker dies between the claim and its
  enqueue, the connector is due again one lease later.
- **Failure backoff.** Applied where the failure is known: a trigger on the listing job's
  terminal transition. `dead` increments `poll_failures` and pushes `next_poll_at` to at least
  now() + 300 s * 2^poll_failures, capped at one day, with +/-10% jitter so an upstream-wide
  outage does not align every connector's retries (300 s = the default poll interval; a
  constant here because connectors carry no poll interval of their own). `done` resets the counter
  (C2 then sets the next poll). Structural, like 0002's in-flight counter, so no exit path,
  including a lease reclaim that buries the job, can skip it.
- `rf.claim_reconcile_connectors(limit, interval_s, lease_s)`: the same claim pattern for the
  nightly `reconcile_deletes` (handoff C2, request 5). The claim takes only a SHORT lease and
  remembers the configured interval (`connectors.reconcile_interval_s`); the next run is
  scheduled by a trigger when the reconcile job ends (`done` or `dead`), at interval +/-5%
  (symmetric, so connectors spread out without drifting later each day). A lost enqueue is
  retried one lease later, not one interval later.

Scope of the twin guarantee: the SCHEDULER never starts a second listing of a connector. A
manual `POST /v1/connectors/{id}/sync` while a scheduled listing runs can still add one (its
enqueue merges only into a QUEUED job); the cost is bounded (GREATEST cursor, page dedupe).

Also in this slot (round 3):
- **Visible backoff.** The dead-listing trigger also writes `connectors.last_error`
  ("listing failed (N in a row): <job error>"), so a backed-off connector is not a silent
  `active` with no error; C2's success path clears it.
- **`rf.heartbeat` no longer reports a false loss.** 0002's version was an UPDATE: it blocked on
  the row lock held by the holder's OWN fenced result transaction and, once that committed
  `done`, found no running row and returned false, so the runtime cancelled the handler's
  post-commit work and logged `job_lease_lost` for a job that had succeeded. Now `false` means
  exactly "another holder took it": the job is gone or its `lease_version` moved on (a reclaim
  plus a new claim bumps it). A row locked by someone else is alive (the holder is inside its
  fenced transaction, which re-checks the version itself); a terminal state at THIS version
  (`done`, `dead`, `failed`, `superseded`, or `queued` after the holder's own retryable fail or
  release) was reached by this holder and is not a loss either. That last claim needs reclaim
  to move the version: 0002's `rf.reclaim_expired_leases` buried or requeued (`superseded` on a
  merge) an expired job at the SAME version, so a zombie holder (VM pause, missed beats) kept
  beating `true` and burning capacity beside the job's new run. This slot replaces it with one
  that bumps `lease_version` first. Correctness never depended on the heartbeat: every result
  transaction is fenced.
- **`rf.defer_job(job, version, delay_s)`**: give a job back without charging an attempt, not
  before `delay_s`. Used for kinds this worker has no handler for (a rolling deploy): failing
  them burned attempts until a new kind went dead; releasing them made the old worker re-claim
  them in a tight loop. Bounded: a job enqueued over 24 h ago that still has no handler is
  buried permanently (`dead`, audited). Cost: during a rolling deploy, an old worker can bury
  a >24 h-old job of a NEW kind (one that waited in backoff that long); drain old workers
  before enqueueing new kinds, or re-enqueue.
- `next_reconcile_at` defaults to a random point within the next day, so existing connectors do
  not all reconcile at the same instant.

Neither claim function enqueues: the worker enqueues in each connector's own tenant transaction, so
the tenant always comes from the claimed row. Both are SECURITY DEFINER per ADR-0013, granted to
rf_worker only. `rf.list_due_connectors` keeps its signature and grant (forward-only; a pre-0008
worker still calls it) but is now itself a CLAIM, with exactly `claim_due_connectors(limit, 300)`
semantics. A pre-0008 worker (a rolling deploy, or a ROLLBACK to the previous image, which
`ragforge-migrate` allows while code_version >= min_code_version) then keeps scheduling, and does
so safely: no twin listings, no backoff reset, claimed rows move to the back. Making it return no
rows instead (round 4) silently stopped all polling after a rollback while readiness stayed green.

Round 6 (blind ops review) also replaces:
- `rf.reclaim_expired_leases`: requeues with `run_after = now() + backoff(attempts)` instead of
  now(). A job that kills its worker (OOM, native crash) no longer kills the next replica within
  seconds, and co-located jobs stop being charged an attempt per incident.
- `rf.assert_lease`: also extends the lease (clock_timestamp() + lease) once the fence passes, so
  a long fenced result transaction that COMMITS keeps its job leased. A transaction that rolls
  back reverts the extension too; the runtime's keepalive beats again right after (locked rows
  are skipped by the heartbeat, not extended).

The `jobs_active_dedupe` index is built without CONCURRENTLY (migrations run in a transaction);
`rf.jobs` holds only live jobs (finished ones move to jobs_history), so the lock is brief.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql(
        """
        ALTER TABLE rf.connectors
            ADD COLUMN next_reconcile_at timestamptz NOT NULL
                DEFAULT now() + random() * interval '1 day',
            ADD COLUMN poll_failures integer NOT NULL DEFAULT 0 CHECK (poll_failures >= 0),
            ADD COLUMN reconcile_interval_s integer NOT NULL DEFAULT 86400
                CHECK (reconcile_interval_s BETWEEN 60 AND 604800);

        -- The "is a listing already queued or running?" probe of both claims (queued-only
        -- jobs_dedupe does not cover running jobs).
        CREATE INDEX jobs_active_dedupe ON rf.jobs (tenant_id, dedupe_key)
            WHERE status IN ('queued', 'running');
        CREATE INDEX connectors_due ON rf.connectors (next_poll_at) WHERE status = 'active';
        CREATE INDEX connectors_reconcile_due ON rf.connectors (next_reconcile_at)
            WHERE status = 'active';

        -- ------------------------------------------------------------ failure counter
        -- Terminal transitions of a sync_connector job. The payload is validated before the cast:
        -- a malformed payload must never abort fail_job/complete_job.
        CREATE FUNCTION rf._connector_poll_outcome() RETURNS trigger
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_raw text := NEW.payload ->> 'connector_id';
            BEGIN
                IF v_raw IS NULL OR v_raw !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' THEN
                    RETURN NULL;
                END IF;
                IF NEW.status = 'dead' THEN
                    UPDATE rf.connectors c
                       SET poll_failures = least(c.poll_failures + 1, 1000),
                           next_poll_at = greatest(c.next_poll_at, now() + make_interval(secs =>
                               least(300 * power(2, least(c.poll_failures + 1, 16)), 86400)
                               * (0.9 + 0.2 * random()))),
                           last_error = left('listing failed ('
                               || (least(c.poll_failures + 1, 1000))::text || ' in a row): '
                               || coalesce(NEW.last_error, 'unknown error'), 500)
                     WHERE c.id = v_raw::uuid AND c.tenant_id = NEW.tenant_id;
                ELSE
                    UPDATE rf.connectors c SET poll_failures = 0
                     WHERE c.id = v_raw::uuid AND c.tenant_id = NEW.tenant_id
                       AND c.poll_failures <> 0;
                END IF;
                RETURN NULL;
            END
            $$;
        REVOKE ALL ON FUNCTION rf._connector_poll_outcome() FROM PUBLIC;
        CREATE TRIGGER jobs_connector_poll_outcome
            AFTER UPDATE OF status ON rf.jobs
            FOR EACH ROW
            WHEN (NEW.kind = 'sync_connector' AND NEW.status IN ('done', 'dead')
                  AND OLD.status IS DISTINCT FROM NEW.status)
            EXECUTE FUNCTION rf._connector_poll_outcome();

        -- ------------------------------------------------------------ reconcile outcome
        -- The reconcile claim only takes a SHORT lease; the next nightly run is scheduled here,
        -- when the reconcile actually ends (done or dead), so a claim whose enqueue was lost
        -- is retried one lease later instead of a whole interval later.
        CREATE FUNCTION rf._connector_reconcile_outcome() RETURNS trigger
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_raw text := NEW.payload ->> 'connector_id';
            BEGIN
                IF v_raw IS NULL OR v_raw !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' THEN
                    RETURN NULL;
                END IF;
                UPDATE rf.connectors c
                   SET next_reconcile_at = greatest(c.next_reconcile_at, now() + make_interval(
                           secs => c.reconcile_interval_s * (0.95 + 0.1 * random())))
                 WHERE c.id = v_raw::uuid AND c.tenant_id = NEW.tenant_id;
                RETURN NULL;
            END
            $$;
        REVOKE ALL ON FUNCTION rf._connector_reconcile_outcome() FROM PUBLIC;
        CREATE TRIGGER jobs_connector_reconcile_outcome
            AFTER UPDATE OF status ON rf.jobs
            FOR EACH ROW
            WHEN (NEW.kind = 'reconcile_deletes' AND NEW.status IN ('done', 'dead')
                  AND OLD.status IS DISTINCT FROM NEW.status)
            EXECUTE FUNCTION rf._connector_reconcile_outcome();

        -- ------------------------------------------------------------ reclaim (replaces 0002)
        -- Differs from 0002's in two ways: it bumps lease_version BEFORE requeueing or burying an
        -- expired job (0002 left the version alone, so a zombie holder's heartbeat could not tell
        -- "I finished it" from "reclaim took it"; see rf.heartbeat below), and a requeue is backed
        -- off like a failure instead of runnable at once.
        CREATE OR REPLACE FUNCTION rf.reclaim_expired_leases(p_limit integer DEFAULT 500)
            RETURNS integer
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_job   rf.jobs;
                v_count integer := 0;
            BEGIN
                FOR v_job IN
                    SELECT * FROM rf.jobs j
                     WHERE j.status = 'running' AND j.lease_expires_at < now()
                     ORDER BY j.lease_expires_at
                     LIMIT greatest(coalesce(p_limit, 500), 1)
                     FOR UPDATE SKIP LOCKED
                LOOP
                    UPDATE rf.jobs j SET lease_version = j.lease_version + 1
                     WHERE j.id = v_job.id;
                    IF v_job.attempts >= v_job.max_attempts THEN
                        PERFORM rf._bury_job(v_job.id, 'lease expired (worker lost)', false);
                    ELSE
                        -- Backed off like a failure (0002 requeued at now()): a job that kills
                        -- its worker must not kill the next replica within seconds.
                        UPDATE rf.jobs j
                           SET last_error = 'lease expired (worker lost)',
                               run_after = now() + make_interval(secs =>
                                   rf._job_backoff_s(v_job.attempts, random()))
                         WHERE j.id = v_job.id;
                        PERFORM rf.requeue(v_job.id);
                    END IF;
                    v_count := v_count + 1;
                END LOOP;
                RETURN v_count;
            END
            $$;
        REVOKE ALL ON FUNCTION rf.reclaim_expired_leases(integer) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.reclaim_expired_leases(integer) TO rf_worker;

        -- ------------------------------------------------------------ heartbeat (replaces 0002)
        -- Never blocks. false = another holder took the job: it is gone, or its lease_version
        -- moved on (a new claim, or reclaim, which bumps it). Locked by someone else = the
        -- holder's own fenced result transaction: alive. Any other state at THIS version
        -- (done, dead, failed, superseded, or queued after the holder's own retryable fail or
        -- release) was reached by this holder: not a loss (see the module doc).
        CREATE OR REPLACE FUNCTION rf.heartbeat(p_job_id uuid, p_lease_version integer)
            RETURNS boolean
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_id uuid;
            BEGIN
                SELECT j.id INTO v_id FROM rf.jobs j
                 WHERE j.id = p_job_id AND j.status = 'running'
                   AND j.lease_version = p_lease_version
                   FOR UPDATE SKIP LOCKED;
                IF FOUND THEN
                    UPDATE rf.jobs j SET lease_expires_at = now() + rf._job_lease()
                     WHERE j.id = v_id;
                    RETURN true;
                END IF;
                PERFORM 1 FROM rf.jobs j
                 WHERE j.id = p_job_id AND j.lease_version = p_lease_version;
                RETURN FOUND;
            END
            $$;
        REVOKE ALL ON FUNCTION rf.heartbeat(uuid, integer) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.heartbeat(uuid, integer) TO rf_worker;

        -- ------------------------------------------------------------ defer_job
        -- Like release_job (no attempt charged, merges into a queued duplicate) but not runnable
        -- before p_delay_s (clamped to 1 s .. 1 h). Bounded: a job enqueued more than 24 h ago
        -- that a worker still cannot run (a removed kind, a typo'd enqueue, a rollback after
        -- new-kind jobs were enqueued) is buried permanently as 'dead' with an audit row,
        -- instead of ping-ponging forever with no signal. Returns queued | merged | dead | lost.
        CREATE FUNCTION rf.defer_job(p_job_id uuid, p_lease_version integer, p_delay_s integer)
            RETURNS text
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_job rf.jobs;
            BEGIN
                SELECT * INTO v_job FROM rf.jobs j WHERE j.id = p_job_id FOR UPDATE;
                IF NOT FOUND OR v_job.status <> 'running'
                   OR v_job.lease_version <> p_lease_version THEN
                    RETURN 'lost';
                END IF;
                IF v_job.enqueued_at < now() - interval '24 hours' THEN
                    PERFORM rf._bury_job(p_job_id,
                        left('no handler for job kind ' || v_job.kind, 500), true);
                    RETURN 'dead';
                END IF;
                UPDATE rf.jobs j
                   SET attempts = greatest(j.attempts - 1, 0),
                       run_after = now() + make_interval(secs =>
                           least(greatest(coalesce(p_delay_s, 60), 1), 3600))
                 WHERE j.id = p_job_id;
                RETURN CASE WHEN rf.requeue(p_job_id) = p_job_id THEN 'queued' ELSE 'merged' END;
            END
            $$;
        REVOKE ALL ON FUNCTION rf.defer_job(uuid, integer, integer) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.defer_job(uuid, integer, integer) TO rf_worker;

        -- ------------------------------------------------------------ claim_due_connectors
        CREATE FUNCTION rf.claim_due_connectors(p_limit integer, p_lease_s integer)
            RETURNS TABLE (connector_id uuid, tenant_id uuid)
            LANGUAGE sql VOLATILE SECURITY DEFINER
            SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            WITH due AS (
                SELECT c.id
                  FROM rf.connectors c
                 WHERE c.status = 'active' AND c.next_poll_at <= now()
                   AND NOT EXISTS (
                       SELECT 1 FROM rf.jobs j
                        WHERE j.tenant_id = c.tenant_id
                          AND j.dedupe_key = 'sync_connector:' || c.id::text
                          AND j.status IN ('queued', 'running'))
                 ORDER BY c.next_poll_at
                 LIMIT least(greatest(coalesce(p_limit, 100), 1), 1000)
                   FOR UPDATE OF c SKIP LOCKED
            )
            UPDATE rf.connectors c
               SET next_poll_at = now() + make_interval(secs =>
                       least(greatest(coalesce(p_lease_s, 300), 30), 86400))
              FROM due
             WHERE c.id = due.id
            RETURNING c.id, c.tenant_id
            $$;
        REVOKE ALL ON FUNCTION rf.claim_due_connectors(integer, integer) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.claim_due_connectors(integer, integer) TO rf_worker;

        -- ------------------------------------------------------------ claim_reconcile_connectors
        CREATE FUNCTION rf.claim_reconcile_connectors(
            p_limit integer, p_interval_s integer, p_lease_s integer)
            RETURNS TABLE (connector_id uuid, tenant_id uuid)
            LANGUAGE sql VOLATILE SECURITY DEFINER
            SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            WITH due AS (
                SELECT c.id
                  FROM rf.connectors c
                 WHERE c.status = 'active' AND c.next_reconcile_at <= now()
                   AND NOT EXISTS (
                       SELECT 1 FROM rf.jobs j
                        WHERE j.tenant_id = c.tenant_id
                          AND j.dedupe_key = 'reconcile_deletes:' || c.id::text
                          AND j.status IN ('queued', 'running'))
                 ORDER BY c.next_reconcile_at
                 LIMIT least(greatest(coalesce(p_limit, 100), 1), 1000)
                   FOR UPDATE OF c SKIP LOCKED
            )
            UPDATE rf.connectors c
               SET next_reconcile_at = now() + make_interval(secs =>
                       least(greatest(coalesce(p_lease_s, 300), 30), 86400)),
                   reconcile_interval_s = least(greatest(coalesce(p_interval_s, 86400), 60),
                                                604800)
              FROM due
             WHERE c.id = due.id
            RETURNING c.id, c.tenant_id
            $$;
        REVOKE ALL ON FUNCTION rf.claim_reconcile_connectors(integer, integer, integer)
            FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.claim_reconcile_connectors(integer, integer, integer)
            TO rf_worker;

        -- ------------------------------------------------------------ assert_lease (replaces 0002)
        -- Same fence; once it passes, the lease is extended from the CURRENT time (not the
        -- transaction start), so a long fenced transaction that commits keeps its job leased.
        CREATE OR REPLACE FUNCTION rf.assert_lease(p_job_id uuid, p_lease_version integer)
            RETURNS void
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
            DECLARE
                v_job rf.jobs;
            BEGIN
                SELECT * INTO v_job FROM rf.jobs j WHERE j.id = p_job_id FOR UPDATE;
                IF NOT FOUND
                   OR v_job.status <> 'running'
                   OR v_job.lease_version <> p_lease_version
                   OR v_job.tenant_id IS DISTINCT FROM rf.current_tenant() THEN
                    RAISE EXCEPTION USING ERRCODE = 'RFL01',
                        MESSAGE = 'lease lost for job ' || p_job_id::text;  -- no percent: run_sql
                END IF;
                UPDATE rf.jobs j SET lease_expires_at = clock_timestamp() + rf._job_lease()
                 WHERE j.id = p_job_id;
            END
            $$;

        -- ------------------------------------------------------------ old scheduler = a claim
        -- Same signature and grant (OR REPLACE keeps the ACL); now VOLATILE because it claims.
        -- Exactly claim_due_connectors(p_limit, 300): a pre-0008 worker (rolling deploy or a
        -- rollback) keeps polling, without twins or backoff resets.
        CREATE OR REPLACE FUNCTION rf.list_due_connectors(p_limit integer DEFAULT 100)
            RETURNS TABLE (connector_id uuid, tenant_id uuid)
            LANGUAGE sql VOLATILE SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
            AS $$
                SELECT d.connector_id, d.tenant_id FROM rf.claim_due_connectors(p_limit, 300) d
            $$;
        """
    )
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
