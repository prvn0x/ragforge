"""Module A (data plane): enqueue_job, claim_jobs, assert_lease, heartbeat, complete_job,
fail_job, release_job, reclaim_expired_leases, report_job_progress, requeue (internal),
connector_inflight upkeep, move_finished_jobs + monthly jobs_history partitions,
list_due_connectors, queue_stats view (ADR-0010, ADR-0013, ADR-0003).

OWNED BY MODULE A. Only module A edits it.

Design notes (the WHY; the ADRs hold the decisions):
- Every function that a login role can call is SECURITY DEFINER owned by rf_system with
  `SET search_path = pg_catalog, rf, ext, pg_temp`, REVOKE ALL FROM PUBLIC and an explicit GRANT.
  Internal helpers (`requeue`, `_claim_one`, ...) are SECURITY INVOKER and granted to nobody:
  they only run inside the definer functions, where current_user is already rf_system.
- Tenant-scoped entry points (enqueue_job, assert_lease, complete_job, report_job_progress) take
  the tenant from `rf.current_tenant()` and refuse to touch another tenant's job, so a worker that
  opened its result transaction under the wrong tenant fails closed (LeaseLost) instead of
  fencing someone else's job.
- Lease loss is signalled with the custom SQLSTATE `RFL01` so the adapter can map it to
  `domain.errors.LeaseLost` without parsing messages.
- connector_inflight is incremented in claim (under the counter row lock, re-checked after the
  lock) and decremented by a trigger on every `running -> *` transition and on delete of a running
  job. Putting the decrement in a trigger makes "decrement on every exit path" structural: no
  future exit path can forget it.
- NOTIFY payloads carry only the job id (PHASE1 conventions); listeners re-read under RLS.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

# Every function below spells out its security clause literally (no string interpolation) so the
# SQL is reviewable as-is; the catalog invariant test checks the definer search_path.


def _helpers() -> str:
    return """
    -- ------------------------------------------------------------ constants
    -- In-flight cap per connector (DESIGN section 3, Flow A step 4).
    CREATE FUNCTION rf._connector_inflight_cap() RETURNS integer
        LANGUAGE sql IMMUTABLE PARALLEL SAFE SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$ SELECT 2 $$;

    -- Lease length (ADR-0010). Workers heartbeat at a third of it.
    CREATE FUNCTION rf._job_lease() RETURNS interval
        LANGUAGE sql IMMUTABLE PARALLEL SAFE SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$ SELECT interval '60 seconds' $$;

    -- Backoff: min(2^attempts * 5 s, 15 min) +/- 20 %. `p_jitter` is uniform in [0, 1); a pure
    -- function so the bounds are testable without sampling random().
    CREATE FUNCTION rf._job_backoff_s(p_attempts integer, p_jitter double precision)
        RETURNS double precision
        LANGUAGE sql IMMUTABLE PARALLEL SAFE SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
            SELECT least(5 * power(2, least(greatest(p_attempts, 0), 30)), 900)
                   * (0.8 + 0.4 * least(greatest(p_jitter, 0), 1))
        $$;

    -- ------------------------------------------------------------ connector in-flight upkeep
    -- Decrement on every exit from `running` (complete, fail, dead, release, reclaim, merge,
    -- tenant cascade delete). DEFINER because cascades can fire it as a non-owner.
    CREATE FUNCTION rf._jobs_release_inflight() RETURNS trigger
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        BEGIN
            IF OLD.connector_id IS NOT NULL THEN
                UPDATE rf.connector_inflight ci
                   SET running = ci.running - 1
                 WHERE ci.connector_id = OLD.connector_id AND ci.running > 0;
            END IF;
            RETURN NULL;
        END
        $$;
    CREATE TRIGGER jobs_release_inflight
        AFTER UPDATE OF status ON rf.jobs
        FOR EACH ROW WHEN (OLD.status = 'running' AND NEW.status <> 'running')
        EXECUTE FUNCTION rf._jobs_release_inflight();
    CREATE TRIGGER jobs_release_inflight_on_delete
        AFTER DELETE ON rf.jobs
        FOR EACH ROW WHEN (OLD.status = 'running')
        EXECUTE FUNCTION rf._jobs_release_inflight();

    -- ------------------------------------------------------------ requeue (internal, race-safe)
    -- Precondition: the caller holds the row lock on p_job_id and the row is `running`.
    -- Returns the id of the row that is now queued (p_job_id itself, or the duplicate it merged
    -- into). The loop terminates: each iteration either succeeds, merges, or observed the
    -- duplicate leave `queued` (claimed), after which the retry cannot conflict with it.
    CREATE FUNCTION rf.requeue(p_job_id uuid) RETURNS uuid
        LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_me  rf.jobs;
            v_dup uuid;
        BEGIN
            SELECT * INTO STRICT v_me FROM rf.jobs j WHERE j.id = p_job_id;
            LOOP
                BEGIN
                    UPDATE rf.jobs j
                       SET status = 'queued', lease_expires_at = NULL
                     WHERE j.id = p_job_id;
                    RETURN p_job_id;
                EXCEPTION WHEN unique_violation THEN
                    -- A queued duplicate exists (same tenant + dedupe_key). The payload is a pure
                    -- function of the key, so merging keeps the most urgent schedule and loses
                    -- nothing.
                    UPDATE rf.jobs d
                       SET priority  = least(d.priority, v_me.priority),
                           run_after = least(d.run_after, v_me.run_after)
                     WHERE d.tenant_id = v_me.tenant_id
                       AND d.dedupe_key = v_me.dedupe_key
                       AND d.status = 'queued'
                       AND d.id <> p_job_id
                    RETURNING d.id INTO v_dup;
                    IF v_dup IS NOT NULL THEN
                        UPDATE rf.jobs j
                           SET status = 'superseded', lease_expires_at = NULL,
                               finished_at = now()
                         WHERE j.id = p_job_id;
                        RETURN v_dup;
                    END IF;
                    -- The duplicate was claimed between the violation and our UPDATE: retry.
                END;
            END LOOP;
        END
        $$;

    -- ------------------------------------------------------------ dead letter (internal)
    CREATE FUNCTION rf._bury_job(p_job_id uuid, p_error text, p_permanent boolean)
        RETURNS void
        LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_job rf.jobs;
        BEGIN
            UPDATE rf.jobs j
               SET status = 'dead', lease_expires_at = NULL, finished_at = now(),
                   last_error = p_error
             WHERE j.id = p_job_id
            RETURNING * INTO STRICT v_job;
            -- Audit in the same transaction as the state change (ADR-0003).
            INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
            VALUES (v_job.tenant_id, 'system:job-queue', 'job.dead', 'job:' || v_job.id::text,
                    jsonb_build_object(
                        'kind', v_job.kind,
                        'attempts', v_job.attempts,
                        'permanent', p_permanent,
                        'error', left(p_error, 500)));
        END
        $$;

    -- ------------------------------------------------------------ claim one job of one class
    -- Uses the (priority, enqueued_at) partial index: one class per query, no computed sort key.
    -- Connectors at their cap are filtered inside the query (no head-of-line blocking) and
    -- re-checked under the counter row lock, because the filter alone races with other claimers.
    CREATE FUNCTION rf._claim_one(p_worker_id text, p_priority smallint) RETURNS rf.jobs
        LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_rejected uuid[] := '{}';
            v_id       uuid;
            v_conn     uuid;
            v_running  integer;
            v_job      rf.jobs;
        BEGIN
            LOOP
                SELECT j.id, j.connector_id INTO v_id, v_conn
                  FROM rf.jobs j
                 WHERE j.status = 'queued'
                   AND j.priority = p_priority
                   AND j.run_after <= now()
                   AND j.id <> ALL (v_rejected)
                   AND (j.connector_id IS NULL OR NOT EXISTS (
                        SELECT 1 FROM rf.connector_inflight ci
                         WHERE ci.connector_id = j.connector_id
                           AND ci.running >= rf._connector_inflight_cap()))
                 ORDER BY j.enqueued_at
                 LIMIT 1
                 FOR UPDATE OF j SKIP LOCKED;
                IF NOT FOUND THEN
                    RETURN NULL;
                END IF;

                IF v_conn IS NOT NULL THEN
                    -- SKIP LOCKED, not a blocking lock: two claimers taking two connectors in
                    -- opposite order would otherwise deadlock. A busy counter just means "try
                    -- the next job"; the skipped one is retried on the next claim.
                    SELECT ci.running INTO v_running
                      FROM rf.connector_inflight ci
                     WHERE ci.connector_id = v_conn
                       FOR UPDATE SKIP LOCKED;
                    IF NOT FOUND THEN
                        -- Normally created by enqueue_job; create defensively if missing.
                        INSERT INTO rf.connector_inflight (connector_id) VALUES (v_conn)
                            ON CONFLICT DO NOTHING;
                        IF NOT FOUND THEN  -- exists but locked by someone else
                            v_rejected := v_rejected || v_id;
                            CONTINUE;
                        END IF;
                        v_running := 0;
                    END IF;
                    IF v_running >= rf._connector_inflight_cap() THEN
                        v_rejected := v_rejected || v_id;
                        CONTINUE;
                    END IF;
                    UPDATE rf.connector_inflight ci
                       SET running = ci.running + 1
                     WHERE ci.connector_id = v_conn;
                END IF;

                UPDATE rf.jobs j
                   SET status = 'running',
                       leased_by = p_worker_id,
                       lease_version = j.lease_version + 1,
                       lease_expires_at = now() + rf._job_lease(),
                       attempts = j.attempts + 1
                 WHERE j.id = v_id
                RETURNING * INTO v_job;
                RETURN v_job;
            END LOOP;
        END
        $$;

    -- Starvation guard target: whichever lower class (maintenance, backfill) has the oldest
    -- runnable head job. Two index probes instead of one sort over both classes.
    CREATE FUNCTION rf._oldest_low_class() RETURNS smallint
        LANGUAGE sql STABLE SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
            SELECT x.priority FROM (
                (SELECT j.priority, j.enqueued_at FROM rf.jobs j
                  WHERE j.status = 'queued' AND j.priority = 30 AND j.run_after <= now()
                  ORDER BY j.enqueued_at LIMIT 1)
                UNION ALL
                (SELECT j.priority, j.enqueued_at FROM rf.jobs j
                  WHERE j.status = 'queued' AND j.priority = 40 AND j.run_after <= now()
                  ORDER BY j.enqueued_at LIMIT 1)
            ) x
            ORDER BY x.enqueued_at
            LIMIT 1
        $$;

    -- ------------------------------------------------------------ jobs_history partitions
    -- Monthly RANGE partitions on finished_at (UTC month boundaries, independent of the session
    -- TimeZone). Dynamic SQL is limited to identifiers/literals computed from a timestamp; no
    -- caller-supplied text reaches it.
    CREATE FUNCTION rf._ensure_jobs_history_partition(p_at timestamptz) RETURNS void
        LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_from timestamp := date_trunc('month', p_at AT TIME ZONE 'UTC');
            v_to   timestamp := v_from + interval '1 month';
            v_name text := 'jobs_history_p' || to_char(v_from, 'YYYYMM');
        BEGIN
            IF to_regclass('rf.' || v_name) IS NOT NULL THEN
                RETURN;
            END IF;
            EXECUTE format(
                'CREATE TABLE rf.%I PARTITION OF rf.jobs_history FOR VALUES FROM (%L) TO (%L)',
                v_name,
                to_char(v_from, 'YYYY-MM-DD"T"HH24:MI:SS') || '+00',
                to_char(v_to, 'YYYY-MM-DD"T"HH24:MI:SS') || '+00');
            EXECUTE format('COMMENT ON TABLE rf.%I IS %L', v_name, 'class: derived');
        END
        $$;
    """


def _api_functions() -> str:
    return """
    -- ------------------------------------------------------------ enqueue (api, worker)
    CREATE FUNCTION rf.enqueue_job(
        p_kind         text,
        p_priority     integer,
        p_payload      jsonb,
        p_dedupe_key   text,
        p_run_after    timestamptz DEFAULT NULL,
        p_connector_id uuid DEFAULT NULL
    ) RETURNS uuid
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_tenant uuid := rf.current_tenant();
            v_id     uuid;
        BEGIN
            IF v_tenant IS NULL THEN
                RAISE EXCEPTION 'enqueue_job requires a tenant transaction'
                    USING ERRCODE = 'insufficient_privilege';
            END IF;
            IF coalesce(length(p_kind), 0) NOT BETWEEN 1 AND 100
               OR coalesce(length(p_dedupe_key), 0) NOT BETWEEN 1 AND 500 THEN
                RAISE EXCEPTION 'invalid job kind or dedupe_key length'
                    USING ERRCODE = 'invalid_parameter_value';
            END IF;
            IF p_priority IS NULL OR p_priority NOT IN (0, 10, 20, 30, 40) THEN
                RAISE EXCEPTION 'invalid job priority %', p_priority
                    USING ERRCODE = 'invalid_parameter_value';
            END IF;
            IF pg_column_size(coalesce(p_payload, '{}'::jsonb)) > 65536 THEN
                RAISE EXCEPTION 'job payload too large' USING ERRCODE = 'invalid_parameter_value';
            END IF;

            -- Tenant-scoped dedupe on queued jobs only (a running job may have read stale
            -- state, so a new queued row is correct). If the conflicting row is concurrently
            -- claimed, Postgres retries the speculative insert and inserts a fresh row.
            INSERT INTO rf.jobs AS j
                (tenant_id, connector_id, kind, priority, payload, dedupe_key, run_after)
            VALUES (v_tenant, p_connector_id, p_kind, p_priority::smallint,
                    coalesce(p_payload, '{}'::jsonb), p_dedupe_key,
                    coalesce(p_run_after, now()))
            ON CONFLICT (tenant_id, dedupe_key) WHERE status = 'queued'
            DO UPDATE SET priority  = least(j.priority, EXCLUDED.priority),
                          run_after = least(j.run_after, EXCLUDED.run_after)
            RETURNING j.id INTO v_id;

            -- The FK (tenant_id, connector_id) above already proved the connector is this
            -- tenant's. Creating the counter row here keeps claim free of first-insert waits.
            IF p_connector_id IS NOT NULL THEN
                INSERT INTO rf.connector_inflight (connector_id) VALUES (p_connector_id)
                    ON CONFLICT DO NOTHING;
            END IF;

            PERFORM pg_notify('rf_jobs', v_id::text);
            RETURN v_id;
        END
        $$;
    """


def _worker_functions() -> str:
    return """
    -- ------------------------------------------------------------ claim (worker; cross-tenant)
    CREATE FUNCTION rf.claim_jobs(p_worker_id text, p_n integer, p_prefer_low boolean DEFAULT false)
        RETURNS TABLE (job_id uuid, tenant_id uuid, kind text, payload jsonb,
                       lease_version integer, attempts integer)
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_job   rf.jobs;
            v_low   smallint;
            v_p     smallint;
            v_taken integer := 0;
        BEGIN
            IF p_n IS NULL OR p_n NOT BETWEEN 1 AND 100 THEN
                RAISE EXCEPTION 'claim size must be 1..100' USING ERRCODE = 'invalid_parameter_value';
            END IF;
            IF coalesce(length(p_worker_id), 0) NOT BETWEEN 1 AND 200 THEN
                RAISE EXCEPTION 'invalid worker id' USING ERRCODE = 'invalid_parameter_value';
            END IF;

            -- Starvation guard (ADR-0010): the worker sets p_prefer_low on every 4th claim; the
            -- first job then comes from the lower class with the oldest head.
            IF p_prefer_low THEN
                v_low := rf._oldest_low_class();
                IF v_low IS NOT NULL THEN
                    v_job := rf._claim_one(p_worker_id, v_low);
                    IF v_job.id IS NOT NULL THEN
                        job_id := v_job.id; tenant_id := v_job.tenant_id; kind := v_job.kind;
                        payload := v_job.payload; lease_version := v_job.lease_version;
                        attempts := v_job.attempts;
                        RETURN NEXT;
                        v_taken := 1;
                    END IF;
                END IF;
            END IF;

            -- Strict class order, no cross-class aging.
            FOREACH v_p IN ARRAY ARRAY[0, 10, 20, 30, 40]::smallint[] LOOP
                EXIT WHEN v_taken >= p_n;
                LOOP
                    EXIT WHEN v_taken >= p_n;
                    v_job := rf._claim_one(p_worker_id, v_p);
                    EXIT WHEN v_job.id IS NULL;
                    job_id := v_job.id; tenant_id := v_job.tenant_id; kind := v_job.kind;
                    payload := v_job.payload; lease_version := v_job.lease_version;
                    attempts := v_job.attempts;
                    RETURN NEXT;
                    v_taken := v_taken + 1;
                END LOOP;
            END LOOP;
        END
        $$;

    -- ------------------------------------------------------------ fencing (tenant tx)
    -- First statement of every result transaction. Row-locks the job so reclaim (which takes
    -- the same lock) cannot interleave between this check and the commit.
    CREATE FUNCTION rf.assert_lease(p_job_id uuid, p_lease_version integer) RETURNS void
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
                RAISE EXCEPTION 'lease lost for job %', p_job_id USING ERRCODE = 'RFL01';
            END IF;
        END
        $$;

    CREATE FUNCTION rf.complete_job(p_job_id uuid, p_lease_version integer) RETURNS void
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        BEGIN
            UPDATE rf.jobs j
               SET status = 'done', lease_expires_at = NULL, finished_at = now()
             WHERE j.id = p_job_id
               AND j.status = 'running'
               AND j.lease_version = p_lease_version
               AND j.tenant_id = rf.current_tenant();
            IF NOT FOUND THEN
                RAISE EXCEPTION 'lease lost for job %', p_job_id USING ERRCODE = 'RFL01';
            END IF;
        END
        $$;

    CREATE FUNCTION rf.report_job_progress(p_job_id uuid, p_lease_version integer, p_progress jsonb)
        RETURNS void
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        BEGIN
            IF pg_column_size(coalesce(p_progress, '{}'::jsonb)) > 16384 THEN
                RAISE EXCEPTION 'progress too large' USING ERRCODE = 'invalid_parameter_value';
            END IF;
            UPDATE rf.jobs j
               SET progress = coalesce(p_progress, '{}'::jsonb)
             WHERE j.id = p_job_id
               AND j.status = 'running'
               AND j.lease_version = p_lease_version
               AND j.tenant_id = rf.current_tenant();
            IF NOT FOUND THEN
                RAISE EXCEPTION 'lease lost for job %', p_job_id USING ERRCODE = 'RFL01';
            END IF;
            -- Delivered on commit; id only (listeners re-read the row under RLS).
            PERFORM pg_notify('rf_job_progress', p_job_id::text);
        END
        $$;

    -- ------------------------------------------------------------ lease lifecycle (own tx)
    CREATE FUNCTION rf.heartbeat(p_job_id uuid, p_lease_version integer) RETURNS boolean
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        BEGIN
            UPDATE rf.jobs j
               SET lease_expires_at = now() + rf._job_lease()
             WHERE j.id = p_job_id AND j.status = 'running' AND j.lease_version = p_lease_version;
            RETURN FOUND;
        END
        $$;

    -- Returns 'queued' | 'merged' | 'dead' | 'lost' (lease no longer held: no-op).
    CREATE FUNCTION rf.fail_job(
        p_job_id        uuid,
        p_lease_version integer,
        p_error         text,
        p_retry_after_s double precision DEFAULT NULL,
        p_permanent     boolean DEFAULT false
    ) RETURNS text
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_job   rf.jobs;
            v_error text := left(coalesce(p_error, ''), 2000);
            v_delay double precision;
        BEGIN
            SELECT * INTO v_job FROM rf.jobs j WHERE j.id = p_job_id FOR UPDATE;
            IF NOT FOUND OR v_job.status <> 'running' OR v_job.lease_version <> p_lease_version THEN
                RETURN 'lost';
            END IF;
            IF coalesce(p_permanent, false) OR v_job.attempts >= v_job.max_attempts THEN
                PERFORM rf._bury_job(p_job_id, v_error, coalesce(p_permanent, false));
                RETURN 'dead';
            END IF;
            -- Retry-After (upstream's explicit instruction) overrides backoff; clamped so a bad
            -- header cannot park a job for days.
            v_delay := CASE
                WHEN p_retry_after_s IS NOT NULL THEN least(greatest(p_retry_after_s, 0), 3600)
                ELSE rf._job_backoff_s(v_job.attempts, random())
            END;
            UPDATE rf.jobs j
               SET last_error = v_error, run_after = now() + make_interval(secs => v_delay)
             WHERE j.id = p_job_id;
            RETURN CASE WHEN rf.requeue(p_job_id) = p_job_id THEN 'queued' ELSE 'merged' END;
        END
        $$;

    -- Graceful shutdown: give the job back without charging an attempt.
    CREATE FUNCTION rf.release_job(p_job_id uuid, p_lease_version integer) RETURNS text
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_job rf.jobs;
        BEGIN
            SELECT * INTO v_job FROM rf.jobs j WHERE j.id = p_job_id FOR UPDATE;
            IF NOT FOUND OR v_job.status <> 'running' OR v_job.lease_version <> p_lease_version THEN
                RETURN 'lost';
            END IF;
            UPDATE rf.jobs j SET attempts = greatest(j.attempts - 1, 0) WHERE j.id = p_job_id;
            RETURN CASE WHEN rf.requeue(p_job_id) = p_job_id THEN 'queued' ELSE 'merged' END;
        END
        $$;

    -- Requeue (or bury, past max_attempts: a job that keeps killing workers must stop) every
    -- running job whose lease expired. SKIP LOCKED: a job whose holder is inside its fenced
    -- result transaction is left alone; that holder still owns it and may legitimately commit.
    -- FOR UPDATE re-evaluates the expiry on the latest row version, so a heartbeat that commits
    -- first wins.
    CREATE FUNCTION rf.reclaim_expired_leases(p_limit integer DEFAULT 500) RETURNS integer
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
                IF v_job.attempts >= v_job.max_attempts THEN
                    PERFORM rf._bury_job(v_job.id, 'lease expired (worker lost)', false);
                ELSE
                    UPDATE rf.jobs j
                       SET last_error = 'lease expired (worker lost)', run_after = now()
                     WHERE j.id = v_job.id;
                    PERFORM rf.requeue(v_job.id);
                END IF;
                v_count := v_count + 1;
            END LOOP;
            RETURN v_count;
        END
        $$;

    -- ------------------------------------------------------------ upkeep (worker, every 5 min)
    -- Moves finished jobs older than p_min_age into monthly jobs_history partitions, creating
    -- every partition a moved row needs (plus current and next month) FIRST, so the DEFAULT
    -- partition stays empty (a non-empty DEFAULT would block creating the matching partition).
    -- Drops whole partitions past retention. Serialized by an advisory xact lock because
    -- partition DDL from two workers would race on the same name.
    CREATE FUNCTION rf.move_finished_jobs(
        p_min_age   interval DEFAULT interval '5 minutes',
        p_limit     integer  DEFAULT 10000,
        p_retention interval DEFAULT interval '30 days'
    ) RETURNS integer
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
        DECLARE
            v_month timestamptz;
            v_part  record;
            v_moved integer;
        BEGIN
            IF p_retention IS NULL OR p_retention < interval '1 day' THEN
                RAISE EXCEPTION 'retention must be at least 1 day'
                    USING ERRCODE = 'invalid_parameter_value';
            END IF;
            PERFORM pg_advisory_xact_lock(hashtext('rf.move_finished_jobs'));

            PERFORM rf._ensure_jobs_history_partition(now());
            PERFORM rf._ensure_jobs_history_partition(now() + interval '1 month');
            FOR v_month IN
                SELECT DISTINCT date_trunc('month', j.finished_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
                  FROM rf.jobs j
                 WHERE j.finished_at IS NOT NULL AND j.finished_at < now() - p_min_age
            LOOP
                PERFORM rf._ensure_jobs_history_partition(v_month);
            END LOOP;

            WITH moved AS (
                DELETE FROM rf.jobs j
                 WHERE j.id IN (
                     SELECT f.id FROM rf.jobs f
                      WHERE f.status IN ('done', 'failed', 'dead', 'superseded')
                        AND f.finished_at IS NOT NULL
                        AND f.finished_at < now() - p_min_age
                      ORDER BY f.finished_at
                      LIMIT greatest(coalesce(p_limit, 10000), 1)
                      FOR UPDATE SKIP LOCKED)
                RETURNING j.*
            )
            INSERT INTO rf.jobs_history SELECT * FROM moved;
            GET DIAGNOSTICS v_moved = ROW_COUNT;

            FOR v_part IN
                SELECT c.relname
                  FROM pg_catalog.pg_inherits i
                  JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid
                 WHERE i.inhparent = 'rf.jobs_history'::regclass
                   AND c.relname ~ '^jobs_history_p[0-9]{6}$'
                   AND (to_date(substr(c.relname, 15), 'YYYYMM')::timestamp
                        + interval '1 month') AT TIME ZONE 'UTC' <= now() - p_retention
            LOOP
                EXECUTE format('DROP TABLE rf.%I', v_part.relname);
            END LOOP;
            RETURN v_moved;
        END
        $$;

    -- Scheduler input (DESIGN Flow A step 2): cross-tenant, ids only.
    CREATE FUNCTION rf.list_due_connectors(p_limit integer DEFAULT 100)
        RETURNS TABLE (connector_id uuid, tenant_id uuid)
        LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, rf, ext, pg_temp
        AS $$
            SELECT c.id, c.tenant_id
              FROM rf.connectors c
             WHERE c.status = 'active' AND c.next_poll_at <= now()
             ORDER BY c.next_poll_at
             LIMIT greatest(coalesce(p_limit, 100), 1)
        $$;
    """


def _views_indexes_grants() -> str:
    return """
    -- ------------------------------------------------------------ metrics view
    -- The ONE view allowed without security_invoker (ADR-0013): aggregates only, no tenant ids.
    CREATE VIEW rf.queue_stats AS
        SELECT j.priority,
               j.kind,
               j.status,
               count(*)::bigint AS jobs,
               count(*) FILTER (WHERE j.status = 'queued' AND j.run_after <= now())::bigint
                   AS ready,
               count(*) FILTER (WHERE j.status = 'running' AND j.lease_expires_at < now())::bigint
                   AS expired_leases,
               min(j.enqueued_at) FILTER (WHERE j.status = 'queued') AS oldest_queued_at
          FROM rf.jobs j
         GROUP BY j.priority, j.kind, j.status;
    COMMENT ON VIEW rf.queue_stats IS 'aggregates only; intentionally not security_invoker';

    -- move_finished_jobs scans finished rows by age.
    CREATE INDEX jobs_finished ON rf.jobs (finished_at) WHERE finished_at IS NOT NULL;

    -- ------------------------------------------------------------ privileges (least privilege)
    REVOKE ALL ON FUNCTION
        rf._connector_inflight_cap(), rf._job_lease(),
        rf._job_backoff_s(integer, double precision), rf._jobs_release_inflight(),
        rf.requeue(uuid), rf._bury_job(uuid, text, boolean),
        rf._claim_one(text, smallint), rf._oldest_low_class(),
        rf._ensure_jobs_history_partition(timestamptz),
        rf.enqueue_job(text, integer, jsonb, text, timestamptz, uuid),
        rf.claim_jobs(text, integer, boolean),
        rf.assert_lease(uuid, integer), rf.complete_job(uuid, integer),
        rf.report_job_progress(uuid, integer, jsonb), rf.heartbeat(uuid, integer),
        rf.fail_job(uuid, integer, text, double precision, boolean),
        rf.release_job(uuid, integer), rf.reclaim_expired_leases(integer),
        rf.move_finished_jobs(interval, integer, interval), rf.list_due_connectors(integer)
        FROM PUBLIC;
    REVOKE ALL ON rf.queue_stats FROM PUBLIC;

    GRANT EXECUTE ON FUNCTION rf.enqueue_job(text, integer, jsonb, text, timestamptz, uuid)
        TO rf_api, rf_worker;
    GRANT EXECUTE ON FUNCTION
        rf.claim_jobs(text, integer, boolean),
        rf.assert_lease(uuid, integer), rf.complete_job(uuid, integer),
        rf.report_job_progress(uuid, integer, jsonb), rf.heartbeat(uuid, integer),
        rf.fail_job(uuid, integer, text, double precision, boolean),
        rf.release_job(uuid, integer), rf.reclaim_expired_leases(integer),
        rf.move_finished_jobs(interval, integer, interval), rf.list_due_connectors(integer)
        TO rf_worker;
    GRANT SELECT ON rf.queue_stats TO rf_metrics;

    -- ------------------------------------------------------------ initial partitions
    SELECT rf._ensure_jobs_history_partition(now());
    SELECT rf._ensure_jobs_history_partition(now() + interval '1 month');
    """


def _run(sql: str) -> None:
    """run_sql goes through psycopg's placeholder parser even without parameters, so literal
    `%` (format() specifiers, RAISE placeholders) must be doubled."""
    run_sql(sql.replace("%", "%%"))


def upgrade() -> None:
    _run(_helpers())
    _run(_api_functions())
    _run(_worker_functions())
    _run(_views_indexes_grants())
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
