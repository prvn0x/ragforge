# ADR-0010: Job scheduling: strict class order with a starvation guard, leases, fencing inside the side-effect transaction, tenant-scoped dedupe

**Status:** Accepted · 2026-10-16 · Revised in rounds 3, 4 and 5

## Context
- Fencing only `complete_job` doesn't stop a zombie's side-effect transaction from committing stale data.
- Notion page IDs are global, so dedupe must be scoped by tenant.
- Requeueing can collide with the queued-only unique index, including through a check-then-update race.
- Advisory **xact** locks are released at commit, so they can't cap running jobs.
- **Round 5:** aging *across* classes caused inversion. After 30 min under a 5 h backfill, backfill outranked fresh incremental jobs and broke the freshness SLO. Sorting by a computed expression also prevented index use.

## Decision
- **Classes** (`priority` smallint): interactive 0, webhook 10, incremental 20, maintenance 30, backfill 40. **No cross-class aging.**
- **Claim order:** `claim_jobs(worker_id, n)` walks the classes in order and, per class, runs `WHERE status='queued' AND priority=$p AND run_after ≤ now() ORDER BY enqueued_at FOR UPDATE SKIP LOCKED LIMIT n`. That uses the `(priority, enqueued_at)` partial index.
- **Starvation guard:** each worker keeps a claim counter. **Every 4th claim** goes to whichever of the lower classes (maintenance, backfill) has the **oldest head job**.
  - Together they get ≥ 25% of claims when queued, and neither starves the other.
  - Interactive, webhook and incremental jobs keep ≥ 75% and are never queued *behind* lower classes. Their wait is bounded by (in-flight jobs ÷ workers) × job duration, which is reported as a metric rather than promised as a constant.
- **Dedupe:** `UNIQUE (tenant_id, dedupe_key) WHERE status='queued'`. Both columns are `NOT NULL`: system jobs use the reserved `SYSTEM` tenant, and `dedupe_key` defaults to `kind || hash(payload)`. **Payload is a pure function of the dedupe key**, so `DO UPDATE` keeping the old payload loses nothing.
  - `enqueue_job` takes the tenant from the setting.
  - It runs `INSERT … ON CONFLICT (tenant_id, dedupe_key) WHERE status='queued' DO UPDATE SET priority = LEAST(…), run_after = LEAST(…)`.
- **Per-connector in-flight cap (Core):** `jobs.connector_id` (nullable) plus a `connector_inflight(connector_id, running)` row, locked `FOR UPDATE` and adjusted in claim, complete, fail, release and reclaim. `claim_jobs` filters `NOT EXISTS (connector at cap)` inside the per-class query, so capped connectors are skipped instead of blocking the head of the queue. This is the same counter-row pattern as the Advanced per-tenant fairness.
- **Tenant context:** the worker calls `set_config('app.tenant_id', job.tenant_id, true)` at the start of every job transaction.
- **Fencing covers side effects.** Every result transaction begins with `assert_lease(id, lease_version)` (`SELECT … FOR UPDATE`, which raises on a version or status mismatch), and `complete_job` runs in the same transaction. Reclaim takes the same row lock and re-checks expiry after acquiring it.
  - Monotonic data guards: document upserts apply only if `fetched_at` increases; cursors use `GREATEST`.
- **Requeue merge (race-safe):** `release_job`, `reclaim_expired_leases` and the retry path of `fail_job` call `requeue(id)`, which is a PL/pgSQL loop:
  1. try `UPDATE … SET status='queued'`;
  2. on `unique_violation` (an `EXCEPTION` block), `UPDATE` the existing queued duplicate with `LEAST(priority)` / `LEAST(run_after)` and mark this row `superseded`;
  3. if that duplicate vanished meanwhile (claimed), loop and retry.

  The loop terminates because every iteration either succeeds or observes a state change.
- **Retry:** backoff `min(2^attempts × 5 s, 15 min) ± 20%`, with `Retry-After` overriding; `max_attempts` → `dead` + an audit row. Dead page syncs are repaired by nightly reconcile.
- **Wake-up:** `NOTIFY rf_jobs`; one dedicated LISTEN connection per worker; 2 s poll fallback.
- **Upkeep:** `move_finished_jobs()` every 5 min to monthly-partitioned `jobs_history` (30 d).
- **Shutdown:** stop claiming; finish within 25 s or `release_job`.
- **[Advanced] Per-tenant fairness:** a `tenant_queue_state(tenant_id, running)` row, locked `FOR UPDATE` during claim and during complete/fail/release/reclaim. Tenants at the cap are skipped.

## Consequences
- No zombie commits, no lost or duplicated queued work, no class inversion, and backfill still progresses.
- Tests:
  - N workers with random kill/pause;
  - lease expiry mid-transaction;
  - concurrent enqueue + requeue on the same key;
  - cross-tenant same-page dedupe;
  - a freshness assertion while a 10k-job backfill is queued.
