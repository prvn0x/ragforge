# Module A handoff: data plane (job queue + worker loop)

## What is built

| Path | Contents |
|---|---|
| `db/migrations/versions/0002_data_plane.py` | All queue SQL (see the function list below), the `queue_stats` view, the `jobs_finished` index, the in-flight trigger, and the initial `jobs_history` partitions (current and next month). |
| `adapters/db/jobs.py` | `PgJobQueue`: implements `JobQueue` exactly and maps SQLSTATE `RFL01` to `LeaseLost`. Extras: `claim(..., *, prefer_low=False)`, `reclaim_expired()`, `move_finished()`, `list_due_connectors()`. |
| `services/jobs/` | `JobHandlers.register(kind, async handler(lease))`, `PermanentJobError`, `StarvationGuard`, `WorkerRuntime`, `WorkerConfig`, plus the module-local ports `FairJobQueue` and `QueueMaintenance`. |
| `apps/worker/` | `main.py`: readiness gate, heartbeat file, runtime, SIGTERM drain. `listener.py`: a dedicated LISTEN `rf_jobs` connection (psycopg, not pooled) that reconnects with backoff. |
| `packages/core/tests/module_a/` | 58 tests. |

### SQL surface (slot 0002)

Every function a login role can call is SECURITY DEFINER and follows ADR-0013. Internal helpers are INVOKER and granted to no role.

| Function | Grant | Notes |
|---|---|---|
| `enqueue_job(kind text, priority int, payload jsonb, dedupe_key text, run_after timestamptz = NULL, connector_id uuid = NULL) → uuid` | rf_api, rf_worker | The tenant comes from `rf.current_tenant()` and the call raises without one. `ON CONFLICT … WHERE status='queued' DO UPDATE` takes `LEAST(priority)` and `LEAST(run_after)`. Sends `NOTIFY rf_jobs, '<id>'`. |
| `claim_jobs(worker_id text, n int, prefer_low bool = false) → (job_id, tenant_id, kind, payload, lease_version, attempts)` | rf_worker | Strict class order, one class per index probe, `FOR UPDATE SKIP LOCKED`. Connectors at the cap are filtered out and re-checked under the counter row lock. With `prefer_low`, the batch's first job comes from whichever lower class has the oldest head. Each claim increments `lease_version` and `attempts` and sets a 60 s lease. |
| `assert_lease(id, version)` | rf_worker | Takes `FOR UPDATE` and raises `RFL01` on a status, version **or tenant** mismatch. |
| `complete_job(id, version)`, `report_job_progress(id, version, jsonb)` | rf_worker | Run in the tenant tx and raise `RFL01` when the lease is lost. Progress is capped at 16 KB and sends `NOTIFY rf_job_progress, '<id>'`. |
| `heartbeat(id, version) → bool` | rf_worker | |
| `fail_job(id, version, error, retry_after_s = NULL, permanent = false) → text` | rf_worker | Backoff is `min(2^attempts·5 s, 15 min) ±20%`. `Retry-After` overrides it, clamped to 1 h. A permanent failure, or reaching max_attempts, marks the job `dead` and writes an `audit_log` row. A lost lease makes it a no-op that returns `'lost'`. |
| `release_job(id, version) → text` | rf_worker | Does not charge an attempt. Merges into a queued duplicate if one exists. |
| `reclaim_expired_leases(limit = 500) → int` | rf_worker | Uses `SKIP LOCKED`, so a holder inside its fenced tx keeps the job. A job past max_attempts is marked `dead`. |
| `move_finished_jobs(min_age = 5 min, limit = 10000, retention = 30 d) → int` | rf_worker | Creates every partition it needs first, so the DEFAULT partition stays empty. Drops partitions past retention. Serialized by an advisory xact lock. |
| `list_due_connectors(limit = 100) → (connector_id, tenant_id)` | rf_worker | |
| view `queue_stats` | rf_metrics SELECT | Aggregates only, with no tenant columns. It is the one view without `security_invoker`. |
| internal: `requeue`, `_claim_one`, `_oldest_low_class`, `_bury_job`, `_ensure_jobs_history_partition`, `_job_backoff_s`, `_job_lease`, `_connector_inflight_cap` | none | |
| trigger fn `_jobs_release_inflight` (DEFINER) | none | Decrements `connector_inflight` on every `running → *` transition and on DELETE of a running job. This makes "decrement on every exit path" structural. |

## Requests for the integrator (outside my paths)

1. **mypy path for test helpers (root `pyproject.toml`)**
   - `make typecheck` reports `import-not-found` for `a_support` because test dirs are not on `mypy_path`. Module D's `d1_fakes` has the same problem.
   - Proposed fix: append `:packages/core/tests/module_a` to `[tool.mypy] mypy_path`, and do the same for the other module dirs that have helpers.
   - With that path set, `MYPYPATH=…:packages/core/tests/module_a uv run mypy packages/core/tests/module_a` is clean.
2. **Settings (`config.py`)**
   - Add `worker_concurrency: int = 4` (ge=1). `apps/worker/main.py` currently derives `max(1, db_pool_size - 2)`, which leaves pool headroom for claim and heartbeat. Switch it to the new setting.
   - The pool should stay at least `worker_concurrency + 2`.
3. **Shutdown grace (`compose.yaml` / k8s)**
   - The worker drains for 25 s, then releases its leases (up to 5 s more).
   - Set `stop_grace_period: 35s` (k8s `terminationGracePeriodSeconds: 35`). Otherwise SIGKILL lands before the release, and jobs wait for lease expiry (60 s) plus a reclaim tick.
4. **Composition root (`bootstrap.py`)**
   - Add `jobs: PgJobQueue` (typed as `JobQueue`) to both containers. Use `PgJobQueue(engine)`: the API only needs `enqueue`, and the worker needs the full set.
   - Then make `apps/worker/main.py` take it from the container instead of constructing it.
5. **Handler wiring (Phase 2)**
   - Fill in `apps/worker/main.register_handlers(handlers, container)`.
   - With **zero** handlers registered the worker deliberately does not claim. It still runs reclaim, so it never burns attempts on jobs it cannot run.
   - A job whose kind has no handler on a worker is retried with backoff, not buried, so a rolling deploy is safe.
   - The scheduler (`list_due_connectors` → `enqueue sync_connector`) is not wired yet. I suggest a `schedule_connectors` maintenance job, or a worker-side periodic task owned by C.
6. **Port extension (proposal, not applied)**
   - The frozen `JobQueue.claim(worker_id, n)` has no starvation-guard flag. I added it as a keyword-only `prefer_low=False` through `services.jobs.FairJobQueue`, which is still a valid `JobQueue`.
   - Proposal: add `*, prefer_low: bool = False` to the port in Phase 2, so other `JobQueue` fakes keep the guard visible.
7. **ADR-0013 amendment**
   - `move_finished_jobs` uses dynamic SQL for partition DDL (`CREATE TABLE … PARTITION OF` / `DROP TABLE`). Identifiers and literals are computed only from timestamps via `format(%I, %L)`, never from caller text.
   - The function list gains these items:
     - the internal helpers above;
     - the definer trigger function;
     - `enqueue_job`'s `connector_id` argument;
     - `claim_jobs`'s `prefer_low` argument;
     - `report_job_progress`.
8. **Schema contract**
   - `move_finished_jobs` inserts `SELECT *` from `rf.jobs` into `rf.jobs_history`. Any column a later migration adds to `rf.jobs` must also be added to `rf.jobs_history`, in the same order.
9. **Dev DB**
   - Slot 0002 changed, and the dev DB has it applied as a no-op. Run `make nuke && make up`, which I did not run (shared stack).

## Notes for other modules

- **All handlers (B/C/D):**
  - Do slow work with no transaction open.
  - Then run one tenant tx: `assert_lease` → side effects → `complete`.
  - Raise `PermanentJobError` (from `ragforge_core.services.jobs`) for definitive failures, and `UpstreamRateLimited(retry_after_s)` for 429/Retry-After.
  - Expect `CancelledError` when the lease is lost or at shutdown.
  - If a handler returns without completing, the runtime completes the job in its own fenced tx.
  - `last_error` stores only `Type: first line` (at most 500 chars). Never put tokens or page content in exception messages.
- **C:**
  - Set `JobSpec.connector_id` **only** on jobs that should count against the per-connector in-flight cap of 2 (for example `sync_page`).
  - `list_due_connectors` does not advance `next_poll_at`; C advances it. Queued-job dedupe absorbs repeats meanwhile.
- **E:**
  - `NOTIFY rf_job_progress` carries the job id only. Re-read `rf.jobs` (or `rf.jobs_history` after the move, which happens after 5 min or more) under RLS.
  - `rf_api` can execute `enqueue_job` only.
- **Not implemented:** [Advanced] per-tenant fairness (`tenant_queue_state`). The counter-row pattern from `_claim_one` extends to it directly.

## Verification

- `RAGFORGE_TEST_TARGET_REVISION=0002 uv run pytest packages/core/tests/module_a -q`: 58 passed, three repeated runs.
- `uv run pytest packages/core/tests/test_catalog_invariants.py packages/core/tests/test_db_foundation.py -q`: 24 passed.
- `uv run pytest packages -q`: 443 passed. That full-suite run included other modules' work in progress.
- ruff check, ruff format --check, mypy (strict) on the module A sources, and `lint-imports` are all clean.
