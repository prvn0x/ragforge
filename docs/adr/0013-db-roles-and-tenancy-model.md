# ADR-0013: Tenancy model, database roles, and named cross-tenant paths

**Status:** Accepted · 2026-10-16 · Revised in rounds 3 and 4: the owner-based bypass replaces FORCE RLS + BYPASSRLS · Amended in Phase 2a (see "Phase 2a amendments")

## Context
- The Notion API has no page-permission endpoint.
- Several paths are inherently cross-tenant: key lookup, webhook routing, the scheduler, the job lifecycle, maintenance, metrics and backups.
- v0.4 used FORCE RLS plus a BYPASSRLS function owner. Creating BYPASSRLS roles needs superuser (or an existing BYPASSRLS role on PG16+), may be restricted on managed Postgres, and still needed the migrator to be a member of the owner.
- In a SECURITY DEFINER function, `search_path` without an explicit `pg_temp` searches temp schemas **first**, and PUBLIC has TEMP by default.
- pgvector's type and operators live in the extension's schema.

## Decision: tenancy
- Personal tenant by default.
- Team tenant **[A]** with recorded consent.
- `document_acl` **[S]**.
- Every key containing an external ID is also scoped by `tenant_id` (dedupe keys, document uniqueness, exclusions).

## Decision: roles (no BYPASSRLS anywhere)
| Role | Login | Notes |
|---|---|---|
| `rf_system` | no | **Owns all tables, views and definer functions.** RLS is `ENABLE`d, **not FORCE**d, so the owner bypasses RLS, and that bypass is reachable only through the definer functions |
| `rf_migrator` | yes | Member of `rf_system`. Privileged; credentials exist only in the migrate job |
| `rf_api`, `rf_worker` | yes | Never members of `rf_system`, so RLS always applies; EXECUTE on their function sets |
| `rf_metrics` | yes | SELECT `queue_stats`, `connector_freshness` (ADR-0018, migration 0013); EXECUTE `dead_jobs_last_day()` (migration 0014) |
| `rf_backup` | yes | **Not** a member of `rf_system`. `pg_read_all_data` + `FOR SELECT TO rf_backup USING (true)` policies on tenant tables; `pg_dump --enable-row-security`. Read-only, no DDL |

**Bootstrap** (`db/init/00-bootstrap.sql`, run once as superuser or the provider's admin):
- create the roles;
- `CREATE SCHEMA rf AUTHORIZATION rf_system`; `CREATE SCHEMA ext`; `CREATE EXTENSION vector SCHEMA ext`;
- `REVOKE TEMP ON DATABASE ragforge FROM PUBLIC`;
- default privileges.

Alembic then runs as `rf_migrator` with `SET ROLE rf_system`.

## Decision: RLS policy
`USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)`. Unset or empty → NULL → no rows (fail closed). The tenant is set with `SELECT set_config('app.tenant_id', $1, true)`, which is transaction-local and takes a bind parameter (`SET LOCAL` can't).

## Decision: named cross-tenant paths (exhaustive)
| Function / view | Caller |
|---|---|
| `authenticate_api_key(prefix, sha256)` | api |
| `provision_tenant(name, kind, owner_display_name, consent)` → (tenant_id, user_id); admin scope only | api, `ragforge-admin` CLI |
| `route_notion_webhook(integration_id, workspace_id)` | api |
| `list_due_connectors()` (0008: a claim, kept for pre-0008 images; current code uses `claim_due_connectors`) | worker |
| `take_tokens(integration_key, n)` | worker |
| `chat_active()` | worker |
| internal (not granted): `requeue`, `connector_inflight` / `tenant_queue_state` counter updates | — |
| `assert_run_lease`, `heartbeat_run` | api (FencedPostgresSaver, node wrapper) |
| `enqueue_job(...)`, tenant taken from the setting | api, worker |
| `claim_jobs`, `assert_lease`, `heartbeat`, `complete_job`, `fail_job`, `release_job`, `reclaim_expired_leases` | worker |
| `move_finished_jobs`, `gc_embeddings`, `purge_tombstones` | worker |
| `search_vectors`, `search_vectors_exact`, `search_keywords`, tenant taken from the setting | api, worker |
| `queue_stats` (view, aggregates only) | metrics |
| `connector_freshness_rows()` (behind the security_invoker view `rf.connector_freshness`) | definer | metrics | ADR-0018 (migration 0013): aggregates only (per connector kind: count, oldest last success, its age), no tenant ids; EXECUTE and SELECT to rf_metrics only |
| `dead_jobs_last_day()` → bigint | definer | metrics | ADR-0018 final review (migration 0014): ONE number, the jobs that went dead in the last 24 h across `rf.jobs` AND `rf.jobs_history` (the upkeep archives a dead job after 5 min, so `queue_stats` alone lost it); no parameters, no tenant or job ids; EXECUTE to rf_metrics only, which still cannot read either table |
| `provision_eval_tenant(owner)` → (tenant_id, user_id) | definer | api | ADR-0017 (migration 0012): source = `rf.current_tenant()`, which must be personal or team; a second pair hits the unique pairing constraint |
| `paired_eval_tenant()` | definer | api, worker | ADR-0017: the session tenant's eval pair, or NULL |
| `begin_eval_snapshot(src, target, id, job)` | definer | api | ADR-0017 r4 guard: source = session tenant (never a parameter), target `kind = 'eval' AND eval_of_tenant_id = source`; needs a queued `eval_snapshot_copy` job of the caller's own tenant; creates the empty snapshot collection, one synthetic `uploads` row (no blob) and the pending row |
| `copy_collection_to_eval(src, target, id)` | definer | worker | ADR-0017 r4 guard again; copies documents (page keys preserved, upload ownership), chunks, embedding rows and links; exclusions honoured; idempotent on `id` |
| `finish_eval_reembed(collection, space)` | definer | worker | ADR-0017: refuses a partial flip, flips the active space, drops unreferenced vectors, re-records the digests |
| `_purge_eval_snapshot_pages()` | definer (trigger) | none | ADR-0017 r4 P2: AFTER INSERT ON `rf.exclusions`, deletes the matching snapshot pages (privacy) |

**Function rules:**
- `SET search_path = pg_catalog, rf, ext, pg_temp` (**pg_temp last**);
- pgvector operators schema-qualified (`OPERATOR(ext.<=>)`, `ext.vector`);
- `REVOKE ALL … FROM PUBLIC`, explicit `GRANT EXECUTE`;
- dynamic SQL only for the integer space ID and dimension from `embedding_spaces`;
- a cross-tenant test per function;
- additions require amending this ADR.

LangGraph tables are reached only via `conversations` (under RLS). Store namespaces start with `(tenant_id, user_id)`. **Isolation for LangGraph tables is application-level** (they're outside RLS), and that's tested explicitly.

**Views:** any view owned by `rf_system` would bypass RLS. Every view is therefore created `WITH (security_invoker = true)`, except `queue_stats` (aggregates only). A migration lint enforces this.

**System jobs** use a reserved `SYSTEM` tenant UUID, so `tenant_id` is never NULL and dedupe always applies.

**Restore** runs as `rf_migrator` (owner member) via `pg_restore`.

## Phase 2a amendments (from the Phase 1 handoffs)
Function table additions and signature changes:

| Function | Kind | Grant | Source |
|---|---|---|---|
| `enqueue_job(kind, priority, payload, dedupe_key, run_after, connector_id)` | definer | api, worker | A: gains `connector_id` |
| `claim_jobs(worker_id, n, prefer_low)` | definer | worker | A: gains `prefer_low` (starvation guard) |
| `report_job_progress(id, version, jsonb)` | definer | worker | A: progress ≤ 16 KB, `NOTIFY rf_job_progress` |
| `penalize_tokens(integration_key, seconds)` | definer | worker | C1: Retry-After clamp |
| `search_vectors(..., p_exact_threshold bigint DEFAULT 20000)` | definer | api, worker | B: 5th arg, clamped to [0, 100000] server-side |
| `heartbeat_run(conversation, run, ttl_seconds)` | definer | api | D2: ttl bounded 5–600 |
| `authenticate_api_key(prefix, sha256)` → `(api_key_id, tenant_id, user_id, scopes, client_kind)` | definer | api | E2: no rows for unknown, wrong, revoked or SYSTEM-tenant keys |
| `provision_tenant(...)` → `(tenant_id, user_id)` | definer | api | E2: writes the `tenant.provisioned` audit row itself |
| `vector_candidates_hnsw` | definer, **never granted** | — | B: called only from the vector definers |
| `_jobs_release_inflight` (trigger) | definer, no EXECUTE | — | A: decrements `connector_inflight` on every `running → *` and on DELETE |
| `_claim_one`, `_oldest_low_class`, `_bury_job`, `_ensure_jobs_history_partition`, `_job_backoff_s`, `_job_lease`, `_connector_inflight_cap` | invoker, not granted | — | A |
| `vector_space_operator`, `vector_sql`, `collections_on_space`, `chunks_tsv_refresh` (trigger) | invoker, not granted | — | B |
| `drop_conversation_checkpoints()`, `drop_user_store_items()` (triggers) | invoker, no EXECUTE | — | D2 |

Rule clarifications:
- **Partition DDL is the second allowed form of dynamic SQL.** `move_finished_jobs` runs `CREATE TABLE … PARTITION OF` / `DROP TABLE` with identifiers and literals computed only from timestamps via `format(%I, %L)`, never from caller text (A).
- **Lease functions are fail closed.** With `app.tenant_id` unset, `assert_run_lease` raises `RFL01` and `heartbeat_run` returns false, like every RLS policy (D2). `RFL01` means "lease lost" for both job and run leases.
- **`rf_worker` has no privilege on any LangGraph table.** A future `summarize_conversation` reads through a tenant-scoped saver plus the conversation lookup under RLS. Any checkpoint-table grant goes in a new slot, and this ADR is amended in the same change (D2).
- **Schema contract:** `move_finished_jobs` copies `SELECT *` from `rf.jobs`, so any column added to `rf.jobs` must be added to `rf.jobs_history` in the same order (A).
- **Slot 0007 (integration):** `pending_approvals.executed_at` (the exactly-once execution claim, updatable by rf_api under RLS) with `CHECK (executed_at IS NULL OR status = 'approved')`; `embedding_spaces` gains `CHECK ((status = 'retired') = (retired_at IS NOT NULL))`. No new functions or grants.

## Threat model
RLS defends against application bugs. It does **not** defend against a compromised `rf_api`/`rf_worker` credential, which can set `app.tenant_id` arbitrarily. That's covered by network policy, secret handling, short-lived credentials in cloud, and audit.

## Consequences
- Works on managed Postgres without BYPASSRLS (verify CREATE ROLE rights on the chosen provider during M9).
- The bypass surface is enumerable and tested.
- Only the migrator is privileged (an owner member), and its credential exists only in the short-lived migrate job. The backup role is read-only.
- Every key containing an external ID is tenant-scoped; exclusions are keyed by `(tenant_id, source_kind, external_id)`, so they survive a connector being re-added.

### Slot 0008 (connector scheduling, Phase 2a review fixes)
- `rf.claim_due_connectors(limit, lease_s)` and `rf.claim_reconcile_connectors(limit, interval_s, lease_s)`: SECURITY DEFINER, pinned search_path, `REVOKE ALL FROM PUBLIC`, EXECUTE to **rf_worker only**. Cross-tenant by nature (the scheduler serves every tenant); they return ids only and take a poll/reconcile lease (`next_poll_at` / `next_reconcile_at` moved forward under `FOR UPDATE SKIP LOCKED`), skipping connectors whose `sync_connector:<id>` / `reconcile_deletes:<id>` job is queued or running. The poll lease is short and fixed (the poll interval, clamped 30 s–1 day): it only guards against a second scheduled listing, so a claim whose enqueue is lost costs one lease. The reconcile claim also takes only a short lease and records the configured interval in `connectors.reconcile_interval_s`; the next run is scheduled by `rf._connector_reconcile_outcome()` when the reconcile job ends, so a lost enqueue is retried one lease later. The worker passes leases that outlive its own scheduling step (polls: max(poll interval, 120 s); reconciles: 240 s). They never enqueue: the worker enqueues in the connector's own tenant transaction, so the tenant always comes from the claimed row. Inputs are clamped (limit 1–1000, lease ≥ 30 s, interval ≥ 60 s).
- `rf._connector_poll_outcome()`: definer trigger function (no EXECUTE grants) on the terminal transition of a `sync_connector` job: `dead` → `connectors.poll_failures + 1` (capped), `next_poll_at` pushed to at least now() + 300 s · 2^poll_failures (capped at one day, ±10% jitter so an upstream-wide outage does not align every connector's retries), and `connectors.last_error` set to "listing failed (N in a row): <job error>" so the backoff is visible; `done` → 0. This is the failure backoff, kept apart from the claim lease. It fires on every path to `dead` (`fail_job` permanent or attempts exhausted, `reclaim_expired_leases`). The payload's `connector_id` is regex-validated before the cast, and the update matches `tenant_id`, so a malformed or foreign payload cannot touch another row or abort `fail_job`. Cross-tenant tests (`tests/test_scheduling.py`): a tenant-A job naming tenant B's connector leaves B untouched; malformed payloads still bury the job; two concurrent claimers get disjoint connectors; the API role cannot execute either claim.
- `rf._connector_reconcile_outcome()`: definer trigger function (no EXECUTE grants) on the terminal transition (`done`/`dead`) of a `reconcile_deletes` job: `next_reconcile_at` = now() + `reconcile_interval_s` ±5% (symmetric, so the nightly time does not drift). Same payload validation and `tenant_id` match as the poll trigger.
- `rf.reclaim_expired_leases(limit)` is replaced (same signature, attributes and grant: rf_worker only). It differs from 0002's in two ways. It bumps `lease_version` before requeueing (or superseding into a queued duplicate) or burying an expired job, so the zombie holder's next heartbeat returns false. And a requeue is backed off (`run_after = now() + rf._job_backoff_s(attempts, random())`) instead of runnable at once, so a job that kills its worker cannot kill the next replica within seconds (round 6). Tests: reclaim superseding and reclaim burying both make the old holder's heartbeat false (each fails with the bump removed); a reclaimed job is queued 8–12 s out at attempts=1.
- `rf.heartbeat(job, version)` is replaced (same signature, attributes and grant: rf_worker only). It never blocks on the job row: a row locked by another session is the holder's own fenced result transaction (alive), and it returns false only when another holder took the job (row gone, or `lease_version` moved on). Any state at the same version (`done`, `dead`, `failed`, `superseded`, or `queued` after the holder's own retryable fail or release) was reached by this holder and is not a loss; that rule is only sound because the replaced reclaim moves the version. It is still fail-safe: every result transaction is fenced by `assert_lease`, independent of the heartbeat. Tests: a heartbeat during the holder's open fenced transaction returns at once, and a runtime handler with a long result transaction plus post-commit work logs no `job_lease_lost`.
- `rf.defer_job(job, version, delay_s)`: SECURITY DEFINER, pinned search_path, EXECUTE to rf_worker only. Like `release_job` (lease-checked, no attempt charged, merges into a queued duplicate), but not runnable before `delay_s` (clamped 1 s–1 h). The runtime uses it for job kinds it has no handler for during a rolling deploy. Bounded: a job enqueued more than 24 h ago is buried instead (`dead`, `job.dead` audit row with `permanent: true`, error 'no handler for job kind <kind>'), so a removed or typo'd kind cannot ping-pong forever unseen. Tested on the real `claim_jobs` path (the claim charges an attempt, the defer refunds it).
- New columns `connectors.next_reconcile_at` (default: a random point in the next day, so existing connectors don't reconcile together), `connectors.poll_failures`, `connectors.reconcile_interval_s` (60 s–7 days); new indexes `jobs_active_dedupe` (queued + running), `connectors_due`, `connectors_reconcile_due`. `rf.list_due_connectors` (0002) keeps its signature and grant, but its body is now `claim_due_connectors(p_limit, 300)` (VOLATILE). The current code no longer calls it, but a pre-0008 image does: during a rolling deploy, or after a rollback that `ragforge-migrate` allows while `code_version >= min_code_version`. Round 4 made it return no rows, and a rollback then silently stopped all polling while readiness stayed green (blind ops review round 6, P1). As a claim, old schedulers keep polling with the twin guard and without backoff resets. Test: `test_the_pre_0008_scheduler_function_is_a_safe_claim`.
- `rf.assert_lease(job, version)` is replaced (same signature and grant). It applies the same fence, and once the fence passes it extends `lease_expires_at` to `clock_timestamp() + rf._job_lease()`. The heartbeat skips a row locked by the holder's own fenced transaction, so a long result transaction that commits keeps its lease. A rollback reverts the extension; the next keepalive beat then extends normally. The function contains no `%`: `run_sql` goes through psycopg's paramstyle, so the message is built with `MESSAGE =` concatenation.

### Slot 0009 (Phase 2a review round 4): why a document is tombstoned
- `rf.documents.deleted_reason` holds `'excluded'` (a user or approved-agent exclusion) or `'source_deleted'` (sync/reconcile confirmed the item gone). It is NULL exactly when `deleted_at` is NULL, enforced by a CHECK.
- `rf._documents_deleted_reason()` is an INVOKER BEFORE trigger with a pinned `search_path` and no EXECUTE grants:
  - clearing `deleted_at` clears the reason;
  - a tombstone written without a reason gets `'source_deleted'`, the sticky default, so writers that don't opt in keep the old behaviour.
- **Revival rules:**
  - Undo (forgetting an exclusion) clears `'excluded'` tombstones at once.
  - A sync clears an `'excluded'` tombstone whatever the fetch time, because sync only reaches the upsert when the item is no longer excluded.
  - The sticky fetched_at rule (handoff B) applies to `'source_deleted'` only.
- **Grants:** none new. `rf_api` already has UPDATE on `rf.documents`, and `rf_worker` writes it through `PgDocumentIndex`.
