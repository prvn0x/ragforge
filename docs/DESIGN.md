# RAGForge — Design Doc

> Ask your Notion (and other docs) anything and get answers with **verified** citations that deep-link to the exact paragraph. Use it from a web UI, a REST API, or inside Claude Code / Cursor through MCP.
> `git clone && make up` sets up everything: database, migrations, models and services. The demo works before you connect anything.

Status: **v0.8. Revised after seven review rounds, five of them independent and blind** ([REVIEW.md](REVIEW.md), [adr/](adr/)) · Last updated: 2026-10-16

---

## 0. Scope tiers and honest effort

One engineer at ~15 h/week. Each tier ships something demo-able on its own.

| Tier | Contents | Effort |
|---|---|---|
| **MVP (M0–M4)** | workspace, compose, roles + RLS + definer functions, job queue (class order + starvation guard, leases, fencing), file + Notion ingestion, hybrid retrieval + rerank, LangGraph agent with verification + streaming sanitizer, short-term memory, ~100-Q public eval gate + verifier eval set, OTel | ~300 h (≈ 20 weeks) |
| **Core (M5–M6)** | long-term memory, conversation summaries, HITL approvals, typed SDK, MCP server, Streamlit UI | +100–150 h (≈ 7–10 weeks) |
| **Advanced (M7–M10)** **[A]** | team tenants, per-tenant fairness, 300-Q set, blue/green space automation, dashboards and alerts as code, load test, restore drill, Helm/kind, NetworkPolicies, KEDA, multi-arch, SBOM | +150–200 h |
| **Stretch** **[S]** | outbox + Redpanda with a real consumer, Google Drive (ACL-aware), LLM gateway | optional |

**Risk-first rule:** the riskiest mechanisms (roles/RLS bootstrap, `claim_jobs` fencing, `search_vectors`, the stream sanitizer) are built **first in M0–M1 as SQL/unit tests**, before any UI. They are where this design is most likely to be wrong.

---

## 1. Users, tenancy, design load

| Persona | Priority | Daily use | Surface |
|---|---|---|---|
| **You** (Notion power user, engineer) | Primary | "What did I decide about X?", "Steps to set up Y from my notes" | UI, MCP in Claude Code |
| **AI coding agent** acting for you | Primary | Pulls your notes and conventions into context while coding | MCP |
| **A small team** **[A]** | Secondary | Runbooks, RCAs, onboarding | UI, API |
| **Operator** (you) | Secondary | Sync health, dead jobs, eval trend | Grafana, admin API |

**Tenancy** ([ADR-0013](adr/0013-db-roles-and-tenancy-model.md))
- **Tenant = a person** by default.
- **Team tenants [A]** require recorded consent, because the Notion API exposes no page permissions.
- Notion page IDs are global, so two tenants can index the same page. **Every key that includes an external ID is also scoped by tenant.**

**Design load:** 1–5 users, ≤ 1 concurrent turn per conversation, ≤ 0.5 chat QPS total, ≤ 10k Notion pages per tenant, a backfill possibly running during chat.

**Non-goals (v1):** billing, SSO, polished frontend, multi-region, write-back to Notion, event broker ([ADR-0003](adr/0003-no-outbox-or-broker-in-v1.md)).

---

## 2. SLOs and budgets

Local-model latency targets are **provisional until measured in M4** on two reference machines. Hosted targets are firm.

| SLO (at design load) | Target |
|---|---|
| `/query` p95 | < 400 ms |
| `/chat` TTFT p95 | hosted: first turn < 1.5 s, follow-up < 2.0 s · local 8B (≥ 24 GB RAM): < 5 s · local 3B: < 4 s |
| `/chat` verified answer p95 | hosted < 5 s · local 8B < 15 s · local 3B < 12 s |
| LLM calls on the critical path | first turn 2 (generate, ground check); follow-up 3 (+contextualize); summarization is an off-path job (Core) |
| Verifier quality (labeled verifier set, ~300 items) | **Wilson 95% lower bound** of recall on unsupported answers ≥ 0.75 and of precision ≥ 0.80 per shipped model; the reported headline metric is the **false-"verified" rate** at the production base rate. Below that, the badge says "unverified" for that model |
| Notion freshness p95 | < 10 min polling · < 1 min webhooks **[A]** |
| Oldest queued interactive/webhook job | < 60 s |
| Retrieval quality (public corpus) | recall@5 ≥ 0.85, MRR ≥ 0.70, never below the frozen baseline − 3 points |
| Freshness during backfill | incremental and webhook jobs are never queued behind backfill; their wait is bounded by (in-flight jobs ÷ workers) × job duration, reported as a metric (strict class order + starvation guard) |

**Hosted critical path (follow-up turn):**
1. auth (5 ms)
2. load_memory (80 ms)
3. contextualize (400 ms)
4. embed (50 ms)
5. vector ∥ keyword search (60 ms)
6. RRF + collapse (2 ms)
7. rerank top-20 at 512 tokens (≈ 250 ms, measured in M3)
8. pack (5 ms)
9. TTFT (600 ms)

That sums to ≈ 1.45 s against a 2.0 s target, leaving ~0.5 s of headroom. The citation check (2 ms) and ground check (≈ 800 ms) come after TTFT.

**Local models:**
- Packed context is 2k tokens (top-5).
- **Every call to a given model uses the same `num_ctx`** (chat 8192, embed 2048), because changing it makes Ollama reload the model.
- Each request sends `keep_alive: 30m` to avoid cold reloads.
- Worker embedding concurrency is 2, and drops to 1 while a chat run lease is active.
- The host's `OLLAMA_MAX_LOADED_MODELS` isn't controlled by us (the default allows ≥ 2). `make doctor` checks it.
- The verifier is evaluated separately (section 12). A model that fails the verifier SLO ships with verification shown as **"unverified"**, never as a false "verified".

**Capacity:** 10k pages × ~20 chunks = 200k vectors × 3 KB ≈ 600 MB per tenant. The revisit trigger is **~5M vectors per space** (indexes are shared across tenants). Initial sync ≈ 30 min per 1k pages (Notion ~3 req/s); it's resumable and goes most-recently-edited first.

---

## 3. High-level design

```
                          Clients
   ┌────────────┐   ┌──────────────┐   ┌─────────────────────────┐
   │ Streamlit  │   │ curl / SDK   │   │ Claude Code / Cursor     │
   └─────┬──────┘   └──────┬───────┘   └────────────┬────────────┘
         │  ragforge-client                         │
         │                 │                 ┌──────▼──────┐
         ▼                 ▼                 │ MCP server  │ thin API client (ADR-0005)
   ┌──────────────────────────────────────┐  └──────┬──────┘
   │ API pods (FastAPI /v1), the only     │◀────────┘
   │ entry point                           │
   │ authenticate_api_key() →              │
   │ set_config('app.tenant_id', …, true)  │
   │ chat SSE (incremental sanitizer) ·    │
   │ query · connectors · docs · jobs ·    │
   │ approvals · audit                     │
   │ LISTEN rf_job_progress → job SSE      │
   └──────┬──────────────────────┬────────┘
          │                      │ app state + job + audit rows commit together
   ┌──────▼─────────────┐ ┌──────▼─────────────────────────────────────┐
   │ LangGraph agent    │ │ Worker pods                                 │
   │ run lease per conv │ │ claim_jobs (class order + guard) · leases · │
   │ verify · sanitize  │ │ lease check INSIDE every side-effect txn    │
   └──────┬─────────────┘ │ scheduler · reconcile · retention · GC      │
          │               │ parser sandbox                              │
          │               └──────┬──────────────────────┬──────────────┘
          │                      ▼                      ▼
          │               Connectors (Notion, Files)    Upload volume / S3
          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ PostgreSQL 17 + pgvector ≥ 0.8 (extension in schema `ext`)      │
   │ tables owned by rf_system (NOLOGIN); RLS on for app roles       │
   │ named SECURITY DEFINER functions = the only cross-tenant paths  │
   │ LangGraph checkpoints/store: separate autocommit pool           │
   └─────────────────────────────────────────────────────────────────┘
   Models: host Ollama (Metal) → container Ollama → OpenAI / Bedrock (ADR-0004)
   Observability: OTel → grafana/otel-lgtm (profile obs); content capture off by default
```
Images: **`server`** (commands `api`, `worker`, `migrate`), **`mcp`**, **`ui`**. The FlashRank and tokenizer models are **baked into `server`** at build time, so containers can run with read-only filesystems.

### Flow A: Notion sync
1. **Connect:** validate with `GET /users/me` plus a test search.
   - The UI asks the user to grant **Read content only** (guidance; capabilities can't be introspected).
   - `Notion-Version` is pinned (2025-09-03, data sources).
   - The token is envelope-encrypted.
2. **Schedule:** `list_due_connectors()` every minute → `enqueue_job(sync_connector)` at priority `incremental`.
3. **Diff:**
   - The cursor is inclusive with a 5-min overlap.
   - Skip pages that are in `exclusions`, and pages that are **fresh**. A page is fresh iff `last_edited_time < date_trunc('minute', documents.fetched_at)`.
     - `fetched_at` is the **`Date` header of the first Notion response of that page's fetch, minus 60 s**. That means it uses Notion's clock (no skew with ours) and it's taken at fetch *start*, so an edit made during a long recursive fetch is never marked fresh.
     - Pages that embed synced blocks record `synced_source_ids`. Reconcile checks those originals' `last_edited_time` and re-enqueues the dependent pages, because editing an original doesn't touch the embedding pages' timestamps.
     - `last_edited_time` is minute-granular, so an edit made in the same minute as the fetch still counts as stale and gets refetched.
     - On refetch with an unchanged `page_sha256`, the document's `fetched_at` is still bumped (guarded), and chunk replacement and embedding are skipped.
   - Enqueue `sync_page` for the rest.
   - The cursor advances when listing completes, independent of page-job outcomes. Lost or dead page jobs are repaired by reconcile (step 9).
4. **Fetch:** recursive and paginated, depth ≤ 10.
   - **Rate limiting is global per connector, not per process:** every API call first runs `take_tokens(integration_key, 1)`, a definer function that atomically refills (using `clock_timestamp()`) and decrements a `notion_rate(integration_key = sha256(token), tokens, refilled_at)` row, and returns the wait in ms. It's capped at 3 req/s across all workers, and keyed by the integration token, which is what Notion actually limits.
   - In-flight `sync_page` jobs are capped at 2 per connector by a **`connector_inflight(connector_id, running)` row**. It's locked `FOR UPDATE` in claim, complete, fail, release and reclaim. `claim_jobs` excludes jobs whose connector is at its cap (`NOT EXISTS … running >= cap`), so capped connectors cause no head-of-line blocking. `Retry-After` drains the bucket.
5. **Normalize, chunk, contextual header:** section 7.
6. **Embed:** compute `embed_sha` for each chunk. For every space that's active or building for the collection:
   1. Upsert `embedding_touches` for **every hash this sync will reference**. This protects them from GC without rewriting vector rows, which would cause HNSW churn.
   2. Embed only the hashes missing from `embeddings`.
   3. `INSERT … ON CONFLICT DO NOTHING`.
7. **Side-effect transaction** (fenced, [ADR-0010](adr/0010-job-scheduling-leases-priorities-fairness.md)):
   1. `assert_lease(job_id, lease_version)` locks the job row `FOR UPDATE`. A mismatch raises, the transaction rolls back, and the zombie's result is discarded.
   2. Upsert the document **only if** `excluded.fetched_at > documents.fetched_at`. A later fetch always observed content at least as new. This makes two concurrent `sync_page` runs for the same page safe in either commit order. **If the guarded upsert affects 0 rows, the remaining steps are skipped**, and the job completes as a no-op.
   - The worker sets the tenant context first: `set_config('app.tenant_id', job.tenant_id, true)` at the start of every job transaction.
   3. Replace its chunks.
   4. Insert the audit row.
   5. `complete_job` in the same transaction.
   6. `NOTIFY rf_job_progress`.
8. **Cursor writes** are monotonic: `cursor = GREATEST(cursor, new)`.
9. **Reconcile (nightly):**
   - Full listing → re-enqueue every page that isn't fresh (the same `fetched_at` rule). This repairs dead jobs and missed polls.
   - Pages missing from the listing are confirmed with `GET /pages/{id}`. 404, `in_trash`/archived or "not shared" → tombstone; any other error → retry next night.
   - Hard delete after 7 days.
10. **Webhooks [A]:** verify → `route_notion_webhook()` → `sync_page` at priority `webhook`.

### Flow B: Ask
`POST /v1/chat` (SSE):
1. **Acquire the conversation run lease.** `UPDATE conversations SET active_run_id=$r, run_expires_at=now()+60s WHERE id=$c AND (active_run_id IS NULL OR run_expires_at < now())`. Zero rows → `409 turn_in_progress`. It's heartbeated every 15 s, and released at turn end or at an interrupt.
2. Run the graph, streaming through the sanitizer.
3. Citation check → ground check → `verification` → **append the final user and assistant messages to state** → `done`. Drafts never enter `messages`.
4. **(Core)** If the unsummarized history exceeds 3k tokens, enqueue `summarize_conversation` (priority `maintenance`, deduped per conversation). It **doesn't take the run lease**, so it never causes a 409.
   - It writes `conversation_summaries` with an optimistic guard: only if its `upto_message_id` is greater than the stored one.
   - The worker defers it while a chat run lease is active, so it doesn't compete with the user for Ollama.

---

## 4. Technology choices

| Concern | **Chosen** | Pros | Cons → mitigation | Rejected |
|---|---|---|---|---|
| Language | **Python 3.12** | LangChain/LangGraph ecosystem | GIL → threads/subprocesses | Go/Java |
| Packaging | **uv workspace, 5 packages** ([0001](adr/0001-uv-workspace-monorepo.md)) | DB boundary enforced; 3 images | import-linter inside core | 7 packages; single package |
| API | **FastAPI + Pydantic v2 + SSE** | Async, typed | — | Flask, Django |
| Datastore | **Postgres 17 + pgvector ≥ 0.8** ([0002](adr/0002-postgres-as-single-datastore.md)) | One transaction; RLS; FTS | ~5M vectors per space → revisit | Qdrant, Elasticsearch |
| Search access | **SECURITY DEFINER `search_vectors` / `search_keywords`** ([0009](adr/0009-embedding-spaces-and-content-addressed-embeddings.md)) | Custom plans hit the partial index; GIN usable; exact path for small sets | Explicit-predicate isolation → per-function cross-tenant tests | Raw queries under RLS |
| Roles | **Tables owned by NOLOGIN `rf_system` (RLS enabled, not forced); app roles under RLS; no BYPASSRLS anywhere; every view `security_invoker=true` except `queue_stats` (lint-enforced)** ([0013](adr/0013-db-roles-and-tenancy-model.md)) | Works on managed Postgres; definer functions bypass RLS as the owner | Migrator is privileged (short-lived job only) | FORCE RLS + BYPASSRLS role |
| Driver | **psycopg 3, two pools** ([0008](adr/0008-psycopg3-single-driver.md)) | One driver | Two pools sized explicitly | asyncpg |
| Migrations | **Alembic, expand/contract, monotonic `schema_meta`** | Rollback-safe readiness | Hand-written pgvector/RLS DDL | `create_all` |
| Queue | **Postgres jobs: strict class order + starvation guard, leases, fencing inside the side-effect txn, tenant-scoped dedupe** ([0010](adr/0010-job-scheduling-leases-priorities-fairness.md)) | No zombie commits; no lost jobs; no class inversion | Revisit above ~500 jobs/s sustained | Celery, SQS |
| Events | **None in v1** ([0003](adr/0003-no-outbox-or-broker-in-v1.md)) | Nothing to operate | — | Outbox, Redpanda |
| Uploads | **Shared volume (local) / S3 (cloud)** | No extra container | RWX PVC on k8s | MinIO, `bytea` |
| Orchestration | **LangGraph 1.x** | Durable threads, interrupts | Interrupted nodes re-run on resume → idempotent node bodies | LCEL |
| Chat model | **Config; `qwen3:8b` (thinking off) if ≥ 24 GB RAM, else `llama3.2:3b`; `make eval-models`** ([0004](adr/0004-host-first-ollama.md)) | Zero keys | Quality → measured | Hard-coded |
| Embeddings | **`nomic-embed-text` 768d, task prefixes, fixed `num_ctx`** | Local | — | — |
| Reranker | **FlashRank ONNX, `max_length=512`, top-20, model baked into image** ([0006](adr/0006-onnx-reranker-and-score-based-grading.md)) | No torch; read-only FS works | Latency measured in M3 | sentence-transformers |
| Answer safety | **Incremental sanitizer (inert overflow, reference links stripped, exact URL allow-list) + citation check + ground check** ([0011](adr/0011-streaming-verification-and-output-safety.md)) | Safe mid-stream | ≤ 512 chars held | Sanitizing only at the end |
| Notion | **Official API, pinned version, polling primary, exclusions** ([0007](adr/0007-notion-polling-primary-webhooks-optional.md)) | Supported | 3 req/s | Export ZIPs |
| MCP | **Official `mcp` SDK, thin API client** ([0005](adr/0005-mcp-as-thin-api-client.md)) | One authz path | ~1 ms hop | Direct DB |
| UI | **Streamlit** | Fast | Not product-grade | React |
| Observability | **structlog + OTel; `grafana/otel-lgtm`** | One container | Dev-grade | Separate stacks |
| Evals | **Recorded-Notion-JSON corpus; path-filtered significance gate (false-block ≤ 5%) + frozen floor; verifier eval with Wilson bounds; τ recalibrated on private set** ([0012](adr/0012-evaluation-strategy.md)) | False blocks bounded by construction; stated power; the "verified" feature itself is measured | ~50% per-PR power for 5-pt drops at n=300 (the floor catches drift) | Point-estimate gate (~20% false blocks); non-inferiority CI gate (needs n≈1k) |
| Testing | **pytest, testcontainers, respx, hypothesis, import-linter** | Real Postgres | Slower → markers | SQLite fakes |
| Supply chain | **Digest-pinned images, `uv.lock`, Trivy; SBOM + multi-arch [A]** | Reproducible | — | — |
| Kubernetes **[A]** | **Helm + kind; KEDA optional** | Deploy story | — | Bitnami subcharts |

---

## 5. Project structure (uv workspace, 5 packages)
```
ragforge/
├── pyproject.toml · uv.lock · Makefile · compose.yaml · .env.example
├── docker/Dockerfile           # targets: server | mcp | ui
├── scripts/                    # bootstrap.sh · doctor.sh
├── packages/
│   ├── contracts/              # API schemas + SSE events
│   ├── core/                   # library + apps; console scripts ragforge-api / -worker / -migrate
│   ├── client/                 # async typed SDK
│   ├── mcp/                    # FastMCP → client
│   └── ui/                     # Streamlit → client
├── evals/  corpus/ · datasets/ · baselines/frozen.json · runner/
├── tests/e2e/
├── deploy/helm/ · deploy/kind/   [A]
└── docs/
```
**import-linter:** `{mcp, ui, evals}` → `client` → `contracts` (never `core`). Inside `core`: `apps` → `agent` → `services`, `rag` → `domain`; `adapters` → `domain` only.

---

## 6. Data model

### 6.1 Roles and bootstrap ([ADR-0013](adr/0013-db-roles-and-tenancy-model.md))
| Role | Login | Purpose |
|---|---|---|
| `rf_system` | no | **Owns all tables, views and SECURITY DEFINER functions.** RLS is enabled but not forced, so the owner, and therefore the definer functions, see all tenants by design |
| `rf_migrator` | yes | Member of `rf_system` (runs DDL as it). Privileged; used only by the short-lived `migrate` job |
| `rf_api` | yes | DML under RLS; SELECT `jobs` / `jobs_history` under RLS; EXECUTE on the API function set |
| `rf_worker` | yes | DML under RLS; EXECUTE on the worker function set |
| `rf_metrics` | yes | SELECT `queue_stats` |
| `rf_backup` | yes | **Not** an owner. `pg_read_all_data` + `FOR SELECT TO rf_backup USING (true)` policies; `pg_dump --enable-row-security`. Read-only and can't run DDL |

- **Bootstrap:** `ragforge_core/db/bootstrap.sql`, run by the one-shot `db-bootstrap` service (`ragforge-db-bootstrap`), runs once as superuser (idempotent). In cloud the same command runs with the provider's admin credentials. It creates the roles, schemas `rf` and `ext` (`CREATE EXTENSION vector SCHEMA ext`), and runs `REVOKE TEMP ON DATABASE … FROM PUBLIC`. No role needs `BYPASSRLS`.
- **RLS policy:** `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid`. Unset or empty → NULL → no rows (**fail closed**). The tenant is set with `SELECT set_config('app.tenant_id', $1, true)`, which is transaction-local and takes a bind parameter.

**Cross-tenant paths are exactly these functions, owned by `rf_system`:**

| Function | Callers |
|---|---|
| `authenticate_api_key(prefix, sha256)` | api |
| `route_notion_webhook(integration_id, workspace_id)` | api |
| `list_due_connectors()` | worker |
| `take_tokens(integration_key, n)` (global Notion rate limit) | worker |
| `chat_active()` (any live run lease; used to throttle embedding) | worker |
| internal, not granted: `requeue`, `connector_inflight` / `tenant_queue_state` updates (called only from the job functions) | — |
| `assert_run_lease`, `heartbeat_run` (checkpoint fencing; LangGraph pool has no tenant setting) | api |
| `enqueue_job(kind, priority, payload, dedupe_key, run_after)`, tenant taken from the setting | api, worker |
| `claim_jobs(worker_id, n)` | worker |
| `assert_lease`, `heartbeat`, `complete_job`, `fail_job`, `release_job` | worker |
| `reclaim_expired_leases()` | worker |
| `move_finished_jobs`, `gc_embeddings`, `purge_tombstones` | worker |
| `search_vectors`, `search_vectors_exact`, `search_keywords`, tenant taken from the setting | api, worker |
| `queue_stats` (view; aggregates only) | metrics |

**Function rules:**
- `SET search_path = pg_catalog, rf, ext, pg_temp` (pg_temp **last**), with pgvector operators schema-qualified (`OPERATOR(ext.<=>)`);
- `REVOKE ALL FROM PUBLIC`; explicit grants;
- dynamic SQL only for integer space ID and dimension loaded from `embedding_spaces`;
- one cross-tenant test per function;
- adding a function requires an ADR amendment.

### 6.2 Tables
```sql
schema_meta      (schema_version int, min_code_version int)                  -- single row, monotonic
tenants          (id, name, kind personal|team, team_visibility_consent_at)
users            (id, tenant_id, display_name)
api_keys         (id, tenant_id, user_id, prefix unique, key_sha256, scopes, last_used_at, revoked_at)
connectors       (id, tenant_id, kind, target_collection_id, config, secret_ciphertext, wrapped_dek,
                  status active|paused|error|deleting, last_error, next_poll_at)
sync_cursors     (connector_id, scope, cursor, updated_at)                    -- monotonic writes
exclusions       (id, tenant_id, source_kind notion|file, external_id, reason, created_at,
                  unique(tenant_id, source_kind, external_id))                -- authoritative; survives connector re-add
uploads          (id, tenant_id, collection_id, blob_key, filename, sha256, created_at)  -- authoritative
embedding_spaces (id int, provider, model, dim, doc_prefix, query_prefix, distance, status building|active|retired)
collections      (id, tenant_id, name, active_space_id, building_space_id null, fts_config, tau real null,
                  unique(tenant_id, name))
documents        (id, tenant_id, collection_id, connector_id null, upload_id null, external_id, external_url,
                  title, external_updated_at, fetched_at, page_sha256, synced_source_ids text[], metadata, deleted_at,
                  UNIQUE NULLS NOT DISTINCT (tenant_id, collection_id, connector_id, external_id))
document_links   (tenant_id, from_document_id, to_external_id)
chunks           (id, tenant_id, collection_id, document_id, anchor, ordinal, heading_path, display_text,
                  display_sha256, embed_sha256, token_count, metadata, tsv,      -- tsv via trigger
                  unique(document_id, ordinal))
embeddings       (tenant_id, collection_id, space_id, embed_sha256, embedding ext.vector, created_at,
                  pk(tenant_id, collection_id, space_id, embed_sha256))
notion_rate      (integration_key bytea pk = sha256(token), tokens real, refilled_at) -- global Notion rate limit (ADR-0007)
jobs             (id, tenant_id NOT NULL  -- system jobs use the reserved SYSTEM tenant,
                  kind, priority, payload, dedupe_key NOT NULL  -- payload is a pure function of dedupe_key,
                  status queued|running|done|failed|dead|superseded,
                  attempts, max_attempts, run_after, enqueued_at, leased_by, lease_version, lease_expires_at, last_error)
jobs_history     (same; monthly partitions; 30 d)
audit_log        (id, tenant_id, actor, action, target, details, created_at)
conversations    (id = thread_id, tenant_id, user_id, collection_ids, title, active_run_id, run_expires_at,
                  head_checkpoint_id)                                         -- advanced only by the lease holder
conversation_summaries (conversation_id pk, tenant_id, upto_message_id, summary, updated_at)   -- Core
pending_approvals(id, tenant_id, conversation_id, run_id, tool_call_id, tool, args,
                  status pending|approved|rejected|expired, created_at, decided_at,
                  unique(conversation_id, tool_call_id))
usage_events     (...)   eval_runs (...)
-- LangGraph checkpoints + store (library-owned, reached only via conversations)
```

### 6.3 Indexes
- **Vectors:** per space, `CREATE INDEX CONCURRENTLY … USING hnsw ((embedding::ext.vector(768)) ext.vector_cosine_ops) WHERE space_id = 1`.
- **Keyword:** GIN on `chunks.tsv`.
- **Btrees:**
  - `embeddings(tenant_id, collection_id, space_id)`;
  - `chunks(tenant_id, collection_id)` and `chunks(embed_sha256)`;
  - `documents(tenant_id, collection_id)`.
- **Jobs:**
  - `jobs(priority, enqueued_at) WHERE status='queued'`. It's usable because `claim_jobs` queries one class at a time (`priority = $p ORDER BY enqueued_at`);
  - **`UNIQUE (tenant_id, dedupe_key) WHERE status='queued'`**;
  - `jobs(lease_expires_at) WHERE status='running'`.

### 6.4 Data classification ([ADR-0014](adr/0014-data-classification-and-backups.md))
- **Authoritative:** schema_meta, tenants, users, api_keys, connectors, collections, embedding_spaces, exclusions, **uploads**, conversations, conversation_summaries, pending_approvals, LangGraph tables, audit_log, usage_events, eval_runs.
- **Derived:** documents, chunks, embeddings, document_links, sync_cursors, jobs, jobs_history.

---

## 7. Ingestion and retrieval

### 7.1 Notion normalization
| Block | Handling |
|---|---|
| headings 1–3 | section boundary; extend `heading_path` |
| paragraph, quote, callout, lists, to_do | Markdown (to_do keeps `[x]`) |
| toggle / toggle heading | children expanded inline |
| code | fenced with language; never split |
| table | Markdown table; row groups with the header repeated |
| synced_block | resolve to the original content |
| child_page | its own document; parent links to it |
| data-source rows | each row is a document; properties → `metadata` |
| mention, link_to_page | text kept; edge in `document_links` |
| equation | LaTeX inline |
| media | caption + file name |
| unknown | skipped; `rf.notion.unsupported_block{type}` |

### 7.2 Chunking and hashing
- Target 400 tokens, max 512, overlap 15% only inside long sections. Never cross a heading unless the section is under 80 tokens.
- `embedded_text = "{title} › {heading_path}\n\n{text}"`; `embed_sha = sha256(embedded_text)`; `display_sha = sha256(text)`.
- **What the hash buys:** on a re-sync, any chunk whose embedded text didn't change keeps its embedding, even if its ordinal moved. Only new or changed text is embedded.
- Identical text on different pages has different headers, so it gets different embeddings (intended). It's collapsed at retrieval by `display_sha`.

### 7.3 Retrieval
```
query → for each distinct active space among the requested collections: embed with that space's query prefix
  ├─ search_vectors(space, q, collections, k=40)
  │    if count(embeddings for tenant ∧ collections ∧ space) < EXACT_THRESHOLD (default 20k, tuned in M3,
  │       measured including TOAST detoast cost):
  │         search_vectors_exact(): a separate function declared with SET enable_indexscan = off, scoped
  │         to the function so it can't leak; bitmap btree filter + ORDER BY distance LIMIT k
  │    else (or if the space's HNSW index isn't yet valid (pg_index.indisvalid), use the exact path):
  │         HNSW (ef_search=100, iterative_scan=relaxed_order, max_scan_tuples=100k),
  │          filter on tenant_id AND collection_id = ANY(...) (both columns live on embeddings),
  │          over-fetch 2×, re-sort by distance in a MATERIALIZED CTE
  │    join chunks on (tenant, collection, embed_sha), DISTINCT ON chunk
  └─ search_keywords(q, collections, k=40): one tsquery per distinct fts_config among the collections
→ RRF (k=60) across all lists (rank-based, so lists from different spaces fuse safely)
→ collapse by display_sha → FlashRank top-20 → MMR → top-5 local / top-8 hosted → pack
```
- **GC:** `gc_embeddings()` deletes unreferenced embeddings whose `embedding_touches.touched_at` is older than 24 h. Retired spaces are purged 7 days after retirement (index dropped, rows deleted).
- **Orphan metric:** `rf.retrieval.orphan_hits`. Should be zero; a repair job re-enqueues affected pages.
- **Blue/green spaces:** while `collections.building_space_id` is set, syncs embed into it too (dual write). The flip (`active_space_id = building_space_id`, `building_space_id = NULL`) requires a **coverage check**: 100% of chunks in the collection have an embedding in the new space.

---

## 8. LangGraph agent

### State
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]        # committed history; appended only at turn end
    turn_messages: Annotated[list[AnyMessage], add_messages]   # scratch: AIMessage(tool_calls) + ToolMessages within a turn
    tenant_id: str; user_id: str; collection_ids: list[str]   # from config, never from the LLM
    question: str; hits: list[RankedChunk]; rewrites: int
    draft: Answer | None
    verification: Literal["pending", "verified", "partial", "unsupported"]
    regenerated: bool; prompt_version: str
```

### Graph (● = LLM call)
```
START → load_memory → contextualize ● (follow-ups) → retrieve → grade (top rerank ≥ τ_collection ?)
  grade ─ weak & rewrites < 1 → rewrite ● → retrieve
        ─ weak & exhausted    → abstain
        ─ ok → generate ● (streamed via sanitizer; no tools on questions, ADR-0015)
  generate ─ tool_calls → tools ─ destructive → approval node (interrupt) → execute → generate
           ─ done → verify_citations → ground_check ● → decide
  decide ─ verified|partial → finalize (commit the answer)
         ─ unsupported & !regenerated → generate(strict) → … answer.replace
         ─ unsupported & regenerated  → abstain → finalize
  finalize ─ extra needed (ADR-0016) → extras ● (labelled "Not from your notes", never committed) → END
           ─ otherwise → END
```

### Run lifecycle
- **Conversation run lease** (section 3, Flow B). One active turn per conversation; heartbeat 15 s; the lease is released when the run stops at an interrupt.
- **Checkpoint fencing, at the write itself:** `FencedPostgresSaver` subclasses the LangGraph saver. Its `aput` / `aput_writes` run in **one transaction** (not autocommit):
  1. `assert_run_lease(conversation, run_id)`, a definer function that does `SELECT … FOR UPDATE` on the conversation row and raises unless the lease matches and hasn't expired;
  2. write the checkpoint;
  3. advance `conversations.head_checkpoint_id`.

  A stalled pod that finishes a node after losing its lease fails at step 1, so nothing it writes becomes visible. Every resume and read uses `head_checkpoint_id`, never "latest checkpoint". A stalled-pod test covers this. (LangGraph's `durability="exit"` mode is a possible simplification to evaluate in M4.)
- A node wrapper also calls `heartbeat_run` before each node, to stop doomed runs early. Within a turn, `generate → tools → generate` exchanges live in `turn_messages`. The final step merges the user message and the final answer into `messages` and clears `turn_messages`. Abandoned turns never reach the committed history.
- **Isolation for LangGraph tables is application-level.** The checkpoint and store tables are outside RLS (autocommit pool, no tenant setting). The API only resolves `thread_id` from `conversations` under RLS, and store namespaces are always built from the authenticated `(tenant_id, user_id)`. Tested in the isolation suite.
- **Disconnect** → cancel at the next node boundary; the partial draft is discarded; the client re-asks.
- **HITL:**
  - All destructive tool calls from one `generate` step are batched into **one** approval node. It runs `INSERT pending_approvals … ON CONFLICT (conversation_id, tool_call_id) DO NOTHING` for each call and then a single `interrupt()`. On resume LangGraph **re-runs the node**, so the inserts are idempotent.
  - Resume carries a per-`tool_call_id` decision map.
  - Approved tools execute in a separate node, idempotently (keyed by `tool_call_id`).
  - Resume: `POST /v1/approvals/{run_id}`. This needs the **`approvals:write` scope, which UI and user keys have and MCP keys never do.** An agent can't approve its own destructive actions.
  - **Resume and new-turn are serialized.** Both paths first acquire the conversation run lease (CAS on `active_run_id`), then lock the run's `pending_approvals` rows `FOR UPDATE` in the same transaction.
    - Resume proceeds only if they're still `pending`.
    - A new turn marks them `expired` (audited) before starting.

    An approved destructive tool can therefore never execute on a thread that has moved on.
  - New input to a thread with a pending interrupt starts a new run from `head_checkpoint_id` with `Command(goto=…)` semantics, never resuming the stale node. This LangGraph 1.x behavior is verified by a contract test in M5.

### SSE contract
`token` · `citation` · `tool_call` · `interrupt` · `verification` · `answer.replace` · `extra` · `done` · `error`.
Clients render pending tokens as plain text until `verification` arrives. `extra` (ADR-0016) is general knowledge, not the user's notes: clients show it apart, under "Not from your notes", never as verified. Its `text` is CommonMark (literal escaped prose, plus one fenced block for code) to be rendered as Markdown. Clients that predate it skip it (unknown types are ignored). A turn ends at `done`: clients must not send the next message on `verification`, because an `extra` may still be coming and the conversation lease is held until `done`.

### Memory
| Kind | Mechanism |
|---|---|
| Short-term | `AsyncPostgresSaver` |
| Context window | trim; **(Core)** summary from `conversation_summaries` (written by the `summarize_conversation` job) |
| Long-term **(Core, M5)** | `AsyncPostgresStore`, namespace `(tenant, user, "facts")` |

`remember_fact(user_quote)`: the stored fact **is** the quote. The quote must be ≥ 20 characters and appear verbatim in the user's latest message (deterministic). An optional LLM label is stored as metadata only. Retrieved documents therefore can't plant lasting memories. `forget_fact` requires HITL.

**Tools:** `search_notes`, `get_page`, `list_sources`, `sync_status`, `remember_fact`, `forget_fact` (HITL), `remove_from_index` (HITL; writes `exclusions` + tombstone).

**Prompts and outputs:** versioned templates. JSON-schema output with one repair retry; failure → `partial`.

---

## 9. API (`/v1`, problem+json, `X-Request-ID`, `traceparent`)
```
POST/GET /v1/connectors   POST /v1/connectors/{id}/sync   DELETE /v1/connectors/{id}
POST/GET /v1/collections  POST /v1/collections/{id}/files
GET /v1/documents         DELETE /v1/documents/{id}        (adds an exclusion; ?forget_exclusion to undo)
GET/DELETE /v1/exclusions
GET /v1/jobs/{id}         GET /v1/jobs/{id}/events (SSE)
POST /v1/query            POST /v1/chat (SSE)               POST /v1/approvals/{run_id}  (scope approvals:write)
POST /v1/interview/question   POST /v1/interview/grade   (scope chat; per-principal rate limit)
GET /v1/audit             POST /v1/webhooks/notion [A]
GET /healthz  /readyz  /metrics
```
- **Interview mode (Phase 3b):** stateless server; the client holds the session (asked chunk ids, topic list, difficulty, history).
  - `/question` picks unasked notes for a topic (title/heading full-text match plus hybrid search, under RLS, active space, live documents) or across all notes, and has the model write ONE question plus expected points. Points are ground-checked; unbacked ones are dropped. It returns a signed `question_token`.
  - `/grade` reloads the chunks through RLS (never trusts client text) and grades at a configurable level (default Senior SDE-1). Each missed point and misconception must be backed by the notes; the model answer is verified like a chat answer.
  - Every matching note asked → 409 `urn:ragforge:problem:interview-exhausted`. The mode is started only by fixed phrases or a UI toggle, never by the model (`ragforge_client/interview.py`).
- **`DELETE /connectors/{id}`:** status `deleting`, destroy the ciphertext and DEK immediately → `delete_connector` job (documents, chunks, then GC) → audit.
- **IDs:** document IDs are `uuidv5(tenant_id, collection_id, connector_id or 'upload', external_id)`, the same tuple as the uniqueness key. Chunk IDs are `uuidv5(document_id, ordinal)`. A reindex or restore reproduces the same IDs, so `ragforge://documents/{id}` links and audit targets stay valid.
- **`/readyz`:** the DB is reachable, `schema_meta.schema_version ≥ CODE_REQUIRED_SCHEMA`, and `CODE_VERSION ≥ schema_meta.min_code_version`.
  - The values are monotonic integers. Every migration bumps `schema_version`; only *contract* migrations raise `min_code_version`.
  - **Rule:** a contract migration may raise `min_code_version` only to a value ≤ the **previous** release's `CODE_VERSION`. Migrations run as a pre-upgrade hook while old pods are still serving, so this keeps them ready mid-rollout.
  - So after a rollback, old code stays ready on a newer expanded schema.

## 10. MCP server
- **Tools:** `search`, `ask` (answer + citations + verification, or `needs_approval{run_id, ui_url}`), `list_sources`, `sync_now`.
- **There's deliberately no `approve` tool.** MCP keys lack the `approvals:write` scope. Content injected into the agent's context can't trigger destructive actions. (MCP elicitation can be added later only if it's human-in-the-client.)
  - **Today an MCP-raised pending action can't be approved anywhere.** The UI can't load it until `GET /v1/approvals/{run_id}` exists. The agent is told the truth: the user asks for the action in the UI and approves it there.
- **Resource:** `ragforge://documents/{id}`. Metadata only until a document/chunks endpoint exists.
- **Transports:** stdio (default) and streamable HTTP.
  - HTTP needs `Authorization: Bearer <token>` (not the API key) and has DNS-rebinding protection.
  - A non-loopback host needs `--allow-remote` and a concrete IP.
- **Agent-facing text:** the answer keeps the escapes of `\ [ ] < > & `` ` `` so it stays inert when rendered. Extras are marked `not_from_notes: true`. After the verdict, a dropped stream, a malformed frame or the 180 s limit returns the saved answer with a warning.

---

## 11. Clone-and-run
```bash
git clone … && cd ragforge && make doctor && make up
```
- `bootstrap.sh`:
  - creates `.env` (generates the KEK and admin key locally);
  - chooses an Ollama runtime:
    - **macOS with host Ollama:** `host.docker.internal` (Metal);
    - **Linux:** always the `ollama` compose profile, which never publishes a host port, with GPU via the NVIDIA override when present. Containers on Linux get GPU passthrough, so there's no reason to bind a host Ollama to a network interface;
    - **otherwise:** the `ollama` profile;
  - picks the chat model by RAM (≥ 24 GB → 8B) and pulls models.
- **Compose order:** `postgres` → `db-bootstrap` (one-shot, superuser) → `[ollama → pull]` → `migrate` (as `rf_migrator`: Alembic, functions, LangGraph setup, default space + index) → `api`, `worker`, `mcp`, `ui` → `seed-demo`.
- Postgres isn't published on the host. Uploads live on a named volume. `make connect-notion` prompts for the token.

## 12. Observability, evals, security, operations

**Observability**
- Spans per node and per LLM call.
- Metrics:
  - retrieval stages, TTFT, verified latency;
  - rewrite and regenerate ratios, verification mix;
  - queue depth and oldest age per priority;
  - sync freshness; dead jobs;
  - orphan hits; unsupported blocks.
- Content capture is off by default.

**Evals** ([ADR-0012](adr/0012-evaluation-strategy.md))
- **Corpus:** recorded Notion API block JSON for a synthetic workspace, replayed through the **real Notion normalizer**.
- **Questions:** ~100 questions × 3 paraphrases.
- **CI gate** (runs only when retrieval, ingestion, chunking or embedding paths change; otherwise scores are identical by determinism):
  - **Blocks** if the drop vs `main` is significant: one-sided clustered paired bootstrap, **p < 0.05**, and a point drop > 2 points. That keeps false blocks ≤ 5% by construction. Power at n = 300 rows is ~50% for a 5-point drop, and ~95% once the set reaches ~1k rows (Advanced).
  - **Warns** (a PR comment, not a block) on any point drop > 2.
  - **Blocks** if a metric falls below the frozen baseline − 3 (catches cumulative drift that per-PR power misses).
  - An audited `eval-override` label with a written justification is allowed.
  - CI runs embeddings in one pinned CPU container, so the only churn comes from code changes.
- **Verifier eval set:** ~300 labeled items (~150 unsupported), including synthetic failure modes. Gated on Wilson lower bounds (section 2), with the false-"verified" rate reported.
- τ is tuned on the public corpus as the default. `make eval-private --calibrate` sets the per-collection `tau` from your private set.

**Threat model note:** RLS protects against application bugs (a missing filter, a wrong ID), **not** against a compromised application credential: any `rf_api` session can set `app.tenant_id` itself. Credential compromise is addressed by network isolation, secret handling, short-lived DB credentials in cloud, and audit.

**Security**
- RLS fail-closed; named definer functions with a safe `search_path`; no BYPASSRLS.
- Envelope encryption; hashed keys with constant-time compare.
- Untrusted-source fencing; incremental sanitizer; HITL with audit; quote-gated memory writes.
- Parser sandbox; non-root, read-only-FS containers.
- Pinned digests; Trivy.

**Operations**
- **Migrations:** expand/contract; concurrent index builds in autocommit blocks.
- **SIGTERM:** the API cancels runs after 25 s and releases their leases; workers `release_job` (which merges into an existing queued duplicate if one exists).
- **Backups:** nightly `pg_dump --enable-row-security` of authoritative tables as the read-only `rf_backup`.
- **Runbook:** dead jobs, stuck sync, 429 storm, space migration, key rotation, restore.

## 13. Deployment [A]
- **Helm:**
  - The migrate Job is a `pre-install,pre-upgrade` hook. It reads DB credentials via `secretKeyRef` from **pre-existing** Secrets, and its ServiceAccount is itself a lower-weight hook. It needs no chart ConfigMaps.
  - **Long `CREATE INDEX CONCURRENTLY` builds are never Helm hooks of any kind** (Helm waits on post-hooks too). The migrate hook **enqueues a `build_space_index` maintenance job**, and queries use the exact path until `indisvalid`.
  - The chart **never templates or generates** the KEK, so upgrades can't rotate it.
  - Postgres is outside the release (managed in cloud; a separate pgvector release on kind; `db/init` runs via the provider's admin user or a one-time Job).
- **NetworkPolicies:** Postgres ingress from api, worker, migrate, metrics and the KEDA namespaces.
- **Autoscaling:** HPA on the API; KEDA on the worker using `queue_stats`.

---

## 14. Milestones

| # | Tier | Milestone | Done when |
|---|---|---|---|
| M0 | MVP | Skeleton + **risk spikes** | 5 packages, CI, `make doctor`/`up`. `db/init` + roles + RLS + `enqueue/claim/assert_lease` + `search_vectors` exist as SQL **with cross-tenant, fencing and EXPLAIN tests passing**. Sanitizer state machine with fuzz tests |
| M1 | MVP | Jobs + file ingestion | full job lifecycle (class order + starvation guard, reclaim, race-safe requeue merge), progress SSE, sandboxed parsing, embeddings |
| M2 | MVP | Notion | connect, backfill, incremental, reconcile (staleness + confirmed deletes), exclusions, block fixtures |
| M3 | MVP | Retrieval + evals | exact/HNSW paths, RRF, rerank, MMR; recorded-Notion corpus; ~100 Q × 3 paraphrases; gate + frozen floor; τ |
| M4 | MVP | Agent + verification | graph, SSE, run lease + checkpoint fencing, citation + ground check, abstain, **verifier eval set**; **local SLOs measured and written here** |
| M5 | Core | Long-term memory + HITL | store, quote-gated memory, batched approvals (UI-only scope), summaries |
| M6 | Core | SDK, MCP, UI | MCP in Claude Code (`needs_approval` → UI link; no approve tool); Streamlit with approvals |
| M7–M10 | Adv | Tenancy, ops, Kubernetes, write-up | as in tier table |
| S1–S3 | Stretch | | outbox + Redpanda · Google Drive · LLM gateway |
