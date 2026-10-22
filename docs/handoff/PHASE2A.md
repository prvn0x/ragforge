# Phase 2a: integration findings and wiring

Status: **approved for Phase 2a.** Blind review round 7 gave 8/10 (integration) and 8/10 (runtime/ops), both with no P0 or P1. `make check` is green (1134 tests), and `make nuke && make up` is green at schema 9.

## Integration bugs found (none caught by the Phase 1 tests)
1. **P0: the agent never reranked.**
   - `AgentDeps` had no reranker, and `agent/retrieval.py` called `retrieve(k=retrieval_k)` directly.
   - `Retriever.retrieve` returns fused candidates with `rerank_score=None` by contract (handoff B), and `passes_grade` passes only on a rerank score. So **every real turn would have abstained**.
   - The D1 fakes return hits that already carry scores, so 99 tests stayed green.
   - Fix:
     - `AgentDeps.reranker` and `AgentSettings.rerank_top` (20);
     - `agent/retrieval.py` retrieves `rerank_top` candidates in the tx, then calls `rerank_and_select(..., final_n=retrieval_k)` with **no tx open**;
     - regression test `tests/module_d1/test_d1_rerank.py`.
2. **`/v1/query` returned un-reranked results.** It now reranks the same way (rerank outside the tx, MMR down to `k`).
3. **`list_sources` / `sync_status` would have failed every call:** they read `rf.sync_cursors`, which `rf_api` cannot SELECT. Found by the new real-Postgres tests; the subquery was removed instead of widening grants, so the tools no longer show a "last synced" time.
4. **`DELETE /v1/documents/{id}` skipped `lock_source_item`.** Handoff C2 requires that lock, and without it a sync in flight could bring an excluded page back. It now goes through `SqlSyncStore.add_exclusion`, which takes the advisory lock. The agent's `remove_from_index` (`adapters/db/agent_data.py`) uses the same path.

## Wiring
- **`bootstrap.py`:** `build_api` / `build_worker` are now **async**: they read `embedding_spaces` and check the LangGraph schema. Each container owns an `AsyncExitStack` that closes the httpx client, the LangGraph pool and the chat model.
  - API container: `jobs`, `retriever`, `reranker`, `embedders`, `agent`, `sanitizer_factory` (`MarkdownStreamSanitizer`), `blobs`, `sync_store`.
  - Worker container: `jobs`, `sync: SyncDeps`.
- **New adapters:** `adapters/db/agent_data.py` adds `SqlConversationStore` and `SqlKnowledgeBase` (handoff D1, request 1).
- **Slot 0007:**
  - `pending_approvals.executed_at`, with a CHECK that only approved rows are executed;
  - the `embedding_spaces` retired_at ⇔ retired CHECK.
  - `CODE_REQUIRED_SCHEMA = 7`. See the ADR-0013 Phase 2a amendments.
- **Config:**
  - `chat_num_ctx` is now 8192 (ADR-0004);
  - new: `embed_batch_size`, `rerank_model` / `rerank_cache_dir` / `rerank_top_n`, `vector_exact_threshold`, `agent_*` (tau, retrieval_k, max_tool_rounds, strict_citations, verifier_trusted, summarize_after_messages), `worker_concurrency`, `sync_poll_interval_s`, `max_upload_mb`, `parse_timeout_s`, `parse_memory_mb`;
  - validator: `db_pool_size >= worker_concurrency + 2`.
- **Engine:** `hide_parameters=True` for every role (E2 asked for the API; the worker binds sealed tokens too).
- **Sync overlap:** decision (a). `SyncPolicy.cursor_overlap = 0`, and the connector owns the 5-min overlap.
- **Dependencies:**
  - added `python-multipart` (uploads);
  - pinned `langgraph-checkpoint-postgres==3.2.0`, `langgraph<2` and `langgraph-checkpoint<5`;
  - dropped `notion-client`, which was unused;
  - dev group: `markdown-it-py`, `linkify-it-py` (E1 oracle);
  - console script `ragforge-admin`.

## Landed in parallel
- **API (`apps/api`):**
  - `POST /v1/collections/{id}/files`: the collection is checked first, so a foreign one is a 404; reads at most the cap + 1 byte (422 when over); calls `accept_upload`; returns 202 `{job_id}`. Tests are in `module_e2/test_e2_uploads.py`.
  - `sync_connector` / `delete_connector` jobs no longer set `connector_id` (handoff A: only jobs that count against the in-flight cap set it).
- **Worker (`apps/worker`):**
  - `SYNC_KINDS` go through `run_sync_job`. The runtime's `_finish` is fenced, so there is no double completion.
  - `summarize_conversation` is a no-op.
  - Upkeep: the scheduler runs every 30 s (`list_due_connectors` → `sync_connector`); GC (`gc_embeddings` until 0, then `purge_tombstones` until fewer than the limit) runs every 6 h.
  - The worker waits for the schema before wiring. Tests are in `module_worker/`.
- **Migrator:** `apps/index/space_indexes.py` (`ragforge-index`, its own one-shot) builds space indexes for non-retired spaces (dim ≤ 2000) and drops them for retired ones, on autocommit as `rf_system`. It is idempotent. Tests are in `tests/test_migrate_space_indexes.py`.
- **Docker / compose:**
  - The FlashRank TinyBERT model is baked into `/opt/models/flashrank` (+7 MB), root-owned and loads on the read-only filesystem.
  - `stop_grace_period: 35s` for api and worker.
  - Prompts, `ragforge-admin` and the parser sandbox were all checked inside the locked-down image.

## First use (2026-10-16, llama3.2:3b + nomic-embed-text on an M1 Pro)
- `ragforge-admin create-tenant` → collection → markdown upload: `ingest_file` was done on its first attempt (2 chunks embedded).
- `/v1/query`: the right chunk reranked to 0.995, the wrong one to 1e-5.
- `/v1/chat`: the answer was correct (on-call rotation and acknowledgement time), in about 25 s over 3 model calls.
- An off-topic question → server abstain with the closest pages (`verified`).
- **Model-quality findings for Phase 2b (not wiring bugs):**
  - The 3B ground check false-flagged a verbatim sentence (`partial`).
  - On a follow-up, the 3B model called `search_notes` with its JSON schema as the argument (`{"query": {"type": "string"}}`). The closed schema rejected it, and the verifier then replaced the unsupported draft with the server abstain. That is correct and safe, but the user did not get the answer.
  - Candidates: a stronger chat model, no tools on the first draft for small models, `agent_verifier_trusted=false` if the verifier eval fails ADR-0012.

## Blind review round 1 (two fresh reviewers, no REVIEW.md access): 6/10 and 5/10
I had written that the scheduler could add "at most one extra listing per poll". The review showed that was wrong: running jobs were never deduped, so listings could multiply. Fixed by five parallel forks; `make check` gives 1031 passed, and `make nuke && make up` is green at schema 8.
- **Scheduler (slot 0008):**
  - `rf.claim_due_connectors` and `rf.claim_reconcile_connectors` are SECURITY DEFINER functions for `rf_worker` only. They skip a connector that has a queued or running listing, and they move `next_poll_at` / `next_reconcile_at` forward as a lease under `SKIP LOCKED`.
  - This fixes duplicate listings, a scheduler merge resetting a retry's backoff, and failing connectors starving healthy ones.
  - A trigger counts dead listings per connector, and the lease doubles per failure up to 1 day.
  - Reconcile is now scheduled daily with 10% jitter.
  - Each upkeep step runs in its own task, and the GC drain checks for shutdown between batches.
  - The worker rechecks readiness every 30 s and exits to restart if the schema becomes incompatible.
  - `worker_db_pool_size` is separate from the API pool size, and `pool_timeout` is 10 s.
- **Migrator index step:** it can no longer fail a deploy (errors are logged, exit 0); it runs with no lock or statement timeout, under an advisory lock; it never drops an index that is still being built.
- **Uploads:** authentication and Content-Length checks happen before the body is read (401/411/413), the body is parsed with a byte counter, unsupported types get 415, the hash runs off the event loop, and compose caps `/tmp` (tmpfs) at 256 MB.
- **Lock order:** `DELETE /documents` and `remove_from_index` now take the source-item advisory lock before the row lock, matching sync. A two-connection test proves it.
- **Reranker:** it is loaded eagerly. `build_api` fails startup if the model is missing, instead of silently abstaining on every turn.
- **Embedder registry:** on a miss it reloads in the background and raises a retryable 503 rather than a permanent `NotFound`.
- **Composed end-to-end test:** `module_integration/test_int_composed.py` runs the real `build_api` with only the models scripted. It covers chat, approval, an exactly-once removal, the exclusion, and a second resume returning Conflict, plus cleanup after a failed startup.
- **Image:** the FlashRank zip comes from a pinned Hugging Face commit with a verified sha256, in a stage that code edits don't invalidate. `flashrank` is pinned to `==0.2.10`.
- **Re-run in the real stack:** an upload without a key → 401, 60 MB → 413, PDF → 415, a markdown ingest succeeds.
  - Chat abstained again: llama3.2:3b called `get_page` with `document_id: "[S2]"`. Across 4 real turns the 3B model misused a tool 3 times; each time the bad arguments were rejected and the server abstained.

## Known gaps (accepted for now)
- The agent, `/v1/query` and the sync indexing service still call the sync `for_space`. A space miss is now a retryable 503 with a background reload, not an awaited one; switching them to `ensure()` needs a port change.
- `?forget_exclusion=true` removes the exclusion but enqueues no re-index.
- `remove_from_index` writes a tenant-wide exclusion (the exclusion key is per tenant), not one per collection.
- The uploads row and blob of a deleted upload are never purged.
- The worker does not use the `chat_active` throttle yet.
- Embedding spaces are loaded at startup, so a newly created space needs a process restart.
- `summarize_conversation` is a no-op handler until Phase 2b, when a tenant-scoped saver read path is needed (ADR-0013 amendment).
- `POST /v1/webhooks/notion` and `/metrics` are not implemented.
- PDF uploads are rejected as unsupported (`pypdf` not added).
- If the schema is behind at startup, the API crash-loops (the spaces query fails) instead of starting and answering `/readyz` 503. Compose ordering (migrate first) hides this locally; it matters for rolling deploys.


## Blind review rounds 2–7 (two fresh reviewers each round, no REVIEW.md access)
| Round | Integration | Ops | Open P1s | Tests |
|---|---|---|---|---|
| 1 | 6.0 | 5.0 | 8 | 991 |
| 2 | 7.0 | 7.0 | 2 | 1031 |
| 3 | 7.5 | 7.5 | 1 | 1064 |
| 4 | 7.5 | 7.5 | 1 | 1088 |
| 5 | 7.5 | 8.0 | 1 | 1108 |
| 6 | 8.0 | 7.5 | 1 | 1123 |
| 7 | **8.0** | **8.0** | **0** | 1134 |

Rounds 3–6 each found one P1, and each time it was in code the previous fix round had just written:
- undo-delete within 60 s did nothing;
- undo revived pages the owner had deleted in Notion;
- the "no rows" fix for `list_due_connectors` stopped all polling after a rollback.

Lesson: every fix needs an end-to-end test through the real handler (not "a job was enqueued"), plus a mutation check. From round 4 on, every fix had both.

The main changes these rounds made:
- **API:** a body limit on every route (auth runs before the body is read); upload slots, deadlines and per-tenant limits.
- **Undo-delete:** slot 0009 `deleted_reason`. Notion undo is a re-fetch; a "page gone" result becomes a confirmed `source_deleted` tombstone.
- **Tombstones:** every copy of a page is tombstoned; an upgrade to `source_deleted` stamps the confirmation time.
- **Embeddings:** `EmbedderRegistry.ensure()` is a deliberate port change. Chunks are capped at 445 tokens so the reranker always scores whole chunks.
- **Runtime:**
  - reclaim bumps `lease_version` and requeues with backoff;
  - the heartbeat never blocks on the holder's own transaction;
  - `assert_lease` extends the lease;
  - jobs have a per-kind max runtime, and a handler that ignores cancellation keeps its slot until the lease expires;
  - unknown job kinds are deferred, then buried after 24 h.
- **Scheduler:** the claim lease is separate from the failure backoff (trigger-based, with jitter), and reconcile has its own lease and trigger.
- **Old images:** the legacy `list_due_connectors` became a safe claim, and migrate accepts a schema newer than the code (rollback).
- **Migrate and index:** concurrent migrate runs serialize. Migrate pre-empts only sessions named `ragforge-index`, and checks checksums before upgrading. Index builds run as a separate one-shot.

## P2 backlog after round 7 (accepted; schedule before production)
**Integration:**
- 0009's backfill marks old exclusions `source_deleted`, so undo can't restore them. Backfill from `rf.exclusions`; file documents should always be restorable.
- The registry's single-flight wait can call a reload "fresh" when it started before the caller.
- No per-tenant upload quota, and no erase path for upload rows or blobs (blob reaper).
- Undo enqueues a refresh for inactive connectors.
- `_tombstone_gone` writes an audit row even when 0 rows changed.

**Ops:**
- Slot 0008 locks connectors before jobs (the reverse of the listing commit), and its volatile default rewrites the whole table. Under live workers, migrate can deadlock or hit the lock timeout, and nothing retries.
- Liveness stays green when abandoned handlers hold every slot.
- The worst-case shutdown is longer than the 35 s grace (pool timeout plus cleanup).
- Docstrings overstate two things: `assert_lease` extends from the start of the fence, not from commit, and crashed jobs still charge an attempt to the other jobs that were running alongside them.
- Reconcile failures are silent on the connector.
- A listing that runs longer than 2 h never makes progress (no cursor checkpoint).
- The BuildKit frontend is not pinned by digest.

## First real Notion sync (2026-10-17): 2 bugs found and fixed in a parallel session
1. **Oversized atomic chunks.** Rule 4 ("code is never split") had no upper bound. One page's code block exceeded nomic-embed-text's 2048-token context: Ollama returned 400 and the page retried until dead. Fixes:
   - The chunker gets a `hard_max_tokens=1024` ceiling. Larger atomic blocks are split at line boundaries, and each code piece is re-fenced.
   - `count_tokens` is conservative for digit-bearing words. The 6-char rule under-counted hex/ids by more than 2x, measured against the REAL model.
   - `OllamaEmbedder`: an input-rejected 400/413/422 → `EmbeddingInputRejected` (a ValidationFailed, so permanent). `embed_documents` isolates a rejected item and embeds its halves (L2-normalised mean, depth ≤ 6), so an estimate error can never make a page unindexable.
   - Verified: 0/80 adversarial chunks rejected by real Ollama (truncate=false, num_ctx 2048).
2. **`heading_4` dropped.** It's a newer Notion block type, and its text was silently skipped. Added to `HEADING_TYPES`.

Result: 152/152 pages, 3213 chunks, 0 dead jobs, 0 chunks without embeddings, and no unsupported blocks. 1146 tests green. A full re-sync was forced (cursor deleted, fetched_at nulled), because unchanged pages would not have been re-fetched to recover the H4 text.
Lesson: verify sync health from job outcomes and worker logs, never from connector status alone.

## First real Q&A against the Notion tenant (2026-10-17, llama3.2:3b, M1 Pro)
- **Setup:** a new `ragforge-admin create-key --tenant <id> --name eval [--scopes read,chat]` mints a least-privilege key for an existing tenant. It can never grant `approvals:write` or `admin`, and the creation is audited.
- **Questions:** 10 topical (Kafka, Redis, databases, LLD, HLD) and 2 off-topic. The runner and raw results stayed in the session scratchpad.
- **Retrieval:** the right page was the top hit on 10/10, with rerank 0.96–0.9999. Off-topic questions top out at about 1e-5.
- **Off-topic:** 2/2 abstained correctly (`verified`) in about 1 s.
- **Topical questions:**
  - **Abstained, 6/10:** the model misused a tool on its first draft, the arguments were rejected, and the server abstained (`unsupported`). The misuses:
    - `get_page` with `document_id: "[S1]"`, 3 times;
    - `search_notes` with `{"query": {"type": "string"}}`, i.e. its own schema, twice;
    - `search_notes` with `"[[query]]"`.
  - **Answered, 4/10, all `partial`:** Q3 exactly-once, Q6 Redis lock, Q7 sharding, Q8 consistent hashing. None cited a single source, though the prompt requires `[S<n>]`. Q6's content was half invented (a Redlock "unique ID per instance" description, plus an unprompted `remember_fact`).
- **Citations:** 0/12 answers cited anything.
- **Latency:** time to first token 6.6–18.9 s (median ≈ 9 s); total 17–65 s; 3–5 model calls per turn.
- **Wiring checks:** tool schemas are well-formed JSON Schema, the sources the model sees carry `document_id`, and the prompt does require citations. **No wiring bugs found; every failure is model quality.**
- **Phase 2b, in priority order:**
  1. No tools on the first draft for small models.
  2. A stronger local chat model (e.g. qwen2.5:7b or llama3.1:8b; fits in 16 GB).
  3. `agent_strict_citations=true` to force cited answers.
  4. Measure all three on the eval set.
