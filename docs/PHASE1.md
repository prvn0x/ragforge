# Phase 1: module ownership and rules for parallel builders

Five builders (A–E) work **in the same folder at the same time** (no git worktrees). These rules keep them from colliding. Each module ends with a blind principal review of that module only.

## Golden rules
1. **Edit only the paths you own** (table below). If you need a change elsewhere, record it in your module's `HANDOFF.md` instead. The integrator (Phase 2) applies it.
2. **Never edit shared contracts:** `packages/contracts/**`, `ragforge_core/domain/**`, `ragforge_core/config.py`, `ragforge_core/db/engine.py`, `0001_base.py`, root `pyproject.toml`, `compose.yaml`, `Makefile`. If a port or schema must change, write the proposed change in `HANDOFF.md`.
3. **Migrations:** edit **only your reserved slot** (0002–0006). Keep `bump_schema_version()` as the last line. Don't create new revision files.
   - Your slot is **already applied** (as a no-op) in every dev database, so Alembic won't re-run it after you edit it. `ragforge-migrate` detects this with a per-revision checksum and fails loudly.
   - In dev, run `make nuke && make up` after editing your slot. Tests always use a fresh database, so they're unaffected.
4. **Dependencies:** list any new library you need in `HANDOFF.md`. The integrator adds it to `packages/core/pyproject.toml`. Run `make sync` to install every package and all extras (`agent`, `ingest`). CI and the server image install them too.
5. **Tests live in your module's test folder** and reuse `packages/core/tests/conftest.py` fixtures (`settings`, `superuser`, `api_engine`, `make_tenant`). Don't fork them; ask for new fixtures via `HANDOFF.md`, or define module-local fixtures in your own `conftest.py`.
6. **No real network or LLM in tests:** fake the `ChatModel`/`Embedder` ports and use recorded Notion JSON. Real Ollama is wired in Phase 2.
7. **Definition of done:** `make lint typecheck imports` is green, your tests pass, and `uv run pytest packages` (everything) still passes.
8. **Transactions:** every port method touching tenant data takes the caller's `tx` (from `TransactionManager.tenant()`), so writes compose atomically. Never open your own transaction inside a tenant-scoped port method. Result transactions start with `JobQueue.assert_lease(tx, lease)` and end with `JobQueue.complete(tx, lease)`.
9. **IDs:** document and chunk IDs come **only** from `ragforge_core.domain.ids` (`document_id`, `chunk_id`). Never derive them yourself.
10. **Every SECURITY DEFINER function** follows ADR-0013:
   - `SET search_path = pg_catalog, rf, ext, pg_temp`;
   - `REVOKE ALL … FROM PUBLIC` plus explicit `GRANT EXECUTE`;
   - schema-qualified pgvector operators;
   - a cross-tenant test.

## Ownership

| Module | Owns (create/edit) | Migration slot | Implements ports | Key docs |
|---|---|---|---|---|
| **A: Data plane** | `ragforge_core/adapters/db/jobs.py` (also: monthly `jobs_history` partitions + `jobs_claim` index tuning in slot 0002), `ragforge_core/services/jobs/**`, `ragforge_core/apps/worker/**`, `packages/core/tests/module_a/**` | `0002_data_plane.py` | `JobQueue` | ADR-0010, 0013 |
| **B: Retrieval** | `ragforge_core/rag/**`, `ragforge_core/adapters/db/search.py`, `ragforge_core/adapters/llm/embedder_*.py`, `ragforge_core/adapters/rerank/**`, `packages/core/tests/module_b/**` | `0003_retrieval.py` | `Chunker`, `Embedder`, `Reranker`, `Retriever` | ADR-0006, 0009; DESIGN §7 |
| **C: Connectors** | `ragforge_core/adapters/connectors/**`, `ragforge_core/adapters/blob/**`, `ragforge_core/services/sync/**`, `packages/core/tests/module_c/**` (incl. recorded fixtures) | `0004_connectors.py` | `SourceConnector`, `RateLimiter`, `BlobStore`, `DocumentParser` | ADR-0007; DESIGN §3 Flow A, §7.1 |
| **D: Agent** | `ragforge_core/agent/**`, `ragforge_core/adapters/llm/chat_*.py`, `ragforge_core/adapters/db/checkpoint.py`, `ragforge_core/prompts/**`, `packages/core/tests/module_d/**` | `0005_agent.py` | `ChatModel` | ADR-0008, 0011; DESIGN §8 |
| **E: Safety + API** | `ragforge_core/agent/sanitize.py` *(only this file under agent/)*, tenant provisioning (`provision_tenant` definer function + `ragforge-admin` CLI for the first key), API keys, `ragforge_core/apps/api/**` except `main.py`, `ragforge_core/services/auth/**`, `packages/client/**`, `packages/core/tests/module_e/**` | `0006_api.py` | `OutputSanitizer` | ADR-0005, 0011; DESIGN §9 |

Module D must not create `agent/sanitize.py`; module E owns it. **Only E calls `OutputSanitizer`. D never sanitizes**: it yields RAW text in `TurnText`/`TurnReplace`. Markdown escaping isn't idempotent, so sanitizing twice corrupts answers. D emits `TurnSources(hits)` after every retrieval and before any text that may link to those hits. E calls `sanitizer.allow(urls)` on each `TurnSources` event.

## Database function ownership (one owner each; matches ADR-0013 and the slot docstrings)
| Slot | Owner | Functions / objects |
|---|---|---|
| 0002 | A | `enqueue_job`, `claim_jobs`, `assert_lease`, `heartbeat`, `complete_job`, `fail_job`, `release_job`, `reclaim_expired_leases`, `requeue` (internal), `move_finished_jobs` + monthly `jobs_history` partitions (keep the DEFAULT partition empty), `list_due_connectors`, `queue_stats` view |
| 0003 | B | `chunks.tsv` trigger + GIN, `search_vectors`, `search_vectors_exact`, `search_keywords`, `gc_embeddings`, space index build support |
| 0004 | C | `take_tokens`, `route_notion_webhook`, `purge_tombstones` |
| 0005 | D | `assert_run_lease`, `heartbeat_run`, `chat_active`, LangGraph checkpoint/store tables + grants |
| 0002 | A | also `report_job_progress` (definer; backs `JobQueue.report_progress`) |
| 0006 | E | `authenticate_api_key`, `provision_tenant`, API views |

`packages/core/tests/test_catalog_invariants.py` checks **every** function and view any module adds:
- definer `search_path` is exact;
- no PUBLIC execute;
- views use `security_invoker`;
- no PUBLIC table grants.

It runs in everyone's test suite.

**Shared-folder tip:** another builder's half-edited slot can break the session fixture for everyone, because migrations run through every slot. While iterating, set `RAGFORGE_TEST_TARGET_REVISION=<your slot>` (for example `0002`) to migrate only up to your slot. The full suite must pass before you report done.

**Conventions every module follows:**
- **No model calls inside a transaction.** Embed or chat first, then open a short transaction. `DocumentIndex` documents the 3-step sync flow.
- **SHA-256 values are raw 32-byte `bytes`** everywhere (DB `bytea`), never hex.
- **Tool-call IDs** are minted by the ChatModel adapter (uuid4). Never pass provider IDs through.
- **`NOTIFY` payloads carry only an ID** (for example `job_id`), never tenant data. Listeners re-read the row under RLS.
- **Module-local persistence:** a module that needs its own tables accessed (C: cursors, connectors, exclusions) defines a small Protocol in its own `services/<module>/ports.py` and implements it in its own adapter path. import-linter forbids `services` from importing `db`/`sqlalchemy` directly.
- **B's space index builds** (`CREATE INDEX CONCURRENTLY`) can't run inside a function or a transaction. Ship them as a migrator-run step (an Alembic `autocommit_block` in slot 0003 for the default space; a `ragforge-migrate` sub-step for new spaces is planned for Phase 2).
- **Shared infra files** (`docker/Dockerfile`, `compose.yaml`, `Makefile`, `config.py`) belong to the Phase 2 integrator. Request changes in `HANDOFF.md`, for example B's baked-in FlashRank model or new settings.
- **Changes to 0001 tables from your slot:** additive only. Allowed: `GRANT`s, new indexes, triggers, and new **nullable** columns. Never drop, rename, retype or tighten an existing column or constraint. If you need that, write a `HANDOFF.md` request.
- **Adapter paths for module-local persistence:** C → `adapters/connectors/**` (cursors, connectors, exclusions); D → `adapters/db/checkpoint.py` and `adapters/db/conversations.py` (`ConversationRuns`). Adapters may import only domain types, so keep module-local Protocols' signatures to domain types and primitives.
- **`documents.fetched_at` is nullable** (uploads have no Notion fetch). Monotonic guards must use `COALESCE(documents.fetched_at, '-infinity')`.
- **New conversations:** when `conversation_id` is null, E sets it on `InterruptEvent`, `ErrorEvent` and `DoneEvent`, so clients always learn the new ID.
- **Slot 0003 (B):** test `CREATE INDEX CONCURRENTLY` inside `op.get_context().autocommit_block()` together with the checksum hook early. It's the one migration mechanic Phase 0 hasn't exercised.

**Seam ports** (frozen): `DocumentIndex` (B, called by C), `AgentRunner` + `ConversationRuns` (D, called by E), `SecretBox` (Phase 0: E seals tokens, C opens them, with `associated_data = b"connector:<tenant_id>:<connector_id>"`), and the adapter helper `adapters.db.tx.connection(tx)`.

## Interfaces between modules (frozen in Phase 0)
- A ← everyone: jobs are enqueued via `JobQueue.enqueue(JobSpec)`. Side-effect transactions start with `assert_lease`.
- C → B: `SourceDocument` (normalized blocks with anchors) goes into `Chunker.chunk()` and becomes `Chunk`s.
- B → D: `Retriever.retrieve()` returns `RankedChunk`s.
- D → E: `ChatModel.stream()` yields `TextDelta | ToolCall | UsageReport`. Text passes through `OutputSanitizer` and comes out as SSE events (`ragforge_contracts.sse`). `UsageReport` feeds `DoneEvent.usage`, and tool calls feed `tool_call` / `interrupt`.
- Contracts: requests forbid unknown fields, while responses and events ignore them. `decode_sse_data` returns `None` for unknown event types, so clients must skip them.

## Phase 2 integration (not a Phase 1 task)
Wiring adapters in `bootstrap.py`, worker job dispatch, real Ollama adapters end to end, the eval corpus, and latency measurement.
