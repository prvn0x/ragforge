// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-phase1-build',
  description: 'Phase 1: 8 parallel module builders, each followed by blind principal review and fix rounds',
  phases: [
    { title: 'Build', detail: '8 builders implement their module + tests in owned paths' },
    { title: 'Review', detail: 'blind principal review per module' },
    { title: 'Fix', detail: 'fix P0/P1 findings, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'

const COMMON = `
You are a senior engineer building ONE module of RAGForge in a shared folder where 7 other engineers build other modules AT THE SAME TIME.
Repository: ${ROOT}  (NOT a git repo: never run git commands. Do not commit.)

READ FIRST: ${ROOT}/docs/PHASE1.md (rules, ownership, conventions: these are binding), ${ROOT}/README.md, ${ROOT}/packages/core/src/ragforge_core/domain/ports.py, domain/models.py, domain/ids.py, domain/errors.py, ${ROOT}/packages/contracts/src/ragforge_contracts/*.py, ${ROOT}/packages/core/src/ragforge_core/db/migrations/versions/0001_base.py, ${ROOT}/packages/core/src/ragforge_core/db/migrations/helpers.py, ${ROOT}/packages/core/tests/conftest.py, plus the design docs named in your module spec.

HARD RULES (violating these breaks other engineers):
1. Create/edit ONLY the paths you own (listed below). Never edit domain/**, contracts/**, config.py, db/engine.py, bootstrap.py, 0001_base.py, root pyproject.toml, packages/*/pyproject.toml, uv.lock, compose.yaml, Makefile, Dockerfile, docs/** (except your HANDOFF file). Never run uv add / uv lock / uv sync.
2. Anything you need outside your paths (a dependency, a setting, a port change, a fixture, a pyproject script entry) goes in your module handoff file: ${ROOT}/docs/handoff/<MODULE_ID>.md (create it). Do NOT change frozen ports; if a port is truly insufficient, implement the closest correct thing and document the proposed change there.
3. Migrations: only your reserved slot file, keep bump_schema_version() as the last statement, keep the slot syntactically valid at all times (others' test sessions migrate through it).
4. Formatting/linting: run ruff ONLY on your own paths, e.g. \`uv run ruff format <your paths>\` and \`uv run ruff check --fix <your paths>\`. NEVER run \`ruff format packages\` or \`make fmt\` (it would rewrite other engineers' in-progress files).
5. Tests: put them under your test dir; test file names MUST be prefixed with your module id (e.g. test_a_claim.py) because test dirs have no __init__.py. Reuse fixtures from packages/core/tests/conftest.py (settings, superuser, api_engine, make_tenant; helper \`from pgtest import connect\`). While iterating run only your tests with RAGFORGE_TEST_TARGET_REVISION set to your slot when you have one, e.g. \`RAGFORGE_TEST_TARGET_REVISION=0002 uv run pytest packages/core/tests/module_a -q\`. Do NOT run make up / make nuke / docker compose down (shared dev stack). Integration tests use testcontainers (Docker is running).
6. Other modules are being built concurrently and are NOT available: depend only on the frozen ports; use small fakes inside your own tests for any other module's port.
7. No real network or LLM in tests (use respx for HTTP; fakes for models).
8. Principal-engineer quality bar: correct under concurrency, secure (tenant isolation, no secret leakage), typed (mypy --strict clean for your files), small focused functions, docstrings explaining WHY for non-obvious decisions, tests that PROVE the guarantees in the design (not just happy paths). Follow ADR-0013 rules for every SECURITY DEFINER function (SET search_path = pg_catalog, rf, ext, pg_temp; REVOKE ALL FROM PUBLIC; explicit GRANT EXECUTE; schema-qualified pgvector operators; a cross-tenant test). The catalog invariant tests in packages/core/tests/test_catalog_invariants.py must keep passing.

DEFINITION OF DONE: your tests pass; \`uv run ruff check <your paths>\` and \`uv run ruff format --check <your paths>\` clean; \`uv run mypy <your source paths>\` clean (strict config from root pyproject); \`uv run lint-imports\` passes; and \`uv run pytest packages/core/tests/test_catalog_invariants.py packages/core/tests/test_db_foundation.py -q\` still passes. Before finishing, also try the full suite once (\`uv run pytest packages -q\`); if failures come ONLY from other modules' in-progress work, note them and do not try to fix them.
`

const MODULES = [
  {
    id: 'A', title: 'Data plane: job queue + worker loop',
    owns: 'packages/core/src/ragforge_core/adapters/db/jobs.py, packages/core/src/ragforge_core/services/jobs/**, packages/core/src/ragforge_core/apps/worker/**, packages/core/src/ragforge_core/db/migrations/versions/0002_data_plane.py, packages/core/tests/module_a/**',
    slot: '0002',
    docs: 'docs/adr/0010-job-scheduling-leases-priorities-fairness.md, docs/adr/0013-db-roles-and-tenancy-model.md, docs/adr/0003-no-outbox-or-broker-in-v1.md, docs/DESIGN.md sections 3 and 6',
    spec: `Implement the JobQueue port and the worker runtime.
- Slot 0002 SQL (all SECURITY DEFINER, owned by rf_system): enqueue_job (tenant from rf.current_tenant(), ON CONFLICT (tenant_id, dedupe_key) WHERE status='queued' DO UPDATE priority=LEAST, run_after=LEAST; NOTIFY rf_jobs with job id only), claim_jobs(worker_id, n, prefer_low boolean) with strict class order per class using the (priority, enqueued_at) index and FOR UPDATE SKIP LOCKED, skipping connectors at their in-flight cap via connector_inflight (lock + increment), lease_version+1, lease 60s; the every-4th-claim starvation guard is driven by the worker passing prefer_low (oldest-head among maintenance/backfill); assert_lease (FOR UPDATE, raise unless version+status match), heartbeat, complete_job, fail_job (backoff min(2^attempts*5s,15m) +/-20% jitter, Retry-After override, permanent flag -> dead, dead writes an audit row), release_job, reclaim_expired_leases, internal requeue() merge loop catching unique_violation (merge into existing queued duplicate, mark superseded), report_job_progress (+ NOTIFY rf_job_progress, payload job id only), connector_inflight decrement on every exit path, move_finished_jobs to jobs_history plus creation of monthly partitions (current + next month; DEFAULT partition must stay empty), list_due_connectors (connectors with next_poll_at <= now(), status active), queue_stats view (aggregates only, the one view allowed without security_invoker). Grants: rf_api EXECUTE enqueue_job; rf_worker EXECUTE the worker set; rf_metrics SELECT queue_stats.
- adapters/db/jobs.py: PgJobQueue implementing JobQueue exactly (use adapters.db.tx.connection(tx) to get the AsyncConnection; map lease loss to domain.errors.LeaseLost).
- apps/worker: claim loop with a handler registry (register(kind, async handler(lease)) API in services/jobs), dedicated LISTEN rf_jobs connection (psycopg, not pooled) + 2 s poll fallback, per-job heartbeat task (stop work on lost lease), concurrency limit, every-4th-claim prefer_low, SIGTERM: stop claiming, finish within 25 s or release; keep the existing schema-readiness gate and heartbeat file.
- Tests that PROVE: zombie fencing (lease reclaimed -> stale worker's result tx fails and commits nothing), N concurrent workers claim each job exactly once, requeue-merge under concurrent enqueue, cross-tenant same dedupe_key does not merge, strict class order + starvation guard, connector in-flight cap race-free, permanent failure goes dead with audit row, backoff bounds, progress NOTIFY carries only the id, catalog invariants still pass.`,
  },
  {
    id: 'B', title: 'Retrieval: chunker, embeddings, search, rerank, DocumentIndex',
    owns: 'packages/core/src/ragforge_core/rag/**, packages/core/src/ragforge_core/adapters/db/search.py, packages/core/src/ragforge_core/adapters/db/index.py, packages/core/src/ragforge_core/adapters/llm/embedder_*.py, packages/core/src/ragforge_core/adapters/llm/__init__.py, packages/core/src/ragforge_core/adapters/rerank/**, packages/core/src/ragforge_core/db/migrations/versions/0003_retrieval.py, packages/core/tests/module_b/**',
    slot: '0003',
    docs: 'docs/adr/0009-embedding-spaces-and-content-addressed-embeddings.md, docs/adr/0006-onnx-reranker-and-score-based-grading.md, docs/adr/0004-host-first-ollama.md, docs/DESIGN.md sections 2, 3 (Flow A), 6, 7',
    spec: `Implement Chunker, Embedder, EmbedderRegistry, Reranker, Retriever and DocumentIndex.
- rag/chunker.py: DESIGN 7.2 exactly (target 400 / max 512 tokens, heading-first, never cross a heading unless section < 80 tokens, 15% overlap only inside long sections, code blocks never split, table row-groups with header repeated, contextual header "{title} › {heading_path}" in embedded_text only, sha256 as raw bytes, ids via domain.ids). Token counting behind one function (a documented approximation is acceptable if no tokenizer is cheaply available; note it in your handoff). Hypothesis property tests: full coverage of source text, code never split, table chunks start with header.
- adapters/llm/embedder_ollama.py: Ollama /api/embed via httpx, per-space doc/query prefixes, fixed num_ctx, keep_alive, batching; an EmbedderRegistry implementation mapping space_id -> embedder from embedding_spaces rows. respx tests.
- adapters/rerank/flashrank.py: FlashRank with max_length=512, top-20, run in a thread; graceful if the model is not available (document Phase 2 baking in handoff).
- Slot 0003: chunks.tsv trigger using the collection's fts_config + GIN index; search_vectors (dynamic EXECUTE splicing ONLY integer space_id and dim from embedding_spaces; tenant from rf.current_tenant(); explicit tenant AND collection predicate; hnsw.ef_search=100, iterative_scan=relaxed_order, max_scan_tuples=100000; over-fetch 2x then re-sort in a MATERIALIZED CTE; join chunks on embed_sha, DISTINCT ON chunk), search_vectors_exact (function attribute SET enable_indexscan = off), search_keywords (one websearch_to_tsquery per distinct fts_config), gc_embeddings (delete unreferenced embeddings whose embedding_touches.touched_at < now()-24h; purge retired spaces after 7 days), and the default space's partial expression HNSW index built CONCURRENTLY inside op.get_context().autocommit_block() (verify the checksum hook still records the revision). Grants to rf_api/rf_worker as needed.
- Retriever: pure functions in rag/ (RRF k=60, collapse by display_sha256, MMR lambda 0.7, packing) + a structural search Protocol defined in rag/; the SQL implementation lives in adapters/db/search.py using domain types only (adapters may not import rag). spaces_for returns ACTIVE spaces only; small-tenant exact path via cached collection_space_stats (< 20000) or index not yet valid.
- adapters/db/index.py: DocumentIndex exactly as documented (3-step flow; monotonic upsert on fetched_at with COALESCE(..,'-infinity'); unchanged page_sha -> bump fetched_at only; touches in embedding_touches; insert_embeddings ON CONFLICT DO NOTHING; collection_space_stats upkeep; tombstone; is_excluded).
- Tests that PROVE: cross-tenant isolation of every definer function, exact vs HNSW path selection, k results under a 1% selective filter, plans captured with auto_explain (log_nested_statements) after 6+ calls show the partial index is used, monotonic upsert in both commit orders, GC never deletes a touched hash, RRF/MMR unit tests, catalog invariants pass.`,
  },
  {
    id: 'C1', title: 'Notion connector: client, rate limiter, normalizer',
    owns: 'packages/core/src/ragforge_core/adapters/connectors/__init__.py, packages/core/src/ragforge_core/adapters/connectors/notion/**, packages/core/src/ragforge_core/adapters/connectors/ratelimit.py, packages/core/src/ragforge_core/db/migrations/versions/0004_connectors.py, packages/core/tests/module_c1/**',
    slot: '0004',
    docs: 'docs/adr/0007-notion-polling-primary-webhooks-optional.md, docs/DESIGN.md sections 3 (Flow A) and 7.1',
    spec: `Implement SourceConnector (Notion) and RateLimiter.
- Notion HTTP client (httpx): Notion-Version 2025-09-03 pinned, bearer token passed in (never logged, never in exceptions), pagination, Retry-After/backoff on 429 and 5xx, data-source endpoints for database rows, recursive block fetch depth <= 10.
- Normalizer per DESIGN 7.1 table: every block type -> domain NormalizedBlock with anchor = block id (deep link url#blockid without dashes), headings with levels, toggles expanded, code (language, never_split), tables as markdown with row groups (header repeated, never_split per group), synced_block resolved and recorded in synced_source_ids, child_page/link_to_page/mentions -> links_to, data-source rows -> separate documents with properties in metadata, equations, media caption only, unknown types skipped with a counter.
- NotionConnector implementing SourceConnector: list_changed(since) via search sorted by last_edited_time desc with an inclusive 5-minute overlap; fetch(ref) sets fetched_at = Date header of the FIRST response minus 60 s; list_all; confirm_deleted (True only for 404, in_trash/archived, or object not shared; any other error -> False).
- RateLimiter adapter calling slot-0004 take_tokens(integration_key bytea, n) (definer; atomic refill with clock_timestamp(); 3 req/s; returns wait ms) keyed by sha256(token); also route_notion_webhook and purge_tombstones (hard-delete documents tombstoned > 7 days) in slot 0004 with grants.
- Hand-authored realistic recorded Notion JSON fixtures (pages, every block type, nested toggles, tables, synced blocks, data source rows, pagination, 429 with Retry-After) under tests/module_c1/fixtures; respx-driven tests. Tests that PROVE: every block type in the table is normalized, anchors and deep links are correct, fetched_at uses the Date header, rate limiter is global across concurrent callers (two asyncio tasks / two connections share the budget), token never appears in logs/exceptions/reprs, confirm_deleted semantics, cross-tenant/definer tests for slot functions.`,
  },
  {
    id: 'C2', title: 'Sync service, reconcile, file ingestion, blob store',
    owns: 'packages/core/src/ragforge_core/services/sync/**, packages/core/src/ragforge_core/adapters/connectors/store.py, packages/core/src/ragforge_core/adapters/connectors/files/**, packages/core/src/ragforge_core/adapters/blob/**, packages/core/tests/module_c2/**',
    slot: 'none (no migration; if you need DDL, write it in your handoff)',
    docs: 'docs/adr/0007-notion-polling-primary-webhooks-optional.md, docs/adr/0009-embedding-spaces-and-content-addressed-embeddings.md, docs/adr/0010-job-scheduling-leases-priorities-fairness.md, docs/adr/0014-data-classification-and-backups.md, docs/DESIGN.md sections 3 (Flow A), 6, 9',
    spec: `Implement the sync job handlers that orchestrate the other modules' ports (all of which you FAKE in tests).
- services/sync: handlers (async functions taking a JobLease + injected ports) for sync_connector (open SecretBox token with associated_data b"connector:<tenant_id>:<connector_id>", list_changed from cursor, skip excluded/fresh pages using the fetched_at rule, enqueue sync_page jobs with tenant-scoped dedupe keys, advance cursor monotonically), sync_page (the exact 3-step flow from the DocumentIndex docstring: short tx touch+find missing per target space -> embed OUTSIDE any tx via EmbedderRegistry -> fenced tx: assert_lease, insert_embeddings, upsert_document (stop if stale), replace_chunks, report_progress, complete; ids via domain.ids; audit row), reconcile_deletes (re-enqueue non-fresh pages, synced-source staleness, confirm_deleted before tombstoning), delete_connector (destroy secret, delete documents, then connector), ingest_file (upload -> parse in sandbox -> same indexing flow, source_kind file). Permanent vs retryable failures via JobQueue.fail(permanent=...).
- Module-local persistence Protocol in services/sync/ports.py (cursors, connector rows incl. sealed secret, exclusions, uploads) using domain types/primitives only; implementation in adapters/connectors/store.py (SQL under the caller's tenant tx; adapters may not import services, so match the Protocol structurally).
- adapters/connectors/files: DocumentParser running parsing in a subprocess with timeout 60 s, memory limit, 50 MB size cap, page cap, MIME sniffing (not extension), markdown and plain text required; PDF/DOCX if a parser library is already installed, else document the dependency in your handoff.
- adapters/blob/local.py: BlobStore on a directory (settings.upload_dir), atomic writes (tmp + rename), path-traversal-safe keys.
- Tests that PROVE (with fakes for other modules' ports, plus real Postgres for the store adapter): no model call happens while a tx is open (fake embedder asserts no open tx), stale outcome skips chunk replacement, exclusions respected, cursor never moves backwards, reconcile never tombstones without confirmation, lease loss mid-sync commits nothing, parser limits enforced (oversize, timeout, zip-bomb-like input), blob path traversal rejected, tenant isolation of the store adapter.`,
  },
  {
    id: 'D1', title: 'LangGraph agent: graph, verification, AgentRunner',
    owns: 'packages/core/src/ragforge_core/agent/** EXCEPT agent/sanitize.py, packages/core/src/ragforge_core/prompts/**, packages/core/tests/module_d1/**',
    slot: 'none (no migration)',
    docs: 'docs/adr/0011-streaming-verification-and-output-safety.md, docs/adr/0006-onnx-reranker-and-score-based-grading.md, docs/DESIGN.md sections 2 and 8',
    spec: `Implement AgentRunner with LangGraph 1.x (installed), against ports only (ChatModel, Retriever, EmbedderRegistry, ConversationRuns, TransactionManager, JobQueue) and an injected LangGraph checkpointer (tests use the in-memory saver; module D2 builds the fenced Postgres saver).
- Graph per DESIGN section 8: load_memory -> contextualize (follow-ups only) -> retrieve -> grade (top rerank score >= collection tau, default from config constant) -> rewrite once -> abstain ("not in your notes" + closest pages) or generate (streams; tools bound) -> destructive tools batched into ONE approval node (idempotent pending_approvals insert keyed by tool_call_id via ConversationRuns/ports, then a single interrupt) -> execute approved tools idempotently -> verify_citations (deterministic: every cited anchor in hits, strict mode one per sentence) -> ground_check (ChatModel.complete_json with a JSON schema, one repair retry, malformed -> partial) -> decide (verified/partial -> done; unsupported -> one strict regeneration with TurnReplace; second failure -> abstain). State uses messages (committed only at turn end) and turn_messages (scratch for tool exchanges).
- Runner: run_turn/resume yield domain TurnEvents in the contracted order: TurnSources after every retrieval and BEFORE any text that may cite it; RAW text (never sanitize: module E does); TurnVerification with unsupported_claims text; TurnDone with usage and prompt_version. Acquire/heartbeat/release the conversation run lease via ConversationRuns (release on interrupt); cancellation on consumer aclose at node boundaries; new message while approvals pending -> expire them.
- Tools: search_notes, get_page, list_sources, sync_status, remember_fact(user_quote) with the deterministic quote gate (>= 20 chars, verbatim in the latest user message; stored fact IS the quote), forget_fact (destructive), remove_from_index (destructive). tenant/user always from the Principal, never from model args.
- prompts/: versioned prompt templates with a version header recorded in TurnDone.
- Tests with a scripted fake ChatModel that PROVE: event ordering (sources before text), abstain path, rewrite loop bound, unsupported -> regenerate -> replace, ground-check malformed -> partial, destructive tools always interrupt and resume executes exactly once (idempotent on node re-run), injected source text cannot trigger tools or memory writes, quote gate, tenant never taken from model output, usage and prompt_version reported.`,
  },
  {
    id: 'D2', title: 'Fenced checkpoint saver, conversation runs, Ollama chat adapter',
    owns: 'packages/core/src/ragforge_core/adapters/db/checkpoint.py, packages/core/src/ragforge_core/adapters/db/conversations.py, packages/core/src/ragforge_core/adapters/llm/chat_*.py, packages/core/src/ragforge_core/db/migrations/versions/0005_agent.py, packages/core/tests/module_d2/**',
    slot: '0005',
    docs: 'docs/adr/0008-psycopg3-single-driver.md, docs/adr/0013-db-roles-and-tenancy-model.md, docs/adr/0004-host-first-ollama.md, docs/DESIGN.md section 8 (Run lifecycle)',
    spec: `Implement ConversationRuns, the fenced LangGraph Postgres saver, and the Ollama ChatModel.
- Slot 0005: definer functions assert_run_lease(conversation_id, run_id) (row lock FOR UPDATE, raise unless active_run_id matches and run_expires_at > now), heartbeat_run, chat_active() (any live run lease; worker uses it to throttle embedding); create the LangGraph checkpoint + store tables in schema rf owned by rf_system using the installed langgraph-checkpoint-postgres package's own migration SQL (so the library's setup() is unnecessary at runtime), classify them (class comments) and grant rf_api what the saver needs. If the library requires its own setup at runtime, document the exact plan in your handoff.
- adapters/db/conversations.py: ConversationRuns exactly as the port documents (CAS acquire raising TurnInProgress, heartbeat/release in own short tenant tx, lock_pending FOR UPDATE, expire_pending).
- adapters/db/checkpoint.py: FencedPostgresSaver subclassing the library's async Postgres saver; aput/aput_writes run in ONE transaction: assert_run_lease -> write -> advance conversations.head_checkpoint_id; reads/resume use head_checkpoint_id. Pin to the installed version in a comment; contract tests (round trip, interrupt/resume, writes ordering) + a stalled-pod test (lease reclaimed by another run -> stale writer's checkpoint is rejected and invisible).
- adapters/llm/chat_ollama.py: ChatModel via Ollama /api/chat streaming (httpx): tools, JSON-schema format for complete_json, think=false where supported, fixed num_ctx, keep_alive, tool-call ids minted with uuid4 (never provider ids), exactly one UsageReport last, aclose cancels the upstream request. respx tests including cancellation and malformed streams.
- Tests that PROVE: only one concurrent turn per conversation, lease expiry allows takeover, fencing rejects stale writes, tenant isolation of the definer functions and conversations adapter, tool ids unique across turns, catalog invariants pass.`,
  },
  {
    id: 'E1', title: 'Streaming output sanitizer + GFM fuzz oracle',
    owns: 'packages/core/src/ragforge_core/agent/sanitize.py, packages/core/tests/module_e1/**',
    slot: 'none (no migration)',
    docs: 'docs/adr/0011-streaming-verification-and-output-safety.md',
    spec: `Implement the OutputSanitizer port exactly as ADR-0011 specifies (incremental state machine; allow(urls); feed(text) -> safe text now; finish()).
- States: text; fenced code ONLY for unambiguous top-level CommonMark fences (column 0, no open list/blockquote container, >= 3 backticks or tildes, no backtick in a backtick fence info string; close only with same char, run >= opener, <= 3 spaces indent, no info string) passed through unheld; inline code held to closing backtick or end of line; constructs (!, [, <, line-start [) held until resolved with 512-char overflow flushed as inert markdown-escaped text; bare URLs held to a delimiter then emitted as the canonical stored allow-listed URL or defanged inside an inline code span whose delimiter is longer than any backtick run inside; remote images dropped -> [image removed]; links kept only on exact normalized match (scheme, lower-cased host, path, credentials stripped; fragments allowed) emitting the stored canonical URL; reference definitions removed and reference-style links/images become plain text; raw HTML escaped; javascript: and data: dropped; autolinks follow the link rule.
- Tests: grammar-based generator (hypothesis) of hostile markdown (container nesting + fences, fence length/char mismatches, info strings with backticks, padded overflow, reference/inline/autolink/bare URLs, HTML, nested constructs) split at random token boundaries; oracle = render the CONCATENATED emitted output with markdown-it-py (installed; enable linkify if linkify-it-py is installed, otherwise implement a conservative bare-URL check and note it) and assert the AST contains no image with a remote src and no link whose href is not in the allow-list. Also: a 5000-char fenced code block round-trips byte-identical; a known corpus of exfiltration payloads; idempotence is NOT required but double-feeding must never produce a live link.`,
  },
  {
    id: 'E2', title: 'API routes, auth, provisioning, SDK',
    owns: 'packages/core/src/ragforge_core/apps/api/** EXCEPT apps/api/main.py, packages/core/src/ragforge_core/services/auth/**, packages/client/**, packages/core/src/ragforge_core/db/migrations/versions/0006_api.py, packages/core/tests/module_e2/**',
    slot: '0006',
    docs: 'docs/adr/0005-mcp-as-thin-api-client.md, docs/adr/0011-streaming-verification-and-output-safety.md, docs/adr/0013-db-roles-and-tenancy-model.md, docs/DESIGN.md sections 8 (SSE contract, approvals) and 9',
    spec: `Implement the /v1 HTTP API and the typed SDK, using ports (JobQueue, Retriever, EmbedderRegistry, AgentRunner, OutputSanitizer, SecretBox, TransactionManager) injected through FastAPI dependencies resolved from app state (Phase 2 wires real adapters in bootstrap.py; your tests override with fakes).
- Slot 0006: authenticate_api_key(prefix, sha256) definer (constant-time compare, revoked keys rejected, updates last_used_at) and provision_tenant(name, kind, owner_display_name, consent) definer (creates tenant + owner user; team requires consent). Grants accordingly.
- services/auth: API key format rf_<prefix>_<secret> (secret from secrets.token_urlsafe), sha256 storage, Principal resolution, scope checks (approvals:write never on MCP keys), an admin CLI module (main function) for creating the first tenant + key; request the console-script entry in your handoff.
- Routes per DESIGN section 9 with contracts models: tenants (admin), api keys, collections, connectors (seal token with SecretBox, associated_data b"connector:<tenant_id>:<connector_id>"; never echo), documents (list; delete -> exclusion + tombstone), exclusions, jobs (GET from jobs+jobs_history under RLS; SSE events via LISTEN rf_job_progress + poll fallback, re-read under RLS; payload ids only), query (embed outside tx via EmbedderRegistry then short tx Retriever), chat SSE (map TurnEvents -> contracts ChatEvents; one OutputSanitizer per answer; sanitizer.allow(urls) on every TurnSources before later text; never emit raw text; set conversation_id on interrupt/error/done), approvals resume (scope approvals:write), audit. Cursor pagination with Page. Problem+json errors, 404 (not 403) for other tenants' ids.
- Client SDK (packages/client): typed async methods for every endpoint, SSE parsing that skips unknown events, ApiError mapping.
- Tests that PROVE: every endpoint returns 404 for another tenant's ids, scopes enforced, approvals impossible with MCP-scoped keys, token never appears in any response/log, SSE mapping order + sanitizer applied to every text event (fake sanitizer that marks text), new conversation id on interrupt/error, auth timing-safe and revoked keys rejected, provisioning consent rule, SDK round trips against the app via ASGITransport.`,
  },
]

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests_passing: { type: 'boolean' },
    test_command: { type: 'string' },
    checks: { type: 'string', description: 'results of ruff/mypy/lint-imports/catalog tests' },
    handoff: { type: 'string', description: 'path of handoff file and its key requests' },
    known_gaps: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files', 'tests_passing', 'test_command', 'checks', 'handoff', 'known_gaps'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    score: { type: 'number' },
    verdict: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          title: { type: 'string' },
          location: { type: 'string' },
          why: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'title', 'location', 'why', 'fix'],
      },
    },
    verified: { type: 'array', items: { type: 'string' } },
  },
  required: ['score', 'verdict', 'findings', 'verified'],
}

function moduleBlock(m) {
  return `
MODULE ${m.id}: ${m.title}
Paths you OWN (create/edit only these): ${m.owns}
Migration slot: ${m.slot}
Design docs to read: ${m.docs}
Handoff file: ${ROOT}/docs/handoff/${m.id}.md

WHAT TO BUILD:
${m.spec}`
}

function reviewPrompt(m, round) {
  return `You are a principal engineer with 25 years at a large tech company (Postgres, distributed systems, security, LLM platforms). Review ONE module of a project BLIND (round ${round}). Be adversarial but calibrated: 9+ = approve with minor comments; 8 = approve after short fixes; 7 = needs another revision.
Repository: ${ROOT} (NOT a git repo: never run git). Do NOT modify any file. Do NOT run make up/nuke or docker compose down.
Other modules are being built concurrently in the same folder; judge ONLY module ${m.id}. Failures caused by other modules' in-progress files are not this module's defects.

Read docs/PHASE1.md (rules + conventions), the frozen ports (packages/core/src/ragforge_core/domain/ports.py, models.py, ids.py), and the design docs: ${m.docs}. Then read ALL of module ${m.id}'s files: ${m.owns}, and its handoff file docs/handoff/${m.id}.md.

Module ${m.id} was supposed to build:
${m.spec}

You may run read-only verification: the module's tests (e.g. uv run pytest packages/core/tests/module_${m.id.toLowerCase()} -q, with RAGFORGE_TEST_TARGET_REVISION set to its slot if it has one), uv run mypy on its source files, uv run ruff check on its paths, uv run lint-imports, uv run pytest packages/core/tests/test_catalog_invariants.py -q.

Check: correctness under concurrency, security (tenant isolation, secret leakage, injection), conformance to the frozen ports and PHASE1 conventions (no model call inside a transaction, ids from domain.ids, sha bytes, definer function rules), whether tests actually PROVE the required guarantees, ownership rule violations (edits outside owned paths), and code quality. Return P0/P1 only for real defects; style nits are P2.`
}

function fixPrompt(m, review, round) {
  return `${COMMON}
${moduleBlock(m)}

You are continuing module ${m.id}, already implemented in the owned paths above. A blind principal-engineer review (round ${round}) scored it ${review.score}/10 and reported these findings. Fix EVERY P0 and P1 (and P2s that are cheap and inside your paths). For each finding you believe is wrong, do not change code; explain why in your summary. Stay strictly inside your owned paths; anything else goes into your handoff file. Re-run the definition-of-done checks.

FINDINGS:
${JSON.stringify(review.findings, null, 2)}`
}

const blocking = r => !r || r.findings.some(f => f.severity === 'P0' || f.severity === 'P1')

const results = await pipeline(
  MODULES,
  async m => {
    const build = await agent(`${COMMON}\n${moduleBlock(m)}`, { label: `build:${m.id}`, phase: 'Build', schema: BUILD_SCHEMA })
    log(`${m.id} built (tests passing: ${build ? build.tests_passing : 'unknown'})`)
    return build
  },
  async (build, m) => {
    const history = []
    let review = await agent(reviewPrompt(m, 1), { label: `review:${m.id}#1`, phase: 'Review', schema: REVIEW_SCHEMA })
    history.push(review)
    log(`${m.id} review 1: ${review ? review.score : '?'}/10`)
    let round = 1
    while (blocking(review) && review && round <= 2) {
      const fix = await agent(fixPrompt(m, review, round), { label: `fix:${m.id}#${round}`, phase: 'Fix', schema: BUILD_SCHEMA })
      round += 1
      review = await agent(reviewPrompt(m, round), { label: `review:${m.id}#${round}`, phase: 'Review', schema: REVIEW_SCHEMA })
      history.push(review)
      log(`${m.id} review ${round}: ${review ? review.score : '?'}/10`)
      if (fix) build = fix
    }
    return {
      module: m.id,
      title: m.title,
      final_score: review ? review.score : null,
      open_blockers: review ? review.findings.filter(f => f.severity !== 'P2') : [],
      open_p2: review ? review.findings.filter(f => f.severity === 'P2').length : null,
      scores: history.map(h => (h ? h.score : null)),
      build,
    }
  },
)

return results.filter(Boolean)
