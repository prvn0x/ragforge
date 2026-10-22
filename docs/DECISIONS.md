# Design decisions, review history and what is not done

This page is the index of the design. It lists every decision record (ADR) with its status in the code, the score of every blind review, and the work that is parked or still open. It is the one place that says what is **not** done. Every row links to the file, test or make target that backs it.

Sources: [adr/](adr/), [REVIEW.md](REVIEW.md), [NEXT.md](NEXT.md), [handoff/](handoff/), [DESIGN.md](DESIGN.md), the [Makefile](../Makefile) and the code under [packages/](../packages/).

## 1. Decision index

How to read the status column:
- **Implemented:** the code for every decision in the ADR's core scope exists, and the note points at it.
- **Partial:** some of the ADR's core decisions exist and some do not; the note names the missing ones.
- Items the ADR itself tags **[A]** (Advanced) or **[S]** (Stretch) do not count toward the status. When one is not built, the note says so.

Status was checked against the code on 2026-10-22, not taken from the ADRs.

| ADR | Decision | Main alternative rejected | Why | Status |
|---|---|---|---|---|
| [0001](adr/0001-uv-workspace-monorepo.md) | One uv workspace with 5 packages: contracts, core (api + worker), client, mcp, ui | A single package; 7 packages | One package lets the UI import the DB layer and ships every dependency in every image; 7 packages add ceremony with no dependency benefit | **Implemented.** Six import-linter contracts in [pyproject.toml](../pyproject.toml), run by `make imports` |
| [0002](adr/0002-postgres-as-single-datastore.md) | Postgres 17 + pgvector is the only datastore; monotonic `schema_meta` integers decide readiness | Postgres + Qdrant or Elasticsearch; Celery/Redis or SQS for jobs | Dual writes and two consistency domains; an enqueue outside the DB is not atomic with the write | **Implemented.** [db/readiness.py](../packages/core/src/ragforge_core/db/readiness.py), [test_readiness.py](../packages/core/tests/test_readiness.py). The Helm rollback part has no target yet (Helm is parked, section 4) |
| [0003](adr/0003-no-outbox-or-broker-in-v1.md) | No outbox and no broker: audit rows in the same transaction, job progress via `NOTIFY` to the API | Outbox + Redpanda | None of the three proposed consumers needed it | **Implemented.** [apps/api/jobfeed.py](../packages/core/src/ragforge_core/apps/api/jobfeed.py) (LISTEN plus a poll fallback), `pg_notify` in [0002_data_plane.py](../packages/core/src/ragforge_core/db/migrations/versions/0002_data_plane.py) |
| [0004](adr/0004-host-first-ollama.md) | Host Ollama on macOS (Metal), the container profile elsewhere; fixed `num_ctx` and `keep_alive` | Ollama in Docker on macOS | Docker on macOS gets no Metal GPU, so a container model is CPU-only | **Partial.** Built: runtime choice in [scripts/bootstrap.sh](../scripts/bootstrap.sh), request settings in [config.py](../packages/core/src/ragforge_core/config.py), the `OLLAMA_MAX_LOADED_MODELS` warning in [scripts/doctor.sh](../scripts/doctor.sh). Not built: the `compose.gpu.yml` override, `make eval-models`, the `openai`/`bedrock` adapters. A Gemini adapter was added instead ([handoff/P2B-G1.md](handoff/P2B-G1.md)); no ADR covers it (section 5) |
| [0005](adr/0005-mcp-as-thin-api-client.md) | The MCP server is a thin client of the public API | MCP calls the service layer directly | A second entry point for auth, tenancy and rate limits, plus DB dependencies in the MCP image | **Implemented.** [packages/mcp/pyproject.toml](../packages/mcp/pyproject.toml) depends on the client, not core; stdio and HTTP in [server.py](../packages/mcp/src/ragforge_mcp/server.py) |
| [0006](adr/0006-onnx-reranker-and-score-based-grading.md) | FlashRank ONNX reranker (top 20, `max_length=512`); grade by reranker score, never by an LLM | sentence-transformers + bge reranker, plus an LLM grader | PyTorch adds about 2 GB; an LLM grader costs seconds per turn | **Partial.** Reranker in [adapters/rerank/flashrank.py](../packages/core/src/ragforge_core/adapters/rerank/flashrank.py). τ is not tuned: the default 0.35 is marked provisional in [agent/settings.py](../packages/core/src/ragforge_core/agent/settings.py), and no command writes `collections.tau` |
| [0007](adr/0007-notion-polling-primary-webhooks-optional.md) | Notion: pinned API version, polling first, nightly reconcile with confirmed deletes, exclusions, a shared rate limit | Webhooks as the primary path | Webhooks need a public HTTPS endpoint and carry no content | **Implemented.** `NOTION_VERSION` in [notion/client.py](../packages/core/src/ragforge_core/adapters/connectors/notion/client.py), `take_tokens` in [0004_connectors.py](../packages/core/src/ragforge_core/db/migrations/versions/0004_connectors.py), `/v1/exclusions` in [routes/content.py](../packages/core/src/ragforge_core/apps/api/routes/content.py). [A] webhooks: the routing function exists, but there is no HTTP route ([handoff/PHASE2A.md](handoff/PHASE2A.md), "Known gaps") |
| [0008](adr/0008-psycopg3-single-driver.md) | psycopg 3 as the only driver, two pools, a fenced LangGraph saver | asyncpg for SQLAlchemy beside psycopg for LangGraph | Two drivers mean two sets of type adapters and pools | **Implemented.** Pools in [bootstrap.py](../packages/core/src/ragforge_core/bootstrap.py), `FencedPostgresSaver` in [adapters/db/checkpoint.py](../packages/core/src/ragforge_core/adapters/db/checkpoint.py), tests in [test_d2_checkpoint.py](../packages/core/tests/module_d2/test_d2_checkpoint.py); `langgraph-checkpoint-postgres` pinned in [packages/core/pyproject.toml](../packages/core/pyproject.toml) |
| [0009](adr/0009-embedding-spaces-and-content-addressed-embeddings.md) | Embedding spaces, content-addressed embeddings, definer search functions, GC with a 24 h grace, dual write to a building space | A fixed vector column | It ties the schema to one model; a model change means downtime and a full re-embed | **Partial.** Built: [0003_retrieval.py](../packages/core/src/ragforge_core/db/migrations/versions/0003_retrieval.py), dual write in [adapters/db/index.py](../packages/core/src/ragforge_core/adapters/db/index.py), plan tests in [test_b_db_search.py](../packages/core/tests/module_b/test_b_db_search.py). Not built: the orphan repair job; the orphan metric counts a narrower case ([handoff/OBS-METRICS.md](handoff/OBS-METRICS.md), item 1). [A] flip automation and the per-collection release step: not built ([handoff/B.md](handoff/B.md), item 8) |
| [0010](adr/0010-job-scheduling-leases-priorities-fairness.md) | Strict class order with an every-4th-claim starvation guard, leases, fencing inside the result transaction, tenant-scoped dedupe | Cross-class aging | Under a long backfill it inverted priority and broke freshness (review round 5) | **Implemented.** [services/jobs/fairness.py](../packages/core/src/ragforge_core/services/jobs/fairness.py), [0002_data_plane.py](../packages/core/src/ragforge_core/db/migrations/versions/0002_data_plane.py), tests in [module_a](../packages/core/tests/module_a/). [A] per-tenant fairness (`tenant_queue_state`): not built |
| [0011](adr/0011-streaming-verification-and-output-safety.md) | An incremental sanitizer before every token event, then a citation check and a ground check | Sanitizing only at the end | The user would read an exfiltration link while it streams | **Implemented.** [agent/sanitize.py](../packages/core/src/ragforge_core/agent/sanitize.py), fuzz and round-trip tests in [module_e1](../packages/core/tests/module_e1/). Rule 4 is amended by ADR-0015 |
| [0012](adr/0012-evaluation-strategy.md) | Recorded-Notion corpus, a significance-based CI gate with a frozen floor, a measured verifier | A non-inferiority CI gate at n = 100; a point-estimate rule | The first fails neutral changes; the second blocks about 20–26% of neutral PRs | **Partial; largely replaced by ADR-0017.** Not built: the recorded-Notion corpus, the CI gate (not built), `--calibrate`, `make label-verdicts`. The verifier set is the 20-item demo set ([evals/demo/verifier.json](../evals/demo/verifier.json)), not about 300 items. Floors are not loaded yet (ADR-0017 P2) |
| [0013](adr/0013-db-roles-and-tenancy-model.md) | Tables owned by a NOLOGIN `rf_system`; RLS enabled, not forced; no BYPASSRLS; a closed list of definer functions | FORCE RLS plus a BYPASSRLS owner | Needs a superuser and is restricted on managed Postgres | **Implemented.** Roles in [db/bootstrap.sql](../packages/core/src/ragforge_core/db/bootstrap.sql); catalog rules in [test_catalog_invariants.py](../packages/core/tests/test_catalog_invariants.py); isolation in [test_e2_api_isolation.py](../packages/core/tests/module_e2/test_e2_api_isolation.py). [A] team tenants: not built. [S] `document_acl`: not built |
| [0014](adr/0014-data-classification-and-backups.md) | Back up authoritative data only; rebuild derived data | No alternative is recorded in the ADR | The ADR's Derived row says derived data can be rebuilt from Notion and the uploads, so only data that cannot be rebuilt needs a backup. Its RTO line sets < 1 h for authoritative data and lets derived data rebuild at Notion's rate | **Partial.** Built: the table classes, checked by `test_every_table_is_classified` in [test_db_foundation.py](../packages/core/tests/test_db_foundation.py); the read-only `rf_backup` role. **Not built:** the nightly `pg_dump`, `make reindex` (so the rebuild path itself does not exist yet), upload snapshots, and retention jobs for `audit_log` and `usage_events` (only `jobs_history` has one). [A] restore drill: not built. `make backup-evals` copies the local eval data instead (ADR-0018) |
| [0015](adr/0015-retrieve-then-answer.md) | Retrieve, then answer with no tools bound; tools only for explicit actions; `verified` means every unit is covered by a citation or an allowlist | Tools on every turn; an LLM intent classifier | The small model misused tools and slowed every turn; a classifier adds a model call per turn | **Implemented.** [agent/coverage.py](../packages/core/src/ragforge_core/agent/coverage.py), [agent/attribution.py](../packages/core/src/ragforge_core/agent/attribution.py), tests in [module_ag](../packages/core/tests/module_ag/). Its "Follow-up measurement" has not run, so `MIN_MATCHED` / `MIN_COVERAGE` are not tuned on data. Identifier exactness is a planned follow-up, not a guarantee |
| [0016](adr/0016-labelled-extras.md) | Interview answer shape; general-knowledge extras as a separate, labelled, unverified event | Mixing general knowledge into the answer | It breaks the guarantee, and the reader cannot tell notes from model | **Implemented.** `agent_extras` in [config.py](../packages/core/src/ragforge_core/config.py), [test_ag_extras.py](../packages/core/tests/module_ag/test_ag_extras.py). The lease is still held while the extra is made; how often extras invent an incident is not measured |
| [0017](adr/0017-eval-layer.md) | An in-house eval layer: retriever, generator, verifier, interview and application evals, a calibrated judge, a gate, an Eval page | RAGAS or DeepEval as the core | They do not measure this project's own `verified` guarantee, do not calibrate the judge, and lack the small-n statistics | **Implemented,** with the open P2s in section 5. [ragforge_client/evals](../packages/client/src/ragforge_client/evals/), `make eval`, `make eval-gate`, `make eval-demo`. Deferred CLI commands `judge-accept`, `overlap`, `calib-sample`: not built |
| [0018](adr/0018-ops-hardening-security-observability.md) | A thin Phase 4: a security pass and a metrics dashboard with alerts | The full Phase 4 (Helm, kind, KEDA, load test, restore drill) | Memory on a 16 GB laptop; those pieces matter mainly for platform roles | **Implemented** (the thin scope). [observability/](../packages/core/src/ragforge_core/observability/), [docker/observability/](../docker/observability/), [scripts/security/](../scripts/security/), [SECURITY.md](SECURITY.md); `make obs-up`, `make audit`, `make secrets-scan`. One open P2 (section 5) |

## 2. Review history

Every score below comes from a blind reviewer who did not read [REVIEW.md](REVIEW.md), except the self-scores marked "self" or "self-claimed". Scores are quoted as recorded. "Not recorded" means no such review took place or no score was written down.

The last column lists findings the reviews caught that the tests at the time did not.

| Phase | Blind design review | Builder reviews | Final integrated review | Caught by review, not by tests |
|---|---|---|---|---|
| **Phase 0, design** ([REVIEW.md](REVIEW.md), rounds 1–7) | Rounds 1–2: 6.2 self, 7.4 self (9.4 self-claimed for v0.3). Blind: 5.5 → 6.3 → 7.5 → 7.8 → 8.0 | — | 8.0 (v0.7), then stopped: the curve had flattened | FORCE RLS made every owner-owned definer function see 0 rows; streamed tokens got past an end-only sanitizer; a zombie job could commit stale data because fencing covered only `complete_job` |
| **Phase 0, code** ([REVIEW.md](REVIEW.md)) | — | — | 7.0 → 7.5 → 8.0 → **8.5, approved** | Per-schema default-privilege REVOKE was a no-op (functions were PUBLIC-executable); embedding inside a locked transaction; plaintext role passwords in DDL |
| **Phase 1** ([REVIEW.md](REVIEW.md), [workflow](workflows/phase1-parallel-build.js)) | Not recorded | A 8.5 · B 8 → 8.5 → 9 · C1 8 → 8.5 → 9 · C2 9 · D1 8 → 8 → 8 → 8.5 · D2 8.5 → 9 · E1 5.5 → 8 → 8 → 8 · E2 8 → 9 | Integrated in Phase 2a | Two runs touching around a dropped construct rendered as a live bare URL ([handoff/E1.md](handoff/E1.md), round-1 P0); IDN e-mail domains were visible only to the linkifier (round-2 P1) |
| **Phase 2a** ([handoff/PHASE2A.md](handoff/PHASE2A.md)) | Not recorded | — | Two reviewers per round: 6/5 → 7/7 → 7.5/7.5 → 7.5/7.5 → 7.5/8 → 8/7.5 → **8/8** | Undo-delete within 60 s did nothing; undo revived pages deleted at the source; a fix for the pre-0008 scheduler function stopped all polling after a rollback |
| **Phase 2b** ([REVIEW.md](REVIEW.md), [NEXT.md](NEXT.md)) | Not recorded | — | 7.5, 7.5, 8.0, 8.0, 8.0, 8.0, 7.5, 8.0, 7.5, then **8.5** | A prose-quote rule joined the text around dropped atoms (round 2); the `make ask` footer printed a block id while the test hid it with fake anchors (round 3); the residue dropped a joined-code unit with a clean twin (round 6) |
| **Phase 2c, ADR-0016** ([REVIEW.md](REVIEW.md)) | Not recorded | — | 7.0, 7.0, 8.0, 7.5, 7.0, then **8.0** | A refusal got an extra (r1); the extras block could be spoofed in `make ask` (r2); stripping emphasis deleted the underscores inside identifiers (r4); a mid-cycle identifier rule scored 5.0 with five P1s and was reverted (r5) |
| **Phase 3** ([REVIEW.md](REVIEW.md), [workflow](workflows/phase3-parallel-build.js)) | Not recorded | MCP 7.5 → 8.5 → 8.4 · UI 8.0 → 8.2 → 8.0 | 8.0 → **8.3** | A 4-space-indented fence ended an MCP code block; the UI lacked a Host check against DNS rebinding; the UI showed a pre-verdict draft as Markdown; MCP texts promised an approval the UI could not load |
| **Phase 3b** ([REVIEW.md](REVIEW.md), [workflow](workflows/phase3b-interview-build.js)) | 6.5 (4 P0s) | IV-CORE 8 → 9 · FLAG 7.5 → 9 · IV-UI 8.2 → 8.7 | 8.0 → **8.6** | The token grammar admitted `3`, `true` and plain words; a reason mapping would have muted a measured false-"verified" class; code questions were starved (about 1 in 20) |
| **Phase 3c** ([REVIEW.md](REVIEW.md), [workflow](workflows/phase3c-interview-levels.js)) | 7.0 (2 P0s) | IV3-CORE 8 → 8.9 · IV3-UI 8.1 → 9 · DOT 7.5 → 8 → 9 | 8.0 → **8.6** | A shape-only dotted-name rule merged "fast.and.it" into one sentence; the general-mode token was optional; the lexical gate rejected hyphens, "vs" and paths; a loose label regex ate "(S3)" |
| **ADR-0017 eval layer** ([adr/0017](adr/0017-eval-layer.md), [workflow](workflows/phase4a-eval-build.js)) | 6.5 → 7.8 → 8.0 → approve to build | EV-API 8 → 9 · EV-SNAP 8.7 · EV-METRICS 7.5 → 7.8 → 8.2 · EV-JUDGE 8 → 8.3 → 8 · EV-RUN 7.5 → 7.8 → 8.3 · EV-UI 7.8 → 8 → 8.5 | **7.0, "ship after fixes"**; all P0/P1 fixed; no re-review score recorded | A verifier set selected on outcome; an unbounded cross-tenant copy function; a private D3 set that could never be built (rows lacked stable keys); `make eval-demo` failing on a fresh install; a gate that accepted a candidate older than its baseline |
| **ADR-0018 Phase 4** ([adr/0018](adr/0018-ops-hardening-security-observability.md), [workflow](workflows/phase4b-ops-build.js)) | 6.5 (4 P0s) | OBS-METRICS 9 · OBS-STACK 8.5 · SEC-SCAN 9 · SEC-THREAT 8.5 | 7.5 → **8.5, ship** | The onnxruntime telemetry client started at import; uvicorn and stdlib logs bypassed redaction; the dead-jobs gauge fell to 0 once jobs were archived; latency and 5xx alerts fired on one event |

`make check` at each close, as recorded: 966 tests (Phase 1), 1134 (Phase 2a), 3422 (Phase 3), 3813 (Phase 3b), 4168 (Phase 3c), 5423 (Phase 4). Sources: [REVIEW.md](REVIEW.md), [NEXT.md](NEXT.md), [handoff/PHASE2A.md](handoff/PHASE2A.md).

## 3. How the work was done

- **Design first.** The Phase 0 design went through 7 review rounds, 5 of them blind, before any code was written (section 2, "Phase 0, design"). Phases 1, 2a, 2b, 2c and 3 were built on that reviewed design and had no design review of their own ("Not recorded" in section 2). From Phase 3b on (3b, 3c, ADR-0017, ADR-0018), each phase had its own blind design review before code, and every P0 and P1 was fixed first (the "Blind design review" column; [CLAUDE.md](../CLAUDE.md), "Non-negotiables").
- **No self-scores.** The Phase 0 self-score was 9.4; the first blind score of the same design was 5.5 ([REVIEW.md](REVIEW.md), round 3).
- **Locked seams.** An integrator locks the shared interfaces (ports, contracts, schemas, migration slots) before builders start. Builders do not change them ([PHASE1.md](PHASE1.md)).
- **Parallel builders with blind review loops.** Each builder owns its paths, writes a handoff for anything outside them, and goes through build → blind review → fix P0/P1, at most two fix rounds ([NEXT.md](NEXT.md), "How parallel phases are run").
- **Integrator fixes.** The integrator applies the handoffs, fixes the P1s left after the round cap, and runs a final blind review of the whole.
- **Verify by running.** Each phase closes with `make check` (lint, `mypy --strict`, import-linter, all tests) and a live `make up`. Running found bugs that tests and reviews missed: the first Phase 0 run found 5 ([CLAUDE.md](../CLAUDE.md)); in Phase 2a the agent never reranked, so every real turn would have abstained, while 99 tests passed on fakes ([handoff/PHASE2A.md](handoff/PHASE2A.md)).
- **Mutation checks.** From Phase 2a round 4 on, each fix gets an end-to-end test and a mutation check that shows the test fails without the fix ([handoff/PHASE2A.md](handoff/PHASE2A.md); [adr/0017](adr/0017-eval-layer.md), "Integration"; [adr/0018](adr/0018-ops-hardening-security-observability.md), "Re-review and close").

The workflow scripts that ran the parallel phases:
[phase1-parallel-build.js](workflows/phase1-parallel-build.js) ·
[phase3-parallel-build.js](workflows/phase3-parallel-build.js) ·
[phase3b-interview-build.js](workflows/phase3b-interview-build.js) ·
[phase3c-interview-levels.js](workflows/phase3c-interview-levels.js) ·
[phase4a-eval-build.js](workflows/phase4a-eval-build.js) ·
[phase4b-ops-build.js](workflows/phase4b-ops-build.js) ·
[phase5-writeup.js](workflows/phase5-writeup.js)

## 4. Parked work, and why

| Item | Why it is parked | Source |
|---|---|---|
| CI pipeline (written, never run) | [.github/workflows/ci.yml](../.github/workflows/ci.yml) exists (lint, types, tests, image build, Trivy, a compose smoke test) but the repository is not on GitHub yet, so it has never run; expect fixes on its first run. Its actions are pinned by tag, not by commit SHA. Locally, `make check` runs the same checks | [adr/0018](adr/0018-ops-hardening-security-observability.md), scope table |
| Helm, kind, KEDA, NetworkPolicies, multi-arch images | Memory on a 16 GB laptop; they matter mainly for platform roles. Probes, the migration job and stateless services already exist | [adr/0018](adr/0018-ops-hardening-security-observability.md); DESIGN §13 |
| Full database backup and restore drill | Notes re-sync from Notion. The data that cannot be rebuilt is the local eval data, which `make backup-evals` copies. There is no database backup today, and `make nuke` deletes all data volumes for good | [adr/0018](adr/0018-ops-hardening-security-observability.md); [Makefile](../Makefile) |
| Load test | One user. The write-up uses the measured latency pass instead | [adr/0018](adr/0018-ops-hardening-security-observability.md) |
| DeepEval comparison | Optional, about half a day: DeepEval as a second, report-only judge next to the calibrated one | [adr/0017](adr/0017-eval-layer.md), "Why not DeepEval or RAGAS" |
| Notion token and owner key rotation | An **accepted risk**, by owner decision. The rotation runbook is written down and can be run at any time | [adr/0018](adr/0018-ops-hardening-security-observability.md), R2-0; [SECURITY.md](SECURITY.md) |
| Blog post | Optional and later, by owner decision | [PHASE5.md](PHASE5.md), "Owner decisions" |

## 5. Known open issues

**Measured limits**
- **The hosted latency SLO is not met.** DESIGN §2 sets a hosted p95 below 5 s. The measured hosted p95 on the demo set is about 6.6 s, so the hosted latency alert fires at the 8 s bucket edge ([adr/0018](adr/0018-ops-hardening-security-observability.md), P1-4).
- **The new held-out set is not sealed.** The old held-out split was used twice. Its replacement is a draft waiting for the owner's rewording ([NEXT.md](NEXT.md), "Current state"; [adr/0017](adr/0017-eval-layer.md)).
- **The verifier baseline and judge calibration on the private set wait for owner labels.** Judge calibration needs at least 50 labels per criterion before a judge score is shown as trusted ([NEXT.md](NEXT.md), "Current state"; [adr/0017](adr/0017-eval-layer.md), "Why not DeepEval or RAGAS").
- **One false "verified" class is known.** The ground check passed an identifier with a one-character typo twice in dev runs. The fix is designed but not built ([adr/0015](adr/0015-retrieve-then-answer.md), "Current state").

**ADR-0017 open P2s** ([adr/0017](adr/0017-eval-layer.md), "Final integrated review")
- Gate floors (`frozen.json`) are not loaded yet.
- A/A results are not persisted per surface for the MDE.
- The MDE is not computed at the Holm-adjusted alpha.
- The CLI label sample drifts: it re-samples excluding labelled items.
- The CLI `label` command does not check the dataset digest.
- `label run1 run2` does not pool runs toward the 50-label floor.
- No dataset-level reveal of the new held-out set, and no reveal run.
- E-V inconclusive items count toward the transient share.
- The additive seams (stages, `trace=shown`, the tokenless grade) do not check the tenant kind.
- Calibration can drift within one label batch.
- Completeness is labelled even when an item has no facts.
- Deferred: a public normalisation API in `citations.py`; extracting `rag.chunker.embed_input`; moving `SqlEvalSnapshotStore` to adapters; the CLI commands `judge-accept`, `overlap` and `calib-sample`; binding the topic in the general-mode token.
- Doc drift: migration 0012's header says the snapshot routes need the `admin` scope; they take a key-manager key. The migration is checksum-guarded, so the ADR is the record.

**ADR-0018 open P2** ([adr/0018](adr/0018-ops-hardening-security-observability.md), "Re-review and close")
- **The MCP HTTP server's logs are not redacted.** [server.py](../packages/mcp/src/ragforge_mcp/server.py) never calls `configure_logging`. The fix is `configure_logging` at MCP start plus `log_config=None`. No secret reaches that log today: the bearer token is a header, not part of the access line.

**Other gaps found while writing this page**
- The Gemini chat provider has no ADR. ADR-0004 still names `openai|bedrock`, which do not exist ([adapters/llm/](../packages/core/src/ragforge_core/adapters/llm/)).
- PDF uploads are rejected as unsupported; Markdown, plain text and DOCX are parsed ([files/parser.py](../packages/core/src/ragforge_core/adapters/connectors/files/parser.py)).
- Earlier phases keep their own P2 lists: [handoff/PHASE2A.md](handoff/PHASE2A.md) ("Known gaps", "P2 backlog") and [NEXT.md](NEXT.md) (Phase 2b, 2c, 3, 3b and 3c sections).

## 6. DESIGN.md sections replaced by later ADRs or by the code

[DESIGN.md](DESIGN.md) is v0.8 from Phase 0. Where it and a later ADR differ, the ADR wins. Where it and the code differ with no ADR, the code, the [Makefile](../Makefile) and [compose.yaml](../compose.yaml) win; those rows say "no ADR".

| DESIGN section | What changed | Replaced by |
|---|---|---|
| §0 Scope tiers, Advanced row; §14 Milestones M7–M10 | Phase 4 became a thin security and observability pass; most Advanced items are parked | [ADR-0018](adr/0018-ops-hardening-security-observability.md) |
| §2 SLOs: hosted verified answer p95 < 5 s | Not met (about 6.6 s on the demo set); the alert edge is 8 s | [ADR-0018](adr/0018-ops-hardening-security-observability.md), P1-4 |
| §2 SLOs: retrieval quality and verifier quality | Measured on the D0 demo set, not on a recorded public corpus or a 300-item verifier set | [ADR-0017](adr/0017-eval-layer.md) |
| §2 LLM calls on the critical path | An extras call can follow the answer | [ADR-0016](adr/0016-labelled-extras.md) |
| §4 Technology choices: Chat model, Observability and Evals rows | No `make eval-models`; Prometheus + Grafana via `prometheus-client` instead of OTel and `grafana/otel-lgtm` (tracing is parked); the eval layer replaces the recorded-Notion gate | [ADR-0018](adr/0018-ops-hardening-security-observability.md), [ADR-0017](adr/0017-eval-layer.md) |
| §8 LangGraph agent: generate and verification steps | No tools on questions; `verified` is defined by coverage and an allowlist, not claim detection | [ADR-0015](adr/0015-retrieve-then-answer.md) |
| §12 Observability | The locked metric catalogue, one dashboard and Prometheus alert rules | [ADR-0018](adr/0018-ops-hardening-security-observability.md) |
| §12 Evals (corpus, CI gate, p < 0.05, τ calibration) | The ADR-0017 datasets, statistics and `make eval-gate`; p-thresholds per ADR-0017 | [ADR-0017](adr/0017-eval-layer.md) |
| §12 Operations: nightly backups | Parked; `make backup-evals` instead | [ADR-0018](adr/0018-ops-hardening-security-observability.md) |
| §11 Clone-and-run | There are no `mcp`, `ui` or `seed-demo` compose services, and the demo does not load on its own. The UI and MCP run on the host (`make ui`, `make mcp`), the demo is `make eval-demo`, and models are pulled by `make models`. Follow the [README](../README.md) quickstart and [GUIDE.md](GUIDE.md) instead | No ADR; [compose.yaml](../compose.yaml), [Makefile](../Makefile) |
| §13 Deployment [A] | Parked | [ADR-0018](adr/0018-ops-hardening-security-observability.md) |

ADR-0012's corpus section is also replaced, by ADR-0017's D0 demo set and eval snapshot ([adr/0017](adr/0017-eval-layer.md), "Consequences").
