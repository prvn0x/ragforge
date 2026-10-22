// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-eval-build',
  description: 'ADR-0017 eval layer: 6 parallel builders (API seams, snapshot/eval tenant, metrics+stats+gate, judge+datasets, runners+CLI, Eval page), each with blind principal review loops',
  phases: [
    { title: 'Build', detail: '6 builders in owned paths' },
    { title: 'Review', detail: 'blind principal review per module' },
    { title: 'Fix', detail: 'fix P0/P1, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'

const COMMON = `
You are a senior engineer building ONE module of RAGForge's eval layer (multi-tenant RAG over the owner's Notion notes, used for interview prep; a Senior/Staff portfolio project) in a shared folder where 5 other engineers build other modules AT THE SAME TIME.
Repository: ${ROOT}  (NOT a git repo: never run git commands. Do not commit.)

READ FIRST: ${ROOT}/CLAUDE.md; ${ROOT}/docs/adr/0017-eval-layer.md (ACCEPTED after 4 blind principal reviews; binding: read ALL of it incl. the review tables and the "Lock step" section); the LOCKED seams ${ROOT}/packages/client/src/ragforge_client/evals/schema.py, runfile.py, interfaces.py and __init__.py (module ownership); the contracts ${ROOT}/packages/contracts/src/ragforge_contracts/api.py (eval section at the end) and sse.py; the SDK ${ROOT}/packages/client/src/ragforge_client/client.py (eval_* methods); the D0 demo set ${ROOT}/evals/demo/ (pages, questions.toml, verifier.json, interview.json, chunks.json) and its golden fixtures ${ROOT}/evals/demo/fixtures/ (run-*.jsonl, *.summary.json, labels.jsonl, judge-results.jsonl, expected_summaries.json; reference_metrics.py is the EXECUTABLE DEFINITION of every metric name).
The locked seams (contracts, SDK, schema.py, runfile.py, interfaces.py, the D0 files, the fixtures) MUST NOT be changed. If one is truly insufficient, say so in your handoff file; the integrator decides.

HARD RULES:
1. Create/edit ONLY the paths you own (listed below). Never edit pyproject.toml, uv.lock, compose.yaml, Makefile, Dockerfile, .env, docs/** (except your handoff file), packages/client/src/ragforge_client/evalrun.py, or any locked seam. Never run uv add / uv lock / uv sync.
2. PRIVACY: NEVER open, read, list the contents of, or test against ${ROOT}/evals/private/** (the owner's real notes, questions and runs; heldout2 is sealed). Build and test ONLY against evals/demo/** and synthetic data you create in your test files. No real network, no real LLM, no live stack: fakes (respx / fake clients / the existing test fakes and test Postgres used by packages/core/tests).
3. Anything you need outside your paths (a Makefile target, a script entry, a dependency) goes in your handoff file ${ROOT}/docs/handoff/<MODULE_ID>.md. The integrator applies it.
4. Format/lint ONLY your paths: uv run ruff format <your paths>; uv run ruff check --fix <your paths>. Never on all of packages/.
5. Tests: files prefixed with your module id lowercased with dashes as underscores (e.g. test_ev_metrics_*.py) in your owned test dir; no __init__.py in test dirs. Do NOT run make up / make nuke / docker compose. Do NOT touch the real .env.
6. Layering (uv run lint-imports must pass): ragforge_client / ragforge_ui / ragforge_mcp import only ragforge_client + ragforge_contracts, NEVER ragforge_core.
7. Secrets: never print, log or put keys in exceptions, URLs or reprs. Judge keys come from key files with the existing hidden-input rules.
8. Principal-engineer quality: typed (mypy --strict clean), small focused functions, docstrings explaining WHY, framework/library primitives over hand-rolled code where they exist (numpy/scipy are NOT dependencies: pure Python unless your handoff justifies adding one), tests that PROVE behaviour (property tests with hypothesis where the ADR asks; mutation-check your key tests), clean failure handling.

DEFINITION OF DONE: your tests pass; ruff check + ruff format --check clean on your paths; mypy clean on your source files and (MYPYPATH=<your test dir>) your test files; uv run lint-imports passes; the suites listed in your module still pass. Write your handoff: what you built, exact integration steps, what you needed outside your paths, what the integrator must test live.
`

const MODULES = [
  {
    id: 'EV-API', title: 'Server eval seams: eval scope, /v1/eval verify|chunks|resolve, query stages, done.shown + prompt_versions, interview chunk_ids + eval-scope frozen grade',
    owns: 'packages/core/src/ragforge_core/apps/api/routes/eval.py (stub exists: implement), packages/core/src/ragforge_core/agent/eval_verify.py (new), packages/core/src/ragforge_core/apps/api/routes/ask.py and agent/retrieval.py and adapters/db/search.py (ADDITIVE: stages), packages/core/src/ragforge_core/apps/api/chat_stream.py + agent/runner.py + prompts/__init__.py (ADDITIVE: shown, prompt_versions per surface), apps/api/routes/interview.py + agent/interview/service.py (ADDITIVE: chunk_ids, eval-scope tokenless frozen grade), apps/api/deps.py + config.py + services/auth/** (ADDITIVE: EvalKey dependency, RAGFORGE_EVAL_API flag, eval scope never combined with an MCP key), packages/core/tests/module_eval_api/**, docs/handoff/EV-API.md',
    tests: 'uv run pytest packages/core/tests/module_eval_api packages/core/tests/module_e2 packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_d1 -q',
    spec: `Implement ADR-0017 §1 (rows: verify, chunks, resolve, query stages, chat done shown, interview chunk_ids, frozen grade, per-surface prompt versions) and the eval-scope rules (scope block + r3/r4 decisions). Every /v1/eval route requires the eval scope AND settings flag RAGFORGE_EVAL_API (off by default except dev; 404 when off). All reads RLS-confined to the caller's tenant.
- /v1/eval/verify: run the SAME pipeline as chat (normalize -> check_citations strict as configured, with dotted names -> attribute -> ground check -> settle) on a supplied answer and shown chunk ids (order = S labels; each within collection_ids; re-read via RLS), returning EvalVerifyResponse with PRE-MERGE evidence: coverage_units (kind, span), verifier_claims (span or null when unplaced), verifier_ran (False on short-circuits), verdict, conclusive, the actual verifier model and prompt@version. Put the pipeline in agent/eval_verify.py reusing agent functions read-only (do not change attribution/coverage/grounding semantics). Per-principal rate limit (reuse the interview limiter pattern).
- GET /v1/eval/chunks?ids= and POST /v1/eval/resolve: stable keys (PageKey/ChunkKey incl. occurrence) -> ids; stale returns candidates under the same (page, heading_path); never auto-remap. PageKey for files = (file, metadata.filename); notion = (notion, external_id).
- /v1/query stages=true (eval scope only, else 403): keyword, vector, fused, reranked lists with ranks, ties broken by chunk_id. Old clients unaffected.
- chat ?trace=shown (eval scope only): DoneEvent.shown (label, chunk_id, document_id, display_sha256 hex). DoneEvent.prompt_versions for every caller: {chat, verifier, interview, judge?} as name@version sets; keep prompt_version.
- interview: chunk_ids (eval scope only, within collection_ids) targets seeds; an eval-scope notes-mode GRADE needs no token and its level is free (frozen D4); general mode still needs its token; real keys keep v3 unchanged.
- Key creation refuses eval combined with an MCP-style key (read the existing key/scope rules).
Tests: scope/flag gating (403/404), cross-tenant ids invisible, verify evidence on planted D0 items (fake chat/verifier), unplaced claims null spans, short-circuit verifier_ran False, stages determinism and tie-break, shown only with scope, prompt_versions per surface, interview targeting and the frozen grade only for eval scope.`,
  },
  {
    id: 'EV-SNAP', title: 'Eval tenant + snapshot: migration 0012, guarded cross-tenant copy definer function, snapshot/reembed jobs and routes, digests, exclusion purge',
    owns: 'packages/core/src/ragforge_core/db/migrations/versions/0012_*.py (new), the CODE_REQUEST_SCHEMA constant (wherever it is defined; bump to 12) and its tests, packages/core/src/ragforge_core/services/eval_snapshot/** (new), packages/core/src/ragforge_core/apps/api/routes/eval_snapshots.py (stub exists: implement), the worker job-handler registration file (ADDITIVE), packages/core/tests/module_eval_snap/**, docs/handoff/EV-SNAP.md',
    tests: 'uv run pytest packages/core/tests/module_eval_snap packages/core/tests/test_db_foundation.py packages/core/tests/test_migrate_concurrency.py packages/core/tests/module_c2 packages/core/tests/module_worker -q',
    spec: `Implement ADR-0017 §1 snapshot row + scope block + "Review r4" P1/P2s and ADR-0013 rules.
- Migration 0012 (checksum-guarded like 0001-0011): tenants.kind allows 'eval'; tenants.eval_of_tenant_id (FK, unique, only for kind eval); a SECURITY DEFINER function (safe search_path, GRANT EXECUTE to the admin role only) copy_collection_to_eval(source_collection_id, target_tenant_id): source tenant derived from the calling session's app.tenant_id NEVER a parameter; target must be kind='eval' AND eval_of_tenant_id = source tenant; refuses otherwise. A bootstrap path (CLI / admin route, see handoff) creates the paired eval tenant + its eval key (scopes eval+chat+read).
- The copy: one synthetic uploads row per snapshot (no blob processing, no ingestion jobs), documents with source_kind/external_id preserved (notion page id) under upload ownership (not the live connector: its ON DELETE CASCADE must not reach the snapshot), chunks AND their embedding rows (no re-embed), document_links; exclusions honoured at copy time; never synced. Run it as a job (progress via the existing job feed).
- Digests: content digest (sorted page_key set + display texts) and index digest (chunk_key with occurrence, embed_sha256, space id); returned in EvalSnapshot when done.
- POST /v1/eval/snapshots (admin key of the SOURCE tenant), GET /v1/eval/snapshots/{id}, POST .../reembed {space_id}: re-embed from stored chunk text using the ingest path's own embed-input function, checking the old embed_sha256 before writing.
- A later exclusion on the source tenant purges matching snapshot pages (privacy) - implement or specify exactly in the handoff if it needs a hook outside your paths.
- The eval tenant is backed up and wiped like live notes (ADR-0014): handoff the backup/nuke integration.
Tests (test Postgres): copy into a non-eval tenant refused; into another owner's eval tenant refused; a collection the caller cannot see refused; no ingestion job enqueued; embeddings identical; digests stable across two copies of the same content; reconnecting/deleting the live connector leaves the snapshot; owner's keys never list the eval tenant's collections; exclusions honoured.`,
  },
  {
    id: 'EV-METRICS', title: 'Pure metrics, statistics and the regression gate (match the golden oracle exactly)',
    owns: 'packages/client/src/ragforge_client/evals/metrics.py, stats.py, gate.py (new), packages/core/tests/module_ev2/test_ev_metrics_*.py, docs/handoff/EV-METRICS.md',
    tests: 'uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_eval_schema.py packages/core/tests/test_client_evalrun.py -q',
    spec: `Implement ADR-0017 §3 metrics, §4 calibration math, §5 gate, as PURE functions over the locked schemas (no I/O except reading via runfile in thin helpers).
- metrics.summarize(run: RunFile, datasets: Datasets, labels: Sequence[Label], judge_results: Sequence[JudgeResult]) -> RunSummary: every metric NAME and point value exactly as evals/demo/fixtures/reference_metrics.py defines it (golden test: on every fixture run, each value in expected_summaries.json reproduced to 1e-9), PLUS the intervals: Wilson for simple proportions, clustered bootstrap by source question for E-V, Clopper-Pearson for interview ranking k/n, bootstrap for kappa; method named per MetricValue; trusted flag for judge-derived metrics from calibration (kappa >= 0.6 AND CI low >= 0.4 AND no significant leniency). Stale rows excluded and counted.
- stats.py: wilson, clopper_pearson, cluster_bootstrap (seeded, deterministic), exact McNemar, paired sign-flip permutation (exact when small), Holm, Page's L with permutation p (ties averaged), linear-weighted kappa + bootstrap CI, paired sign test, MDE from A/A discordance. Hypothesis property tests (bounds, symmetry, known values from textbooks, permutation exactness at small n).
- gate.evaluate(baseline: RunFile, candidate: RunFile, aa: Sequence[RunFile], datasets) -> GateVerdict: refuse (not_evaluable) on content-digest mismatch; retrieval exact McNemar on hit@5 + sign-flip on MRR, Holm, block p<0.025 & drop>2 pts, floor-3; chat McNemar correct/verified Holm a=0.05, hard floors (no 'ours' errors; E-V exact McNemar on planted items + absolute SLO on badge-flip recall lower bound >= 0.75 and verifier-caused false-reject upper bound <= 0.20); interview + latency WARN-ONLY; every check prints n, method, MDE; override recorded. A/A acceptance: the gate on an A/A pair must pass.
Tests: golden oracle match, stats properties, gate scenarios built from fixture runs (identical -> pass; synthetic drop -> block; digest mismatch -> not_evaluable; transient errors excluded).`,
  },
  {
    id: 'EV-JUDGE', title: 'Judge port (local Ollama primary, Gemini fallback), cache, calibration inputs; datasets loading, D3 generator for private runs, held-out tooling, label store',
    owns: 'packages/client/src/ragforge_client/evals/judge.py, datasets.py (new), packages/client/src/ragforge_client/evals/prompts/** (new: judge prompt files), packages/core/tests/module_ev2/test_ev_judge_*.py, docs/handoff/EV-JUDGE.md',
    tests: 'uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_eval_schema.py -q',
    spec: `Implement ADR-0017 §4 (judge) and §2 (datasets) against interfaces.py.
- judge.py: the Judge protocol implementations: OllamaJudge (HTTP to the host Ollama via httpx; format = the JSON schema; fixed num_ctx; short keep_alive; one item per call; temperature 0) and GeminiJudge (fallback; batching DEFERRED per r2 P2: one item per call); make_judge(config) with explicit opt-in for any provider other than local; the provider-RETURNED model version in JudgeKey; prompts judge_qa@1 and judge_question@1 as files with 0/1/2 anchors per criterion ("length is not quality"), blind (no arm/model/verdict), every item fenced with a per-item random nonce delimiter the reply must echo (else invalid -> score None). JudgeCache: judge-cache.jsonl keyed by digest({judge key, item digest}); a changed JudgeKey invalidates. Judge calls log one structured line (provider, model version, prompt, latency, item digest, outcome; no text). Acceptance helper for D0: per-item latency, valid-JSON rate.
- datasets.py: load_questions (TOML -> QuestionSet, v2 schema; private file form too), load_verifier_set, load_interview_set, Datasets bundle; LabelStore implementing LabelReader + LabelWriter (atomic append with a file lock, newest per (item_digest, kind) wins); seal/verify for heldout2 (digest in header; edits fail); the question<->page lexical overlap (Jaccard on content lemmas) flag > 0.5; a D3 builder for PRIVATE runs mirroring evals/demo/build_demo.py's mutation + validity-check logic but sourced from v2 run rows + owner-correct labels + /v1/eval/chunks text (via the SDK), never from verdicts; the stratified calibration sample (50 items, all criteria each, excluding heldout2).
Tests: fake Ollama/Gemini HTTP via respx (valid, invalid JSON, wrong nonce, injection text in the answer), cache hit/miss/invalidation, label store concurrency (two writers), sealing, overlap flag, D3 builder on synthetic rows reproduces demo-style items with mechanical checks, sample stratification.`,
  },
  {
    id: 'EV-RUN', title: 'Runners for E-R, E-G, E-V, E-I, E-A (+ latency pass) and the v2 CLI (run, resume, label, summarize, gate, snapshot, demo)',
    owns: 'packages/client/src/ragforge_client/evals/runners.py, cli.py (new), packages/core/tests/module_ev2/test_ev_run_*.py, docs/handoff/EV-RUN.md',
    tests: 'uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_client_evalrun.py -q',
    spec: `Implement ADR-0017 runners (§3) and the CLI against the SDK (eval_* methods, query(stages=True), chat(trace_shown=True), interview_question/grade), interfaces.py and runfile.RunWriter. Other modules are concurrent: call metrics.summarize / gate.evaluate / judge.make_judge / datasets.* through their interface names, test with fakes.
- run_retrieval, run_generator (answer then judge post-hoc: the judge NEVER runs concurrently with answering or the latency pass), run_verifier (resolve D3 chunk keys -> ids; /v1/eval/verify; re-verify each clean source in the same run), run_interview (frozen D4 grades at seed level + the level sweep for gold/partial, 2 replicates; generated questions via chunk_ids for question quality), run_application (+ a separate latency pass; first 2 cold questions reported apart). Every run: header with snapshot digests (from eval_snapshot_status), retrieval config, models, prompt_versions (from done events), judge key; stable keys resolved at start, stale items marked; provider-transient errors (429/5xx) retried up to 2 times then classified; resume via RunWriter; summary file written at finish (metrics.summarize). Pacing for the free tier (configurable RPM).
- cli.py: main(argv) with subcommands: run <kind> --dataset demo|private --arm A [--resume], summarize <run>, label (blind queue in the terminal: no judge score/verdict shown; never heldout2), gate --surface S [--override "..."], baseline set <run> --note, snapshot create|status|reembed, demo (create the demo tenant via the API, upload evals/demo/pages, run all five kinds). The existing ragforge-eval (evalrun.py) is untouched: put the wiring (script name ragforge-eval2 or a subcommand dispatch) in your handoff.
Tests: each runner against a fake SDK client producing the D0 shapes (row-for-row equivalent to the golden fixtures), resume after a crash, transient retry and classification, stale-key handling, header contents, judge-after-answer ordering, CLI argument parsing and the blind label flow.`,
  },
  {
    id: 'EV-UI', title: 'Eval page in the Streamlit UI: runs, compare, drill-down, blind label queue + calibration panel, verifier and interview views',
    owns: 'packages/ui/src/ragforge_ui/eval_page.py and packages/ui/src/ragforge_ui/eval_view/** (new), packages/ui/src/ragforge_ui/app.py (ADDITIVE: register the page/navigation only), packages/core/tests/module_ui/test_ev_ui_*.py, docs/handoff/EV-UI.md',
    tests: 'uv run pytest packages/core/tests/module_ui -q',
    spec: `Implement ADR-0017 §6 against the LOCKED files only: runfile.read_run / iter_runs, RunSummary (<run>.summary.json), GateVerdict files, Calibration, LabelReader/LabelWriter (interfaces.py; EV-JUDGE implements the store concurrently: code against the protocol and use a small in-test fake). Data directory configurable (default evals/private/runs; tests use evals/demo/fixtures and tmp copies ONLY).
- Runs view: header strip (surface, snapshot content digest, models, prompt sets), headline metrics with CIs and n, gate state and any override. Compare: two runs with the same content digest only (else explain), paired deltas labelled exploratory, per-question flips; heldout2 rows hidden unless revealed. Drill-down: answer, shown notes (fetched text only on demand via the SDK eval_chunks with digest check; never stored), verdict + catching layer, judge scores/reasons, owner label. Label queue: BLIND (question, notes, answer only: no judge score, no reason, no verdict), keyboard-friendly, never heldout2, writes via LabelWriter. Calibration panel: per criterion kappa + CI, raw agreement, confusion matrix, leniency, trusted state; "judge not trusted for X" when untrusted, and the no-trusted-judge outcome. Verifier view: badge-flip recall (SLO line), span recall by mutation and layer, caught-unlocated, false-reject split. Interview view: ranking accuracy (k/n exact CI), detection given vs kept, level trend + test-retest SD.
- Render server/answer text as Markdown safely (existing UI rules: no unsafe HTML).
Tests: AppTest on the golden fixtures (every view renders the expected numbers from *.summary.json), compare refuses mismatched digests, the label queue shows no judge/verdict text and never heldout2 items, label writes go through the writer, untrusted criteria shown as not trusted.`,
  },
]

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests_passing: { type: 'boolean' },
    test_command: { type: 'string' },
    checks: { type: 'string' },
    handoff: { type: 'string' },
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
Handoff file: ${ROOT}/docs/handoff/${m.id}.md
Existing suites you must keep green: ${m.tests}

WHAT TO BUILD:
${m.spec}`
}

function reviewPrompt(m, round) {
  return `You are a principal engineer with 25 years at a large tech company (distributed systems, security, search/RAG, LLM products). Review ONE module of RAGForge BLIND (round ${round}). Be adversarial but calibrated: 9+ = approve with minor comments; 8 = approve after short fixes; 7 = needs another revision.
Repository: ${ROOT} (NOT a git repo: never run git). Do NOT modify any file. Do NOT run make up/nuke or docker compose. Do NOT open docs/REVIEW.md.
Five other modules are being built concurrently in the same folder; judge ONLY module ${m.id} (failures clearly caused by another module's in-progress files are not this module's defects, but say so).

Read CLAUDE.md, docs/adr/0017-eval-layer.md (the ACCEPTED eval design: binding, including every review table and the Lock step section), packages/client/src/ragforge_client/evals/schema.py, runfile.py and interfaces.py (LOCKED seams), the contracts packages/contracts/src/ragforge_contracts/api.py (eval section) and sse.py, the SDK client.py, evals/demo/** (the D0 demo set and golden fixtures; evals/demo/fixtures/reference_metrics.py is the executable metric definition). Then read ALL of module ${m.id}'s files and changes: ${m.owns}, and its handoff.

Module ${m.id} was supposed to build:
${m.spec}

Run read-only verification: ${m.tests}; uv run mypy on its source files and tests; uv run ruff check on its paths; uv run lint-imports. Write any scratch probes ONLY under <SCRATCH>/review-${m.id}/ and run them with uv run python (never name a probe after a library, e.g. h2.py).

Check: correctness against ADR-0017 (metric definitions exactly as reference_metrics.py, statistics as specified), security (eval scope and tenant isolation, cross-tenant copy guards, RLS, prompt injection into the judge, key handling, privacy: never note text in run files, never private data in tests), failure handling, test quality (do tests PROVE behaviour; try to construct a counterexample the tests miss), code quality, ownership violations. Return P0/P1 only for real defects; style nits are P2.`
}

function fixPrompt(m, review, round) {
  return `${COMMON}
${moduleBlock(m)}

You are continuing module ${m.id}, already implemented in the owned paths. A blind principal-engineer review (round ${round}) scored it ${review.score}/10 with these findings. Fix EVERY P0 and P1 (and cheap P2s inside your paths). If you believe a finding is wrong, do not change code; explain why in your summary. Stay strictly inside your owned paths; anything else goes into your handoff file. Re-run the definition-of-done checks.

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
      final_score: review ? review.score : null,
      open_blockers: review ? review.findings.filter(f => f.severity !== 'P2') : [],
      open_p2: review ? review.findings.filter(f => f.severity === 'P2') : [],
      scores: history.map(h => (h ? h.score : null)),
      build,
    }
  },
)

return results.filter(Boolean)
