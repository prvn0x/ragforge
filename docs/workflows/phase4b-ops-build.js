// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-ops-build',
  description: 'ADR-0018 thin Phase 4: 4 parallel builders (metrics instrumentation, Prometheus/Grafana stack, security scans, threat model + egress), blind principal review loops',
  phases: [
    { title: 'Build', detail: '4 builders in owned paths' },
    { title: 'Review', detail: 'blind principal review per module' },
    { title: 'Fix', detail: 'fix P0/P1, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'

const COMMON = `
You are a senior engineer building ONE module of RAGForge's Phase 4 (multi-tenant RAG over the owner's Notion notes; a personal project and a Senior/Staff portfolio piece) in a shared folder where 3 other engineers build other modules AT THE SAME TIME.
Repository: ${ROOT}  (NOT a git repo: never run git. Do not commit.)

READ FIRST: ${ROOT}/CLAUDE.md; ${ROOT}/docs/adr/0018-ops-hardening-security-observability.md (ACCEPTED: its "Revision 2" section is BINDING and overrides the earlier sections); the LOCKED seams ${ROOT}/packages/core/src/ragforge_core/observability/catalog.py (every metric name, type, label, allowed value, bucket, port) and prompt_context.py (the prompt/surface ContextVars); DESIGN.md §2 and §12; compose.yaml; Makefile.
The locked seams MUST NOT be changed; if one is insufficient, say so in your handoff.

HARD RULES:
1. Create/edit ONLY the paths you own. Never edit pyproject.toml, uv.lock, compose.yaml, Makefile, Dockerfile, docker/bootstrap.sh, .env, .env.example, docs/** (except your handoff and files your module owns), or the locked seams. Never run uv add / uv lock / uv sync. prometheus-client is already installed in ragforge-core.
2. NEVER read or print the real .env, ~/.config/ragforge/**, evals/private/**, or ~/.claude/**. Never print a secret. Test only with synthetic data.
3. Changes outside your paths (Makefile targets, compose services, version bumps, bootstrap.sh) go in your handoff file ${ROOT}/docs/handoff/<MODULE_ID>.md as exact snippets; the integrator applies them.
4. Format/lint ONLY your paths (uv run ruff format <paths>; uv run ruff check --fix <paths>).
5. Tests: files prefixed with your module id (lowercase, dashes as underscores) in your owned test dir; no __init__.py in test dirs. Do NOT run make up / make nuke / docker compose up|down (the live stack is in use). You MAY run one-off scanner/tool containers with \`docker run --rm\` (no volume mounts of ~/ or the Docker socket) and \`uvx\`.
6. Layering: uv run lint-imports must pass.
7. Principal-engineer quality: typed (mypy --strict), small functions, docstrings explaining WHY, library primitives over hand-rolled code, tests that PROVE behaviour (mutation-check the key ones), clean failure handling.

DEFINITION OF DONE: your tests pass; ruff check + format --check clean on your paths; mypy clean on your sources and tests; lint-imports passes; the suites listed for your module still pass. Write your handoff: what you built, exact integration steps, what the integrator must test live.
`

const MODULES = [
  {
    id: 'OBS-METRICS', title: 'Instrumentation: every catalog.py metric, InstrumentedChatModel, DB collector (rf_metrics), migration 0013 freshness view, /metrics on internal ports, log redaction',
    owns: 'packages/core/src/ragforge_core/observability/** (NOT catalog.py, prompt_context.py), additive hooks in packages/core/src/ragforge_core/apps/api/** (middleware.py, app.py, chat_stream.py, routes/interview.py, main.py) and apps/worker/**, the call sites in agent/** that build prompts (wrap with prompt_context.for_prompt), agent/retrieval.py, adapters/llm/** (embedder metrics only), adapters/connectors/notion/client.py (request status counter), bootstrap.py (wrap the ChatModel with InstrumentedChatModel), db/engine.py (pool gauges), config.py (ADDITIVE: metrics port settings), db/migrations/versions/0013_*.py + the CODE_REQUEST_SCHEMA constant bump to 13 + its tests, packages/core/tests/module_obs/**, docs/handoff/OBS-METRICS.md',
    tests: 'uv run pytest packages/core/tests/module_obs packages/core/tests/module_e2 packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_d1 packages/core/tests/module_worker packages/core/tests/test_db_foundation.py -q',
    spec: `Implement ADR-0018 Revision 2 R2-1..R2-6 and the log-redaction part of R2-8.
- Instrument EVERY metric in catalog.CATALOG at its stated source, with exactly its labels/buckets; unknown label values map to "other"; never a tenant/user/chunk id or text as a label. Define metrics once at import (create_app is called repeatedly in tests); a registry helper for tests.
- InstrumentedChatModel wraps the ChatModel port (bootstrap): times, counts tokens from the usage reports, classifies outcomes into catalog.LLM_OUTCOMES using the adapters' own error classes; reads prompt_context.prompt_name. Every node/route that builds a prompt (answer, answer_strict, ground_check, rewrite, contextualize, abstain, actions, action_reply, extras_*, interview_*) wraps its call in for_prompt("name@version"). Test: no path records prompt="unknown".
- surface ContextVar set by the API per request from the key's tenant kind / route (eval tenant -> eval; demo tenant name -> demo; interview routes -> interview; else chat).
- HTTP middleware per R2-5 (outermost; matched route template; "unmatched"; SSE excluded from the histogram; disconnected; health excluded).
- DB collector: a prometheus Collector in the API process that, at scrape time, reads rf.queue_stats and rf.connector_freshness as rf_metrics (a small dedicated engine with the metrics role from Settings.db roles; 5 s timeout; failures -> the gauges absent + a counter, never a 500). Migration 0013 (checksum-guarded like 0001-0012): CREATE VIEW rf.connector_freshness (aggregates only, NO tenant ids; per connector kind: the age of the STALEST enabled connector's last successful sync, where a no-change poll counts as success: derive it from finished sync jobs in rf.jobs + rf.jobs_history or the existing connector state, whichever is authoritative) GRANT SELECT TO rf_metrics; bump CODE_REQUEST_SCHEMA to 13.
- /metrics via prometheus_client.start_http_server on API_METRICS_PORT / WORKER_METRICS_PORT (addr 0.0.0.0 in the container; configurable; disabled by default in tests); NOT on the FastAPI app (GET :8000/metrics stays 404; absent from OpenAPI). Single-process guard (R2-6). build_info from package metadata.
- Log redaction: a structlog processor in observability/logging.py masking token shapes (ntn_, secret_, rf_<prefix>_<secret>, AIza..., Bearer ...), with tests through configure_logging.
Tests: names/labels/buckets equal catalog; cardinality (labels only from allowed sets); each metric moves on a real code path (fakes); no prompt=unknown; /metrics not on :8000; migration 0013 view has no tenant columns and rf_metrics can read it but not rf.jobs; redaction.`,
  },
  {
    id: 'OBS-STACK', title: 'Prometheus + Grafana stack: scrape config, alert rules (+ promtool unit tests), provisioned dashboard, compose profile snippet',
    owns: 'docker/observability/** (prometheus.yml, rules/*.yml, rules tests, grafana/provisioning/**, grafana/dashboards/ragforge.json), packages/core/tests/test_obs_stack_*.py, docs/handoff/OBS-STACK.md',
    tests: 'uv run pytest packages/core/tests/test_obs_stack_*.py packages/core/tests/module_g2 -q',
    spec: `Implement ADR-0018 Revision 2 R2-4, R2-6 (scrape side), R2-7 (as a compose snippet in the handoff: the integrator adds it to compose.yaml).
- prometheus.yml: scrape api:9101 (static) and worker via dns_sd_configs (A record "worker", port 9102); 15s interval; rule_files. No admin/lifecycle API flags (in the compose snippet).
- Alert rules exactly per R2-4 (volume guards, per-provider latency on bucket edges, verified share over answered owner-surface turns only, quota, rate limit, stale sync with for:15m, interactive/webhook queue age, dead jobs, up==0), each with a summary/description saying what to do. promtool unit tests (rules tests YAML) covering firing AND not firing (one-user low volume must not fire); run them with the official prom/prometheus image via \`docker run --rm --entrypoint promtool\` (no socket mount) and wire that into a pytest that skips cleanly when docker is unavailable; also \`promtool check config\` and \`check rules\`.
- Grafana: provisioning (datasource Prometheus, dashboards from file, alert list panel), dashboard JSON with the 4 rows of ADR-0018 §6 using ONLY catalog metrics (a test parses every PromQL expression in the dashboard and the rules and asserts every metric name is in catalog.CATALOG with valid labels). Settings in the compose snippet: phone-home off (GF_ANALYTICS_REPORTING_ENABLED=false, CHECK_FOR_UPDATES=false, CHECK_FOR_PLUGIN_UPDATES=false, GF_NEWS_NEWS_FEED_ENABLED=false), anonymous + signup off, enforce_domain, bound 127.0.0.1:3000, admin password from an env var the integrator generates in bootstrap.sh; images pinned by digest (look up current stable digests with \`docker pull\` + \`docker inspect\`? NO: do not pull into the shared daemon if unsure; propose tags + leave digest pinning as a handoff step if you cannot verify), mem_limit (prometheus 256m, grafana 192m), retention 15d/512MB, named volumes, profiles: [obs].
Tests: the catalog-name test; promtool rule tests (firing and quiet); dashboard JSON validity; compose snippet has no published prometheus port and grafana only on 127.0.0.1.`,
  },
  {
    id: 'SEC-SCAN', title: 'Security scans: dependency audit (pip-audit), image scan (Trivy vuln+secret+misconfig on docker save), secret scan (gitleaks with custom rules), ignore-file coverage',
    owns: 'scripts/security/** (new), .gitleaks.toml (new), packages/core/tests/test_sec_scan_*.py, docs/handoff/SEC-SCAN.md',
    tests: 'uv run pytest packages/core/tests/test_sec_scan_*.py -q',
    spec: `Implement ADR-0018 §Security 1-3 + Revision 2 R2-8 (scan parts).
- scripts/security/audit.sh (or .py): pip-audit via \`uvx pip-audit\` over \`uv export --frozen\` requirements (no install into the project); Trivy via \`docker run --rm aquasec/trivy\` over \`docker save\` tarballs of the built api/worker images (scanners vuln,secret,misconfig; also config scan of compose.yaml and Dockerfiles), HIGH/CRITICAL fail unless listed in a justified ignore file. scripts/security/secrets.sh: gitleaks via \`docker run --rm\` over the working tree with .gitleaks.toml (explicit path allowlist since the tree is not a git repo: exclude .venv, node caches, run folders; DO scan docs/, evals/demo/, docs/workflows/, packages/; custom rules for ntn_, secret_ (Notion), rf_<16 hex>_<secret> (RAGForge keys), AIza[0-9A-Za-z_-]{35} (Gemini)). NEVER print a found secret value (redact in output). Mount only the repo read-only into the containers, never ~/ or the Docker socket.
- Run them now; FIX what is in your paths, and for findings outside your paths write exact fixes (version bumps for uv to apply, Dockerfile base bumps, .gitignore/.dockerignore additions: *.key, *.pem, .env.*) in the handoff. Report a summary of findings (counts by severity, package/image, fixed vs proposed vs justified-ignored).
- A pytest that checks .gitignore/.dockerignore coverage of the sensitive paths (.env, .env.*, *.key, *.pem, evals/private/, evals/*/runs/, the eval key dir) and that the gitleaks config's custom rules detect synthetic samples of each token shape and do not fire on the repo's known fixture strings.
- Makefile targets (handoff): make audit, make secrets-scan.`,
  },
  {
    id: 'SEC-THREAT', title: 'Threat model + rotation runbook (docs/SECURITY.md), egress audit tests (socket guard, host inventory, LangSmith off, flashrank no-download)',
    owns: 'docs/SECURITY.md (new), packages/core/tests/test_sec_threat_*.py, packages/core/tests/egress_guard.py (new pytest plugin module; the integrator registers it in conftest), docs/handoff/SEC-THREAT.md',
    tests: 'uv run pytest packages/core/tests/test_sec_threat_*.py -q',
    spec: `Implement ADR-0018 §Security 4-5 + Revision 2 R2-0 and R2-8 (threat-model parts).
- docs/SECURITY.md: assets, trust boundaries (diagram in text), top threats with existing mitigations (cite code/ADRs) and residuals: tenant isolation (RLS, definers, eval-tenant guard, cross-tenant copy guard), keys (hashing, scopes, MCP/eval never approvals:write, 0600 files, hidden input), prompt injection (fencing, verifier, literal extras, MCP indirect injection into agentic tools as a residual), local UI (loopback, Host/Origin, XSRF), egress (exact list), Docker socket (docker inspect shows env secrets: residual; compose file-secrets parked), iCloud Desktop sync (whether the repo leaves the machine: explain how the owner checks it; the repo is under ~/Desktop), KEK + DB role passwords backup/rotation, log redaction, backups (backup-evals writes to ~/.local/share), out of scope (host OS/Docker/Gemini compromise). R2-0 ACCEPTED RISK section: secrets pasted into a chat and a chat app (NO values), where copies may persist (the chat history, the chat app, ~/.claude/projects transcripts on disk), and the exact ROTATION RUNBOOK for: Notion integration token (regenerate in Notion, then make connect-notion with hidden input), RAGForge API keys (mint replacement via POST /v1/keys with the owner key, update key files, revoke old via DELETE), Gemini key, KEK (note: rotating it requires re-connecting Notion), DB role passwords; plus a pre-first-commit checklist (gitleaks git on history, .gitignore check).
- Egress audit tests: inventory outbound hosts from Settings defaults + URL literals in packages/** (expected: Notion api.notion.com, Gemini generativelanguage.googleapis.com, host Ollama host.docker.internal/localhost) and FAIL on a new one; egress_guard.py: a pytest plugin that refuses non-loopback socket connects during unit tests (opt-out marker for tests that legitimately need it; respx/testcontainers keep working); LangSmith tracing pinned OFF (env in tests; compose.yaml env has no LANGCHAIN_TRACING_V2/LANGSMITH_TRACING/LANGCHAIN_API_KEY); flashrank allow_download False pinned.
Do not read .env or ~/.config; reason from code and compose.yaml.`,
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
Three other modules are being built concurrently in the same folder; judge ONLY module ${m.id} (failures clearly caused by another module's in-progress files are not this module's defects, but say so).

Read CLAUDE.md, docs/adr/0017-eval-layer.md (the ACCEPTED eval design: binding, including every review table and the Lock step section), packages/client/src/ragforge_client/evals/schema.py, runfile.py and interfaces.py (LOCKED seams), the contracts packages/contracts/src/ragforge_contracts/api.py (eval section) and sse.py, the SDK client.py, evals/demo/** (the D0 demo set and golden fixtures; evals/demo/fixtures/reference_metrics.py is the executable metric definition). Then read ALL of module ${m.id}'s files and changes: ${m.owns}, and its handoff.

Module ${m.id} was supposed to build:
${m.spec}

Run read-only verification: ${m.tests}; uv run mypy on its source files and tests; uv run ruff check on its paths; uv run lint-imports. Write any scratch probes ONLY under <SCRATCH>/review-${m.id}/ and run them with uv run python (never name a probe after a library, e.g. h2.py).

Check: correctness against ADR-0018 Revision 2 (metric names/labels/buckets exactly as catalog.py; cardinality rules: no tenant/user/chunk ids or text in labels; alert soundness at one user; exposure: never on :8000/v1, no published ports; Grafana phone-home off), security (secret handling, nothing leaves the machine, scanner usage without the Docker socket), test quality (do tests PROVE it, mutation-checked), ownership, failure handling, test quality (do tests PROVE behaviour; try to construct a counterexample the tests miss), code quality, ownership violations. Return P0/P1 only for real defects; style nits are P2.`
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
