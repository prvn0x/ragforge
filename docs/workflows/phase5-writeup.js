// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-phase5-writeup',
  description: 'Phase 5 write-up: 5 parallel builders (key commands, README + doc test, usage guide, generated results, decisions page) with blind principal review loops',
  phases: [
    { title: 'Build', detail: '5 builders in owned paths' },
    { title: 'Review', detail: 'blind principal review per deliverable' },
    { title: 'Fix', detail: 'fix P0/P1, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'
const SCRATCH = '<SCRATCH>'

const COMMON = `
You are a senior engineer on Phase 5 (the write-up) of RAGForge: a multi-tenant RAG over the owner's Notion notes, used daily and also a Senior/Staff portfolio piece. Four other engineers work in the same folder AT THE SAME TIME on other deliverables.
Repository: ${ROOT}  (NOT a git repo: never run git, never git init, never commit.)

READ FIRST: ${ROOT}/CLAUDE.md; ${ROOT}/docs/PHASE5.md IN FULL (the plan; "Revision 2" and "Revision 3" are BINDING and override the earlier text where they differ); ${ROOT}/Makefile; the current ${ROOT}/README.md.

HARD RULES:
1. Create/edit ONLY the paths you own. Never edit pyproject.toml, uv.lock, compose.yaml, Dockerfile, .env, .env.example, or files owned by another deliverable. Never run uv add / uv lock / uv sync. Anything you need outside your paths goes in your handoff ${ROOT}/docs/handoff/<ID>.md as an exact snippet.
2. NEVER read or print .env, ~/.config/ragforge/**, evals/private/**, ~/.claude/**. Never print a secret. Use only synthetic data and the public Acme demo set (evals/demo/**) for examples. No note titles, private question text, tenant/collection ids, key prefixes, or absolute home paths in anything you write.
3. Do NOT run make up / make down / make nuke / docker compose, do not call a model, do not touch the live stack (it is in daily use). Reading code, docs, ADRs, handoffs, \`make help\`, and \`uv run <cli> --help\` is fine.
4. Python: format/lint ONLY your paths (uv run ruff format <paths>; uv run ruff check --fix <paths>); mypy --strict clean; tests in the owned test dir; no __init__.py in test dirs.
5. Writing style (docs): plain English, short sentences, active voice, no superlatives or marketing ("blazing", "state of the art", "seamless", "robust", "powerful"). Say what is NOT done as plainly as what is. Every factual claim must be checkable: point at a file, a make target or a test. If you cannot confirm a behaviour from code or a test, leave it out and list it under "Open questions" at the end of your handoff. Never guess.
6. Secrets in instructions: tokens and keys are entered only through the hidden-input commands or written to 0600 files by the tools; never tell a reader to paste a token into a chat, a file by hand, or a command line.
7. Principal-engineer quality for code: typed, small functions, docstrings explaining WHY, library primitives over hand-rolled code, tests that PROVE behaviour (mutation-check the key ones and report the result).

Locked cross-document names: README.md, docs/GUIDE.md, docs/RESULTS.md, docs/DECISIONS.md. Locked commands (built by W0 in parallel; write against them): make tenant, make mcp-key (key at ~/.config/ragforge/mcp/mcp.key), make mcp (defaults to that key), make connect-notion (reads the key file), ragforge-eval2 results --check|--write (W3). The quickstart is exactly R3-1.

DEFINITION OF DONE: your tests pass; ruff and mypy clean on your Python; your handoff ${ROOT}/docs/handoff/<ID>.md lists what you built, integration steps for the integrator, and an "Open questions" list.
`

const MODULES = [
  {
    id: 'W0-KEYS', title: 'Key commands a new user can run: make tenant, make mcp-key, make mcp default key, connect-notion key discovery',
    owns: 'Makefile (ONLY the targets tenant, mcp-key, mcp, mcp-install, connect-notion and their help text; add new targets near the related ones), scripts/create_mcp_key.py (new), scripts/create_tenant.sh or .py (new, if you need one), scripts/connect_notion.py, packages/core/tests/test_w0_keys_*.py, docs/handoff/W0-KEYS.md',
    tests: 'uv run pytest packages/core/tests/test_w0_keys_*.py -q',
    spec: `Implement docs/PHASE5.md R2-1 as amended by R3-3 and R3-5 exactly.
- make tenant (NAME default personal, OWNER default $USER): runs ragforge-admin create-tenant inside the api container (see packages/core/src/ragforge_core/apps/api/admin.py for its exact stdout lines: tenant_id=, user_id=, scopes=, api_key=; and the eval-demo target in the Makefile for the existing pattern). Writes ONLY the api_key value to ~/.config/ragforge/owner.key (dir 0700, file 0600, umask 077), never printed or echoed. Refuses if ANY *.key already exists directly in ~/.config/ragforge/ (a second key breaks make ask/ui which refuse to choose). Shows admin.py's stderr except the "Store this key now" line (no 2>/dev/null). Removes a partial/empty file on failure with a clear message ("is the stack up? make up"). Prints tenant_id and the key file path only. Prefer a small script (testable) over a long Makefile one-liner.
- make mcp-key: scripts/create_mcp_key.py (run via uv run python) finds the owner key with ragforge_client.ask's discovery rule (import it; --owner-key-file overrides), mints "mcp:claude-code" with scopes read, chat, write via RagforgeClient.create_api_key, writes ~/.config/ragforge/mcp/mcp.key (dir 0700, file 0600, O_EXCL; refuses if it exists). Prints prefix + scopes only, never the secret. RAGFORGE_URL env overrides http://localhost:8000.
- make mcp: defaults to --key-file ~/.config/ragforge/mcp/mcp.key when ARGS is empty. make mcp-install: prints the claude mcp add command with the new key path and points to docs/GUIDE.md (not docs/handoff/MCP.md).
- connect-notion: target becomes uv run python scripts/connect_notion.py; the script imports resolve_key_file/read_key (or the actual names; read ragforge_client/ask.py) and reads the RAGForge key from --key-file, else $RAGFORGE_KEY_FILE, else the single ~/.config/ragforge/*.key; falls back to the hidden getpass prompt only when no key file is found. The Notion token stays hidden input. Keep its existing behaviour otherwise.
Tests (synthetic temp HOME / dirs, fakes for the client and for the docker exec; never the live stack): file modes 0700/0600; refuse-on-exists (owner.key and any other *.key); the secret never appears in stdout/stderr; partial file removed on failure; the "Store this key now" line filtered; mcp key discovery order and O_EXCL; connect_notion discovery order and the getpass fallback; make mcp default key path (parse the Makefile). Mutation-check the key ones.`,
  },
  {
    id: 'W1-README', title: 'README.md rewrite + the docs consistency test',
    owns: 'README.md, packages/core/tests/test_docs_consistency.py, docs/handoff/W1-README.md',
    tests: 'uv run pytest packages/core/tests/test_docs_consistency.py -q',
    spec: `README.md (the front door, at most 200 lines), per docs/PHASE5.md deliverable W1 plus R2-8 items 1, 2, 5:
- What it is in 2 lines. The 3 guarantees (citations verified against the source and deep-linked to the paragraph; tenant isolation by Postgres RLS; local-first: embeddings and the index always local, chat model local Ollama by default or Gemini by choice), EACH linking to the code and the named tests that prove it (find them: RLS policies + isolation tests, the verifier + its attack/adversarial tests, the egress guard pytest plugin + no-silent-egress tests).
- A mermaid architecture diagram (packages contracts/core/client/mcp/ui; api, worker, Postgres+pgvector, host Ollama or Gemini, Notion, Prometheus/Grafana obs profile). Check names against compose.yaml and packages.
- Quickstart exactly R3-1 (6 commands), each with a one-line comment; "check it without Docker: make test-unit".
- A headline results table: 3-5 rows, inside a generated block if W3's marker format allows referencing, otherwise a short table that links to docs/RESULTS.md and says the numbers there are generated. Do NOT invent numbers: if you show any, take them from evals/demo/runs/*.summary.json and name the run id; prefer linking.
- Doc map: GUIDE, RESULTS, DECISIONS, DESIGN.md ("design as of Phase 0; later changes are in the ADRs"), SECURITY.md, docs/adr/, REVIEW.md, NEXT.md. Three limit bullets linking to DECISIONS and RESULTS.
- The current README's accurate provider/privacy detail moves to the GUIDE (W2 owns it; just link). An optional screenshot is an HTML comment TODO, never a broken image.
test_docs_consistency.py (R2-2, R2-8 item 4): over README.md, docs/GUIDE.md, docs/RESULTS.md, docs/DECISIONS.md (skip a file that does not exist yet with a clear xfail-free skip, so it passes while others are in progress; the integrator removes the skip later: make that a single module-level constant REQUIRED_DOCS the integrator flips): every \`make X\` mentioned is a real target parsed from rule lines ^[a-z0-9-]+: (not .PHONY); every relative link resolves and every #anchor matches a heading (GitHub slug rules); every docs/adr/*.md has a row (link) in DECISIONS.md; every top-level docs/*.md is linked from README (explicit exclusion list allowed, e.g. PHASE1.md, PHASE5.md); README at most 200 lines; a banned-word list (blazing, state-of-the-art, seamless, robust, powerful, cutting-edge, revolutionary); no UUID and no "/Users/" string in the four documents; the ragforge-eval2 subcommands mentioned in \`make eval ARGS="<sub> ...\` exist (parse cli.py's argparse or run --help). Mutation-check: each check fails on a planted violation (write those as tests over temp files, so the real-doc checks and the checker logic are both tested).`,
  },
  {
    id: 'W2-GUIDE', title: 'docs/GUIDE.md: the step-by-step usage guide for a new user',
    owns: 'docs/GUIDE.md, docs/handoff/W2-GUIDE.md',
    tests: 'uv run pytest packages/core/tests/test_docs_consistency.py -q (owned by W1; may not exist yet; if it exists it must pass for GUIDE)',
    spec: `docs/GUIDE.md per docs/PHASE5.md deliverable W2, R2-5..R2-8 and R3-1, R3-6, R3-7. Audience: someone who wants to use RAGForge for their own notes (study, interview prep, team docs). Every step: the command, what you should see, how to tell it worked, what to do if not. Sections in this order:
1. What you need (tested platform: macOS Apple silicon with Rancher Desktop; Linux differences: Ollama in a container, from the Makefile/compose/doctor.sh) and make doctor (quote its real checks from scripts/doctor.sh).
2. Install and start: make up; the provider prompt (Ollama default vs Gemini, with the privacy trade-off moved from the current README, accurately); make models (always, R3-1); make setup to switch later.
3. Create your tenant and keys: make tenant; make mcp-key (only if you use Claude Code); where the files go and why the MCP key has its own folder.
4. Connect Notion: create an internal integration in Notion, share the pages with it, make connect-notion (hidden input), first sync, how to see progress (find the real way: UI, API sources/jobs routes, logs). Plus "Add files" via the REST upload route in packages/core/src/ragforge_core/apps/api/routes/sources.py (exact path, method, form field, accepted types, size limit from code).
5. Ask: make ui (127.0.0.1:8501), make ask, the REST API (curl example to the chat or query route; note the process-list caveat for keys on the command line, R3 note), Claude Code via make mcp-install.
6. Reading the answer: verified/partial/unsupported badges, flagged claims (red vs muted reasons), extras; from the UI code and ADR-0011/0015/0016.
7. Interview mode: levels SDE-1..Staff, notes vs the opt-in general-knowledge mode (never verified, needs explicit opt-in), how grading works; from the code.
8. Advanced: evaluate on your own notes (R2-7 + R3-6), the full real sequence with flags taken from \`uv run ragforge-eval2 <sub> --help\`: make eval-tenant, snapshot create --source-collection <COLLECTION_ID> (how to find it), questions in evals/private/questions.toml (format from evals/demo/questions.toml), make eval ARGS="run retrieval --dataset private --arm A", baseline set, make eval-gate, labelling in the Eval page. Say what each step costs in model calls where knowable.
9. Monitoring: make obs-up, Grafana at localhost:3000 (user admin, password in .env: tell them where, not to print it), what the board and alerts show, the memory cost.
10. Backups and data: make backup-evals; "there is no database backup today; make nuke deletes everything and cannot be undone" (R2-6); make down vs nuke.
11. Troubleshooting: ONLY entries whose exact error text you quote from code or scripts/doctor.sh, naming the file (R2-6): e.g. Ollama unreachable, ports busy, Gemini quota (QuotaExhausted's user-facing text), no API key found (ask.py's message), more than one key file, a sync that fails on an unshared page, the eval runner's lease-lost error if it exists.
12. Uninstall: make down, make nuke (with the warning), removing ~/.config/ragforge.
Verify every make target, CLI flag, route, port and message against the code. Use placeholders like <COLLECTION_ID> and the Acme demo set for examples.`,
  },
  {
    id: 'W3-RESULTS', title: 'Generated results: ragforge-eval2 results exporter, manifest, sidecars, docs/RESULTS.md',
    owns: 'packages/client/src/ragforge_client/evals/export.py (new), the "results" subcommand wiring in packages/client/src/ragforge_client/evals/cli.py (ADDITIVE only: a new subparser + dispatch; back the file up to ' + SCRATCH + '/phase5/cli.py.bak before editing and edit with the Edit tool, never a script), packages/client/tests/test_w3_export*.py, evals/results.toml (new), evals/demo/runs/*.header.json (generated by your tool), docs/RESULTS.md, .gitignore (ONLY add the two ! exceptions of R2-3/R3-2), docs/handoff/W3-RESULTS.md',
    tests: 'uv run pytest packages/client/tests/test_w3_export*.py packages/client/tests -q -k "export or cli"',
    spec: `Implement docs/PHASE5.md R2-3 + R3-2 exactly, then write docs/RESULTS.md.
Exporter (ragforge_client.evals.export, CLI \`ragforge-eval2 results --check | --write [--doc docs/RESULTS.md] [--manifest evals/results.toml]\`):
- Manifest evals/results.toml: tables, each with id, dataset (demo|private), title, run_ids (explicit), provider, model, and the metric names to show in order with a human label. Fails on a missing or unlisted run; never picks the latest file.
- Reads from each <run>.summary.json ONLY name, value, ci_low, ci_high, n, method, trusted (via the RunSummary model in evals/schema.py is fine, then select fields); note only when it matches a fixed allow-list of templates (derive from metrics.py's note strings); refuses a metric name outside ^[a-z_]+(\\.[a-z0-9_@]+)+$. From the header (line 1 of <run>.jsonl, via runfile.py) ONLY chat_model, verifier_model, dataset.name, dataset.splits, started_at; never arm, snapshot, collection ids or rows. --write writes the sidecar <run>.header.json with only those fields (0644; deterministic key order). --check uses summary + manifest + sidecar only; the header-vs-manifest model cross-check runs only when the .jsonl exists; when the header model is null the caption says "model from the manifest".
- Output between markers <!-- generated:<table-id> run_ids=... digests=<summary header_digest...> --> ... <!-- /generated --> in the doc; --write replaces only the block contents; --check re-renders and exits 1 on any diff. Private tables: --check skips them cleanly (with a printed note) when evals/private/runs does not exist. Rows show value, 95% CI, n, method; captions show provider, model, dataset, n, date.
- Tests: planted summary with a note holding a page title -> note dropped; odd metric name -> refused; header with a collection id / arm -> not in the sidecar or output; manifest/run mismatch -> error; --check passes on fresh output and fails on a one-character edit; private skip; deterministic output. Mutation-check the privacy filters.
- Run --write for the DEMO tables only (evals/demo/runs; the six demo summaries: retrieval, application, verifier, generator, interview, latency) to create the sidecars and fill RESULTS.md. Put the PRIVATE table blocks in RESULTS.md as marker blocks listed in the manifest with run_ids the integrator will fill (generator-A-20261022T071117Z and retrieval-A-20261021T191959Z are the private run ids; you may NOT read them; leave their blocks with a "not generated yet: run ragforge-eval2 results --write" line and make --check skip them when the files are absent). Verifier/judge rows on the private set: "pending owner labels".
- .gitignore: add !evals/demo/runs/*.summary.json and !evals/demo/runs/*.header.json after the evals/*/runs/ line (check git ignore semantics: a negation cannot re-include a file whose parent dir is excluded; use evals/*/runs/* style so it works, and keep .jsonl/target.json/baselines ignored; prove it with \`git check-ignore --no-index -v\` in a temp copy, since the repo is not a git repo, or reason it carefully and say which).
docs/RESULTS.md: datasets (Acme demo: public, 15 synthetic pages, rerunnable with make eval-demo which needs the live stack plus a model; private: the owner's notes, numbers only, no text), each metric's meaning (hit@k, MRR, nDCG, recall, below_tau, correct, verified, abstain_correct, answer_rate, badge-flip recall, caught_by, false_reject, interview bands/detection/level test, latency p50/p95/TTFT), CI methods and why (Wilson for proportions, cluster bootstrap for per-question metrics, exact Clopper-Pearson when degenerate, exact permutation for the level test), the statistical limits only (small n, single labeller, model-specific: Gemini Flash-Lite numbers; Ollama will differ), and how to regenerate. Not-done items belong to DECISIONS; link there.`,
  },
  {
    id: 'W4-DECISIONS', title: 'docs/DECISIONS.md: ADR index with status, review history, parked work, open issues',
    owns: 'docs/DECISIONS.md, docs/handoff/W4-DECISIONS.md',
    tests: 'uv run pytest packages/core/tests/test_docs_consistency.py -q (owned by W1; may not exist yet)',
    spec: `docs/DECISIONS.md per docs/PHASE5.md deliverable W4 + R2-6 (status column) + R2-8 items 1 and 5. This page is the single owner of "not done".
1. ADR index: one row per docs/adr/0001..0018 (link each): decision (one line), the main alternative rejected, why, and status = implemented / partial / not built, with a short note for partial/not built verified against code (e.g. ADR-0014's nightly pg_dump and make reindex are NOT built; check every ADR's promises against the Makefile and code; never mark implemented without finding the code).
2. Review history: a table per phase (Phase 0, 1, 2a, 2b, 3, 3b, 3c, ADR-0017 eval layer, ADR-0018 Phase 4): blind design review score(s), builder scores, final integrated score, and what the blind reviews caught that tests did not (concrete, one line each). Source: docs/REVIEW.md, docs/NEXT.md, docs/handoff/*.md, the ADR review sections. You MAY read REVIEW.md (you are not a reviewer). Quote scores exactly as recorded; do not round or improve them.
3. How the work was done (short): design-first with blind principal reviews, locked seams, parallel builders with blind review loops, integrator fixes, verify by running (make check + live), mutation checks. Link docs/workflows/*.js.
4. Parked work and why: CI (until the repo is on GitHub), Helm/kind/KEDA/NetworkPolicies/multi-arch, the full backup/restore drill, the load test, the DeepEval comparison, the Notion token rotation (owner accepted risk ADR-0018 R2-0: say it is an accepted risk, without details of what was pasted where), the blog post.
5. Known open issues: the ADR-0017 P2 list and the ADR-0018 open P2s (incl. the MCP HTTP logs not redacted), the hosted 5 s latency SLO not met (p95 about 6.6 s on the demo), the held-out set not sealed, verifier/judge on the private set pending owner labels.
6. DESIGN.md sections known to be superseded by later ADRs (list section -> ADR).
Every claim checkable (link the file). No private content (no private question topics or note titles; NEXT.md has a topic list: do NOT copy it).`,
  },
]

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests_passing: { type: 'boolean' },
    checks: { type: 'string' },
    open_questions: { type: 'array', items: { type: 'string' } },
    known_gaps: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files', 'tests_passing', 'checks', 'open_questions', 'known_gaps'],
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
DELIVERABLE ${m.id}: ${m.title}
Paths you OWN (create/edit only these): ${m.owns}
Handoff file: ${ROOT}/docs/handoff/${m.id}.md
Checks: ${m.tests}

WHAT TO BUILD:
${m.spec}`
}

function reviewPrompt(m, round) {
  return `You are a principal engineer with 25 years at a large tech company who has hired many Staff engineers and written and reviewed a lot of developer documentation. Review ONE Phase 5 deliverable of RAGForge BLIND (round ${round}). Be adversarial but calibrated: 9+ = approve with minor comments; 8 = approve after short fixes; 7 = needs another revision.
Repository: ${ROOT} (NOT a git repo: never run git). Do NOT modify any file. Do NOT run make up/down/nuke or docker compose; do not call a model. Do NOT open docs/REVIEW.md. Never read .env, evals/private/**, ~/.config/ragforge/**, ~/.claude/**.
Four other deliverables are being built concurrently; judge ONLY ${m.id} (a failure clearly caused by another deliverable's in-progress file is not this one's defect, but say so).

Read CLAUDE.md and docs/PHASE5.md in full (Revisions 2 and 3 are binding). Then read ALL of ${m.id}'s files and its handoff: ${m.owns}.

${m.id} was supposed to build:
${m.spec}

Verify by checking, not by trusting: run ${m.tests}; for Python, mypy and ruff on its files. For DOCS: check EVERY command, make target, CLI flag, route, port, file path, error message, number and claim against the code, Makefile, scripts and summary files (run \`make help\` and \`uv run ragforge-eval2 <sub> --help\` freely); follow the steps as a new user would and find where they would get stuck; look for privacy leaks (private topics, ids, home paths, key prefixes), overclaiming, secrets-in-instructions, and contradictions with the other docs that exist. For CODE: correctness, privacy/secret handling, file modes, failure handling, and whether tests PROVE behaviour (try a counterexample they miss). Scratch probes only under ${SCRATCH}/review-${m.id}/ run with uv run python (never name a probe after a library).
Return P0/P1 only for real defects; style nits are P2.`
}

function fixPrompt(m, review, round) {
  return `${COMMON}
${moduleBlock(m)}

You are continuing ${m.id}, already built in the owned paths. A blind principal review (round ${round}) scored it ${review.score}/10 with these findings. Fix EVERY P0 and P1 (and cheap P2s inside your paths). If you believe a finding is wrong, do not change anything for it; explain why in your summary. Stay strictly inside your owned paths; anything else goes into your handoff. Re-run your checks.

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
      open_questions: build ? build.open_questions : [],
      files: build ? build.files : [],
    }
  },
)

return results.filter(Boolean)
