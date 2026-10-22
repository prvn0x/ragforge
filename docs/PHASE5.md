# Phase 5: the write-up (design note, 2026-10-22)

Status: DRAFT, for a blind design review before any writing starts.

## Goal
Make RAGForge usable and checkable by people who were not in the room:
- **A reviewer** (hiring manager, Staff/Principal engineer) who gives it five minutes: what it is, why it is built this way, what was measured, what is not done.
- **A user** who wants to run it on their own Notion workspace or files and use it for their own work (study, interview prep, team docs).

## Owner decisions
- In: README, a step-by-step usage guide, measured results tables, a design-decisions and review-history page.
- Out: a recorded demo script. The blog post is optional and later.
- Results rows that need the owner's labels (verifier on the private set, judge calibration) ship as "pending labels" and are filled later by rerunning the exporter.
- No git. The owner creates the repository; Phase 5 ends with a pre-publish checklist they run.

## Deliverables (one owner each, so writers never edit the same file)
| ID | Files | What |
|---|---|---|
| W1 README | `README.md`, `tests/docs/test_docs_consistency.py` | The front door. What it is (2 lines), the 3 guarantees (verified citations that deep-link, tenant isolation by Postgres RLS, local-first), a mermaid architecture diagram, a 5-command quickstart, a headline results table (3–5 rows, linking to RESULTS.md), the doc map, limits. At most about 200 lines; the current README's accurate detail (provider choice, privacy) moves to the guide, not lost. |
| W2 GUIDE | `docs/GUIDE.md` | Task-oriented steps for a new user. Each step: the command, what you should see, how to tell it worked, what to do if not. Sections: prerequisites and `make doctor`; install and `make up`; choosing Ollama or Gemini (with the privacy trade-off); connecting Notion (create an internal integration, share pages with it, `make connect-notion`, first sync, how to see progress); asking (UI, `make ask`, the REST API with a curl example, MCP in Claude Code via `make mcp-install`); reading the badges (verified/partial/unsupported, flagged claims); interview mode (levels, the opt-in general mode); evaluating on your own notes (`make eval-tenant`, writing questions in the TOML format, `make eval`, `make eval-gate`, labelling); observability (`make obs-up`, the board, the alerts); backups and what `make nuke` destroys; troubleshooting (the real failure modes from NEXT.md/CLAUDE.md: Ollama unreachable, Gemini quota, ports busy, laptop sleep during a run); uninstall. |
| W3 RESULTS | `docs/RESULTS.md`, `evals/tools/export_results.py`, `evals/tools/tests/test_export_results.py` | The exporter reads `*.summary.json` files (metric numbers only: name, value, CI, n, method) and prints markdown tables. Nothing is typed by hand. RESULTS.md holds: what each dataset is (Acme demo: public and rerunnable; private: the owner's notes, numbers only), what each metric means, the CI methods and why (Wilson, cluster bootstrap, exact), the tables, the honest limits (small n, a single owner labelling, the hosted 5 s SLO not met, the held-out set not sealed yet), and how to reproduce the demo tables (`make eval-demo`). |
| W4 DECISIONS | `docs/DECISIONS.md` | An index of ADR-0001..0018: one row each (decision, the main alternative rejected, why, link). The review history from REVIEW.md as a table (phase, blind design score, final score, what the reviews caught that tests did not). Parked work and why (CI until GitHub, Helm/kind, the load test, the restore drill, DeepEval). Known open P2s. |

The integrator owns the cross-links (fixed names: `README.md`, `docs/GUIDE.md`, `docs/RESULTS.md`, `docs/DECISIONS.md`), runs the exporter on the private summaries, pastes those tables, and writes the pre-publish checklist into NEXT.md.

## Rules for every writer
1. **Checkable claims only.** Every factual claim points at a file, a make target or a test. No number that does not come from the exporter or a cited summary file.
2. **Commands must exist.** Every `make X` in README/GUIDE is a real Makefile target, and every relative link resolves. `tests/docs/test_docs_consistency.py` (W1) enforces both for all four documents, so it runs in `make check`.
3. **Privacy.** Nothing from `evals/private/**` except the exporter's numeric output (the integrator runs it). No note titles, question text, tenant or collection ids, key prefixes or `.env` values. Examples use the Acme demo set or obvious placeholders.
4. **Secrets in instructions.** Tokens are entered only through the hidden-input commands (`make setup`, `make connect-notion`). No step tells the reader to paste a token into a file by hand, a chat or a command line.
5. **Plain style.** Short sentences, active voice, no superlatives ("blazing", "state of the art"). Say what is not done as plainly as what is.
6. **Accuracy over completeness.** If a writer cannot confirm a behaviour from code or a test, they leave it out or mark it as an open question for the integrator; they do not guess.

## What writers may and may not do
- May: read the code, docs, ADRs, handoffs and the Makefile; run `make help`; run the exporter and its tests on `evals/demo/runs`; run `uv run pytest` on their own test files; run ruff and mypy on their own Python files.
- May not: read `.env`, `evals/private/**`, `~/.config/ragforge/**` or `~/.claude/**`; start, stop or touch the stack; call a model; run git.

## Review plan
- This note: one blind principal reviewer before writing starts.
- Each writer: a blind review loop, at most two fix rounds. The W2 reviewer follows the guide step by step against the code and the Makefile, as a newcomer would.
- Final: one blind reviewer reads all four documents as an outside hiring manager, then as a new user; then `make check`.

## Risks
- **Doc drift** after Phase 5: mitigated by the consistency test (targets and links) and by generated tables.
- **Overclaiming from small n** (n = 4 to 32): every table shows n and the CI; the RESULTS limits section says what the numbers do not show.
- **A guide that only works on this machine:** the guide states the tested platform (macOS on Apple silicon, Rancher Desktop) and the expected platform differences (Linux runs Ollama in a container).

---

## Revision 2: binding changes from the design review (6/10, approve after changes). Overrides the above where they differ.

### R2-1 (P0-1): a key a new user can actually get. New builder W0 KEYS, runs in parallel; the interface below is LOCKED, so W2 writes against it.
- `make tenant` (optional `NAME=...`, default `personal`; `OWNER=...` display name, default `$USER`): runs `ragforge-admin create-tenant` in the api container and writes the owner key to `~/.config/ragforge/owner.key` (dir 0700, file 0600, `umask 077`), never printed. It refuses if that file exists ("already have a key at ..."), and removes a partial file on failure (the `eval-demo` pattern).
- `make mcp-key`: `scripts/create_mcp_key.py` (stdlib + the workspace client) mints `mcp:claude-code` with scopes read, chat, write using the owner key file, and writes `~/.config/ragforge/mcp/mcp.key` (its OWN folder: `make ask`/`make ui` pick the single `*.key` in `~/.config/ragforge/` and refuse to guess between two). Refuses if the file exists. Prints the key prefix and scopes, never the secret.
- `make mcp-install` points at `docs/GUIDE.md` and the new key path; `make mcp`'s help text uses the new path.
- `scripts/connect_notion.py` reads the RAGForge key from `--key-file`, else `$RAGFORGE_KEY_FILE`, else the single `~/.config/ragforge/*.key` (the same rule as `ask`), and falls back to the hidden prompt only when none is found. The Notion token stays hidden input.
- Tests for all four (key file modes, refuse-on-exists, never printed, discovery order), in `packages/core/tests/` so `make check` runs them. No live-stack run by W0; the integrator runs them live.

### R2-2 (P0-2): every new test runs in `make check`.
- The exporter moves into the package: `packages/client/src/ragforge_client/evals/export.py`, exposed as `ragforge-eval2 results --check | --write` (W3), tests in `packages/client/tests/`.
- The doc consistency test is `packages/core/tests/test_docs_consistency.py` (W1).
- Exit criterion: `uv run pytest --collect-only -q` lists both.

### R2-3 (P1-1..4): generated tables, checked, private-safe.
- `.gitignore`: add `!evals/demo/runs/*.summary.json` (integrator). `.jsonl`, `target.json`, baselines stay ignored (they hold local UUIDs).
- A committed manifest `evals/results.toml` maps each table to explicit `run_id`s plus the `provider` and `model` that produced them. The exporter fails on a missing or unlisted run and never picks "the latest" file.
- The exporter reads from a summary ONLY `name, value, ci_low, ci_high, n, method, trusted`; `note` only when it matches a fixed allow-list of templates; it refuses a metric name outside `^[a-z_]+(\.[a-z0-9_@]+)+$`. From the run header (`<run>.jsonl` line 1) ONLY `chat_model, verifier_model, dataset.name, dataset.splits, started_at`; never `arm` (free text: the manifest names it), `snapshot`, collection ids or any row. If the header model disagrees with the manifest it fails; if the header has none (interview and retrieval runs record `chat_model: null` today), the manifest value is used and the caption says "model from the manifest".
- Output goes between markers `<!-- generated:<table> run_ids=... digests=... -->` ... `<!-- /generated -->` in RESULTS.md. `--check` re-renders and diffs; the test runs `--check` for the demo blocks and for private blocks only when `evals/private/runs` exists. The integrator runs `--write`; nothing is pasted.
- Exporter tests include a planted summary (a note holding a page title, an odd metric name, a header with a collection id) that must be refused or stripped.
- Every table caption states provider, model, dataset, n and the date.

### R2-4 (P1-5): what ships, and a pre-publish gate (integrator, in NEXT.md).
- A decision per path: `docs/handoff/` and `docs/workflows/` ship (they show the process) after scrubbing; `PHASE*.md`, `NEXT.md`, `REVIEW.md` ship with a note at the top of NEXT.md that RESULTS.md is the source of truth for numbers.
- Scrub: the private set's topic list (NEXT.md), absolute home paths, hand-typed private numbers that contradict RESULTS (marked "historical, v1").
- Gate script `scripts/security/prepublish.sh` (integrator): greps the files to be published for UUIDs, `/Users/`, key prefixes, `ntn_`, `rf_` tokens and anything under `evals/private`; then `make secrets-scan`. The owner runs it before the first push.

### R2-5 (P1-6): the quickstart, exactly. Local default: `make doctor` → `make up` → `make models` (Ollama only) → `make tenant` → `make connect-notion` → `make ui` (or `make ask`). Gemini replaces `make models` with `make setup`. The checklist adds an owner step: from a fresh clone, under a separate macOS user (so the live data and keys are untouched), follow the README quickstart word for word; every deviation is a guide fix.

### R2-6 (P1-7): troubleshooting and backups are real.
- Each troubleshooting entry quotes the exact error text from code or `scripts/doctor.sh` and names the file. No entry without a source (so "laptop sleep" appears only as the eval runner's real "Lease lost" error, if W2 finds it in code).
- Backups: the guide says plainly that there is no database backup today and that `make nuke` is unrecoverable; it explains `make backup-evals`. DECISIONS has a status column per ADR (implemented / partial / not built), so ADR-0014's nightly dump and `make reindex` show as not built.

### R2-7 (P1-8): evaluating on your own notes, the full sequence, in the GUIDE as an "advanced" section: `make eval-tenant` → `ragforge-eval2 snapshot create` → writing questions (the TOML format, from `evals/demo/questions.toml`) → `make eval ARGS="run retrieval ..."` → `ragforge-eval2 baseline set` → `make eval-gate`; labelling for answer metrics. Every flag is taken from `--help` output, not memory.

### R2-8 (P2s adopted)
1. **Not-done has one owner:** DECISIONS. RESULTS keeps only the statistical limits; README has three limit bullets linking to both.
2. **Proof hooks:** each README guarantee links to the code and the named test that proves it (RLS policies + the isolation tests; the verifier + its attack tests; the local-first egress guard). "Check it without Docker: `make test-unit`." A screenshot of a verified answer on the Acme tenant is an OPTIONAL owner step (no browser tool here), placed as a TODO comment, never a broken image.
3. **Files:** the GUIDE gains a curl upload step for the REST upload route (`routes/sources.py`), so "Notion or files" is true.
4. **Consistency test checks:** targets parsed from rule lines (`^[a-z0-9-]+:`), not `.PHONY`; relative links and `#anchor`s resolve; every `docs/adr/*.md` has a DECISIONS row; every top-level `docs/*.md` is in the README doc map; README ≤ 200 lines; a short banned-word list; no UUID or `/Users/` string in the four documents.
5. **Linked docs:** README links DESIGN.md and SECURITY.md with "design as of Phase 0 / updated through Phase 4" notes; W4 lists in DECISIONS the DESIGN sections known to be superseded by later ADRs.
6. **Open questions:** each writer ends its handoff (`docs/handoff/W<n>.md`) with an "Open questions" list; the integrator clears it before the final review; nothing uncertain goes into the published docs.
7. **Demo caveat:** RESULTS says `make eval-demo` needs the live stack plus a model, and that numbers on Ollama will differ from the Gemini Flash-Lite numbers shown.

## Revision 3: from the re-review (8/10, approve after changes). Overrides R2 where they differ.
- **R3-1 quickstart (both providers):** `make doctor` → `make up` (the first run in a terminal asks for the provider) → `make models` (always: embeddings are local Ollama on both paths; on Gemini it also pulls an unused chat model, say so) → `make tenant` → `make connect-notion` → `make ui` or `make ask`. `make setup` is only for switching provider later.
- **R3-2 fresh-clone check:** `--write` also writes a committed sidecar `<run>.header.json` holding ONLY the allow-listed header fields (`chat_model, verifier_model, dataset.name, dataset.splits, started_at`); `.gitignore` gets `!evals/demo/runs/*.header.json`. `--check` uses the summary, the manifest and the sidecar only; the header-vs-manifest cross-check runs only when the `.jsonl` exists. Manifest arm names and run_ids (which embed the arm: `{kind}-{arm}-{ts}`) are public by review.
- **R3-3 connect_notion:** the target becomes `uv run python scripts/connect_notion.py`, and the script imports `resolve_key_file` / `read_key` from `ragforge_client.ask` (one rule, one implementation). Fallback to the hidden prompt only when no key file is found.
- **R3-4 pre-publish gate:** `scripts/security/prepublish.sh`, run by the owner after `git init && git add -A`, before the first commit. It scans `git diff --cached --name-only` and fails on any staged `.env*` (except `.env.example`), `*.key`, `*.pem`, anything under `evals/private/`, or anything under `evals/*/runs/` other than `*.summary.json` / `*.header.json`. Content patterns, exact: `rf_[0-9a-f]{16}_[A-Za-z0-9_-]{43}` (the shape in `services/auth/keys.py`; DB role names like `rf_api` do not match), `ntn_[A-Za-z0-9]{20,}`, `secret_[A-Za-z0-9]{43}`, `AIza[0-9A-Za-z_-]{35}`, the UUID regex (with an allow-list file for the documented example UUIDs, if any), and `/Users/`. Then `make secrets-scan`. A test in `packages/core/tests` runs it in a temp git repo with planted files.
- **R3-5 key details:** `make tenant` refuses if ANY `*.key` already exists in `~/.config/ragforge/` (a second key breaks `ask`/`ui`), shows admin.py's stderr except the "Store this key now" line (no `2>/dev/null`), and writes only the `api_key=` value to the file. `make mcp-key` finds the owner key with the same discovery rule (`--owner-key-file` to override). `make mcp` defaults to `--key-file ~/.config/ragforge/mcp/mcp.key` when ARGS is empty.
- **R3-6 eval guide details:** `snapshot create --source-collection <UUID>`, with how to find the id (`curl -H "Authorization: Bearer $(cat ~/.config/ragforge/owner.key)" localhost:8000/v1/collections` and a placeholder in the doc); `--dataset` accepts `demo|private`, so the user's own questions go in `evals/private/questions.toml` (git-ignored).
- **R3-7 owner fresh run:** stop the live stack first (`make down`; ports 8000, 8501, 3000 are fixed), run as a separate macOS user, then `make down` there and `make up` back in the owner's account.
- Note on `curl` with `$(cat key)`: the key appears in the process list for the length of the call. The guide prefers the SDK or `make ask`; where curl is shown, it uses `-H @file` style (`curl -H @<(printf 'Authorization: Bearer %s' "$(cat ...)")` is still a process-list read), so the guide says this is for a single-user machine.

## Integrator notes (2026-10-22)
- The gate is `scripts/security/prepublish.py` (Python, not `.sh`: testable, and it never prints a matched value), run by `make prepublish`, which then runs the gitleaks scan. UUID and home-path rules apply to docs and eval data only; code and test fixtures hold many synthetic UUIDs. The exact check is the live one: real tenant, collection and connector ids and key prefixes, read from the running database, must not appear in any staged file (the all-zero SYSTEM tenant is a schema constant and is skipped).
- The README headline table is generated by the same exporter (`evals/readme-results.toml`); W1's separate README results checker was removed, so there is one mechanism.
- Retrieval tables check the manifest model against the header's embed and rerank models (`models_from = "retrieval"`); a retrieval run records the server's chat model too, which did not produce its numbers.
