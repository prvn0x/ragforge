# Next session: start here

> **Numbers:** [RESULTS.md](RESULTS.md) is the source of truth. Figures in this log are historical (some are v1-eval numbers) and may differ.


## TO DO NEXT (owner-agreed, 2026-10-18)
1. **[DONE 2026-10-19, Phase 3b]** **Interview mode (owner's top priority).**
   - **Problem, seen live:** "ask me interview questions" and "act as a principal engineer interviewing me" went through the Q&A pipeline. They got summarised, a raw code fence was dumped, and "where's the question?" became a new search.
   - The Q&A path has one job by design (answer from the notes, verified). Interviewing needs its own mode, chosen EXPLICITLY (a UI toggle / `/interview <topic>`), not inferred.
   - **Flow:**
     1. pick a topic and retrieve diverse notes;
     2. ask ONE Senior-SDE-1 question grounded in a note;
     3. the user answers;
     4. grade against the notes: right / missed / misconceptions, plus a model answer with citations (verified);
     5. an adaptive next question;
     6. a session summary with weak topics.
   - Shares the LLM-as-judge with item 2.
   - **Owner decisions (2026-10-18):**
     - Start: deterministic phrases ("interview me [on X]", "quiz me", "ask me [interview] questions", `/interview X`) plus a UI toggle. Never inferred by the model.
     - Scope: "on Kafka" means ALL Kafka pages (title/heading match plus retrieval), with broad coverage and no repeats. With no topic, questions mix topics across all the notes.
     - Question types: ALL of concepts, scenarios/production issues (the MNC pages), system design/trade-offs, and code/LLD.
     - Grading at the **Senior SDE-1 bar**: trade-offs, failure modes, the why. A score out of 10, then right / missed / misconceptions, plus a model answer with verified citations.
     - Session commands: repeat ("where's the question?"), skip, harder, easier, next topic, stop (summary of weak topics and pages to revisit). A normal question leaves the mode.
   - **Plus, false "Not verified" flags (owner report, 2026-10-18):** the Memento answer was flagged with `code_unbacked=1`, the verifier did NOT object. The model backticked `createMemento()`, which sits inside a larger code block on the page.
     - (a) Accept a SINGLE identifier token that appears as a whole token inside a cited page's code block. Whole commands still need an exact whole-element match. This is a guarantee change, so it needs a blind review.
     - (b) Split the UI flags: red "Not found in your notes" (verifier / residue) vs a muted, collapsed "Couldn't match exactly" (code/format only). This needs a reason per claim in `VerificationEvent` (an additive contract field).
     - Why: frequent false alarms train the owner to ignore the warning.
2. **Eval layer.** **Design ACCEPTED 2026-10-21: [ADR-0017](adr/0017-eval-layer.md) rev 5. Blind principal reviews: 6.5 → 7.8 → 8.0 → approved. Exporter dropped; local Ollama judge with Gemini Flash fallback; eval tenant + frozen snapshot; D0 demo set.** **Lock step DONE 2026-10-21** (ADR-0017 "Lock step": contracts + SDK, `ragforge_client/evals/schema.py`, D0 pages and datasets, golden fixtures with the stdlib oracle of 60 expected values; `make check`: 4178 passed). Next step: builders B1–B4, then the integrator's live steps (eval tenant, snapshot, A/A run, baselines, the owner's labelling sessions). Original scope:
   1. retrieval metrics (recall@k, MRR, nDCG) from the expected pages;
   2. an LLM-as-judge with a rubric (faithfulness, relevance, completeness against the facts);
   3. **judge calibration against the owner's labels (agreement %, Cohen's kappa)**;
   4. a planted-error test set for the verifier's false-verified and false-reject rates;
   5. an exporter to LangSmith or self-hosted Langfuse (traces per question);
   6. a regression gate (`make eval-gate`, paired test vs a baseline);
   7. an Eval page in the UI.
   - Also: a fresh held-out set (the current one has been used twice).
3. **Phase 4, thin (owner decision 2026-10-22, [ADR-0018](adr/0018-ops-hardening-security-observability.md)):** BUILD the security pass and the observability stack (metrics + dashboard + alerts). PARKED, with reasons in the ADR:
   - CI (until the repo is on GitHub);
   - Helm/kind/KEDA/NetworkPolicies/multi-arch;
   - the full backup/restore drill (`make backup-evals` instead);
   - the load test;
   - the DeepEval comparison.
4. Then Phase 5 (write-up + the owner's git repo).

## Current state (2026-10-22): read this first
- **Phase 5 (write-up) is CLOSED:** README, [GUIDE](GUIDE.md), [RESULTS](RESULTS.md) (generated), [DECISIONS](DECISIONS.md), `make tenant` / `make mcp-key` / `make prepublish`. Final review 8 → re-check **8.5, ship**. Left: the owner's checklist below, the labels, `make models` skipping the unused chat model on Gemini (P2).
- **Before the first push (owner): the pre-publish checklist** (PHASE5 R2-4, R3-4, R3-7).
  1. `make backup-evals`.
  2. Optional but recommended: the fresh-user run. `make down`, then as a separate macOS user, follow the README quickstart word for word; note every place it differs and tell the integrator; then `make down` there and `make up` back in your own account.
  3. `git init && git add -A`, then `make prepublish` (with the stack up, so the live tenant/collection/connector ids and key prefixes are checked too). It reports `path:line  rule`, never the value, then runs the gitleaks scan. Fix or unstage until both are clean; a dry run on a copy on 2026-10-22 gave 0 findings (634 staged files, 19 live ids checked).
  4. CI: `.github/workflows/ci.yml` runs on the first push to `main` (never run before; expect fixes). Delete it before pushing if you do not want public CI yet; otherwise watch the first Actions run, and consider pinning its actions by commit SHA.
  5. Add a LICENSE file (your choice; none exists today).
  6. Look at `git status --ignored` once: `.env`, `evals/private/`, the raw run files and `.idea/` must be listed as ignored.
- **Phase 4 (thin) is CLOSED** ([ADR-0018](adr/0018-ops-hardening-security-observability.md) "Re-review and close"): final review 7.5 → fix round → re-review **8.5, ship**. Live on schema 14 with the obs profile up.
- **Next:** the owner's optional label tasks (50 blind labels, the negation-flip check, rewording `heldout2.draft.toml`), then `d3-build` + the verifier baseline + the full chat gate + judge calibration, then **Phase 5** (write-up).
- **Open P2:** MCP HTTP logs are not redacted (see ADR-0018); the ADR-0017 P2 list.

## State on 2026-10-19
- **Phase 3c (interview levels, general-knowledge fallback, acronym topics, dotted-name false flags) is CLOSED.**
  - **Process:**
    - contracts locked first;
    - a blind DESIGN review: 7.0, 2 P0s. The first: a shape-only dotted-name rule merged "fast.and.it", so the rule is now source-backed. The second: the general-mode token was optional. Both were fixed in the specs before building.
    - 3 builders with blind loops (`docs/workflows/phase3c-interview-levels.js`): IV3-CORE 8 → 8.9, IV3-UI 8.1 → 9, DOT 7.5 → 8 → 9;
    - integrated final review 8.0. Its P1: the integrator's lexical gate was too strict, rejecting hyphens, "vs" and paths; now fixed by normalising both sides.
    - re-review 8.6. Its P1-minor: the loose-label regex ate "(S3)" and "(T2)"; it is narrowed, and the P2s are fixed.
    - `make check`: 4168 passed.
  - **Built:**
    - **Levels** (`InterviewLevel`): SDE-1, SDE-2, Senior SDE-1 (default), Senior SDE-2, Staff.
      - Each has a per-level rubric with 3/6/8/10 anchors; difficulty is one step within a level.
      - The level is fixed per session, bound into token v3, and shown as a dropdown in the UI and as phrases in `make ask`.
    - **General-knowledge fallback** (`source=general`, ADR-0016 amended):
      - offered only after 409 `interview-no-notes` / `interview-exhausted`, and only on a click;
      - the token is required; never verified, no sources;
      - labelled "General knowledge · not verified · graded by the model against its own answer key";
      - not exposed in MCP; scores kept apart in the summary.
    - **Acronym topics:** HLD matches "High Level Design" through title and heading initials, then the topic is expanded for search.
    - **Lexical gate:** a search-only candidate must contain every topic word (`/-_.` read as spaces, filler words dropped). Live: "Kubernetes operators" had returned a Kafka note.
    - **Dotted names:** source-backed, on the chat and interview paths (ADR-0015 "Dotted names").
    - **Question replies:** no more 502 when the expected points lack labels.
  - **Live** on the real notes:
    - HLD, Spring Boot actuator, event-driven architecture and Kafka vs RabbitMQ all give questions from the notes.
    - Kubernetes operators and Terraform give `interview-no-notes`; general mode then gives good questions and grades.
    - The `{logger.name}` answer is verified.
  - **Open (P2):**
    - The topic is not bound in the general token. This is an accepted deviation: `InterviewGradeRequest` has no topic field, and the topic never reaches the grade prompt.
    - The client always sends `senior_sde_1`, so the server's configured default level does not apply to the UI or `make ask` (the client default wins).
    - No chat-path test pins "dotted names never come from tool results" (it holds by construction).
    - The "below the verifier's quality bar" badge also shows for an inconclusive verdict.
    - A bare "(S1)" in a question is left as is (product guard).
    - The acronym expansion searches are uncached.
    - Lowercase 3–6 letter topics may expand.

- **Phase 3b (interview mode + false "Not verified" flags) is CLOSED.** TO DO item 1 is done; next is item 2 (the eval layer: design review FIRST, then build).
  - **Process:**
    - interfaces locked by the integrator;
    - a blind DESIGN review before building: 6.5, 4 P0s, all fixed in the specs; the builders were stopped and relaunched;
    - 3 parallel builders with blind review loops (workflow `docs/workflows/phase3b-interview-build.js`): IV-CORE 8 → 9, FLAG 7.5 → 9, IV-UI 8.2 → 8.7;
    - integrated final review 8.0 (one P1: code questions starved, about 1 in 20; fixed by ranking on the due kind, with a mutation check);
    - re-review 8.6 with no P0/P1, and its P2s fixed.
    - `make check`: 3813 passed.
  - **Built:**
    - `POST /v1/interview/question` and `/grade`: stateless; the session lives in the client; per-principal rate limit; a signed `question_token`; 409 `urn:ragforge:problem:interview-exhausted`; one `interview_model_call` log line per model call, for the eval.
    - Deterministic start phrases (`ragforge_client/interview.py`, including topic-first phrasing like "ask me Kafka interview questions") plus a UI toggle. Topic lists alternate. A normal question SUSPENDS the mode, and it is recoverable.
    - Grading at a configurable level (default Senior SDE-1). Missed points and misconceptions must be backed by the notes; the model answer is verified.
    - One narrow guarantee change (ADR-0015 amended 2026-10-19): a single identifier-shaped name backed by a whole token of a cited chunk's code.
    - A reason per flagged claim, `not_found` (red) vs `inexact_code` (muted, "code shortened from your notes, check it"), in the UI, `make ask` and MCP.
    - answer@11 keeps the label off the closing fence line.
  - **Live on the real notes:**
    - Kafka: questions in 3–5 s, grades in 2–4 s. Good, weak and wrong answers scored 10 / 0 / 0, and the planted misconception was caught and quoted back.
    - Memento:
      - 7 questions covering all 4 kinds, then the 409 exhaustion;
      - the owner's "explain Memento" case is verified 3 of 3 times (it was flagged before).
  - **Open residuals (P2):**
    - ~~Bare dotted config names cut as sentence ends~~ (fixed in Phase 3c).
    - Near-duplicate questions from different chunks of the same page (Memento Q3/Q7, both on encapsulation): no semantic de-duplication yet.
    - Interview prompt tags change the chat `PROMPT_VERSION` (needs a per-surface version before `make eval-gate`).
    - `answer_strict@5` has no code or closing-fence rule.
    - A misconception quote can be very short ("a").
    - No HTTP-level 429 concurrency test.
    - The rate limiter lives in each process.
    - `question_token` is optional (enforce it later).
    - An old-checkpoint resume is covered by unit tests only.

- **Phase 3 is CLOSED.** The final blind re-review scored **8.3/10, no P0/P1**; its P2s are fixed (post-verdict honesty in UI and MCP, clean key-file errors, wiring tests for the Host/Origin checks, DESIGN section 10 and handoffs synced). `make check`: 3422 passed. The UI is running at http://localhost:8501 (`make ui`). MCP is ready (`make mcp-install` prints the command; the owner adds it later).
- **Phase 3 (daily-use surfaces) is BUILT and INTEGRATED**, workflow `docs/workflows/phase3-parallel-build.js`:
  - 2 parallel builders, each with blind reviews: MCP 7.5 → 8.5 → 8.4, UI 8.0 → 8.2 → 8.0.
  - The integrator fixed the last P1 of each:
    - MCP: a closing fence is indented 0–3 spaces only (CommonMark), so a 4-space-indented ``` inside code no longer ends the block and strips backslashes;
    - UI: a per-session Host check (`config.host_refused`) plus `--server.allowedHosts` against DNS rebinding.
  - Applied the handoffs:
    - `make ui` (127.0.0.1:8501, allowedHosts, no telemetry);
    - `make mcp` / `make mcp-install` (prints the `claude mcp add` command; the owner will run it later);
    - the MCP direct dependencies declared;
    - the MCP doc corrected (an MCP-raised approval can't be loaded in the UI until `GET /v1/approvals/{run_id}` exists).
  - `make check`: 3405 passed.
  - **Live, real key/API/notes:**
    - UI (Streamlit AppTest driving the real app): sections, Verified badge, sources with the Notion link, the "Not from your notes" block after them, "Show the code"; first token 3.0 s, total 5.3 s. Bound to 127.0.0.1 only.
    - MCP over stdio: 4 tools (no approve); list_sources, search, ask (verified, cited, extra flagged `not_from_notes`), a follow-up in the same conversation (verified).
  - **Open API gaps** from the handoffs:
    - `GET /v1/approvals/{run_id}`;
    - `GET /v1/documents/{id}` + chunks;
    - `Connector.last_synced_at`;
    - `ApiError.retry_after`;
    - SDK generators closing their inner generators.
- **Phase 3 integrated blind review: 8.0, two P1s, both fixed** (UI: a pre-verdict draft is now shown as plain "Unverified draft … not saved" text, never Markdown; MCP: the approval texts no longer promise an approval the UI can't load).
  - P2s fixed:
    - UI Origin check;
    - fences close on a longer closer;
    - `max_chars`;
    - a 180 s UI turn deadline;
    - MCP keeps the saved answer when the connection drops after the verdict;
    - combined warnings;
    - `display_url` drops the query;
    - UI pyproject deps.
  - `make check`: 3415 passed.
  - UI restarted and re-verified live: Verified badge, sources, extras after them, first token 1.4 s, total 4.1 s, bound to 127.0.0.1.
- **Phase 2c (interview-prep answer shape + labelled extras, ADR-0016) is CLOSED.** Blind reviews: 7.0 → 7.0 → 8.0 → 7.5 → 7.0 → **8.0 final, no P0 or P1**. The r5 P1s were all in an identifier-exactness rule that was REVERTED (ADR-0015 records the measured residual and the proper design). The final review's P2s are fixed (see the Phase 2c section). The build to quote is C7, plus the r6 P2 fixes (confirmation run C8).
- **Phase 2b is CLOSED.** The final blind review scored **8.5/10 with no P0 or P1**: about 150 attacks plus two full Unicode sweeps, all failing closed. Its cheap P2s are fixed (below). The remaining ones are recorded under "Open".
- **Stack:** Gemini `gemini-3.5-flash-lite` (free tier) for chat; local Ollama `nomic-embed-text` for embeddings; schema 11; `make check` green. The tool mode is `actions` (retrieve → answer; tools only for explicit commands).
- **Phase 2b:** blind reviews 7.5 → 7.5 → 8.0 → 8.0 → 8.0 → 8.0 → 7.5 → 8.0 → 7.5, with no P0 in any round. Each round found at least one P1. Most of them came from the code-quote channel: rounds 2 and 4–9, eight P1s across them. Each was fixed with tests and an independent oracle, but the score did not converge.
- **Measured** (one Flash-Lite run each):
  - dev: 100% correct, verified 62–85%, on the round-1 to round-4 builds (B5, B7–B9). Intermediate B6/B6b had provider errors (see below);
  - held-out H2: 85% correct (CI 58–96), p95 4.8 s, on the B5 build;
  - latency on B9 (round-4 build, heavy swap): p50 9.3 s, p95 15.1 s, FAIL.
  - **FINAL build (prose quotes removed), B11: 100% correct (77–100), 0 errors, p50 2.9 s, p95 4.9 s, TTFT 1.7 s, verified 62%.** This is the number to quote for Phase 2b on dev. It is one run at n = 13; B10's 54% → 62% is noise.
  - **Re-measured on the round-7 (single-token) build, B10: 100% correct (77–100), 0 errors, p50 3.1 s, p95 4.7 s, TTFT 1.8 s. Verified 54%.** This is consistent with the B9 slowness being the machine (swap was still about 10 GB during B10, so the machine wasn't quiet either). The verified% drop from B9's 77% is the expected cost of the single-token rule: multi-word backticked spans are now residue. It is one run at n = 13.
- **H2 invented% (owner labels, 13/13).**
  - First pass: 0 invented, 3 unsure. That is ≤ 23% under the pre-registered rule, so it FAILED.
  - Then the owner relabelled all 13 as ok (sidecar timestamps 06:33:21–06:33:48 UTC, 2026-10-18). Asked about it, **the owner states they checked the 3 unsure answers against their cited Notion pages** and every claim was there.
  - **The page titles were not recorded**, as the adjudication rule asked, and the label tool kept no history then. So this pass rests on the owner's attestation, not on an auditable record.
  - Since round 7, `save_labels` keeps an append-only `history` of verdict changes.
  - `compare` now reports H2 invented% 0, meets bar.
  - **Adjudication rule, written before any unsure is resolved (2026-10-18):** an "unsure" may change only by opening the page the answer cites and checking each sentence against it. If every claim is on the page, it becomes ok; if any is not, invented. Record the page title for each one in this file. No other relabelling.
  - The verifier's false-verified rate: no labelled answer was invented, but **dev runs C4 and C5 each measured one false "verified"**: the same identifier typo (`exactly_one_v2`), which the model repeats. It is disclosed in ADR-0015 as a residual.
- **Prose-quote rule REMOVED (owner decision, 2026-10-18, after round 9).** It was first narrowed to single tokens after round 7. Code in an answer is now backed only by a whole code element of the cited page. Rationale: it was the channel for most of the P1s, and its measured verified% benefit was within noise at n = 13. The answer-side `joined_code` check stays, since it protects real code spans too.
- **Final review P2s:**
  - fixed:
    - blank glyphs written as entities: whole-answer checks run on both raw and decoded text;
    - the Gemini SSE splitter is linear (a perf test: an 8 MiB endless line is rejected in < 2 s);
    - stale docstrings and docs;
    - `answer_strict@5` pins the exact refusal sentence;
  - open (deliberate):
    - **the inferred-citation path is unused on the shipped config** (B11/H2: inferred 0%). It is about 1,000 lines. Measure it on an Ollama arm or remove it in a later cleanup;
    - long-tail zero-width or tofu characters (U+1D159, U+16FE4, U+034F, U+17B4) between a look-alike terminator and the next word;
    - for a new bar claim on the final build: a fresh held-out set, labelled with page titles recorded, plus the verifier's false-verified rate.
- Latency: B10/B11 meet p95 < 10 s (4.7 s and 4.9 s) even with about 10–11 GB of swap, so swap does not drive latency on the Gemini path. The B9 slowness was provider or machine variance.
- **Open:**
  - the bar on invented%;
  - list lead-ins are the main `partial` cause;
  - P2s listed per round.

## Phase 2c: interview-prep answer format (2026-10-18): BUILT, see ADR-0016
- **Prompts:**
  - `answer@10`: sections In simple terms / Technically / In production (notes only), bold names inline, code only when asked, and a firmer no-backticks-for-plain-text rule.
  - `extras_incident@5`: one "Common failure pattern:" line, never a company name, NONE for non-engineering topics; question and answer passed as fenced data.
  - `extras_code@2`.
- **Extras node:** after `finalize`, routed only when needed. `TurnExtra` → SSE `extra`, rendered LITERALLY by the API (`sanitize.literal` / `sanitize.fenced`), never committed. It is dropped when off-format, NONE or empty, and only the first paragraph is kept.
- **Live finding:** with "name a company only if publicly attributed", Flash-Lite invented a company outage. So extras are failure patterns only, and real MNC incidents come only from the notes (the incident pages), verified.
- **`make ask`:**
  - `**bold**` is rendered (ANSI on a TTY, private-use markers the server text cannot inject);
  - extras are printed under "── Not from your notes: … ──";
  - the footer offers `→ ask "show the code"`.
- **C1 (dev):** 100% correct (77–100), **verified 77%**, p50 4.5 s, p95 6.0 s, TTFT 1.9 s, 0 errors. p50/p95 run to `done`, so they include the extra. The extra's own time is not recorded separately (the about-1.5 s figure is an estimate from p50 differences, not a measurement).
- **Phase 2c blind review r1: 7.0, three P1s, all fixed:**
  - a refusal got an extra: now never on refusals or uncited answers;
  - the `make ask` footer printed under the extra: extras now come last, in a closed block;
  - the code trigger fired on "give me an example of X": code now needs the word code, snippet or implementation, with a negative-case table.
  - Plus the P2s:
    - server-side label and part-name stripping, a robust first-paragraph cut, caps;
    - broad failure isolation (except lease loss);
    - the question and answer passed as data in the user message;
    - `**In production:**` anchored;
    - a generic label for unknown kinds, no code offer after code;
    - extras recorded in eval `Outcome.extras`;
    - a real-sanitizer test, a second-turn "not committed" test;
    - the dead escape phrase removed, and docs/version refs fixed.
- **C2 (dev, after the r1 fixes): 85% correct (58–96), verified 83%, p50 3.8 s, p95 4.5 s, 0 errors; extras on 8/15 turns.**
  - The two misses are model variance on the unchanged answer path. a Kafka question: a correct draft that the verifier rejected (a false reject, fail-safe abstain). a system-design question: 1 of 2 fact keywords.
  - C1→C2 is +0/−2, McNemar p = 0.50.
- **Phase 2c blind review r2: 7.0** (P1s: a spoofable extras block, and no code extra after a refusal). Fixed structurally:
  - `make ask` prefixes every extra line with "│ ";
  - incident extras are one line on the server; control and bidi characters are stripped;
  - code extras keep the code verbatim (no label stripping inside fences), capped and re-closed;
  - the label and part-name regexes are wider;
  - `answer_already_shown` and `question` are fence tags;
  - the code trigger is clause-bounded, with more forms and a negative table;
  - inferred provenance counts as cited;
  - a code extra is made after a refusal when code was asked for;
  - tests for LeaseLost, usage in TurnDone, spoofing and the fences;
  - DESIGN.md SSE contract and graph synced.
- **2 code-request questions added to the dev set** (a producer code question, a streams code question).
- **C3 (dev, 17 questions): 100% correct (80–100), verified 60%, p50 4.2 s, p95 6.9 s, TTFT 1.6 s, 0 errors.** Extras on 11/17, including 1 code extra (a streams code question: verified answer, no code in the notes, so the code came as an extra). a producer code question copied the config from the notes. verified% 83 → 60 is within noise at n = 15.
  - **Correction: C3 ran WITHOUT a rebuild, so it measured the r1 build** (the r2 server changes were not deployed). Kept for the record; use C4.
- **C4 (dev, 17 questions, the actual r2 build): 93% correct (70–99), verified 67%, p50 3.8 s, p95 5.0 s, TTFT 1.7 s, 0 errors. Extras on 9/17, 1 code extra.**
  - The one miss is a **verifier false "verified"**, the first one measured. a streams code question answered `processing.guarantee=exactly_one_v2`, a typo for `exactly_once_v2`, and the ground check passed it.
  - This is ADR-0015's disclosed verifier-dependent residual. The lexical layer doesn't compare identifiers character by character inside a labelled sentence; it can't, since that is the paraphrase room labels need.
  - The code extra showed the right setting.
  - Follow-up candidate: an exact-match check for identifier-like tokens (dotted, snake_case) in labelled sentences against the cited page.
- **Phase 2c final review r4: 7.5, one P1:** the r3 "remove emphasis" step deleted every `_`/`*`, corrupting identifiers (`max_poll_interval_ms`).
  - Fixed structurally: the node only EXTRACTS (incident line; code via markdown-it's CommonMark fence), and the API renders extras LITERALLY (`sanitize.literal`: all Markdown escaped, URLs defanged; `sanitize.fenced`: a fence longer than any inner backtick run).
  - `make ask` wraps by display width.
  - The code trigger is case-aware, with a large positive/negative table.
  - `extras_incident@5` adds a NONE reply.
- **C5 (rebuilt, after r4 fixes): 87% correct (62–96), verified 80%, p95 6.1 s, 0 errors.** The `exactly_one_v2` false "verified" repeated: a systematic model typo; the notes say `exactly_once_v2`, 11 chunks, 0 with the typo.
  - So we added identifier exactness, later REVERTED after review r5 (see below).
- **C6 (rebuilt; WITH the identifier rule that was later reverted): 100% correct (80–100), verified 80%, p50 3.9 s, p95 5.8 s, TTFT 1.5 s, 0 errors. Extras on 8 turns (1 code), 0 naming a known company.** a streams code question was correct and verified this time (the model did not repeat the typo).
- **Final review r5: 7.0.** The extras scored 8.0; **the identifier-exactness rule scored 5.0, with five P1s**:
  - it was checked per unit after segmentation cut dotted names;
  - it used substring matches;
  - it read raw Markdown, so escaped underscores hid names;
  - the `=` sentence-end change reopened the sentence-merge class through the TLD exemption;
  - it broke tool-cited replies.
  - **Decision: REVERTED** (the rule and the `=` change). It was added mid-cycle without a design, and a broken guarantee rule is worse than a disclosed residual. ADR-0015 records the measured typo residual and the proper design as a follow-up.
  - The r5 extras P2s are fixed:
    - code-trigger over-fires ("write a summary … using", "how does Kafka implement … in Java clients?", "codes of the …", "dead code");
    - empty fences and NONE after the opener produce no extra;
    - U+2028 no longer shifts the fence map;
    - a `~~~` fence in the answer counts as code;
    - "In production" is detected right after a label;
    - the client's code offer accounts for a revised answer;
    - stale comments and docs fixed.
  - (Older r3 notes below mention emphasis/part-name stripping: r4 replaced that with extract-only plus literal rendering.)
- **Final review r6: 8.0, no P0/P1. The phase is closed.** P2s fixed:
  - citation look-alikes folded again (NFKC, Cf; `[1]`, `[S1 & S2]`);
  - `sanitize.literal` / `fenced` clean their own input (controls, bidi, format characters, the language tag);
  - an 8 s budget on the extras call (`EXTRAS_TIMEOUT_S`);
  - form feeds and line separators in code become spaces, never deletions;
  - code-trigger cases in both directions (source code, "code of the …", provide … implementation, "Code for …", see a Java example; NOT code path, code-level, Hamming/Huffman/error-correcting codes, haiku/blog);
  - the client checks the whole streamed answer for a fence;
  - eval extras are stored unescaped;
  - the contract documents `extra.text` as CommonMark.
  - Open, deliberate:
    - releasing the lease before extras;
    - an explicit `/code` command (the client's "show the code" offer is the primary path today);
    - identifier exactness, done properly (ADR-0015).
- **C8 (confirmation run, rebuilt with the r6 fixes): 93% correct (70–99), verified 67%, p50 3.9 s, p95 28.3 s, 0 errors; extras on 10 turns (1 code), none naming a company, stored unescaped.**
  - The p95 is a **cold start**. The first two questions after the rebuild had TTFT 7.8 s and 22.2 s (retrieval/rerank/embedder warm-up, before any extras code; a Kafka question had no extra), and the rest had TTFT about 1.6 s.
  - The miss is a system-design question' usual 1-of-2 fact match.
  - Follow-up: a warm-up call in `make up` (one embed and one rerank) so the first question isn't the cold one.
- **C7 (rebuilt; identifier rule reverted, r5 extras fixes): 100% correct (80–100), verified 80%, p50 3.9 s, p95 5.1 s, TTFT 1.6 s, 0 errors.** This is the build to quote for Phase 2c.
- **Phase 2c blind review r3: 8.0, no P0/P1.** Its P2s are fixed:
  - look-alike labels folded (NFKC, Cf, blank glyphs); this protection was lost in r4's rewrite and restored after r6 (`_without_labels`);
  - emphasis removed from extras prose;
  - the code trigger handles status/error/QR/Morse "codes" and "example code"/"in Java";
  - the fence info string is reduced to a language tag;
  - a bare failure-pattern prefix is dropped;
  - cuts happen at a sentence or word;
  - "In production" is detected in any emphasis;
  - `make ask` unescapes outside code fences only and wraps extras to the terminal width under the "│ " prefix;
  - the lease held during extras is disclosed (ADR-0016, DESIGN "wait for done");
  - tests incl. real sanitizer → render_turn → format_footer.

## (historical) Phase 2c plan as requested (owner request, 2026-10-18)
- **Done already:** `make ask` showed the server's Markdown escapes (`\[S1\]`). The `TerminalText` streaming unescaper in `ask.py` fixes this. It is display only, code spans are kept literal, and `clean` still runs after it.
- **Answer shape (owner decisions):**
  1. **In simple terms:** 2–3 plain sentences.
  2. **Technically:** the precise mechanism, configs and trade-offs.
  3. **In production:** one real MNC incident (broke → why → fix), 3–4 lines.
  4. A follow-up offer: "Show the code?".
- **Sources: "notes first, labelled extras".**
  - Sections 1–3 come from the owner's notes (the incident pages) and keep the existing guarantee unchanged.
  - When the notes have no incident, or no code on request, a SEPARATE extras step generates one. It reaches the client as its own additive SSE event, clearly labelled "Not from your notes", and is never part of the verified answer. That way the model cannot route claims around verification.
- **Code: a follow-up, not inline.** "Show the code" returns a focused snippet, from the notes when they have one, otherwise labelled extras.
- **Work:**
  - ADR-0016 (extras channel);
  - answer@9 (sections, crisp, the follow-up line);
  - the extras node and a contract event;
  - `make ask` rendering;
  - eval re-measure: correctness, latency, verified%. The format change affects grading and the free-tier request count, since extras cost one more call.
- **Estimate:** 2–3 h.

## Phase 2b handoff (2026-10-17, end of session; historical, superseded by "Current state")
Nothing is running. No eval in progress. The last run (A1) finished; A2 was NOT started.

**Done (plan steps 5 and 1, part of 3):**
- **`make ask`**: `packages/client/src/ragforge_client/ask.py`, script `ragforge-ask`, Makefile target `ask` (`ARGS=` passthrough).
  - Key: `--key-file` > `$RAGFORGE_KEY_FILE` > the single `~/.config/ragforge/*.key` (refuses several, or group/other-readable). Never printed.
  - Strips C0/C1 control chars from model text (terminal-escape injection). `/new`, `/quit`; footer shows citations, verdict, first-token and total time.
  - Verified live: two-turn chat against the owner's tenant, follow-up resolved.
- **Eval set**: `evals/private/questions.toml` (gitignored via `.gitignore` → `evals/private/`). 30 questions: 26 answerable (the owner's interview-prep topics) + 4 must-abstain (2 near-miss). Split 15 dev / 15 heldout. Expected pages/facts were grounded with Postgres FTS over `rf.chunks` (not the retriever); every fact was string-checked against the notes.
  - **Owner has NOT reviewed the facts yet.** Re-score after edits; no re-run needed.
- **Runner/grader**: `packages/client/src/ragforge_client/evalrun.py`, script `ragforge-eval`, target `make eval-private ARGS="run --label <arm> [--split dev|heldout|all]"` / `ARGS="score [-v] <run.jsonl>..."`. Raw outcomes → `evals/private/runs/<ts>-<label>.jsonl`.
  - correct = answered AND cites an expected page AND ≥ ceil(2/3) facts matched. Also reports content% (no citation needed), draft% (pre-verifier draft), cited%, abstain%, verifier-flagged% (proxy only), p50/p95, ttft, tool turns, model (from `DoneEvent.usage`).
- **Tests**: `packages/core/tests/test_client_ask.py`, `test_client_evalrun.py` (16 pass). ruff/format, mypy strict and lint-imports clean on touched paths. **Full `make check` NOT run this session** (deferred so it wouldn't skew eval latency): run it first.
- **compose.yaml**: api now passes `RAGFORGE_AGENT_MAX_TOOL_ROUNDS` (default 3), `RAGFORGE_AGENT_STRICT_CITATIONS` (false), `RAGFORGE_AGENT_VERIFIER_TRUSTED` (true); defaults mirror `config.py`.
- **`.env` currently has `RAGFORGE_AGENT_MAX_TOOL_ROUNDS=0`** (arm A1, appended at the end with a comment). The running api uses it. Keep it for further arms, or remove it and `docker compose up -d --no-deps --wait api` to restore defaults.

**Results (dev split, n=15: 13 answerable + 2 abstain; p95 at n=15 = slowest answer):**

| arm | correct | content | draft | cited | abstain ok | p50 | p95 | ttft | tool turns |
|---|---|---|---|---|---|---|---|---|---|
| A0 llama3.2:3b, tools on (default) | 15% | 54% | 77% | 15% | 2/2 | 22.0 s | 52.8 s | 8.5 s | 13 |
| A1 llama3.2:3b, `max_tool_rounds=0` | **46%** | 46% | 69% | 46% | 2/2 | 12.7 s | 22.0 s | 3.2 s | 0 |

**Findings:**
- `max_tool_rounds=0` already IS retrieve→answer (retrieval runs before `generate`; tools are unbound at 0). No code change needed to measure it. ADR (plan step 2) is supported by A0→A1 but not yet written; the open design question is how explicit actions (remove page, remember) get tools without binding them on every turn.
- Dominant failure: drafts with correct facts but **zero `[S<n>]` citations** → `verify_citations` fails with no claims → the strict regeneration also fails → abstain. E.g. A0 cassandra/livelock drafts were right and discarded. So plan step 4 (citation prompt/format for small models) is the biggest lever, alongside a stronger model.
- Invented content seen in drafts: CHM answer described Java 7 segments as Java 8; Cassandra "RF = Read Failure". Keyword grading can't catch contradictions: the ≤2% invented bar needs human labels (not yet designed).
- Grader fix already applied: CHM fact matched `"segment"` (credited the wrong answer); now matches removed/single-array wording. A0 draft% 85→77 after re-score.
- Not yet run: A2 qwen3:4b, A3 qwen3:8b, hosted upper bound, any heldout scoring.

**Machine state:**
- Ollama has only `llama3.2:3b` + `nomic-embed-text` (2.1 GB). **qwen3:4b and qwen3:8b are not pulled.**
- Memory is tight: ~13 GB swap used, ~22% free. Rancher's built-in Kubernetes is enabled and not needed until Phase 4 (owner may disable). Unload llama3.2 (`ollama stop llama3.2:3b`) before loading another chat model. Watch swap for 8b.
- Docker cleanup (owner-approved): removed 80 dangling images and all build cache (16.5 GB inside the VM); the next `make up` rebuilds uncached. Named images from other projects were left alone.
- The owner wants plain-language status updates (what's running, what it's for).

**Next, in order:** `make check` → owner reviews `questions.toml` facts → A2 qwen3:4b (no tools) → citation prompt fix, re-measure → A3 only if needed and memory allows → ADR → heldout run on the chosen setup → blind review. All arms are sequential local runs with no plan-token cost.

### Gemini arms (2026-10-17, later)
- **gemini-3.5-flash free tier = 5 RPM and 20 requests/day**: B1 hit the daily cap (23% correct, 11 errors; the answered ones were correct and cited). Not usable for daily use without billing.
- **gemini-3.5-flash-lite rejected `thinkingBudget: 0`** with Google's bare, detail-less "Request contains an invalid argument." (budget 1/128/-1 and no thinkingConfig are all accepted, with 0 thought tokens). Fix: `chat_gemini.py` treats that exact bare message (no details) as a thinking rejection, so the existing one-retry-without-thinkingConfig fallback applies. The budget is still only dropped after an accepted retry. 2 new tests in `test_g1_chat_gemini.py`; module G1 212 pass.
- **B2b gemini-3.5-flash-lite (dev, paced 13 s): 77% correct, content 85%, cited 85%, abstain 2/2, 0 errors, p50 3.0 s, p95 4.2 s, ttft 1.6 s.** Misses the 80% bar by one question. The misses: a Java concurrency question (a thin answer: no CAS or synchronized), a hashing question (cited a Cassandra page, not the Consistent Hashing page), a Redis question (correct noeviction, but cited a Redis incidents page, which is not in the expected pages). Answers oddly start with "Yes," (look at the rewrite/answer prompt). Citations come back markdown-escaped `\[S1\]`, and the parser accepts them.
- The owner approved questions.toml (2026-10-17). a Redis incidents page was added to a Redis question's pages (B2b re-scores to 85%).
- **Fixes after B2b:**
  1. **`answer@7`:** the only example sentence in answer@6 began "Yes, the bridge…", and Flash-Lite started every answer with "Yes,". The yes/no rule now applies only to yes/no questions. B3b has 0 such answers.
  2. **Migration 0010 (`rf.search_keywords`):** the query is now the OR of its own lexemes (at most 32, quoted, no re-parse) ranked by `ts_rank_cd`, instead of `websearch_to_tsquery`'s AND. The AND form missed the vnode chunk because it lacks the word "use". `CODE_REQUIRED_SCHEMA = 10`.
  3. **Observability:** `chat_turn_failed` warning log in `apps/api/chat_stream.py` (error type, status and title; no detail, since adapter detail can carry upstream text). evalrun now records `problem.detail`. B3's first 4 turns failed after a 200 with no trace in the logs.
- **Results:**

| arm | split | correct | cited | abstain ok | err | p50 | p95 |
|---|---|---|---|---|---|---|---|
| B3b flash-lite, answer@7 + 0010 | dev | **100%** | 100% | 2/2 | 0 | 3.2 s | 4.3 s |
| H1 flash-lite, same build | **heldout** | **77%** | 92% | 2/2 | 0 | 3.4 s | 4.2 s |

  Dev was tuned against, so it is optimistic; heldout is the honest number and misses the 80% bar by one question. invented% is still unlabelled (`ragforge-eval label`). B3, the first run on a cold stack, had 4 provider errors on turns that returned 200; the rerun had none.

### Phase 2b review round 1 (blind, 7.5/10, no P0) and fixes, 2026-10-17
**The reviewer's P1s:**
1. The eval graded the model's own refusal ("Your notes don't cover this.") as an invented answer. Now `ABSTAIN_PHRASES`, with a drift test against the templates.
2. `make ask` didn't mark inferred citations. Now it shows "(inferred)" and also strips bidi controls.
3. Results hid the `partial` rate. `compare` now has verified%, and the api logs `answer_settled` counts (verifier claims, residue, code_unbacked; no text).
4. The ADR-0015 effect claim was overstated. It's corrected: McNemar p ≈ 0.22, latency is the real effect. `compare` now prints a Wilson 95% CI and a paired McNemar against the first arm.
5. invented% is unlabelled. **Still open: the owner labels H2** (below).

**P2s fixed:**
- bidi as HTML entities;
- look-alike full stops (ꓸ ·) end a sentence;
- fact keywords match whole words (+ inflections);
- **0011:** keyword rank = distinct query terms matched + `d/(1+d)`. A chunk repeating "node" 8× no longer beats one covering every term;
- `chat_turn_failed` logs no detail;
- `gemini_thinking_config_dropped` logged once;
- setup comment corrected.

**Left as P2:**
- setup does no generateContent probe;
- a failed bare-400 retry is paid again on the next bare 400;
- the ADR status line is still a long amendment history (a "Current state" summary was added at the top).

**Partial-rate root cause, from the `answer_settled` logs:** in B4, every non-verified answer was `code_unbacked`, and the verifier flagged nothing. The model puts setting names (`replica.lag.time.max.ms`, `noeviction`) in backticks while the notes state them in prose.
- Fix (ADR-0015 current state): an inline span that the cited page states verbatim in prose, as whole tokens, is covered like the same words without backticks. Code inside page code blocks keeps the whole-element rule.
- Tests cover both sides, and the property-test oracle got its own implementation of the rule.

**Round 2 (blind, 7.5, one P1):** the prose-quote rule joined the text around dropped atoms (code, links, URLs, images, HTML, emphasis), so `rm -rf /` quoted "rm -rf /`var/cache`". Fixed fail-closed: dropped atoms become U+FFFC, and a quote must be bounded by whitespace, start or end (only opening or closing quotes/brackets and `.,;:!?` may sit between). The oracle has an independent reading. Also: dot look-alikes are a Unicode-category rule; an RTL letter plus a digit on one line makes the answer residue; abstain detection uses the whole text with apostrophes folded. (The round-2 note claimed the `make ask` footer showed `[S<n>]`. It did not: it printed `Citation.anchor`, a Notion block id, and the test hid this with fake anchors. Fixed in round 3.) Left as P2: setup writes budget 0 for flash-lite (one logged 400 per process start); 0011 has no IDF.

**Round 3 (blind, 8.0, one P1):**
- `make ask` footer: new additive `label` field on `TurnCitation` and the contracts `Citation`, so the footer shows the answer's `[S3]`. Tests use real block-id anchors, with an end-to-end SSE assertion.
- Dot-like ends now see through ZWSP, word joiner, soft hyphen and closing emphasis.
- Heading Markdown is parsed like the body.
- Any RTL character makes the answer residue.
- Emphasis delimiters are transparent in prose quotes, and struck text is hidden. The property-test pages gained an S6 "hidden elements" page; the oracle models it independently.
- Fact digits ignore `[S<n>]` labels and decimals. Abstain accepts the refusal variants `refusal_only` allows.
- Setup resets its own -1 when leaving a Pro model. `ragforge-eval` explains a missing question set.
- All earlier runs re-scored with the new grader: the numbers are unchanged.
- **B8 (round-3 build, dev): 100% correct (77–100), 0 errors, p95 4.6 s, verified 62%** (B7: 77%; McNemar on correctness +0/−0).
  - The verified% drop is *consistent with* model variance (one run per arm, n = 13; B7's answers were not re-settled under the round-3 code): the same 3 questions were `partial` in B7 (a Kafka question, a Redis question, a system-design question).
  - The 2 new ones are uncited list lead-ins ("…in three cases:") that answer@8 asks the model to label. None of the round-3 changes touch that path.
  - Lead-ins are now the main recall loss on dev. The next lever is a prompt or strict-regeneration change, not loosening coverage.

**Round 4 (blind, 8.0, one P1):**
- Two answer code spans that touch, or are joined by ZWSP, an entity, emphasis or a hard break, rendered as one command although each passed alone. `attribution.joined_code` now applies the fail-closed `_bounded` rule on the answer side, and the oracle has its own `_oracle_span_alone`. The generator emits joined spans.
- Boundaries are visible spaces only (`_gap`: ASCII space/tab/LF, Zs), not `isspace`.
- RTL characters written as entities count as RTL.
- Emoji and symbol ?/!/ellipsis/therefore end sentences.
- `label` is a `PositiveInt`. Inferred footer tags are `[S<n>*]`, with a one-line legend.
- **B9 (round-4 build, dev): 100% correct, verified 77%, 0 errors, but p50 9.3 s / p95 15.1 s** (B8: 4.0 / 4.6).
  - TTFT also doubled (median 4.7 s per `compare` vs 2.0 s). TTFT is retrieval plus the model's first chunk, before any coverage code runs, and nothing in round 4 touches that path.
  - No retries or 429s in the api log. A direct Gemini call varied 1.3–3.6 s.
  - The Mac had 11.2 of 12.3 GB swap in use with almost no free pages.
  - Read: environment (memory pressure plus provider variance), not a code regression. Re-measure latency on a quiet machine before quoting p95.

**Round 5 (blind, 8.0, one P1):**
- Notion plain text isn't Markdown-escaped, so literal `*`/`_` parsed as emphasis. Dropping the delimiters let `/srv/data` quote "/srv/_data_".
  - Fix: delimiters are kept as `_EMPHASIS` markers. No quote may contain one, and `_bounded` steps over them only at a quote's edges.
  - The oracle models this independently. The globs, dunders and paths from the review are tests.
- Letters and digits posing as terminators (ǃ ʔ ٠ ۰ ...) end a sentence by rule (`_foreign_tail`), not by list.
- Arabic-Indic digits (bidi AN) count as RTL.
- The `label` contract comment now says inferred labels are not in the text.

**Round 6 (blind, 8.0, one P1):**
- `residue` keys survivors by text, so a joined-code unit (decided by position) with a clean twin of the same text dropped out.
  - Fix: `settle` adds every `attribution.uncited` unit to the residue. Uncited units can never be covered later, so this only fails closed.
- `_foreign_tail` is now a rule: any non-ASCII, non-cased, visible character before a space, unless it continues a non-ASCII word. Free-standing, only letters and digits. It's tested by a scan over Unicode, not a list. The CJK "不" residual is now blocked (pinned).
- A failed bare-400 retry stops further bare-400 guessing (`gemini_bare_400_not_thinking`).
- The invented% headline is the conservative number.

**Round 7 (blind, 7.5, three P1s):**
- A struck table cell with `\|` became an escape atom and leaked into prose. Fixed by the narrowing above.
- `_trust_bare` was cleared by any retry failure (a timeout would disable the fallback for good). Now it is cleared only on a definitive 400 (`ModelRequestRejected`), with tests for ConnectTimeout, 500 and 503.
- The H2 label discrepancy: see "Current state".
- P2s fixed: HTML-wrapped text and line breaks no longer read as prose.
- The oracle now also matches a kept unit without its trailing terminators: production may cut before a " !" that the oracle's own cut keeps.

**Round 8 (blind, 8.0, one P1):**
- A code span ending in a terminator (`` `2017.` ``) hid the sentence end. Fixed: `_view` cuts after it, `_quotes_prose` rejects a span ending in one, and the oracle cuts after such a code mark.
- `_` emphasis delimiters are no longer stepped over: `__init__` no longer quotes `init`.
- The bare-400 trust needs 2 consecutive definitive rejections (`BARE_MISSES_TO_DISTRUST`), and an accepted retry resets it.
- Thin Zs spaces (U+2006/2009/200A/202F) are not gaps.
- Label history: refuses a malformed record, stores both answer digests, keeps the reveal stamp.
- ADR example corrected: my round-7 edit of that sentence had silently not applied.
- The oracle treats trailing symbols ("partitions.™") as stating nothing.

**Round 9 (blind, 7.5, one P1):** a code span ending in a terminator plus closers (`` `1850.)` ``) hid the sentence end.
- **Owner decision: remove the prose-quote exception entirely** (see "Current state").
- Also fixed:
  - `_ends_inside` steps back over closers, emphasis marks and Cf before testing for a terminator;
  - blank glyphs (U+2800 and the Hangul fillers) make the answer residue;
  - the label history also logs a changed answer with the same verdict;
  - the bare-400 miss count resets on any accepted response;
  - ADR §3 now matches how `(source N)` is normalised.

**answer@8:** every line needs a label, including list lead-ins; backticks only for code copied from a code source.

| arm | split | correct (95% CI) | verified% | cited | err | p50 | p95 |
|---|---|---|---|---|---|---|---|
| B3b answer@7 + 0010 | dev | 100 (77–100) | 38 | 100 | 0 | 3.2 s | 4.3 s |
| B4 answer@8 + 0011 | dev | 85 (58–96) | 67 | 92 | 1 (provider timeout) | 3.6 s | 18.6 s |
| B5 + inline prose quotes | dev | 100 (77–100) | **85** | 100 | 0 | 4.0 s | 7.7 s |
| H1 (before the round-1 fixes) | heldout | 77 (50–92) | 58 | 92 | 0 | 3.4 s | 4.2 s |
| **H2 frozen (B5 build), run once** | **heldout** | **85 (58–96)** | **77** | **100** | 0 | 3.9 s | 4.8 s |

After round 2 (dev only; held-out NOT rerun, it has been used twice):
| B6b fail-closed prose quotes | dev | 77 (50–92) | 70 | 77 | **3 provider connect timeouts/errors** | 4.2 s | 14.4 s |
| **B7 + httpx connect retries (2)** | dev | **100 (77–100)** | **77** | 100 | 0 | 3.0 s | 4.5 s |
| B6 same build, first try (see note) | dev | 85 (58–96) | 82 | 85 | 2 provider errors | 4.5 s | 47.4 s |
B6b's misses were all `model provider timed out` at about 5.5 s (the 5 s connect timeout) or `unreachable`: TLS connects from the container to Google failed. The Gemini client now uses `httpx.AsyncHTTPTransport(retries=2)`, which retries connection establishment only, so nothing is sent twice. The fail-closed rule's effect on verified% is NOT measured: B5 85% vs B7 77% is one question at n = 13 (and B6, same rule, scored 82%), well inside noise. Note on B6: a `make up` failed right before it and passed on rerun; the api was later confirmed to run the new code, but B6 ran before that check, so it's reported here and not used for comparisons.

H2 is the best number available, with three caveats. (1) The held-out split has now been used **twice** (H1, then H2 after H1 missed by one question), so it is no longer untouched: any further change needs a fresh held-out set. (2) H1→H2 is +1/−0, McNemar p = 1.00: not significant. (3) 85% is 11/13 with a 95% CI of 58–96%, one question over the bar. It meets correct ≥ 80% and p95 < 10 s as a point estimate. **H2's verified% (77) was measured on the B5 build, i.e. with the round-2 prose-join bug that round 2 fixed; its correctness numbers do not depend on that rule. Labels: see "Current state" (13/13, 0 invented, 3 unsure: the bar fails as it stands).** (in a separate terminal; the run stays the same, labels go in a sidecar):
`make eval-private ARGS="label --reveal <H2 run file>"`. Then measure the verifier from those labels: count `invented` answers that were badged `verified`. That decides `verifier_trusted`.

---

**State (2026-10-16):** Phase 0 is approved (8.5). Phase 1 is complete (8 modules at 8–9). **Phase 2a integration is wired, running and reviewed:**
- `make check` is green: 1134 tests, mypy strict, 6 import contracts (re-verified independently).
- `make nuke && make up` is green at schema 9 (0009: documents.deleted_reason).
- First use works end to end: tenant → collection → upload → ingest → query → chat.
- Blind review: 6/5 → 7/7 → 7.5/7.5 → 7.5/7.5 → 7.5/8 → 8/7.5 → **8/8, with no P0 or P1 (approved)**. History and the P2 backlog are in `docs/handoff/PHASE2A.md`.

**Next:**
1. Optional hardening: the round-7 P2 backlog in `PHASE2A.md`. Highest priority: 0008's lock order under live workers, liveness when every slot is held, and the 0009 backfill.
2. **Notion is connected and fully indexed** (the owner's tenant, one Notion connector, 152 pages). See the 2026-10-17 section in `PHASE2A.md` for the two sync bugs fixed.
3. **Phase 2b, with the model quality problem first.** llama3.2:3b misused a tool on the first draft in 4 of 5 real turns:
   - it sent its own JSON schema as an argument;
   - it passed `get_page` the label `"[S1]"` instead of an id.
   Every time the bad arguments were rejected and the server abstained, which is safe but leaves the user without an answer. Build the eval corpus and verifier set first, then measure:
   - (a) no tools on the first draft for small models (`agent_max_tool_rounds=0` or a first-draft-only switch);
   - (b) a stronger local chat model;
   - (c) `agent_verifier_trusted=false` (ADR-0012).
   Also measure latency: 10–25 s per turn on an M1 Pro.
4. Open gaps from Phase 2a (`PHASE2A.md`, "Known gaps"):
   - blob reaper;
   - tenant-wide exclusions;
   - `chat_active` throttle;
   - `/docs` public;
   - `summarize_conversation` is a no-op.

**Rules:**
- No git until the project is finished; the user will do it.
- The machine is an M1 Pro with 16 GB: Rancher (5 GB VM) plus host Ollama with `llama3.2:3b` and `nomic-embed-text`.

## Plan agreed with the owner (2026-10-17): supersedes "Next" above where they conflict
The owner wants to USE RAGForge to prepare for interviews (Q&A over their Notion notes) AND keep building phases.

**Phase 2b: make answers good and measurable (sequential, normal cost)**
1. **Success bar, defined before any experiment:**
   - ≥ 80% correct with citations on ~30 questions (the 12 smoke questions plus ~18 more, each with an expected page and key facts; half held out);
   - p95 under 10 s;
   - ≤ 2% invented content.
2. **ADR:** remove tools from the answer path (retrieve → answer). Tools only for explicit actions (remove page, remember). Measure it.
3. **Arms, one change at a time:**
   - retrieve→answer on llama3.2:3b;
   - the same on qwen3:4b (thinking off);
   - qwen3:8b, recording RAM and swap (16 GB machine with apps open);
   - optionally a hosted model as an upper bound (the owner provides a key).
4. **Citations:**
   - First fix the prompt and format for small models.
   - Server-inferred citations must be LABELLED inferred and pass the verifier, or be dropped. Never show unsupported provenance.
5. **`make ask`:** an interactive terminal chat over the SDK that reads the key from `~/.config/ragforge/*.key` (never printed). This is the owner's first daily-use interface.
6. Report a table per arm, recommend the cheapest setup that meets the bar, get a blind review, and keep `make check` green.

**Phase 3: daily-use surfaces.** Streamlit UI (chat with citation badges, sync status, approvals) and an MCP server, so the owner can ask about their notes from inside Claude Code. 2–3 parallel builders; state the cost first.

**Phase 4: Advanced** (Helm/kind, dashboards, load test, restore drill). **Phase 5: write-up** (README, demo, measured tables, blog).

**Interview-prep loop (every phase):** the owner reviews the eval set's expected answers. These are their own Kafka/Redis/DB/LLD/HLD questions, so curating them doubles as study. After each phase, quiz the owner on that phase's design decisions (ADRs, review findings), and have them rewrite one piece themselves.

## How parallel phases are run (proven in Phase 1: 8 builders, about 1h40m, every module 8–9)
Template: `docs/workflows/phase1-parallel-build.js` (the exact Workflow script used). Reuse it for Phase 3/4 by replacing the MODULES list. The method:
1. **Lock shared interfaces first, sequentially** (ports/contracts/schema), so builders never need to change them.
2. **Each builder gets exclusive owned paths**, at most one reserved migration slot, and a `docs/handoff/<id>.md` for anything outside its paths. No `uv add`. ruff runs only on its own paths. Test files are prefixed with the module id.
3. **Pipeline per module:** build → blind principal review (no access to REVIEW.md) → fix P0/P1 → re-review, at most 2 fix rounds. Leftover P1s get a targeted fix plus re-review afterwards.
4. **Then, sequentially:** run `make check` over the whole repo, apply the handoffs, `make nuke && make up`, and use it for real (expect integration bugs; Phase 2a found a P0 that no module test caught).
5. **State the cost and get a yes first.** Phase 1 used about 6.6M subagent tokens; the owner's plan limits are real.
