# Handoff EV-RUN: the runners (E-R, E-G, E-V, E-I, E-A + latency pass) and the v2 CLI (ADR-0017 §3, §5, §7)

**Status:** built and tested against the LOCKED seams (`schema.py`, `runfile.py`, `interfaces.py`, the contracts, the SDK), the D0 demo set and the golden fixtures. I edited nothing outside my paths. No locked seam had to change. Three contract gaps are listed in §4; each one is worked around and labelled in the data, or warned about at run time.

**Owned files:**
- `packages/client/src/ragforge_client/evals/runners.py`
- `packages/client/src/ragforge_client/evals/cli.py`
- `packages/core/tests/module_ev2/test_ev_run_{fakes,runners,oracle,units,cli,resilience}.py`

`test_ev_run_fakes.py` is a helper module with no tests: the fake SDK client that replays the golden fixtures, plus a fake judge, summarizer and label store.

## Review round 2 (2026-10-21): what changed
All 3 P1s and all 6 P2s are handled inside my paths. One P2 (latency cold rows) needs a change in EV-METRICS, which is listed in §4 item 5.
- **P1: a resumed retry lost to the row it recovered.** `drive` now reads each item's highest attempt from the resumed file (`prior_attempts`), and `execute` writes `attempt = prior + this session's attempts`. So `attempt` only goes up across sessions, and the "highest attempt wins" rule in `metrics.select_rows` and `gate._latest_rows` picks the recovery. Test: an item fails with 503 three times, the process crashes, and the run resumes. The rows are then attempts `[3, 4]`, and the real `metrics.summarize` reports `errors_transient == 0` and `errors_ours == 0`.
- **P1: the chat gate skipped the E-V hard floor.** `cli.HARD_FLOORS[chat]` adds the verifier surface's verdict, comparing `baseline-verifier-verifier.json` with the newest verifier run. If that baseline or a verifier candidate is missing, `gate --surface chat` **refuses** with exit 2 and says to re-run `run verifier`. It never passes unchecked. The verifier checks appear in `gate-chat.json`. Tests: the chat kinds pass but the verifier blocks, which exits 1; with no verifier baseline it exits 2 and writes no gate file.
- **P1: `--override` turned not_evaluable into exit 0.** The exit code is now `gate.all_pass(verdicts)` over every per-kind verdict, floors included. Each verdict carries THIS invocation's `--override`, so an override excuses a block and never a not_evaluable. An override recorded on an earlier gate does not pass a new one. The override is still written into every candidate's header. Tests are parametrized over not_evaluable→1, block→0 and warn→0, plus a test that an earlier override does not carry over.
- **P2: an inconclusive verifier counted as an objection.** I recorded the decision in `runners.InconclusiveVerifier`. When `verifier_ran` is true and `verifier_conclusive is False`, the row is **provider-transient**: it is retried, then classified, and it is never an objection. Transient rows are left out of E-V metrics and gate pairing. A model that is always inconclusive leaves nothing paired, and the gate blocks on that. Tests: one inconclusive reply, then a retry that matches golden; an always-inconclusive item becomes a transient error, and metrics count it under `errors_transient`.
- **P2: unmapped stage chunks were scored silently as misses.** `PageLookup.load` fetches in batches of `CHUNKS_BATCH = 100` (the server accepts 200) and raises `ItemFailure` when any stage or shown chunk is absent. That makes it an error of ours, and the `unresolved:` page is gone. Tests: one dropped reranked chunk produces an ours error and no `unresolved:` page anywhere; 207 ids are fetched as `[100, 100, 7]`; absent ids are not cached.
- **P2: `--rpm` counted our requests, not provider calls.** This is now documented in `Pacer`, the module docstring and the `--rpm` help: set `rpm` to about provider RPM / provider calls per item (a chat turn makes at least 2). Example: `--rpm 5` on a 10-RPM tier for E-G/E-A. No per-kind factor was added, because the number of server calls per item is the server's to report.
- **P2: a crash after the judge finished wedged `--resume`.** `find_resumable` deletes a staging file whose run file has finished. `answer_then_judge` (`drop_finished_staging`) does the same for an explicit resume, then raises `RunAborted("... already finished: removed its leftover staging file")`. Both cases are tested by crashing `_remove` right after `finish()`.
- **P2: no CLI path for adjudication.** `label <run> --adjudicate` works on judged generator runs. It queues the verified answers whose faithfulness score in the judge cache is below 2 (an invalid reply, score None, flags nothing). The view is blind: question, notes and answer, with no judge score or reason. It asks "the notes do NOT support this answer (false-verified)?" and writes `LabelKind.adjudication` with `confirmed`, keyed by the judge's item digest, so `metrics.confirmed_false_verified` pairs it. Two tests cover this.
- **P2: cold rows after a resume (EV-METRICS).** The runner's `grades["cold"]` is right, but `metrics.cold_and_warm_rows` ignores it (see §4 item 5). The runner docstring no longer claims these rows are "all reported apart". `test_without_a_resume_the_cold_marks_agree_with_metrics` passes through the real metrics. `test_after_a_resume_metrics_reports_every_marked_cold_row_apart` is `xfail(strict=True)`, so it flips to a failure, prompting removal of the mark, once metrics honours the grade.
- Mutation check: 4 new hand mutations, all killed (prior attempts dropped, inconclusive classed as ours, the old exit rule, the hard floor dropped). Each source was restored from a backup and diffed afterwards.

## Review round 1 (2026-10-21): what changed
All 4 P1s and every P2 are fixed inside my paths; nothing outside them changed.
- **P1 gate candidate.** `gate` picks the candidate by `header.started_at` (never by filename), and takes `--arm`. Without `--arm`, if finished runs of the kind come from several arms, it refuses and names them, instead of gating a stale experiment.
- **P1 judge key on resume.** The staging (answering) header no longer names a judge. `judge_phase` writes the run file's header with the judge that actually scores. Resuming a judge phase whose run file already has rows from another judge is refused (`RunAborted`, nothing appended). Switching judges before any row is judged (e.g. Ollama down, Gemini instead) works, and the header names the new judge.
- **P1 E-V probe.** The header now waits for a *complete* observation (E-V: the verifier ran and reported its model and prompt). Earlier rows are buffered (at most `PROBE_LIMIT = 5` answers; after that the header opens with what was reported, plus a warning). `conflicts()` treats a field that either side does not report as "not reported", never as a change. A mid-run model or prompt change is still an error of ours.
- **P1 chat model.** `Observed.chat_model` is the model of the turn's first `done.usage` entry. The verifier, which also appears in `usage`, only runs on a finished draft, so it is never first. It goes into the header (the server's value wins over `--arm-config`, with a warning when they differ) and into `conflicts()`. On resume, only the fields the server reports for that kind (`runners.REPORTED`) are inherited from the header; the rest are rebuilt from the config, so a changed config is still refused.
- **P2s.**
  - Latency `cold` now marks the first 2 answers of each process, so a resumed pass marks its own cold start.
  - E-I warns when the header has no interview prompt (contract gap 3 below).
  - The judge is paced by its own `--judge-rpm` (`RunContext.judge_pacer`).
  - `demo` runs application and its latency pass before the judged kinds.
  - `load_judge_results` reads through `datasets.complete_lines`: a half-written last line is skipped, and a bad complete line raises `RunAborted` with `file:line`.
  - The demo key moved to `~/.config/ragforge/eval/demo/demo.key`. `--dataset demo` uses only that directory; `--dataset private` refuses the demo key, even when passed explicitly.
  - `--collection` is refused for `--dataset private`.
  - A chat stream without a verification event is an error of ours, not `unverified`.
  - A generated question's row `shown` (what the judge and `code_ref` see) is now the question's OWN `chunk_ids`, not only the seed's.
  - v1's cite_source is kept as the grade `cited_by_model`: True when the model itself cited an expected page, not counting server-inferred citations.
- **P2 (c) timeouts: no code change, on purpose.** A client-side `item_timeout_s` (300 s) stays an error of ours. The server bounds its own provider calls and turns a stalled provider into a 5xx or `error` event, which IS transient here. If the client timer expires, our server failed to do that, which is what the gate's "no errors of our own" floor should catch. The reasoning is in the `classify` docstring.
- Mutation check for this round: 17 new hand mutations, all killed. On the first pass, 1 survived (dropping the "reference did not report" guard in `conflicts`); `test_a_value_first_reported_after_the_header_fills_a_gap` now kills it. A script restored the sources after each mutation, and I diffed them against a backup.

## Checks run (2026-10-21, after review round 2)
- `uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_client_evalrun.py -q`: **381 passed, 1 xfailed** (the strict xfail above).
- `ruff check` and `ruff format --check` are clean on my 8 files. `mypy --strict` is clean on `runners.py` and `cli.py`, and with `MYPYPATH=packages/core/tests/module_ev2` on my 6 test files. `uv run lint-imports`: 6 kept, 0 broken.

## Checks run (2026-10-21, after review round 1)
- `uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_client_evalrun.py -q`: **365 passed**. 103 of those tests are mine (`-k ev_run`; there were 83 before this round).
  - One run before my changes had 2 failures, both in `test_ev_metrics_trust.py`, which EV-METRICS was editing at that time. The final run is fully green.
- `ruff check` and `ruff format --check` are clean on my 7 files.
- `uv run mypy` is clean on `runners.py` and `cli.py`, and `MYPYPATH=packages/core/tests/module_ev2 uv run mypy` is clean on my 5 test files (strict mode, from the project config).
- `uv run lint-imports`: 6 kept, 0 broken. `runners` and `cli` import only `ragforge_client` and `ragforge_contracts`.
- **Mutation check:** I applied 18 hand mutations, one at a time, and the tests killed all 18. (A script restored the source after each one, and I diffed the files against a backup afterwards.)
  - 4xx counted as transient;
  - one extra retry;
  - mutations verified before their clean source;
  - all tiers swept across the levels;
  - ranks left unsorted;
  - a non-verified verdict with no claims not counted as an objection;
  - the self-hash check on the shown text dropped;
  - errored rows skipped on resume;
  - stale pages ignored;
  - the judge allowed on unfinished answers;
  - errors of ours retried;
  - an unanswerable question always correct;
  - abstains judged;
  - cold latency rows not marked;
  - heldout2 allowed into the label queue;
  - a verdict leaked into the label view;
  - labelled items asked again;
  - a failed gate exiting 0.
  - The first pass left 5 survivors. Each now has a test that kills it (the last 5 tests in `test_ev_run_runners.py`, `test_checked_texts_*` in units, and `test_the_answer_queue_*` in cli).

## 1. What was built

### `runners.py`: one driver, six runners
Each runner returns the path of the run file it wrote, `<runs>/<kind>-<arm>-<UTC stamp>.jsonl`, and writes `<run>.summary.json` next to it through the injected `metrics.summarize`.

| Runner | What it does |
|---|---|
| `run_retrieval(ctx, qs)` | `query(stages=True)`. Each stage's chunks are de-duplicated to pages, every page at its best rank, with ties broken by chunk_id. Unanswerable questions record the top reranked score. |
| `run_generator(ctx, qs, judge)` | Answers into a **staging file** (`<run>.answers.staging`, not `*.jsonl`, so the Eval page never lists it). Only after answering has finished does the judge score the answered rows, never abstains. The judge sees relevance, completeness and faithfulness, plus the shown notes fetched by id and digest-checked. The final run file is the staged rows plus `judge_refs`. |
| `run_verifier(ctx, vs)` | Resolves the D3 chunk keys to ids, then calls `/v1/eval/verify`. Items are ordered **clean source first, then its mutations**, so each clean source is re-verified in the same run, moments before its mutations. The client reduces the response to `VerifyOutput` spans; `caught_by` is left to metrics. |
| `run_interview(ctx, iv, judge)` | Frozen D4 grades, tokenless (eval scope), at the seed's own level for every tier. The 5-level sweep covers **gold and partial only**, with 2 replicates each. Then `questions_per_seed` generated questions targeted through `chunk_ids`, with these deterministic checks: answer leak (content-lemma share > 0.5), `seed_chunk_included`, and `code_ref` for code seeds. The optional judge for question quality runs post-hoc. |
| `run_application(ctx, qs)` | The full stack: correct%, verified%, tokens and classified errors. No judge. |
| `run_latency(ctx, qs)` | The dedicated latency pass, with nothing else running. The first 2 questions are graded `cold: true`. The CLI runs it right after `application`. |

**What every run gets** (the `drive` loop):
- **Stable keys, resolved once at the start** (`/v1/eval/resolve`, in batches of 500). Only `ok` counts. A stale or missing key marks the item `stale`: it is recorded without a call and never re-mapped.
- **A header from what the server reports it ran.**
  - Snapshot digests come from `eval_snapshot_status`, and a run refuses to start on a snapshot with no digest yet.
  - `prompt_versions` come from the `done` event, or from `verifier_prompt`/`verifier_model` on verify.
  - The header is therefore opened on the first item that answers (the probe). Stale or errored rows before it are buffered.
  - Every later item is checked against the header. A prompt or verifier model that changes mid-run turns that row into an **error of ours**, so the gate blocks.
  - The header also records the retrieval config, the chat and verifier models from `--arm-config`, the judge key, and `transient_retries`.
- **Errors:** 429 and 5xx (HTTP or an SSE `error` event) are provider-transient. They are retried up to 2 times with backoff, then recorded as transient. Everything else (4xx, transport failures, timeouts, malformed responses, a missing `shown`) is ours: final at once and counted. Error messages are `clean()`ed and truncated, and never hold keys or note text.
- **Resume:**
  - It goes through `RunWriter(resume=True)`, which refuses a changed configuration before any call.
  - Rows that errored are retried. A retry's `attempt` continues from the item's highest earlier attempt, so the recovery wins the "highest attempt" rule in metrics and the gate. The label queue takes the newest row.
  - E-G and E-I resume in either phase: answering continues in the staging file, judging continues by row count.
  - A judge failure keeps every answer. Its message names only the exception type, because provider messages may hold URLs.
- **Pacing:** `Pacer(rpm)` gives an even gap of 60/rpm between SDK item requests. `rpm` counts our requests, not the server's provider calls, so set it to about provider RPM / calls per item. Setup calls (resolve, chunks, snapshot) are not paced.
- **Privacy:** run files hold chunk ids and digests, never note text. Notes are fetched for the judge, checked against BOTH the recorded digest and their own SHA-256, and then dropped. A note that changed marks the row stale, and that row is not judged.
- **heldout2:** a runner refuses it outright (`RunAborted`).

### `cli.py`: `main(argv, env)`, the `ragforge-eval2` command
Subcommands:
- `run <kind> --dataset demo|private --arm A [--resume] [--rpm N] [--arm-config f.toml] [--judge-config f.toml] [--collection id ...] [--questions-per-seed 2]`.
- `summarize <run>`: refuses an unfinished run, or a dataset whose digest changed since the run.
- `label <run> [--batch b] [--sample 50] [--seed 0] [--adjudicate]`:
  - `--adjudicate` on a judged generator run gives adjudication labels on the verified answers the judge flagged (faithfulness < 2);
  - generator runs give criteria labels on the stratified calibration sample (`datasets.calibration_sample`);
  - application runs give correct/invented labels;
  - it is blind: the question, the shown notes and the answer only;
  - it never includes heldout2, stale, errored or abstained rows, and never asks an already-labelled digest again;
  - the label's `item_digest` is `datasets.item_digest_of`, which is the judge's digest, so the two pair up for calibration.
- `gate --surface chat|verifier|interview [--dataset private] [--arm A] [--candidate run] [--aa run ...] [--override "..."]`:
  - the default candidate is the newest finished run (by `started_at`) of `--arm`; without `--arm`, the runs must all be of one arm;
  - it gates each eval kind that has a baseline;
  - it merges the per-kind verdicts into the locked `gate-<surface>.json`, where the worst outcome wins (not_evaluable > block > warn > pass);
  - `--surface chat` also requires the E-V hard floor (the verifier baseline and a verifier run) and refuses without it;
  - an override is written into each candidate's header (`override` is outside the header digest);
  - exit codes: `gate.all_pass` decides between 0 and 1 (an override excuses a block, never a not_evaluable), and 2 means a clean error.
- `baseline set <run> --note "..."`: writes `runs/baseline-<surface>-<kind>.json`. Only finished runs are accepted, and only from that dataset's runs dir.
- `snapshot create --source-collection id | status | reembed --space n`: uses the **owner's** key. `create` writes `private/runs/target.json`.
- `demo [--admin-key-file f]`:
  - provisions a fresh demo tenant;
  - creates an eval key (`eval+chat+read`) and saves it 0600 to `~/.config/ragforge/eval/demo/demo.key` (never printed);
  - creates the `acme-demo` collection;
  - uploads `evals/demo/pages/*.md` and waits for each ingest job;
  - writes `demo/runs/target.json`;
  - runs all five kinds, plus the latency pass, in the order retrieval, application + latency, verifier, generator, interview. The judged kinds come last, so no judge model is loaded on the 16 GB host during the latency pass.

**File layout** under `--root` (default `evals`): `<root>/<dataset>/` holds the dataset files, `labels.jsonl` and `judge-cache.jsonl`, and `<root>/<dataset>/runs/` holds the runs, summaries, baselines, gate verdicts and `target.json`.

**Keys:**
- `run` and `label` read the eval key for the run's dataset:
  - **private:** `--key-file`, or `$RAGFORGE_EVAL_KEY_FILE`, or `~/.config/ragforge/eval/*.key`. The demo key is refused.
  - **demo:** `--key-file`, or `~/.config/ragforge/eval/demo/*.key`, where `demo` saves it. The env var is ignored for demo because it names the private key.
  - Both directories are separate from the owner's daily key, so `make ask` never picks one up.
- Both go through the existing `ask.read_key` rules: 0600, and never printed.
- The judge reads its own key file (`[judge] key_file`, through `judge.make_judge`).

**Seams:** the sibling modules are reached only through the `Components` dataclass: `metrics.summarize`, `gate.evaluate`, `judge.make_judge` + `CachingJudge`/`JudgeCache`, and `datasets.LabelStore`. Tests replace each one with a fake.

## 2. Integration steps (the integrator applies these)
1. **Script entry**, in `packages/client/pyproject.toml` `[project.scripts]` (the existing `ragforge-eval` is untouched):
   ```toml
   ragforge-eval2 = "ragforge_client.evals.cli:run"
   ```
   Then run `uv sync --all-packages --all-extras` (I did not run it).
2. **Makefile targets** (suggested):
   ```make
   eval-demo: ## D0 end to end: demo tenant, upload, all five evals (live stack + model; not CI)
   	@uv run ragforge-eval2 demo $(ARGS)
   eval: ## v2 eval (ARGS="run retrieval --dataset private --arm A")
   	@uv run ragforge-eval2 $(ARGS)
   eval-gate: ## Regression gate (ARGS="--surface chat")
   	@uv run ragforge-eval2 gate $(ARGS)
   ```
3. **.gitignore:** `evals/*/runs/` (run files, summaries, baselines, `target.json`) and `*.answers.staging`. `evals/demo/runs/` holds answers to synthetic questions only, but it should still stay out of git.
4. **Package docstring:** `evals/__init__.py` says "`cli` - B3: `ragforge-eval` v2 subcommands". The command is `ragforge-eval2`, per step 1. This is a one-word fix in a locked file, so it is yours to make.

## 3. What the integrator must test live (`make up`, then `make eval-demo`)
- **Demo setup.**
  - `provision_tenant` creates a `personal` tenant, because `TenantKind` has no `eval` yet. Once B1's migration 0012 lands, check whether key creation with the `eval` scope requires `tenants.kind = 'eval'`. If it does, `setup_demo` must pass the eval kind (a one-line change in `cli.py`).
  - Check that an eval-scope key cannot be created on an MCP key, and that the owner's daily keys never list the demo tenant.
- **Upload.** The SDK has no upload method, so `cli.http_uploader` posts multipart to `/v1/collections/{id}/files` with the tenant's provisioned key (it needs write). httpx sets `Content-Length`, which the route requires. Check that the file `page_key` comes out as `(file, <filename>)`, so that D0 keys resolve.
- **E-R:** that `stages` are returned under the eval key, and that two repeated runs are identical (A/A, tie-break by chunk_id).
- **E-G:**
  - that `?trace=shown` is honoured and `done.shown` is present;
  - that `prompt_versions` is present on `done`;
  - that the judge runs only after the last chat call. The Ollama and Gemini logs should show no judge call during answering.
- **E-V:** that `verifier_model` and `verifier_prompt` come back on verify (otherwise the header falls back to `--arm-config`).
- **E-I:** that a tokenless eval-scope notes-mode grade at any level is accepted, and that `chunk_ids` on the interview question is honoured.
- **E-V inconclusive:** force the server's verifier provider to 429 (or time out), and check that `verifier_conclusive: false` comes back with HTTP 200 and the row becomes `provider_transient`.
- **Gate:** `make eval-gate ARGS="--surface chat"` with no verifier baseline should exit 2. After `baseline set` on a verifier run it should gate both.
- **Pacing:** `--rpm 5` (provider RPM 10 / 2 calls per chat turn) on Gemini Flash-Lite. Confirm that no 429 appears; if one does, confirm it is retried and recorded as `provider_transient`. With the Gemini judge, pass `--judge-rpm` too; it is paced apart from the stack.
- **Chat model in the header:** check that `done.usage[0].model` is the answering model on a real turn, with the verifier's usage after it, so that `header.chat_model` names the chat model.
- **E-V header:** check that `verifier_model` and `verifier_prompt` are present when `verifier_ran` is true and absent when the coverage gate caught the item. The header waits for the first item where the verifier ran.
- **Resume:** kill a long run with Ctrl-C, then `run <kind> --resume` with the same arm. Kill E-G again during its judge phase and resume once more.

## 4. Contract gaps (no seam changed; the integrator decides)
1. **Interview `misconceptions` given vs kept** (§3 E-I). `InterviewGrade` returns only the quotes that survived the server's quote rule and verifier. So the runner records `misconceptions_given = misconceptions_kept` and grades the row `given_is_kept: true`, which makes "given" a lower bound.
   - **Proposed fix:** an additive, eval-scope-only `misconceptions_given: list[str] | None` on `InterviewGrade`. The runner change is one line.
2. **No SDK upload method.** `demo` uses its own small httpx uploader (multipart built by httpx). An SDK method `upload_file(collection_id, name, content) -> Accepted` would let `cli.http_uploader` go away.
3. **Interview responses report no prompt versions** (round-1 P2). `InterviewGrade` and `InterviewQuestion` carry no `prompt_versions`, so an E-I header holds only what `--arm-config` says. The interview baseline is keyed by the per-surface prompt set, so an interview prompt change would go undetected. Until this is fixed, the runner logs a warning when an E-I header has no `interview` prompt.
   - **Proposed fix:** an additive, eval-scope-only `prompt_versions: dict[str, str]` on both responses. The runner change is to report it in `Observed` from `_grade` and `_generate`, and add `EvalKind.interview` to `runners.REPORTED`.
4. **For EV-METRICS (not a seam change):** E-G and E-A answer rows now carry the boolean grade `cited_by_model` (None when provenance is unknown). Metrics may report it like v1's model-cited share. Nothing breaks if they ignore it.
5. **For EV-METRICS: latency cold rows after a resume.** `metrics.cold_and_warm_rows` should take cold/warm from `grades["cold"]` when a row has that grade, and fall back to the first-asked order only when it does not. Today a resumed latency pass's own cold starts, marked by the runner, end up in the warm p50/p95. The gate is unaffected because latency is warn-only. When that lands, remove the `xfail` from `test_ev_run_resilience.py::test_after_a_resume_metrics_reports_every_marked_cold_row_apart`.
6. **For EV-METRICS/gate (FYI):** both select the latest row by the highest `attempt`. The runner now guarantees that `attempt` grows across resumed sessions, so that rule is enough. No change is needed.

## 5. Known limits
- **Judge failures:** the judge phase treats any judge exception as stop-and-resume. A transient `JudgeTransportError` is not retried inside the runner; `CachingJudge` and the judge adapters own that.
- **Running `run latency` by hand right after a judged run** can still find the judge model resident (Ollama `keep_alive`). Wait out the keep_alive first, or run latency before the judged kinds, as `demo` does.
- **Run ids:** a run id has one-second resolution. Two runs of the same (kind, arm) started in the same second would collide, and `RunWriter` refuses the second one (`O_EXCL`).
- **Gate scope:** `gate` uses `datasets` from `--dataset` (default `private`). Gating the demo set needs `--dataset demo`.
