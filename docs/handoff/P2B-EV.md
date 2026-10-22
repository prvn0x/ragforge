# P2B-EV: eval tooling handoff

Owned and changed: `packages/client/src/ragforge_client/evalrun.py`, `packages/core/tests/module_ev/**` (new).
`packages/core/tests/test_client_evalrun.py` is unchanged and still passes (16), so the old API stays compatible.

## What changed (the CLI is backward compatible: `run` and `score` take the same arguments as before)
- **Citations:** `Outcome.inferred_titles` lists pages the server cited ONLY by inference (`Citation.inferred`). If the model cited a page even once, it counts as model-cited.
  - `correct` accepts either kind of citation of an expected page.
  - New columns in both `score` and `compare`: `model-only%` (correct counting only model citations) and `cited model/inferred%`.
  - Runs recorded before this change show `?` or `-` here (provenance unknown). Nothing is guessed.
- **Arms:** `run --notes "..."` is stored in the header.
  - Each outcome records providers, models, prompt_version and token totals from `DoneEvent.usage`.
  - A trailing `{"footer": ...}` line holds `complete`, the observed arm and resource peaks. The footer is also written when a run fails or is interrupted.
- **`compare <runs...>`:** prints a markdown table with these columns: arm, split, asked k/N, provider: model, notes, correct, model-only, content, cited model/inferred, abstain ok, invented, err, p50/p95/ttft, peak swap GB, swap Δ GB, peak ollama GB, meets bar.
  - Bar: correct ≥ 80, p95 < 10 s, invented + unsure ≤ 2% of answered, AND every question of the run's split asked.
  - A run with footer `complete=false`, or fewer outcomes than its split has questions, is `no: incomplete k/N`, however good its answers. A run with no footer (old format, or SIGKILL) is at best `unknown: no footer`.
  - With partial labels, (invented + unsure) / answered is a lower bound on the final rate. If it already exceeds 2%, the verdict is `no`. Otherwise it is `unknown` while labels are incomplete.
  - **Errors and timeouts (review round 2 P1):** any errored turn makes the verdict `no: N errored turn(s) (k timed out)`.
  - A timed-out turn counts in p50/p95 at max(measured, deadline), because it took at least that long. The cell then shows `≥` to mark a lower bound. Other errors (a fast 5xx) stay out of latency, and the bar already fails them.
  - `run` writes `question_timeout_s` to the header (None when disabled) and sets `Outcome.timed_out`. Older rows are recognised from the `timeout after …` error text.
  - A run whose question timeout was under 10 s is `no: question-timeout Xs<10s`, because such a deadline makes p95 < 10 s true by construction.
  - The table shows aggregates only, so it is safe for held-out runs.
- **Resilience:** a transport error (connection reset or refused, protocol error) or a per-question timeout (`--question-timeout`, default 300 s, 0 disables) becomes that question's `error`. The run continues. Ctrl-C still stops the run and writes `complete=false`. If SIGKILL truncates the last line of the run file, `load_run` skips that line with a warning, and the run loads without a footer.
- **`--label`** must match `[A-Za-z0-9._:-]{1,64}`, because it becomes part of the run filename.
- **Resources:** `run --sample-interval 2` (default; 0 turns it off) samples memory in a worker thread.
  - macOS: `sysctl -n vm.swapusage` and the summed RSS of `ollama*` processes from `ps`.
  - Linux: swap comes from `/proc/meminfo`.
  - Any failure records None (shown as `-`), never 0.
  - The first swap reading is stored as `swap_baseline_mb`. `compare` shows `swap Δ GB` = peak − baseline, because this Mac already carries about 11–13 GB of swap before any run. Footers written before this change have no baseline, so their Δ shows as `-`.
- **`label <run> [--reveal] [--relabel]`:** marks each answered result ok / invented / unsure, or lets you skip or quit.
  - Labels go to `<run>.labels.json` next to the run file. The file is saved atomically after every answer, so labelling can be resumed.
  - Each label is bound to the SHA-256 of the answer text, so a stale label is ignored.
  - `score` and `compare` say `unlabelled` or `partial k/n` until every answered result is labelled.
- **Held-out:** per-question output for held-out questions is hidden unless `--reveal` is passed. This covers `score -v`, `run` progress and `label`. Without `--reveal`, `run` asks held-out questions last, as one block, and prints one line before it and one aggregate line (`[N/N] k held-out question(s) done`) after it. There is no per-question line, so the terminal's timing exposes only the block's total latency (see round 4).
- **Held-out labelling protocol:** invented% needs a human to read each answer, so a held-out (or `all`) run stays `unknown: invented% partial` until `label --reveal` is run, and that pass shows the held-out answers.
  - Do this ONCE, after the arm is frozen, to get the final held-out verdict. Never do it during tuning.
  - The sidecar stores `heldout_revealed_at` at the first reveal, written before the first held-out answer is shown. Every later `--reveal` session prints it, so a tuning session can see that those rows have been seen.
- **Setup errors** (no collections, bad key file, API error, or a server that is down, `cannot reach <url>: ConnectError`) exit with a one-line message instead of a traceback. The key is never printed.
- **Label sidecar durability:** the temp file is fsynced before `os.replace`, and the directory is fsynced (best effort) afterwards.

## Requests for other owners
1. **Makefile (integrator):** update the `eval-private` help text to list the new subcommands:
   `ARGS="run --label <arm> [--notes ..]" | "score [-v] <run>" | "compare <runs..>" | "label <run>"`.
2. **docs/NEXT.md (owner/integrator):** in the Phase 2b handoff, point arm comparisons at `make eval-private ARGS="compare evals/private/runs/*.jsonl"`.
   - Note that invented% now needs `make eval-private ARGS="label <run>"`.
   - Old runs A0 and A1 re-score to the same numbers (verified: 15/54 and 46/46 correct/content, p95 52.8 s and 22.0 s). They have no provenance split or memory peaks.
3. **Caveat to record with A3 (qwen3:8b):** on Apple Silicon, Ollama's Metal allocations may not fully show up in RSS. Treat `peak ollama GB` as a lower bound. Peak swap is the reliable pressure signal.
4. **Git (owner, at the end of the project):** label sidecars live under `evals/private/runs/`, which is already gitignored via `evals/private/`. No change is needed. Just confirm before the first commit.

## Review round 1 fixes (blind score 8/10)
All 9 findings were addressed: the P1 (incomplete runs met the bar) and all 8 P2s. Each has a regression test in `module_ev/test_ev_reports.py` or `module_ev/test_ev_run.py`.

## Verification (round 1, superseded below)
- `uv run pytest packages/core/tests/module_ev packages/core/tests/test_client_evalrun.py`: 71 passed.
- Clean on these paths:
  - `ruff check` and `ruff format --check`;
  - `mypy` (strict) on `packages/client/src`, on `test_client_evalrun.py`, and on `module_ev` with `MYPYPATH=module_ev`;
  - `lint-imports` (6 kept).
- Full suite `uv run pytest packages`: 1277 passed, 8 failed.
  - All 8 failures are in `module_d1/*` and `module_integration/*` (agent turns, evidence and approval flow), which other modules are changing in parallel.
  - None of them import `evalrun`.
- Live read-only sampler check on this Mac: swap is about 11.4 GB used. The Ollama server's RSS is about 18 MB because no model was loaded.

## Verification (round 2, this session)
- `uv run pytest packages/core/tests/module_ev packages/core/tests/test_client_evalrun.py`: 93 passed.
- Clean on the owned paths: `ruff check`, `ruff format --check`, `mypy` (strict) on `packages/client/src` and on `module_ev` + `test_client_evalrun.py` (with `MYPYPATH=module_ev`), and `lint-imports` (6 kept).
- Full suite `uv run pytest packages -q`: 1573 passed, 1 skipped, 0 failed.

## Review round 2 fixes (blind score 8/10)
- **P1 (errored or timed-out turns could still meet the bar):** fixed as described under compare above. Regression tests are in `module_ev/test_ev_bar_errors.py`: the reviewer's 300 s probe, the `--question-timeout 9` gaming case, the deadline floor, legacy timeout rows, and a fast 5xx.
- **P2s:** the `httpx.HTTPError` setup exit, fsync of the sidecar, the documented and stamped held-out protocol, and the missing tests. The new tests are a CLI `label` test, an end-to-end `run()` held-out hiding test with `--reveal`, and a structured-timeout header test, in `test_ev_run.py` and `test_ev_labels.py`.
- `label_run(prompt=None)` now resolves `input` at call time, like `out`, so a patched or redirected terminal is honoured.

## Review round 3 fixes (blind score 8/10)
- **P1 (an arm that answered every must-abstain question could still be `**yes**`):** an answer to a must-abstain question now counts as invented **by construction**. It needs no human label, a label on it is ignored, and it counts as both labelled and invented (`InventedStats.forced`).
  - The reviewer's reproduction (dev-off answered, every answer labelled ok) is now `no: invented 25.0>2 (incl. 1 answered must-abstain)`. Without any labels the partial lower bound already fails.
  - The `label` walk prints a one-line rubric ("invented = any claim your notes do not support, even if true"). It does not ask about must-abstain answers, because the label would be ignored; it prints how many were counted instead.
- **Question fingerprint:** each new row stores `question_split` and `question_sha256` (a digest of the text and the answerable flag).
  - If the text, split or answerable flag changes after the run, the verdict is `no: k question(s) changed since the run`. `score` warns with a count and `-v` marks the row.
  - Fact and page corrections are deliberately not in the digest, so re-scoring after a correction still works.
  - Legacy rows (A0/A1) carry no fingerprint and are not flagged. They still re-score to the same numbers.
  - Held-out hiding now also applies when the question was held-out at run time, so moving it to dev after the run does not reveal it.
- **Run files are 0600** and created with `O_EXCL`. A same-second filename collision is a setup error, never a truncated earlier run.
- **Server text cleaned:** the ErrorEvent title, per-question ApiError text, the setup ApiError exit message and the `score` model column (also `_md_cell`) are cleaned.
- **Provenance must be sent:** a citation that arrives without the `inferred` field (checked via `model_fields_set`) makes that turn's provenance unknown (`?`). A test proves the current `encode_sse` always sends the field.
- **Rough edges:**
  - The compare column is renamed `asked`.
  - The partial-label lower bound uses `invented` only, because `unsure` may be relabelled ok.
  - The swap baseline comes only from the first sampling attempt. If that attempt fails, the baseline is None and Δ shows `-`.
  - An interrupt while the footer waits for the sampler is deferred until after the footer is written.
  - Ctrl-C now exits 130 with one line instead of a traceback.
- New tests: `module_ev/test_ev_integrity.py`, plus 5 in `test_ev_run.py`. They include a test that the reveal stamp survives a later non-reveal `--relabel` session.

## Verification (round 3)
- `uv run pytest packages/core/tests/module_ev packages/core/tests/test_client_evalrun.py`: 131 passed.
- ruff check/format, mypy strict (client src; module_ev + test_client_evalrun.py with MYPYPATH), lint-imports (6 kept): all clean.
- Full suite: 1400 passed, 2 skipped, 0 failed, 422 errors. Every error is a testcontainers setup error: `/var/run/docker.sock` was absent during this session (the Rancher VM was down), and I was not allowed to start it. None of the errors are in module EV.

## Review round 4 fixes (blind score 8/10)
- **P1 (deleting a question after the run flipped the verdict to `**yes**`):** the run header now records the asked set, `questions: {id: question_digest}`.
  - `removed_questions` returns the ids the run asked, taken from the header or from any fingerprinted row, that the current file no longer has.
  - `judge_bar` fails such a run with `no: k question(s) removed since the run`. `score` prints a warning with the count only, never the ids, so held-out ids cannot leak.
  - Legacy runs (A0/A1) carry neither the header field nor row fingerprints, so they are not flagged, as before.
  - `score` and `compare` now share `summarize_run`, so they cannot drift apart on what they grade against.
  - Regression test: the reviewer's reproduction. A dev run with a wrong answer is `no`; with that `[[q]]` deleted it is still `no`.
- **Edit matrix:** a parametrised test runs `compare` on a passing run after each kind of edit and asserts `no:` with the specific reason:
  - text, split, answerable → changed;
  - deleted → removed;
  - added → `incomplete 3/4`.
  - A second test checks that fact and page corrections still pass.
- **Held-out timing leak (P2):** as described under Held-out above. With `--reveal`, questions are asked in file order. Otherwise held-out questions come last, so on an `all` split their latency is measured after the dev block (a warm model). That is a small ordering effect, and it is the same for every arm.
- **Setup failure left a 0-row run file (P2):** the client is opened and the collections are resolved before the run file is created. A server that is down, an API error or an empty tenant now leaves no file. Once the file exists, a failure mid-run still writes the `complete=false` footer, and a test covers this.
- **Ctrl-C in `label` (P2):** exits with code 130 and one line on stderr, `interrupted; labels saved, resume with: label <run>`. Labels given before the interrupt are kept, and a test covers this.
- Tests: new `module_ev/test_ev_question_set.py`; the updated and new run tests are in `test_ev_run.py`. `ev_support.write_run` gained an optional `questions=` header field.

## Verification (round 4)
- `uv run pytest packages/core/tests/module_ev packages/core/tests/test_client_evalrun.py`: 145 passed.
- Clean on the owned paths:
  - `ruff check` and `ruff format --check`;
  - `mypy` (strict) on `packages/client/src`, and on `module_ev` + `test_client_evalrun.py` with `MYPYPATH=module_ev`;
  - `lint-imports` (6 kept).
- Full suite `uv run pytest packages -q`: 1465 passed, 2 skipped, 0 failed, 422 errors. Every error is a testcontainers setup error, because `/var/run/docker.sock` is absent (Rancher Desktop is stopped, and I did not start it). None of the errors are in module EV.

## Needs live evidence (integrator)
None of the round 4 fixes needs a live server to verify, since every one is proven with respx or fixture tests. One live smoke run would still confirm the new header and the held-out progress shape against the real API. Run it only when the stack is already up:
1. `make eval-private ARGS="run --label smoke-r4 --split all --sample-interval 2"`. Expect the dev rows first, then `asking k held-out question(s); progress hidden`, then `[N/N] k held-out question(s) done`, with no held-out id printed.
2. `python -c "import json,sys; h=json.loads(open(sys.argv[1]).readline())['header']; print(len(h['questions']))" evals/private/runs/<that run>.jsonl`. Expect the number of questions in the `all` split. This prints a count only, never ids or text.
3. `make eval-private ARGS="compare evals/private/runs/*.jsonl"`. Expect no new `0/N` junk rows from earlier failed setups. A stale 0-row file written by an older build can be deleted by hand.
