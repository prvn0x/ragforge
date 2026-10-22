# EV-JUDGE handoff: judge port, cache, datasets, D3 builder, calibration sample

Owned and created:
- `packages/client/src/ragforge_client/evals/judge.py`
- `packages/client/src/ragforge_client/evals/datasets.py`
- `packages/client/src/ragforge_client/evals/prompts/` (`__init__.py`, `judge_qa_v1.md`, `judge_question_v1.md`)
- `packages/core/tests/module_ev2/ev_judge_support.py` and `test_ev_judge_{ports,cache,datasets,d3}.py` (123 tests)

No locked seam, contract, SDK, D0 file or fixture was changed. Nothing in `evals/private/**` was opened. I didn't run git, uv add/lock/sync, or the stack.

## Round 3 (after the blind round-2 review, 8.3/10): what changed
The P1 and every P2 inside my paths are fixed. Each fix has a test, and each was mutation-checked: with the fix reverted, a test fails (10 mutants, all killed).
- **P1: a torn tail no longer bricks `labels.jsonl` or `judge-cache.jsonl`.** Under the lock, `append_jsonl` now cuts the file back to just after its last newline (`os.ftruncate`). It reads backwards from the end in 4 KiB blocks, and a well-formed file costs one 1-byte `pread`. It no longer adds a `\n` before the new line, which used to turn the fragment into a corrupt line in the middle of the file. A short write (disk full) also truncates back to the pre-write size before it re-raises, so none of our own bytes are left behind. Readers still raise on a corrupt line in the middle of a file, by choice: with the cut in place, such a line means real corruption, and silently skipping it would hide lost hand labels. Tests cover: a torn tail followed by an append, after which `latest()` returns both labels; a file that is a single 10 KB fragment with no newline; a short write leaving the file byte-identical; and the same torn-tail case for `JudgeCache`, both with a fresh reader and with an already-indexed one.
- **P2: every httpx error is classified.** `TooManyRedirects` maps to `JudgeTransportError(transient=False)`. Any other `httpx.HTTPError` (e.g. `DecodingError`) maps to `transient=True`. The message carries the type name only.
- **P2: completeness needs reference facts.** `make_item` drops `completeness` when `facts` is empty. The item is still judged on its other criteria, and completeness simply has no result. `score()` refuses a hand-built item that asks for completeness with no facts.
- **P2: calibration candidates.** `calibration_candidates(..., *, kinds=frozenset({EvalKind.generator}))` skips `stale` rows and whole runs of any other kind. The existing positional signature is unchanged, so `cli.calibration_queue` keeps working.
- **P2: a length pre-check runs before the Ollama call.** If the system prompt plus the item is longer than `3 chars × (num_ctx − 512)`, the item is scored `truncated` and nothing is sent. The `prompt_eval_count` check stays as the second guard. The bound errs towards flagging: a false `truncated` shows up as an acceptance failure, while a missed one would be a silently wrong score.
- **P2: a garbled provider envelope is transient and never cached.** This covers an Ollama body that is not JSON, has no `message.content`, or has non-string content, and a Gemini body that is not a JSON object or has no candidates without a block reason. All raise `JudgeTransportError("malformed reply envelope", transient=True)`, so the runner retries them. A Gemini safety block (`promptFeedback.blockReason`, or a candidate finished with no content) is still a bad reply: scored None, cached and counted. See "Decisions" below.
- **P2: `build_d3` collapses labels itself.** It applies the new `newest_labels(labels)`, which `LabelStore.latest()` also uses, before it picks owner-correct answers. Passing `store.all()` therefore gives the same set as `store.latest()`, and an overruled "correct" no longer admits an answer.
- **P2: every Gemini reply must report its `modelVersion`.** A reply without it raises a non-transient `JudgeTransportError`. `open()` already required the field, so a reply without it means the API changed.
- **P2: overlap wording and the schema comment.** These are not my files. They are listed under "For the integrator" below.

### Decisions to confirm (round 3)
- **Safety blocks are cached.** A Gemini block is the provider's decision on this exact input at temperature 0, so it repeats like an invalid model reply. Counting it as None matches the "invalid is None, counted, never imputed" rule. Only an envelope that is actually garbled is retried.
- **Completeness is dropped rather than refused in `make_item`.** Refusing would abort a whole E-G run (the runner turns any judge exception into `RunAborted`) because one private question has no facts. Metrics see no completeness result for that item, so it counts as unscored, not as 0.

## Round 2 (after the blind round-1 review, 8/10): what changed
Every P0/P1 and every P2 inside my paths is fixed. Each fix has a test, and each test was mutation-checked (the fix reverted, the test fails, the file restored).
- **P1 D3 clusters per question.** `build_d3` keeps ONE clean source per question: the owner-correct answer with the smallest item digest, so the pick does not depend on run order. Every other answer to that question (another arm, the A/A repeat) is counted under `skipped["duplicate_question"]`. The bootstrap's `source_id` is therefore the source question, as ADR-0017 §3 E-V assumes. Test: two runs answering every question differently, in both orders, plus an A/A repeat.
- **P1 Ollama cloud models need the opt-in.** `OllamaJudge.open` refuses a model whose `/api/tags` entry has `remote_host` or `remote_model`, or whose name ends in `:cloud`/`-cloud`, unless `allow_third_party=True`. The local daemon forwards the prompt (note text) to ollama.com for these. Tests: four tags shapes, refused then accepted with the opt-in, and a local model named `cloudy-llm` stays local.
- **P1 Truncated prompts score None.** The Ollama reply's `prompt_eval_count` is compared with `num_ctx`. At or above `num_ctx - 512` (the reply reserve) the call's outcome is the new `truncated`: every criterion is None, the call is logged, and `acceptance` counts it as invalid. A reply with no count is not flagged (nothing to check; see live check 1). Tests: the boundary on both sides, a non-default `num_ctx`, and no count.
- **P2 Gemini key binds the served model.** `GeminiJudge.open` now also makes one tiny generate call (the fixed word "ping", `maxOutputTokens: 1`, no note text) and puts the served `modelVersion` in the key: `model_version = "<model>@<version>/<served modelVersion>"`. A probe without `modelVersion` is refused (`JudgeConfigError`). Every reply's `modelVersion` must equal the key's, else a non-transient `JudgeTransportError`. Cost: one ~1-token call per judge open.
- **P2 A partial cache miss re-asks the whole item.** `CachingJudge` asks the inner judge for the item's full criteria set whenever any criterion misses, so every result of an item comes from one prompt and one reply schema. The fresh lines supersede the cached ones (the newest line per key wins). `batch_key` stays None.
- **P2 heldout2 is always sealed.** A file named `heldout2.toml` must be sealed on every load (`load_questions` and `load_datasets`), whatever the caller passes. Deleting the seal line and then editing fails. `require_seal=True` still extends this to heldout2 rows in any other file. Test: the bundle refuses an unsealed or seal-stripped `heldout2.toml`.
- **P2 Lemmatizer is symmetric.** `_lemma` now (1) drops a plural `s` only, (2) drops `-ing`/`-ed`, (3) drops a final `e` and maps a final `y` to `i`. `cache/caches/cached/caching`, `box/boxes` and `policy/policies` now meet. A hypothesis property test checks that `x` and `x + "s"` always share a lemma.
- **P2 Appends are O(new lines).** `append_jsonl` takes the same lock, then does ONE `O_APPEND` write and an fsync (it no longer reads and renames the whole file). `JudgeCache` indexes only the bytes appended since its last read. Readers (`LabelStore`, `JudgeCache`) leave an unterminated last line for the next read instead of failing on it. A file that shrank is re-read from the start. Tests: same inode after an append, a half-written line ignored and then picked up, a replaced file re-read.

## Status (verified by running)
- Round 3: my 123 tests pass. The full `module_ev2` + `test_eval_schema.py` run gives 323 passed and 2 failed, and both failures belong to other modules editing in parallel. `test_ev_metrics_trust` fails a different test on each run while `stats.py` is being edited. `test_ev_run_runners::test_generator_records_the_deterministic_correct_grade` sees a new `cited_by_model` grade that `runners.py` added a minute before my run and its test doesn't expect yet. Neither imports `judge.py` or `datasets.py` in the failing path. ruff, `mypy --strict` (sources, and tests with MYPYPATH) and lint-imports are all clean.
- Round 2: `uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_eval_schema.py -q`: 275 passed (my 90 included).
- `packages/core/tests/module_ui` + `module_ev` (not mine, run as a neighbour check): 2 to 7 failures that change from run to run, all in the UI module's own tests (`STORE_MODULE`, `RunsRead`, the flip measures). The same suites fail with my round-1 files swapped back in, so they're the UI builder's in-progress work, not this change. Re-run once every module has landed.
- `ruff format --check` and `ruff check` are clean on my paths. `mypy --strict` is clean on `judge.py` and `datasets.py`. `MYPYPATH=packages/core/tests/module_ev2 mypy` is clean on my test files. `uv run lint-imports`: 6 contracts kept. The client imports only `ragforge_client` and `ragforge_contracts`.
- **Mutation checks:** I applied each mutant below, ran the tests, and restored the file. Every mutant was killed:
  - the file lock removed;
  - the nonce check removed;
  - the criterion dropped from the cache key;
  - D3 selecting on the verdict instead of labels;
  - the seal covering only question text;
  - no floor of one per stratum;
  - heldout2 allowed into the sample;
  - the chunk text-digest check removed;
  - heldout2 allowed into D3;
  - partial replies voided;
  - the third-party opt-in removed.

## What I built
### judge.py (ADR-0017 §4)
- **`await make_judge(JudgeConfig(...))`** returns an `OllamaJudge` or `GeminiJudge`. Use it as `async with`.
  - **It is async**, because `JudgeKey.model_version` must hold the version the provider returns *before* the first score (the run header records it):
    - Ollama: probes `/api/tags` and sets `model_version = "<model>@<first 12 hex of the digest>"`;
    - Gemini: probes `GET models/<m>` for `version`, then one tiny generate call for the served `modelVersion`, and sets `"<model>@<version>/<served>"` (round 2).
  - **Refused before any network call:**
    - an unknown prompt;
    - any provider other than an Ollama on this machine (localhost, 127.0.0.1, ::1, host.docker.internal) unless `allow_third_party=True`;
    - after the tags probe (round 2): an Ollama cloud model (`remote_host`/`remote_model` in its tags entry, or a `:cloud`/`-cloud` name) unless `allow_third_party=True`;
    - a Gemini key file that other users can read. It uses `ragforge_client.ask.read_key`, so the mode-600 rule is the same one as today.
  - **The key** goes in the `x-goog-api-key` header only: never in the URL, a repr or an exception. A test checks all of these.
- **Ollama call:** `/api/chat` with:
  - `format` = the reply JSON schema;
  - `options = {temperature: 0, seed: 0, num_ctx: 8192}`;
  - `keep_alive: "2m"`, `think: false`, `stream: false`.
- **Gemini call:** `generateContent` with `temperature: 0`, `responseMimeType: application/json` and `responseJsonSchema`.
  - One item per call. Batching is deferred per r2 P2, so `batch_key` is always None.
  - A reply whose `modelVersion` is not the key's served version raises a non-transient `JudgeTransportError`, because every result must belong to the key it is filed under.
- **Prompts** are `judge_qa@1` (relevance, completeness, faithfulness) and `judge_question@1` (grounded, on_topic, at_level).
  - Each is a file with 0/1/2 anchors per criterion and "length is not quality".
  - Each file is **pinned by sha256** in `PROMPTS`. Editing a released version's text fails at load time. To change wording, add `judge_qa_v2.md` + `"judge_qa@2"`.
- **Blind and fenced:**
  - The request is exactly the pinned rubric plus `render_item(...)`. A test asserts byte equality, so no arm, model or verdict can be sent.
  - Each item is fenced as `<<<ITEM nonce>>> … <<<END nonce>>>` with a fresh `secrets.token_hex(8)` per call.
  - The reply schema keeps `nonce` a plain string, not a `const`, so constrained decoding can't echo it for free.
  - A reply that isn't JSON or has the wrong nonce sets every criterion to None. A criterion whose object fails the schema is None while the valid ones stand (outcome `partial`).
  - Ollama only: a `prompt_eval_count` ≥ `num_ctx - 512` makes the call `truncated` and every criterion None (round 2).
  - Only transport failures raise `JudgeTransportError(status, transient)`. `transient` covers 429, 5xx, timeouts, connection errors, any other httpx error except `TooManyRedirects`, and a garbled provider envelope (round 3). An error body is never surfaced, since it may echo notes.
  - Ollama only: before the call, a prompt longer than `3 × (num_ctx − 512)` chars is `truncated` and not sent (round 3).
- **Items and digests:** build items with `make_item(criteria, question, answer, notes, facts, level)`.
  - `item_digest = datasets.item_digest_of(question, answer, notes, facts, level)`. `score()` refuses an item whose digest doesn't match, because the result would otherwise be cached under another item.
  - For `judge_question@1`, `question` is the requested **topic** and `answer` is the **generated question**.
- **Cache:**
  - `cache_key(judge_key, item_digest, criterion[, batch_key])` = `digest({"judge", "item", "crit"[, "batch"]})`. This is exactly the formula in the golden `judge-results.jsonl`, and a test recomputes every fixture key.
  - **Note for the integrator:** the `schema.py` comment on `JudgeResult.cache_key` omits `"crit"`. The fixtures and this module include it, and they must, since there is one result per (item, criterion). Consider updating the comment.
  - `JudgeCache(<dataset dir>/judge-cache.jsonl)` stores results under that key.
  - `CachingJudge(inner, cache)` implements `Judge`; on any miss it re-asks the item's full criteria set (round 2).
  - A changed JudgeKey field or item misses the cache; nothing is overwritten in place.
  - Invalid replies are cached too. At temperature 0 a re-ask gives the same reply, and the metric must count it.
- **Log and acceptance:**
  - Each call logs one line on logger `ragforge_client.evals.judge` as JSON: `{event: "judge_call", provider, model_version, prompt, latency_ms, item_digest, outcome}`. It contains no text, which a test checks.
  - `judge.calls` holds the records.
  - `acceptance(calls)` returns per-item latency (max ≤ 60 s), p50 and the valid-JSON rate (≥ 99%; `ok` and `partial` count as valid), plus `passed` and reasons. `truncated` counts as invalid.

### datasets.py (ADR-0017 §2, D5)
- **Loaders:**
  - `load_questions(*paths, require_seal=False)` reads v2 TOML. It accepts several files (for example private `questions.toml` + `heldout2.toml`), refuses duplicate ids, and refuses a v1 `[[q]]` file with a migration hint.
  - `load_verifier_set` checks unique ids, that clean items are their own source, and that no planted item lacks a clean source.
  - `load_interview_set`.
  - `load_datasets(dir)` returns the `Datasets` bundle.
  - Errors are `DatasetError` (a ValueError) and name the file and field.
- **Sealing:** `seal_questions(path)` prepends `seal = "sha256:<digest>"`. The digest covers the parsed `heldout2` rows in id order, so a comment, whitespace or dev-row edit doesn't break it and any change of meaning does.
  - Every load of a sealed file verifies it.
  - A file named `heldout2.toml` must always be sealed (round 2). `require_seal=True` also fails unsealed heldout2 rows in other files.
  - Re-sealing is refused, so unsealing stays a deliberate manual edit.
- **`LabelStore(labels.jsonl)`** implements LabelReader and LabelWriter.
  - Appends: a lock on `labels.jsonl.lock` via `fcntl.flock`, then one `O_APPEND` write and an fsync (round 2). The file mode is 0600. Readers skip an unterminated last line.
  - The newest line per (item_digest, kind) wins, by file order rather than `labelled_at`, since clocks can disagree.
  - A test with two concurrent writer processes loses no line and interleaves none.
- **Overlap flag:** `content_lemmas` (stopwords plus light suffix stripping; identifiers kept whole), `jaccard`, and `question_page_overlap` = the max Jaccard against any single sentence or line of the page.
  - It works per sentence because a whole-page Jaccard is near 0 for every question.
  - `overlap_flags(questions, {page_key_str: text})` flags heldout2 questions above 0.5.
- **D3 builder:** `await build_d3(runs, questions, labels=store.latest(), chunks=<RagforgeClient>)` returns `D3Build(verifier, skipped)`.
  - **Sources:** v2 answer rows with `shown` chunk keys, the text from `/v1/eval/chunks` and the owner's `answer` labels (`correct=True` and not invented) on `item_digest_of(question, answer, texts, facts)`. **The verdict is never used.**
  - **Skipped:** heldout2 rows, errored or stale rows, and rows whose chunk text doesn't match the run's `display_sha256`. It checks both the server-reported digest and sha256 of the text, so a server that misreports is caught too.
  - **Spans:** `planted_span` is build_demo's algorithm. A test loads `evals/demo/build_demo.py` and reproduces all 12 committed demo answers and spans byte for byte.
  - **Validity checks:** `check_validity` mirrors build_demo for the token, number and entity checks. It makes the three coverage-layer checks mechanical (no label and < 50% content-lemma containment in every shown chunk; the new chunk < 50%; the dropped line present in a shown chunk), and every demo item passes them.
  - **One clean source per question** (round 2): the smallest digest among its owner-correct answers; the rest are `skipped["duplicate_question"]`.
  - **Mutations:** each source gets every mutation that applies (typo, number, entity from another page, negation, fabricated sentence from another page, citation moved, guard dropped). Donors rotate by source digest.
  - **Output:** `negation_flip` items are written `owner_reviewed=False`. On synthetic rows built from D0, all 7 types are produced and every item passes its check.
  - `ChunkSource` is a Protocol. A mypy-checked test function proves `RagforgeClient` satisfies it.
- **Calibration sample:** `calibration_candidates(runs, questions, digest_for, *, kinds={generator})` takes answered, error-free, non-stale rows of E-G runs (round 3). The strata are tier (`grades["correct"]`: correct / incorrect / ungraded) × verdict (`status`). `digest_for(row)` must return `item_digest_of(...)` over the fetched texts.
  - `calibration_sample(cands, n=50, seed=0, exclude_digests=...)` uses Hamilton allocation with at least one per stratum when n allows. It is capped per stratum, never includes heldout2, is reproducible and doesn't depend on input order (hypothesis property tests).
  - Each sampled item gets one `criteria` label covering all its criteria.

## Integration steps (for the integrator / B3b runners / B4 page)
1. **Digest convention (load-bearing):** the label queue (B4) must compute `Label.item_digest` with `datasets.item_digest_of(question, answer, notes, facts, level)` over exactly the fields the judge item used. That means `facts` = the D1 fact texts for E-G, and `level` only for E-I. Otherwise calibration joins nothing. For `answer` labels that feed D3, use the same E-G fields.
2. **Runners:**
   - `async with await make_judge(cfg) as raw:` then `judge = CachingJudge(raw, JudgeCache(dataset_dir / JUDGE_CACHE_FILE))`;
   - record `judge.key` in `RunHeader.judge`;
   - put `[r.cache_key for r in results]` into `RunRow.judge_refs`;
   - catch `JudgeTransportError` and map `transient` to `ErrorClass.provider_transient` (re-run up to 2 times), else `ours`.
3. **CLI (B3b):** suggested subcommands, none of which exist yet:
   - `ragforge-eval judge-accept --set demo` runs the judge over the D0 generator fixture items and prints `acceptance(judge.calls)`;
   - `ragforge-eval seal <file>` → `seal_questions`;
   - `ragforge-eval overlap <file>` → `overlap_flags` with page texts from `/v1/eval/chunks`;
   - `ragforge-eval d3-build <runs...> --labels <file> -o verifier.json`;
   - `ragforge-eval calib-sample`.
   - Pass Gemini only with an explicit `--judge gemini --allow-third-party --judge-key-file …`.
4. **Makefile (integrator):** `typecheck` already loops over `packages/core/tests/module_*`, so `module_ev2` is covered. `ev_judge_support.py` is a test helper named with this module's prefix.
5. **Package data:** the prompt `.md` files live inside `ragforge_client/evals/prompts/` and are loaded with `importlib.resources`. Hatchling includes them under `packages = ["src/ragforge_client"]`. Confirm once with a built wheel if you ship one.

## Nothing needed outside my paths
No new dependency: httpx, pydantic and the stdlib (`fcntl`, `tomllib`, `secrets`, `importlib.resources`). hypothesis and respx are already dev dependencies. No Makefile or script change is required to test.

## Live checks the integrator must run (I couldn't, by the rules: no live stack or model)
1. **Ollama:**
   - `/api/chat` accepts `think: false` together with a JSON-schema `format` on the installed Ollama version and `qwen3:8b`;
   - `/api/tags` returns `digest` for the pulled model;
   - run the D0 acceptance (≤ 60 s per item, ≥ 99% valid JSON) and watch swap growth (≤ 2 GB) with evalrun's sampler, which is outside this module.
   - **Truncation guard:** confirm `/api/chat` with `stream: false` returns `prompt_eval_count` and that it counts the WHOLE prompt even when the rubric prefix comes from the KV cache (one item per call shares the system prompt). Send one item with a prompt longer than `num_ctx` and check the count is ≥ `num_ctx - 512` (outcome `truncated`). Since round 3, a character-length pre-check runs first (more than 3 chars per token of the window means no call), so a count that leaves out a cached prefix can no longer fail open. Still confirm that the count covers the whole prompt, so the second guard works for items between the two bounds. Also check that the D0 items are well under the pre-check bound, so it never fires on them.
   - Pull a cloud model (e.g. `gpt-oss:120b-cloud`) once and check its `/api/tags` entry carries `remote_host`/`remote_model`; `make_judge` must refuse it without `allow_third_party`.
2. **Gemini:**
   - the `open()` probe (`"ping"`, `maxOutputTokens: 1`) returns `modelVersion` on the configured model (a thinking model may return no text: that is fine, only `modelVersion` is read), and every scoring reply carries the same `modelVersion`;
   - `responseJsonSchema` is accepted by the configured model. Some older models only take `responseSchema`; if so, switch the key name in `GeminiJudge._complete`, which is one line;
   - `GET models/<m>` returns `version`;
   - `modelVersion` is present in replies.
3. **D3 on real data:** run `build_d3` on a private v2 run with real `/v1/eval/chunks` and check that `sha256(display_text) == display_sha256` holds for server chunks. It holds for all 30 D0 chunks. The owner then reviews every negation item and spot-checks 10% of the rest (ADR D3).
4. **Sealing:** seal the real `heldout2.toml` once the owner has finished rewording it, then load it with `require_seal=True`.

## For the integrator outside my paths
- **`schema.py` comment on `JudgeResult.cache_key` (r2 review P2):** add `"crit"` to the documented key, i.e. `digest({judge key, item digest, criterion[, batch key]})`. This is the formula the golden fixtures already use.
- **Runners / metrics (round 3):** an E-G item whose question has no facts now carries no `completeness` result. Metrics should treat that criterion as unscored for the item, not as a missing judge reply.
- **ADR-0017 §2 D1 step 4 wording (r2 review P2):** the overlap flag is the max Jaccard between the question and any one SENTENCE (or line) of a relevant page, not the whole page. A whole page has hundreds of lemmas, so a page-level Jaccard is near 0 for every question and could never flag. Please note this in the ADR (docs are not mine to edit).
- **EV-RUN/CLI:** `heldout2.toml` is now always seal-checked by `load_datasets`, so `cli.load_datasets` gets the bar for free. If a runner ever loads heldout2 rows from another file, it should pass `require_seal=True` whenever `Split.heldout2` is in the run's splits.
- **Runners:** `JudgeCall.outcome` has a new value `truncated`. Anything that switches on outcomes (a report, the Eval page) should show it as a judge failure, not a model failure.
- `ev_judge_support.py` (a helper, not a `test_ev_judge_*` file) was left unchanged this round; the new fakes live in the test files.

## Known gaps and decisions to confirm
- The entity-swap and identifier proposers are heuristics: capitalised words or acronyms, and dotted or underscored identifiers. On private notes some sources will get fewer mutation types, and the `skipped` counts and per-type totals show it. Swaps inside code blocks aren't excluded.
- The 50% containment threshold for "attributable" is a proxy for the server's ADR-0015 attribution, not the server's own rule.
- **Hazard:** the shared scratchpad directory is used by every parallel builder. Another module's `check.sh` overwrote mine, and once two mutants (`if False:`) were left in `judge.py`. I found them before handoff, restored them, and re-ran with an integrity check, and the final files are clean. Builders should prefix scratch files with their module id.
