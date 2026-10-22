# Handoff EV-UI: the Eval page (ADR-0017 §6)

Built against the LOCKED seams only: `runfile.read_run` / `iter_runs`, `RunSummary`
(`<run>.summary.json`), `GateVerdict` (`<runs dir>/gate-<surface>.json`), `Calibration`,
`JudgeResult`, `Label`, `LabelReader` / `LabelWriter` (interfaces.py), `EvalChunk` and the SDK's
`eval_chunks`. No locked file, contract, SDK, server, pyproject, Makefile or other docs file was
touched. Built and tested only on `evals/demo/**` (tmp copies) and synthetic data; nothing opens
`evals/private`.

## What was built

`packages/ui/src/ragforge_ui/`
- `eval_page.py`: the entry `eval_page()`: loads everything once per script run, shows load
  problems, and dispatches a horizontal radio of 7 views. Tests inject `EvalPaths`
  (`rf_eval_paths`), a `ChunkSource` (`rf_eval_chunks`) and a label store (`rf_eval_labels`)
  through session state, the way `app.BACKEND` is injected; the real page never sets them.
- `app.py` (additive only): `"Eval"` added to the sidebar page radio; `elif page == "Eval":
  eval_page()`. The exposure / Host / Origin guards still run first.
- `eval_view/`
  - `config.py`: `--eval-runs DIR`, `--eval-data DIR`, `--eval-key-file FILE` after `--` (or
    `RAGFORGE_EVAL_RUNS`, `RAGFORGE_EVAL_DATA`, `RAGFORGE_EVAL_KEY_FILE`). Defaults:
    `evals/private/runs`, data dir = its parent. `EvalPaths.under(runs, data)` applies the locked
    names (`labels.jsonl`, `judge-cache.jsonl`) plus `questions.toml`, `heldout2.toml`
    (`datasets.HELDOUT2_FILE`), `verifier.json`, `interview.json`, `calibration.jsonl`. A
    malformed flag (`--eval-runs` with no value) raises `EvalConfigError` (argparse
    `exit_on_error=False`), which the page shows as "Bad command-line flags", never a silent stop.
  - `data.py`: datasets are parsed by EV-JUDGE's own loaders (`datasets.load_questions` over
    `questions.toml` + `heldout2.toml` when present, `load_verifier_set`, `load_interview_set`),
    so a SEALED file (top-level `seal` key) and the separate sealed `heldout2.toml` read exactly
    as the CLI and metrics read them; a failed seal (`SealError`) is a problem line and leaves the
    questions unknown, so the held-out rule fails closed. Read-only loaders that degrade (a broken file is a problem line naming the file and
    the exception TYPE, never its contents). Runs are read per file with `read_run` (the same
    skip rule as `iter_runs`), so one file the OS refuses is a problem line, not an empty Runs
    list. `label_store(path)` is EV-JUDGE's `LabelStore(path)` (direct import). Labels are read
    through `read_labels`: when the store refuses the file (any invalid line raises
    `DatasetError`), the page shows the problem, still shows the good lines (`FileLabelReader`,
    line by line) and turns labelling OFF for that rerun, so no item is labelled twice. The page
    never writes `labels.jsonl` itself (a second writer without the store's lock could interleave
    with the CLI). `EvalContext.api_url` holds only `display_url(...)` (it is printed in errors).
  - `model.py` (pure): formatting; principle 1 (an untrusted judge metric shows "not trusted",
    never its number, CI or delta; one `hides_value` rule behind `value_text` / `ci_cell` serves
    every table, tile and caption; `calibration.*` metrics are EXEMPT: kappa and raw agreement
    measure trust, they are not judge results, and they explain an untrusted state);
    held-out policy, FAIL CLOSED (a row whose split can't be proven in a run declaring `heldout2`
    is hidden; a D3 item whose id or source id is a heldout2 question is held out, one linked to
    a dev question is dev, one linked to NO question is unknown, hence hidden when the run
    declares heldout2); compare rules (same eval kind, same dataset digest, same snapshot
    CONTENT digest; index digest may differ; demo vs snapshot refused); per-item flips with an
    exact two-sided McNemar p labelled exploratory, where "pass" is always the GOOD outcome: E-V
    rows split into `planted: caught at the span` / `planted: badge flipped` and
    `clean: verified clean` (never pooled), E-R into `relevant page in top 5` (need=any: the
    questions of hit@5) and `all relevant pages in top 5` (need=all); span-based `caught_layer`
    (incl. "caught, unlocated"); the SLO verdict on interval BOUNDS only (badge-flip recall
    `ci_low >= 0.75` and verifier-caused false-reject `ci_high <= 0.20`; point-only summaries are
    "not evaluable"); the blind `QueueItem` (question, answer, shown refs, item digest: no field
    can hold a judge score, reason or verdict); the queue is the STRATIFIED calibration sample
    (`datasets.calibration_candidates` + `calibration_sample`, n = `CALIBRATION_N`, seed 0, by
    answer tier and verdict, never heldout2), drawn from all labelable items without excluding
    labelled ones so the set is stable while the owner works through it (k of n, then done);
    newest row per item id only (resumed runs); a row with no recorded shown notes is never
    queued (as the CLI's `labelable`); calibration selection (newest record per
    criterion for the chosen judge key; other keys counted as stale).
  - `notes.py`: `shown_notes(refs, source)` fetches the shown chunks in one `eval_chunks` call and
    shows a text ONLY when the server's `display_sha256` AND our own SHA-256 of the text equal the
    run's recorded digest ("changed" / "missing" otherwise). `SdkChunkSource` reads the EVAL key
    file per fetch with `ask.read_key` (refuses group/other-readable files); its repr shows the
    display-safe URL only. Note text is never put in session state or a cache.
  - Views (thin): `runs.py` (Runs: overview table, header strip, gate state + checks + override,
    headline tiles with CI/n/method, all metrics; Compare, which NEVER reveals held-out rows and
    says how many it left out), `drill.py` (answer with the verdict badge, judge scores/reasons
    with trust, owner label, D3 planted span + caught-by, interview grade with misconceptions
    given/kept, retrieval stages, notes on demand; the only view with a held-out reveal),
    `labels.py` (blind queue over E-G runs ONLY, named neutrally in the
    picker (start time + a short hash of the file path; never the run id, arm or model); fetches the shown notes itself, digest-checked and never stored,
    and offers an item only when every note is unchanged; an item whose notes changed or
    vanished is skipped with its reason and remembered for the session by item digest (no text),
    so a rerun fetches only the item on screen; no eval key or a failed fetch = nothing offered, so
    faithfulness is never labelled without the notes; radios without a default, Enter in the
    note submits, batch id field, writes only via `LabelWriter`, never heldout2 even after a
    reveal, skips abstained/stale/unjudged rows and says how many), `quality.py` (Verifier: SLO
    line + badge, recall by mutation with the expected layer, caught-by layer, false-reject
    split; plus the false-verified block: the synthetic curve
    `verifier.false_verified@<rate>` beside the newest E-G run's owner-confirmed
    `generator.false_verified.confirmed` k/n (EV-METRICS names; a lower bound); Interview:
    ranking as k/n when it is a whole count, detection given vs kept, bands,
    Page's L + retest SD, mean score per level; Calibration: kappa + CI, raw agreement, leniency
    p, n, batch, confusion matrices, "Judge not trusted for X", the no-trusted-judge outcome,
    summary fallback when no records exist: every `calibration.<c>.*` value and CI shown even when
    untrusted, with its note (leniency p, label-count rule), "Judge not trusted for X" from the
    calibration flag (the judge metric's copy only when no calibration metric exists), and the
    no-trusted-judge outcome).
  - Text safety: every file string placed in Markdown goes through `md_literal`; answers through
    `answer_markdown` / `for_streamlit`; note text with `st.text`; tables are `st.dataframe`.
    Nothing uses `unsafe_allow_html` (asserted on every view).

Tests (`packages/core/tests/module_ui/`): `test_ev_ui_app.py` (AppTest, 52 functions),
`test_ev_ui_model.py` (pure + hypothesis, 38 functions), `test_ev_ui_data.py` (loaders, config,
notes, respx SDK path, 20 functions); 130 test cases with parametrization;
`test_ev_ui_support.py` holds the helpers and fakes (no tests). The golden check: every metric
of every `run-*.summary.json` is rendered with its value, n and method, AND every value of the
locked oracle `expected_summaries.json` for that run is what the page shows (so the test does not
merely echo whatever summary file it is given; ranking accuracy's n is not pinned, see below); the drill-down's per-item
catching layer, tallied over the golden verifier run, reproduces the oracle's `caught_by` shares
in `expected_summaries.json`. Ten mutants of the key rules (held-out filter, fail-closed split,
trust hiding, digest double-check, content-digest refusal, reveal confirmation, layer naming,
SLO bound, partial-label refusal, judge-key filter) were each killed by the suite; round 2 added
eight more (CI hiding, verifier fail-closed split, planted-item direction, need=any restriction,
note-state gate in the queue, labelling off on a refused label file, label-read degradation,
per-file run problem), all killed. Round 3 added eight more (calibration exemption from the
trust rule, no-shown-notes skip, newest row per item, heldout2.toml loading, session skip cache,
stratified sampling, neutral picker name, calibration flag first in the fallback), all killed.

## Integration steps
1. Nothing to install; no dependency added (pure Python + Streamlit's own pandas for tables).
2. `make ui` works as is (private defaults). To fetch note text, pass the eval key:
   `make ui ARGS="--key-file ~/.config/ragforge/ui.key --eval-key-file ~/.config/ragforge/eval.key"`.
   Suggested (optional) Makefile line for the help text: mention `--eval-runs/--eval-data/--eval-key-file`.
3. Optional demo target (integrator's call), browsing the golden fixtures without private data:
   `make ui ARGS="... --eval-runs evals/demo/fixtures --eval-data evals/demo"`. Note: the fixtures
   name the judge file `judge-results.jsonl` and keep `labels.jsonl` beside the runs, so judge
   reasons and owner labels show only for a private-style layout (or after `make eval-demo`
   writes one). Not wired, by design.

## Decisions the integrator must confirm (outside my paths)
- **Calibration records file (needed for the full panel)**: nothing in the client writes
  `Calibration` records today, so only the summary fallback is reachable: kappa, CI, raw
  agreement, trust and the metric notes, but NO confusion matrix and no leniency column. Please
  lock a path and have EV-METRICS / EV-RUN write it: the page reads `<data dir>/calibration.jsonl`,
  one `Calibration` per line, append-only, newest `computed_at` per (criterion, judge key) wins
  (one line in `EvalPaths.under` if the path differs).
- **EV-METRICS ranking-accuracy n**: the golden `run-interview.summary.json` says
  `interview.ranking_accuracy` n = 92; the D0 set has 16 tier pairs. Not an EV-UI defect (the
  page shows whatever n the summary has, as k/n once value x n is a whole count); my tests no
  longer pin n, so the B2 fix will not break them.
- **The UI depends on unlocked EV-JUDGE functions**: `datasets.load_questions`,
  `load_verifier_set`, `load_interview_set`, `HELDOUT2_FILE`, `calibration_candidates`,
  `calibration_sample`, `CALIBRATION_N`, `LabelStore`. Please treat their signatures as part of
  the seam (or lock them); the UI must keep reading datasets the way the CLI does.
- **`LabelStore(path: Path)`** from `ragforge_client.evals.datasets` is imported directly (it
  landed at that signature). If it changes, change only `data.label_store`.
- **Label item digest** = the `item_digest` of the row's judge results (via `judge_refs`), so a
  label always pairs with its judge results without the UI re-deriving EV-JUDGE's digest formula.
  Consequence: a row is labelable only after it was judged (the queue says "N not judged yet").
- **Held-out reveal record (please lock one)**: ADR-0017 D1 allows heldout2 ONE reveal, and
  there is no locked v2 reveal stamp to count it. Until there is, the page reveals only in the
  Drill-down (one row at a time), only after the owner types the run id, for the session only,
  and says it is NOT recorded; Compare flips and the label queue never show held-out rows, even
  after a reveal. Proposal: lock `<run>.reveal.json` (who / when / run id) in interfaces.py,
  written by the CLI and the page and honoured by both (the page would then refuse a second
  reveal).
- **The Eval page needs the daily UI key**: `app.main()` returns before the page radio when
  `_backend()` has no key, so Eval (local files, plus the eval key for notes) is unreachable
  without it. My app.py edit is additive only, so I left this; if wanted, the integrator can
  render Eval after the exposure/Host/Origin guards but before the key load.
- The exploratory McNemar p on Compare is computed locally (`model.mcnemar_exact`, exact binomial).
  Once `evals.stats` lands, it may replace it; behaviour must stay "exploratory, not a decision".

## Known gaps (not built)
- Labelling kinds other than E-G criteria: E-A `answer` labels (correct / invented), adjudication
  labels for judge-flagged verified answers and E-I question-quality labels. The UI queue offers
  E-G runs only and says "E-A answer labels and adjudications: use the CLI".
- The held-out reveal is NOT recorded (no locked `<run>.reveal.json`; see above), so the page
  cannot enforce ADR-0017 D1's single reveal; it limits the reveal to the Drill-down and says so.
- The false-verified view uses EV-METRICS's names (`verifier.false_verified@<rate>`,
  `generator.false_verified.confirmed`), which the D0 oracle does not define; please lock them.
- E-A / latency runs render generically (header, all metrics, drill-down), no dedicated view.
- Runs are re-read on every rerun (no cache): fine at D0/D-snap sizes; cache by mtime if slow.

## What the integrator must test live
1. `make ui` → sidebar "Eval": the Runs view lists the private runs; a run without a summary says so.
2. Drill-down on a v2 E-G run with `shown`: toggle the notes → text appears via `/v1/eval/chunks`
   with the eval key; edit a note in the snapshot (or use a stale run) → "changed since the run".
   With the owner's daily key as `--eval-key-file` the fetch must fail (eval tenant RLS).
3. Label queue on a judged E-G run with `shown` notes and the eval key: the notes show without
   a toggle; edit one shown note's page and its item is skipped ("whose shown notes changed");
   without `--eval-key-file` nothing is offered. Label one item → one new line in `labels.jsonl` written by
   the store while the CLI `label` runs concurrently (no interleaving); the queue advances; no
   heldout2 question ever appears. Hand-break one line of `labels.jsonl`: the queue says the
   file is unreadable, shows the good lines' progress and offers no form.
4. After EV-METRICS writes `calibration.jsonl`: the panel shows kappa CI / confusion / leniency
   and "Judge not trusted for X" for untrusted criteria.
5. On the real sealed private layout (`questions.toml` + `heldout2.toml`): no "unreadable"
   problem line, Drill-down lists the dev rows of a run declaring heldout2, and the label queue
   shows "The calibration sample: up to 50 items" with dev items only.
6. After `make eval-gate`: the Runs view shows the verdict, checks with n/method/MDE and any override.

## Round 3 (review round 2 findings)
- P1 sealed / heldout2.toml not loaded: fixed (EV-JUDGE loaders; AppTests seal a copy with
  `seal_questions` plus a separate sealed `heldout2.toml`: dev rows visible, heldout2 hidden, dev
  items labelable; an edited sealed `heldout2.toml` is "heldout2.toml: unreadable (SealError)"
  and the run fails closed).
- P1 untrusted calibration statistics hidden: fixed (`calibration.*` exempt; fallback warns from
  the calibration flags and shows notes; AppTest with an untrusted kappa asserts value, CI, raw
  agreement and note).
- P2s fixed: no-shown-notes rows skipped; stratified calibration sample; neutral E-G-only picker;
  newest row per item and a per-session skip cache; oracle assertions without the n pin;
  malformed flags shown as an error; false-verified synthetic-vs-real block. Still open: the
  reveal record and other label kinds (above).
