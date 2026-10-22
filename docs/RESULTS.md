# Results

This page shows what RAGForge's evals measured. Every table below is generated. The command
`ragforge-eval2 results --write` reads the runs listed in
[`evals/results.toml`](../evals/results.toml) and writes the blocks between the
`<!-- generated:... -->` markers. Do not edit those blocks by hand:
`ragforge-eval2 results --check` re-renders them and fails on any difference.

This page covers the numbers and their statistical limits only. What is not built or not
measured yet is listed in [DECISIONS.md](DECISIONS.md).

## Datasets

**Acme demo (public).** Fifteen synthetic Markdown pages about an invented company, "Acme
Shipping" ([`evals/demo/pages/`](../evals/demo/pages/)). They were written for the evals and are
not a sample of anyone's real notes. The demo set has:

- 20 questions, all in the `dev` split: 12 that one page answers, 1 that needs several pages,
  and 7 that the pages do not answer ([`evals/demo/questions.toml`](../evals/demo/questions.toml)).
- 20 verifier items: 8 clean answers and 12 copies with one planted error each
  ([`evals/demo/verifier.json`](../evals/demo/verifier.json)).
- 4 interview seeds, each with a gold, partial, wrong and misconception answer
  ([`evals/demo/interview.json`](../evals/demo/interview.json)).

You can rerun it with `make eval-demo`. It needs the live stack (`make up`) and a chat model.
The demo numbers below come from Google Gemini `gemini-3.5-flash-lite`, the model named in
[`evals/demo/arm.toml`](../evals/demo/arm.toml). On a local Ollama model the numbers will
differ. The run summaries (`*.summary.json`) and header sidecars (`*.header.json`) in
`evals/demo/runs/` are committed. The run files themselves (`*.jsonl`) are not: they hold local
database ids.

**Private (the owner's notes).** The owner's own Notion notes, frozen as an eval snapshot. This
page publishes only the exporter's numbers for it: no question text, note text, page titles or
ids. `evals/private/` is git-ignored, so a fresh clone has none of this data. On such a machine
`--check` skips the private tables and says so.

## How to read a table

- **Caption:** provider, model, dataset and split, the range of n, and the run date (UTC). When
  the run header does not record a model (interview runs, and the demo retrieval run, do not
  today), the caption says "model from the manifest". Retrieval tables are checked against the
  embedding and rerank models, not the chat model.
- **Value:** a share from 0 to 1 unless the row says otherwise. `n/a` means the run produced no
  value.
- **95% CI:** a two-sided 95% interval. `-` means no interval (see the Method column).
- **n:** the number of items behind the value. It differs between rows of one table.
- **Method:** how the interval was computed (next section).
- **Note:** a fixed explanation from [`metrics.py`](../packages/client/src/ragforge_client/evals/metrics.py),
  such as "bootstrap degenerate". The exporter keeps a note only when it matches a known
  template. Any other note is dropped.
- **"pending owner labels":** a judge-scored metric whose judge is not calibrated against the
  owner's labels yet, or a row whose labelled data does not exist yet. When the note says "no
  judge key in the run header", no judge ran at all for that run: the judge runs only after the
  owner's labels exist to calibrate it.

## What each metric means

The definitions live in [`metrics.py`](../packages/client/src/ragforge_client/evals/metrics.py)
and [ADR-0017 §3](adr/0017-eval-layer.md). Their point values are pinned by
[`evals/demo/fixtures/reference_metrics.py`](../evals/demo/fixtures/reference_metrics.py).

### Retrieval

Retrieval is scored per page. Chunks from the same page count once. Each metric is reported for
four stages: `keyword`, `vector`, `fused` (both lists merged), and `reranked`.

| Metric | Meaning |
|---|---|
| hit@k | Share of questions with a relevant page in the top k (k = 5, 10). Only questions that any one relevant page answers. |
| MRR | Mean of 1 / rank of the first relevant page (0 when none is retrieved). Same questions as hit@k. |
| nDCG@10 | Ranking quality against the graded relevance in the question file (2 = primary page, 1 = also relevant). All answerable questions. |
| recall@k | Share of the needed pages found in the top k. Reranked recall@5 and @10 use only questions that need several pages; fused recall@20 uses every answerable question and is the most the reranker can reach. |
| below_tau | For questions the notes do not answer: share whose top reranked score is below the abstain threshold tau. Needs tau in the run header. |

### Answers

`generator` runs ask each question once through the chat path. `application` runs do the same
on the full stack and also count tokens.

| Metric | Meaning |
|---|---|
| correct | Deterministic. An answerable question is correct when the answer does not abstain, cites an expected page, and matches at least 2/3 of the expected facts on word boundaries. An unanswerable one is correct when the answer abstains (`answer_grades` in [`runners.py`](../packages/client/src/ragforge_client/evals/runners.py)). |
| verified | Share of answers that got the server's `verified` badge. A server-written abstain counts as `verified`, so badge-flip recall (Verifier, below) is the direct test of the citation and ground checks. |
| abstain_correct | Share of questions where the answer abstained exactly when the question is unanswerable. |
| answer_rate | Share of questions answered rather than abstained. Shown next to the judge scores, so an arm that abstains a lot cannot look good. |
| tokens per question | Prompt plus completion tokens, averaged over questions. |
| judge: relevance, completeness, faithfulness | An LLM judge scores answered rows 0, 1 or 2. A judge counts as trusted only after calibration against the owner's labels: at least 50 per criterion, linear kappa at least 0.6 with a CI lower bound of at least 0.4, and no significant leniency. Until then the row shows "pending owner labels". |

### Verifier

The verifier eval sends clean answers and copies with one planted error (identifier typo,
number drift, entity swap, negation flip, fabricated sentence, citation moved, guard dropped)
through the server's verification pipeline.

| Metric | Meaning |
|---|---|
| span recall | Share of planted errors where a coverage unit or a located verifier claim overlaps the planted span. A flag elsewhere in the answer does not count. Also reported per mutation type. |
| badge-flip recall | Share of planted errors that turn a `verified` answer into a non-verified one. Counted only on sources whose clean answer the same run verified. |
| caught_by | Which layer overlapped the span: coverage only, verifier only, both, or missed. |
| caught, unlocated | The badge flipped because the verifier objected, but nothing overlaps the span. Not counted in span recall. |
| false_reject | Share of clean answers that did not get `verified`. Split into verifier-caused (the verifier objected) and coverage-caused (it did not). |
| false verified at a base rate | P(answer is wrong given it shows `verified`), from badge-flip recall and false reject, at an assumed 5, 10, 20 or 30% share of wrong answers. The note gives conservative bounds from the two CIs. They are not a joint interval. |

### Interview

| Metric | Meaning |
|---|---|
| ranking accuracy | Share of tier pairs (gold > partial, partial > wrong, gold > wrong, gold > misconception) that the grader ranks in the right order at the seed's own level. A tie counts as wrong. |
| misconception detected, given / kept | Share of misconception answers where the planted false sentence is among the grade's quoted misconceptions, before (given) and after (kept) the server's quote rule and verifier. |
| bands | Share of gold answers scored 7 or more, and of wrong answers scored 3 or less. |
| level test | Gold and partial answers are graded at all five levels, twice each. Page's L tests for scores that rise from Staff down to SDE-1. The row gives L and its exact one-sided p-value. A pooled sign test compares SDE-1 with Staff. |
| test-retest SD | Noise of one grade, in score points, from cells graded twice. |
| question checks | For generated questions: the question gives its answer away (leak), it is built on its seed chunk, and a code question refers to the code. |

### Latency

Time to first token (TTFT) and total answer time, in seconds, at p50 and p95. The first two
questions of the pass are cold and reported apart (`latency_metrics` in
[`metrics.py`](../packages/client/src/ragforge_client/evals/metrics.py)). Percentiles have no
interval.

## Confidence intervals: which method, and why

All intervals are two-sided at 95%. The code is
[`stats.py`](../packages/client/src/ragforge_client/evals/stats.py) (standard library only) and
`_conservative_ci` in [`metrics.py`](../packages/client/src/ragforge_client/evals/metrics.py).

- **Wilson** for a share of independent items (hit@k, correct, verified, abstain correct,
  answer rate, interview bands and detection, question checks). It stays inside [0, 1] and keeps
  near-nominal coverage at small n, where the plain normal (Wald) interval does not.
- **Cluster bootstrap** for per-question means (MRR, nDCG, recall: one cluster per question) and
  for the verifier metrics (one cluster per source answer). The planted errors in one answer are
  correlated, so resampling single items would make the interval too narrow. It uses 2000
  resamples with a fixed seed, so a summary is reproducible.
- **Exact Clopper-Pearson** when a bootstrap is degenerate (every resample equal, for example all
  ones) and the values are 0/1, and for interview ranking accuracy, where four seeds are too few
  clusters for a bootstrap. For other 0/1 verifier metrics each bootstrap bound is widened to the
  Clopper-Pearson bound when that is more conservative; the note says so. A degenerate bootstrap
  of a non-binary mean reports no interval.
- **Exact permutation** for the interview level test: the p-value of Page's L is computed over
  every within-item permutation, not sampled. The SDE-1 vs Staff comparison is an exact sign
  test.
- **Point** means no interval: latency percentiles, the test-retest SD, and the false-verified
  values (their bounds are in the note).

## Statistical limits

- **Small n.** The demo set has 20 questions, 20 verifier items and 4 interview seeds, so most
  rows rest on 20 items or fewer (see the n column). Intervals are wide, and a value of 1.000
  can still have a lower bound far below 1. Recall@5 and recall@10 on the demo set come from one
  question.
- **Many rows, no correction.** Each interval stands alone. Nothing here corrects for the number
  of rows. (The regression gate, `make eval-gate`, does adjust its own p-values with Holm; these
  tables do not.)
- **One labeller.** Every label (answer labels, planted-error reviews, judge calibration and
  adjudication) comes from one person, the owner. There is no second rater, so label
  agreement between people is not measured.
- **Model-specific.** The answer, verifier, interview and latency numbers were measured with
  `gemini-3.5-flash-lite` on the dates shown. Another model, including a local Ollama model, will
  give different numbers; no run on a local Ollama chat model has been measured yet, although
  Ollama is the default provider. Retrieval uses local embeddings and a local reranker on both
  providers.
- **Synthetic demo.** The demo pages and questions were written for the evals. They show that the
  pipeline and the metrics work end to end. They do not show quality on real notes.
- **Not on the demo set.** The demo runs had no judge configured (their summaries say "no judge
  key in the run header"), so the demo tables have no judge rows. The demo retrieval run header
  has no tau, so below_tau shows `n/a`.

## Tables: Acme demo

<!-- generated:demo-retrieval run_ids=retrieval-demo-20261021T173653Z digests=8602792ab46b0f7492faf08b276adb60073ff4205117e7cefa1af872f9e916ba -->
**Retrieval, Acme demo set.** Provider: Ollama embeddings, FlashRank reranker (in process). Model: nomic-embed-text, ms-marco-TinyBERT-L-2-v2 (model from the manifest). Dataset: demo (set `demo`, split dev). n = 0 to 13 (per row). Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| hit@5, keyword | 1.000 | [0.758, 1.000] | 12 | Wilson |  |
| hit@5, vector | 1.000 | [0.758, 1.000] | 12 | Wilson |  |
| hit@5, fused | 1.000 | [0.758, 1.000] | 12 | Wilson |  |
| hit@5, reranked | 1.000 | [0.758, 1.000] | 12 | Wilson |  |
| hit@10, reranked | 1.000 | [0.758, 1.000] | 12 | Wilson |  |
| MRR, keyword | 0.958 | [0.875, 1.000] | 12 | cluster bootstrap | bootstrap over questions |
| MRR, vector | 0.958 | [0.875, 1.000] | 12 | cluster bootstrap | bootstrap over questions |
| MRR, fused | 0.917 | [0.792, 1.000] | 12 | cluster bootstrap | bootstrap over questions |
| MRR, reranked | 0.958 | [0.875, 1.000] | 12 | cluster bootstrap | bootstrap over questions |
| nDCG@10, fused | 0.925 | [0.846, 0.987] | 13 | cluster bootstrap | bootstrap over questions |
| nDCG@10, reranked | 0.948 | [0.885, 0.990] | 13 | cluster bootstrap | bootstrap over questions |
| recall@5, reranked (need=all questions) | 0.500 | - | 1 | point (no CI) | bootstrap degenerate (every resample equal): no interval |
| recall@10, reranked (need=all questions) | 1.000 | - | 1 | point (no CI) | bootstrap degenerate (every resample equal): no interval |
| recall@20, fused (the reranker's ceiling) | 1.000 | - | 13 | point (no CI) | bootstrap over questions; bootstrap degenerate (every resample equal): no interval |
| unanswerable: top score below tau | n/a | - | 0 | point (no CI) | no tau in header |
<!-- /generated -->

<!-- generated:demo-generator run_ids=generator-demo-20261021T175028Z digests=7875363e02779a22b289ab61b387097c668eac262dc355d1df4b5bd546efd114 -->
**Answers (generator), Acme demo set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite. Dataset: demo (set `demo`, split dev). n = 20. Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| correct | 0.950 | [0.764, 0.991] | 20 | Wilson |  |
| verified badge | 0.900 | [0.699, 0.972] | 20 | Wilson |  |
| abstain correct | 1.000 | [0.839, 1.000] | 20 | Wilson |  |
| answer rate | 0.650 | [0.433, 0.819] | 20 | Wilson |  |
<!-- /generated -->

<!-- generated:demo-application run_ids=application-demo-20261021T174004Z digests=6d992b2721a2d8a87e4bb613e33327f1902d514a6e83eac6318d5c337d72369d -->
**Full chat path (application), Acme demo set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite. Dataset: demo (set `demo`, split dev). n = 20. Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| correct | 0.900 | [0.699, 0.972] | 20 | Wilson |  |
| verified badge | 0.900 | [0.699, 0.972] | 20 | Wilson |  |
| abstain correct | 1.000 | [0.839, 1.000] | 20 | Wilson |  |
| tokens per question | 1974 | [1378, 2515] | 20 | cluster bootstrap | bootstrap over questions |
<!-- /generated -->

<!-- generated:demo-latency run_ids=latency-demo-20261021T174336Z digests=d738185c41abe6167fcddea753fdb4b2a1b24cfb4ca11ec6498c29fa70153893 -->
**Latency, Acme demo set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite. Dataset: demo (set `demo`, split dev). n = 2 to 18 (per row). Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| time to first token, p50 | 1.69 s | - | 18 | point (no CI) |  |
| time to first token, p95 | 5.65 s | - | 18 | point (no CI) |  |
| total, p50 | 4.31 s | - | 18 | point (no CI) |  |
| total, p95 | 6.57 s | - | 18 | point (no CI) |  |
| cold: time to first token, mean | 2.19 s | - | 2 | point (no CI) | first questions of the pass, excluded above |
| cold: total, mean | 4.82 s | - | 2 | point (no CI) | first questions of the pass, excluded above |
<!-- /generated -->

<!-- generated:demo-verifier run_ids=verifier-demo-20261021T174717Z digests=9da010a4b144a6fa3870e81d40152c6470394c2cd2c6c046e54119dab15e9508 -->
**Verifier on planted errors, Acme demo set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite. Dataset: demo (set `demo`, split dev). n = 1 to 12 (per row). Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| badge-flip recall | 0.909 | [0.587, 1.000] | 11 | cluster bootstrap | on 7 clean-verified sources; a bound widened to the exact Clopper-Pearson bound |
| span recall (all mutations) | 0.917 | [0.615, 1.000] | 12 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| caught by: coverage only | 0.167 | [0.000, 0.484] | 12 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| caught by: verifier only | 0.417 | [0.152, 0.723] | 12 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| caught by: both | 0.333 | [0.099, 0.651] | 12 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| caught by: missed | 0.083 | [0.000, 0.385] | 12 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| caught, unlocated | 0.000 | [0.000, 0.285] | 11 | exact Clopper-Pearson | on 7 clean-verified sources; exact CP: bootstrap degenerate |
| false reject (clean answers) | 0.125 | [0.000, 0.527] | 8 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| false reject, verifier-caused | 0.000 | [0.000, 0.369] | 8 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| false reject, coverage-caused | 0.125 | [0.000, 0.527] | 8 | cluster bootstrap | a bound widened to the exact Clopper-Pearson bound |
| false verified at a 5% wrong-answer rate | 0.005 | - | 8 | point (no CI) | P(wrong \| verified) at base rate 5%; conservative plug-in bounds [0.000, 0.044] from the recall and false-reject CI bounds (not a joint CI) |
| false verified at a 10% wrong-answer rate | 0.011 | - | 8 | point (no CI) | P(wrong \| verified) at base rate 10%; conservative plug-in bounds [0.000, 0.088] from the recall and false-reject CI bounds (not a joint CI) |
| false verified at a 20% wrong-answer rate | 0.025 | - | 8 | point (no CI) | P(wrong \| verified) at base rate 20%; conservative plug-in bounds [0.000, 0.179] from the recall and false-reject CI bounds (not a joint CI) |
| false verified at a 30% wrong-answer rate | 0.043 | - | 8 | point (no CI) | P(wrong \| verified) at base rate 30%; conservative plug-in bounds [0.000, 0.272] from the recall and false-reject CI bounds (not a joint CI) |
| span recall: identifier typo | 1.000 | [0.292, 1.000] | 3 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: number drift | 1.000 | [0.158, 1.000] | 2 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: entity swap | 1.000 | [0.158, 1.000] | 2 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: negation flip | 1.000 | [0.025, 1.000] | 1 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: fabricated sentence | 1.000 | [0.158, 1.000] | 2 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: citation moved | 0.000 | [0.000, 0.975] | 1 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
| span recall: guard dropped | 1.000 | [0.025, 1.000] | 1 | exact Clopper-Pearson | exact CP: bootstrap degenerate |
<!-- /generated -->

<!-- generated:demo-interview run_ids=interview-demo-20261021T175559Z digests=44126f54dd9f33ad127ecd601ca219c5df70a509b23c6b9542d0451d5e09c365 -->
**Interview mode, Acme demo set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite (model from the manifest). Dataset: demo (set `demo`, split dev). n = 2 to 40 (per row). Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| ranking accuracy (tier pairs) | 1.000 | [0.794, 1.000] | 16 | exact Clopper-Pearson | tier pairs at the seed level; a tie counts as wrong |
| misconception detected, given | 0.750 | [0.301, 0.954] | 4 | Wilson |  |
| misconception detected, kept | 0.750 | [0.301, 0.954] | 4 | Wilson |  |
| band: gold answer scores 7 or more | 1.000 | [0.510, 1.000] | 4 | Wilson |  |
| band: wrong answer scores 3 or less | 1.000 | [0.510, 1.000] | 4 | Wilson |  |
| level test: Page's L statistic | 370 | - | 8 | exact | exact permutation p=0.06 (one-sided, staff -> sde_1 increasing) |
| level test: Page's L p-value | 0.06 | - | 8 | exact | P(L\* >= L) under within-block exchangeability |
| level test: SDE-1 vs Staff sign test p | 1 | - | 8 | exact | pooled paired sign test: +1 -0 =7 |
| test-retest SD (score points) | 0.3354 | - | 40 | point (no CI) | cells graded twice |
| generated question leaks its answer | 0.000 | [0.000, 0.324] | 8 | Wilson |  |
| generated question cites its seed chunk | 1.000 | [0.676, 1.000] | 8 | Wilson |  |
| code question references the code | 1.000 | [0.342, 1.000] | 2 | Wilson |  |
<!-- /generated -->

## Tables: private set

These tables hold numbers only. They are generated on the owner's machine.

<!-- generated:private-retrieval run_ids=retrieval-A-20261021T191959Z digests=ecabf6ffa68baa48691e06a204876e8994e94755befad27f554934c9bf0cade1 -->
**Retrieval, private set.** Provider: Ollama embeddings, FlashRank reranker (in process). Model: nomic-embed-text, ms-marco-TinyBERT-L-2-v2. Dataset: private (set `private`, split dev). n = 28. Run date (UTC): 2026-10-21.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| hit@5, reranked | 1.000 | [0.879, 1.000] | 28 | Wilson |  |
| hit@10, reranked | 1.000 | [0.879, 1.000] | 28 | Wilson |  |
| MRR, reranked | 0.982 | [0.946, 1.000] | 28 | cluster bootstrap | bootstrap over questions |
| nDCG@10, reranked | 0.940 | [0.893, 0.979] | 28 | cluster bootstrap | bootstrap over questions |
| recall@20, fused (the reranker's ceiling) | 0.958 | [0.911, 1.000] | 28 | cluster bootstrap | bootstrap over questions |
<!-- /generated -->

<!-- generated:private-generator run_ids=generator-A-20261022T071117Z digests=d8cfb881a65f5883543c16c566befe3e78e5f8632358600adfc2ee1b1b021472 -->
**Answers (generator), private set.** Provider: Google Gemini. Model: gemini-3.5-flash-lite. Dataset: private (set `private`, split dev). n = 0 to 32 (per row). Run date (UTC): 2026-10-22.

| Metric | Value | 95% CI | n | Method | Note |
|---|---|---|---|---|---|
| correct | 0.969 | [0.843, 0.994] | 32 | Wilson |  |
| verified badge | 0.781 | [0.612, 0.890] | 32 | Wilson |  |
| abstain correct | 1.000 | [0.893, 1.000] | 32 | Wilson |  |
| answer rate | 0.875 | [0.719, 0.950] | 32 | Wilson |  |
| judge: relevance (0 to 2) | pending owner labels | - | 0 | point (no CI) | no judge key in the run header |
| judge: completeness (0 to 2) | pending owner labels | - | 0 | point (no CI) | no judge key in the run header |
| judge: faithfulness (0 to 2) | pending owner labels | - | 0 | point (no CI) | no judge key in the run header |
<!-- /generated -->

<!-- generated:private-labels-pending run_ids=- digests=- -->
**Verifier and judge calibration, private set.** Not run yet: these rows need the owner's labels.

| Metric | Value |
|---|---|
| verifier: badge-flip recall | pending owner labels |
| verifier: span recall | pending owner labels |
| verifier: false reject, verifier-caused | pending owner labels |
| judge calibration: relevance, linear kappa | pending owner labels |
| judge calibration: completeness, linear kappa | pending owner labels |
| judge calibration: faithfulness, linear kappa | pending owner labels |
<!-- /generated -->

## How to regenerate

All commands run from the repository root.

1. Check the tables against the committed summaries. This needs no stack and no model:

   ```bash
   make eval ARGS="results --check"
   ```

   On a machine without `evals/private/runs`, the private tables are skipped with a note.

2. Rerun the demo evals. This needs the live stack and a model, and writes new runs with new ids
   under `evals/demo/runs/`. If `.env` runs another chat model than the one in
   [`evals/demo/arm.toml`](../evals/demo/arm.toml) (for example local Ollama instead of Gemini),
   set it there first: interview and verifier runs take their model from `arm.toml` when the
   server reports none, so a stale value would caption the tables with the wrong model.

   ```bash
   make up
   make eval-demo
   ```

3. Put the new run ids into [`evals/results.toml`](../evals/results.toml). The exporter never
   picks the latest run by itself. If the runs used another provider or model, also set the
   manifest's `provider` and `model` to match. Then write the tables and the header sidecars:

   ```bash
   make eval ARGS="results --write"
   ```

4. The README's headline tables are generated the same way, from their own manifest:

   ```bash
   make eval ARGS="results --write --doc README.md --manifest evals/readme-results.toml"
   ```

5. Run step 1 again, and the same command with `--check` for the README. Both should pass.
