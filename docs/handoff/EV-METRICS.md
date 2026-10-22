# Handoff EV-METRICS: pure metrics, statistics and the regression gate (ADR-0017 §3, §4, §5)

Status: built and tested against the LOCKED seams (`schema.py`, `runfile.py`, `interfaces.py`), the D0 demo set and the golden fixtures. I edited nothing outside my paths, and no locked seam needed a change.

Owned files:
- `packages/client/src/ragforge_client/evals/metrics.py`
- `packages/client/src/ragforge_client/evals/stats.py`
- `packages/client/src/ragforge_client/evals/gate.py`
- `packages/core/tests/module_ev2/test_ev_metrics_{support,golden,stats,gate,completeness,trust}.py`

`test_ev_metrics_support.py` is a helper module with no tests. It lives under the test prefix because that is the only test path I own.

## Round 3 (2026-10-21, after the blind review round 2 scored 7.8/10)
Every P1 is fixed, and so is every P2 that fits in my paths. New tests are in `test_ev_metrics_completeness.py` and `test_ev_metrics_trust.py`.

- **P1, an incomplete candidate passed the gate.** A new `coverage` check runs first on every surface. It returns **`not_evaluable`** in four cases: the candidate has no footer (crashed or unfinished), the baseline has no footer, the baseline has no rows, or any baseline item has no candidate row. The note says how many items are missing.
  - I chose `not_evaluable` over `block` on purpose. A truncated run is not a measurement, so an override must not be able to pass it, and `passes()` never passes `not_evaluable`.
  - This also covers the warn-only interview and latency surfaces, because the ADR's "warn-only" describes the statistical findings, not the input's validity.
  - Inside each paired check, a baseline item that the candidate has no value for blocks a block-severity check. Examples are an error of our own or a dropped `correct` grade. The note reads "k baseline item(s) have no candidate value".
  - A check that paired nothing because **neither** arm records the metric still only warns. One case is the ungraded generator fixture, which has no `correct` grade in either arm. There is no evidence of a regression, and blocking would make every ungraded A/A fail.
  - Tests cover an empty candidate, a 3-row candidate with no footer, a finished run with a dropped item, an unfinished baseline, and a missing grade. They run on retrieval, chat, interview and verifier.
- **P1, a judge was trusted after 2 labels.**
  - `stats.kappa_ci` now returns None when the interval has zero width, or when fewer than half the resamples define kappa (`KAPPA_MIN_DEFINED_SHARE`, enforced through a new `bootstrap_ci(min_defined=...)`).
  - `calibrate` also requires `MIN_CALIBRATION_LABELS = 10` pairs before it trusts a judge. The ADR collects 50 per criterion; 10 is the floor below which trust means nothing. The calibration note says so.
  - Tests: 2 or 3 perfectly agreeing labels give `trusted False` with `kappa_ci None`; a hypothesis property checks that a kappa CI is never zero-width; a test confirms the label floor is part of trust.
  - **Integrator / owner:** the fixture's 12 labels still make `completeness` trusted. If you want the ADR's 50 as the hard floor, change the constant to 50. The golden test `test_judge_metrics_carry_the_calibration_trust_state` would then expect False.
- **P1, the false-verified curve was missing.** It is now built in `metrics.py`:
  - `false_verified_rate(base_rate, miss, clean_pass)` applies Bayes to give P(wrong | verified).
  - `false_verified_curve(badge_flip_recall, false_reject, base_rates)` returns a `FalseVerifiedPoint` for each base rate, with conservative plug-in bounds. The curve is monotone in both inputs, so the upper bound pairs recall CI low with false-reject CI high. These bounds are not a joint CI.
  - The curve uses badge-flip recall (r4 P2), not span recall, and the total false-reject across every layer.
  - The verifier summary now emits `verifier.false_verified@{0.05,0.10,0.20,0.30}`. They use method `point` because the locked enum has no plug-in kind; the bounds are stated in the note.
- **P1, the confirmed false-verified k/n was missing.** It is built too:
  - `confirmed_false_verified(run, judge_results, labels)` returns k owner-confirmed adjudications over the n verified answers, with a Clopper-Pearson CI. The note marks k/n as a lower bound, because the judge screen misses some.
  - It is emitted as `generator.false_verified.confirmed` **only when** at least one adjudication label belongs to the run. "Not adjudicated yet" never prints as 0/n.
  - **Integrator (B4 adjudication flow):** an adjudication label is matched to a row by `label.item_digest` equal to the `item_digest` of a judge result that the row references. In other words, label the flagged answer under the judge's item digest, not just its question id.
- **P2, binary inferred from the values.** `clustered_mean` now takes a required keyword, `binary`. E-V passes `True`. Judge means and `iid_mean` pass `False`. A declared-binary metric that holds another value raises `MetricsError`. A 0..2 judge mean whose scores are all 1 reports no interval instead of a capped CP [0.40, 1.0].
- **P2, judge refs from another key.** `_judge_means` now excludes and counts refs whose `jr.judge` differs from the header's key. The note reads "k judge refs from another judge key excluded".
- **P2, kappa labelled `cluster_bootstrap`.** This is still open, because the fix needs a locked seam. **Integrator: add `Method.bootstrap` to schema.py**, then change the `method=` line in `metrics._calibration_metrics`. I left the code unchanged, as the finding asks.
- **P2, duplicate grades and the two-sided trend test.**
  - `interview_rows` now raises `MetricsError` when two item ids carry the same (seed, tier, level, replicate) grade. The gate reports that as `not_evaluable`.
  - `stats.sign_flip_test` gained `alternative="less"|"greater"`, and the level-trend warn now uses the one-sided p. A test builds 5 lowered blocks out of 8: the one-sided p is 1/32 and warns, where the two-sided 2/32 would have missed it.
- **Mutation check (scratch script `evmetrics_r3_mutate.py`):**
  - 15 hand mutations of the round-3 fixes. The first pass killed 13 of them.
  - One survivor was a dead branch: an n = 0 check with a non-empty baseline is always `missing > 0`. I removed the branch.
  - The other survivor was the one-sided trend test. I added the test above.
  - The re-run killed all 14 remaining mutations.

### Checks run (round 3)
- `uv run pytest packages/core/tests/module_ev2 packages/core/tests/test_eval_schema.py packages/core/tests/test_client_evalrun.py -q`: **354 passed, 1 failed.**
  - The failure is `test_ev_run_runners.py::test_generator_records_the_deterministic_correct_grade`, in another builder's in-progress file. That file changed on disk during my run. The test asserts the runner's grade key names (`cited_expected` vs `cited_by_mo...`) and never touches metrics or gate.
  - The same suite gave 338 passed, 0 failed a few minutes earlier.
- My 6 test files: **126 passed.**
- `ruff check` and `ruff format --check` are clean on my 9 files.
- `mypy` is clean on my 3 sources and, with `MYPYPATH`, on my 6 test files.
- `lint-imports`: 6 kept, 0 broken.

## Round 2 (2026-10-21, after the blind review scored 7.5/10)
Every P1 and every P2 is fixed in my paths:
- **P1, retrieval floor.** The floor is now **absolute and configured only** (ADR-0012 `frozen.json`), passed as `evaluate(..., floors={...})`. With no floor configured, `floor.*` passes with the note "no absolute floor configured". It is never the baseline's value any more. A new test runs an A/A pair with real noise (one question demoted between the runs): no paired test and no floor fires. The test that enshrined the old behaviour is replaced by two tests: one demoted question passes, and a configured floor blocks exactly below floor − 3.
- **P1, E-V SLO at tiny n.** `metrics.clustered_mean` now never reports a zero-width percentile CI. For binary items, each bound is the more conservative of the cluster-bootstrap bound and the exact Clopper-Pearson bound on the item counts. When the bootstrap is degenerate or there is only one cluster, the method is reported as `clopper_pearson`. A degenerate non-binary bootstrap reports **no interval** (`point`, with a note). The tests:
  - 3/3 flipped fails the SLO (CP low 0.292);
  - 0/2 false rejects fails ≤ 0.20 (CP high 0.842);
  - a hypothesis property: a binary clustered CI is never narrower than CP.
- **P2, E-V hard floor in the chat gate.** New `gate.all_pass(verdicts)`. See §3 for how the CLI must use it.
- **P2, kappa label.** The note now says "iid pairs bootstrap over labelled items (not clustered)". The locked `Method` enum has no plain `bootstrap`, so the field still says `cluster_bootstrap`. **Integrator: consider adding `Method.bootstrap` to schema.py** and switching `metrics._calibration_metrics` to it.
- **P2, comparability and stale rows.**
  - A retrieval run with the retrieval config in only one arm is now `not_evaluable`.
  - Items stale in either arm are excluded and counted in each check's note ("k stale item(s) excluded").
- **P2, latency order.** The new `metrics.cold_and_warm_rows` orders items by their FIRST row in the file, so a retry appended at the end keeps its original position and is still cold. The values come from the latest attempt. `total_s == 0.0` is kept (`is not None`). The run file has no explicit order field, so this is the best the locked schema allows.
- **P2, process.** Nothing more to do. This round I used one uniquely named scratch script, `evmetrics_r2_mutate.py`, and touched only my own files.
- **Mutation check of the round-2 fixes:**
  - 7 of 8 hand mutations were killed: baseline-relative floor, no CP widening, degenerate bootstrap kept, file-order latency, a 0.0 total dropped, the one-arm config check removed, `all_pass([])` passing.
  - The survivor is equivalent: stale ids are already dropped by `select_rows` before pairing, so the explicit exclusion only drives the note.
- **Effect on D0 (expected, not a bug).** The fixture verifier's badge-flip bound is now 0.555 (it was 0.625) and its false-reject upper bound is 0.53, so D0 can never certify a verifier model. The verifier A/A acceptance test uses a synthetic D0 ×3 set: 24 clean items, 36 planted, with the dataset digest recomputed.

## Checks run (2026-10-21)
- `uv run pytest packages/core/tests/test_eval_schema.py packages/core/tests/test_client_evalrun.py packages/core/tests/module_ev2/test_ev_metrics_*.py -q`: **132 passed** (92 of them mine; round 2).
- `uv run pytest packages/core/tests/module_ev2 -q` (the whole shared directory): all of mine pass. During the run, failures appeared only in another builder's in-progress `test_ev_judge_*` files.
- `ruff check` and `ruff format --check` are clean on my 7 files.
- `uv run mypy` on my 3 sources is clean, and so is `MYPYPATH=packages/core/tests/module_ev2 uv run mypy` on my 4 test files.
- `uv run lint-imports`: 6 kept, 0 broken.
- **Mutation check:** I applied 16 hand mutations, one at a time, and the tests killed all 16:
  - ties counted as correct ranking;
  - span overlap made inclusive;
  - badge flip counted on every source;
  - first attempt wins instead of the latest;
  - leniency ignored;
  - coverage excluded from verifier-caused false rejects;
  - McNemar made one-sided;
  - Holm's running maximum removed;
  - Page's L tail made strict;
  - ties not averaged in ranks;
  - bootstrap that does not resample;
  - an improvement that blocks;
  - content digest ignored;
  - floor ignored;
  - an override that lets not_evaluable pass;
  - transient errors not excluded.

## 1. What was built

### `stats.py` (pure, stdlib only, seeded)
- **Binomial intervals:**
  - `wilson`.
  - `clopper_pearson`: exact, found by bisecting the binomial tails in log space. A test checks it against the definition in exact `Fraction` arithmetic.
  - `binom_cdf`.
- **Bootstrap:**
  - `bootstrap_ci`: generic percentile bootstrap.
  - `cluster_bootstrap`: resamples clusters, and the statistic is the pooled item mean.
  - `quantile`: type 7.
- **Paired tests:**
  - `mcnemar_exact`;
  - `sign_test`;
  - `sign_flip_test`: exact over all 2^m sign patterns when m ≤ 16, otherwise seeded Monte Carlo with an add-one estimator;
  - `holm`.
- **Page's L:** `average_ranks`, `page_l`, `page_l_test`. The p-value is **always exact**: the null distribution is the convolution of the per-block distributions over k! permutations each, and since ranks are multiples of ½, 2L is an integer. No Monte Carlo is needed for k ≤ 8.
- **Agreement:** `linear_weighted_kappa` and `kappa_ci` (bootstrap over pairs).
- **MDE:**
  - `mde_from_discordance`: the normal approximation for McNemar, floored at the exact test's discreteness `min_discordant_for_significance(alpha)/n` and capped at 1;
  - `mde_from_paired_sd`.

### `metrics.py`
- **`summarize(run, datasets, labels, judge_results) -> RunSummary`:** dispatches on `eval_kind` and returns metrics sorted by name.
- **Row hygiene (`select_rows`):** the latest attempt per item wins. Stale rows are excluded and counted. Errored rows are excluded and counted by class. An output-less row with no error counts as an `ours` error.
- **Refusal:** `MetricsError` if the passed dataset's digest differs from `header.dataset.digest`, if a row's output has the wrong kind, or if an item id is unknown.
- **Interval methods:**
  - Wilson for binary shares (hit@k, below_tau, answer_rate, raw agreement, detection, bands);
  - a bootstrap with one cluster per question for non-binary means (MRR, nDCG, recall);
  - a **bootstrap clustered by `source_id`** for every E-V metric;
  - Clopper-Pearson for `interview.ranking_accuracy`;
  - a pairs bootstrap for kappa;
  - `exact` for Page's L.
- **Trust:** judge-derived metrics (`generator.judge.*`, `calibration.*`, `interview.question.judge.*`) carry `trusted` from `calibrate()`. Trusted means kappa ≥ 0.6 AND CI low ≥ 0.4 AND no significant sign-test bias of judge − owner in either direction at 0.05. The E-I question-quality judge also needs n ≥ 40 labels, otherwise it is reported as "uncalibrated".
- **Judge replies:** invalid replies (`score=None`) and missing refs are counted in the note, never imputed. Calibration uses only results whose `JudgeKey` equals the run header's key, with the newest result per item and the newest label per (item_digest, kind).
- **Exposed building blocks** (used by the gate, and usable by the Eval page):
  - `select_rows`, `retrieval_item_values`, `verifier_rows`, `interview_rows`;
  - `ranking_outcomes`, `detection_outcomes`, `level_blocks`, `retest_sd`;
  - `calibrate`, `calibration_pairs`, `latest_labels`, `metric_map`.

### `gate.py`
- **`evaluate(baseline, candidate, aa_runs, datasets, *, floors=None) -> GateVerdict`:**
  - The locked positional signature is unchanged; `floors` is an optional keyword.
  - **not_evaluable** when any of these differ:
    - eval kind, surface or dataset digest;
    - snapshot presence or content digest (the index digest may differ, so a re-embed stays evaluable);
    - chunker version.
  - It is also not_evaluable when the metric layer refuses the dataset.
- **Pairing:** by item id. Items with a provider-transient error in either arm are excluded and counted in the note.
- **Retrieval:**
  - exact McNemar on reranked hit@5 and a sign-flip test on reranked MRR, Holm across the two;
  - block when adjusted p < 0.025 AND the drop is > 2 points;
  - plus `floor.*` checks that block below floor − 3 points;
  - plus `aa.retrieval_identical` (warn).
- **Chat (generator/application):** exact McNemar on `correct` (from `grades["correct"]`) and on verified, Holm at 0.05. `errors.ours` is a hard floor.
- **Verifier:**
  - exact McNemar on the planted items (badge flipped);
  - the absolute SLO on the candidate: badge-flip recall CI low ≥ 0.75 and verifier-caused false-reject CI high ≤ 0.20.
- **Interview:** warn only. McNemar on ranking pairs and on detection_kept. The level-trend check warns only when the baseline's Page's L is significant AND the candidate's per-block L is significantly lower (paired sign-flip over blocks).
- **Latency:** warn only. A sign-flip test on paired warm `total_s`.
- **Every check prints n, its method and an MDE.** The MDE comes from the consecutive comparable A/A pairs. When it is n/a, the note says why: absolute thresholds, a rank statistic, or an identical A/A pair on a continuous metric.
- **Completeness (round 3):** an unfinished or item-dropping run is `not_evaluable` (the `coverage` check). A baseline item with no candidate value blocks a block-severity check. A metric that neither arm records (n = 0) warns, and never passes silently.
- **Outcome order:** not_evaluable > block > warn > pass.
- **Override:** `verdict.override` comes from the candidate header. The outcome is never rewritten. `passes(verdict)` returns True for pass or warn, and for block only with an override; never for not_evaluable.
- **`format_verdict(verdict)`:** the CLI text, one line per check.

## 2. Decisions the integrator should confirm
1. **Floor − 3 (changed in round 2).** The floor is ADR-0012's `frozen.json`: absolute, and moved only by a dedicated PR. The gate CLI must load it and pass `evaluate(..., floors={"retrieval.reranked.hit@5": x, "retrieval.reranked.mrr": y})`. Without it, `floor.*` passes with the note "no absolute floor configured". **Integrator: add `evals/<dataset>/frozen.json` (or similar) plus a CLI flag or loader; the CLI is outside my paths.**
2. **The verifier SLO is a hard block, even on A/A.** On the fixture verifier run (10 eligible planted items, 8 clean), badge-flip CI low is 0.555 and the false-reject upper bound is 0.53 (both CP-widened), so the fixture's A/A blocks **on the SLO only**. Certifying "≤ 0.20" with zero false rejects needs at least 17 clean items, and "≥ 0.75" with all flipped needs at least 13 planted items; the paired test passes (this is tested). The A/A acceptance test for the verifier uses a synthetic run that meets the SLO. If "A/A must pass" should cover the verifier surface at small n, the SLO checks could become warn while n is below some threshold. That is the integrator's or owner's call.
3. **Extra metric names beyond the oracle** (pinned in `test_ev_metrics_golden.EXTRA_NAMES` for the fixture runs):
   - `generator.verified`, `generator.abstain_correct`, and `generator.correct` (only when rows carry `grades["correct"]`);
   - `interview.level.page_l.p_value` and `interview.level.sde1_vs_staff.sign_p`;
   - for kinds the fixtures do not cover: `application.{verified,abstain_correct,correct,tokens_per_question}`, `latency.{ttft,total}.{p50,p95}` and `latency.cold.{ttft,total}.mean` (the first 2 rows in file order count as cold);
   - `interview.question.<bool grade>` and `interview.question.judge.<criterion>`.
4. **Oracle edge cases** (the oracle would crash or misbehave here; none of them occurs in the fixtures, and all 60 oracle values match to 1e-9):
   - a Page's L block with a missing cell is dropped rather than shortened;
   - an unanswerable question with `top_rerank_score=None` counts as below tau;
   - no tau in the header gives `below_tau` = None.
5. **`n` is the true denominator**, for example 12 for need=any hit@5, not the run's 20 rows. The `run-*.summary.json` files in fixtures/ are placeholders (point values, n = row count). Their writer should regenerate them with `metrics.summarize`; I did not touch them because they are locked fixtures.

## 3. Integration steps
- Nothing to apply outside my paths: no new dependency, no Makefile change. `make typecheck` already covers `module_ev2` through its `module_*` loop.
- Runners (B3) write `<run>.summary.json` like this:

  ```python
  metrics.summarize(read_run(p), datasets, label_store.latest(), judge_results).model_dump_json()
  ```
- The gate CLI:

  ```python
  v = gate.evaluate(base, cand, aa, ds)
  print(gate.format_verdict(v))
  ```

  It should write `gate-<surface>.json` and exit non-zero when `not gate.passes(v)`.
- Pass the run files as `read_run` returns them, keeping the footer. A run without a footer is `not_evaluable` by design, so the gate must run only on finished runs.
- For the gate's chat checks, the E-G/E-A runner must record `grades["correct"]` as a bool. Without it, `chat.correct` warns with n = 0.
- **The E-V hard floor on a chat change (ADR §5).** `evaluate` judges one run pair of one kind. On any chat or verifier change, the CLI must also evaluate the verifier-surface pair (E-V baseline vs a fresh E-V run) and exit non-zero unless `gate.all_pass([chat_verdict, verifier_verdict])`. Both verdicts should be written and printed.

## 4. What to test live
- A real A/A pair on the snapshot for each surface: retrieval must be identical (`aa.retrieval_identical` passes), and the printed MDEs should look sane.
- On the real D3 set (about 40 clean and 100 planted items), whether the verifier model meets the SLO. The CIs will be much tighter than on D0.

## 5. Incident during the build (resolved)
Another builder and I both wrote a script named `mutate.py` in the shared scratchpad, and I ran theirs by mistake. It mutates and restores `judge.py` and `datasets.py`, and it overlapped with their own run. For a few seconds after that, `judge.py` and `datasets.py` held mutated lines (`if False:`). Re-checked at 18:15: their originals were back and no mutation lines were left. My own files were verified intact. I have moved to uniquely named scripts. **The owner of judge/datasets should still re-run their suite once.**
