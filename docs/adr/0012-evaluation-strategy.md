# ADR-0012: Evaluation strategy: recorded-Notion corpus, a passable regression gate, frozen floor, verifier evaluation

**Status:** Accepted · 2026-10-16 · Revised in rounds 3, 4 and 5

## Context
- Private Notion data can't be in git or CI.
- **Round 5:**
  - A non-inferiority CI gate at a −2-point margin with n = 100 fails neutral changes. If 10% of questions flip (5 up, 5 down), the SE ≈ 3.2 points and the CI lower bound ≈ −6. Holding −2 needs n ≈ 9604 × (fraction of questions that change), about 1k at 10% churn.
  - The headline "verified" feature had no evaluation of the verifier itself.
  - A Markdown corpus bypasses the Notion block normalizer.

## Decision
- **Corpus:** a synthetic "Acme Logistics" workspace (~150 pages) authored once in a throwaway Notion workspace. Its **API responses (block JSON, data-source rows) are recorded** into `evals/corpus/notion/`, and a `replay` connector feeds them through the **real Notion normalizer and chunker**. Markdown fixtures are kept only for the file connector.
- **Retrieval dataset:** ~100 reviewed questions in MVP (300 in Advanced). Each has **3 paraphrases** (the generated paraphrases are human-checked), expected anchors and a type; 15% unanswerable; 10 injection fixtures.
- **CI regression gate** (recall@5 and MRR over all question × paraphrase rows). It runs only when retrieval, ingestion, chunking or embedding paths change. Other PRs produce identical scores by determinism.
  - **Block on a significant drop:** a one-sided paired bootstrap clustered by question gives **p < 0.025 per metric** (Bonferroni across recall@5 and MRR), *and* the point drop is > 2 points.
    - The false-block rate is **approximately** ≤ 5%. The bootstrap is approximate when few clusters change, so with < 10 discordant questions the test falls back to an exact sign test.
    - A plain point-estimate rule at −2 would block ~20–26% of neutral PRs at 10% churn.
    - Power for a true 5-point drop: ~50% at 100 question clusters (MVP), ~85–90% at 300 clusters (Advanced).
  - The **path filter** includes the retrieval, ingestion, chunking and embedding code, plus `uv.lock`, model config and the container image digests.
  - **Warn** (a PR comment) on any point drop > 2.
  - **Block** if a metric is below `frozen.json` − 3 points. This catches cumulative drift that per-PR power misses.
  - **Override:** an `eval-override` label with a written justification, listed in the release notes.
  - **Determinism:** CI embeds with a pinned model in one pinned CPU container image, with embeddings cached by hash. Score changes come from code, not hardware.
  - Moving `frozen.json` needs a dedicated PR with a justification.
- **Verifier eval set:** ~300 labeled items (~150 unsupported; question, sources, answer, `supported: bool`, unsupported spans). It includes realistic failure modes: a fabricated citation, a claim from the right page but the wrong section, an outdated version, a numerical drift.
  - The report gives ground-check precision and recall per model.
  - **SLO on Wilson 95% lower bounds:** recall on unsupported ≥ 0.75 and precision ≥ 0.80. With ~150 unsupported items the CI half-width is about ±0.065.
  - The **headline metric** is the false-"verified" rate, P(unsupported | badge = verified). The eval set is ~50% unsupported by construction, so it isn't a production base rate. The rate is therefore reported **as a curve over base rates of 5–30%**, and optionally from locally labeled production samples (`make label-verdicts`, private, never committed).
  - A model below the SLO ships with the badge "unverified" (never "verified").
- **Report-only:** faithfulness, answer relevance, correct-abstain, injection resistance, rewrite/regenerate ratios, latency. Posted as a PR comment and stored in `eval_runs`.
- **Private set:** `make eval-private` before releases and model changes; `--calibrate` fits τ per collection into `collections.tau`.

## Consequences
- The gate is passable for neutral changes and still catches real drops of more than 2 points.
- The "verified" claim is backed by a measured verifier.
- Notion-specific normalization is covered by evals, not only by unit fixtures.
