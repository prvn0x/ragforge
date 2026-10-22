# ADR-0017: The eval layer: retriever, generator, verifier, interview and application evals, a calibrated judge, a gate and an Eval page

**Status:** Accepted, revision 5 · 2026-10-21 · Approved to build after round 4 · Extends [ADR-0012](0012-evaluation-strategy.md) (keeps its statistics; replaces its corpus plan) · Blind reviews: r1 6.5 (4 P0s), r2 7.8 (2 new P0s, 10 P1s), r3 8.0 (1 new P0, 5 P1s), r4 **approve to build** (1 P1 folded into lock step 1, 5 P2s). Every finding is addressed and mapped in the tables at the end.

**Owner decisions (2026-10-21):**
- **Judge:** a local model through Ollama, with Gemini Flash as the fallback. Calibration decides which one is trusted.
- **Snapshot:** a frozen eval snapshot collection of the notes, approved.
- **Demo set:** a committed synthetic demo set, approved.
- **Exporter:** dropped (proposed in revision 1; the reviewer agreed).
- **Held-out set:** model-drafted, then reworded and approved by the owner.

## Context
- **What exists** (`ragforge_client/evalrun.py`, 1,680 lines): answer-quality evals on the owner's private set (`evals/private/questions.toml`: 32 questions, 16 held out and already revealed twice, 5 unanswerable).
  - Grading: correct% is deterministic (no abstain, an expected page cited, at least 2/3 of facts matched on word boundaries). verified% comes from the server's verdict. invented% comes from owner labels only.
  - Statistics: Wilson CIs and paired McNemar.
  - Held-out protection: one `--reveal` only.
- **What is missing:**
  - retrieval metrics;
  - a calibrated judge;
  - a measurement of the verification guarantee itself;
  - interview-mode evals;
  - a regression gate;
  - a page to see results on.
- **Constraints:**
  - **Private:** notes stay on the machine except through the configured providers, and never go into git.
  - **Cost and hardware:** the Gemini free tier allows Flash-Lite at a modest RPM and Flash at about 20 requests/day; one M1 Pro with 16 GB.
  - **Users:** one owner, who can label about 50 items.
  - **The corpus changes daily:** the owner edits Notion and sync re-chunks.
  - **Builders:** the agents that build this must never read the private data.

## Decision

### 0. Principles
1. **Deterministic first.** A model judges only what code can't check. A judge number is shown only after calibration against the owner.
2. **Measure the guarantee as shipped.** A `verified` badge comes from coverage/attribution AND the LLM ground check (ADR-0015). E-V measures the whole pipeline and reports which layer caught each error.
3. **A fixed corpus for anything compared over time.** Gates and baselines run on a frozen snapshot, never the live notes.
4. **Every number has a definition, an n, a CI, a method and, for judge numbers, a trust state.** Small-n limits are printed, never implied away.
5. **Builders never touch private data.** Everything is built and tested on the committed demo set (§2 D0) and its golden offline fixtures.
6. **Stable identity (r2 P0-A).** Datasets never store database ids, because `document_id = uuidv5(tenant, collection, …)` changes with each copy, tenant and install. They store:
   - for a page: `page_key`:
     - Notion pages: `(notion, external_id)`, the Notion page id; the snapshot copy job writes `source_kind = 'notion'` and the original `external_id` on its upload-owned documents (the CHECK allows it);
     - files (D0): `(file, metadata.filename)`, because the upload handler's `external_id` is a random upload id per install; filenames are unique per collection;
   - for a chunk: `chunk_key = (page_key, heading_path, display_sha256, occurrence)`, where `occurrence` is its rank among equal keys in ordinal order, so two identical chunks under one heading don't collide.
   The harness resolves keys to ids at run time through the eval seam (§1). A key that no longer resolves, or whose `display_sha256` changed, marks the item **stale**: it is reported, never silently re-mapped. `resolve` returns any candidate under the same `(page_key, heading_path)`, so the owner can re-approve the item instead of re-authoring it.

### 1. Server seams: additive, eval-scoped (lock before building)
A new API scope, `eval`.
- **Who holds it:** only a dedicated eval key, whose scopes are `eval` + `chat` + `read`. **It belongs to a dedicated eval tenant** (r3 P0-C), not the owner's tenant. Key creation refuses to combine `eval` with an MCP key.
- **The eval tenant** holds the snapshot and the demo set. The owner's daily UI, `make ask` and MCP never see either: no duplicate pages in search, citations or interview coverage. A test proves the owner's keys list neither. `/eval/verify`, `chunks` and `resolve` run under the eval tenant's RLS, so they **can't touch the live notes at all**.
- **Switch:** the whole surface sits behind `RAGFORGE_EVAL_API`, on in dev and off by default otherwise.
- **Ownership:** `Scope` and `ApiScope` are frozen and mirrored, so the **integrator** adds the scope in the lock step.

Every endpoint below is additive and requires the scope.

| Seam | Purpose | Shape |
|---|---|---|
| `POST /v1/eval/verify` | run the **full** ADR-0015 pipeline on a supplied answer: normalise → check_citations (strict as configured) → attribute → ground check → settle | in: `answer`, `shown_chunk_ids` (ordered = S labels; each must lie in `collection_ids`), `collection_ids`.<br>out: the final `status` and flagged claims with reasons and spans, **plus the per-layer evidence BEFORE the server merges it** (r3 P1-c):<br>• `coverage_units`: residue, uncited and code-unbacked, each with its span and reason;<br>• `verifier_claims`: each with its span, or `null` when the server couldn't place it;<br>• `verifier_ran`, `verifier_verdict`, `verifier_conclusive`;<br>• the **verifier model and prompt@version that actually ran**.<br>`caught_by` is computed **in the client** from this evidence (§3 E-V). Chunks are re-read through RLS. Under the per-principal rate limit. |
| `GET /v1/eval/chunks?ids=` (r2 P1-1) | the shown TEXT, for the client judge, the drill-down and the label queue | `[{chunk_id, document_id, page_key, heading_path, display_text, display_sha256}]`. RLS read. The client **checks the digest matches** the run's recorded digest before judging or labelling |
| `POST /v1/eval/resolve` | stable keys to current ids (principle 6) | in: `page_keys`/`chunk_keys` + `collection_ids`. out: ids, or `stale`/`missing` per key |
| `POST /v1/query` with `stages=true` | per-stage retrieval | adds `stages: {keyword: [...], vector: [...], fused: [...], reranked: [...]}`. Each is a list of `{chunk_id, document_id, rank, score}`; ties are broken by `chunk_id` |
| chat `done` event: `shown` (eval scope only, requested with `?trace=shown`) | what the model was SHOWN | `[{label, chunk_id, document_id, display_sha256}]`. Ids and digests only, never text |
| interview question: `chunk_ids` (eval scope only) | target a seed note for the generated question-quality items | `chunk_ids` (each within `collection_ids`) overrides selection. The token is a normal v3 token |
| interview grade: frozen D4 items (r3 P1-e) | grade a FROZEN question (no token can exist for it) at any of the 5 levels | **An eval-scope notes-mode grade needs no token, and its `level` is free.** This is today's optional-token path, **exempted explicitly for the `eval` scope** when enforcement lands. The grade route checks the eval scope first. Every other key keeps the v3 binding unchanged. The chunks are still re-read through RLS (eval tenant only), and general mode still requires its token. |
| per-surface prompt versions | `chat`, `verifier`, `interview`, `judge` | `PROMPT_VERSION` split into `prompt_versions: {surface: "name@v,..."}` on `done` and in logs. The old field stays for compatibility |
| eval snapshot (r2 P1-5, r3 P0-C/P1-b) | §2 D-snap | `ragforge-eval snapshot create` runs an admin job **as `rf_system` through a named ADR-0013 definer function**. It copies the owner's live collection **into the eval tenant**, as collection `eval-snapshot-<date>`:<br>• **Source:** its own frozen upload-style source (one synthetic `uploads` row per snapshot, no blob processing, no ingestion jobs enqueued). It is not owned by the live connector, whose `ON DELETE CASCADE` would delete it on reconnect. Never synced; excluded from exclusions.<br>• **What it copies:** chunks, **their embedding rows** (no re-embed), and `document_links`. Each page keeps its `page_key`.<br>• **Two digests:** a **content digest** (the sorted `page_key` set plus display texts), which the gate requires to match; and an **index digest** (`chunk_key`, `embed_sha256`, space id), recorded per run and allowed to differ between arms.<br>• **Embedder changes:** `ragforge-eval snapshot reembed --space <new>` re-embeds the snapshot from its stored chunk text, so an embedder change is evaluable.<br>• **Chunker or normaliser changes are NOT evaluable on the snapshot.** It stores chunks, not raw source. The path filter prints "warn: not evaluable" for those paths. Recorded raw-source replay (ADR-0012's idea) is a Phase 4 item. |

### 2. Datasets
**D0 · Demo set (committed, synthetic):** `evals/demo/`.
- About 15 Markdown pages for an invented "Acme Shipping" (Kafka, Redis, API, incidents, a design page with code), ingested through the **file connector**.
- About 20 Q&A questions: 4 unanswerable and 3 near-misses.
- About 20 planted verifier items.
- 4 interview seeds, each with 4 frozen answers.
- `make eval-demo` creates a demo tenant, uploads the pages, runs all five evals and fills the Eval page. It is a **smoke test that needs a live model and stack, not CI.**
- **Golden offline fixtures (r2 P1-6, committed in the lock step):** v2 run files for every eval kind, synthetic D5 labels, judge results and the expected summaries for D0. **The expected summaries are computed by hand or with an independent reference script, never with B2's code**, so the oracle isn't circular. B2 and B4 build and test against these with no model and no stack. The keys in D0 are file paths (principle 6), so they hold across installs.
- It's what builders and CI test against, and what someone who clones the repo can run. The recorded-Notion "Acme" corpus of ADR-0012 stays deferred.

**D-snap · Eval snapshot:** the private notes, frozen (owner approved). **Every** private run targets the snapshot: D1, D3, D4, E-A, and the v2 runs that source D3. Every run header records the snapshot digest, and the gate **refuses** to compare runs with different digests. Refreshing the snapshot is an explicit command: it starts a new baseline and marks changed items stale (principle 6).

**D1 · Q&A set:**
- **Additive fields:**
  - `relevance = { "<page_key>" = 2 | 1 }`: graded, keyed by the stable page key (principle 6); titles are shown for humans only;
  - `need = "any" | "all"`: whether any one of the pages answers it, or all of them are needed.
  - The existing `pages` stays for citation grading.
- **Splits:**
  - the burned `heldout` is **reclassified as `dev`**;
  - a new **`heldout2`** of about 30 questions;
  - about 5 more unanswerable questions and about 5 near-misses (adjacent topics the notes don't cover).
- **Drafting `heldout2`:**
  1. Sample pages uniformly per collection; the procedure is recorded.
  2. The model drafts a question with its facts and relevance.
  3. **The owner rewrites each question in their own short, acronym-heavy style**, and edits `facts` and `match`.
  4. The harness reports **question↔page lexical overlap** (Jaccard on content lemmas) and flags any question above 0.5 for rewording.
  5. The file is sealed: a digest goes in the header, and any edit fails the bar (existing rule). Builder agents must never read it; this is in the workflow prompt.

**D3 · Planted-error verifier set** (built only from v2 runs, which record `shown`):
- **Clean items:** answers the owner labelled **correct, whatever the server's verdict**. This removes the selection on the outcome. Each item freezes its `shown` **chunk keys** (principle 6).
- **Each mutation stores its character span** in the mutated answer (r2 P0-B).
- **Mutations:** each has a declared **expected catching layer** and a **mechanical validity check**:

  | Mutation | Expected layer | Validity check |
  |---|---|---|
  | identifier typo (one edit in a whole identifier) | verifier (the 3c token rule can't back it) | the mutated token is absent from every shown chunk |
  | number drift | verifier | the new number is absent from every shown chunk |
  | entity swap (an entity from another page) | verifier | the swapped entity is absent from the cited chunk |
  | negation flip | verifier | **owner review of every item** (a flip can be true) |
  | fabricated uncited sentence | coverage | has no label and no attribution |
  | citation moved to a wrong shown page | coverage or verifier | the new page lacks the sentence's content words |
  | guard line dropped from code | coverage (inexact or not found) | the line exists in the source block |

- **Size:** about 40 clean items, about 100 planted.
- **CIs:** a **bootstrap clustered by source question**, not iid Wilson, because mutations of one answer are correlated.
- **Owner checks:** every negation item, and a 10% spot-check of the others.

**D4 · Interview set:** about 10 seeds. Each seed freezes the **question text, expected points, chunk keys, kind and seed level**, plus 4 answers written in the owner's terse style:
- gold;
- partial (about half the points);
- wrong (on topic, wrong content);
- misconception = **gold plus one planted false sentence** (a minimal pair).

The misconception sentence is stored separately, so detection can be matched (§3). The owner edits the partial and misconception tiers into their own style.

**D5 · Human labels:**
- **Format:** an append-only JSONL, one record per **item** (all its criteria at once), keyed by item digest. Writes are atomic (write to a temp file, fsync, rename) under a file lock, since the CLI and the Eval page may both write.
- **Blind labelling:** the queue shows the question, the shown notes and the answer only, with no judge score, no reason and no server verdict.
- **No held-out rows:** the calibration sample never includes `heldout2` rows.

### 3. Metrics
- **E-R · Retriever** (deterministic; on D0 and D-snap):
  - **Level:** page level, chunks de-duplicated to `document_id`.
  - **Metrics:**
    - `need=any` questions: **hit@k** (k = 5, 10) and **MRR**;
    - `need=all` questions: recall@k;
    - all answerable questions: **nDCG@10** with the graded `relevance`.
  - **Stages:** keyword, vector, fused and reranked, from `stages`. Also **fused recall@20**, the rerank's ceiling.
  - **Unanswerable questions:** the top reranked score against τ (abstain correctness by retrieval).
- **E-G · Generator (Q&A):**
  - **Deterministic (existing):** correct%, abstain correctness, verified%.
  - **Judge, answered rows only** (reported next to the answer rate, so an abstain-heavy arm can't look good):
    - *relevance*: does it answer what was asked?
    - *completeness*: does it cover the D1 facts?
    - *faithfulness*: is every claim supported by the SHOWN chunks? The text is fetched through the eval seam by chunk id.
  - All three use 0/1/2 anchors written in the judge prompt.
  - *Citation accuracy* is **dropped**: it is the deterministic checks plus the verifier, which E-V measures.
  - **Real-distribution false-verified (r2 P1-3):** the judge is a **screen**, not an estimator, because its own error rate exceeds the 2–5% being measured.
    - The judge's faithfulness check runs over the `verified` answers.
    - The **owner adjudicates every answer the judge flags** (a handful).
    - The page reports **confirmed** false-verified as k/n with an exact CI, next to D3's synthetic rate.
- **E-V · Verifier** (D3, through `/v1/eval/verify`, the full pipeline):
  - **Caught is SPAN-based (r2 P0-B):** a planted item counts as caught only when a coverage unit or a located verifier claim **overlaps the mutation's span** (the server's positional overlap rule). A flag elsewhere in the answer is not a catch.
  - **`caught_by`** (computed in the client from the pre-merge evidence): `coverage` when only a coverage unit overlaps; `verifier` when only a located verifier claim does; `both` when both do.
  - **Caught, unlocated (r3 P1-c):** the badge flipped and the verifier ran with an objection, but nothing overlaps the span (ADR-0015 rule 4a: unplaced objections, or `unsupported` with no claims). It is reported apart and **not** counted in span recall.
  - **Planted recall** (span-based), per mutation type and per **catching layer** of the overlapping claim (coverage / verifier / both / missed).
  - **Badge flip:** the share of mutations that turn a clean `verified` into not-`verified`. Computed **only on sources the current build verifies clean in the same run**: each clean source is re-verified alongside its mutations, because the verifier is stochastic.
  - **False-reject** on clean items, **split by layer** (r3 P1-d):
    - *verifier-caused*: the verifier objected and coverage was clean;
    - *coverage-caused*: lead-ins, residue. This is the verified% cost of the coverage guarantee and has no SLO.
  - **False-verified curve:** derived from span-recall and false-reject over base rates of 5–30%.
  - **CIs:** a clustered bootstrap by source question; with about 15 clusters for the clean items the false-reject CI is wide, and it is printed.
  - **SLO per verifier model (r2 P1-4, r3 P1-d):** the lower bound of span-based planted recall ≥ 0.75 AND the upper bound of **verifier-caused** false-reject ≤ 0.20. "Precision" is dropped: it depends on the synthetic 100:40 mix. A model below the SLO is reported as "should be `verifier_trusted=false`". The eval recommends; it never flips config by itself.
- **E-I · Interview** (D4 + generated questions):
  - **Question quality**, on about 20 generated questions per run:
    - *deterministic*:
      - answer leak = the expected points' content lemmas appearing in the question above a threshold;
      - kind = code ⇒ the question references the code;
      - the seed chunk is among `chunk_ids`;
    - *judge, 0/1/2*: grounded in the given notes; on topic; at the requested level. These are **report-only, labelled "uncalibrated"** unless 40 questions are labelled (r2 P1-7: at n = 20 a good judge is trusted only about 39% of the time).
  - **Grading quality, at the seed's own level:**
    - pairwise ranking accuracy over the tier pairs (gold > partial > wrong; gold > misconception), as k/n with an exact (Clopper–Pearson) interval (10 clusters are too few for a bootstrap);
    - **misconception detection**: the planted sentence appears, normalised by containment, in the grade's misconception quotes, **given** (before server filtering) vs **kept** (after the quote rule and the verifier);
    - score bands: gold ≥ 7 and wrong ≤ 3.
  - **Level calibration** (gold and partial tiers only, each (answer, level) graded **twice**):
    - **test-retest SD**;
    - **Page's L test, permutation version**, over cell means of the two replicates (integer scores tie), for a decreasing trend across the 5 ordered levels, with blocks = items;
    - the pooled paired sign test, SDE-1 vs Staff.
    - "Share of monotone items" is dropped (it measured noise).
  - **General mode:** reported apart, never pooled.
- **E-A · Application** (D-snap; the full stack):
  - correct%, verified%, abstain correctness;
  - tokens per question;
  - errors, **classified**: provider-transient (429/5xx, re-run up to 2 times before counting) vs ours.
  - **Latency comes from a dedicated latency pass** (the first 2 cold questions excluded and reported apart): TTFT and total p50/p95.

### 4. LLM-as-judge
- **Port:** `Judge.score(items) -> results`.
  - **Primary:** a **local model through Ollama** (proposed `qwen3:8b`, thinking off). Operating rules (r2 P1-9):
    - it runs **only post-hoc** over recorded outputs, never during E-A or the latency pass;
    - fixed `num_ctx`, a short `keep_alive`, and Ollama `format` = the JSON schema.
    - **Acceptance on D0:** ≤ 60 s per item, swap growth ≤ 2 GB, and valid JSON on ≥ 99% of calls. A candidate that fails moves on to the next one, then to the fallback.
    - **If no judge passes calibration,** the page shows "not trusted" and only the deterministic checks plus the owner's labels count. That is an acceptable, written-down outcome.
  - **Fallback:** Gemini Flash.
  - A different provider (a paid key) needs explicit config and an opt-in, because note text would go to a new third party.
  - The judge runs **in the client** with its own key file (hidden-input rule, never printed). These calls skip the server's rate limits and usage logs, so the harness logs them itself in the same structured shape.
- **Prompting:**
  - pointwise, with a JSON schema reply, a short reason then the score;
  - anchors written per criterion, including "length is not quality";
  - blind: no arm, model or server verdict shown;
  - every item fenced with a **per-item random nonce delimiter**, and the reply must echo the nonce (stops cross-item injection).
- **Batching** (r2 P2: **deferred**; built only if the local judge fails and the capped Gemini fallback is needed):
  - members sorted by item digest, at most 5;
  - the cache key covers the whole batch and its order;
  - a one-off **singly-vs-batched agreement** on 20 items is reported. Batching is used only if agreement is at least the calibration agreement.
- **Cache and calibration keys:** (judge provider, **the model version the provider returns**, decoding config, judge prompt@version, item digest, and the batch key when batched). A moved provider alias invalidates them.
- **Calibration:**
  - **50 owner-labelled E-G items, each labelled on all 3 criteria**, so n = 50 per criterion. The items are stratified by answer tier and verdict, and exclude held-out rows.
  - E-I question quality reuses about 20 labelled questions, all criteria each.
  - **Reported per criterion:** linear-weighted kappa with a bootstrap 95% CI, raw agreement, the confusion matrix, and the **leniency bias** (a paired sign test of judge − owner).
  - **Trusted = the point estimate ≥ 0.6 AND the CI lower bound ≥ 0.4 AND no significant leniency bias.**
  - Calibration records are versioned and recomputed only on a new label batch or a new judge key (no flapping).
  - **A new judge version re-scores the same labelled items**, about 50 local calls, so the owner never labels again.

### 5. Regression gate: `make eval-gate`
- **Acceptance criterion:** an **A/A run must pass.** Running the same build twice on the same snapshot gives a gate PASS, and its discordance rate sets the **MDE the gate prints**. If retrieval is nondeterministic (host Ollama near-ties), ranks are tie-broken by `chunk_id`, and the A/A check verifies identical retrieval.
- **Baselines:** one per surface (`chat`, `verifier`, `interview`), keyed by the snapshot digest + the per-surface prompt set + the model. A baseline moves only through `ragforge-eval baseline set <run> --note "..."`.
- **Tests:**
  - **Retrieval:** one row per question, so binary hit@5 uses the **exact McNemar** and MRR a **paired sign-flip** permutation test. Block if p < 0.025 (Holm across the two) and the drop is > 2 points, or a metric is below floor − 3.
  - **Chat:** exact McNemar (a sign-flip on binary pairs) on correct% and verified%, Holm-adjusted at α = 0.05. Hard floors:
    - no errors of our own;
    - false-verified (E-V, re-run on any verifier or chat change) passes **an exact McNemar on the planted items against the baseline AND the absolute SLO**.
  - **Interview (r2 P1-8): warn-only.** With 10 seeds the MDEs are large. The level trend warns when it was significant on the baseline AND the candidate's L is significantly lower; "not significant" alone is never a block.
  - **Latency:** warn-only, from the latency pass.
- **Output:** each verdict is printed with its n, MDE and method. The p-values on the Compare page are labelled *exploratory*.
- **Override:** `--override "<reason>"`, stored in the run and shown on the page.

### 6. Eval page (Streamlit)
- **Data access:** it reads run files and label files **locally through `ragforge_client.evals`**, which the UI may import (ADR-0005; `make ui` runs on the host). No new server API is needed beyond §1. Writes go only to D5 labels, atomic and locked.
- **Views:**
  - **Runs:** a header strip per surface (snapshot digest, models, prompt sets), the headline metrics with CIs, the gate state and any override.
  - **Compare:** two runs with the same snapshot digest, paired tests (exploratory), and per-question flips. Held-out rows stay hidden until revealed.
  - **Drill-down:** the answer, the shown notes, the verdict and layer, the judge scores and reasons, and the owner's label.
  - **Label queue:** blind, keyboard-driven, never held out, plus the calibration panel (kappa with CI, confusion matrix, leniency, trusted state per criterion).
  - **Verifier:** recall by mutation and layer, false-verified (synthetic vs real), and the SLO verdict.
  - **Interview:** ranking, detection, the level trend and test-retest SD.

### 7. Run file v2 and package layout
- **Header:**
  - schema version, eval kind, surface, arm, the dataset digests and the **snapshot digest**;
  - the retrieval config (embed model, chunker version, τ, rerank model, `rerank_top`);
  - the chat model and the verifier model;
  - the per-surface prompt sets;
  - the judge key (§4);
  - started and finished times, and the provider-error policy.
- **Rows:** item id, input digest, the `shown` chunk ids + digests, outputs, deterministic grades, judge result refs, timings, tokens, error class.
- **Resume:** it refuses if the header config differs. v1 files load read-only.
- **Privacy:** run files are private (git-ignored, 0600 like today) and may hold answer text. They never hold note text, only chunk ids and digests; text is fetched on display.
- **Package:** `evalrun.py` is split into `ragforge_client.evals/`:
  - `runfile`, `datasets`, `metrics` (pure), `stats` (pure), `judge`, `gate`, `runners`, `cli`.
  - The CLI is unchanged for existing commands.

### 8. Deliberately NOT built
- **The LangSmith / Langfuse exporter:**
  - it duplicates the structured logs and run files;
  - LangSmith would send note text to a third party;
  - self-hosted Langfuse is too heavy for 16 GB.
  - An OpenTelemetry GenAI adapter over the JSONL is a documented later step.
- **The recorded-Notion "Acme" corpus and a CI gate on it:** deferred (D0 covers builders, CI and cloners).
- Online or production sampling (single user).

### 9. Build plan (after this ADR is approved)
- **The integrator locks first, in this order (r2):**
  1. stable keys (principle 6), **the eval tenant (r3 P0-C)**, the snapshot spec and **the content/index digest split (r3 P1-b)**;
  2. the span-based E-V definition, with **caught-unlocated and the pre-merge evidence (r3 P1-c)** and **the layered false-reject SLO (r3 P1-d)**;
  3. the §1 contracts: `chunks`/`resolve`, the `eval` scope and key rules, `RAGFORGE_EVAL_API`, **the eval-scope tokenless frozen grade (r3 P1-e)** and the `prompt_versions` field. B1 then only implements them;
  4. ~~the move-only split of `evalrun.py`~~ **Done differently (lock step, 2026-10-21):** a new package `ragforge_client.evals/` whose modules are each owned by ONE builder (`schema` locked; `metrics`, `stats` and `gate` B2; `judge`, `datasets`, `runners` and `cli` B3; the UI page B4). `evalrun.py` (v1) stays untouched and read-only for every builder. That removes the collision risk the split was for, without a 1,680-line move under test;
  5. the run v2, dataset, judge-result, cache-key and summary JSON Schemas;
  6. D0: the pages and questions plus the golden offline fixtures.
- **B1 · Server seams:** §1 (verify endpoint with `caught_by`, `stages`, `shown`, interview `chunk_ids` and the level override, per-surface prompt versions, snapshot tooling, the `eval` scope).
- **B2 · Metrics and stats** (pure, property-tested on D0):
  - E-R metrics; clustered bootstrap, sign-flip, exact sign, McNemar, Holm, Page's L, weighted kappa with CI;
  - the false-verified curve, the MDE from A/A, and the gate logic.
- **B3 · Datasets, judge and runners:** the D3 mutation generator with the validity checks, the D1/D4 tooling (overlap flag, sealing), the judge port (Ollama + Gemini, nonce fences, batch keys, caching), and the E-G/E-V/E-I/E-A runners. Uses B1's fakes until B1 lands.
- **B4 · Eval page:** the views in §6, built against the D0 fixture runs.
- **Integrator:**
  - `make eval-demo` end to end;
  - the snapshot;
  - the **A/A run**;
  - the first baselines;
  - the owner's sessions (`heldout2` rewording, 50 labels, the D3 negation review).

## Options considered
- **RAGAS / DeepEval as the framework:** rejected as the core. They hide rubrics and statistics, can't batch or cache under the free tier, and don't know the verification contract. Their metric definitions informed §3.
- **The LLM verifier alone for E-V:** rejected. It misstates the shipped guarantee, since coverage catches 3 of the 7 mutation types.
- **The live notes for baselines:** rejected. They compare corpora, not code.
- **A Gemini judge as primary:** rejected. It shares the family with the answer model, which is a self-preference risk, and it is capped at about 20 calls/day. It stays as the fallback.

## Consequences
- **Owner time:** about 2 h, for rewording `heldout2`, 50 blind labels, the D3 negation review and a spot-check.
- **Runs:** resumable. Judge results are cached. Local-judge runs aren't capped, though slower (measured on D0). Gemini runs are free-tier-throttled.
- **What the gate can detect at this n:** only large changes, and it prints its MDE every time.
- **Docs to sync:**
  - DESIGN §2: the hosted p95 and the retrieval SLO move to D0, not "the public corpus";
  - DESIGN §12: the p-thresholds;
  - ADR-0012's corpus section is superseded by D0/D-snap.

## Review r1 → revision 2
| Finding | Where it's addressed |
|---|---|
| P0-1 seams missing | §1 (eval scope: verify with `caught_by`, `stages`, `shown`, interview `chunk_ids` and level override) |
| P0-2 D3 bias / context / clustering | §2 D3 (clean items by owner label regardless of verdict; v2 `shown`; clustered bootstrap; validity checks; negation fully reviewed) |
| P0-3 live corpus | §2 D-snap, §5 (digest refusal, A/A) |
| P0-4 calibration n | §4 (label items on all criteria, n = 50 per criterion; CI lower bound; leniency; versioned) |
| P1-1 blind queue, no held-out | §2 D5, §6 |
| P1-2 level monotonicity | §3 E-I (test-retest, Page's L, sign test, gold and partial only) |
| P1-3 D4 frozen inputs, minimal pair, token, match rule | §2 D4, §1, §3 E-I (given vs kept) |
| P1-4 interview cost | §2 D4 (10 seeds), §3 (sweep only gold and partial) |
| P1-5 batch reproducibility, nonce, model version | §4 |
| P1-6 judge choice | §4 (local primary, owner decision) |
| P1-7 E-G scope | §3 E-G (citation accuracy dropped; faithfulness on verified as a real-distribution estimate; answered rows only) |
| P1-8 D2 relevance | §2 D1 (`relevance` by document_id, `need`), §3 E-R (hit@k, fused recall@20) |
| P1-9 held-out drafting | §2 D1 (owner rewording, overlap flag, near-misses, burned split → dev) |
| P1-10 gate false blocks | §5 (error classes and re-run, latency warn-only, A/A acceptance, exact sign fallback, sign-flip, Holm, exploratory label) |
| P1-11 gate gaps | §5 (interview gate metrics; E-V tied to the SLO; exact McNemar on planted items) |
| P1-12 demo fixture | §2 D0 (`make eval-demo`) |
| P1-13 run header | §7 |
| P2s | DESIGN sync (Consequences); deterministic leak/kind checks (§3); owner-style answers (D4); no flapping (§4); builders never read `heldout2` (D1); package split + atomic labels (§7, D5); judge keys and logging (§4); exporter (§8) |

## Review r2 → revision 3
| Finding | Where it's addressed |
|---|---|
| P0-A identity across collections, tenants and installs | principle 6 (`page_key`, `chunk_key`, resolve at run time, stale marking); §1 `resolve`; D0/D1/D3/D4 keyed by keys; every private run on the snapshot |
| P0-B planted items caught for free | §2 D3 mutation spans; §3 E-V span-based catch, badge flip on clean-verified sources only, `caught_by` per overlapping claim and `both` only when the verifier ran |
| P1-1 chunk text seam | §1 `GET /v1/eval/chunks` + digest check |
| P1-2 token and scope security | §1 HKDF purpose key, `eval` scope also required at grade, key scopes, `RAGFORGE_EVAL_API`, rate limit, chunk_ids within collections, integrator owns Scope |
| P1-3 real-distribution false-verified | §3 E-G: judge as a screen, owner adjudicates, confirmed k/n |
| P1-4 precision SLO | §3 E-V: span-recall lower bound and false-reject upper bound |
| P1-5 snapshot cascade / re-embed / digest | §1 snapshot row |
| P1-6 offline fixtures | §2 D0 golden fixtures; eval-demo is a smoke test |
| P1-7 20 labels for question quality | §3 E-I: report-only "uncalibrated" unless 40 are labelled |
| P1-8 interview gate | §5: warn-only; Page's L permutation over replicate means |
| P1-9 local judge on 16 GB | §4 operating rules, acceptance on D0, the no-trusted-judge outcome |
| P1-10 build order | §9 lock order, move-only split first, integrator owns frozen contracts |
| P2s | §5 exact McNemar / sign-flip stated plainly; ranking k/n exact CI; verify returns the actual model and prompt; batching deferred; owner edits the D4 tiers |

## Review r3 → revision 4
| Finding | Where it's addressed |
|---|---|
| P0-C snapshot pollutes daily use | a dedicated eval tenant (§1 scope and snapshot rows); a test proves the owner's keys never see it |
| P1-a D0 page_key | principle 6: `(file, metadata.filename)`; the snapshot writes `notion` + the original external_id |
| P1-b digest / not evaluable | §1 snapshot row: content vs index digest, `reembed`, chunker/normaliser changes flagged not evaluable, raw-source replay moved to Phase 4 |
| P1-c unlocated verdicts | §1 verify returns pre-merge evidence; §3 E-V caught-unlocated category, caught_by computed in the client |
| P1-d false-reject SLO | §3 E-V false-reject split by layer; the SLO uses the verifier-caused part only |
| P1-e token vs frozen D4 | §1 eval-scope tokenless notes-mode grade, level free; real keys keep v3 |
| P2s | chunk_key occurrence; resolve candidates; clean re-verify in the same run; independent golden oracle; synthetic uploads row, no jobs, document_links copied |

## Review r4 (approved) → revision 5
- **P1, cross-tenant copy guard (lock step 1):** the snapshot definer function takes `(source_collection_id, target_tenant_id)`.
  - **Source:** the source tenant is derived from the calling admin session's `app.tenant_id`, never from a parameter.
  - **Target:** it must have `tenants.kind = 'eval'` (set only at bootstrap) and `eval_of_tenant_id` = the source tenant.
  - **Locking down:** a safe `search_path`, and `GRANT EXECUTE` to the admin role only.
  - **Tests:** copying into a non-eval tenant, into another owner's eval tenant, or from a collection the caller can't see is refused.
- **P2, privacy:** exclusions are honoured at copy time. A later exclusion purges the matching snapshot pages and marks their items stale.
- **P2, reembed:** it rebuilds the embed input with the ingest path's own server function (prefix, heading path, text) and checks the old `embed_sha256` before writing new vectors.
- **P2, false-reject layers:** *verifier-caused* = the verifier objected (located or not), whatever coverage did. Only coverage-alone rejects count as coverage-caused.
- **P2, the curve and the SLO** use **badge-flip recall over sources verified clean in the same run**, which is what the user sees. Span recall and `caught_by` stay as the per-layer diagnostic. Caught-unlocated is counted only on those clean-verified sources.
- **P2, eval tenant data class:** it holds a full copy of the notes, so it is classified and backed up like the live notes (ADR-0014), and deliberately included in backups and in `make nuke`.

## Lock step (2026-10-21): what is locked
- **Contracts** (`ragforge_contracts`):
  - `ApiScope.eval` (mirrored in `domain.Scope`);
  - `QueryRequest.stages` → `QueryResponse.stages` (`RetrievalStages` / `StageHit`);
  - `DoneEvent.prompt_versions` and `DoneEvent.shown` (`ShownSource`);
  - `InterviewQuestionRequest.chunk_ids`;
  - `PageKey`, `ChunkKey`;
  - `EvalVerifyRequest/Response` (with `coverage_units`, `verifier_claims`, `verifier_ran`, the verdict and the actual model and prompt);
  - `EvalChunk(s)Response`, `EvalResolveRequest/Response`, `EvalSnapshotRequest`, `EvalSnapshot`, `EvalReembedRequest`.
- **SDK** (new optional request fields are SENT ONLY WHEN SET, because a server that predates them forbids unknown request fields; `make check` caught this): `query(stages=)`, `chat(trace_shown=)`, `eval_verify`, `eval_chunks`, `eval_resolve`, `eval_snapshot`, `eval_snapshot_status`, `eval_reembed`.
- **Schemas** (`ragforge_client/evals/schema.py`): datasets (Question / QuestionSet, VerifierItem / VerifierSet with the `EXPECTED_LAYER` table, InterviewSeed / InterviewSet, Label), the judge (JudgeKey, JudgeResult), run file v2 (RunHeader / RunRow / RunFooter, with one output shape per eval kind), and summaries (MetricValue, RunSummary, Calibration, GateCheck / GateVerdict), plus `digest`.
- **D0:**
  - `evals/demo/pages/` (15 pages), `questions.toml` (20);
  - `build_demo.py`, which runs the server's own parser and chunker, writing `chunks.json` (30 chunks), `verifier.json` (8 clean + 12 planted, all 7 types, with mechanical validity checks) and `interview.json` (4 seeds × 4 tiers).
- **Golden fixtures** (`evals/demo/fixtures/`): v2 runs for retrieval, verifier, interview and generator, plus labels and judge results.
  - `reference_metrics.py` is a stdlib-only oracle and **the executable definition of every metric name**. It writes `expected_summaries.json` (60 values); B2 must match them to 1e-9.
  - `packages/core/tests/test_eval_schema.py` pins all of the above.
- **Not locked here (B1 implements to the §1 spec):**
  - migration 0012: `tenants.kind` gains `eval`, plus `tenants.eval_of_tenant_id`;
  - the snapshot definer function with the r4 guards;
  - the routes.

## Integration (2026-10-21/22): 6 parallel builders
- **Builders and blind review scores:** EV-API 8 → 9; EV-SNAP 8.7; EV-METRICS 7.5 → 7.8 → 8.2; EV-JUDGE 8 → 8.3 → 8; EV-RUN 7.5 → 7.8 → 8.3; EV-UI 7.8 → 8 → 8.5. The workflow is `docs/workflows/phase4a-eval-build.js`.
- **The six P1s left open after the 2-round cap, fixed by the integrator,** each with a regression test and a mutation check:
  1. a mostly-transient candidate is `not_evaluable` (> 20% of a check's baseline items lost);
  2. an error of ours on ANY attempt blocks `errors.ours`;
  3. JSONL is split on "\n" only (U+2028/2029/0085 no longer corrupt labels or run files);
  4. a local judge never uses environment proxies (`trust_env=False`);
  5. a resume with a different judge setting refuses instead of re-answering or crashing;
  6. Compare counts "badge flipped" only on sources verified clean in the same run.
- **Test hygiene:** the snapshot tests delete their priority-0 copy jobs (they starved the integration tests' claims).
- **Handoffs applied:**
  - **Switch and keys:** `RAGFORGE_EVAL_API` on in the dev compose; eval keys refused outside an eval tenant (403, tested).
  - **Code cleanup:** public `HybridRetriever` accessors; `CoverageUnitKind.invalid_citation` and `ResolveStatus.ambiguous`; `Method.bootstrap`; `MIN_CALIBRATION_LABELS = 50` (owner decision).
  - **CLI and make targets:** `eval_tenant_create` (SDK) + `ragforge-eval2 tenant create` (`make eval-tenant`); `seal` and `d3-build`; `make eval`, `eval-demo` and `eval-gate`; the script entry `ragforge-eval2`.
  - **Eval page:** calibration records `calibration.jsonl` (once per label batch); the held-out reveal recorded in `<run>.reveal.json`; the page opens without the daily key, still behind the Host and Origin guards.
  - **Docs:** `.gitignore` for runs; ADR-0013's definer table.
- **Deferred (documented):**
  - a public normalisation API in `citations.py` (`eval_verify` imports private names; a drift check fails loudly);
  - extracting `rag.chunker.embed_input` (`services/eval_snapshot` mirrors it; pinned by a test on all D0 chunks, and the job re-hashes every input);
  - moving `SqlEvalSnapshotStore` to adapters;
  - the CLI commands `judge-accept`, `overlap` and `calib-sample` (the functions exist);
  - binding the topic in the general-mode token.

## Final integrated review (2026-10-22): 7.0, "ship after fixes"; all P0/P1 fixed
- **P0, a private D3 could never be built.** Runner rows recorded shown notes without stable keys, so `build_d3` skipped every row.
  - Fix: `EvalChunk.chunk_key` (occurrence computed server-side by the same rule as resolve); `PageLookup` records it and `answer_question` attaches it to every `ShownRef`.
  - Tested across builders: a real runner run → owner labels → `build_d3` gives a non-empty set (`test_ev_run_d3_seam.py`). Mutation-checked.
- **P1, the demo on a fresh install.**
  - It failed on the scopes and on the eval-tenant check.
  - Fix: `make eval-demo` bootstraps the demo tenant with the operator CLI (the key goes straight to a 0600 file) and follows the owner path: upload → `POST /v1/eval/tenant` → snapshot → evals with the eval key on the snapshot.
  - Found live, also fixed: the eval-tenant and snapshot routes (create, status, re-embed) accept the owner's key-manager key, because owners never hold `admin`.
- **P1, stale gate evidence.**
  - A candidate older than its baseline is refused.
  - The candidates of one surface must be one build (models and prompt sets).
  - The E-V hard-floor run must have run the SAME verifier model and verifier prompt as the chat candidate.
- **P1, CLI keys:** `d3-build` uses the eval key. Snapshot status and re-embed work with the owner's key since the route fix.
- **P2s fixed:** the gate comment; the demo dataset's labels, cache, calibration and target are gitignored.
- **P2s open (recorded):**
  - gate floors (`frozen.json`) are not loaded yet;
  - persisted A/A per surface for the MDE;
  - the MDE at the Holm-adjusted alpha;
  - the CLI label sample drifts (it re-samples excluding labelled items);
  - the CLI label does not check the dataset digest;
  - pooled `label run1 run2` for the 50-label floor;
  - a dataset-level heldout2 reveal plus a reveal run;
  - E-V inconclusive items count toward the transient share;
  - the tenant-kind check on the additive seams (stages, `trace=shown`, tokenless grade);
  - calibration drift within one label batch;
  - completeness is labelled with no facts.
- **Doc drift:** migration 0012's header comment says the snapshot routes need the `admin` scope. The routes use a key-manager key. The comment is left as is because migrations are checksum-guarded; this ADR is the record.

## Why not DeepEval or RAGAS: a principal-engineer note (2026-10-22, owner request)
**Context.** DeepEval and RAGAS are the best-known open-source LLM eval libraries. Neither is an "official" standard. In industry, the three common patterns are:
- open-source libraries (DeepEval, RAGAS, TruLens, promptfoo), typical for startups and prototypes;
- eval/observability platforms (LangSmith, Braintrust, Arize Phoenix, W&B Weave);
- **in-house harnesses** built around a product's own quality bar, common in mature AI teams.

We chose an in-house harness (`ragforge_client.evals`) for four reasons.

1. **The metric that matters is our own guarantee.**
   - RAGForge's claim is "a *verified* badge means every visible unit is backed by the notes" (ADR-0015). Measuring it needs the shipped pipeline (coverage + attribution + LLM ground check), planted errors with known spans, and per-layer attribution (`caught_by`).
   - A library's *faithfulness* metric is a separate LLM judge's opinion of the answer. It does not tell us how often OUR badge is wrong, and it would double-count our own verifier.
2. **A judge must earn trust.**
   - Library metrics report LLM scores against pass thresholds. They don't calibrate the judge against a human.
   - We show a judge criterion only when it agrees with the owner: linear-weighted kappa ≥ 0.6, a CI lower bound ≥ 0.4, no significant leniency, and n ≥ 50 labels. Otherwise the page says "not trusted". An uncalibrated LLM judge is the most common silent failure in LLM evals.
3. **Small-n statistics.**
   - With about 30 questions, per-case thresholds can't separate a regression from noise.
   - The gate needs paired exact tests (McNemar, sign-flip), Holm correction, clustered bootstraps, an A/A acceptance run and a printed MDE. These are not what the libraries provide.
4. **Cost, privacy and determinism on this hardware.**
   - Claim-decomposition metrics make several LLM calls per answer, which the Gemini free tier and a local 8B judge on 16 GB can't sustain.
   - Our judge makes one call per item, cached by (provider-returned model version, decoding, prompt version, item digest), so re-runs are free and reproducible.
   - Hosted dashboards (and the LangSmith/Langfuse exporters, §8) would send note text off the machine.

**What we kept from them.** Their metric *definitions* (faithfulness, relevance, context recall) informed §3, so the vocabulary maps one to one.

**Trade-off accepted.** More code to own: metrics, statistics and the gate, about 4k lines plus tests. This is mitigated by pure functions, property tests, and an independent stdlib oracle that pins all 60 metric values.

**Parked (owner, 2026-10-22).** An optional DeepEval comparison, about half a day:
- run DeepEval's faithfulness and G-Eval as a **second, report-only judge** with the local model;
- report its agreement with the owner's labels next to ours on the Eval page.

That gives a measured answer to "how good is an off-the-shelf judge on my data?"
