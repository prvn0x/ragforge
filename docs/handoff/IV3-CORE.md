# Handoff IV3-CORE: interview Phase 3c, server side

Status: built and tested against the locked contracts and the DOT seam (`coverage.dotted_names`, `names=` on `segment` / `attribute` / `settle`, already present on disk). I edited nothing outside my paths. `bootstrap.py` needed no change.

Checks run (2026-10-19, after the round-1 review fixes, see section 7):
- `uv run pytest packages/core/tests/module_iv packages/core/tests/module_e2 -q`: **313 passed** (module_iv: 166).
- Neighbouring suites `module_d1`, `module_ag`, `test_client_interview.py`, `test_foundation_contracts.py` and `test_api_health.py`: 1945 passed.
- ruff check and ruff format --check are clean on my paths. `uv run mypy packages/core/src` is clean. `MYPYPATH=packages/core/tests/module_iv uv run mypy packages/core/tests/module_iv/test_*.py` is clean. `uv run lint-imports`: 6 kept.
- Mutation checks:
  - dropping `names` in `verify._backed` fails 5 tests;
  - each half of the NO_NOTES existence logic (topic and no topic) fails a test when removed.
- **Not run:** the live stack (see section 5).

## 1. What was built

### Levels
- `model.Level` / `model.Source` mirror `InterviewLevel` / `InterviewSource` (a test pins both).
- `InterviewSettings.target_level` is now a `Level`.
- `config.interview_target_level` is now `Literal[sde_1..staff]`, default `senior_sde_1`. A before-validator maps the Phase 3b spellings: "Senior SDE-1" becomes `senior_sde_1`, "SDE 2" becomes `sde_2`, "senior" becomes `senior_sde_1`. Any other value fails at startup ("Principal", "L7" and "" are tested). `deps.get_interview_settings` wraps the value in `Level(...)`.
- `prompts.LEVELS` holds, per level, its cumulative focus, its DELTA over the level below (`adds`, disjoint from the lower levels' foci, with its key terms in `aspects`) and its 10/8/6/3 anchors. The rubric is generic and names no company. Difficulty is one step within the level (`difficulty_rule`), bounded by deltas, never by a whole focus:
  - "harder" must not ask for what the next level ADDS;
  - "easier" must still ask for what this level ADDS over the one below.

  So harder-L forbids exactly what easier-(L+1) requires, and no rule forbids part of its own level's focus (a test checks every level's forbidden aspects against its own focus).
- In notes mode, the grade prompt judges only aspects the notes support: a Staff rubric does not penalise notes that lack cost or cross-team material.
- Templates: `interview_question@2` and `interview_grade@2` (bumped), plus the new `interview_general_question@1` and `interview_general_grade@1`.
- The `/question` route resolves `level=None` to the configured level once and always sets `level` and `source` on the response. `/grade` uses the request's echoed level (None means the default).

### Token v3
- `QuestionBinding` gains `level` (the resolved one) and `source`. `v3.<issued_at>.<mac>`.
- v1 and v2 tokens are rejected as malformed (400).
- With a valid token, swapping the level, swapping notes for general (or back), changing the key or changing the question returns 403 (API tests).

### General-knowledge fallback (`source=general`)
- **Question:** `service._general_question` runs only the collection access check (`require_collections`), then `question.write_general_question`. That is one model call plus at most one repair: no retrieval, no chunk and no verifier.
  - Kind: the client's first requested kind, with general mode's own `GENERAL_KIND_RULES`.
  - `exclude_questions`: only the last 15, each cut to 300 characters, fenced as data; `excluded_given` and `excluded_used` are logged.
  - The prompt forbids company, incident, date and person names.
- **Grade:** `grade.grade_general` is one judge call. The client-echoed question and key are fenced data, and the prompt treats the key as a HINT the model may override, saying so in the summary.
  - `verification` is always "unverified", with `unsupported_claims=[]` and `citations=[]`.
  - `verify` is never called.
  - A misconception renders as `"<said> (general knowledge: <correction>)"`. `said` must still quote the user (`quoted`).
- `strip_labels` runs on every general text: question, topic, key, model answer and each item. The sanitizer gets an empty link allow-list.
- Tests prove:
  - no note read (a SQL spy sees only `require_collections`; the retriever, embedder and reranker are untouched);
  - no `[S/T<n>]` label and no Markdown link in any general output, with the REAL `MarkdownStreamSanitizer`;
  - a notes request that finds nothing returns a 409 and never switches to general.
- Topic required: the contract validator rejects a blank topic, and a general grade without a token, with the API's uniform **422** validation Problem (the spec said "400"; the shared handler answers 422 for every body validation failure). The domain adds its own 400 `TopicRequired` guard for non-HTTP callers.

### Acronym topics (`agent/interview/acronyms.py`, pure)
- When the topic's direct title/heading FTS matches nothing before exclusion, and the topic has an acronym-shaped word, ONE extra read runs: `SqlInterviewNotes.topic_phrases`.
  - It returns the distinct titles plus `unnest(heading_path)` (UNION) under RLS and the same visibility filter, `ORDER BY phrase` (deterministic truncation), `LIMIT phrase_pool + 1` (5000).
  - Truncation is logged as `interview_topic_phrases_truncated` (no text).
  - The result is cached per (tenant, collection set) for 60 s in a process-wide, bounded `agent/interview/phrases.PhraseCache` (256 entries), so a session repeating its topic scans once. Staleness is harmless: an expansion only feeds the normal reads, which re-apply RLS and every filter.
- Matching runs in Python, so no user text ever reaches SQL. The rules:
  - acronym-shaped = 3 to 6 ASCII letters, or 2 typed in capitals; never a stop word;
  - initials are tried both with and without stop words, and a run never starts or ends on one;
  - hyphens split words; for 3+ letter acronyms ", " and " & " also join a spelled-out list ("Atomicity, Consistency, Isolation & Durability" spells ACID); any other punctuation splits phrases;
  - a non-ASCII word breaks a run;
  - a 2-letter acronym must cover a whole phrase, which is why "HL" does not match "High Level Design" but does match "High Level";
  - a word that already has a hit of its own is never expanded.
- Expansions (at most `max_expansions`, 3) go through the existing `topic_chunks` FTS and `_search`. Their hits rank BELOW direct hits: `pick_seed(secondary=...)`, tier 1b.
- Tested on the DB:
  - "HLD" finds the "High Level Design" page, and "lld" finds a "Low-Level Design" heading;
  - "HLDX" and "hl" find nothing (NO_NOTES);
  - another tenant's or an unrequested collection's "High Level Design" never matches;
  - a direct hit ("Kafka") runs no extra read.

### NO_NOTES vs EXHAUSTED
- `InterviewNoNotes(InterviewExhausted)`. The route maps it to `INTERVIEW_NO_NOTES`, and every other exhaustion to `INTERVIEW_EXHAUSTED`. The details are plain text ("No notes on X" / "No more unasked notes on X").
- The decision is made BEFORE exclusion:
  - with a topic, `_candidates` already knows from EVIDENCE: direct FTS (or one `LIMIT 1` existence read when exclusions exist), raw search hits, and the same for each expansion. An expansion merely existing proves nothing ("HLD caching" -> "High Level Design caching" may match no note);
  - with no topic, `_any_note` makes one existence read, only on the failure path.
- Tested: a session that used the Kafka notes and asks for Redis (no notes) gets NO_NOTES; Kafka again gets EXHAUSTED.

### Question-reply robustness
- `replies.ask_json` takes a `review` that returns `Reviewed(reply|None, problem, reason, metrics)`. The outcome is `parsed`, `degraded` or `malformed`.
  - After the one repair, the best USABLE reply seen is accepted (`Parsed.degraded`): the one with the most `points_kept`, ties to the later (repaired) one.
  - An unusable reply (schema-invalid or too-short question) is still a 502.
- `QuestionReply.expected_points` accepts 0 to 6 items (the prompt still asks for 3 to 6). `question.review_question` judges the question and the points separately:
  - points-only schema errors are salvaged per point (`points_schema`);
  - unbacked points are dropped (`points_unbacked`, one repair, then accepted with 0 or 1 points);
  - an unlabelled point gets its label back via `verify.recover_label` when EXACTLY one shown note lexically backs it. It still faces the ground check.
- **Root cause of the live Spring Boot 502, confirmed:**
  - `check_citations(strict=True)` cuts a bare `max.in.flight.requests.per.connection` / `spring.datasource.url` at its last dot and calls the second half uncited. This is asserted in a test.
  - `verify._backed` now runs strict mode's `needs_label` rule over `coverage.segment(text, names=dotted_names(shown notes))`, plus non-strict `check_citations` for rule 1, plus `attribute(..., names=)`.
  - `check_answer` passes `names` to `attribute` and `settle`.
- Every `interview_model_call` line now also carries `level`, `kind`, `difficulty` and `source` (`model.CallTags`), the verifier calls included. The question call adds `reason` (the category), `detail` (our validator's message: DECLARED field paths, rules and counts, never values; an extra key's model-written name is rendered `<extra>`), `points_given`, `points_backed` and `points_kept`.

## 2. Integration steps
1. Nothing to apply: the router, deps, config and templates are all in my paths.
2. Run `make check` (and `make typecheck`, which covers module_iv's test files in isolation).
3. Live smoke test as in section 5.

## 3. Needed outside my paths / contract notes
1. **`InterviewGradeRequest` has no `topic` field**, so the design decision "token v3 binds the topic for general" cannot be honoured: the grade cannot re-present the topic for the MAC. The token binds the question text and the key, which pin the content the grade uses (the topic does not reach the grade prompt). If topic binding matters, add `topic: str | None` to `InterviewGradeRequest` (a contract change) and I will add it to `QuestionBinding` (v4).
2. **`citations.check_citations` has no `names` parameter.** `verify._backed` mirrors strict mode's `needs_label` expression (5 lines, documented) over `segment(names=)` instead. If the integrator adds `names` to `check_citations`, replace `_backed`'s mirror with `check_citations(..., strict=True, names=known.names)`.
3. **`PROMPT_VERSION`** changes again: interview prompts at @2, two new templates. This is the known per-surface-version residual.
4. **The SDK docstring** `interview_grade` ("at the Senior SDE-1 bar") is now stale (client.py is not mine).
5. **Clients:**
   - Branch on `INTERVIEW_NO_NOTES` vs `INTERVIEW_EXHAUSTED`, and offer general mode on either.
   - For general mode, send back `source="general"`, `chunk_ids=[]`, the token, the echoed `level`, and `question` / `expected_points` byte for byte.
   - Label general output "General knowledge, not from your notes".
   - Send `exclude_questions` (the server uses the last 15).

## 4. Eval seams
- **Pure:**
  - `prompts.question_prompt`, `grade_prompt`, `general_question_prompt`, `general_grade_prompt`, `difficulty_rule`, `LEVELS`, `recent_questions`;
  - `question.review_question(notes)(payload)`, which gives reason categories `schema`, `points_schema`, `question_too_short` and `points_unbacked`;
  - `verify.lexically_backed`, `verify.recover_label`;
  - `acronyms.expanded_topics`, `phrase_matches`;
  - `selection.pick_seed(..., secondary=)`.
- **Model:**
  - `question.write_question(..., level=)`, `question.write_general_question(...)`;
  - `grade.grade_answer` (uses `request.level`), `grade.grade_general`.
- **Logs:** `interview_model_call` (with tags, reason, detail and point counts), `interview_question_written` (degraded, points kept), `interview_graded`, `interview_topic_expanded` (counts only).

## 5. What the integrator must test live
1. Spring Boot notes, "interview me on Spring Boot": no 502. In the logs, `points_backed` should now be close to `points_given`.
2. "HLD": questions come from the "High Level Design" page. Check `interview_topic_expanded` with expansions=1, and the latency of the extra phrase read on the real note count.
3. Redis after a Kafka session (no Redis notes): 409 `INTERVIEW_NO_NOTES`. Kafka until done: `INTERVIEW_EXHAUSTED`.
4. `source=general, topic="Redis"`: a sensible question with no company names, and a grade that is "unverified" with no citations. Check that `exclude_questions` avoids repeats over about 5 questions.
5. Levels: the same Kafka note at `sde_1` vs `staff`. Check by eye that the questions differ in depth, and that the Staff grade does not mark the answer down for the cost or cross-team material the notes lack.
6. Gemini repair rate: count `outcome=degraded` and its `reason` over a session.

## 6. Known gaps
- **Acronym phrase read.** It is linear in visible chunks (`UNION` + `unnest`, no index), capped at 5000 and cached 60 s per tenant and collection set. Fine at personal scale; a distinct-phrase table filled at index time would be the next step (worker/migration paths, not mine).
- **Expansion breadth.** Each expansion costs one FTS read plus one embed and rerank; the cap is 3.
- **Acronym rule (deliberate).** A 3-letter acronym may match inside a longer title (e.g. "Highly Lazy Dogs" spells HLD). The rule is generic by design, and such hits rank as expansions, below direct hits.
- **General mode has no semantic de-duplication** beyond the prompt's "do not repeat" over the last 15 questions.

## 7. Round-1 review fixes (2026-10-19)
- P1 difficulty rules: rebuilt on disjoint per-level deltas (`LevelRubric.adds` / `aspects`); generic harder wording is level-neutral ("push on an edge case"); grammar and ".)." fixed. New tests: harder-L's forbidden text equals easier-(L+1)'s required delta; no forbidden aspect appears in the level's own focus or in what the rule asks; every rendered rule reads cleanly. The rendered question system prompts change (template versions unchanged, still @2: unshipped).
- P1 "HLD caching" EXHAUSTED: `matched` now comes from expansion evidence only. API test: fresh session "HLD caching" -> NO_NOTES; "HLD" with every HLD chunk excluded -> EXHAUSTED.
- P2 log leak: `schema_summary` keeps only declared field names and indexes; test with a text-bearing extra key proves the text is absent from logs and the repair turn.
- P2 best fallback: implemented (most kept points, ties to the later); test.
- P2 phrase scan: ORDER BY + `PhraseCache`; unit test for TTL, eviction and tenant/collection keying; API test proves the scan runs once across two questions.
- P2 ACID/CAP: list joins for 3+ letters; positive (ACID, CAP) and negative (colon, comma without space, 2 letters) unit cases.
- P2 topic binding for general: unchanged, needs the contract change in section 3 item 1 (integrator's call). SDK docstring: section 3 item 4.
