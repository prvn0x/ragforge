# W1 README: handoff

Owner paths: `README.md`, `packages/core/tests/test_docs_consistency.py`, this file. Nothing else was edited.

## What I built

### README.md (137 lines, limit 200)
- What it is in two lines. An HTML-comment TODO marks where the owner's optional screenshot goes, so there is no broken image.
- Three guarantees. Each one links the code and names the tests that prove it, and each says what it does not cover:
  1. Verified citations; Notion citations deep-link to the block: `agent/citations.py`, `agent/grounding.py`, `agent/coverage.py` and the Notion `normalize.py`. Tests: `test_review_attacks`, `test_a_negation_the_source_lacks_gets_no_source`, `test_anchors_are_dashless_block_ids_and_deep_links_land_on_the_block` and `test_citation_chip_uses_the_answer_label_and_links_the_paragraph`. "Not covered" says uploaded files are cited by title with no link (`files/parser.py` sets `external_url=None`; `apps/api/chat_stream.py` then emits `url=None`).
  2. RLS: `db/migrations/helpers.py` `tenant_table`. The guarantee is scoped to the app's runtime login roles `rf_api` and `rf_worker` (ADR-0013: never members of `rf_system`). Tests: `test_rls_isolates_tenants`, `test_rls_fails_closed_without_tenant`, `test_tenant_tables_have_rls_enabled_not_forced`, `test_every_endpoint_answers_404_for_another_tenants_ids` and `test_lists_never_show_another_tenants_rows`. These need Docker, and the README says so. "Not covered" names `rf_system` (cannot log in), `rf_migrator` (acts as `rf_system` through `SET ROLE`) and the read-only `rf_backup` (reads every tenant for `pg_dump`), with `test_rf_system_cannot_log_in` and `test_backup_role_reads_all_tenants`.
  3. Local first: `embedder_ollama.py`, `provider.py`, the `config.py` default and the `egress_guard.py` plugin. Tests: the egress inventory, egress guard and no-silent-egress tests. "Not covered" names both egress paths from SECURITY.md §3.5: Google on Gemini, and the MCP client's model provider (B6).
- A mermaid diagram. Its names were checked against `compose.yaml` (api, worker, postgres, prometheus/grafana in the obs profile, the one-shot services), `docker/observability/prometheus.yml` (both api and worker are scraped), `bootstrap.py` (only the worker wires the Notion connector, and both api and worker wire the Ollama embedders) and the import-linter contracts in `pyproject.toml`.
- The quickstart is exactly R3-1: doctor, up, models, tenant, connect-notion, ui (or ask). Each command has a one-line comment. A line under the prerequisites says to run `make sync` once after cloning: the root `pyproject.toml` installs only contracts, core and client, so `make ui` (Streamlit) and `uv run ragforge-mcp` cannot start without it. It links GUIDE section 2, which says the same. Prerequisites: Docker, uv, python3 and openssl (`scripts/doctor.sh` checks both; the Makefile's `setup` calls python3), plus host Ollama on macOS only. Under it: `make models` runs on both paths (on Gemini it also pulls an unused chat model, per `scripts/models.sh`), `make setup` is for switching provider later, and `make mcp-key` and `make mcp-install` are named.
- The headline results come from ONE cited run: `application-demo-20261021T174004Z` (Acme demo, dev split, n = 20, Gemini `gemini-3.5-flash-lite`, 2026-10-21). The table has three rows: `application.correct`, `application.verified` and `application.abstain_correct`. They sit in a `<!-- readme-results run_id=... -->` block, and a test checks them against the summary file, so they are not hand-typed and unchecked. The `verified` row says plainly that a server-written abstain also counts as `verified` (`agent/nodes.py`, `abstain`).
- Three limit bullets. They link to RESULTS, DECISIONS and ADR-0018. The held-out set is cited to DECISIONS (RESULTS does not mention it), and n is "20 or fewer on the demo set (see the n column)".
- A doc map with GUIDE, RESULTS, DECISIONS, DESIGN ("as of Phase 0; later changes are in the ADRs"), SECURITY ("updated through Phase 4"), `docs/adr/`, REVIEW and NEXT.
- The old README's provider and privacy detail is gone from the README. It now lives in the GUIDE (W2), and the README links to it.

### packages/core/tests/test_docs_consistency.py (55 tests, about 0.9 s, no network, no Docker)
Markdown is parsed with markdown-it-py (CommonMark plus tables), which is already a core dependency. So fences, inline code, tables and HTML comments are told apart the way a renderer does it. Each checker is a pure function that returns its violations.

| Check | Real-doc test | Planted test (mutation) |
|---|---|---|
| `make X` in code is a Makefile rule (`^[a-z0-9-]+:`, not `.PHONY`, not `X :=`); flags and `VAR=v` before the target are skipped (`make -s X`, `make X=1 Y`); code includes raw-HTML `<pre>` and `<code>` bodies outside comments | `test_real_make_targets_exist` | `test_planted_makefile_parse_uses_rule_lines_not_phony`, `test_planted_unknown_make_target_fails`, `test_planted_make_flags_and_raw_html_code_are_checked` |
| ADR "not built" targets (`eval-models`, `label-verdicts`, `reindex`) are allowed in DECISIONS.md only, and must still be absent from the Makefile and named in an ADR | `test_real_not_built_targets_are_still_not_built_and_come_from_an_adr` | the same planted make test |
| `make eval ARGS="<sub> [<sub2>] ..."` and `ragforge-eval2 <sub> [<sub2>]` name real (nested) subcommands; leading flags are skipped (`ARGS="--verbose nope"` is read as `nope`) (argparse is asked with `<words> --help`, public API only) | `test_real_eval2_subcommands_exist` | `test_planted_unknown_eval2_subcommand_fails`, `test_planted_real_eval2_parser_is_asked` |
| Relative links (Markdown, and `<a href>`/`<img src>` in raw HTML outside comments) resolve inside the repo; `#anchor`s match GitHub heading slugs (with `-1` de-duplication) or `<a id>` | `test_real_links_and_anchors_resolve` | `test_planted_broken_link_and_anchor_fail`, `test_planted_raw_html_links_are_checked`, `test_planted_github_slugs`, `test_planted_html_anchor_counts` |
| Every `docs/adr/*.md` has a TABLE-ROW link in DECISIONS.md | `test_real_every_adr_has_a_decisions_row` | `test_planted_adr_without_a_decisions_row_fails` |
| Every top-level `docs/*.md` has a TABLE-ROW link in the README, so a link in the guarantees does not stand in for a doc-map row (excluded: `PHASE1.md`, `PHASE5.md`) | `test_real_readme_maps_every_top_level_doc` | `test_planted_doc_missing_from_readme_fails` |
| README is at most 200 lines | `test_real_readme_is_short` | `test_planted_long_readme_fails` |
| Banned words in prose (code identifiers are exempt) | `test_real_prose_avoids_banned_words` | `test_planted_banned_words_fail_in_prose_only` |
| No UUID and no home-directory path anywhere in the four docs (comments and code included) | `test_real_docs_hold_no_uuid_or_home_path` | `test_planted_uuid_and_home_path_fail_anywhere` |
| Every backticked `test_*` name exists as a test in `packages/*/tests`, so a README proof cannot go stale | `test_real_named_tests_exist` | `test_planted_unknown_test_name_fails` |
| A test name that follows a `.py` link in the same paragraph or bullet is defined in THAT file | `test_real_named_tests_are_in_the_file_cited_beside_them` | `test_planted_test_cited_beside_the_wrong_file_fails` |
| README results rows equal the cited summary (value and CI to 2 decimals, n exactly); every data row in the block names a metric; the caption's interval method ("Wilson intervals") equals each row's `method`; the caption's model equals the run header, and every ISO date in the block is the header's date (at least one must appear) (the `.header.json` sidecar, else line 1 of the `.jsonl`) | `test_real_readme_results_match_their_summary` | `test_planted_results_row_differing_from_its_summary_fails`, `..._caption_differing_from_its_header_fails`, `..._block_without_summary_or_rows_fails`, `..._row_without_a_metric_name_fails`, `..._caption_method_differing_from_the_summary_fails`, `..._caption_date_must_be_the_only_date` |

`REQUIRED_DOCS = False` is the single switch. While it is False, a missing one of the four documents is skipped (not xfailed), and links to a missing one are tolerated. All four exist now, and every test passes with no skips.

The test file holds no literal UUID and no literal home path. Both are built from parts, so the file passes the R3-4 pre-publish scan.

### Mutation check (run on scratch copies; the real file was not changed)
I applied 20 mutants, one at a time. Every one was killed by at least one test except the first round's outside-repository mutant, and I then strengthened `test_planted_broken_link_and_anchor_fail` until it was killed too. The mutants:
- rule lines taken in any case;
- `make` also scanned in prose;
- no anchor de-duplication;
- anchors not checked;
- missing files skipped;
- outside-repository links allowed;
- ADR links accepted outside tables;
- banned words scanned in code too;
- no UUID check;
- no home-path check;
- the README length check off by one;
- result values, model, date or empty blocks not compared;
- the eval subcommand check always true;
- test names not checked;
- not-built targets allowed in every doc;
- the doc map not checked.

### Round 2 mutation check (after the blind review)
Run on a scratch copy outside the repo (deleted afterwards). Each of these 11 mutants was killed:
- the `make VAR=v target` form dropped;
- raw-HTML links ignored;
- HTML comments not stripped before reading hrefs;
- only the first eval2 word asked;
- the eval2 regex reading one word;
- the cited `.py` link never recorded;
- the cited-file check skipped;
- the `.py` link carried over to the next paragraph;
- a results row without a metric name skipped;
- the caption method not compared;
- a missing caption method allowed.

### Round 3 mutation check (after blind review round 2)
Run on a scratch copy outside the repo (deleted afterwards), planted tests only. Each of these 8 mutants was killed:
- flags before the make target not skipped;
- raw-HTML block `<pre>`/`<code>` not read as code;
- inline `<code>...</code>` not read as code;
- HTML comments not stripped before reading `<pre>`/`<code>`;
- flags before the eval2 subcommand not skipped;
- the doc map accepting a link anywhere;
- the caption date check back to "the right date appears somewhere";
- a block with no date accepted.

## Round 3 changes (blind review round 2, 8/10)
- P1 quickstart without `make sync`: fixed. The six R3-1 commands are unchanged; one line under the prerequisites says to run `make sync` once after cloning, and why.
- P2 checker gaps: flags before the make target and before the eval2 subcommand are skipped, and raw-HTML `<pre>`/`<code>` (block and inline) is read as code. Plain prose such as "run make ghost" is still not checked, on purpose: see open question 3. A flag that takes a separate value (`make -C dir up`, `--root x run`) makes the checker read the value as the target. That is a false alarm that fails loudly, not a silent pass. No such form is in the four docs.
- P2 doc map: only table-row links count now.
- P2 caption date: every ISO date in the block must equal the run's date, and one must appear. I did not use the reviewer's `, (date)\.` regex: it ties the check to one caption wording, and a set compare catches a wrong date wherever it sits. The run id's compact timestamp is not an ISO date, so it does not match.
- P2 two results mechanisms: not changed, as in round 2. It is integration step 3.

## Round 2 changes (blind review round 1, 8/10)
- P1 deep links: fixed as described under guarantee 1.
- P1 RLS roles: fixed as described under guarantee 2.
- P1 MCP egress: fixed as described under guarantee 3.
- P2 limits and prerequisites: fixed.
- P2 checker gaps: all six closed except flags. `run retrieval --nope` still passes: `--help` exits before argparse reports an unknown flag, and checking flags would need a per-subcommand parse with placeholder values. The `(Gemini)` provider is not compared: the run header has no provider field (only `chat_model`, `dataset`, `started_at`, `verifier_model`), and inferring it from the model name would be a guess.
- P2 two results mechanisms: not changed in my paths. It is an integration choice (step 3 below).

## Checks run
- `uv run pytest packages/core/tests/test_docs_consistency.py -q`: 55 passed.
- `uv run ruff format` and `uv run ruff check --fix` on the test file: clean.
- `uv run mypy packages/core/tests/test_docs_consistency.py`: clean. The `make typecheck` shared pass (`mypy packages/contracts/tests $(ls packages/core/tests/*.py)`): clean, 32 files.
- `uv run pytest --collect-only -q packages/core/tests/test_docs_consistency.py`: 55 collected. It sits in `packages/core/tests/`, so `make test`, `make test-unit` and `make check` run it.

## Integration steps
1. Set `REQUIRED_DOCS = True` in `packages/core/tests/test_docs_consistency.py` (line 52), then run the test.
2. Commit the demo run's `evals/demo/runs/application-demo-20261021T174004Z.summary.json` (and the `.header.json` sidecar from W3's `--write`). `test_real_readme_results_match_their_summary` fails on a fresh clone without the summary. This depends on the R2-3 `.gitignore` change (`!evals/demo/runs/*.summary.json`, and `!evals/demo/runs/*.header.json` from R3-2).
3. Pick ONE results mechanism (review finding: two formats now coexist, and they print the same metric to 2 and 3 decimals). Recommended: W3's renderer (`ragforge-eval2 results --doc README.md --manifest evals/readme-results.toml`, `generated:` markers, `--check`). Then replace the README `readme-results` block with W3's markers and delete `results_mismatches`, `_body_rows`, `_expected_cells`, `_run_header`, `test_real_readme_results_match_their_summary` and the five `test_planted_results_*` tests. If you keep mine instead: after a new demo run, update the run id, the three rows and the caption; the test names any row that differs.
4. If `make reindex`, `make eval-models` or `make label-verdicts` is ever built, the not-built test fails. Then drop the name from `NOT_BUILT_TARGETS` and fix the DECISIONS row.

## Open questions
1. **The headline run is on Gemini, not on the local default.** The only public demo summary is a Gemini Flash-Lite run, and the README says so. Should the integrator add an Ollama demo run, or is one Gemini run the right headline?
2. **`application.verified` counts server-written abstains as `verified`** (`agent/nodes.py`, `abstain`; 7 of the 20 demo questions are unanswerable). The README states this in the row. Should RESULTS (W3) also show a verified rate over answered questions only? That metric does not exist today, so I did not invent it.
3. **`make` mentions in plain prose are not checked.** Only code spans, fences and raw-HTML `<pre>`/`<code>` are checked, because prose like "make sure" would be a false positive. Every command in the four docs is in backticks today.
4. **External links are not fetched** (unit tests stay offline, per the egress guard). The README has one, the uv docs link.
5. **DESIGN.md's banner says "v0.8, last updated 2026-10-16".** I used the phrase "design as of Phase 0" from the brief. Please confirm that phrase matches the owner's view of that date.
6. **`helpers.py` `tenant_table` docstring** says the policy filters every login role, which is false for `rf_backup` and `rf_migrator` (review round 1). I do not own that file. Suggested docstring line: `Enable RLS with the tenant policy. rf_api and rf_worker are always filtered; rf_backup gets a read-all policy for pg_dump (ADR-0013).`
7. **`make doctor` calls uv "only needed for tests/dev"** (review round 2). Yet `make ui` and the MCP server need `make sync` through uv. I do not own `scripts/doctor.sh`. Suggestion for its owner: change the uv line to say uv is needed for `make sync`, `make ui`, `make ask` and the MCP server.
