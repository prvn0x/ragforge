# W3-RESULTS handoff: generated results tables

Implements docs/PHASE5.md R2-3 and R3-2: the `ragforge-eval2 results` exporter, the manifest,
the demo header sidecars and docs/RESULTS.md.

## What I built

| Path | What |
|---|---|
| `packages/client/src/ragforge_client/evals/export.py` (new) | The exporter. Manifest models, allow-listed summary and header readers, the note template allow-list, the sidecar writer (0644, atomic, sorted keys), the marker-block renderer, `--check` (unified diff on stderr, exit 1). |
| `packages/client/src/ragforge_client/evals/cli.py` (additive) | One docstring line, one import (`export as export_mod`), the `results` subparser (`--check` or `--write`, required and exclusive; `--doc`, `--manifest`, `--root`, repeatable `--table`), a `case "results"` in `dispatch`, and `cmd_results`, which maps `ExportError` to `CliError` (exit 2). Backup before editing: `phase5/cli.py.bak` in this session's scratchpad. |
| `packages/client/tests/test_w3_export.py` (new) | 38 tests plus 1 owner-only test that is skipped for now (see below). |
| `evals/results.toml` (new) | 9 tables: 6 demo, 2 private (run ids given to me, not read), 1 private table of pending rows only. |
| `evals/demo/runs/*.header.json` (6, generated) | Written by `--write`, for the six demo runs. |
| `docs/RESULTS.md` (new) | Datasets, how to read a table, metric meanings, CI methods and why, statistical limits, the generated blocks, how to regenerate. |
| `.gitignore` | See "gitignore" below. |

### Behaviour, as the tests prove it

- **Summary allow-list.** The exporter reads only `name, value, ci_low, ci_high, n, method, trusted`, plus `note`. It validates the file through `RunSummary` first.
- **Notes.** A note is split on `"; "`. It is kept only when every part fully matches one of the 28 templates in `NOTE_TEMPLATES`. Those templates are derived from every `note=` in metrics.py, and their placeholders accept only numbers. Otherwise the whole note is dropped, and stderr names the metric only. `test_every_note_in_the_committed_demo_summaries_is_allowed` fails if metrics.py's notes drift from the templates.
- **Metric names.** A name outside `^[a-z_]+(\.[a-z0-9_@]+)+$` is refused, both in the manifest and in a summary. The error does not echo the name.
- **Header allow-list.** Only line 1 of `<run>.jsonl` is read, and rows are never parsed. It is projected onto `HeaderFields`: `chat_model, verifier_model, dataset.name, dataset.splits, started_at`. The sidecar's schema is `extra="forbid"`, so a sidecar with a `snapshot` field is refused.
- **Model cross-check.** It runs only when the `.jsonl` exists. A header `chat_model` must equal the manifest's `model`. A header `verifier_model` must equal `verifier_model`, or `model` when that is unset. When both header models are null, the caption says "model from the manifest".
- **Sidecar vs run file.** With the `.jsonl` present, `--check` fails when the sidecar differs from the header ("out of date; run --write"). Without the `.jsonl` (a fresh clone), it uses the sidecar alone.
- **Run listing.** Each table lists explicit `run_ids`. A missing summary fails, and so does a run with neither a `.jsonl` nor a sidecar. A summary whose `run_id` differs from its file name fails. A newer run in the directory is ignored (tested). A doc block that the manifest does not list fails. So do a manifest table with no block, a malformed or unclosed marker, a metric that no listed run has, and a metric found in two of a table's runs.
- **Private tables.** A private table with run ids is skipped, with a printed note, when `evals/private/runs` is not a directory. When that directory exists, the table is rendered and checked. A missing file there is an error.
- **Pending rows.** A metric with `trusted == False` renders "pending owner labels" in place of its number. A table of `pending = true` rows needs no runs and always renders.
- **Determinism.** Fixed-precision formatting, manifest order, sorted sidecar keys. A second `--write` is a no-op, and a second tree built from the same inputs gives identical bytes (tested).

### Mutation check (each mutant applied to export.py, tests run, file restored)

| Mutant | Result |
|---|---|
| keep every note | 8 failed |
| keep a note when ANY part matches | 3 failed |
| templates not anchored | 3 failed |
| no metric-name check on summaries | 1 failed |
| no metric-name check in the manifest | 4 failed |
| sidecar also writes `arm` | 7 failed |
| no header-vs-manifest model check | 1 failed |
| private tables always skipped | 1 failed |
| `trusted=False` shows its number | 1 failed |
| `--check` ignores a stale sidecar | 1 failed |

All 10 mutants were killed, and export.py is byte-identical to the original afterwards.

### Checks run

- `uv run pytest packages/client/tests/test_w3_export.py packages/client/tests -q -k "export or cli"`: 38 passed, 1 skipped.
- `uv run pytest packages/core/tests/module_ev2/test_ev_run_cli.py -q`: 50 passed (the existing CLI is unaffected).
- `uv run ruff format --check` and `uv run ruff check` on my three Python files: clean.
- `uv run mypy packages/client/src`: clean. `uv run mypy packages/client/tests/test_w3_export.py`: clean.
- `uv run lint-imports`: 6 kept, 0 broken.
- `uv run pytest --collect-only -q` lists `test_w3_export.py` (R2-2 exit criterion).
- `ragforge-eval2 results --check --table <the 6 demo tables> --table private-labels-pending`: up to date.

### gitignore

I changed `evals/*/runs/` to `evals/*/runs/*` and added the two `!` lines after it. A
negation cannot re-include a file whose parent directory is excluded. With `runs/*`, the
directory itself is not excluded, so the negations work. I proved this in a throwaway git repo
in the scratchpad (not the project) with a copy of the new `.gitignore`. `git add -A` there
staged exactly `evals/demo/runs/a.summary.json` and `evals/demo/runs/a.header.json`. It did
not stage these, which stayed ignored:

- `*.jsonl`, `target.json`, `baseline-*.json` and `gate-*.json` in `evals/demo/runs/`;
- anything in a subdirectory of `runs/`;
- `evals/other/runs/*.summary.json`;
- everything under `evals/private/` (still excluded by the `evals/private/` line).

`git check-ignore --no-index -v` printed the matching rule for each file, including the `!`
rules for the two re-included files. This matches R3-4's gate rule, which allows only
`*.summary.json` and `*.header.json` under `evals/*/runs/`.

## Integration steps (integrator)

1. **Private tables, on the owner's machine.** First confirm the open questions on the private
   tables below. Then run:
   ```bash
   make eval ARGS="results --write --table private-retrieval --table private-generator"
   make eval ARGS="results --check"
   ```
   The first command writes the sidecars next to the private runs, which are git-ignored. After
   that, `test_committed_private_tables_are_up_to_date` stops skipping on this machine. It skips
   while any block still holds the "not generated yet" line.
2. **Type-check the new test directory in `make check`.** `make typecheck` does not cover
   `packages/client/tests/`. Add this line to the `typecheck` target in the Makefile, after the
   `packages/contracts/tests` line:
   ```make
   	@uv run mypy $$(ls packages/client/tests/*.py)
   ```
3. **W1 (README headline table).** Copying numbers into README would bring back hand-typed
   numbers. Options: link to the RESULTS.md tables, or quote only values with an explicit
   "see RESULTS.md". One manifest serves one doc: every manifest table must have a block in the
   doc it renders, so a README block would need its own manifest file
   (`--doc README.md --manifest evals/readme-results.toml`).
4. **W1 consistency test.** RESULTS.md links to `DECISIONS.md` (W4) and to repository files
   under `../evals/` and `../packages/`. Every `make` target it names exists: `up`, `eval-demo`,
   `eval`, `eval-gate`.
5. **After any new demo run:** put the new run ids into `evals/results.toml` by hand, then run
   `make eval ARGS="results --write"`. Commit the new `*.summary.json` and `*.header.json` files.

## Open questions

1. **Provider and model of the private runs.** I could not read them. The manifest assumes the
   demo values: Gemini `gemini-3.5-flash-lite` for `generator-A-20261022T071117Z`, and
   `nomic-embed-text` plus `ms-marco-TinyBERT-L-2-v2` for `retrieval-A-20261021T191959Z`. The
   generator run's header `chat_model` is cross-checked, so a wrong value fails loudly. The
   retrieval header records no model, so nothing can check it. Please confirm.
2. **Rows in the private tables.** I listed only metrics that metrics.py always emits for those
   kinds:
   - retrieval: reranked hit@5, hit@10, MRR, nDCG@10, and fused recall@20;
   - generator: correct, verified, abstain_correct, answer_rate, and the three judge criteria.

   `retrieval.unanswerable.below_tau`, `recall@k` and `generator.false_verified.confirmed` depend
   on the data and the header, so I left them out. Add them if the private summaries have them.
   `generator.correct` appears only when rows carry a boolean `correct` grade. If it is missing,
   the export fails and names it.
3. **The demo retrieval header has `retrieval: null`.** `evals/demo/arm.toml` has a `[retrieval]`
   table, but the recorded header does not carry it. So tau is unknown and below_tau is `n/a`
   on the demo. Is this expected from `ragforge-eval2 demo`, or is it a bug in the runner? I did
   not change it (not my file). RESULTS.md states the effect plainly.
4. **Interview and retrieval headers record no model** (`chat_model: null`), so their captions
   say "model from the manifest". If the runner should record the server's chat model for
   interview runs, that is a runners.py change for its owner.
