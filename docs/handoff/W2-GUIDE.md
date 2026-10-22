# Handoff W2-GUIDE: docs/GUIDE.md

## What I built
`docs/GUIDE.md`, the step-by-step usage guide, in the 12 sections the plan asks for (PHASE5 W2,
R2-5..R2-8, R3-1, R3-6, R3-7):
1. What you need: tested platform, requirements, one-stack-per-machine ports, `make doctor` with a
   table of its real checks (from `scripts/doctor.sh`), Linux differences (from `bootstrap.sh`,
   `compose.yaml` and doctor).
2. Install and start: `make sync`, `make up`, the provider prompt with the privacy notice quoted
   from `setup_provider.py` (the README's provider detail moved here), `make models` (always, with
   the unused-chat-model note), `make setup` to switch later.
3. Tenant and keys: `make tenant`, creating a collection (curl), the curl/key note, `make mcp-key`,
   a table of key files and why the MCP and eval keys have their own folders.
4. Add your notes: Notion integration, sharing, `make connect-notion` (prompts quoted from the
   script), three ways to watch the sync (worker log line, UI Sources page, API routes), and file
   upload via `POST /v1/collections/<COLLECTION_ID>/files` (form field, types, 50 MB, one upload per
   tenant, from `routes/sources.py`), using an Acme demo page as the example.
5. Ask: `make ui`, `make ask` (flags from argparse), `/v1/query` and `/v1/chat` curl examples,
   Claude Code via `make mcp-install`.
6. Reading the answer: badge table quoted from `ui/render.py`, citations, red vs muted flagged
   claims (`draw_flags`), extras (ADR-0016).
7. Interview mode: levels, start phrases, commands, how notes grading works (`grade.py`), the opt-in
   general mode (button text quoted from `interview.py`).
8. Evaluate on your own notes: the full sequence with flags from `ragforge-eval2 <sub> --help`
   (`eval-tenant`, `snapshot create --source-collection`, `snapshot status`, questions TOML,
   `run retrieval --dataset private --arm A`, `baseline set <run> --note`, `eval-gate --surface chat
   --kind retrieval`, generator runs, labelling in the Eval page), with the model-call cost of each
   step where the code shows it.
9. Monitoring: `make obs-up`, where the Grafana password is (not printed), board rows from the
   dashboard JSON, all 12 alert summaries from the rules file, memory caps.
10. Backups: "there is no database backup today", `make backup-evals`, `make down` vs `make nuke`
    (volume list from `compose.yaml`; what nuke leaves behind).
11. Troubleshooting: 25 entries (the upload entry covers 4 errors), each quoting exact text and
    naming the file.
12. Uninstall.

No Python code, so no ruff/mypy targets. No test files of my own (W1 owns the docs test).

## Checks run
- `uv run pytest packages/core/tests/test_docs_consistency.py -q`: 45 passed.
- Mutation check of that test against GUIDE: appending `make no-such-target` and a link to
  `nope.md` made 2 tests fail with both problems named; restoring the file made all 45 pass again.
- A scratch check (not committed) found no UUID, no `/Users/`, no banned words, and every relative
  link and `#anchor` in GUIDE resolves.
- Every `ragforge-eval2` flag was taken from `uv run --no-sync ragforge-eval2 <sub> --help`.
- Nothing was run against the live stack, as the rules require. No curl example was executed.

## Integration steps for the integrator
1. **Collection gap: closed in round 3.** `connect_notion.py` now creates a first collection
   (W0-KEYS item 5); GUIDE section 3's curl step and both "no collections" entries are gone.
2. **`make sync` (Open question 2).** If the fresh-clone run confirms `make ui` fails without it, add
   `make sync` to the README quickstart before `make tenant`. If it is not needed, delete the
   "Install the Python tools (once)" subsection in GUIDE section 2.
3. During the R3-7 fresh-user run, follow GUIDE sections 2 to 5 word for word and also try the
   curl examples (`-H @<(printf ...)`) in zsh, the default macOS shell.
4. After W0 finishes, recheck the quoted output lines of `make tenant`, `make mcp-key` and
   `make connect-notion` in GUIDE sections 3 and 4 against `scripts/create_tenant.py`,
   `scripts/create_mcp_key.py` and `scripts/connect_notion.py`.
5. README should link GUIDE's anchors `#1-what-you-need`, `#2-install-and-start`, and so on if it
   points into sections.

## Open questions
1. **Closed (round 3).** `scripts/connect_notion.py` calls `create_first_collection` when
   `GET /v1/collections` is empty, so the R3-1 quickstart works without a curl step.
2. **Fresh clone and `make ui` / `make mcp`.** The root `pyproject.toml` dev group lists only
   ragforge-contracts, -core and -client. ragforge-ui (Streamlit) and ragforge-mcp are installed by
   `make sync` (`--all-packages`). I believe a plain `uv run streamlit` or `uv run ragforge-mcp` on a
   fresh clone fails, but I could not run `uv sync` to confirm. GUIDE tells the reader to run
   `make sync` once.
3. **`make doctor` says uv is "only needed for tests/dev"**, but `make tenant`,
   `make connect-notion` (now `uv run python ...`), `make ask`, `make ui`, `make mcp` and the evals
   need it. Suggest rewording or making it blocking in `scripts/doctor.sh` (not my path).
4. **Doctor's Linux line names an NVIDIA override that does not exist** in the repository. GUIDE says
   so plainly. Either add the override file or change the message.
5. **Doctor checks only port 8000.** Ports 8501 (UI) and 3000 (Grafana) are not checked. GUIDE
   says so.
6. **Notion UI wording** ("Settings, Connections, Develop or manage integrations"; a page's
   connections menu) comes from `docs/SECURITY.md`, not from a check against today's Notion. I left
   out whether sharing a parent page shares its children, because the code cannot confirm it.
7. **"~30 s" to the first sync** is the text `connect_notion.py` prints; I did not trace it in the
   scheduler code.
8. **No "laptop sleep" entry.** There is no eval-runner lease-lost error. `LeaseLost` ("Lease lost",
   HTTP 409) is the server's job and conversation lease. GUIDE only mentions `--resume` for an
   interrupted eval run.
9. **Snapshot completion.** GUIDE says to repeat `snapshot status` until neither digest is
   `pending`. The output format is from `cmd_snapshot`; that both digests leave `pending` when the
   job ends is not traced.
10. **Gate output and inputs.** `gate chat: <OUTCOME>` and "reads local run files only" come from
    `cmd_gate`. `baseline chat/retrieval: ...` assumes a retrieval run's header surface is `chat`
    (`SURFACE_KINDS`); not checked on a real run file.
11. **After `make nuke`**, GUIDE advises moving `~/.config/ragforge/*.key`, the eval key folder and
    `evals/private/runs/target.json` aside. This follows from the refuse-on-exists rules and the
    deleted tenant, but was not run.
12. **Uninstall step `claude mcp remove --scope user ragforge`** is the Claude Code CLI, not code
    in this repository.

## Round 2 (review fixes)
All nine round-1 findings were applied in docs/GUIDE.md; none was disputed.
- P1: section 5 REST chat now lists `answer.replace` (drop tokens, show its `text`; ADR-0011
  rule 6, `chat_stream.py` TurnReplace), `error` (carries `problem`, no `done` follows), names
  `tool_call` and `interrupt`, says to ignore unknown types (`decode_sse_data` in `sse.py`), and
  recommends `make ask` or the SDK.
- Section 8: keep `name = "private"` (`cli.py` line 792 error quoted); gate needs `--arm` with
  several arms (`newest_finished`, `cli.py` line 825); labels wording narrowed.
- Section 4: failed first sync is invisible (only decrypt failure calls `mark_connector_error`);
  tells the user to press Sync now and read Recent syncs, and quotes the worker log line
  `job %s (%s) failed: %s` (`services/sync/handlers.py` line 570). Upload example now creates an
  `acme-demo` collection first. Read-only Notion claim now cites `notion/connector.py` (GET,
  POST /search, POST /data_sources/<id>/query) instead of REVIEW.md.
- Section 1: both Ollama value pairs quoted inline from scripts/bootstrap.sh.
- Section 5 MCP: approval wording matches server.py INSTRUCTIONS.
- Section 6: "the reason never changes the status" (ADR-0015).
- Re-check: `uv run pytest packages/core/tests/test_docs_consistency.py -q`: 45 passed. No banned
  words, UUIDs or home paths in GUIDE; new links resolve.

Open question added: 13. The guide says the first sync's job id is shown nowhere and there is no
job-listing route; confirmed only by reading `routes/jobs.py` and handlers, not by a run.

## Round 3 (review fixes)
All five round-2 findings were applied in docs/GUIDE.md; none was disputed.
- P1: removed the "Create a collection" subsection and its anchor. Section 4 step 3 now describes
  both paths of `connect_notion.py` (`pick_collection` and `create_first_collection`, quoted
  prompts and output) and the name rule from `CollectionCreate`. The stale troubleshooting entry
  is gone; "this key sees no collections" now points at `make connect-notion`. Section 8 step 2
  points at the connect-notion output or a `GET /v1/collections` curl. The acme-demo curl in
  "Add files" stays.
- P2: the Acme eval example uses `file:kafka-basics.md` (as in `evals/demo/questions.toml`), the
  `notion:<PAGE_ID>` form is shown separately, and relevance must be in the snapshotted collection.
- P2: worker log lines are described as the `"event"` field of JSON lines (`log_json = True` in
  `config.py`), with a `grep 'indexed document'` filter.
- P2: OLLAMA_MAX_LOADED_MODELS fix gives `launchctl setenv ... 2` and a restart command.
- P2: the gate exits 2 when it cannot gate (`CliError`, mapped to `return 2` in `evals/cli.py`).
- Re-check: `uv run pytest packages/core/tests/test_docs_consistency.py -q`: 53 passed.
