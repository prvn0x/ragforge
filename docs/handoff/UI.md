# Module UI handoff: Streamlit UI (thin API client)

Owner paths: `packages/ui/src/ragforge_ui/**`, `packages/core/tests/module_ui/**`, this file.
Imports only `ragforge_client` and `ragforge_contracts` (ADR-0005). `lint-imports` passes.

## What it is
| file | job |
|---|---|
| `app.py` | `streamlit run` entry: builds the `Backend`, sidebar page switch (Chat / Sources) |
| `config.py` | API URL + key file (the `make ask` rules, reused from `ragforge_client.ask`), `display_url` |
| `api.py` | `Backend` protocol (sync) and `SdkBackend` over `RagforgeClient`: one client per call/stream, private event loop per stream, closes the HTTP stream on early stop |
| `state.py` | pure reducer: SSE events → `Turn`/`Conversation`, approval `decide`, `offers_code` |
| `render.py` | pure presentation: Markdown safety for Streamlit, badges, citation chips, extras split |
| `errors.py` | exceptions and stream `error` events → friendly `UiError` (never `str(exc)`) |
| `views.py` | the Streamlit drawing; thin |

Behaviour: streamed answer (plain text until `verification`, then Markdown, per the SSE
contract; a draft that fails or is cut off BEFORE its verdict stays plain text under "Unverified
draft: … not saved", integration fix), section names on their own paragraphs, verdict badge + plain-language meaning,
unsupported claims as "Not verified" warnings, citation chips `[S1] Title › heading` linked to
the Notion URL, inferred ones `[S2*] … · matched by the server`; then, last and in a bordered box,
"Not from your notes — general knowledge (not verified)" extras (code in `st.code` with its
language). "Show the code" after a finished cited answer without code. Approve/Reject for a
pending interrupt (`resume`), a new question expires it. Errors (API down, 401, 403, 409, 429,
503, dropped stream, mid-stream `error`) are friendly and keep the conversation. Sources page:
collections table, connectors with status/last error and a "Sync now" button, and the status of
syncs started in this browser session.

## 1. Requests for the integrator

### 1.1 Makefile target
Add `ui` to `.PHONY` and:
```make
ui: ## Streamlit UI on http://localhost:8501 (key from ~/.config/ragforge/*.key; ARGS="--key-file F")
	@uv run streamlit run packages/ui/src/ragforge_ui/app.py --server.headless true \
		--server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false \
		--server.allowedHosts localhost --server.allowedHosts 127.0.0.1 \
		--client.showErrorDetails type -- $(ARGS)
```
**`--server.address 127.0.0.1` is required.** Streamlit headless binds every interface by
default (verified: it printed a LAN "Network URL" and a public "External URL"). The UI acts with
the owner's key, which has `approvals:write`, so anyone on the same Wi-Fi could read the notes
and approve destructive actions. **Since review round 2 the app enforces this itself:** if
`server.address` is unset or not loopback (`127.x`, `::1`, `localhost`), the page shows an error
and never reads the key file. The explicit opt-in `RAGFORGE_UI_ALLOW_REMOTE=1` exists only for a
setup where something else guards the port (see 1.4). So forgetting the flag fails closed.
`gatherUsageStats false` turns Streamlit's telemetry off. `showErrorDetails type` shows only the
exception type in the browser if anything is uncaught (the app also sets this itself).

### 1.2 Key for the UI
The UI key needs `read`, `chat`, `write` (Sync now) and `approvals:write` (Approve/Reject): the
owner scopes from `ragforge-admin create-tenant`. Put it in `~/.config/ragforge/<name>.key`
(mode 600). If that directory holds more than one key, pass `ARGS="--key-file ~/.config/ragforge/ui.key"`
or set `RAGFORGE_KEY_FILE`. `RAGFORGE_URL` (or `RAGFORGE_API_URL`, or `--url`) overrides
`http://localhost:8000`.

### 1.3 Nothing else is required
- `make typecheck` already covers `packages/ui/src` and loops over `packages/core/tests/module_*`,
  so `module_ui` is included without changes.
- No new dependencies (streamlit 1.66.0 and httpx are already installed; tests use
  `streamlit.testing.v1.AppTest` and `httpx.MockTransport`).

### 1.4 Optional: compose service (not needed for daily use; unverified sketch)
Only if the UI should run in the stack. The image installs packages non-editable, so there is no
`packages/ui/src/...` path inside it. Simplest route: add a console script to
`packages/ui/pyproject.toml` (integrator-owned) that runs the INSTALLED `app.py`:
```python
# packages/ui/src/ragforge_ui/serve.py  (UI module can add this on request)
import sys
from pathlib import Path
from streamlit.web import cli
def run() -> None:
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
```
```toml
[project.scripts]
ragforge-ui = "ragforge_ui.serve:run"
```
```yaml
  ui:
    build: { context: ., dockerfile: docker/Dockerfile, target: runtime, args: { PACKAGE: ragforge-ui } }
    command: ["ragforge-ui", "--server.headless", "true", "--server.address", "0.0.0.0",
              "--server.port", "8501", "--browser.gatherUsageStats", "false",
              "--client.showErrorDetails", "type"]
    environment:
      RAGFORGE_URL: "http://api:8000"
      RAGFORGE_KEY_FILE: /run/secrets/ui_key
      # 0.0.0.0 inside the container is needed for the port mapping; the app refuses it unless
      # this opt-in is set. Set it ONLY together with the 127.0.0.1 host-side port below.
      RAGFORGE_UI_ALLOW_REMOTE: "1"
    secrets: [ui_key]
    ports: ["127.0.0.1:8501:8501"]   # host side localhost only (see 1.1)
    depends_on: { api: { condition: service_healthy } }
```
Not run (no compose in this module). The secret file must be mode 600 inside the container and
owned by the runtime uid (10001), or `read_key` refuses it.

## 2. Gaps found (API, not UI)
1. **`Connector` has no last-sync time.** The task asked for "last sync"; the API exposes only
   `status`, `last_error`, `created_at`. The page shows those, plus the status of syncs started
   in this session (`get_job`). Suggest an additive `last_synced_at: datetime | None` on
   `Connector` (response model, so older clients are unaffected); the UI would then show it with
   a one-line change in `views._draw_connector`.
2. **No "list jobs" endpoint**, so "recent jobs" can only be jobs this browser session started.
3. **No way to read a pending approval (blocks MCP approvals and post-reload recovery).**
   DESIGN section 10 makes the UI the only place a human can approve (MCP has no approve tool,
   MCP keys never get `approvals:write`). But `resume()` needs the `tool_call_id`s, and those
   exist only in the `interrupt` SSE event of the stream that raised it. So (a) a destructive
   call raised through MCP can never be approved anywhere, and (b) a UI-raised interrupt is lost
   when the tab reloads. **Request:** an additive `GET /v1/approvals/{run_id}` (scope
   `approvals:write`, tenant-scoped, 404 for another tenant's run) returning
   `{run_id, conversation_id, status, pending: [ToolCallEvent]}`. With it, the UI change is
   small: in `views._approval_link`, when `foreign_run(...)` is not None, fetch it and draw the
   existing approval card (`_draw_interrupt`) instead of the notice.
   **What the UI does today:** it reads `?run_id=` (MCP's `approval_url` link). A run this browser
   session holds is shown with Approve/Reject as usual, and the UI writes its own open run into
   the URL. A run it does not hold (an MCP link, or a reloaded page) gets an honest warning: the
   page can't load it, nothing happens unless it is approved, and to do it you ask for the action
   again here. A Dismiss button clears the link.
4. *(Resolved at integration: MCP.md and the MCP agent-facing texts no longer claim the approval is listed in the UI.)*

## 3. How to run and test
```bash
make ui                                                    # after 1.1
uv run pytest packages/core/tests/module_ui -q             # 170+ tests, no network, ~7 s
uv run mypy packages/ui/src
MYPYPATH=packages/core/tests/module_ui uv run mypy packages/core/tests/module_ui/test_*.py
uv run lint-imports
```
Tests: `test_ui_render.py` (Markdown safety, badges, chips, extras), `test_ui_state.py` (event
reducer, approvals, code offer), `test_ui_errors_config.py` (friendly errors, key-file rules,
no key in messages), `test_ui_api.py` (`SdkBackend` over `httpx.MockTransport`: bearer header,
pagination bound, 401, stream closed on early stop), `test_ui_app.py` (AppTest: full turn order
answer < badge < claims < citations < extras, follow-ups, Show the code, 401/429/drop/API down,
Approve/Reject/expiry, Sources + Sync now, key never rendered, inputs disabled mid-stream,
cut-off turns, per-call approval, resume retry, duplicate collection names, failed refresh,
unreadable key file / bad flag).

## 4. Review round 1 fixes (in owned paths, no integrator action needed)
- **P1, input while streaming.** A `rf_busy` flag is set by the widget callback that starts a
  turn (chat_input `on_submit`, Show the code, Approve/Reject), so the run that streams already
  draws every input disabled: chat box (also `submit_mode="disable"`, which greys it out in the
  browser on Enter), Show the code, approvals, New conversation, Refresh sources, the collection
  picker and the Chat/Sources switch. It is cleared in a `finally`, also when Streamlit stops the
  script. A turn cut off after `verification` now reads "the answer and its check above are
  complete; general-knowledge extras, if any, were cut off" instead of "Send it again". AppTest
  covers both (a `StopException` injected after verification), and a mutation check confirmed the
  tests fail when the flag is removed.
- **Per-call approval.** With 2+ pending calls: one unchecked "Approve this one" box per call,
  "Approve checked, reject the rest" and "Reject all". One call keeps Approve/Reject.
- **Failed resume stays retryable.** The interrupt is marked decided only once the resume stream
  produced an event or ended cleanly (`state.settle_resume`). If it failed before that, the
  buttons stay up.
- **Emoji/icon shortcodes.** I read the frontend bundled with streamlit 1.66
  (`static/js/StreamlitMarkdown.*.js`). The reviewer was right: `:material/` is rewritten in the
  raw source, and remark-emoji and the icon plugin match text nodes after escapes resolve, so
  `\:smile\:` would still render as an emoji. A shortcode-shaped `:name:` now also gets an
  invisible U+2060 word joiner after its first colon. Known limit: a link destination that holds
  a `:name:` sequence would get the joiner too. Notion URLs never contain one.
- Collections are keyed by ID, and duplicate names get an ID suffix. A failed "Refresh sources"
  keeps the last good list and the conversation. Section labels split only on `**X:**` /
  `**X**:`. `app.py` catches `OSError`/`ValueError` (type name only, because a decode error
  quotes key bytes) and `SystemExit` from argparse, and escapes the key-file message with
  `md_literal`.
- Known gap (contract, not UI): the conversation id only arrives on `done`, so a turn stopped
  by a tab reload before `done` cannot be followed up in the same conversation. With inputs
  disabled, only a reload or a closed tab can cause that now.

## 5. Review round 2 fixes (in owned paths)
- **P1, LAN exposure: now fails closed in code.** `config.exposure_refused` and
  `app._backend()` read `st.get_option("server.address")` BEFORE anything else (also before an
  injected test backend). If it is unset or not loopback and `RAGFORGE_UI_ALLOW_REMOTE` is not
  exactly `1`, the page shows how to fix it and never reads the key file. AppTests prove the key
  file is never read (with `read_key` patched to record calls) for unset, `0.0.0.0` and a LAN
  address, and that the opt-in works. `tests/module_ui/conftest.py` gives every other UI test
  the loopback address `make ui` uses. I did not ship `packages/ui/.streamlit/config.toml`:
  Streamlit reads it only from the working directory, so it would not apply to `make ui` from
  the repo root, and the in-code check covers every case anyway.
- **P1, pending approvals unrecoverable:** see gap 2.3 for the honest notice, the URL
  round-trip, the AppTests and the API request. Flagged the false MCP handoff claim (gap 2.4).
- **P2, failed question no longer expires an approval.** `start_turn` no longer expires
  anything. `state.settle_question` expires open approvals only if the question reached the
  server, using `state.reached_server` (any event, or a clean end). `settle_resume` uses the
  same rule, and `open_interrupt` skips any turn that never reached the server. AppTest:
  interrupt → question fails with ConnectError → Approve is still offered and works.
- **P2, the approval card names the page.** `render.call_summary` shows
  `Remove “Kafka basics” from the index (document 1234abcd…)` above the raw JSON (still shown).
  Titles come from the conversation's citations for free. Only if an id is still unnamed does
  the UI do one bounded `documents()` walk (new `Backend.documents()` over the SDK's
  `list_documents`, at most 1,000 documents), cached on the interrupt. An id it can't name is
  labelled "unknown document … Check the id", which is what an injected call would look like.
  A failed lookup falls back to the id and retries on the next rerun. `forget_fact` ids have no
  read endpoint, so they stay raw.
- **P2, Sync now.** It is now a button callback, so the click is handled before the rerun. That
  rerun lists the job and draws the button as a disabled "Syncing…" while the connector has a
  queued or running job. A callback guard (`rf_syncing`) drops the second click of a double
  click. Unknown job statuses count as finished, so they never lock the button. A job that
  returns 404 is dropped from the session. Other errors are shown once per rerun.
- **P2, early-close test.** `api.drain` consumes a stream and closes it in a `finally`, so a
  Streamlit `StopException` closes the HTTP response at once. Before, closing waited on garbage
  collection, because the traceback keeps the loop's frame alive. The new test raises
  `StopException` from the consumer and asserts that the body closed while the traceback was
  still alive. `SdkBackend` teardown (`_shut_down`) now runs each step on its own, and a test
  proves that a failing `aclose` still closes the client.
- **P2, nits.** Unsupported claims render through `for_streamlit`, not `answer_markdown`, so
  they get no section split. `client.showErrorDetails` is set to `type` in code and in the
  suggested commands.
- Mutation check: I removed, one at a time, the drain close, the per-step suppress, the
  double-click guard, the reached-server rule, the exposure check and the 404 drop. Each
  removal made its test fail.

## 6. Verified
- 158 UI tests pass; ruff check/format clean on owned paths; mypy --strict clean (src and tests);
  `lint-imports` 6 kept; `test_client_ask.py` still passes.
- `streamlit run … --server.headless true` started and `/_stcore/health` returned `ok`.
- **Not verified:** a real chat against the live stack in a browser (the stack is shared and a
  chat calls the LLM). First manual check after `make ui`: ask one question, check the badge,
  a citation link opening the Notion paragraph, and the extras box at the bottom.


## 7. Integration (2026-10-18)
- **Integrated-review P1 fixed:** a pre-verdict draft is never rendered as Markdown (`views._draw_answer`).
- **Integrated-review P2s fixed:**
  - Host check (`config.host_refused`) plus an Origin check (`config.origin_refused`): another local web page can't drive the UI;
  - `make ui` sets `--server.allowedHosts`;
  - fences close on a longer closer indented 0–3 spaces;
  - `max_chars=8000` on the chat input;
  - a 180 s whole-turn deadline (`api.TURN_DEADLINE_S`, `errors.TurnTimeout`);
  - pyproject declares `ragforge-contracts`, `httpx`, `pydantic` and `streamlit>=1.66`.
- **Still open:**
  - `RAGFORGE_UI_ALLOW_REMOTE=1` turns off the bind, Host and Origin checks together. A localhost port mapping only needs the bind check off, so split the opt-in if the UI ever runs in compose;
  - the three fence parsers (UI, MCP, client) should become one shared `split_fences` in `ragforge_client`.
- After the re-review (8.3, no P0/P1):
  - a failure AFTER the verdict says the answer is saved ("No need to send it again");
  - AppTests prove `_backend` calls the Host and Origin checks.
- Threat note: on loopback the page has no login. Another OS user on the same machine could use it, and a non-browser client sends no Origin. That is acceptable for a single-user Mac; add a token if the machine is shared.
