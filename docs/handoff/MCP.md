# MCP: RAGForge MCP server handoff (Phase 3)

Owned and created: `packages/mcp/src/ragforge_mcp/**` (`server.py`, `service.py`, `turn.py`, `models.py`, `errors.py`, `config.py`, `http_auth.py`) and `packages/core/tests/module_mcp/**` (100 tests, `mcp_support.py` + `test_mcp_ask.py`, `test_mcp_tools.py`, `test_mcp_server.py`, `test_mcp_http.py`).
Nothing outside these paths was edited. `ragforge_mcp` imports only `ragforge_client`, `ragforge_contracts`, `mcp`, `httpx`, `pydantic`, `anyio` and `uvicorn` (import-linter: 6 contracts kept).

## What it is
- `ragforge-mcp` (console script already registered → `ragforge_mcp.server:main`), built on the mcp 2.3 `MCPServer`. It is a thin client of the public API (ADR-0005): every call goes through `RagforgeClient`, and the key is sent only as the bearer token.
- **Transports:**
  - stdio (default);
  - `--http --port N [--host 127.0.0.1]` serves streamable HTTP at `http://127.0.0.1:N/mcp`. **Every request needs `Authorization: Bearer <token>`** (review r2, `http_auth.BearerAuth`, pure ASGI, constant-time compare, lifespan passes through). The token comes from `--http-token-file` / `RAGFORGE_MCP_TOKEN_FILE` (same mode-600 rules as the key, at least 20 chars), or is generated per start (256 bits) and printed once on stderr. It is NOT the API key. A non-loopback `--host` is still refused (exit 2) unless `--allow-remote`, because the transport is plain HTTP and the token would cross the network unencrypted. DNS-rebinding protection stays on, with explicit `TransportSecuritySettings` (loopback hosts, plus the bound host only with `--allow-remote`). Real run checked: no/wrong token 401, right token 200 `initialize`, foreign `Host` 421, key absent from the log.
- **Config:**
  - `--url` / `RAGFORGE_URL` (falls back to `RAGFORGE_API_URL`; default `http://localhost:8000`);
  - `--ui-url` / `RAGFORGE_UI_URL` (default `http://localhost:8501`);
  - `--collection NAME_OR_ID`, repeatable, at most 20 (default: every collection the key sees, first 20).
- **Key lookup** (first match wins): `--key-file` → `RAGFORGE_API_KEY` → `RAGFORGE_KEY_FILE` → the single `~/.config/ragforge/*.key`. The file rules are `ragforge_client.ask.resolve_key_file`/`read_key`, so a group- or other-readable file is refused. The key is held as a `SecretStr`. A key under 20 characters is refused at startup (review r2: a placeholder such as `dev` would make redaction rewrite ordinary words; real keys are ~50 chars).
  - When the key came from the directory fallback (usually the OWNER's key: approvals:write, key manager), `ragforge-mcp` prints a warning on stderr at startup recommending an MCP key + `--key-file`. It is not refused, since the spec keeps that fallback; the "no approve tool" design keeps the approval invariant either way.
  - **Redaction:** every tool output is redacted as a whole (`errors.redact_model`: every string VALUE of the dumped model is redacted, then re-validated, so field names and JSON syntax can never be hit, review r2), every error message passes through the `Redactor`, and a post-verdict SSE error is redacted where it is stored. Tested with a hostile API that echoes the key into errors before AND after the verdict, and into the answer, a citation title and an extra.
- **Tools** (typed pydantic outputs → `outputSchema` + `structuredContent`):
  - `ask(question, conversation_id?)` → `status` (answered | needs_approval), `answer`, `citations` [{label "S<n>", title, heading_path, url, inferred, document_uri}], `verification`, `unsupported_claims`, `extras` [{not_from_notes: true, label "Not from your notes: …", kind, text}], `conversation_id`, `needs_approval`, `warning`.
    - Field order puts extras AFTER the citations and the verdict.
    - **Answer text stays inert** (review r1 P1). `turn.readable` removes only cosmetic escapes (`max\_poll` → `max_poll`), never `\ [ ] < > & \``, so the sanitizer's ADR-0011 invariant 1 still holds: no link, image, raw HTML or entity can come back to life. `\[S1\]` becomes `[S1]` unless the next character would make it link syntax (`(`, `[`, `:`). Fenced code and code spans stay byte for byte. Fences follow CommonMark (review r2 P1, `turn.split_fences`): backtick or tilde openers of 3+, a closer of the same char at least as long, and an unclosed fence runs to the end of the text. Indentation is accepted at any depth, on purpose: reading prose as code only keeps a few cosmetic backslashes, reading code as prose corrupts it. Tested on the real sanitizer output for `~~~`, a longer closer, a short inner fence, a backtick fence inside a tilde fence and an unclosed fence. Nothing is added (an unbalanced `**` stays as is). Tested on the REAL `MarkdownStreamSanitizer` output for `<img onerror>`, `<a href=javascript:>`, a pre-escaped link, `&lt;`, `[S1](…)`, `[S1]:` and a backslash-then-bracket. The field description and the server instructions tell the agent the remaining escapes are deliberate.
    - **Extras are passed on as the server sent them** (literal CommonMark, ADR-0016), so `**In simple terms:**`, headings and links in an extra can never render as the answer's own markup.
    - A whole turn has a deadline (`ASK_TIMEOUT_S` = 180 s, `anyio.fail_after`) with a clear "took longer than 180 s" tool error, and the chat event generator is closed deterministically (shielded `aclose`) on every exit path.
    - **Failure rule:** an SSE `error` before `verification` means nothing was committed, so the call fails, the draft is dropped and the error carries the conversation_id. An `error` after the verdict (the extras step) returns the committed answer with a `warning`. A stream that ends with neither `done` nor a verdict is an error.
    - **Interrupt:** `needs_approval {run_id, ui_url, pending_tools}`. The unverified draft is dropped, and nothing is ever sent to `/v1/approvals` (tested). `conversation_id` falls back to the caller's for a follow-up (the server sends it only for a new conversation). The warning, the tool description and the server instructions tell the agent not to ask again in that conversation until the user has decided, because a new question cancels the pending action (DESIGN section 8).
  - `search(query, k=8, 1..20)` → ranked hits (title, heading path, url, snippet, rerank score, document_uri). No LLM.
  - `list_sources()` → collections (with `in_scope`) and connectors (status, last_error, collection name). All pages are walked. It never fails on a broken scope (review r1 P1): an unknown `--collection` or an empty tenant is reported in `config_warning`, and everything visible is still listed. ask/search still fail with the same message.
  - `sync_now(connector_id?)` → `started` [{connector_id, display_name, job_id}] and `skipped` [{…, reason}]. The default is every `kind == notion` connector of the key. Each connector is tried on its own (review r2): a 409 (already syncing), 403 or any API error lands in `skipped` with the agent-facing reason, and the jobs that did start are returned. The call fails only when nothing started and every failure has the same message (one clear error instead of N identical skips); a `deleting` connector is skipped without a call. Annotated `idempotent_hint=False` (each call enqueues a job or gets a 409).
  - **There is no approve tool.** The test asserts the exact tool set `{ask, search, list_sources, sync_now}`.
- **Resource** `ragforge://documents/{id}` (application/json): document metadata, with `"chunks": null` and a note (see API gap 2). It honours the same `--collection` scope as ask/search (review r2): it lists documents per in-scope collection (`collection_id` filter, so out-of-scope collections are never even listed), a document outside the scope gets the same "not visible" answer as a missing one (no existence oracle), and a broken scope fails with the same message as ask/search. Cost: about one call per collection plus one per 200 documents, until gap 2 is closed.
- **Errors:** messages are written for the agent:
  - 401: bad or revoked key, fix it and restart;
  - 403: scope (MCP keys never hold approvals:write; sync_now needs `write`);
  - 404, 409, 422;
  - 429/503: retry hint;
  - other 5xx: with the request id;
  - connect error: "unreachable at <url>, is `make up` running";
  - timeout.
  
  An unexpected exception (a bug) is NOT converted. The SDK then shows only `Error executing tool <name>` and logs the traceback.
- **Collection scope** is resolved on first use, not at startup, so the server starts and lists its tools while the API is down. It is then cached for the process lifetime: a new collection needs a restart, the same as `make ask`.

## Integration steps (for the integrator)

### 1. Makefile (add `mcp mcp-install` to `.PHONY`)
```make
mcp: ## MCP server over stdio for Claude Code / Cursor (ARGS="--key-file ~/.config/ragforge/mcp.key")
	@uv run ragforge-mcp $(ARGS)

mcp-install: ## Print (does not run) the `claude mcp add` command for this checkout
	@echo "1) Create an MCP key (no approvals:write) if you have none: see docs/handoff/MCP.md, step 2"
	@echo "2) Then run:"
	@echo "claude mcp add --scope user ragforge -- uv --directory $(CURDIR) run ragforge-mcp --key-file $(HOME)/.config/ragforge/mcp.key"
```
- `--key-file` is explicit because `~/.config/ragforge/` may already hold the owner's key. With two `*.key` files, the "single key" rule refuses to guess.
- For HTTP instead: make a stable token file once (`umask 077; python -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.config/ragforge/mcp-http.token`; give it a non-`.key` suffix so the single-key lookup ignores it), run `uv run ragforge-mcp --http --port 8765 --key-file ~/.config/ragforge/mcp.key --http-token-file ~/.config/ragforge/mcp-http.token`, then `claude mcp add --transport http ragforge http://127.0.0.1:8765/mcp --header "Authorization: Bearer $(cat ~/.config/ragforge/mcp-http.token)"`. (That stores the endpoint token, not the API key, in Claude Code's config.) Without `--http-token-file` the token changes on every start.
- Avoid `-e RAGFORGE_API_KEY=…` in `claude mcp add`: Claude Code would store the key in its own config file.

### 2. Creating an MCP-scoped key (owner runs it; the key is never shown)
The key name must contain `mcp`. The server then marks it `client_kind=mcp`, and it can never hold `approvals:write` or `admin` (`services/auth/scopes.py`, also enforced in the DB). Scopes `read, chat` are enough for ask/search/list_sources/resource. `write` is needed only for `sync_now` (drop it for a read-only agent; sync_now then returns a clear 403 message).
```bash
uv run python - <<'EOF'
import asyncio, getpass, os, pathlib
from ragforge_client.client import RagforgeClient
from ragforge_contracts.api import ApiScope

async def main() -> None:
    owner = getpass.getpass("Owner API key (hidden; it must hold approvals:write to mint keys): ").strip()
    url = os.environ.get("RAGFORGE_URL", "http://localhost:8000")
    async with RagforgeClient(url, owner) as rf:
        created = await rf.create_api_key("mcp:claude-code", [ApiScope.read, ApiScope.chat, ApiScope.write])
    path = pathlib.Path.home() / ".config/ragforge/mcp.key"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(created.secret + "\n")
    print(f"wrote {path} (prefix {created.prefix}, scopes {', '.join(created.scopes)})")

asyncio.run(main())
EOF
```
The snippet parses, and its `ApiKeyCreate` validates. It was NOT run against the live stack (shared, in use). Optionally wrap it as `scripts/create_mcp_key.py` + `make mcp-key`.

### 3. UI coordination (approval deep link)
`needs_approval.ui_url` is `{RAGFORGE_UI_URL}/?run_id=<uuid>`. Today the UI CANNOT load such a run (no `GET /v1/approvals/{run_id}`, UI gap 3): the agent-facing text (INSTRUCTIONS, `NEEDS_APPROVAL_WARNING`, `ui_url` description) tells the user to ask for the action in the UI and approve it there (integrated-review P1). If the UI picks another shape, change only `approval_url()` in `ragforge_mcp/turn.py` (and its test).

### 4. Docs
In DESIGN section 10, add: the resource returns metadata only (no chunk text, API gap 2), plus the deep-link format, the `--allow-remote` rule and the "answer keeps `\ [ ] < > & \`` escaped" rule.

### 5. `packages/mcp/pyproject.toml` (review r1 + r2 P2): declare the direct imports
The code imports `httpx`, `pydantic`, `anyio`, `uvicorn` (new in r2, HTTP serving with the auth gate) and `ragforge-contracts` directly (today they arrive transitively through `ragforge-client` and `mcp`). Add them to `[project].dependencies` (`ragforge-contracts` as a workspace source, as `ragforge-client` is), then `uv lock`. I am not allowed to edit any pyproject.toml or run `uv lock`, so this stays with the integrator even though it is this module's manifest.

### 6. Track in `docs/NEXT.md`
- Makefile `mcp` / `mcp-install` targets (step 1).
- UI support for the `?run_id=` deep link (step 3). Until a `GET /v1/approvals/{run_id}` API exists (UI handoff gap 3), the UI cannot load an approval raised through MCP: `ui_url` opens the UI with an honest notice, and the user asks for the action again in the UI.
- SDK gaps 1 and 4 below, API gaps 2 and 3.

## SDK / API gaps found (outside my paths; not worked around beyond what is stated)
1. **`ApiError` drops `Retry-After`.** `RagforgeClient._error` keeps only the Problem body. Proposal: `ApiError.retry_after: float | None`, parsed from the response header (delta-seconds). `errors._retry_hint` already reads `getattr(exc, "retry_after", None)` and prints "retry after N s" once it exists (tested). Until then it prints "retry shortly". (The API sends Retry-After on 503 ServiceBusy/UpstreamRateLimited. I found no 429 in the API today.)
2. **No `GET /v1/documents/{id}` and no chunk text endpoint.** The resource walks `GET /v1/documents?collection_id=…` for each in-scope collection (200 per page, at most 50 pages per collection) and returns metadata only, with `chunks: null` and an explicit note. Proposal: `GET /v1/documents/{id}` and `GET /v1/documents/{id}/chunks` (read scope, RLS), plus SDK methods `get_document` / `list_chunks`.
3. **`Connector` has no last-sync time or last job.** `list_sources` shows `status` + `last_error` only. Proposal: additive `last_synced_at` / `last_job_id` on the `Connector` response.
4. **SDK generators do not close their inner generators.** `RagforgeClient.chat` iterates `_chat_stream`, which iterates `_stream`, with plain `async for`. `aclose()` on the outer generator does not close the inner ones, so the HTTP response is released by the asyncio GC finalizer, not at once. Proposal: wrap each inner iteration in `contextlib.aclosing(...)`. The MCP side already closes the outer generator (tested).

## Verification (run in this session, after review round 2)
- `uv run pytest packages/core/tests/module_mcp -q` → **93 passed**.
- `uv run ruff check` and `ruff format --check` on `packages/mcp/src/ragforge_mcp packages/core/tests/module_mcp` → clean.
- `uv run mypy packages/mcp/src/ragforge_mcp` → clean (strict). `MYPYPATH=packages/core/tests/module_mcp uv run mypy packages/core/tests/module_mcp/test_*.py` → clean.
- `uv run lint-imports` → 6 kept, 0 broken.
- `uv run pytest packages/core/tests/test_client_ask.py -q` → passes (116 together with the module tests).
- **Mutation check (r2 P1):** with the old fence regex patched back in, 4 of the 6 new fence cases fail (`~~~`, longer closer, tilde-around-backtick, unclosed); the 2 plain-backtick baselines pass on both.
- **Real process runs:** the stdio subprocess test still passes; `ragforge-mcp --http --port 8799` gave 401 without or with a wrong token, 200 on `initialize` with the printed token, 421 for a foreign `Host`, and the key was absent from its log.
- **Not run:** against the live stack (`make up` is shared and in use) and from Claude Code itself. First thing for the integrator: `make mcp-install`, add it, then ask one question and check that extras are shown apart.

## Review round 1: response
All P0/P1 and the P2s inside my paths are fixed (see above). Outside my paths, so handed off: pyproject deps (step 5), Makefile targets and NEXT.md tracking (steps 1 and 6), the UI deep link (step 3). For P2 "human key", the reviewer's softest option was taken (a stderr warning), not a refusal, because the task spec requires the `~/.config/ragforge/*.key` fallback.

## Review round 2: response
- **P1 fences:** fixed (CommonMark fence rules, see "Answer text stays inert"). markdown-it-py was not used: it is not a declared dependency of this package, and a line scanner of ~25 lines keeps the byte-for-byte guarantee easy to prove (the output is the input with only prose segments touched).
- **P2 sync_now:** fixed (per-connector errors, started jobs kept).
- **P2 resource scope:** fixed (filtered by the configured collections, server-side `collection_id` filter); gap 2 still open.
- **P2 redaction:** both fixes: values-only redaction and a 20-char key minimum.
- **P2 pyproject deps:** cannot be done inside my paths (hard rule: no pyproject.toml, no `uv lock`); step 5, now including `uvicorn`.
- **P2 HTTP auth:** fixed (bearer token, file or generated).
- **P2 Makefile targets:** integrator, step 1 (unchanged; HTTP variant updated for the token).

## Known limits
- `ask` waits for the whole turn (5–60 s, cut off at 180 s) and returns once; it doesn't stream progress notifications.
- The answer keeps a few backslash escapes (`\[`, `\<`, `\&`, …) on purpose: readability traded for staying inert.
- Citation/title text from notes is cleaned of control and bidi characters but otherwise passed as data. The server `instructions` tell the agent that note text is data, not instructions. That is a prompt rule, not a guarantee.
- Blind review rounds 1 and 2 are addressed; round 3 has not run yet.
- The HTTP token is shared by every client of one server process; there is no per-client identity or revocation short of changing the token file and restarting.


## Integration (2026-10-18)
- Integrated-review P1 fixed: approval texts no longer promise an approval the UI cannot load.
- P2s fixed:
  - a connection drop AFTER the verdict returns the saved answer with a warning (`collect_turn`);
  - warnings are combined (a failed later step no longer hides a partial verdict);
  - `display_url` drops query and fragment;
  - a closing fence is indented 0–3 spaces only.
- `--allow-remote --host 0.0.0.0` is NOT usable for LAN clients: the allowed hosts become `0.0.0.0:<port>`, so a LAN client's real `Host` gets 421. Use a concrete IP with `--allow-remote`.
- After the re-review (8.3, no P0/P1):
  - the 180 s limit runs inside `collect_turn`: a timeout or a malformed frame AFTER the verdict returns the saved answer with a warning;
  - an unreadable key or token file is a clean `ConfigError` (exit 2, type only).
