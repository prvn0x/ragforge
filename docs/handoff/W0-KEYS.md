# W0-KEYS handoff: key commands a new user can run

Scope: docs/PHASE5.md R2-1, as amended by R3-3 and R3-5. No live-stack run, no model call, no git.

## What I built

| Path | What |
|---|---|
| `scripts/create_tenant.py` (new) | `make tenant`. Refuses if any `*.key` is directly in `~/.config/ragforge/`. Then it reserves `owner.key` (O_EXCL, 0600, folder 0700, `umask 077`) **before** running `docker compose exec -T api ragforge-admin create-tenant --name N --owner O`. Its stdout is captured and parsed. Only the `api_key=` value is written to the file. It prints `tenant_id=...` and the file path. Compose and admin stderr is shown, minus the `Store this key now` line (and minus any line holding the key, as a guard). On any failure (non-zero exit, no `api_key` line, compose missing) the reserved file is removed and the message says `is the stack up? make up` and `no key file was written`. Ctrl-C also removes the file, prints `interrupted; no key file was written` and exits 130. If the tenant is created but the key write fails, the message gives the tenant id and says to start clean on a dev stack (`make nuke && make up && make tenant`). It no longer suggests `ragforge-admin create-key`. That command prints the key on the terminal, and its key cannot mint the MCP key. |
| `scripts/create_mcp_key.py` (new) | `make mcp-key`. Finds the owner key with `ragforge_client.ask.resolve_key_file` / `read_key` (`--owner-key-file` overrides). Reserves `~/.config/ragforge/mcp/mcp.key` (O_EXCL, 0600, folder 0700) before minting, so an existing file means nothing is minted on the server. Mints `mcp:claude-code` with `read, chat, write` through `RagforgeClient.create_api_key`. Prints path, prefix and scopes, never the secret. `RAGFORGE_URL`, else `RAGFORGE_API_URL`, overrides `http://localhost:8000` (`api_url`). On an API or connection error the reserved file is removed. If the key is minted but the write fails, the message names the key prefix and `DELETE /v1/api-keys/<id>` to revoke it (`save`). A reply the SDK cannot parse gives a plain message, never pydantic's error text, which would quote the secret. Ctrl-C prints `interrupted; no key file was written` and exits 130. Extra flag: `--out` (default above). It refuses a path directly in `~/.config/ragforge/`, since a second key there breaks `make ask`/`ui` (`check_out_path`). |
| `scripts/connect_notion.py` (edited) | Reads the RAGForge key via `resolve_key_file` / `read_key`: `--key-file`, else `$RAGFORGE_KEY_FILE`, else the single `~/.config/ragforge/*.key`. The hidden getpass prompt for the RAGForge key is used only when none of the three points at anything. A named file that is missing, several keys, or a group/other-readable file is an error, not a prompt (`find_key_file`, `load_key`). The Notion token is still read with getpass. New: when `GET /v1/collections` is empty (a fresh `make tenant` creates no collection), it asks `Collection name [notes]`, creates it with `POST /v1/collections` using the key it already loaded, and connects Notion to it (`create_first_collection`). Before this, the quickstart stopped here. The rest is unchanged (collection pick, display name, `POST /v1/connectors`, `RAGFORGE_API_URL`). One small addition: an empty typed key exits with `empty key`. |
| `Makefile` | New `tenant` (`NAME ?= personal`, `OWNER ?= $(USER)`, passes `--compose "$(COMPOSE)"`), new `mcp-key`, `MCP_KEY := $(HOME)/.config/ragforge/mcp/mcp.key`. `mcp` runs `--key-file "$(MCP_KEY)"` when `ARGS` is empty, otherwise `ARGS` alone. `mcp-install` prints `make mcp-key (see docs/GUIDE.md)` and the `claude mcp add ... --directory '<checkout>' ... --key-file '<MCP_KEY>'` line, with both paths single-quoted so a path with a space still pastes as one argument. `connect-notion` is now `uv run python scripts/connect_notion.py $(ARGS)`. I also added `tenant` and `mcp-key` to the `.PHONY` line (the only edit outside those target blocks). |
| `packages/core/tests/test_w0_keys_tenant.py` | Subprocess runs with a temp HOME and a fake compose program. |
| `packages/core/tests/test_w0_keys_mcp.py` | In-process, using the real SDK with an `httpx.MockTransport`. It also checks the `make -n` output for `mcp`, `mcp-install`, `mcp-key`, `connect-notion` and `tenant`. |
| `packages/core/tests/test_w0_keys_connect_notion.py` | In-process, with fakes for `call`, `input` and `getpass`. |

## Checks run

- `uv run pytest packages/core/tests/test_w0_keys_*.py -q`: 48 passed (round 2; 35 in round 1).
- `uv run ruff format` / `ruff check` on the 6 Python files: clean.
- `uv run mypy --strict` on the 3 scripts, and `uv run mypy` (project config) on the 3 test files: clean.
- `make help` lists `tenant`, `mcp-key`, `mcp`, `mcp-install` and `connect-notion` with the new text.
- Mutation check: 19 mutants, each applied alone to a script or the Makefile and then reverted. All 19 were killed. The mutants: tenant file mode 0644; tenant folder mode 0755; no Store-notice filter; no secret-line filter; no unlink on failure; no refusal on another `*.key`; tenant O_EXCL→O_TRUNC; ignore the admin exit code; mcp O_EXCL→O_TRUNC; mcp no unlink; mcp prints the secret; mcp ignores `--owner-key-file`; mcp ignores `RAGFORGE_URL`; mcp drops `write`; connect never reads the key file; connect prompts on a bad or ambiguous key; connect ignores `--key-file`; `make mcp` without the default key; `mcp-install` pointing at the old doc. The first run had 3 survivors (file mode hidden by the umask, O_EXCL hidden by the glob refusal, exit code hidden by the missing key). I added tests that kill them (`test_reserve_*`, the `exit-after-key` mode).
- Mutation check, round 2: 16 more mutants, each applied alone and then reverted. All 16 were killed. They cover the 5 survivors the review found: mcp O_EXCL mode 0644; mcp umask removed; tenant umask removed; mcp folder chmod removed; tenant folder chmod removed. They also cover the new code: connect exits on an empty collection list; connect default collection name changed; `--out` check removed; `ValidationError` not caught; minted-but-not-saved not caught; mcp Ctrl-C handler removed; tenant Ctrl-C handler removed; `mcp-install` without quotes around `$(CURDIR)`; the same for `$(MCP_KEY)`; `RAGFORGE_API_URL` not read; URL variable order swapped. The umask mutants are killed by recording the umask in force during the run (the fake API in-process, the fake compose as a child process). The `$(CURDIR)` mutant first survived because the checkout path has no space. The test now runs `make mcp-install` with `CURDIR` and `HOME` overridden to paths that hold a space, then splits the printed line with `shlex`.

## Integration steps (integrator)

1. Run the checks above. They need only the workspace venv. No Docker, no stack.
2. Optional, so `make check` also lints and type-checks the new scripts. These edits are outside my paths:
   - `lint`: `uv run ruff check packages scripts/security scripts/create_tenant.py scripts/create_mcp_key.py scripts/connect_notion.py && uv run ruff format --check packages scripts/security scripts/create_tenant.py scripts/create_mcp_key.py scripts/connect_notion.py`
   - `typecheck`, first line: append `scripts/create_tenant.py scripts/create_mcp_key.py scripts/connect_notion.py`.
3. Live run (owner, when the stack can be touched): `make tenant`, `make mcp-key`, `make connect-notion`, `make mcp-install`. Each one should print no secret. `ls -l ~/.config/ragforge ~/.config/ragforge/mcp` should show `-rw-------` files.
4. `docs/handoff/MCP.md` step 2 still shows the old snippet and the old path `~/.config/ragforge/mcp.key`. That path is directly in the discovery folder, so it would break `make ask` and `make ui`. Mark it superseded, or point it at `make mcp-key`.
5. W2 (GUIDE): the quickstart `make tenant` then `make connect-notion` now works on a fresh tenant: connect-notion offers to create the first collection (default name `notes`). Before round 2 it stopped with "No collections yet". The commands behave exactly as described in the table above. One thing the GUIDE should state: `make connect-notion` now runs through `uv run`, so it needs the workspace installed (`make sync`), the same as `make ask`.

6. Optional, outside my paths (`packages/client/src/ragforge_client/ask.py`): give the "nothing points at a key" case its own exception, so `connect_notion.py` can drop the guard that mirrors `resolve_key_file`'s three sources. Snippet:

   ```python
   class NoKeyFound(KeyFileError):
       """No flag, no $RAGFORGE_KEY_FILE and no *.key in the folder: callers may prompt."""

   # in resolve_key_file, the empty-candidates branch:
       raise NoKeyFound(
           f"no API key found: put one in {directory}/<name>.key (mode 600) or pass --key-file"
       )
   ```

   Then in `scripts/connect_notion.py`, `find_key_file` becomes `try: return resolve_key_file(explicit, dict(env), key_dir)` / `except NoKeyFound: return None`. Until then a comment there ties the guard to `resolve_key_file`.

## Open questions

1. **URL variable names.** Round 2: `create_mcp_key.py` now reads `RAGFORGE_URL` first, then `RAGFORGE_API_URL`, so one exported `RAGFORGE_API_URL` works for `ragforge-ask`, `connect_notion.py` and `make mcp-key`. `connect_notion.py` still reads only `RAGFORGE_API_URL` (unchanged behaviour). Should the project settle on one name?
2. **Key printed, then non-zero exit.** If `ragforge-admin` prints an `api_key=` line but exits non-zero, `make tenant` treats the run as failed and keeps no key. A tenant may then exist without a stored key. The way out on a dev stack is `make nuke && make up && make tenant`, which deletes all data. On a stack with data to keep, a key that is never shown can be made with `(umask 077; docker compose exec -T api ragforge-admin create-key --tenant <id> --name owner --scopes read,write,chat | sed -n 's/^api_key=//p' > ~/.config/ragforge/owner.key)`. That key works for `make ask`/`ui`/`connect-notion`, but it cannot mint the MCP key: `create-key` cannot grant `approvals:write`, so `make mcp-key` gets a 403. I did not put this pipe in the script's message because I have not run it. I chose this because admin.py prints the key only on success, so this case means something unexpected happened.
3. **Tenant created, key write fails.** Same as 2: if the tenant is created but the key write fails (for example, the disk is full), the message names the tenant id and suggests `make nuke && make up && make tenant` on a dev stack. Not tested live.
7. **Ctrl-C during `make mcp-key`.** If Ctrl-C lands while the request is in flight, the server may already have minted the key. The message says `interrupted; no key file was written`, which is true, but it cannot name a key it never received. `GET /v1/api-keys` would show it. There is no `make` target to list or revoke keys; should there be one?
8. **First collection name.** `connect-notion` suggests `notes`. The API accepts only lowercase letters, digits, `-` and `_` (`CollectionCreate.name`); a bad name ends with the API's 422 message, and the user runs the command again. Should the GUIDE name the rule?
4. **`NAME` and `OWNER` from the environment.** These are `?=` target-specific defaults, so a shell that exports `NAME` would override `personal`. Rename them to `TENANT_NAME` / `TENANT_OWNER`? The locked interface says `NAME=` / `OWNER=`, so I kept them.
5. **Folder modes on existing folders.** Both scripts `chmod 0700` the key folder even when it already exists (`~/.config/ragforge`, `~/.config/ragforge/mcp`). I believe this is right for a folder that holds tenant keys. Please confirm.
6. **No live checks yet.** I did not check the live `docker compose exec -T` path, the real `POST /v1/api-keys` with an owner key, or the `claude mcp add` line. The integrator runs them (step 3).
