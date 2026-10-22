# P2B-G2: provider setup UX (make setup, hidden key entry, compose/doctor/docs)

## What G2 built (owned paths only)
- `scripts/setup_provider.py` (stdlib only, runs on system python3 ≥ 3.9). It has three modes:
  - **(default):** the interactive menu, run by `make setup`.
  - **`--if-unset`:** used by `make up`. It asks only when `RAGFORGE_LLM_PROVIDER` is absent from `.env` and from the shell, and stdin is a TTY. Otherwise it stays silent. If that first-run setup can't finish (rejected key, Google unreachable), it prints the reason and exits 0, so `make up` starts on local Ollama. Nothing is written, so the next `make up` asks again. Ctrl-C still exits 130 and stops `make up`.
  - **`--check`:** used by `make doctor`. Exit 0 = ok, 1 = blocking, 2 = warning (API unreachable).
- `.env` is parsed the way compose (compose-go) parses it: `export ` prefix, blanks around `=`, single and double quotes (escapes in double quotes), anything after a closing quote ignored, and an unquoted value ending at the first ` #`. I checked each rule against `docker compose config` and pinned it in a table test. A parity test also checks python-dotenv where it agrees with compose. They disagree on `KEY= # x`, on a tab before `#`, and on text after a closing quote; compose wins because it is what feeds the containers. Multi-line quoted values are not followed, since no managed value can contain a newline.
- `--check` overlays shell-exported managed variables on `.env` (compose's precedence, empty-but-exported included). It names the overridden variables, never their values. It also rejects a key outside the AI Studio charset before any request, so a mangled value is never sent to Google.
- The merge changes only the value: an `export ` prefix, the quoting and any inline `# comment` stay as they were.
- Writes `RAGFORGE_LLM_PROVIDER` / `RAGFORGE_GEMINI_API_KEY` / `RAGFORGE_GEMINI_MODEL` / `RAGFORGE_GEMINI_THINKING_BUDGET`.
  - The write is atomic: mkstemp 0600 in the same directory, fsync, `os.replace`, then fsync the directory. `.env` is re-read right before the merge, so edits made while the user was at the prompts survive. The read, merge and rename all happen under an exclusive `flock` on the `.env` directory. No lock file is created. Because no other run can be mid-write while the lock is held, leftover `.env.setup-*.tmp` files are swept whatever their age. Without flock, it falls back to sweeping files older than 10 minutes. The file ends up mode 600. Lines it doesn't manage are kept byte for byte, and lines split on `\n` only.
  - **Round 4:** `.env` is now read as raw bytes decoded as UTF-8. The old `read_text()` used universal newlines, which rewrote CRLF to LF and split a value holding a lone CR. The byte-for-byte claim is now proved on disk (`test_g2_bytes_and_pro_budget.py`): CRLF, lone-CR and quoted lines go through `update_env` and the interactive `main()`, and every unmanaged byte is compared. `docker compose config` keeps a lone CR inside the value (`B=x\ry` gives `'x\ry'`, checked read-only), and `read_env_values` now agrees.
  - **Round 4:** a `.env` that is not UTF-8 raises `SetupError` naming only the line number, never the bytes. `make up --if-unset` then falls back to local (rc 0), `--check` blocks with one line (rc 1, no traceback), and `make setup` refuses to rewrite the file. These are proved in-process and through the real script as a subprocess.
  - Choosing local writes only `RAGFORGE_LLM_PROVIDER=ollama`. The saved key stays so switching back is easy.
  - A hand-set thinking budget is kept if it is within the Settings bounds (-1..32768). Otherwise it is reset to 0 and the user is told, so the saved config always starts.
  - **Round 4:** a Pro model (`-pro` in the id) is never saved with budget 0, because Gemini rejects `thinkingBudget=0` for Pro on every `generateContent`. Setup saves `-1` (dynamic) instead and says so. `-1` is used rather than 128 because 128 is a version-specific floor. `--check` returns 1 for a Pro model whose effective budget is 0, counting an unset budget as compose's `:-0` default (pinned to compose.yaml by a test), and it does so without calling Google.
- The key is checked with `GET /v1beta/models` and sent only in the `x-goog-api-key` header.
  - Values are checked against a strict character set before anything is written, so a pasted key or a remote model name can't inject lines into `.env`.
  - Error output shows only the HTTP code and Google's UPPER_SNAKE reason, never free text from the server.
- `Makefile`: `setup` target. `up` calls `--if-unset` before `compose up`, and only if python3 exists.
- `compose.yaml`: an `x-llm-env` anchor that only `api` and `worker` merge (`<<: [*db-env, *llm-env]`). Compose v2 rejects two `<<` keys in one mapping, so it is a single list merge. I checked it with `docker compose config` (read-only).
- **Redirects are refused.** The default opener (`_open_without_redirects`) swaps urllib's redirect handler for one that raises on any 30x. urllib would otherwise copy `x-goog-api-key` onto the redirected request, to any host or over plain http. A real loopback server pair proves the key reaches the first host once and the redirect target never sees a request (301/302/303/307/308). A control test shows the same opener works without a redirect.
- **Real-terminal proof.** `test_g2_terminal_and_redirects.py` runs the real script under a pseudo-terminal that is the child's controlling tty, with the real `TerminalConsole`, `input()` and `getpass`. Only the HTTP opener is a local fake. It runs on the test interpreter and on macOS `/usr/bin/python3` (3.9). It asserts:
  - the key never appears in the terminal output, while a visible answer is echoed, which proves the capture would catch a leak;
  - getpass printed no echo-fallback warning;
  - the header carried the key;
  - `.env` is 0600 with exactly the four values;
  - `--if-unset` asks under a terminal and is silent when a provider is saved.
  
  A mutation check (getpass replaced by input) fails it. `make -n up` is pinned to run that exact `--if-unset` command before compose, with stdin not redirected.
- `doctor.sh`: provider output is rendered line by line by `report_provider`. A `Note: ...` line (a shell export overriding .env) is always its own warning, and every other line carries the verdict's marker. A non-zero rc with no verdict line still counts as blocking. doctor.sh can be sourced (it then only defines its helpers), which is how the tests exercise this.
- `doctor.sh`: reports the provider through `--check`, which also blocks (rc 1) on a model id or thinking budget that Settings would reject, so a doctor "ok" means the api will start. The Ollama checks always run, because embeddings stay local.
- `bootstrap.sh`: a comment only. The provider is deliberately not written there, because the `make up` gate depends on it being absent.
- `.env.example`: every provider line is commented out, including `RAGFORGE_LLM_PROVIDER`, so `cp .env.example .env` keeps the first-run question. Uncommented, those lines parse to a config that `--check` accepts (tested). README section "Choosing a model provider".
- Tests: `packages/core/tests/module_g2/` (178 tests, including a hypothesis property test and real-subprocess non-TTY runs). An autouse `conftest.py` clears shell-exported provider variables. There is one `integration`-marked test that runs the real `docker compose config --format json` (read-only, no daemon needed) on a temp env file that uses export, quotes and inline comments. It asserts that only `api` and `worker` resolve the four variables, to exactly what `read_env_values` reads, and that the key appears nowhere else in the resolved config.

## Requests for other owners
1. **config.py owner (Settings):** `LLMProvider` currently has `ollama|openai|bedrock`. It needs `gemini`, plus `gemini_api_key: SecretStr`, `gemini_model: str` and `gemini_thinking_budget: int = 0` (the frozen names). If they're missing, `RAGFORGE_LLM_PROVIDER=gemini` makes api/worker fail at startup.
2. **config.py owner (resolved by G2, keep in sync):** compose now passes `RAGFORGE_GEMINI_MODEL=${RAGFORGE_GEMINI_MODEL:-gemini-3.5-flash-lite}`, equal to `Settings.gemini_model`'s default. `test_g2_settings_parity.py` fails if they drift, and it also pins the setup/`--check` model pattern and thinking-budget bounds (-1..32768) to the Settings field constraints. If you change the default or the bounds, tell G2/the integrator so compose, `.env.example` and `setup_provider.DEFAULT_GEMINI_MODEL` move with it. Alternative: set `env_ignore_empty=True` on Settings, after which compose could pass `${RAGFORGE_GEMINI_MODEL:-}` and the default would live only in config.py.
3. **pyproject owner:** `test_g2_compose_and_make.py` imports `yaml`. pyyaml is in `uv.lock` only as a transitive dependency, and its stubs aren't installed (hence the `# type: ignore[import-untyped]`). Please add `pyyaml` and `types-PyYAML` to the dev group, then drop the ignore.
4. **models.sh owner (unowned in Phase 2b):** with `RAGFORGE_LLM_PROVIDER=gemini`, `make models` could skip pulling the chat model and pull only `RAGFORGE_EMBED_MODEL`. That saves about 2 GB on a memory-tight machine. It's optional.
5. **Integrator (live end-to-end, not yet done; compose resolution is now proven by the real `docker compose config`):** after the config change, run `make setup`, pick Gemini, then `make up`, `make doctor` and one `make ask` turn. Then run `docker compose exec <svc> env | grep -c RAGFORGE_GEMINI_` for postgres (and `docker compose run --rm` for the one-shot db-bootstrap/migrate/index) and expect 0, and expect 4 for api and worker.
6. **.gitignore / .dockerignore owner (still recommended; G2 now sweeps leftovers on the next write whatever their age):** add `.env.setup-*.tmp`, or more broadly `.env.*` with `!.env.example`. The atomic write uses a temp file named `.env.setup-XXXX.tmp`, which is 0600 and holds the key. If the write is SIGKILLed or the machine loses power between mkstemp and rename, that file is left behind. The next write sweeps it, but until then they would be picked up by `git add` or sent in the docker build context.
7. **Test environment note:** during this round the Docker engine socket was down (`/var/run/docker.sock` was missing; I did not start it, per the rules). The full suite showed 1309 passed, 2 skipped, and 422 errors, all of them testcontainers `DockerException` connection errors in other modules. With `-m "not integration"` it showed 1299 passed and 26 errors, again all Docker connection errors, from the unmarked Postgres tests in module_a. All 160 G2 tests pass.
8. **ADR owner (requested by the round-3 review):** record the hosted-provider decision. Revise ADR-0004 (it still says `LLM_PROVIDER=openai|bedrock`) or add a short ADR covering:
   - hosted chat via Gemini, with local Ollama as the default;
   - an explicit consent prompt before any key is asked for;
   - the key visible only to api and worker (compose `x-llm-env`);
   - embeddings and the index always local;
   - free-tier vs paid-tier data use (relevant to ADR-0014 data classification);
   - shell-over-.env precedence.

   When it exists, G2 (or the README owner) links it from README "Choosing a model provider". G2 can't link it now without pointing at a file that doesn't exist.
9. **Integrator: live run, still not done by G2.** The G2 rules forbid `make up` and `docker compose up` on this machine, because the owner's stack is live. Run handoff item 5 and record the results:
   - `make setup` (Gemini), `make up`, then `curl localhost:8000/readyz` should return 200;
   - `make doctor` should show "API reachable";
   - one `make ask` answer should come from Gemini;
   - `env | grep -c RAGFORGE_GEMINI_` should give 4 in api and worker, and 0 in postgres, db-bootstrap, migrate and index.

## Needs live evidence (round 4; G2 cannot produce these under its rules: no make up, no real key)
These are the review's to9 items 4 and 5. They are not faked here.
1. **Integrator, live run.** The commands are listed in item 9 above:
   - `make setup` (pick Gemini, paste a real key; the input is hidden);
   - `make up && curl -fsS localhost:8000/readyz` should return 200;
   - `make doctor` should print `LLM provider: gemini (<model>), key set, API reachable`;
   - `make ask` should give one answered turn, with the api logs showing the Gemini adapter;
   - for each of `api worker`, `docker compose exec <svc> sh -c 'env | grep -c ^RAGFORGE_GEMINI_'` should print 4, and for `postgres` it should print 0;
   - `docker compose run --rm --no-deps <svc> sh -c 'env | grep -c ^RAGFORGE_GEMINI_'` should print 0 for `db-bootstrap migrate index`.
   Optional Pro check: set `RAGFORGE_GEMINI_MODEL=gemini-2.5-pro` and `RAGFORGE_GEMINI_THINKING_BUDGET=0`, run `make doctor` and expect a blocking line. Then set the budget to `-1` and expect one `make ask` to be answered.
2. **ADR owner.** Write the hosted-provider ADR (item 8). G2 does not own `docs/adr/**`. Once the ADR exists, add one line under README "Choosing a model provider": `Decision record: [ADR-00NN](docs/adr/00NN-....md).` G2 has not added that link yet, because it would point at a file that doesn't exist.
