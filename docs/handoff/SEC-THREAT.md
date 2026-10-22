# Handoff SEC-THREAT: threat model, rotation runbooks, egress audit

**Scope:** ADR-0018 §Security 4–5, plus Revision 2 R2-0 and R2-8 (the threat-model parts). Built 2026-10-22, in parallel with OBS-METRICS, OBS-STACK and SEC-SCAN. No git was used: the tree is not a repo.

## What was built
| Path | What it is |
|---|---|
| `docs/SECURITY.md` | The threat model, R2-0 and the runbooks. Sections: <ul><li>assets and trust boundaries (a text diagram, B1–B7);</li><li>11 threat areas, each with mitigations (cited to code, tests and ADRs) and its residual;</li><li>out of scope;</li><li>the **R2-0 ACCEPTED RISK** (no values);</li><li>rotation runbooks for the Notion token, API keys, Gemini key, KEK and DB roles;</li><li>the pre-first-commit checklist;</li><li>the "how this stays true" test map.</li></ul> |
| `packages/core/tests/egress_guard.py` | A pytest plugin. While a test's setup, call and teardown run, it refuses non-loopback `connect`, `connect_ex`, `sendto`, `sendmsg` and name lookups (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo`), and **fails the test even if the code swallowed the error**. <ul><li>Opt-outs: `@pytest.mark.allow_egress`, or `RAGFORGE_TEST_EGRESS_GUARD=off`.</li><li>It also pins LangSmith tracing OFF (env plus `langsmith.configure(enabled=False)` plus clearing langsmith's lru-cached env reads).</li></ul> |
| `packages/core/tests/test_sec_threat_egress_inventory.py` | The host inventory. Every URL literal of **any scheme** (`https`, `postgresql+psycopg`, `grpc`, ...; our `ragforge://` resource URIs excepted) in `packages/*/src`, `scripts`, `docker`, `compose*.yaml` and the `Makefile` (`.sql` included) is compared EXACTLY with `INVENTORY` (host → kind → files). <ul><li>It fails on a new host, on a known host used from a new file, and on a stale entry.</li><li>A literal host followed by a template (`https://x.org:{port}`, `https://x.org{path}`, `u:${PW}@x.org`) is reported; a partly templated host (`api.{region}.x.org`) is reported as non-local (fails closed).</li><li>It pins runtime egress = {api.notion.com, generativelanguage.googleapis.com}.</li><li>It pins the destination-valued Settings: URL defaults plus str fields ending `_host`/`_url`/`_endpoint`/`_addr`/`_uri` = {`ollama_base_url`, `db_host`, `metrics_addr`}, all local.</li><li>It pins the eval judge as local, opt-in for a third party, and unproxied.</li></ul> |
| `packages/core/tests/test_sec_threat_egress_guard.py` | The guard. <ul><li>Unit level: what it refuses and allows, that the socket layer is restored exactly (guards nest), and httpx/anyio reaching it.</li><li>End to end: a child pytest whose root `pyproject.toml` has the exact two keys from step 1 below. It runs only a sibling package's tests directory, which has no conftest. A swallowed refusal → failure, a leaky fixture → error, the marker and env opt-outs work, and LangSmith is pinned off even when the env says `true`.</li><li>A strict-xfail tripwire for the root registration.</li></ul> |
| `packages/core/tests/test_sec_threat_no_silent_egress.py` | Silent egress. <ul><li>LangSmith off, including what LangChain sees.</li><li>Nothing shipped (sources, scripts, `docker/**` incl. the Dockerfile, compose, Makefile) names `LANGSMITH_`/`LANGCHAIN_` or `langsmith`; no non-Python shipped file sets a `*_proxy` variable (both scans proven on planted input).</li><li>No compose service, in any compose file checked on its own (no merge by name), gets tracing, `*_PROXY` or `env_file`, or mounts `docker.sock`.</li><li>FlashRank: `allow_download` defaults False, no source passes anything but a literal `False`, and an empty cache dir refuses before flashrank is imported.</li></ul> |

## Integration steps (exact)
1. **Register the plugin from the ROOT `pyproject.toml`**, so every run is guarded, including a targeted `pytest packages/contracts` that never loads `packages/core/tests/conftest.py` (round-1 review finding). Add these two keys to the existing `[tool.pytest.ini_options]` table:
   ```toml
   # ADR-0018 R2-8: unit tests may only reach loopback; LangSmith pinned off (egress_guard.py).
   addopts = ["-p", "egress_guard"]
   pythonpath = ["packages/core/tests"]
   ```
   - The table has no `addopts` or `pythonpath` today. If another module adds `addopts`, append `"-p", "egress_guard"` to its list (the tripwire checks for the adjacent pair, not list equality, so other options may sit around it).
   - `pythonpath` only adds the directory to `sys.path`. No helper module in `packages/core/tests` shares a name with one in `packages/contracts/tests` (checked).
   - I checked that pytest 9 applies `pythonpath` before it imports `-p` plugins taken from `addopts`, both with a scratch project and with the child run in `test_plugin_fails_tests_that_leave_the_machine`. That test writes exactly these two keys and runs only a sibling package's tests directory, which has no conftest.
   - Do **not** register it in `packages/core/tests/conftest.py` as well. That is harmless (pytest skips a second import of the same plugin name), but it is redundant.
2. **In the same change, delete the `@pytest.mark.xfail(strict=True, ...)` decorator** above `test_root_pyproject_registers_the_plugin` in `packages/core/tests/test_sec_threat_egress_guard.py`. After step 1 that test XPASSes, which fails a strict xfail. From then on it is a tripwire: it checks the two keys and that the plugin is active.
3. **Nothing else to apply:**
   - No Makefile, compose or bootstrap change. The plugin registers its own `allow_egress` marker.
   - `make test` and `make test-unit` run `pytest packages`, so they pick up step 1 with no further change.

## Verified here (not live)
- **Round 3** (review findings fixed): `uv run pytest packages/core/tests/test_sec_threat_*.py -q` → **57 passed, 1 xfailed**; the same with `PYTHONPATH=packages/core/tests … -p egress_guard`. ruff check/format, mypy (strict) clean on the 4 Python files; `lint-imports` 6 kept, 0 broken. Mutation checks, each caught and restored:
  - restore the old URL lookbehind `(?<![A-Za-z0-9+.-])` → the `${X:-https://…}` / `${X:+…}` scanner test and the real-compose Ollama-default test fail;
  - drop the Dockerfile `ENV KEY value` proxy form → the teeth test fails;
  - drop the parent-directory socket check → the planted `/var/run:/var/run`, `/run/`, `~/.rd`, `${HOME}/.docker`, `/` mounts test fails;
  - compare addopts by equality → the appended-addopts tripwire test fails.
- **Earlier rounds:**
- `uv run pytest packages/core/tests/test_sec_threat_*.py -q` → **53 passed, 1 xfailed** (the tripwire). The result is the same with `-p egress_guard`.
- **The full unit suite with the guard on** (`PYTHONPATH=packages/core/tests uv run pytest -p egress_guard packages -m "not integration"`), after round 2 → **4866 passed**, 3 skipped, 15 xfailed, 0 failed.
  - An earlier run of the same command had one failure, `module_obs/test_obs_metrics_server.py::test_the_listener_serves_the_catalogue_and_build_info`. It was an assertion on catalogue content, not an `egress_guard` refusal.
  - That module then passed 3 runs out of 3 under the guard, and the full rerun was green. It looked like OBS-METRICS's files changing in the middle of the run. If it recurs, look there first.
- `uv run pytest -o pythonpath=packages/core/tests -p egress_guard packages/contracts` → 13 passed. This is the targeted-package case that step 1 now covers.
- **testcontainers under the guard:** `test_db_foundation.py` → 17 passed. The Docker API is a unix socket and the container ports are on localhost.
- **Lint and types:** ruff check and format --check are clean on my 4 Python files; `mypy` (strict, project config) is clean on them; `lint-imports` → 6 kept, 0 broken.
- **Round-2 mutation checks**, each caught by ≥ 1 test, all files restored:
  - skip any authority that contains a template (the round-1 P1 bug);
  - treat a partly templated host as local;
  - narrow the scheme back to http(s)/ws(s);
  - drop the `*_host`/`*_addr` Settings check;
  - unpatch `sendmsg`, `gethostbyname` or `getnameinfo`;
  - merge compose services by name;
  - drop the proxy scan for non-Python files;
  - drop `LANGCHAIN_` from the tracing words.
- **The §6.5 superuser pipe** (also step 3 of the §3.9 recovery) was run in **bash and zsh** against a throwaway `docker run --rm` of the project's pinned pgvector/pg17 image, with a synthetic value. Afterwards the new value authenticated over TCP (scram), the old value was refused, and nothing was printed. The SECURITY.md text is byte-identical to the tested script, with `docker exec -i <name>` swapped for `docker compose exec -T postgres`.
- **Round-1 mutation checks**, each caught by ≥ 1 test, all files restored:
  - drop the post-test `pytest.fail` (a swallowed refusal passes);
  - drop the langsmith cache and global reset from the pin;
  - make every IP local;
  - leave `sendto` unguarded;
  - rename an INVENTORY host;
  - drop a file from a host's file set.
- **gitleaks** (the project's pinned image and `.gitleaks.toml`) over SECURITY.md, the three test files and the plugin → no leaks. This handoff contains no token shapes.
- **The runbook's `rfapi` helper** was exercised in **bash and zsh** against a dummy loopback server, with a synthetic key file: the header arrives and no key appears in argv. This caught a zsh bug: `local path` clobbers `$PATH`, so the variable is now `route`.
- **The iCloud / Drive / Time Machine check** (SECURITY.md §3.8) was run on this machine: none of them sync `~/Desktop`.

## What the integrator must test live
1. **After step 1:** `make check`. This includes `make typecheck`, which mypy-checks `packages/core/tests/*.py`, so egress_guard is covered. Then `make test`, **with integration tests**, on Docker: the whole integration suite now runs under the guard.
   - **If Docker is not on a local unix socket** (for example `DOCKER_HOST=tcp://…`), testcontainers' host is not loopback. Run with `RAGFORGE_TEST_EGRESS_GUARD=off`, or point DOCKER_HOST at the local socket.
2. **When OBS-STACK's compose services land in `compose.yaml`:**
   - Rerun `test_sec_threat_no_silent_egress.py`. It parses every `compose*.yaml` and fails on any `LANGSMITH_*`/`LANGCHAIN_*`/`*_PROXY` env, on `env_file:`, or on a `docker.sock` mount.
   - Rerun the inventory test too. Dotless service names (`prometheus`, `grafana`) are local by rule. A **dotted** public host in `docker/observability/**` (say a `grafana.com` link in dashboard JSON) fails it **on purpose**: add it to `INVENTORY` as `inert` (with the file) and to SECURITY.md §3.5 in one change.
3. **Confirm the facts SECURITY.md cites from the other modules once they are integrated:**
   - `observability/logging.py` `redact_secrets` is in the processor chain (OBS-METRICS);
   - `make secrets-scan` and `make audit` exist (SEC-SCAN);
   - Grafana has `GF_ANALYTICS_*`/news off and binds `127.0.0.1:3000` (OBS-STACK);
   - its admin password is generated into `.env` by `bootstrap.sh` (R2-7). SECURITY.md §1 and §6.5 assume this.
   
   If one did not land as designed, fix the matching line in §3.4/§3.5/§3.6.
4. **Optional, read-only:** run `make doctor`, then follow SECURITY.md §6.0 → `rfapi <owner key> GET /v1/api-keys` against the live API. That confirms the helper end to end, and it prints prefixes only.

## Locked seams
`observability/catalog.py` and `prompt_context.py` were not touched, and this module does not depend on them. Nothing is missing from them for this module.

## Gaps found (not fixed: outside my paths, or new features)
0. **Not mutation-proven:** the `tracing_env` fixture's teardown re-pins LangSmith off when the plugin is active (round-1 P2). It only matters once step 1 lands, and the session-wide pin is already proven by the child run's `test_langsmith_pinned_off`. A dedicated test would depend on test order, so I did not add one.
1. **There is no way to replace a connector's token.** Notion rotation means connecting again plus a full re-sync, then deleting the old connector (runbook §6.1). The fix: a `PATCH /v1/connectors/{id}` that re-seals the token under the same associated data.
2. **There is no KEK rewrap command.** `EnvelopeSecretBox.rewrap` exists but nothing wires it up, so rotating the KEK requires re-connecting Notion (§6.4). The fix: a `ragforge-admin rewrap-kek` that reads the old and new KEKs as hidden input and rewraps every `wrapped_dek` in one transaction.
3. **An eval key cannot be rotated, and its id is not recorded.** `POST /v1/eval/tenant` is once-only (409), and `ragforge-admin create-key` cannot grant `eval`. A leaked eval key can only be revoked, and only by the superuser (`make psql`, `UPDATE rf.api_keys SET revoked_at = now() WHERE prefix = …`; §6.2), because the eval key cannot list keys and the owner key cannot see eval-tenant rows (RLS). Suggested fix (outside my paths): `ragforge-eval2 tenant create` should print the key id and prefix (never the secret) and save the id next to the key file; or add `DELETE /v1/api-keys/self` so any key can revoke itself without knowing its id.
3a. **Eval keys are guarded at creation only.** `effective_scopes` and the 0006 CHECK cover `ClientKind.mcp` only; eval keys are `ClientKind.user`. A DB CHECK such as `NOT ('eval' = ANY(scopes) AND scopes && ARRAY['approvals:write','admin'])` would give them the same defense in depth (a new migration; not this module).
3b. **The egress guard does not see uvloop or subprocesses** (SECURITY.md §3.5 residual, plugin docstring). Optional hardening for later: have the plugin fail a test whose running loop is a uvloop loop unless it is marked `allow_egress`. Not done here because uvicorn's `loop="auto"` picks uvloop, and a blanket refusal could break future loopback server tests.
4. **The key routes are `/v1/api-keys`**, not `/v1/keys` as the task text said. The runbook uses the real paths.
5. **The file name differs from the ADR.** ADR-0018's build plan names `test_security_egress.py`; this module's rule required the `test_sec_threat_` prefix. Worth a one-word ADR touch-up when the integrator edits docs.
6. **The secrets live in container env** (`docker inspect`). Compose file secrets are parked, as recorded in §3.7.

## Policy note
The managed policy asks for a Jira ticket on git operations. None were run: the tree is not a repo, and the task forbids git.
