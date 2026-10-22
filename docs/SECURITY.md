# RAGForge security: threat model, accepted risks and rotation runbooks

**Status:** Phase 4 security pass ([ADR-0018](adr/0018-ops-hardening-security-observability.md) §Security 4–5, Revision 2 R2-0 and R2-8) · written 2026-10-22 · owner: the repo owner (single user).

**What it covers:** RAGForge as it runs today: one Mac (M1 Pro, Rancher Desktop VM, host Ollama), one owner, the compose stack in `compose.yaml`, and the clients `make ask`, `make ui` and `make mcp`.

**How to read it:**
- Every mitigation names the code, test or ADR that implements it, so a claim can be checked.
- Every threat ends with its **residual**: what is still possible after the mitigation.
- The tests that keep this document true are listed in [§9](#9-how-this-document-stays-true). If one of them fails, this document is wrong somewhere.

---

## 1. Assets

| Asset | Where it lives | If it leaks |
|---|---|---|
| **The owner's notes** (page text, chunks, embeddings, conversation history) | Postgres volume `pgdata` (Docker VM); Notion itself | The owner's private notes are exposed. They can be re-synced from Notion, so loss matters less than disclosure |
| **Notion integration token** | `rf.connectors.secret_ciphertext`, envelope-encrypted under the KEK (`adapters/crypto/envelope.py`) | Read access to every Notion page shared with the integration, until the token is regenerated in Notion |
| **RAGForge API keys** (owner, MCP, eval, demo) | Plaintext only in the owner's key files `~/.config/ragforge/**/*.key` (0600); the DB stores only `sha256(key)` (`services/auth/keys.py`) | Whatever the key's scopes allow, inside its own tenant, until it is revoked |
| **Gemini API key** (only when the hosted provider is chosen) | `.env` (`RAGFORGE_GEMINI_API_KEY`), passed only to `api` and `worker` (`compose.yaml` `x-llm-env`) | Billing abuse on the owner's Google account; no access to the notes |
| **KEK** (`RAGFORGE_KEK`) | `.env` only, mode 0600 (`scripts/bootstrap.sh`) | With a DB copy: every connector token. It also keys the interview question tokens (an HKDF subkey; `bootstrap._question_tokens`) |
| **DB role passwords** (superuser, migrator, api, worker, metrics, backup) | `.env` only, 0600 | Postgres is not published on the host, so this needs access to the Docker network first |
| **Eval data** (labels, runs, the held-out set) | `evals/private/` (git-ignored); backups under `~/.local/share/ragforge/eval-backups/` | Contains questions and answers derived from the notes. It is **irreplaceable**: it cannot be re-synced |
| **The Grafana admin password** (obs profile, opt-in) | `.env`, generated like the DB passwords (R2-7) | Read access to metrics. Labels are bounded enums with no note text or ids (`observability/catalog.py`) |

**Classification:** [ADR-0014](adr/0014-data-classification-and-backups.md) splits tables into authoritative and derived (DESIGN §6.4).

---

## 2. Trust boundaries

```
 ┌──────────────────────────── the owner's Mac (macOS user account) ─────────────────────────────┐
 │                                                                                                │
 │  Browser ──(B1: 127.0.0.1:8501, Host/Origin checks)──▶ Streamlit UI ──┐                        │
 │  Terminal: make ask / connect-notion / setup (hidden input) ──────────┤ bearer API key         │
 │  Claude Code / Cursor ──(B2: stdio, or HTTP + bearer token)──▶ MCP ───┤ (read from 0600 files)  │
 │        │                                                              ▼                        │
 │        │ (B6: MCP answers go to the agent's own model provider)                                │
 │        │                                                   (B3: 127.0.0.1:8000 only)          │
 │  ~/.config/ragforge/*.key (0600)   .env (0600)   ~/.claude/projects/*.jsonl  (B7: disk)      │
 │                                                              │                                 │
 │  ┌──────────────── Rancher Desktop VM: compose network (B4: Docker socket = root) ─────────┐  │
 │  │  api :8000 ──▶ agent ──▶ Postgres (RLS, definer functions; not published)              │  │
 │  │  worker ──▶ Notion client, parser sandbox      Prometheus/Grafana (profile obs)         │  │
 │  └──────┬──────────────────┬──────────────────────┬────────────────────────────────────────┘  │
 │         │ host.docker.internal:11434               │                                          │
 │         ▼                                          │                                          │
 │    host Ollama (chat + embeddings, local)          │                                          │
 └────────────────────────────────────────────────────┼──────────────────────────────────────────┘
                     (B5: the internet)               ▼
         api.notion.com (token)        generativelanguage.googleapis.com (only if Gemini is chosen)
```

| Boundary | What crosses it | Main control |
|---|---|---|
| **B1** Browser → UI | Page loads, uploads | Loopback bind, Host/Origin checks, Streamlit XSRF (§3.4) |
| **B2** Agent client → MCP | Tool calls, tool results | No `approvals:write` on MCP keys, HTTP bearer + DNS-rebinding protection (§3.3, §3.4) |
| **B3** Clients → API | Bearer key, questions | Key hashing, scopes, RLS (§3.1, §3.2) |
| **B4** Docker socket | Full control of every container | Only the owner's user holds it; no container mounts it (§3.7) |
| **B5** Stack → internet | Notion calls; Gemini calls if chosen | The egress inventory test (§3.5) |
| **B6** MCP client → its model provider | Answers, note excerpts | **Outside RAGForge's control**: using MCP sends notes to the agent's provider by design |
| **B7** Disk outside the stack | `.env`, key files, chat transcripts | File modes; the sync check (§3.8) |

---

## 3. Threats, mitigations and residuals

### 3.1 Tenant isolation (one tenant reads another's notes)
**Mitigations:**
- **RLS fails closed.** `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid`: an unset tenant sees nothing. The setting is transaction-local and bound as a parameter (DESIGN §6.1, [ADR-0013](adr/0013-db-roles-and-tenancy-model.md)).
- **No role bypasses RLS.** Tables are owned by `rf_system` (NOLOGIN); no app role is a superuser or has `BYPASSRLS` (`db/bootstrap.sql`).
- **The only cross-tenant paths are named `SECURITY DEFINER` functions.** Each one has a pinned `search_path` with `pg_temp` last, `REVOKE ALL FROM PUBLIC`, explicit grants, and one cross-tenant test per function (ADR-0013).
- **The eval tenant guard.** The `/v1/eval/*` surface 404s unless `RAGFORGE_EVAL_API` is on, before authentication. It then needs the `eval` scope **and** a tenant whose `kind = 'eval'` (`apps/api/deps.py` `require_eval_key`). So an eval key minted into an owner tenant by some future path still cannot read live notes through `/v1/eval/chunks`.
- **The cross-tenant copy guard.** A snapshot copies a collection only into the source tenant's **paired** eval tenant. The pairing comes from the definer (`rf.paired_eval_tenant()`), never from the request, and the database re-checks it (`kind = 'eval' AND eval_of_tenant_id = source` in `rf.copy_collection_to_eval`, migration 0012; `apps/api/routes/eval_snapshots.py`).

**Residual:** RLS protects against application bugs, **not** against a stolen `rf_api` credential: any `rf_api` session can set `app.tenant_id` itself (DESIGN §12, threat model note). Here that credential lives only in `.env` and in the api container's env (§3.7).

### 3.2 API keys (a key is stolen, or a key does more than intended)
**Mitigations:**
- **Hashed at rest.** Only `sha256(full key)` is stored, as raw bytes. The 256-bit secret makes a fast hash correct, and hashing the full key binds the digest to its prefix (`services/auth/keys.py`).
- **Shown once.** A key is returned once at creation with `Cache-Control: no-store` (`apps/api/routes/identity.py`). `GeneratedKey.__repr__` masks it.
- **Scoped, with no escalation.** A key mints only subsets of its own scopes (`services/auth/scopes.py`).
- **MCP and eval keys never hold `approvals:write` (or `admin`).** The two are enforced differently:
  - **MCP keys, three ways:** at creation (`check_mcp_scopes`), at authentication (`effective_scopes` drops the forbidden scopes for `ClientKind.mcp`), and by a DB CHECK (`api_keys_mcp_never_privileged`, migration 0006).
  - **Eval keys, at creation only:** `check_eval_scopes`, and the eval-tenant definer path mints the fixed `EVAL_KEY_SCOPES`. An eval key is a `ClientKind.user` key, so neither `effective_scopes` nor the 0006 CHECK applies to it. A row edited by hand to add `approvals:write` would be honoured.
  - Privileged scopes also need a key name that positively declares a human client (`ui:` / `user:`).
- **Any key can revoke itself.** No scope is needed: the holder of a leaked MCP key can kill it (`DELETE /v1/api-keys/{id}`).
- **Key files must be private.** Clients refuse a key file that group or others can read (`ragforge_client/ask.py` `read_key`). `make eval-demo` / `make eval-tenant` write keys with `umask 077` and never print them.
- **Secrets are entered as hidden input** (`getpass`): `make connect-notion` and `make setup`. They never appear in shell history, `ps` output or a chat (CLAUDE.md "Secrets").

**Residual:**
- A key that leaks stays valid until it is revoked. There is no expiry (the R2-0 accepted risk, §5).
- An owner key holds `approvals:write`: whoever has it can mint keys.
- The audit log records key creation and revocation (`api_key.create`, `api_key.revoke`), but nothing alerts on them.

### 3.3 Prompt injection (note or tool text steers the model)
**Mitigations:**
- **Fencing.** Retrieved notes, tool results and the answer being verified are fenced and neutralized before they reach a model (`agent/fencing.py`; [ADR-0011](adr/0011-streaming-verification-and-output-safety.md) rule 7).
- **An incremental output sanitizer** runs on the server before any `token` event, with link and image rules (ADR-0011 §1–2).
- **The verifier.** It runs a deterministic citation check and an LLM ground check (ADR-0011 §4–5, [ADR-0015](adr/0015-retrieve-then-answer.md)). A model below the verifier SLO shows "unverified", never a false "verified".
- **Literal extras.** General-knowledge extras are never verified, cited or committed to `messages`. The agent node only extracts structure from them and never rewrites characters, and clients show them apart, under "Not from your notes" ([ADR-0016](adr/0016-labelled-extras.md)).
- **Human in the loop.** Destructive actions need an approval with the `approvals:write` scope, which only human-held keys have (DESIGN §10).
- **Inert agent-facing text.** The MCP answer keeps `\ [ ] < > & `` ` `` escaped and marks extras `not_from_notes: true` (DESIGN §10).

**Residual: MCP indirect prompt injection into agentic tools.** A note (or a Notion page someone else can edit) can carry instructions. Through `ask` or `search`, those instructions reach Claude Code or Cursor, **which also have their own tools** (shell, file edits, web fetch). RAGForge's controls stop the injected text from approving RAGForge actions. They **cannot** stop the agent client from acting on that text with its own tools. Two mitigations are partial:
- **Keep the agent client's tool approvals on.** Don't run it with auto-approve while it is reading notes.
- **Give the MCP key only `read, chat`.** Leave out `write`, so `sync_now` is the only side effect it could trigger (docs/handoff/MCP.md, step 2).

### 3.4 Local UI and local endpoints (another web page or host reaches them)
**Mitigations:**
- **API.** Published only on `127.0.0.1:8000` (`compose.yaml`). Auth is a bearer header, never a cookie, and there is no CORS middleware. So a page the owner visits cannot ride the owner's credentials: the browser sends no key, and the preflight for an `Authorization` header fails.
- **UI.** It binds to `127.0.0.1:8501` with `--server.allowedHosts` set to localhost and 127.0.0.1 (`Makefile` `ui`). The app also checks the session's Host and Origin itself (`ragforge_ui/config.py` `host_refused` / `origin_refused`), because a loopback bind does not stop DNS rebinding. `RAGFORGE_UI_ALLOW_REMOTE=1` is the only opt-out.
- **Streamlit XSRF.** Protection stays at Streamlit's default (on); nothing in the Makefile or the UI turns it off.
- **MCP over HTTP.** It needs `Authorization: Bearer <token>`, a separate token rather than the API key. DNS-rebinding protection is always on, with explicit allowed hosts and origins (`ragforge_mcp/server.py` `transport_security`). A non-loopback bind needs `--allow-remote` and a concrete IP.
- **Metrics and Grafana.** `/metrics` is never on the API port. The metrics ports 9101 and 9102 are not published. Grafana binds to `127.0.0.1:3000` with anonymous access and sign-up off (R2-6, R2-7).

**Residual:**
- Any **local process running as the owner** can call the API with a key it can read. The file modes protect against other users only, not against the owner's own processes (see §3.7).

### 3.5 Egress (data leaves the machine)
**The exact list.** Pinned by `packages/core/tests/test_sec_threat_egress_inventory.py`, which fails on a new host, on a known host used from a new file, and on a stale entry:

| Destination | When | What is sent | Code |
|---|---|---|---|
| `api.notion.com` | Every sync poll (worker) | The integration token; page and block ids | `adapters/connectors/notion/client.py` |
| `generativelanguage.googleapis.com` | **Only** when `RAGFORGE_LLM_PROVIDER=gemini`, or when the eval judge runs with `--allow-third-party`, or once during `make setup` to check the key | The question, packed note excerpts, prompts; the key goes in the `x-goog-api-key` header only | `adapters/llm/chat_gemini.py`, `ragforge_client/evals/judge.py`, `scripts/setup_provider.py` |
| Host Ollama (`host.docker.internal:11434` / `localhost:11434`) | Every chat and embedding with the local provider | Prompts, notes | `config.py` `ollama_base_url`; `compose.yaml` |

**Not runtime egress:**
- **Image build only:** `huggingface.co` (the reranker model, pinned to a commit and a sha256 checksum, `docker/Dockerfile`), the base images (pinned by digest), and PyPI through `uv` (locked).
- **Model pulls:** `make models` pulls from the Ollama registry, through Ollama itself.
- **Inert strings, never fetched:** `www.notion.so` (deep links the owner's browser opens), `schemas.openxmlformats.org` (an XML namespace) and `aistudio.google.com` (a link printed for the owner).

**Silent egress, pinned off:**
- **LangSmith.** It is installed with LangChain and as a pytest plugin, and it would trace if its env said so. No compose service gets `LANGSMITH_*` / `LANGCHAIN_*` variables, and no shipped source names them. The test process forces tracing off and drops any LangSmith key (`packages/core/tests/egress_guard.py` `pin_langsmith_off`). Pinned by `test_sec_threat_no_silent_egress.py`.
- **FlashRank** downloads its model when the model directory is missing. The adapter defaults to `allow_download=False` and refuses before importing flashrank (`adapters/rerank/flashrank.py`). Bootstrap never passes the flag, and the model is baked into the read-only image. Pinned by the same test file.
- **onnxruntime telemetry** (under FlashRank, in the API). The locked onnxruntime 1.31 links Microsoft's 1DS telemetry client with a built-in collector, `mobile.events.data.microsoft.com`, in both the macOS wheel and the Linux aarch64 one in the image, and starts it when onnxruntime is imported (in the container it logs `Failed to persist telemetry device ID` at import). Two switches, both checked against the installed library:
  - `ORT_DISABLE_TELEMETRY=1` stops the client from starting at all. In the container, an import with it set logs nothing; with it unset or `0` the device-id line appears. It is set in the image's `ENV` (`docker/Dockerfile`, every entrypoint) and forced to `1` when `adapters/rerank/flashrank.py` is imported, before anything in the process can import onnxruntime.
  - `onnxruntime.disable_telemetry_events()` turns event collection off. The adapter calls it before flashrank builds its first `InferenceSession`. If a future onnxruntime drops the function the call is skipped, not crashed, and the env switch still applies.
  - Pinned by `test_sec_threat_no_silent_egress.py`: the call happens before the session, the env reads `1` at session time even when the shell said `0`, a fresh interpreter has it forced after import, the locked onnxruntime has the function and its native library reads the variable, and the image sets it.
- **Proxies.** No compose service gets `HTTP(S)_PROXY`/`ALL_PROXY`. A local eval judge builds its HTTP client with `trust_env=False` (`judge.py`).
- **Grafana phone-home** (obs profile) is off: `GF_ANALYTICS_*` and the news feed (R2-7, OBS-STACK).
- **Streamlit usage telemetry** is off on every launch path: `make ui` passes `--browser.gatherUsageStats false`, and the repo-root `.streamlit/config.toml` sets `gatherUsageStats = false` for any `streamlit run` started from the repo. Pinned by `test_sec_threat_streamlit_telemetry.py`.

**In unit tests:** the `egress_guard` pytest plugin refuses every non-loopback `connect`/`connect_ex`, `sendto`/`sendmsg` and name lookup (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo`), and fails the test even when the code under test swallowed the error. It is registered from the root `pyproject.toml`, so a run of one package (`pytest packages/contracts`) is guarded too. `@pytest.mark.allow_egress` opts one test out. testcontainers and respx keep working.

**Residual:**
- **Notes do leave the machine in two places, by the owner's choice:** to Google when Gemini is selected, and to the MCP client's model provider whenever the owner uses MCP (B6). The local default (Ollama) sends nothing out.
- Server-side httpx clients honour proxy variables (the default `trust_env=True`). Compose passes them none, but a proxy exported in the owner's shell does apply to `make ask` and the CLIs.
- The inventory reads URL **literals** of any scheme (`https://`, `postgresql://`, `grpc://`, ...). A literal host followed by a templated port or path is still reported, and a host that is only partly literal (`api.{region}.x.org`) fails the test. A host assembled entirely at runtime slips past it; that is why the destination-valued Settings (URLs and `*_host`/`*_url`/`*_endpoint`/`*_addr` fields) are pinned separately to local defaults.
- **What the socket guard cannot see.** It patches Python's `socket` module, so anything that opens sockets or resolves names without it is unguarded:
  - **C/Rust network stacks:** libpq (it reaches only the test Postgres on localhost), and **uvloop**, which is installed through `uvicorn[standard]`. Under `uvloop.run(...)`, `asyncio.open_connection("api.notion.com", 443)` resolves and connects with nothing refused. No test runs a uvloop loop today; a test that does must be treated as unguarded.
  - **Child processes:** a subprocess (`pytest`, `uv`, `curl`, `docker`) runs without the patch. The plugin's own end-to-end test starts a child pytest, which registers the guard itself.
  - **Bound early:** a lookup function bound with `from socket import gethostbyname` before the test started (the patch replaces module attributes).
  - **Outside a test:** code that runs at import or collection time.

### 3.6 Secrets at rest, in logs and in errors
**Mitigations:**
- `.env` is created with mode 0600 and never overwritten (`scripts/bootstrap.sh`). `make setup` rewrites it atomically, keeping mode 600.
- Every service receives **only** the secrets it needs. There is no `env_file:` (`compose.yaml`; a G2 compose test and `test_sec_threat_no_silent_egress.py`).
- `.env` is excluded from the image (`.dockerignore`), and the image runs as uid 10001 with a read-only filesystem, `cap_drop: ALL` and `no-new-privileges`.
- Secrets are pydantic `SecretStr` (`config.py`). The Gemini validator echoes only the empty key, never the DB passwords.
- **Log redaction.** The final rendered line of every log record is masked for the token shapes: `rf_…` (secret part), `ntn_…`, `secret_…`, `AIza…`, bearer and `Basic` credentials, and the password in a URL's userinfo (`scheme://user:[REDACTED]@`, a DSN in a driver error). This is `observability/logging.py` `_RedactingRenderer` (R2-8, OBS-METRICS). It covers standard-library records too (uvicorn, httpx, SQLAlchemy, LangGraph, flashrank): `configure_logging` installs the one root handler, so a library's later `logging.basicConfig` adds nothing raw. uvicorn's access log is off (`apps/api/main.py`, `access_log=False`). The Notion client also strips its own token from any error text (`notion/client.py`).
- **Problem responses** never echo a key; `KeyFileError` names the file, never the key.
- **Secret scan.** `make secrets-scan` runs gitleaks with custom rules for the four token shapes (`.gitleaks.toml`, SEC-SCAN).

**Residual:** `docker compose config` prints `.env` values after interpolation. Treat its output as secret, and never paste it into a chat.

### 3.7 The Docker socket (`docker inspect` shows env secrets)
Secrets reach containers as environment variables, so anyone who can talk to the Docker socket can read them. `docker inspect ragforge-api-1` lists `RAGFORGE_KEK`, `RAGFORGE_DB_API_PASSWORD` and `RAGFORGE_GEMINI_API_KEY` in clear text; `docker exec … env` does too. The socket is root-equivalent on the VM.

**Mitigations:**
- No RAGForge container mounts the socket (pinned by `test_sec_threat_no_silent_egress.py`).
- Trivy scans `docker save` tarballs, not the live socket (R2-8).
- Only the owner's macOS user can open the Rancher socket.

**Residual (accepted):** any process running as the owner, **including an AI coding agent with shell access in this repo**, can read every secret through `docker inspect`, or simply `cat .env`. Only policy stops it (CLAUDE.md: never read `.env`). Compose file-based secrets (`secrets:` mounted under `/run/secrets`) would take them out of `docker inspect`, but are **parked**: every service would need a read-from-file path for each setting.

### 3.8 Does the repo leave the machine? (iCloud Desktop sync)
The repo is under `~/Desktop`. With macOS "Desktop & Documents Folders" in iCloud Drive turned on, that folder (`.env`, `evals/private/`, key-free but note-derived run files) would be copied to Apple's servers.

**Checked 2026-10-22, on this machine:**
- iCloud Drive is not in use: `~/Library/Mobile Documents/com~apple~CloudDocs` does not exist, so neither does its `Desktop` folder.
- No File Provider sync folders exist (`~/Library/CloudStorage` is absent).
- Google Drive is installed but was never signed in: there is no account folder under `~/Library/Application Support/Google/DriveFS`, and it is not running.
- Time Machine has no destination (`tmutil destinationinfo`).

**Result:** the repo does **not** leave the machine through sync today.

**How the owner re-checks it** (after an OS update, a new Apple ID sign-in, or installing a sync tool):
1. **System Settings → [your name] → iCloud → iCloud Drive:** "Desktop & Documents Folders" must be **off**.
2. **In a terminal,** both of these must say "No such file":
   - `ls -d ~/Library/Mobile\ Documents/com~apple~CloudDocs/Desktop`
   - `ls ~/Library/CloudStorage`
3. **Google Drive / Dropbox / OneDrive:** if signed in, their "back up folders" lists must not include Desktop.
4. **Time Machine:** `tmutil destinationinfo`. A local, encrypted Time Machine disk is acceptable. It does copy `.env`, so treat the disk as secret.

**Residual:** none of these checks is automated. If sync is ever needed, move the repo out of `~/Desktop` (for example to `~/Dev/ragforge`). Running `make nuke && make up` after the move is not required: volumes are named, not bind-mounted.

### 3.9 KEK and DB role passwords: backup and rotation
- **Where they live:** only in `.env` (0600). There is no other copy, and none should be made in a synced folder or a chat.
- **If `.env` is lost:**
  - **The KEK:** connector tokens become unreadable. Recover in this order: run `make env` (a fresh KEK), `make up`, **delete the old connectors** (`rfapi … DELETE /v1/connectors/<id>`; it needs no decryption, and their syncs can only fail now), then re-connect Notion (the token can always be regenerated in Notion; runbook §6.4 steps 3–5). As in §6.4, the notes are unavailable until the new connector's first sync completes.
  - **The DB passwords, in this order** (the superuser first: `db-bootstrap` logs in as the superuser over TCP, so `make up` fails before any app role is touched while the volume still holds the old superuser password):
    1. `make env` writes new values for every missing password, the superuser's included.
    2. `docker compose up -d --wait postgres` starts only the database. Postgres ignores `POSTGRES_PASSWORD` on an existing volume.
    3. Set the superuser to the new `.env` value with the §6.5 pipe. It is read from `.env` inside the pipe and never printed; `psql` connects over the container's local socket, which needs no password.
    4. `make up`. `db-bootstrap` re-runs `ALTER ROLE … PASSWORD` for every app role on each start, so the app roles follow `.env` (`db/bootstrap.sql`: "also rotates passwords").
- **Backup advice:** an offline copy of the KEK (a password manager entry) saves only one Notion re-connect, so it is optional. Never back `.env` up to a synced folder.
- **Rotation:** runbooks §6.4 (KEK) and §6.5 (DB roles).

### 3.10 Backups
- **What is backed up:** `make backup-evals` copies `evals/private/` (the only irreplaceable data) to `~/.local/share/ragforge/eval-backups/<timestamp>/`. The directory is created 0700, the copy is made under `umask 077`, and `chmod -R go-rwx` follows (`Makefile`).
- **Why that location:** `~/.local/share` is **outside the repo and outside `~/Desktop`**, so it is not swept up by iCloud Desktop sync even if that is turned on later.
- **What is not backed up:** notes, chunks and embeddings (re-synced from Notion), and conversations (accepted loss). A full `pg_dump` and restore drill are parked (ADR-0018 scope table).

**Residual:** the backups are plaintext on the same disk. They protect against an accidental `rm` or `make nuke`, not against losing the disk. FileVault (full-disk encryption) is the control for a stolen laptop. Check it is on under **System Settings → Privacy & Security → FileVault**.

### 3.11 Supply chain
- **Pinned inputs:** base images and the Postgres/Ollama images are pinned by digest. Python dependencies are locked in `uv.lock`. The reranker model is pinned by commit and sha256.
- **Scans:** `make audit` runs pip-audit over the lock and Trivy (`--scanners vuln,secret,misconfig`) over the image tarballs. Findings are fixed by version bumps or ignored with a written justification (SEC-SCAN, R2-8).

**Residual:** a compromised upstream release that is pinned *before* it is known bad. The scans catch it only once it is published as an advisory.

---

## 4. Out of scope
- **A compromised host OS or macOS user account,** including malware running as the owner. It can read `.env`, the key files and the Docker socket. FileVault and the OS are the controls.
- **A compromised Docker / Rancher Desktop VM or the Docker daemon.**
- **A compromised Gemini service, Notion service or Ollama build.** Each sees what it is sent, by design.
- **The MCP client's model provider** (B6) and the agent client's own tools: that trust is the owner's choice when installing MCP.
- **Physical attacks, shoulder-surfing, and Apple ID compromise** (beyond the sync check in §3.8).
- **Multi-user deployment hardening** (Helm, NetworkPolicies, short-lived DB credentials): parked in ADR-0018.

---

## 5. ACCEPTED RISK (R2-0): secrets pasted into a chat
**What happened:**
- During earlier development a **Notion integration token** and the **owner API key** were pasted into a chat with an AI assistant.
- The owner key was also kept in a **chat app**.
- No values are recorded here, and none should ever be.

**Owner decision (2026-10-22):** do not rotate now (no new keys, no re-runs). The risk is **accepted, not fixed** (ADR-0018 R2-0).

**Where copies may persist:**
1. **The chat history** on the assistant's service (the conversation itself).
2. **The chat app** where the key was stored: its local database, its sync service and any device it syncs to.
3. **Claude Code transcripts on this disk:** `~/.claude/projects/*/*.jsonl`. These are plaintext JSON lines, readable by any process running as the owner. They are **not touched** without the owner's say-so (R2-0).
4. **Anything that indexes or backs up the above:** Spotlight indexes the home folder, and a Time Machine disk would copy it if one is ever configured.

**What an attacker with a copy could do:**
- **With the Notion token:** read every page shared with that integration, directly from Notion's API, from anywhere. RAGForge is not involved.
- **With the owner key:** **nothing from off the machine**, because the API is bound to `127.0.0.1`. A process **on this machine** could read and change all of the owner's notes and mint new keys (`approvals:write`).

**Why rotation, not deletion, is the fix:** rotation makes **every** copy worthless at once, wherever it is. Deleting transcripts only removes the copies you know about. The runbooks below take about ten minutes each.

**When to stop accepting the risk:**
- before the repo is pushed anywhere;
- before the API is exposed beyond loopback;
- if the chat app or the assistant account is compromised;
- if any page shared with the integration becomes sensitive.

---

## 6. Rotation runbooks
**Ground rules:**
- Secrets go in only through hidden input or straight into a 0600 file, and are **never** printed.
- Run commands in your own terminal (in Claude Code, prefix them with `!`). Never paste a value into a chat.
- The stack must be up (`make up`).

### 6.0 A safe way to call the API with a key file
These two helpers keep the key out of `ps` (no `-H "Authorization: …"` on the command line), out of shell history and off the screen. `printf` is a shell builtin, so the key never appears in a process's arguments. Paste them into your shell:

```bash
# Usage: rfapi <key-file> <METHOD> <path> [json-body]
rfapi() {
  local kf="$1" method="$2" route="$3"  # not `path`: zsh ties $path to $PATH
  local -a data=()
  if [ -n "${4:-}" ]; then data=(--data "$4"); fi
  curl -sS --fail-with-body -X "$method" "http://127.0.0.1:8000$route" \
    -H @<(printf 'Authorization: Bearer %s\n' "$(cat "$kf")") \
    -H 'Content-Type: application/json' "${data[@]}"
}
# The id of the key stored in a key file (prefix only, which is public):
rfprefix() { cut -d_ -f2 "$1"; }
```

### 6.1 Notion integration token
1. **Regenerate the secret in Notion.**
   - **Where:** notion.so → Settings → Connections → **Develop or manage integrations** → your RAGForge integration → **Internal Integration Secret** → **Refresh** (regenerate).
   - **Effect:** the old token stops working at once, wherever its copies are. Page access (which pages are shared with the integration) is unchanged.
   - **Don't copy the new secret anywhere yet.** You will paste it into a hidden prompt in step 3.
2. **Note the old connector's id:** `rfapi ~/.config/ragforge/<owner>.key GET /v1/connectors | python3 -m json.tool`. This shows ids, names and statuses, no secrets. After step 1 the old connector's syncs fail with 401 from Notion.
3. **Create the replacement connector: `! make connect-notion`.**
   - It asks for the RAGForge key and the new Notion token as **hidden input**, and lets you pick the same collection.
   - The worker starts the first sync within about 30 s (about 30 min per 1,000 pages).
4. **Delete the old connector** once the new one's first sync has finished (`make logs SERVICE=worker`, or `GET /v1/jobs/{id}`):
   - **Command:** `rfapi ~/.config/ragforge/<owner>.key DELETE /v1/connectors/<old-id>`.
   - **What it does:** destroys the old ciphertext immediately, and a job removes the old copy of the documents.
   - **Order:** create first, then delete, so the notes never disappear. Until the delete job finishes, some answers may cite duplicate pages.
   - **What survives:** exclusions are keyed by Notion page id and survive the swap (DESIGN §6.2).
5. **Verify:**
   - `GET /v1/connectors` shows one active Notion connector;
   - `make ask` answers with citations;
   - the audit log has `connector.create` and `connector.delete` (`GET /v1/audit`).

**Known gap:** there is no "replace token" endpoint, so a rotation re-syncs every page. A `PATCH /v1/connectors/{id}` that re-seals the token would avoid that; it is recorded in the SEC-THREAT handoff.

### 6.2 RAGForge API keys (owner, MCP, and others)
Repeat these steps for each key file under `~/.config/ragforge/` (`<owner>.key`, `mcp.key`, and the eval/demo keys under `~/.config/ragforge/eval/`).

1. **List the keys** (ids, names, prefixes and scopes; never secrets): `rfapi ~/.config/ragforge/<owner>.key GET /v1/api-keys | python3 -m json.tool`. Match each row to its file with `rfprefix <file>`.
2. **Mint the replacement with the owner key.** The secret goes straight into a new 0600 file and is never shown. The subshell keeps `umask 077` from leaking into your shell:
   ```bash
   ( umask 077
     rfapi ~/.config/ragforge/<owner>.key POST /v1/api-keys \
       '{"name": "user:owner-2026-10", "scopes": ["read", "write", "chat", "approvals:write"]}' \
       | python3 -c 'import json, sys; print(json.load(sys.stdin)["secret"])' \
       > ~/.config/ragforge/owner.key.new )
   ```
   If the call fails, the error body is not valid key output, so the new file stays empty and step 3's `test -s` stops you.
   - **Privileged scopes** (`approvals:write`) need a name that starts with `user:` or `ui:` (`services/auth/scopes.py`).
   - **For the MCP key,** use a name containing `mcp` (for example `mcp:claude-code-2026-10`) and the scopes `["read", "chat"]`. Add `"write"` only if you want `sync_now`.
   - **A key can mint only subsets of its own scopes.**
3. **Swap the files.** Check that the new file is non-empty (`test -s …`). Then move the old file aside (`mv <owner>.key <owner>.key.old`) and the new one into place (`mv owner.key.new <owner>.key`).
   - **Why the move matters:** with no `--key-file`, clients pick the single `*.key` in the directory and refuse with "several keys" when there are more (`resolve_key_file`). A stray `owner.key.new` or `.old` is harmless (wrong suffix). If `<owner>.key` and `mcp.key` sit side by side, every client must name its file with `--key-file` or `RAGFORGE_KEY_FILE` (set it in the MCP server's client config). Otherwise keep `mcp.key` in its own directory.
   - **For the MCP key:** restart the MCP server, or Claude Code's session.
4. **Revoke the old key:**
   - **Command:** `rfapi ~/.config/ragforge/<owner>.key DELETE /v1/api-keys/<old-id>`, using the **new** key.
   - **Effect:** idempotent; it answers 204.
   - **If the old owner key is the only key manager left,** revoke it last, with the new key.
5. **Verify:**
   - the old file now gets **401**: `rfapi ~/.config/ragforge/<owner>.key.old GET /v1/collections` fails;
   - the new one works;
   - the audit log shows `api_key.create` and `api_key.revoke`.
   
   Then delete `*.key.old`. It is dead once revoked, so a plain `rm` is enough (`rm -P` does nothing on APFS anyway).

**Eval keys have no rotation path today.**
- **Why:** the eval key is minted once, together with the paired eval tenant (`make eval-tenant`, `POST /v1/eval/tenant`, which answers 409 once the tenant exists). `ragforge-admin create-key` cannot grant the `eval` scope.
- **To kill a leaked eval key,** revoke it as the database superuser. The API path does not work: `DELETE /v1/api-keys/{id}` needs the key's UUID, and nothing hands it over. The eval key is not a key manager (no `GET /v1/api-keys` or `GET /v1/audit`). The owner key cannot see eval-tenant rows (RLS). `ragforge-eval2 tenant create` saves only the secret. The prefix, which is public, is enough for the superuser, who bypasses RLS:
  1. Get the prefix: `rfprefix ~/.config/ragforge/eval/<file>.key`.
  2. Run `make psql` (the superuser over the container's local socket), then:
     ```sql
     UPDATE rf.api_keys SET revoked_at = now()
      WHERE prefix = '<prefix>' AND revoked_at IS NULL
     RETURNING id, tenant_id, name;
     ```
     Expect exactly one row. Zero rows means a wrong prefix or a key that is already revoked.
  3. Confirm: `rfapi ~/.config/ragforge/eval/<file>.key GET /v1/collections` now gets **401**. Keys are not cached, so this takes effect on the next request.
  4. This bypasses the API, so no `api_key.revoke` audit row is written. Add a line to §5 instead.
- **To get a replacement:** a new path is needed. This gap is recorded in the SEC-THREAT handoff.

**The demo tenant key** (`~/.config/ragforge/eval/demo/owner.key`) is recreated by `make eval-demo` when the file is missing.

### 6.3 Gemini API key
1. **Create a new key:** Google AI Studio → **API keys** → **Create API key**. Don't paste it anywhere yet.
2. **Run `! make setup`.** Choose Gemini and paste the new key at the **hidden** prompt. `make setup` validates it with one call, then rewrites `.env` atomically with mode 600.
3. **Run `make up`.** Compose recreates `api` and `worker`, because their env changed.
4. **Delete the old key** in AI Studio (or in the Google Cloud console, under APIs & Services → Credentials). Then check that `make doctor` and `make ask` still work.
5. **If the eval judge uses a Gemini key file** (`--key-file`), replace that file too, with `umask 077`.

### 6.4 KEK (`RAGFORGE_KEK`)
Rotating the KEK **requires re-connecting Notion**. `EnvelopeSecretBox.rewrap` exists (`adapters/crypto/envelope.py`), but no command wires it up yet, so the stored connector tokens cannot be re-wrapped in place.

1. **Delete every connector:** `rfapi … DELETE /v1/connectors/<id>` for each one. This needs no decryption: it destroys the ciphertext and DEK.
   - **Side effect:** a delete also removes that connector's documents. **The notes are unavailable from here until the first sync after step 4 completes** (about 30 min per 1,000 pages), and `make ask` has nothing to cite meanwhile. Rotate when you can wait for that. Unlike §6.1, create-then-delete is not possible here: the new connector would be sealed under the old KEK.
2. **Replace the KEK.**
   - Remove the `RAGFORGE_KEK=` line from `.env` with an editor; **don't print the file**.
   - Run `make env`. `bootstrap.sh` adds a fresh KEK only for lines that are missing.
3. **Restart with `make up`.** Compose recreates `api` and `worker` with the new KEK.
   - **Side effect:** interview question tokens are keyed by the KEK, so a question asked before the rotation must be asked again.
4. **Re-connect Notion: `! make connect-notion`.** Use the same token, or a freshly regenerated one (§6.1).
5. **Verify:** a sync completes, and `make ask` answers.

### 6.5 DB role passwords
**App roles** (`rf_migrator`, `rf_api`, `rf_worker`, `rf_metrics`, `rf_backup`):
1. Remove their `RAGFORGE_DB_<ROLE>_PASSWORD=` lines from `.env` with an editor.
2. Run `make env` to generate new values.
3. Run `make up`.
   - The `db-bootstrap` one-shot re-runs `ALTER ROLE … PASSWORD` for each app role, SCRAM-hashed on the client side (`db/bootstrap.py`).
   - `api` and `worker` are recreated with the new passwords.
   - Existing connections keep working until they reconnect; a restart closes them.

**Superuser** (also step 3 of the lost-`.env` recovery in §3.9). Postgres reads `POSTGRES_PASSWORD` only when it first initialises the data directory, so changing `.env` alone does nothing, and `db-bootstrap` fails until the database matches `.env`:
1. Remove the `RAGFORGE_DB_SUPERUSER_PASSWORD=` line from `.env` with an editor, then run `make env` to generate a new value.
2. Start only the database: `docker compose up -d --wait postgres`.
3. Set the database to the `.env` value without it ever reaching the screen, shell history or a process argument list. `sed` reads it, the container's `sh` takes it from stdin into an env variable, and psql's `\getenv` reads it from there:
   ```sh
   sed -n 's/^RAGFORGE_DB_SUPERUSER_PASSWORD=//p' .env |
     docker compose exec -T postgres sh -c "IFS= read -r PGPW && export PGPW &&
       printf '%s\n' '\\getenv pw PGPW' \"ALTER ROLE postgres PASSWORD :'pw';\" |
       psql -v ON_ERROR_STOP=1 -q -U postgres -d ragforge"
   ```
   It prints nothing on success. Verified in bash and zsh against the project's pinned pgvector/pg17 image with a synthetic value: afterwards the new value authenticates over TCP and the old one is refused.
4. Run `make up`.

The value travels in plain text over the container's local socket and is hashed by the server (SCRAM). Unlike `\password`, it is not hashed on the client side. It is logged only if the statement fails, so check `docker compose logs postgres` after an error and treat those logs as secret.

**Grafana admin** (obs profile): change it in the Grafana UI (or with `grafana cli admin reset-admin-password` inside the container), then update `.env` to match.

### 6.6 After any rotation
- **Check the audit log:** `GET /v1/audit` should show only actions you expect.
- **Optional, with the owner's explicit say-so only:** remove old transcripts that contain the dead secrets. Once a secret is rotated, those copies are harmless.
- **Update §5:** add a line with the date and what was rotated. If everything was rotated, close the accepted risk.

---

## 7. Pre-first-commit checklist
The tree is not a git repo yet (CLAUDE.md: the owner does git at the end). Before the first `git add`:
1. **Scan the working tree:** `make secrets-scan` (gitleaks with `.gitleaks.toml`'s rules for `rf_…`, `ntn_…`, `secret_…` and `AIza…`). It must report zero findings outside the allowlists.
2. **Check what git would take:** run `git init`, then `git status --ignored --porcelain`. `.env`, `.env.*` (except `.env.example`), `*.key`, `*.pem`, `evals/private/`, `evals/*/runs/` and the eval label/cache files must all show as ignored (`!!`). Then run `git check-ignore -v .env evals/private` to see which rule matches.
3. **Check the image context:** `.dockerignore` must exclude `.env`, `evals` and `docs`, so no secret or private eval data reaches an image.
4. **After the first commit, scan the history:** `gitleaks git --redact -v` (run it as a container, the same way `make secrets-scan` does). The history must be clean before any `git push`. A secret that reached a commit **must be rotated** (§6), not just removed with a new commit.
5. **Before pushing, install the scan as a pre-push hook** (`make secrets-scan` is built to be reusable that way). Turn on the remote's own push protection, such as GitHub secret scanning.
6. **Re-read §5.** Pushing the repo is a trigger to stop accepting the R2-0 risk.

---

## 8. Reporting
This is a single-owner project. A finding goes to the owner directly. Never include a secret value in the report: name the file and line instead.

---

## 9. How this document stays true

| Claim | Test |
|---|---|
| Runtime egress is exactly Notion + Gemini (+ local Ollama); every URL literal is inventoried | `packages/core/tests/test_sec_threat_egress_inventory.py` |
| Destination-valued Settings (URLs, `*_host`, `*_addr`, ...) default to this machine; the eval judge is local and unproxied by default | same file |
| Unit tests cannot reach the network; a swallowed refusal still fails the test | `test_sec_threat_egress_guard.py` (plugin: `egress_guard.py`) |
| LangSmith tracing is off; nothing shipped (sources, scripts, Dockerfile, Makefile, compose) names a tracing variable or sets a proxy; no compose service, in any compose file, gets tracing or proxy env or mounts the Docker socket | `test_sec_threat_no_silent_egress.py` |
| FlashRank never downloads | same file |
| Key hashing, scopes, MCP/eval never `approvals:write` | `module_e2` auth tests; migration 0006 CHECK |
| RLS, definer functions, cross-tenant paths | `test_db_foundation.py`, per-function cross-tenant tests (ADR-0013) |
| No `env_file:`; the Gemini key reaches only api and worker | `module_g2/test_g2_compose_and_make.py` |
| Token shapes are redacted in logs | OBS-METRICS redaction tests (`observability/logging.py`) |
| No token shape in the tree | `make secrets-scan` (SEC-SCAN) |
