# RAGForge user guide

This guide takes you from a fresh checkout to asking questions about your own notes. It is for
people who want to use RAGForge for their own work: study, interview prep, or team docs.

Each step gives the command, what you should see, how to tell it worked, and what to do if it
did not. Commands run from the root of the checkout. Values in angle brackets, such as
`<COLLECTION_ID>`, are placeholders: replace them with your own.

The short path is the [README](../README.md) quickstart. This guide explains each of its steps and
everything after them.

## Contents
1. [What you need](#1-what-you-need)
2. [Install and start](#2-install-and-start)
3. [Create your tenant and keys](#3-create-your-tenant-and-keys)
4. [Add your notes](#4-add-your-notes)
5. [Ask](#5-ask)
6. [Reading the answer](#6-reading-the-answer)
7. [Interview mode](#7-interview-mode)
8. [Advanced: evaluate on your own notes](#8-advanced-evaluate-on-your-own-notes)
9. [Monitoring](#9-monitoring)
10. [Backups and your data](#10-backups-and-your-data)
11. [Troubleshooting](#11-troubleshooting)
12. [Uninstall](#12-uninstall)

---

## 1. What you need

**Tested platform:** macOS on Apple silicon with Rancher Desktop as the container runtime, and
Ollama installed on the Mac itself. Other setups are not tested. The differences the scripts
know about are listed under "Linux" below.

You need:
- A container runtime with Docker Compose v2 (Rancher Desktop or Docker Desktop).
- `python3` on your `PATH`. `make up` and `make setup` run [scripts/setup_provider.py](../scripts/setup_provider.py) with it.
- `openssl`. [scripts/bootstrap.sh](../scripts/bootstrap.sh) uses it to generate the database
  passwords and the encryption key.
- [uv](https://docs.astral.sh/uv/). `make tenant`, `make connect-notion`, `make ask`, `make ui`,
  `make mcp` and the eval commands all run through `uv run` (see the [Makefile](../Makefile)).
  The workspace needs Python 3.12 (`requires-python` in `pyproject.toml`); `uv python install 3.12`
  installs one if you have none.
- On macOS: Ollama on the Mac (for example `brew install ollama`, then `brew services start ollama`).
  Without it the stack falls back to a CPU-only Ollama container (see "Linux" below).

**One stack per machine.** The API listens on port 8000, the UI on 8501 and Grafana on 3000, all
on 127.0.0.1 (`compose.yaml`, `make ui`). These ports are fixed. If another RAGForge checkout is
running, stop it with `make down` there first.

### Run the preflight check
```bash
make doctor
```
[scripts/doctor.sh](../scripts/doctor.sh) prints one line per check: `✓` passed, `!` a warning,
`✗` a blocking problem. It checks:

| Check | Blocking when |
|---|---|
| RAM | never (below 16 GB it warns: "expect swapping; prefer a hosted LLM") |
| Free disk in the checkout | under 10 GB |
| Docker engine reachable, its VM memory and CPUs | engine unreachable, or VM under 4 GB |
| `docker compose` v2 | missing |
| Rancher Desktop Kubernetes | never (warns that it uses VM memory you do not need) |
| Port 8000 | in use by something other than the container runtime |
| `uv` | missing |
| `openssl` | missing |
| The chat model provider (`setup_provider.py --check`) | a bad provider setting or a rejected Gemini key |
| `python3` | missing |
| macOS: host Ollama on `localhost:11434`, whether containers can reach it through `host.docker.internal`, and `OLLAMA_MAX_LOADED_MODELS` | never (warnings only) |

**It worked if** the last line is `doctor: ready`. **If not**, the last line is
`doctor: blocking problems found`: fix each `✗` line (see [Troubleshooting](#11-troubleshooting))
and run it again.

Doctor only checks port 8000. Ports 8501 (UI) and 3000 (Grafana) are not checked.

### Linux
These differences come from the scripts, not from a tested Linux run:
- `make env` (run by `make up`) writes `RAGFORGE_OLLAMA_MODE=container` on Linux, and on a Mac
  where host Ollama did not answer at that moment ([scripts/bootstrap.sh](../scripts/bootstrap.sh)).
  The stack then starts the `ollama` compose profile, a container with no host port.
- `make doctor` prints `Linux: containerized Ollama (profile 'ollama'); add the NVIDIA override for GPU`.
  This repository does not ship such an override file, so the Ollama container runs on the CPU.
- `.env` keeps the first choice. If you install host Ollama on a Mac later, the stack stays on the
  container until you change `RAGFORGE_OLLAMA_MODE` and `RAGFORGE_OLLAMA_BASE_URL` in `.env`
  (host: `host` and `http://host.docker.internal:11434`; container: `container` and
  `http://ollama:11434`, both written by [scripts/bootstrap.sh](../scripts/bootstrap.sh)).

---

## 2. Install and start

### Install the Python tools (once)
```bash
make sync
```
This runs `uv sync --all-packages --all-extras`. The root `pyproject.toml` lists only the
contracts, core and client packages, so the UI (Streamlit) and the MCP server are installed by
this step, not by a plain `uv run`.

**It worked if** it ends without an error. **If not**, check that `uv` is installed and that
you have network access to download packages.

### Start the stack
```bash
make up
```
What happens, in order (from the [Makefile](../Makefile)):
1. `make env` creates `.env` with fresh random secrets, mode 600. It never overwrites a value
   that is already there. It also picks the local chat model from your RAM: `qwen3:8b` at 24 GB
   or more, else `llama3.2:3b`. The embedding model is `nomic-embed-text`.
2. **First run in a terminal only:** you are asked which chat model provider to use (next section).
3. Docker Compose builds the image and starts Postgres, the one-shot setup jobs, the API and the
   worker, and waits up to 300 seconds for them to be healthy.

**It worked if** the last lines are:
```
API      http://localhost:8000/docs
Ready    http://localhost:8000/readyz
```
`curl -s localhost:8000/readyz` then returns JSON with `"ready": true`. This route needs no key.

**If not**, run `make ps` to see which service is unhealthy and `make logs SERVICE=api` (or
`SERVICE=worker`) to see why. Ctrl-C stops `make up`.

### Choose the chat model provider
The first `make up` in a terminal asks:
```
Do you have an API key for a hosted model provider? [y/N]:
```
- **Answer N (the default): local Ollama.** Everything runs on your machine. Nothing leaves it.
- **Answer y: Google Gemini.** The only hosted provider offered today. Before it asks for the key,
  [scripts/setup_provider.py](../scripts/setup_provider.py) shows this notice:
  ```
  Before you continue: what Gemini means for your notes
    * Each question you ask, and the note snippets retrieved to answer it, are sent to Google.
    * Free tier: Google may use that data to improve its products (and humans may review it).
    * Paid tier (billing enabled on the key's project): Google does not use it for that.
    * Embeddings and the search index stay on this machine; only the chat model moves.
    Switch back any time with `make setup` -> local Ollama.
  ```
  It then asks `Send questions and note snippets to Google? [y/N]`. Answering N keeps you local.

If you choose Gemini:
1. Create a key at <https://aistudio.google.com/apikey>.
2. Enter it at the prompt `Gemini API key, input hidden`. Nothing is echoed. Do not put the key
   on a command line, in a chat, or in a file by hand.
3. The script checks the key against Google's model list. It sends the key only in a request
   header to `generativelanguage.googleapis.com` and refuses redirects. If Google rejects the
   key, `.env` is not changed.
4. Pick a model from the list. The suggested default is `gemini-3.5-flash-lite`. Flash models run
   with thinking off (`RAGFORGE_GEMINI_THINKING_BUDGET=0`). Pro models cannot turn thinking off,
   so the script saves `-1` for them, and answers are slower.

**It worked if** you see `Saved to ... (mode 600): provider=gemini, model=...` (or
`provider=ollama (local, nothing leaves).`). Only the `api` and `worker` containers receive the
key (`compose.yaml`). `make doctor` then reports the provider and checks the key without
printing it.

Things to know:
- **No terminal, no question.** In CI or a script, `make up` never asks and stays on local Ollama.
- **A failed first setup still starts.** If the key is rejected or Google cannot be reached,
  `make up` prints `setup: ... Starting on local Ollama for now; run make setup to retry.` and
  starts locally. Nothing is saved, so the next `make up` asks again.
- **Shell variables win.** An exported `RAGFORGE_LLM_PROVIDER` or `RAGFORGE_GEMINI_*` in your
  shell overrides `.env`, as in Docker Compose. `make setup` and `make doctor` name such variables
  (never their values). Unset them if a change seems to have no effect.

### Pull the models (always)
```bash
make models
```
Run this on both paths. Embeddings always run on local Ollama, even with Gemini.
[scripts/models.sh](../scripts/models.sh) pulls the embedding model and the local chat model
into the Ollama that `.env` selected. On the Gemini path the local chat model is pulled too and
not used.

**It worked if** you see `models: pulling <model> (host Ollama)` (or `(container Ollama)`) for
each model and no error. **If not**, check that Ollama is running (`make doctor`).

### Switch provider later
```bash
make setup
make up
```
`make setup` shows the same menu. Pressing Enter at the key prompt keeps the saved key, so you can
rotate a key by entering a new one. Answering N at the first question switches back to local: it
only sets `RAGFORGE_LLM_PROVIDER=ollama`, and the saved key stays in `.env`. To remove the key
completely, delete the `RAGFORGE_GEMINI_*` lines from `.env`. `make up` applies the change.

---

## 3. Create your tenant and keys

A tenant is your private space in RAGForge. Postgres row-level security keeps each tenant's data
apart ([ADR-0013](adr/0013-db-roles-and-tenancy-model.md)). You need one tenant and its owner key.

### Create the tenant
```bash
make tenant
```
Optional: `make tenant NAME=study OWNER="Ada"`. The defaults are `NAME=personal` and
`OWNER=$USER`. The stack must be up.

[scripts/create_tenant.py](../scripts/create_tenant.py) runs `ragforge-admin create-tenant` in
the `api` container and writes the owner key to `~/.config/ragforge/owner.key`. The folder is
mode 700, the file mode 600, and the key is never printed.

**It worked if** you see `tenant_id=<TENANT_ID>` and
`owner key written to .../.config/ragforge/owner.key (mode 600, not shown)`.

**If not:**
- `make tenant: already have a key at ...`: a `*.key` file already sits in `~/.config/ragforge/`.
  The script refuses on purpose, because `make ask` and `make ui` use the single key there and
  refuse to guess between two. Keep using the existing key, or move it away first.
- `ragforge-admin failed with exit code ... (is the stack up? make up)`: start the stack and retry.
  No key file is left behind on failure.

The owner key has the scopes `read`, `write`, `chat` and `approvals:write`
(`services/auth/scopes.py`). Treat it like a password.

### About curl and your key
The curl examples in this guide read the key from its file into a request header, so you never
type or paste it. The `printf` in them is a shell builtin. This pattern is meant for a
single-user machine. For daily use, prefer `make ask`, `make ui` or the Python SDK
(`ragforge_client.client.RagforgeClient`), which read the key file themselves.

### The MCP key (only if you use Claude Code)
```bash
make mcp-key
```
[scripts/create_mcp_key.py](../scripts/create_mcp_key.py) uses your owner key to create a key
named `mcp:claude-code` with the scopes `read`, `chat` and `write`, and writes it to
`~/.config/ragforge/mcp/mcp.key` (mode 600). It prints the key prefix and scopes, never the key.

**It worked if** you see `wrote .../mcp/mcp.key (prefix ..., scopes read, chat, write; secret not shown)`.
**If not:** `make mcp-key: ... already exists` means you made one before: keep using it.
`owner key: ... (create one with make tenant)` means the owner key is missing.

Because its name contains `mcp`, the API marks it an MCP key that can never hold
`approvals:write` or `admin`. An AI assistant using it cannot approve its own destructive actions
([ADR-0005](adr/0005-mcp-as-thin-api-client.md)).

### Where the key files live, and why
| File | Made by | Used by |
|---|---|---|
| `~/.config/ragforge/owner.key` | `make tenant` | `make ask`, `make ui`, `make connect-notion`, `make mcp-key`, `make eval-tenant`, eval snapshots |
| `~/.config/ragforge/mcp/mcp.key` | `make mcp-key` | `make mcp` (its default) and Claude Code |
| `~/.config/ragforge/eval/eval.key` | `make eval-tenant` | eval runs and the UI's Eval page ([section 8](#8-advanced-evaluate-on-your-own-notes)) |
| `~/.config/ragforge/eval/demo/` | `make eval-demo` | the public Acme demo evals only |

`make ask`, `make ui` and `make connect-notion` look for a key in this order: `--key-file`, then
`$RAGFORGE_KEY_FILE`, then the single `*.key` directly in `~/.config/ragforge/`
(`resolve_key_file` in [ask.py](../packages/client/src/ragforge_client/ask.py)). Subfolders do not
count. That is why the MCP key and the eval keys have their own folders: a second key next to
`owner.key` would stop those commands: they refuse to guess.

---

## 4. Add your notes

### Connect Notion
**Step 1: create an internal integration in Notion.** In Notion, go to Settings, Connections,
then "Develop or manage integrations" (the path [SECURITY.md](SECURITY.md) uses), and create an
internal integration for your workspace. Give it **Read content** only: RAGForge never writes to
Notion. Its Notion client sends only `GET` requests and Notion's read-only `POST /search` and
data source query calls
([adapters/connectors/notion/connector.py](../packages/core/src/ragforge_core/adapters/connectors/notion/connector.py)). Notion shows an "Internal Integration Secret". You will
enter it at a hidden prompt in step 3; do not paste it anywhere else.

**Step 2: share your pages with it.** The integration sees only pages shared with it. Open each
page you want RAGForge to read and add the integration from the page's connections menu. Notion
answers "not found" both for a deleted page and for a page that is not shared, and does not say
which (`NotionNotFound` in `adapters/connectors/notion/errors.py`).

**Step 3: create the connector.**
```bash
make connect-notion
```
[scripts/connect_notion.py](../scripts/connect_notion.py) reads your RAGForge key from its file
(the rule above) and then:
1. picks the collection the pages go into. A new tenant has none (`make tenant` creates none), so
   the script prints `No collections yet: creating one for the Notion pages.`, asks
   `Collection name [notes]:` and prints `Created collection notes (<COLLECTION_ID>).` A name is
   lowercase letters, digits, `-` and `_`, starting with a letter or digit, up to 100 characters
   (`CollectionCreate` in `packages/contracts`). If you already have collections, it lists them as
   `  1. notes  (<COLLECTION_ID>)` and asks `Target collection [1-N]:` instead. Note the
   `<COLLECTION_ID>`: the API examples and the eval snapshot use it;
2. asks `Display name [Notion]:`;
3. asks `Notion integration token (hidden):`. Enter the secret from step 1. Nothing is echoed.

The token is sealed with the server's encryption key before it is stored, and is never echoed or
logged (`apps/api/routes/sources.py`).

**It worked if** you see:
```
Connector <CONNECTOR_ID> created (active).
The worker schedules its first sync within ~30 s. Watch: make logs SERVICE=worker
```
After that the worker polls Notion every 300 seconds (`RAGFORGE_SYNC_POLL_INTERVAL_S` in
`compose.yaml`).

**If not:** an HTTP error line names the route and status.

### Watch the first sync
Three ways, from most to least direct:
- **Worker logs:** `make logs SERVICE=worker`. The logs are JSON lines (`log_json` defaults to
  true in `config.py`). Each page written to the index logs a line whose `"event"` field is
  `indexed document <DOCUMENT_ID> outcome=<OUTCOME>`. To see only those:
  `make logs SERVICE=worker | grep 'indexed document'`. Ctrl-C stops following the log.
- **The UI:** `make ui`, then the **Sources** page. It shows each connector with a status badge
  (`active`, `paused`, `error` or `deleting`), its last error if any, and a **Sync now** button.
  A sync started with that button appears under its recent syncs with the job's status.
- **The API:** list the documents in a collection:
  ```bash
  curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
    "http://localhost:8000/v1/documents?collection_id=<COLLECTION_ID>"
  ```
  Each item has the page `title`, its Notion `external_url` and `fetched_at`. Other routes:
  `GET /v1/connectors` (status and `last_error`), `POST /v1/connectors/<CONNECTOR_ID>/sync`
  (start a sync; returns a `job_id`), `GET /v1/jobs/<JOB_ID>` and
  `GET /v1/jobs/<JOB_ID>/events` (a live event stream of that job). All are in
  `apps/api/routes/sources.py`, `content.py` and `jobs.py`, and in the API docs at
  <http://localhost:8000/docs>.

**It worked if** the documents list shows your pages. If a page is missing, check that it is
shared with the integration (step 2), then start a sync.

**If no documents appear within a minute or two,** the first sync may have failed, for example on
a mistyped Notion secret. That first job's id is not shown anywhere, and the connector keeps its
`active` badge: only a secret that cannot be decrypted sets the connector to `error`
(`services/sync/handlers.py`). Press **Sync now** on the Sources page and read that job's error
under **Recent syncs**. The worker log (`make logs SERVICE=worker`) also shows a JSON line whose `"event"` field has the
form `job <JOB_ID> (<KIND>) failed: NotionApiError`.

### Add files
You can also upload files to a collection. The route is in
[routes/sources.py](../packages/core/src/ragforge_core/apps/api/routes/sources.py):

- `POST /v1/collections/<COLLECTION_ID>/files`, a key with the `write` scope (the owner key has it);
- `multipart/form-data` with exactly one file part named `file`;
- accepted types: Markdown, plain text and `.docx`. The type is detected from the content, not
  the file name; the name only decides Markdown versus plain text. PDF and other binary files are
  refused;
- at most 50 MB (`max_upload_mb`, capped at 50 in `config.py`);
- one upload at a time per tenant.

To try it, upload a page of the public, synthetic Acme demo set into its own collection. A
separate collection does **not** keep it out of your answers by itself: the UI's "Search in" list
and `make ask` search every collection your key can see, so after this upload Acme pages can be
cited in answers to your own questions. Untick `acme-demo` under "Search in", or use
`make ask ARGS="--collection notes"`. A collection cannot be deleted today; a single document can
(`DELETE /v1/documents/<DOCUMENT_ID>`, in `routes/content.py`). Create the collection first, and
note the `id` in the reply:
```bash
curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
  -H 'Content-Type: application/json' -d '{"name": "acme-demo"}' \
  http://localhost:8000/v1/collections
```
Then upload the page, with that `id` as `<COLLECTION_ID>`:
```bash
curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
  -F 'file=@evals/demo/pages/kafka-basics.md' \
  http://localhost:8000/v1/collections/<COLLECTION_ID>/files
```
**It worked if** the reply is HTTP 202 with `{"job_id": "<JOB_ID>"}`. Follow the job with
`GET /v1/jobs/<JOB_ID>` (same header) until its `status` is `done`. **If not**, the reply is a
problem JSON: see [Troubleshooting](#11-troubleshooting) for the upload errors.

---

## 5. Ask

### In the browser
```bash
make ui
```
This serves the Streamlit UI on <http://localhost:8501>, bound to 127.0.0.1 only. It reads your
key with the same rule as `make ask`; to use another file, run `make ui ARGS="--key-file <FILE>"`.

The sidebar has three pages: **Chat**, **Sources** and **Eval**. On Chat, pick the collections in
the sidebar and type a question. **It worked if** the answer streams in with a badge under it
([section 6](#6-reading-the-answer)). **If not**, the page shows an error box with a short reason,
for example `Cannot reach the RAGForge API` when the stack is down.

The UI refuses to load your key if it is reachable from other machines, or if another web page or
host name opened it ([app.py](../packages/ui/src/ragforge_ui/app.py)). Open
<http://localhost:8501> directly.

### In the terminal
```bash
make ask
```
`ragforge-ask` uses every collection the key sees (up to 20). To pick some, pass names or ids:
`make ask ARGS="--collection notes"` (repeatable). `make ask ARGS=--help` lists the flags
(`--url`, `--key-file`, `--collection`).

**It worked if** you see a line `RAGForge · http://localhost:8000 · 1 collection(s) · /new, /quit, /interview [topic]`
and a `ragforge ›` prompt. Each answer ends with its citations and a footer of the form
`(<STATUS> · first token <SECONDS>s · total <SECONDS>s)`. `/new` starts a new conversation; `/quit` or Ctrl-D
exits.

### The REST API
Use this to script RAGForge. Read [About curl and your key](#about-curl-and-your-key) first.

Search only, no chat model (`POST /v1/query`, in
[routes/ask.py](../packages/core/src/ragforge_core/apps/api/routes/ask.py)):
```bash
curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does Acme keep all events of one shipment in order in Kafka?", "collection_ids": ["<COLLECTION_ID>"], "k": 5}' \
  http://localhost:8000/v1/query
```
The reply has `hits`, each with a `citation` (title, heading path, snippet, URL) and scores.

A full answer (`POST /v1/chat`) is a server-sent event stream, so pass `-N`:
```bash
curl -sS -N -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
  -H 'Content-Type: application/json' \
  -d '{"collection_ids": ["<COLLECTION_ID>"], "message": "How does Acme keep all events of one shipment in order in Kafka?"}' \
  http://localhost:8000/v1/chat
```
The events are defined in [sse.py](../packages/contracts/src/ragforge_contracts/sse.py). A
script must handle at least these:
- `event: token`: a piece of the answer text. Show it as plain text until `verification` arrives.
- `event: citation`: one source of the answer.
- `event: verification`: the verdict (`status`) and the flagged claims.
- `event: answer.replace`: the server replaced the answer (for example after an unsupported answer
  was regenerated, ADR-0011 rule 6). Drop all token text received so far and show this event's
  `text` instead. `make ask` does this (`packages/client/src/ragforge_client/ask.py`).
- `event: extra`: general knowledge that is not from your notes ([section 6](#extras-not-from-your-notes)).
- `event: done`: the end of a good turn, with the `conversation_id`.
- `event: error`: the turn failed, for example on a Gemini quota. It carries a `problem` and ends
  the turn; no `done` follows.

The list is not complete: there are also `tool_call` and `interrupt` (an action waiting for your
approval in the UI). A newer server may add event types; ignore the ones you do not know, as
`decode_sse_data` in `sse.py` does. For anything beyond a quick test, use `make ask` or the Python
SDK, which handle all of this.

### In Claude Code (MCP)
1. Create the MCP key once: `make mcp-key` ([section 3](#the-mcp-key-only-if-you-use-claude-code)).
2. Print the install command:
   ```bash
   make mcp-install
   ```
   It prints, and does not run, a `claude mcp add --scope user ragforge -- uv --directory <CHECKOUT> run ragforge-mcp --key-file .../mcp/mcp.key`
   line for this checkout. The command holds a file path, not the key.
3. Run the printed `claude mcp add ...` line yourself.

The server offers the tools `ask`, `search`, `list_sources` and `sync_now`, and a `document`
resource ([server.py](../packages/mcp/src/ragforge_mcp/server.py)). There is no approve tool:
when an action needs your approval, `ask` returns status `needs_approval` with the UI address
(`needs_approval.ui_url`). You cannot approve it from Claude Code. Open the UI, ask for the action
again there, and approve it there. Your next question in Claude Code cancels the pending action.
`make mcp` runs the same server by hand over stdio with the MCP key.

---

## 6. Reading the answer

RAGForge checks every answer against your notes before it calls it verified
([ADR-0011](adr/0011-streaming-verification-and-output-safety.md),
[ADR-0015](adr/0015-retrieve-then-answer.md)). An answer is `verified` only if the verifier calls
it supported **and** every substantive part of it is covered by a citation to a note you were
shown, or is an allow-listed phrase.

### The badge
The UI shows one badge per answer ([render.py](../packages/ui/src/ragforge_ui/render.py)). The
terminal prints the same status in the footer.

| Status | UI badge | What it means |
|---|---|---|
| `verified` | Verified (green) | "Every statement is backed by your notes." |
| `partial` | Partly verified (orange) | "Most of it is backed by your notes; the statements listed below are not." |
| `unsupported` | Not supported (red) | "Your notes don't back this answer. Don't rely on it." |
| `unverified` | Not checked (gray) | "The model in use is below the verifier's quality bar, so nothing was checked." |
| `unverified`, nothing cited, nothing flagged | Nothing to check (gray) | No answer came from your notes, for example a refusal. |

A status the UI does not know is shown in gray, never green.

### Citations
Each citation reads `[S1] Title › heading` and links to the place in the Notion page. The label is
the one the answer used. `S2*` with "matched by the server" means the server, not the model,
attached that source to an uncited sentence, after checking it.

### Flagged claims
When an answer is `partial` or `unsupported`, the statements that failed are listed in two groups
([interview_views.py](../packages/ui/src/ragforge_ui/interview_views.py), `draw_flags`):
- **"Not found in your notes"**, in red and always visible. The notes do not support these. Read
  them before you rely on the answer.
- **"Couldn't match exactly (code/format)"**, muted and collapsed. Code that was shortened or
  reformatted from your notes: every name in it is in the cited note, but lines around it may be
  missing. Check it.

The reason changes only how a claim is shown; it never changes the status
([ADR-0015](adr/0015-retrieve-then-answer.md)). The
terminal prints the same groups, the first in red with `!`, the second dim with `~`.

### Extras: "Not from your notes"
For many questions RAGForge adds one more block after the answer: a common production failure
pattern, or a code example when you asked for code ([ADR-0016](adr/0016-labelled-extras.md)).
This is the model's general knowledge. It is **never verified, cited or saved** with the
conversation. The UI shows it last, in its own box headed
"Not from your notes — general knowledge (not verified)". The terminal prints it after the footer,
every line prefixed `│ `, closed by `── end (general knowledge, not verified) ──`. It never runs
after a refusal or an answer that cites nothing (except an explicit code request). It is on by
default; `RAGFORGE_AGENT_EXTRAS=false` turns it off (`compose.yaml`).

---

## 7. Interview mode

Interview mode asks you one question at a time from your notes and grades your answer
([agent/interview](../packages/core/src/ragforge_core/agent/interview/__init__.py),
[interview.py](../packages/client/src/ragforge_client/interview.py)).

### Start it
- **UI:** turn on **Interview mode** in the sidebar, enter topics (optional; empty means mixed
  topics across all your notes), pick a level, and press **Start interview**. Or type
  "interview me on Kafka".
- **Terminal:** `/interview Kafka`, "interview me on Kafka", "quiz me on Redis", or a list:
  "interview me on Kafka, Redis" (topics then alternate).

### Levels
SDE-1, SDE-2, Senior SDE-1 (the default), Senior SDE-2 and Staff. The level is the bar for both
the questions and the grading, and it is fixed for a session: changing it starts a new session.
Set it in the UI's **Level** box, or in words: "interview me on Kafka at staff level", "as a
senior SDE 2", or `/interview kafka --level staff`. Within a level, **Harder** and **Easier** make
a fine step.

### During a session
Questions rotate through four kinds: concept, scenario, system design, and code / LLD. The
session commands are repeat, skip, next, harder, easier, next topic and stop. Type them, or use
the UI's buttons (Skip or Next question, Harder, Easier, Next topic, Stop).
A message that looks like a normal question ("What is ISR?") pauses the interview and goes to
normal Q&A. If that was really your answer, the UI offers "That was my answer — grade it", and
the terminal takes `/answer`. `/resume` returns to the question; `/stop` ends with a summary.

### How grading works (notes mode)
For each answer the server makes one grading call, then two checks against your notes at the same
time ([grade.py](../packages/core/src/ragforge_core/agent/interview/grade.py)):
- The model answer, the points you missed and the corrections to misconceptions claim to state
  what your notes say, so they are verified like a chat answer. An item that fails is dropped,
  never shown.
- The score (0 to 10), the one-line summary and "what you got right" judge your answer. Nothing
  in your notes can confirm them, so they are the model's judgement.

A score of 6 or below marks the topic as weak in the session summary, with the pages to revisit.

### General-knowledge mode (opt-in, never verified)
When your notes have nothing on a topic, or every matching note was already asked, the server
answers so, and the client offers one button per topic:
`Interview me on "<topic>" from general knowledge (not from your notes, not verified)`. In the
terminal, `/general [topic]` takes the offer. Only that explicit choice starts it; the server
never switches by itself, and a mixed-topic session is never offered it.

In this mode the questions and grades come from the model's own knowledge. They are labelled
"General knowledge · not verified · graded by the model against its own answer key", have no
citations, and are never checked against anything. Their scores are kept apart from notes scores
in the summary. The MCP server does not offer this mode
([ADR-0016](adr/0016-labelled-extras.md), amendment of 2026-10-19).

Interview calls have a per-user rate limit (`interview_rate_per_minute`, default 10, in
`config.py`); over it the API answers HTTP 429 with `Retry-After`.

---

## 8. Advanced: evaluate on your own notes

The eval layer measures retrieval and answers on a frozen copy of one of your collections
([ADR-0017](adr/0017-eval-layer.md)). Everything here runs against your live stack and is
optional. Results tables and what they mean are in [RESULTS.md](RESULTS.md).

Your questions and results stay on your machine: `evals/private/` and `evals/*/runs/` are
git-ignored (`.gitignore`).

The sequence below uses flags from `uv run ragforge-eval2 <subcommand> --help`. `make eval`
passes `ARGS` to `ragforge-eval2`.

### Step 1: create the eval tenant (once)
```bash
make eval-tenant
```
Your owner key provisions a separate eval tenant and writes its key to
`~/.config/ragforge/eval/eval.key` (mode 600, never printed).
**You should see** `eval tenant <TENANT_ID>: eval key written to .../eval/eval.key`.
**If you see** `... already exists: the eval tenant was provisioned before`, it is already done.
Cost: one API call, no model call.

### Step 2: freeze a copy of your collection
Find the collection id in the `make connect-notion` output (`Created collection <NAME> (<COLLECTION_ID>).`
or its numbered list), or list your collections:
```bash
curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragforge/owner.key)") \
  http://localhost:8000/v1/collections
```
Then:
```bash
make eval ARGS="snapshot create --source-collection <COLLECTION_ID>"
```
This copies the collection's chunks and their stored embeddings into the eval tenant, and writes
`evals/private/runs/target.json`, which later runs use. It prints
`snapshot <SNAPSHOT_ID> job <JOB_ID>: content <DIGEST or pending>, index <DIGEST or pending>`.
The copy runs as a job: repeat
```bash
make eval ARGS="snapshot status"
```
until both digests are no longer `pending`. Cost: no model call; the copy reuses the stored
embeddings (ADR-0017).

### Step 3: write your questions
Create `evals/private/questions.toml`. The format is the one in
[evals/demo/questions.toml](../evals/demo/questions.toml) (schema in
`packages/client/src/ragforge_client/evals/schema.py`). An example with the Acme demo content. It
works only if step 2 snapshotted the `acme-demo` collection; for your Notion notes the relevance
keys are `notion:<PAGE_ID>` instead of `file:<name>`:
```toml
schema_version = 2
name = "private"

[[questions]]
id = "kafka-ordering"
split = "dev"
question = "How does Acme keep all events of one shipment in order in Kafka?"
pages = ["Kafka Basics at Acme Shipping"]
relevance = { "file:kafka-basics.md" = 2 }
facts = [
  { fact = "Records with the same key go to the same partition", match = ["same partition"] },
]

[[questions]]
id = "unanswerable-graphql"
split = "dev"
question = "How does Acme version its GraphQL schema?"
answerable = false
```
- `name`: keep it `"private"`. The eval CLI uses it to find the runs folder and the key; with
  another name, `baseline set` fails with `a baseline must live in evals/<NAME>/runs`.
- `id`: lowercase letters, digits and `-`, unique.
- `split`: `dev` for questions you look at while tuning.
- `pages`: page titles, used to grade citations.
- `relevance`: the pages that answer the question (2 = the main page, 1 = also relevant). They must
  be pages in the collection you snapshotted in step 2. The example uses `file:kafka-basics.md`
  because the Acme page was uploaded as a file ([section 4](#add-files)), as in
  `evals/demo/questions.toml`. A Notion page is keyed `notion:<PAGE_ID>`, for example
  `relevance = { "notion:<PAGE_ID>" = 2 }`. `<PAGE_ID>` is the Notion page id written as a lowercase, dashed UUID
  (`canonical_id` in `adapters/connectors/notion/ids.py`); the 32 hex characters at the end of the
  page's `external_url` (from the documents list) are that id without dashes. Uploaded files use
  `file:<filename>`.
- `facts`: what a correct answer must contain; `match` lists accepted phrasings.
- `answerable = false`: a question your notes do not cover; it has no `relevance` or `facts`.

### Step 4: run a retrieval eval
```bash
make eval ARGS="run retrieval --dataset private --arm A"
```
`--arm` names the configuration you are measuring. The run uses the eval key and the snapshot from
step 2. It prints the path of the run file it wrote under `evals/private/runs/`. Cost: one
`POST /v1/query` per question, which embeds the question on local Ollama and reranks locally; no
chat model call.

If a run stops part way, continue it with `--resume` ("continue the newest unfinished run").

### Step 5: set the baseline
```bash
make eval ARGS="baseline set evals/private/runs/<RUN>.jsonl --note 'first baseline'"
```
Use the path step 4 printed. The baseline must be a finished run in `evals/private/runs/`. It
prints `baseline chat/retrieval: <RUN>.jsonl`. No model call.

### Step 6: change something, run again, gate
After a change (for example to retrieval settings), run step 4 again with a new arm name
(`--arm B`), then:
```bash
make eval-gate ARGS="--surface chat --kind retrieval"
```
`--surface` is required. `--kind retrieval` gates only the retrieval run, without the verifier's
hard floor that a chat change needs. It compares the newest run with the baseline and prints
`gate chat: <OUTCOME>`. It exits 1 when the gate does not pass, and 2 when it cannot gate at all (for example no
baseline, or the several-arms case below; `CliError` in `evals/cli.py`). It reads local run files only. If you have finished runs of several arms besides the baseline
(say A again and B), it stops with `runs of several arms (...): choose one with --arm`; pick one:
`make eval-gate ARGS="--surface chat --kind retrieval --arm B"`.

### Answer quality and labelling
`make eval ARGS="run generator --dataset private --arm A"` asks each question through
`POST /v1/chat`. Each question costs at least two model calls on the server (the `--rpm` help text:
"chat: >= 2"), plus a judge call per answer if you pass `--judge-config`. On the Gemini free tier,
lower `--rpm` to stay inside your quota.

Judge-based and application answer metrics need your labels; the
deterministic grades are recorded without them. Label in the UI:
```bash
make ui ARGS="--eval-key-file ~/.config/ragforge/eval/eval.key"
```
then **Eval** in the sidebar and the **Label queue** view. The Eval page reads
`evals/private/runs` and uses the eval key only to show the notes a model was shown. You can also
label in the terminal with `make eval ARGS="label evals/private/runs/<RUN>.jsonl"`.

---

## 9. Monitoring

Prometheus and Grafana are optional and off by default ([ADR-0018](adr/0018-ops-hardening-security-observability.md)).
```bash
make obs-up
```
**It worked if** you see
`Grafana  http://localhost:3000  (user admin; password = RAGFORGE_GRAFANA_ADMIN_PASSWORD in .env)`.
Open <http://localhost:3000> and log in as `admin`. The password is the value of
`RAGFORGE_GRAFANA_ADMIN_PASSWORD` in `.env`; open `.env` in your editor to read it. Do not print
`.env` in a shared terminal or a screen share. On macOS you can copy it without showing it:
`grep '^RAGFORGE_GRAFANA_ADMIN_PASSWORD=' .env | cut -d= -f2- | tr -d '\n' | pbcopy`. Grafana
applies this password only the first time it creates its data.

**The board** ("RAGForge", in `docker/observability/grafana/dashboards/ragforge.json`) has four rows:
- **Answers:** time to first token and turn duration (p50/p95), turns by outcome, the
  verification mix and verified share, flagged claims by reason, streams in flight;
- **Models:** model call p95 by prompt, tokens per hour, failed model calls, calls by prompt,
  embedding p95 and failures, interview calls;
- **Retrieval & jobs:** retrieval stage p95, ready jobs and oldest queued job by priority, job
  outcomes and duration, dead jobs, expired leases, Notion API requests by status;
- **Freshness & health:** time since the last successful sync, HTTP 5xx ratio and requests by
  route, HTTP p95, the database pool, scrape targets, build info, and the alert list.

**The alerts** (`docker/observability/rules/ragforge.rules.yml`) show only in Grafana; nothing is
sent anywhere else:

| Alert | Fires when (its summary) |
|---|---|
| RagforgeChatLatencyHostedHigh | chat p95 above 8 s on a hosted provider |
| RagforgeChatLatencyLocalHigh | chat p95 above 12 s on a local provider |
| RagforgeVerifiedShareLow | under half of answered owner turns were verified (6 h) |
| RagforgeLlmQuotaExhausted | the provider's quota is exhausted |
| RagforgeLlmRateLimited | the provider rate-limits more than 5 times in 10 min |
| RagforgeHttp5xxHigh | API 5xx above 2% (6 h), for 15 min |
| RagforgeSyncStale | a sync is stale for over 30 min |
| RagforgeInteractiveQueueStuck | jobs waiting over 2 min |
| RagforgeDeadJobsRising | new dead jobs in the last 15 min |
| RagforgeTargetDown | a metrics target is down |
| RagforgeDbGaugesMissing | the database gauges are missing, so queue, dead-job and sync alerts are blind |
| RagforgeWorkerTargetsMissing | no worker metrics target was discovered |

**Memory cost:** the two containers are capped at 256 MB (Prometheus) and 512 MB (Grafana),
768 MB in all (`compose.yaml`); the Makefile notes about 430 MB in typical use. Stop them with
`make obs-down` (their data is kept).

---

## 10. Backups and your data

**There is no database backup today.** Your indexed notes, conversations, connectors and API key
records live in Docker volumes only. The nightly dump that
[ADR-0014](adr/0014-data-classification-and-backups.md) describes is not built; see
[DECISIONS.md](DECISIONS.md) for what is and is not built. Notion stays the source of your notes:
after a loss you can rebuild the index by connecting Notion again. Uploaded files must be uploaded
again, and conversations are lost.

**Eval data is the exception.** `evals/private/` holds your questions, labels and runs. It is
git-ignored and cannot be rebuilt. Back it up with:
```bash
make backup-evals
```
It copies `evals/private` to a new dated folder under `~/.local/share/ragforge/eval-backups/`,
readable only by you, and prints `backed up evals/private -> <FOLDER>`.

**`make down` versus `make nuke`:**
- `make down` stops every container, including Ollama and the monitoring profile. Your data stays.
  `make up` brings it back.
- `make nuke` stops everything **and deletes every data volume**: the database (`pgdata`), uploaded
  files (`uploads`), the container Ollama's models (`ollama`), and Prometheus and Grafana data.
  **It cannot be undone.** Models in a host Ollama on the Mac are not touched. `.env` and the
  key files in `~/.config/ragforge/` are not deleted, but the keys no longer match anything: move
  them away before `make tenant` (which refuses while a key is there) and `make mcp-key`. If you
  used the eval layer, also move `~/.config/ragforge/eval/` and `evals/private/runs/target.json`
  aside before creating the eval tenant again.

---

## 11. Troubleshooting

Each entry quotes the exact message and the file that prints it.

**`✗ Docker engine not reachable (start Rancher Desktop / Docker Desktop)`**
([scripts/doctor.sh](../scripts/doctor.sh)). Start the container runtime and wait until it is
ready, then run `make doctor` again.

**`✗ Docker VM has N GB; give it at least 4 GB`** (doctor.sh). Raise the VM memory in Rancher
Desktop or Docker Desktop.

**`✗ Disk free N GB (< 10 GB)`** (doctor.sh). Free disk space on the volume that holds the checkout.

**`✗ Port 8000 is in use by <PROCESS>`** (doctor.sh). Something else listens on the API port.
Stop it, or stop the other RAGForge checkout with `make down`. A line
`Port 8000 in use by the container runtime (likely RAGForge itself)` is fine.

**`! Host Ollama not running: brew services start ollama (otherwise the CPU-only container is used)`**
(doctor.sh). Start Ollama on the Mac. If `.env` already says `RAGFORGE_OLLAMA_MODE=container`,
see [Linux](#linux) for how to switch.

**`! Containers cannot reach host Ollama via host.docker.internal (check Docker networking)`**
(doctor.sh). Rancher Desktop and Docker Desktop resolve `host.docker.internal` to the Mac. Do not
add an `extra_hosts: host-gateway` entry: it points at the VM instead and breaks host Ollama
(comment in `compose.yaml`).

**`! OLLAMA_MAX_LOADED_MODELS=1: chat + embedding models will evict each other`** (doctor.sh).
Set it to 2 or more with `launchctl setenv OLLAMA_MAX_LOADED_MODELS 2` (doctor reads it with
`launchctl getenv`), then restart Ollama: `brew services restart ollama`, or quit and reopen the
Ollama app.

**Ollama down while the stack runs.** A question fails with `Embedding provider failed` /
`embedding service unreachable` ([embedder_ollama.py](../packages/core/src/ragforge_core/adapters/llm/embedder_ollama.py))
or `Model provider error` / `model provider unreachable`
([chat_ollama.py](../packages/core/src/ragforge_core/adapters/llm/chat_ollama.py)). Start Ollama
and ask again.

**`embedding service error: model not found`** (embedder_ollama.py). The embedding model is not
pulled. Run `make models`.

**Gemini daily quota: `Model provider quota exhausted` / `model provider daily quota exhausted`**
([chat_gemini.py](../packages/core/src/ragforge_core/adapters/llm/chat_gemini.py)). The key's
daily allowance is spent, typically on the free tier. Retrying does not help. Wait for the daily
reset, move the key to a paid tier, or switch to local with `make setup` and `make up`.

**`Google rejected this API key (HTTP ...)`** ([setup_provider.py](../scripts/setup_provider.py)).
Create a new key and run `make setup` again. `.env` was not changed.

**`✗ LLM provider: gemini, but RAGFORGE_GEMINI_API_KEY is empty (run make setup)`** (from
setup_provider.py, shown by doctor). Run `make setup` and enter the key at the hidden prompt.

**`ragforge ask: no API key found: put one in .../.config/ragforge/<name>.key (mode 600) or pass --key-file`**
([ask.py](../packages/client/src/ragforge_client/ask.py)). You have no owner key yet. Run
`make tenant`; do not write a key file by hand. The UI shows the same text after `No usable API key.`

**`ragforge ask: several keys in .../.config/ragforge (a.key, b.key): choose one with --key-file`**
(ask.py). Keep one key in that folder (move the others to a subfolder), or pass
`ARGS="--key-file <FILE>"`.

**`ragforge ask: key file <FILE> is readable by others: run chmod 600 <FILE>`** (ask.py). Run the
`chmod` it names.

**`ragforge ask: this key sees no collections`** (ask.py). Run `make connect-notion`: it creates a
first collection when there is none ([section 4](#connect-notion)).

**A sync job fails with `NotionApiError: Notion API <STATUS> <CODE>: ...`.** A failed job's
`last_error` is the error class and its message (`_safe_error` in `services/sync/handlers.py`;
the `Notion API <STATUS> <CODE>: <MESSAGE>` form is built in
`adapters/connectors/notion/errors.py` from Notion's reply). You see it under **Recent syncs** on
the UI's Sources page (for a sync started there) or in `GET /v1/jobs/<JOB_ID>`. The status and
code are Notion's: for example, if Notion does not accept the token, create a new secret in Notion
and run `make connect-notion` again.

**A page is missing from answers and from the documents list.** Notion lists only pages shared
with the integration, and answers "not found" alike for a deleted page and an unshared one
(`adapters/connectors/notion/errors.py`). Share the page ([section 4](#connect-notion)), then
start a sync.

**UI: `Cannot reach the RAGForge API` / `Nothing answered at http://localhost:8000. Is the stack running (make up)?`**
([errors.py](../packages/ui/src/ragforge_ui/errors.py)). Start the stack with `make up`.

**UI: `The previous answer is still finishing`** (errors.py, HTTP 409). Wait until the answer
ends, then send again.

**Upload errors** ([routes/sources.py](../packages/core/src/ragforge_core/apps/api/routes/sources.py)):
- HTTP 415 `supported types: markdown, plain text, docx`: the file is not one of those types;
- HTTP 413 `file exceeds 52428800 bytes`: the file is over 50 MB;
- HTTP 422 ``expected one file part named `file` ``: use `-F 'file=@<PATH>'`;
- HTTP 503 `too many uploads in progress`: wait for your other upload to finish.

**Eval: `nothing to gate on chat: set a baseline first (baseline set)`**
([evals/cli.py](../packages/client/src/ragforge_client/evals/cli.py)). Run step 5 of
[section 8](#8-advanced-evaluate-on-your-own-notes) first.

**Eval: `that is the demo tenant's key: a private run needs the private eval key`** (evals/cli.py).
Private runs use `~/.config/ragforge/eval/eval.key`; do not pass the demo key.

**Eval: `--collection is for --dataset demo: a private run uses the snapshot`** (evals/cli.py).
Drop `--collection`; private runs read the snapshot from `target.json`.

---

## 12. Uninstall

1. Stop the stack: `make down`.
2. Delete its data: `make nuke`. **This deletes your index, uploads and conversations and cannot be
   undone** ([section 10](#10-backups-and-your-data)). Run `make backup-evals` first if you want to
   keep your eval data.
3. If you added RAGForge to Claude Code, remove it there: `claude mcp remove --scope user ragforge`.
4. Delete your keys: `rm -rf ~/.config/ragforge`. This removes the owner, MCP and eval keys.
5. Delete eval backups if you made any: `~/.local/share/ragforge/eval-backups/`.
6. Delete the checkout. Its `.env` holds the database passwords, the encryption key and your
   Gemini key if you saved one.
7. Optional: remove the image (`docker image rm ragforge/server:dev`), and the models from host
   Ollama (`ollama rm <MODEL>` for the `RAGFORGE_CHAT_MODEL` and `RAGFORGE_EMBED_MODEL` values that
   were in `.env`).
8. In Notion, delete the integration you created, or remove its access to your pages.
