# ADR-0018: Phase 4 (thin): a security pass and an observability dashboard

**Status:** Accepted, revision 2 · 2026-10-22 · Blind design review r1: 6.5 (4 P0); every finding is addressed in "Revision 2" below, which overrides the sections above where they differ · Amends the Phase 4 plan in DESIGN §0/§14 and NEXT.md

## Context
- **What RAGForge is:** the owner's personal project (daily interview prep over their own notes), and also a Senior/Staff portfolio piece. The machine is an M1 Pro with 16 GB (Rancher VM + host Ollama).
- **The original Phase 4** ("Advanced", DESIGN §0) planned Helm/kind, KEDA, NetworkPolicies, multi-arch images, a load test, a restore drill, dashboards and alerts as code, and CI.
- **The question:** what is worth building for a single-owner project that still has to read as production-minded?
- **Facts found while scoping:**
  - **No metrics exist today.** `/metrics` returns 404, and nothing imports a metrics library. DESIGN §12 lists metrics that were designed but never built.
  - **Secrets have been exposed outside the repo:** a Notion token and an owner API key were pasted into a chat, and the key was also kept in a chat app. The repo itself must be proven clean, and the rotation advice must be written down.

## Decision: scope (owner decisions, 2026-10-22)
| Item | Decision | Why |
|---|---|---|
| **Security pass** | **BUILD** | It holds the owner's notes, the Notion token and API keys, and it talks to Gemini. This helps daily use and the portfolio alike |
| **Observability: metrics + dashboard + alerts** | **BUILD** | It shows whether sync is stale, verified% is dropping, or something errors, at a glance |
| CI pipeline | parked until the repo is pushed to GitHub (~1–2 h then) | Locally the owner runs `make check` |
| Helm / kind / KEDA / NetworkPolicies / multi-arch | parked | Memory on 16 GB, and they matter mainly for platform/infra roles; the deployment readiness pieces (probes, migration job, stateless services) already exist |
| Full backup + restore drill | parked; `make backup-evals` instead (copies `evals/private/` to a dated 0600 folder) | Notes re-sync from Notion; the irreplaceable data is the local eval data (labels, runs, held-out set), which is git-ignored |
| Load test | parked | Single user; the write-up uses the measured latency pass (p95 ≈ 6.6 s on the demo) |
| DeepEval/RAGAS comparison | parked (ADR-0017 note) | — |

## Decision: observability
1. **Library:** `prometheus-client`, the de-facto standard and the lightest dependency. It is added to `ragforge-core` in the lock step.
2. **Exposure:**
   - The API serves `GET /metrics` on its own port. It is never public: it is bound in compose to the internal network, and Prometheus scrapes it there. The application router does not serve it, so it never appears under `/v1` or `/docs`.
   - The worker serves `/metrics` on its own internal port.
   - Neither needs auth inside the compose network; neither is published on the host.
3. **Stack:** an opt-in compose profile `obs` (`make obs-up` / `obs-down`) with Prometheus + Grafana. Grafana is provisioned from files: datasource, dashboard JSON, alert rules. Grafana is bound to 127.0.0.1 with an admin password from `.env` (hidden input, the existing setup flow). Expected memory is about 300 MB; it is off by default.
4. **Cardinality rules:**
   - labels are bounded enums only: route template, status class, provider, model, prompt name@version, verification status, job kind, priority, outcome;
   - **never** a tenant id, user id, question text, chunk id or key prefix as a label.
5. **The locked metric catalogue.** The names are the seam between the instrumentation builder and the dashboard builder:

   | Metric | Type | Labels | Source |
   |---|---|---|---|
   | `ragforge_http_requests_total` | counter | `route`, `method`, `status_class` | API middleware |
   | `ragforge_http_request_seconds` | histogram | `route`, `method` | API middleware |
   | `ragforge_chat_ttft_seconds` | histogram | — | chat SSE (first token) |
   | `ragforge_chat_total_seconds` | histogram | — | chat SSE (done) |
   | `ragforge_chat_turns_total` | counter | `outcome` (answered/abstained/refused/error/interrupted) | chat SSE |
   | `ragforge_verification_total` | counter | `status` (verified/partial/unsupported/unverified) | chat SSE verification event |
   | `ragforge_flagged_claims_total` | counter | `reason` (not_found/inexact_code) | chat SSE |
   | `ragforge_llm_call_seconds` | histogram | `provider`, `model`, `prompt` | every model call (chat, verifier, interview, extras) |
   | `ragforge_llm_tokens_total` | counter | `provider`, `model`, `direction` (prompt/completion) | same |
   | `ragforge_llm_errors_total` | counter | `provider`, `kind` (rate_limited/timeout/server/other) | same |
   | `ragforge_retrieval_seconds` | histogram | `stage` (embed/search/rerank) | retrieval |
   | `ragforge_retrieval_orphan_hits_total` | counter | — | retrieval (DESIGN §7.3) |
   | `ragforge_jobs_queue_depth` | gauge | `priority` | worker, sampled each loop |
   | `ragforge_jobs_oldest_age_seconds` | gauge | `priority` | worker |
   | `ragforge_jobs_total` | counter | `kind`, `outcome` (done/failed/dead/requeued) | worker |
   | `ragforge_sync_last_success_timestamp_seconds` | gauge | `connector_kind` | worker (per tenant's max, without a tenant label) |
   | `ragforge_interview_calls_total` | counter | `endpoint` (question/grade), `source` (notes/general), `outcome` | interview routes |
   | `ragforge_build_info` | gauge (=1) | `version`, `schema_version` | both |

6. **The dashboard (one board, 4 rows):**
   - **Answers:** TTFT/total p50/p95, turns by outcome, verification mix, flagged reasons;
   - **Models:** LLM latency by prompt, tokens per hour, errors by kind (429s visible);
   - **Retrieval & jobs:** retrieval stage latency, queue depth, oldest age, job outcomes, orphan hits;
   - **Freshness & health:** time since the last successful sync, HTTP 5xx rate, build info.
7. **Alert rules (Grafana, file-provisioned):**
   - Notion sync stale > 30 min;
   - HTTP 5xx > 2% for 10 min;
   - chat p95 total > 10 s for 15 min (the SLO in ADR-0012);
   - verified share < 50% over 1 h (a drift signal);
   - LLM rate-limit errors > 5 in 10 min;
   - a dead job appears;
   - queue oldest age > 15 min.

   Delivery is Grafana's own UI only (no external webhook: nothing leaves the machine).

## Decision: security pass
1. **Dependency audit:** `pip-audit` (run via `uvx`, no new runtime dependency) over the locked requirements. Findings are fixed by version bumps (the integrator re-locks) or documented with a justified ignore. Target: `make audit`.
2. **Image scan:** Trivy (run as a container, not installed) on the built API/worker images. HIGH/CRITICAL are fixed (base-image bump) or documented. It is part of `make audit`.
3. **Secret scan:**
   - **Tool:** gitleaks (run as a container) over the working tree, honouring `.gitignore`, plus a check that `.gitignore` covers `.env`, `*.key`, `evals/private/`, run files and the eval key directory.
   - **Scope:** a scan of the docs, handoffs, fixtures and the workflow scripts for any real token shape (Notion `ntn_`/`secret_`, `rf_<prefix>_`, Gemini `AIza`).
   - **Target:** `make secrets-scan`, reusable as a pre-push hook later.
4. **Egress audit:**
   - **Method:** enumerate every outbound network destination in the code (Notion API, the Gemini API, host Ollama, nothing else) and prove it with a test that fails when a new outbound host appears in the code without being listed.
   - **What it confirms:** the eval judge is local by default and not proxied; LangSmith tracing is OFF. `langsmith` is installed as a pytest plugin, so the audit must confirm nothing sends traces (`LANGCHAIN_TRACING_V2` unset, no API key) and pin that in a test.
5. **The threat model** (`docs/SECURITY.md`): assets (notes, keys, the Notion token, eval data), trust boundaries, and the top threats with mitigations already in place and residuals:
   - **tenant isolation:** RLS, definer functions, the eval tenant guard;
   - **keys:** hashed at rest, scoped; eval and MCP never `approvals:write`;
   - **prompt injection:** fencing, the verifier, literal extras;
   - **local UI exposure:** Host/Origin guards;
   - **egress;**
   - **secrets handling:** hidden input, 0600 files.

   It also says plainly what is out of scope: compromise of the host OS, of Docker, or of Gemini itself. It includes the **rotation runbook** for an exposed Notion token or API key.

## Build plan (after the design review)
**The lock step (integrator):** add `prometheus-client` to ragforge-core and re-lock; lock this metric catalogue; create the stub files.

**Five parallel builders:**

| Builder | Builds | Owns |
|---|---|---|
| **OBS-METRICS** | the instrumentation (API middleware, chat/verification/LLM/retrieval/interview metrics, worker queue/jobs/sync gauges, `/metrics` on the internal ports, `build_info`) + tests (names, labels, the cardinality rules, no PII in labels) | `packages/core/src/ragforge_core/observability/**` and additive hooks in apps/api, apps/worker, agent and adapters/llm |
| **OBS-STACK** | the `compose.obs.yaml` profile (Prometheus + Grafana, provisioning), dashboard JSON, alert rules, scrape config; a test that the dashboard queries only catalogue metrics | `docker/observability/**`, `compose.obs.yaml` |
| **SEC-SCAN** | pip-audit + Trivy runs and findings; fixes proposed as exact version bumps | `scripts/security/audit*`, `docs/handoff/SEC-SCAN.md` |
| **SEC-SECRETS** | the gitleaks config + run, the `.gitignore` coverage test, the token-shape scan, and the findings with their fixes | `scripts/security/secrets*`, `.gitleaks.toml`, tests |
| **SEC-THREAT** | `docs/SECURITY.md` (threat model + rotation runbook) + the egress audit test + the LangSmith-off test | `docs/SECURITY.md`, `packages/core/tests/test_security_egress.py` |

**The integrator:** applies the Makefile targets (`obs-up`/`obs-down`, `audit`, `secrets-scan`, `backup-evals`), the version bumps and the compose wiring of the internal metrics ports; runs `make check`, `make up` and `make obs-up` live (dashboard screenshot, alerts loaded); then a final blind review.

## Consequences
- **One new runtime dependency** (`prometheus-client`). There is a small per-request overhead (counters and histograms in-process).
- **The observability stack is opt-in,** so daily use is unchanged unless `make obs-up` is run.
- **Security tools run as containers or `uvx`**, so nothing new is installed on the host.
- **Parked items stay recorded** in this ADR and in NEXT.md, with the reasons.

## Revision 2: binding decisions from the design review (overrides the above)
**R2-0: the exposed secrets are an ACCEPTED RISK (owner decision).**
- The reviewer's P0 was to rotate the pasted Notion token and owner key now.
- The owner declined (2026-10-22: no new keys, no re-runs). This is recorded as an accepted risk, not fixed.
- SECURITY.md documents the exposure (the chat history, the chat app, and `~/.claude/projects/*.jsonl` on disk) and the exact rotation runbook, so it can be done any time.
- No transcript on disk is touched without the owner's say-so.

**R2-1: the catalogue lives in code.** It is `observability/catalog.py` (LOCKED): names, types, labels, allowed values, buckets and ports. It replaces the table above.
- **Naming:** Prometheus conventions (`_duration_seconds`; `_total` for counters only).
- **LLM metrics:** `ragforge_llm_requests_total{provider,model,prompt,outcome}`, so the error ratio is computable. The outcomes map to the adapters' own classes, including `quota_exhausted`.
- **Buckets:** locked, with the SLO edges at 5 s (hosted) and 12 s (local), per DESIGN §2.
- **Added:**
  - embed latency and errors;
  - Notion requests by status;
  - an in-flight chat streams gauge;
  - DB pool gauges;
  - a job duration histogram;
  - `surface` on the chat metrics, so eval and demo traffic never pollutes the owner's signals.
- **Dropped:** `method`. The DESIGN §12 rewrite/regenerate ratios and unsupported blocks are parked; tracing spans are parked.

**R2-2: DB-derived gauges are in ONE place.**
- A scrape-time collector in the API process connects as `rf_metrics` (the existing role, with its 5 s statement_timeout) and reads `rf.queue_stats` (aggregates, no tenant ids). This avoids a new DB holder in the worker's claim loop and replica double-counting.
- **Migration 0013** adds `rf.connector_freshness`: aggregates only, no tenant ids, granted to rf_metrics.
  - It gives, per connector kind, the age of the **stalest** enabled connector's last successful sync, which is `min(last_success)`.
  - A poll that changes nothing still counts as a success.
- The worker exports only what it alone knows: job outcomes and durations, and Notion request statuses.

**R2-3: the prompt label comes from a ContextVar plus a decorator.**
- `observability/prompt_context.py` (LOCKED) holds `prompt_name` and `surface` ContextVars.
- `InstrumentedChatModel` wraps the ChatModel port in bootstrap. It times calls, counts tokens from the usage reports and classifies outcomes.
- Every node and route that builds a prompt wraps its call in `for_prompt(name@version)`.
- A test asserts that no chat, verifier, interview or extras path records `prompt="unknown"`.

**R2-4: alerts are sound at one user.**
- They are **Prometheus** alerting rules, unit-tested with `promtool test rules`, plus `promtool check config`. Grafana only displays them.
- **Volume guard:** every ratio and latency alert needs at least 5 events per hour.
- **Latency:** p95 per provider, over 5 s hosted or 12 s local, for 30 min.
- **5xx:** above 2% for 15 min.
- **Verified share:** below 50% over 6 h. Answered turns only (abstain, refusal and error excluded), owner surface only.
- **Quota exhausted:** any quota-exhausted outcome in 10 min.
- **Rate limits:** more than 5 rate-limited in 10 min.
- **Stale sync:** over 30 min, `for: 15m`, so waking from sleep does not fire it instantly.
- **Interactive or webhook queue:** oldest age over 120 s, for 5 min. Backfill is exempt.
- **Dead jobs:** the dead gauge rises.
- **Scrape targets:** `up == 0`.
- **Honest limit:** delivery is the Grafana UI only, so nobody sees an alert unless the board is open.

**R2-5: HTTP middleware.**
- `route` is the matched template, read after the call; unrouted requests (404, and 413 before routing) use a fixed `unmatched`.
- The middleware is added LAST, so it is the outermost layer.
- SSE responses are excluded from the duration histogram.
- Client aborts get `status_class="disconnected"`.
- `/healthz` and `/readyz` are excluded.

**R2-6: process model.**
- The API is ONE uvicorn process, so no multiprocess mode. A test or guard enforces it: `workers > 1` would need `PROMETHEUS_MULTIPROC_DIR` on the tmpfs, because the root filesystem is read-only.
- Metrics are defined once at import (tests call `create_app` repeatedly).
- **Ports:** the API serves metrics on 9101 and the worker on 9102. `start_http_server(addr="0.0.0.0")` inside the container, with NO `ports:` entry; the port is configurable, and off in tests.
- **Worker scrape:** Prometheus uses `dns_sd_configs` (A record `worker`), so a scaled worker is scraped per replica.
- **Tests:** `GET :8000/metrics` returns 404 and is absent from the OpenAPI schema; no metrics or Prometheus port is published; Grafana is the only UI, on 127.0.0.1:3000.

**R2-7: the stack shape.**
- `profiles: [obs]` services in compose.yaml (shared network). `make obs-up` / `obs-down`; `make down` and `nuke` include the obs profile.
- **Image hardening:** images pinned by digest; `mem_limit` (Prometheus 256 MB, Grafana 512 MB: raised from 192 MB after the live measurement below); Prometheus retention 15 d or 512 MB; named volumes; no Prometheus admin or lifecycle API.
- **Grafana:** phone-home off (`GF_ANALYTICS_*`, news feed), anonymous access and sign-up off, `enforce_domain`, and an admin password generated by `bootstrap.sh` like the DB passwords.
- **Compose test:** the "no env_file" test is extended to the new services.

**R2-8: the security pass, tightened.**
- **Builders:** the SEC-SCAN and SEC-SECRETS builders merge into one.
- **Secret scan:** the tree is not a git repo, so `.gitleaks.toml` has explicit path allowlists plus custom rules for `ntn_`, `secret_`, `rf_<prefix>_` and `AIza[0-9A-Za-z_-]{35}`.
- **Trivy:** `--scanners vuln,secret,misconfig` on `docker save` tarballs (no Docker socket mount).
- **Ignore files:** `.gitignore` and `.dockerignore` gain `*.key`, `*.pem` and `.env.*`.
- **Logs:** a structlog redaction processor masks the token shapes, with a test. OBS-METRICS owns `observability/**`, so it owns this.
- **Egress audit:**
  - an inventory of Settings defaults plus URL literals;
  - a socket-level guard in unit tests (non-loopback connects are refused);
  - LangSmith tracing pinned off in conftest and absent from the compose env;
  - flashrank `allow_download=False` pinned.
- **SECURITY.md also covers:**
  - Docker-socket risk: `docker inspect` shows env secrets;
  - iCloud Desktop sync: does the repo leave the machine? Checked and documented;
  - KEK and DB-role password backup and rotation;
  - MCP indirect prompt injection into agentic tools;
  - Streamlit XSRF;
  - where `backup-evals` writes (`~/.local/share`, outside the repo and iCloud);
  - a pre-first-commit checklist (`gitleaks git` on history once git exists).

**R2-9: build plan (4 builders).** OBS-METRICS (instrumentation, migration 0013, the DB collector, redaction), OBS-STACK, SEC-SCAN (deps + images + secrets), SEC-THREAT (SECURITY.md + egress). The integrator applies the Makefile, compose and bootstrap changes from the handoffs and measures `make obs-up` memory live.

## Live measurement (2026-10-22, integrator)
- **Memory:** Prometheus about 100 MiB (cap 256m). Grafana about 190 MiB idle, but about 330 MiB about 3 min after start (the final review's measurement; the first reading here, about 280 MiB, was taken too early), so the r2 cap of 192m would OOM-kill it and the interim 384m left under 60 MiB of headroom. The cap is **512m** (final review fix). The obs profile is about 300 MiB typical, with a 768 MiB cap.
- **Exposure:** `:8000/metrics` is 404, and `:9101` is refused from the host. Both targets are up inside compose (api 160 series, worker 130). There are 12 alert rules, 0 firing; the Notion sync age was 157 s; schema 13.
- **Image:** the base is now Debian 13 (trixie) with pip removed.

## Final review fixes (2026-10-22, after the final blind review)
These override the sections above where they differ.

**P1-1: onnxruntime telemetry is pinned off.**
- **Finding:** the locked onnxruntime 1.31 (under FlashRank, in the API) links Microsoft's 1DS telemetry client with a built-in collector (`mobile.events.data.microsoft.com`) and starts it at import. This was checked in both the macOS wheel and the image's Linux aarch64 one.
- **Fix:** `ORT_DISABLE_TELEMETRY=1` in the image `ENV` and forced at the reranker adapter's import. In the container it stops the client from starting: with the variable set, the import logs nothing. The adapter also calls `onnxruntime.disable_telemetry_events()` before the first session, and skips it if a future onnxruntime drops the function.
- **Pinned by:** SECURITY.md §3.5 and `test_sec_threat_no_silent_egress.py`.

**P1-2: stdlib logging is redacted.**
- **Finding:** `log_config=None` did not turn uvicorn's access log off. flashrank's `logging.basicConfig(level=INFO)` then printed every uvicorn, httpx, LangGraph and SQLAlchemy record to stderr, unredacted.
- **Fix:** `configure_logging` installs one root handler: `structlog.stdlib.ProcessorFormatter` through the same redacting renderer. A later `basicConfig` is a no-op, and a raw handler attached earlier is replaced. The API passes `access_log=False` and keeps `uvicorn.error`. The token shapes gain URL-userinfo passwords and `Basic <base64>`.

**P1-3: `ragforge_jobs_dead` no longer resets.**
- **Finding:** the gauge read `rf.queue_stats` (rf.jobs only), and the upkeep archives dead jobs to `rf.jobs_history` after 5 min. The gauge, the panel and `RagforgeDeadJobsRising` all fell back to 0 with nothing fixed.
- **Fix:** migration 0014 adds `rf.dead_jobs_last_day()`. It is a SECURITY DEFINER function returning one number (deaths in the last 24 h across both tables), with EXECUTE for rf_metrics only (ADR-0013 table amended). The collector reads it in the same statement as the queue view, so `absent(ragforge_jobs_dead)` still means "the queue read failed". The metric name is unchanged; only its help text changed. `CODE_REQUIRED_SCHEMA` is 14.

**P1-4: ratio alerts no longer fire on one event (amends R2-4).**
- **The rule:** `RagforgeChatLatencyHostedHigh`, `RagforgeChatLatencyLocalHigh` and `RagforgeHttp5xxHigh` now need three things, all over one **6 h** window:
  - the ratio over budget;
  - at least **3** bad events;
  - at least **20** events in all.
- **Why 20:** below 20 events a 5% budget cannot be expressed (1 of 20 is already 5%).
- **Why 3:** one stray slow turn or failed request (a cold model, one provider hiccup) never fires on its own, while a real regression still does.
- **Why 6 h:** it is the window in which one owner's normal use reaches 20 events. The cost is that an alert clears up to 6 h after the fix.
- **Hosted latency threshold:** it moves from the 5 s SLO edge to the **8 s** bucket edge.
  - The measured hosted p95 on the demo is about 6.6 s, so **the DESIGN §2 hosted SLO of 5 s is NOT met today**. An alert on it would fire on every normal day: noise, not news.
  - 8 s is the next bucket edge above the measured baseline, so the alert now means "a regression past today's latency".
  - The 5 s SLO remains the target, and the board still shows p95 against it.
  - The local threshold stays at 12 s.
- **Pinned by** promtool cases, each guard from both sides:
  - 1 slow turn of 6 is quiet, and so are 1 of 20, 2 of 20 and 3 of 19;
  - 3 of 20 fires;
  - turns between 5 and 8 s are quiet;
  - the same cases for 5xx;
  - plus mutation tests on every guard.

**P2:**
- The Grafana cap is 512m (see "Live measurement").
- The queue-age gauge's help text now says that it includes jobs waiting out a retry backoff.
- OBS-METRICS gap 6 is corrected.


## Re-review and close (2026-10-22)
- **Blind re-review of the fix round: 8.5/10, ship.** All four P1s closed, no regressions; 103 targeted tests rerun (egress, logging, DB metrics on testcontainers, promtool).
- **Its two P2s, fixed by the integrator:**
  - the DSN-userinfo redaction now also masks a raw `/`, `?` or `#` in the password (two new cases; a mutation back to the old pattern fails 2 tests);
  - R2-7's "Grafana 192 MB" is updated to 512 MB.
- **Live after `make up` (migration 0014) and `make obs-up`:** readyz schema 14; both targets up; 12 rules loaded, none firing; `ragforge_jobs_dead` reported; Prometheus stores `le="8.0"` (the hosted alert edge matches real series); uvicorn lines come out as redacted JSON; no `telemetry_1ds` line and `ORT_DISABLE_TELEMETRY=1` in the api container; Grafana 302 MiB of 512; one live chat turn answered, cited and verified.
- **Open P2:** the MCP HTTP server (`packages/mcp/src/ragforge_mcp/server.py`) never calls `configure_logging`, so its uvicorn lines are not redacted. Passing `log_config=None` alone would drop them; the fix is `configure_logging` at MCP start plus `log_config=None`. No secret is logged there today (the bearer token is a header, not in the access line).
- **Phase 4 (thin) is CLOSED.**
