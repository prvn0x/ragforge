# OBS-STACK: Prometheus + Grafana (ADR-0018 revision 2: R2-4, R2-6 scrape side, R2-7)

## What OBS-STACK built (owned paths only)
- **`docker/observability/prometheus.yml`**
  - 15 s scrape and evaluation; `scrape_timeout: 14s` (review r2). The DB collector does two sequential reads, each bounded by a 5 s connect / checkout / statement timeout, so a hung Postgres can hold one `/metrics` answer for about 10 s. 14 s covers that and stays at or below the 15 s interval, so a hung DB shows as `RagforgeDbGaugesMissing` (right advice) and not as `RagforgeTargetDown` (wrong advice, plus gaps on every API panel). A test pins `timeout > 2 × db_collector.TIMEOUT_S` and `timeout <= interval`. See the "Scrape-time budget" seam note for the residual risk OBS-METRICS should close.
  - `ragforge-api`: a static target, `api:9101`.
  - `ragforge-worker`: `dns_sd_configs` on the A record `worker`, port 9102, so every replica is scraped.
  - `rule_files: [rules/*.yml]`. This is relative and not recursive, so `rules/tests/` is never loaded as rules.
  - Nothing else is scraped. The admin and lifecycle APIs are command-line flags, and the compose snippet sets neither.
- **`docker/observability/rules/ragforge.rules.yml`** holds 12 alerting rules for R2-4. Each has a `severity` label and a summary plus a description that says what to do.

  | Alert | Condition |
  |---|---|
  | `RagforgeChatLatencyHostedHigh` / `...LocalHigh` | More than 5% of owner (`surface="chat"`) turns are over the SLO edge. That edge is `le="8.0"` for every provider except ollama (the 5 s SLO is not met today, measured hosted p95 about 6.6 s, so the alert sits on the next bucket edge: ADR-0018 "Final review fixes"), and `le="12.0"` for ollama. Measured over 6 h, needs ≥ 3 slow turns AND ≥ 20 turns in that window, `for: 30m`. Because it counts the exact share above a bucket edge, it avoids histogram_quantile interpolation. |
  | `RagforgeHttp5xxHigh` | 5xx share over 2% in 6 h, with ≥ 3 failures AND ≥ 20 requests in that window, `for: 15m`. `disconnected` is excluded from the denominator, so client aborts can't dilute the errors. |
  | `RagforgeVerifiedShareLow` | Verified share under 50% over 6 h, owner surface only. Volume guard: ≥ 30 answered turns in that window, i.e. 5/h × 6 h. `or vector(0)` lets it fire when there is no `verified` series at all. |
  | `RagforgeLlmQuotaExhausted` | Any quota-exhausted call in 10 min (see the note below). |
  | `RagforgeLlmRateLimited` | More than 5 rate-limited calls in 10 min (see the note below). |
  | `RagforgeSyncStale` | Stalest connector over 1800 s, `for: 15m`. |
  | `RagforgeInteractiveQueueStuck` | `priority=~"interactive\|webhook"` oldest age over 120 s, `for: 5m`. Backfill is exempt. |
  | `RagforgeDeadJobsRising` | The dead gauge (deaths in the last 24 h across `rf.jobs` and `rf.jobs_history`, migration 0014; it used to read `rf.jobs` only and fell to 0 when the upkeep archived the job) is above its lowest value in the last 15 min (`max(x) > min_over_time(max(x)[15m:1m])`). A requeue followed by a new death inside the window still fires; a two-point "now vs 15 min ago" comparison missed it (review r1). |
  | `RagforgeTargetDown` | `up == 0` for 2 min. |
  | `RagforgeDbGaugesMissing` | `absent(ragforge_jobs_dead) and on () (up{job="ragforge-api"} == 1)` for 10 min. The scrape-time DB read fails soft (gauges omitted, `/metrics` still 200, `up` stays 1), which would leave the stale-sync, queue-age and dead-job alerts blind with nothing firing. The queue gauges are emitted on every good read (zero-filled), so their absence on a healthy scrape means the read failed. Quiet when the API itself is down (TargetDown covers that). |
  | `RagforgeWorkerTargetsMissing` | `absent(up{job="ragforge-worker"})` for 5 min. With DNS discovery a stopped worker *disappears* rather than reading 0, so `up == 0` alone would never fire for it. |

  **Note on the quota and rate-limit alerts.** Both also count a labelled counter that is *born* inside the window: `x unless x offset 10m`. increase() never sees that first step. They only trust it for targets Prometheus already scraped 10 min ago (`and on (job, instance) (up offset 10m)`). Without that, a fresh `make obs-up` would raise yesterday's count as a new event.
- **`docker/observability/rules/tests/ragforge.rules.test.yml`** holds the promtool unit tests: 39 cases with 62 checks, and every alert has both a firing case and a quiet case.
  - **Every R2-4 number is pinned from both sides** (review r2): hosted latency 4% slow quiet / 7% firing at 100 turns/min; 5xx 1.9% quiet / 2.5% firing at 1000 req/min; verified share 52% quiet / 48% firing at 50 answered turns; rate limits exactly 5 quiet / 6 firing. The counters start at 0 inside the window, so increase() is exact (no extrapolation) and the boundary is the real one.
  - **Every `for:` grace has a check inside its window**, shortly after the condition starts (TargetDown at +1 min, WorkerTargetsMissing at +3 min, DbGaugesMissing at +3 and +9 min, VerifiedShareLow at +4 min, plus the existing latency, 5xx, stale-sync and queue cases).
  - **Disappearing series are modelled with staleness markers** (`stale` in the input), as real Prometheus writes them: a worker dropped by DNS SD and a gauge left out of a successful scrape are absent at once, not after the 5 min lookback.
  - The quiet cases cover:
  - one user's 3 slow turns an hour;
  - 2 failed requests in a quiet hour;
  - a light day of unsupported answers;
  - eval traffic;
  - waking from sleep;
  - a backfill queue that is hours old;
  - an old, steady dead count, and a falling one (requeues);
  - DB gauges present, and an API that is down (not a missing-gauges alert);
  - a fresh Prometheus that finds an old quota count.
- **`docker/observability/grafana/provisioning/`**
  - `datasources/prometheus.yml`: uid `ragforge-prometheus`, proxy access, `manageAlerts: true`. This lists the Prometheus rules read-only.
  - `dashboards/ragforge.yml`: a file provider with `allowUiUpdates: false`, reading `/etc/grafana/dashboards`.
- **`docker/observability/grafana/dashboards/ragforge.json`** is one board, uid `ragforge-overview`.
  - It has the 4 rows of ADR-0018 §6, in order (Answers / Models / Retrieval & jobs / Freshness & health), and 30 panels under them, one of them an alert list.
  - It uses all 27 catalogue metrics plus `up`, and nothing else.
  - Variables: `surface` defaults to `chat`, so eval and demo traffic stay out of the owner's view unless selected. `window` (15m/1h/6h/24h, default 1h) exists because a 5 min rate is mostly empty at one user.
- **`docker/observability/compose.obs.snippet.yaml`** is the R2-7 compose services. Apply it with step 1 below.
- **Tests**, all in `packages/core/tests/`:

  | File | What it covers |
  |---|---|
  | `test_obs_stack_support.py` | Helpers only, no tests. A strict PromQL selector reader, plus a promtool runner that streams files into the container as a tar on stdin, with no bind mount and `--network none`. |
  | `test_obs_stack_catalog.py` | Every metric in every rule and dashboard query is in `catalog.CATALOG`. Each one is checked for: labels the metric really has (plus job/instance, and `le` on `_bucket`); `le` values that are real bucket edges; enum values (each surface the dashboard dropdown can send, outcome/status per metric, provider, priority, job names); grouping labels that exist. The dashboard must also cover every catalogue metric. It also checks that functions fit the metric type: rate/irate/increase/resets only over counters or a histogram's `_bucket`/`_count`/`_sum`, delta/idelta/deriv/predict_linear only over gauges, and histogram_quantile only over `_bucket` (the reader records the calls enclosing each selector). The checker's rejections are unit-tested, one per kind of drift, plus accept cases so it has no false positives. |
  | `test_obs_stack_promtool.py` | Marked `integration`. Runs the pinned image's promtool for `check config` (1 rule file found), `check rules`, and `test rules`. Runs the real parser over every dashboard query, with variables substituted. Asserts each alert has a firing case and a quiet case. Runs **35 mutation checks** (the final review added the 3-event floor and the 20-event guard of the latency and 5xx alerts, each from both sides, and the hosted edge), each a plausible design mistake that the unit tests must catch (including every R2-4 threshold moved up and down, `> 5` → `>= 5`, and each `for:` dropped or shortened), plus a re-serialisation control. It skips cleanly without Docker. |
  | `test_obs_stack_config.py` | Snippet guarantees, the scrape config, provisioning, and dashboard structure (rows, ids, grid overlap, datasource uid, alert list). Has one `integration` test: `docker compose -f compose.yaml -f snippet --profile obs config`, read-only, with a fake env file. The real `.env` is never read. |

  These tests read compose.yaml's `prometheus`/`grafana` once they are merged, so they guard what actually runs. A drift test then requires the merged services to equal the snippet.

**Images.** The digests are index digests, checked against the registry on 2026-10-22 with `docker buildx imagetools inspect`; both include arm64.
- `prom/prometheus:v3.15.0@sha256:efd719c9…8753e`, released 2026-09-25. The tests run exactly this image.
- `grafana/grafana:13.2.3@sha256:b28bae15…f9e572`, released 2026-09-29. It is Alpine-based with busybox `wget`, which the healthcheck uses. I read that from the image config without pulling it.
- I chose releases at least 3 weeks old rather than the newest ones (Prometheus v3.15.1 came out yesterday).
- Only the Prometheus image was pulled into the daemon. Grafana was not pulled.
- The Grafana setting names added in review round 2 (`[plugins] public_key_retrieval_disabled`, `preinstall_disabled`, `[grafana_com] url`/`api_url`, `[grafana_net] url`, `[metrics] enabled`) were checked against `conf/defaults.ini` at the `v13.2.3` tag on GitHub, not against a running container.

## Integration steps (integrator)
1. **compose.yaml.**
   - Paste the two services from `docker/observability/compose.obs.snippet.yaml` under `services:`, after the `ollama` profile block, below a `# ---- profile: obs` comment. Copy them verbatim, comments included.
   - Add the volumes `prometheus-data:` and `grafana-data:` to the top-level `volumes:`.
   - Keep the snippet file. `test_compose_yaml_matches_the_snippet_once_merged` then fails on any drift between the two.
2. **scripts/bootstrap.sh** (the task text says `docker/bootstrap.sh`; the real one is `scripts/bootstrap.sh`). After the DB-role loop:
   ```bash
   put RAGFORGE_GRAFANA_ADMIN_PASSWORD "$(secret)"   # Grafana admin (make obs-up); applied on Grafana's first start only
   ```
   compose interpolates `${RAGFORGE_GRAFANA_ADMIN_PASSWORD:?run make env}` for every command, even without the obs profile. `make up` already runs `env` first, but on an existing checkout `make ps`, `make logs` and `make psql` would abort with "run make env" until then. Make them depend on `env` too (it never overwrites existing values, so this is safe and idempotent):
   ```make
   ps: env ## Show service status
   logs: env ## Tail logs (SERVICE=api to filter)
   psql: env ## psql into the database as the superuser
   ```
   (Only the prerequisite changes; keep the recipes.) If you would rather not, tell the owner to run `make env` once after this lands.

   On the password being a plain env var: it is visible to `docker inspect` through the Docker socket, exactly like the DB passwords today (documented in SECURITY.md), so it is consistent rather than new. If the owner wants it off the container's env, the alternative is a compose secret plus `GF_SECURITY_ADMIN_PASSWORD__FILE: /run/secrets/grafana_admin`; I did not do that, to keep one secret pattern across the stack.
3. **.env.example.** Add:
   ```
   # Grafana admin password for `make obs-up` (user: admin). Generated by `make env`.
   RAGFORGE_GRAFANA_ADMIN_PASSWORD=
   ```
4. **Makefile.** Add `obs-up obs-down` to `.PHONY`, then:
   ```make
   obs-up: env ## Prometheus + Grafana (opt-in, ~450 MB cap); Grafana on http://localhost:3000 (user admin)
   	@$(COMPOSE) --profile obs up -d --wait --wait-timeout 120 prometheus grafana
   	@echo; echo "Grafana  http://localhost:3000  (user admin; password = RAGFORGE_GRAFANA_ADMIN_PASSWORD in .env)"

   obs-down: ## Stop Prometheus + Grafana (keeps their data)
   	@$(COMPOSE) --profile obs stop prometheus grafana && $(COMPOSE) --profile obs rm -f prometheus grafana
   ```
   Also change `down` and `nuke` so the obs services stop with everything else (R2-7):
   ```make
   down: ## Stop the stack (keeps data)
   	@$(COMPOSE) --profile ollama --profile obs down

   nuke: ## Stop the stack AND delete all data volumes
   	@$(COMPOSE) --profile ollama --profile obs down --volumes --remove-orphans
   ```
5. **OBS-METRICS wiring (seam).** It must match what these files assume:
   - The api serves metrics on `0.0.0.0:9101` and the worker on `0.0.0.0:9102`, inside the containers, with **no** `ports:` entry.
   - The `priority` label carries the **`JobPriority` name** (`interactive`, `webhook`, `incremental`, `maintenance`, `backfill`), never the integer that `rf.queue_stats.priority` holds. If it emits the int, `RagforgeInteractiveQueueStuck` can never fire.
   - The `provider` label carries the `LLMProvider` value (`ollama` / `gemini`). The hosted-vs-local split is "`provider != "ollama"`" vs "`provider = "ollama"`".
   - Recommended: pre-initialise the bounded label combinations at import. Examples: `ragforge_chat_turns_total{surface, outcome}`, `ragforge_verification_total{surface="chat", status}`, `ragforge_http_requests_total` for the 5xx class. That way increase() also counts the very first event after a restart. Without it, the ratio alerts undercount by one event per new series, which errs quiet and never causes a false alert.
6. Run `uv run pytest packages/core/tests/test_obs_stack_*.py packages/core/tests/module_g2 -q`. G2's real-compose test fills every `${VAR}` it finds, so the new variable is covered. Then run `make check`.

## What the integrator must test live (`make up && make obs-up`)
- `docker stats --no-stream ragforge-prometheus-1 ragforge-grafana-1` stays under the 256m/192m caps. Measure and record it for R2-9.
- Both containers start **read-only**. If Grafana refuses to start, read `make logs SERVICE=grafana` for the path it cannot write. Log mode is console and `/tmp` is a tmpfs. Any other path needs a tmpfs, not `read_only: false`.
- `http://localhost:3000` shows a login, with no anonymous access and no sign-up link. `http://127.0.0.1:3000` redirects to `localhost` (enforce_domain). From another machine on the LAN, port 3000 is closed.
- `docker compose port prometheus 9090` prints nothing (not published).
- Prometheus targets: `docker compose exec prometheus wget -qO- http://127.0.0.1:9090/api/v1/targets`. It should show `ragforge-api` and one `ragforge-worker` target per replica, all `up`. Try `docker compose up -d --scale worker=2 worker` and confirm 2 worker targets.
- `wget -qO- http://127.0.0.1:9090/api/v1/rules` (from inside the container) lists 12 rules, all `health: ok`.
- `curl -s -X POST localhost:9090/-/quit` is impossible from the host, because the port is not published. Inside the container it must return 403/404 (lifecycle off).
- Grafana opens the **RAGForge** board in the **RAGForge** folder; the 4 rows render. After a few chats, check that the Answers panels fill. The **Alerts** panel lists the Prometheus rules. This panel's `datasource` option (`"Prometheus"`, the datasource name) is the one setting I could not verify without a running Grafana. If the panel is empty while the board's **All alert rules** link (Grafana's Alerting > Alert rules page, `/alerting/list`) shows the 12 rules, open the panel editor, pick the datasource, and export the working option value back into `ragforge.json` (and into `test_dashboard_lists_the_prometheus_alerts`). The link is the fallback alert UI either way.
- Stop the worker (`docker compose stop worker`). `RagforgeWorkerTargetsMissing` goes pending at once (DNS SD drops the target and Prometheus writes a staleness marker, so `absent()` is true within one SD refresh, 30 s) and fires after about 5 to 6 min (`for: 5m`). Then `docker compose start worker` and confirm it resolves. A plain `make up` that recreates the worker must NOT fire it.
- **Grafana makes no outbound requests, captured from container start.** A single netstat misses a request made once at startup, so start the capture first and Grafana second. The capture runs in the Docker host's (VM's) network namespace, where it sees the compose bridge, so it catches Grafana's very first packet:
  ```bash
  SUBNET=$(docker network inspect ragforge_default -f '{{(index .IPAM.Config 0).Subnet}}')
  docker run --rm --net host --cap-add NET_RAW --cap-add NET_ADMIN nicolaka/netshoot:v0.13 \
    tcpdump -ni any "src net $SUBNET and not dst net $SUBNET" &
  docker compose --profile obs up -d grafana
  ```
  (Use the network name `docker network ls` shows for the project; pin the netshoot tag the integrator trusts.) Leave it running for about 5 minutes and through one board load and one login. Nothing may come from Grafana's IP (`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ragforge-grafana-1`). Traffic from other services (for example the API to a hosted model) is expected and can be ignored. Also check that `make logs SERVICE=grafana` shows no `grafana.com` or `plugins/ci/keys` error that would mean a retry loop. The snippet already sets `GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED`, `GF_PLUGINS_PREINSTALL_DISABLED` and `GF_GRAFANA_COM_URL`/`_API_URL`/`GF_GRAFANA_NET_URL` = `http://127.0.0.1:9` (refused inside the container, so any leftover path fails closed).
- `curl -s localhost:3000/metrics` returns 404 (`GF_METRICS_ENABLED=false`; this endpoint is served before auth when enabled).
- Make the DB read fail and confirm `RagforgeDbGaugesMissing` fires: for example, `docker compose stop postgres` for about 11 minutes (the gauges are left out of the next successful scrape, Prometheus writes staleness markers at once, then `for: 10m`), with Grafana's Alerting > Alert rules page open (Prometheus has no host port; from a shell use `docker compose exec prometheus wget -qO- http://127.0.0.1:9090/api/v1/alerts`). Confirm `RagforgeTargetDown` does NOT fire meanwhile (the scrape must still succeed within the 14 s timeout). Then start Postgres and confirm it resolves.

## Seam notes (the locked catalogue is unchanged)
- **Catalogue gap 1: `ragforge_metrics_collect_errors_total{source}` is not in the catalogue.** It exists in `metrics.py` (OBS-METRICS) with both sources pre-created, but the locked catalogue has no entry, so the catalogue test forbids using it. `RagforgeDbGaugesMissing` covers the `queue_stats` read. A `connector_freshness`-only failure (for example, only that view's grant missing) is still undetected, because "no freshness series" cannot be told apart from "no connectors". Proposed for the next catalogue revision (agree with OBS-METRICS):
  ```python
  Metric("ragforge_metrics_collect_errors_total", "counter",
         "Scrape-time DB reads that failed (their gauges are absent from that scrape)",
         ("source",), source="db_collector (api)"),
  # with COLLECT_SOURCES = ("queue_stats", "connector_freshness") added to VALUES_BY_LABEL
  ```
  and then this rule in `ragforge.rules.yml`, plus a firing case (counter rising), a quiet case (flat at 0) and a mutation check:
  ```yaml
      - alert: RagforgeDbGaugeReadFailing
        expr: sum by (source) (increase(ragforge_metrics_collect_errors_total[10m])) > 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Scrape-time DB read {{ $labels.source }} is failing"
          description: >-
            The API's rf_metrics read of rf.{{ $labels.source }} has failed on every scrape for
            10 min, so the alerts fed by it are blind. Run make logs SERVICE=api and look for
            metrics_collect_failed.
  ```
  (Both sources are pre-created at 0, so increase() needs no newborn-series handling.)
- **Catalogue gap 2: the model and retrieval metrics have no `surface` label.** An eval run therefore moves the Models row and the retrieval/orphan panels even while `$surface=chat`. Every affected panel says so in its description, and a test enforces that. A later revision could add `surface` to `ragforge_llm_*`, `ragforge_embed_*` and `ragforge_retrieval_*` (the `prompt_context` ContextVar already carries it).
- **The catalogue locks allowed values but not which label they belong to,** and it has no values at all for `priority`, `provider`, `connector_kind`, `kind`, `process` or `pool`. The tests keep that mapping in one place (`VALUES_BY_LABEL` / `VALUES_BY_METRIC_LABEL` in `test_obs_stack_catalog.py`). `priority` uses the `JobPriority` names and `provider` uses the `LLMProvider` values, as in step 5. A future catalogue revision could add `Metric.values: dict[label, tuple]` and let the tests read it.
- **`RagforgeVerifiedShareLow` counts `unverified` turns in the denominator,** because R2-4 excludes only abstain, refusal and error. For a model shipped with the badge forced to "unverified" (DESIGN §2), the alert would fire whenever there are 30 or more turns in 6 h. That is a correct statement ("nothing is verified"), but it is not drift. If the owner wants it to mean drift only, change the denominator to `status=~"verified|partial|unsupported"`; the quiet tests need no change.
- **The latency alerts use `ragforge_chat_turn_duration_seconds`** (the turn to done, which is the DESIGN §2 "verified answer" SLO), not per-model-call latency. Per-call latency by prompt is on the Models row.

- **Scrape-time budget (seam with OBS-METRICS, review r2).** `scrape_timeout: 14s` covers two sequential 5 s-bounded reads. Two residual gaps remain in `db_collector.py`, which OBS-STACK does not own:
  - `pool_pre_ping` on an already-pooled connection to a Postgres that *hangs* (TCP open, no reply) has no client-side bound: `statement_timeout` is enforced by the server, and `connect_timeout` only covers new connections. One scrape could then block past 14 s, `up` drops to 0, and the wrong alert fires.
  - Proposed fix (pick one, agree with OBS-METRICS): one combined read (both SELECTs on one connection checkout, so the worst case is one connect + one statement), or one shared deadline across both reads; and libpq TCP keepalives / `tcp_user_timeout` in `connect_args` (for example `"keepalives": 1, "keepalives_idle": 5, "keepalives_interval": 2, "keepalives_count": 2, "tcp_user_timeout": 5000`) so a hung socket is bounded client-side. Once the worst case is ≤ 10 s, `scrape_timeout` can stay at 14 s with margin; the config test only requires it to exceed 2 × `TIMEOUT_S`.

## Policy note
The org git/Jira policy asks for a ticket ID before any git operation. This repo is not a git repository, and per CLAUDE.md no git was run and nothing was committed, so no ticket was needed.
