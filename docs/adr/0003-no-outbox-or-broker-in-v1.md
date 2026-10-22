# ADR-0003: No outbox and no broker in v1. Audit in-transaction, progress via NOTIFY

**Status:** Accepted · 2026-10-16 · Revised in round 3 (supersedes "outbox with a Postgres dispatcher")

## Context
v0.1 proposed an outbox relaying to Redpanda. v0.3 kept the outbox with an in-process dispatcher and named three consumers: an audit-log projection, job-progress fanout, and sync-health metrics. Independent review showed:
- **Audit** can be inserted directly in the same transaction as the state change. A projection adds latency and a failure mode, and gives nothing in return.
- **Job progress** was dispatched inside *worker* processes, but SSE subscribers are connected to *API* pods, so the design was broken as written.
- **Sync-health metrics** can be computed from tables (`connectors`, `documents`, `queue_stats`).

So the outbox had no consumer that needed it.

## Pros and cons of an outbox and broker (for when the triggers fire)
- **Pros:** reliable fan-out to independent consumers, replay, decoupling across teams.
- **Cons:** relay process, at-least-once delivery (idempotent consumers), ordering per aggregate (bigserial isn't commit order), DLQs, schema evolution, lag monitoring, +1 GB RAM for Redpanda.

## Decision
- **Audit:** `audit_log` rows inserted in the same transaction as the change.
- **Job progress:**
  - The worker runs `NOTIFY rf_job_progress, '<job_id>'` in the same transaction as its progress update. NOTIFY is delivered on commit.
  - Each **API pod** keeps one dedicated (non-pooled) `LISTEN` connection, re-reads the job row for subscribed job IDs, and pushes SSE.
  - A 2 s poll fallback covers missed notifications and reconnects.
- **No `outbox_events` table and no `EventPublisher` port in v1.** YAGNI: the seam is added together with its first consumer.

## Triggers for Stretch S1 (outbox + Redpanda), any one of:
1. A consumer outside this deployable (analytics pipeline, another service, external webhook subscribers). Planned first one: a daily "docs changed" Slack digest plus an eval re-run trigger.
2. Multiple consumers needing independent progress or replay.
3. > ~500 events/s sustained.

## Consequences
- Fewer moving parts in Core.
- The interview story: "I designed it, measured the need, found no consumer, and here are the triggers. Adding it is one table, one relay and one adapter."
