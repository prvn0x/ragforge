# ADR-0007: Notion sync: polling primary, webhooks optional, confirmed deletes, staleness reconcile, exclusions

**Status:** Accepted · 2026-10-16 · Revised in rounds 3 and 4

## Context
- Webhooks need a public HTTPS endpoint, and their payloads don't carry content.
- The rate limit is ~3 req/s.
- `last_edited_time` is minute-granular.
- Search isn't a guaranteed-exhaustive listing.
- The data-source model arrived in API version 2025-09-03.
- Capabilities can't be introspected.
- If page jobs die after the cursor has advanced, those pages are never retried.
- A deleted or removed document comes back on the next sync unless it's suppressed.

## Decision
- Pinned `Notion-Version: 2025-09-03`; database rows via data-source endpoints.
- **Polling:** `list_due_connectors()` every minute (default interval 5 min).
  - The cursor is inclusive with a 5-min overlap and advanced monotonically when listing completes.
  - Pages are skipped if excluded or **fresh**. Fresh means `last_edited_time < date_trunc('minute', documents.fetched_at)`. Because `last_edited_time` is minute-granular, an edit made in the fetch's own minute still counts as stale, so it's refetched, and an unchanged `page_sha256` skips re-chunking.
  - **`fetched_at` = the `Date` header of the first Notion response of that page's fetch, minus 60 s.** It's on Notion's clock (immune to our clock skew) and taken at fetch *start* (an edit during a long recursive fetch stays stale).
  - Document upserts apply only if `fetched_at` increases. Two concurrent `sync_page` runs for one page are safe in either commit order. An unchanged `page_sha256` still bumps `fetched_at` but skips re-chunking.
  - **Synced blocks:** pages that embed synced blocks record `synced_source_ids`. Reconcile checks those originals' `last_edited_time` and re-enqueues dependent pages, because editing an original doesn't update the embedding pages' timestamps.
- **Reconcile (nightly):**
  1. Full listing → re-enqueue every page that isn't fresh (same rule). This repairs dead jobs, missed polls and same-minute edits.
  2. Missing pages → confirm with `GET /pages/{id}`. 404, `in_trash`/archived or "not shared" → tombstone; any other error → retry tomorrow.
  3. Hard delete 7 days after the tombstone.
- **Exclusions:** `DELETE /documents/{id}` and `remove_from_index` write an authoritative `exclusions(tenant_id, source_kind, external_id)` row, which sync and reconcile skip. It's keyed by source kind, not connector, so it survives re-adding a connector, and file uploads (no connector) can be excluded too. `GET/DELETE /v1/exclusions` lets the user undo it.
- **Webhooks [A]:** verify → `route_notion_webhook()` → `sync_page` at priority `webhook`.
- **Global rate limit per connector:** `take_tokens(integration_key, n)` atomically refills (with `clock_timestamp()`) and decrements a `notion_rate` row keyed by `sha256(token)`, so the 3 req/s budget is shared across **all** workers. Per-process buckets would multiply it by the worker count. In-flight `sync_page` jobs are capped at 2 per connector via a locked `connector_inflight` counter (ADR-0010); `Retry-After` drains the bucket. Backfill runs at the lowest class, with a starvation guard (ADR-0010).
- "Read content only" is presented as setup guidance (it can't be verified).

## Consequences
- Freshness < 10 min.
- The index converges within a day even after dead jobs.
- No accidental deletes, and no zombie re-ingestion of removed pages.
