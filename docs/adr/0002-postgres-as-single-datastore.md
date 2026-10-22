# ADR-0002: Postgres (+pgvector) as the single datastore; expand/contract with monotonic schema versions

**Status:** Accepted · 2026-10-16 · Revised in rounds 3 and 4

## Context
We need relational metadata, vector search, keyword search, a durable work queue, LangGraph persistence and tenant isolation. Rollbacks must be safe, but Alembic revision IDs are hashes, so old code can't tell that an unknown revision is "newer".

## Options
| Option | Pros | Cons |
|---|---|---|
| **Postgres + pgvector + FTS + jobs table** | One transaction; RLS; one backup | Shared per-space HNSW degrades at a few million vectors; queue fine to hundreds of jobs/s |
| Postgres + Qdrant | Strong vector features | Dual writes, two consistency domains |
| Postgres + Elasticsearch | BM25 | Heavy, dual writes |
| Celery/Redis or SQS | Mature queues | Enqueue isn't atomic with the DB write |

## Decision
- Postgres 17 + pgvector ≥ 0.8 in schema `ext`.
- `tsvector` maintained by trigger (it uses the per-collection `fts_config`, which a generated column can't read); GIN index.
- Jobs per [ADR-0010](0010-job-scheduling-leases-priorities-fairness.md).
- **Schema compatibility:** a single-row `schema_meta(schema_version int, min_code_version int)`.
  - Every migration increments `schema_version`. Only *contract* migrations raise `min_code_version`.
  - The code declares `CODE_VERSION` and `CODE_REQUIRED_SCHEMA`.
  - `/readyz` passes iff `schema_version ≥ CODE_REQUIRED_SCHEMA` and `CODE_VERSION ≥ min_code_version`.
  - Expand migrations never break the previous release, so Helm rollbacks keep pods ready.
  - **Rule:** a contract migration may raise `min_code_version` only to a value ≤ the previous release's `CODE_VERSION`. Migrations run pre-upgrade while old pods still serve.

## Revisit when
- > 5M vectors in a space, or retrieval p95 > 400 ms after tuning.
- > 500 jobs/s sustained.
