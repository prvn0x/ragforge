# ADR-0014: Data classification drives backup and recovery

**Status:** Accepted · 2026-10-16 · Revised in rounds 3 and 4

## Decision
| Class | Tables | Protection | Recovery |
|---|---|---|---|
| **Authoritative** | schema_meta, tenants, users, api_keys, connectors, collections, embedding_spaces, exclusions, **uploads** (upload → collection mapping), conversations, conversation_summaries, pending_approvals, LangGraph checkpoints/store, audit_log, usage_events, eval_runs | Nightly `pg_dump --enable-row-security` as the read-only `rf_backup` (RPO 24 h locally); PITR in cloud (RPO ≤ 5 min) | Restore, then `make reindex` |
| **Derived** | documents, chunks, embeddings, document_links, sync_cursors, jobs, jobs_history | Not backed up | `make reindex`: reset cursors and re-sync connectors; re-ingest every `uploads` row from the blob store; exclusions respected |
| **Uploads** | volume (local) / S3 (cloud) | Snapshot / versioned bucket | Restore, then reindex |

- **Restore order:** authoritative tables (`pg_restore` as `rf_migrator`) → uploads → reindex.
- Document and chunk IDs are deterministic (uuidv5 of tenant + source kind + external ID, and of document + ordinal), so links and audit targets survive a rebuild.
- `exclusions` is authoritative, so removed pages stay removed after a rebuild.
- **RTO:** < 1 h for authoritative data; derived data rebuilds at Notion's rate.
- **Retention:** jobs_history 30 d; tombstones 7 d; audit_log 1 y; usage_events 90 d; conversations until the user deletes them (which also deletes checkpoints and approvals).
- A CI check fails if a migration creates a table without a `-- class: authoritative|derived` comment.
- Restore drill: Advanced (M8).
