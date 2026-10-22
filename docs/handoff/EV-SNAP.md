# Handoff EV-SNAP: the paired eval tenant and the frozen eval snapshot (ADR-0017 §1 snapshot row, r3 P0-C, r4 P1/P2)

Status: built, tested on the test Postgres with fakes (no network, no model, no live stack), green. I edited no locked seam, and none was insufficient for the server side. One SDK/CLI gap is listed under §3 (bootstrap has a route but no client method).

## 1. What I built (owned files)
- `packages/core/src/ragforge_core/db/migrations/versions/0012_eval_snapshot.py` (checksum-guarded like 0001-0011, ends with `bump_schema_version()`):
  - `tenants.kind` gains `'eval'`. New column `tenants.eval_of_tenant_id`: UNIQUE (one eval tenant per owner), FK `ON DELETE CASCADE` (deleting the owner deletes their note copy). It is CHECKed both ways (`kind = 'eval'` ⇔ paired) and is never self-paired.
  - `rf.eval_snapshots`: tenant table (RLS + backup policy), classified `authoritative`. It holds the synthetic upload, the source collection, the job that last changed it, counts, `excluded`/`purged_pages`, both digests, `space_id`, `ready_at`/`purged_at`, and CHECKs that tie ready ⇔ digests ⇔ space.
  - Definer functions (ADR-0013: `SET search_path = pg_catalog, rf, ext, pg_temp`, `REVOKE ALL FROM PUBLIC`, explicit grants, explicit tenant predicates):

    | Function | Grant | Rule |
    |---|---|---|
    | `provision_eval_tenant(owner)` → (tenant_id, user_id) | rf_api | source = `rf.current_tenant()`, which must be personal/team; a second one → 23505 on the pairing constraint |
    | `paired_eval_tenant()` | rf_api, rf_worker | the session's pair or NULL |
    | `begin_eval_snapshot(src, target, id, job)` | rf_api | r4 guard; requires a queued `eval_snapshot_copy` job **of the caller's own tenant**; creates the empty collection `eval-snapshot-<date>-<id8>`, the ONE synthetic `uploads` row (size 0, no blob) and the pending row |
    | `copy_collection_to_eval(src, target, id)` | rf_worker | r4 guard again (source = session tenant, target `kind='eval' AND eval_of_tenant_id = source`, collection visible to the source, otherwise P0002); idempotent on `id` |
    | `finish_eval_reembed(collection, space)` | rf_worker | refuses a partial flip (55000), flips the active space, drops unreferenced vectors, re-records the digests |
    | `_purge_eval_snapshot_pages()` | trigger only | AFTER INSERT ON `rf.exclusions` |
    | `_eval_copy_guard`, `_eval_snapshot_digests`, `_uuid5`, `_netstring`, `_eval_page_key`, `_eval_snapshot_lock_key` | none | invoker helpers called by the definers |

  - **The copy** is one fenced transaction with one statement per table: documents (only live ones, not excluded, one per page key), chunks, embedding rows in the source's ACTIVE space (copied, not re-embedded), embedding touches, `collection_space_stats`, and `document_links` (minus links to excluded pages).
    - Documents are **upload-owned**: `connector_id` is NULL and `upload_id` is the synthetic upload. `source_kind` and `external_id` (the Notion page id) and the metadata are preserved, so the page_key survives. The live connector's cascade can't reach the copy.
    - Ids are `domain.ids.document_id/chunk_id` (uuidv5 in SQL, pinned to Python by a test).
    - A chunk without a vector means the source is still indexing. The copy refuses with 55000; the job retries and nothing is written.
    - No ingestion or sync job is ever enqueued.
  - **The digests** are computed in SQL inside the same transaction (copy, purge and re-embed), so a stored digest always describes committed content:
    - *content*: the sorted page_key set plus display texts in ordinal order;
    - *index*: the active space id, plus each `chunk_key` (with occurrence) and its `embed_sha256`.
    - Fields are netstring-encoded, so separators can't collide. Ordering is `COLLATE "C"`.
  - **Privacy purge (r4 P2):** an exclusion inserted on the SOURCE tenant hard-deletes the matching pages from every snapshot of that owner's eval tenant, in the same transaction. Chunks and links cascade, orphan vectors and touches are dropped, stats are recounted, both digests are re-recorded and an audit row is written. Purged pages then resolve as `missing`, and the gate refuses to compare across the new content digest. The purge is not undone when the exclusion is later removed: a snapshot is frozen, and a refresh is a new snapshot.
  - **Concurrency:** a copy or re-embed takes a SHARED advisory xact lock on the eval tenant, and the purge an EXCLUSIVE one. So an exclusion racing a copy is either filtered out by the copy or purged right after the copy commits.
- `CODE_REQUIRED_SCHEMA = 12` (`packages/core/src/ragforge_core/__init__.py`; the task calls it CODE_REQUEST_SCHEMA). It is pinned by `test_db_foundation.py` and `test_api_health.py`.
- `packages/core/src/ragforge_core/services/eval_snapshot/`:
  - `policy`: job kinds, pydantic payloads, dedupe keys, bounds, and the eval key's scopes `{eval, chat, read}`, named `eval` so it is never an MCP key;
  - `digests`: the readable reference definition of both digests (pins the SQL);
  - `embed_input`: the ingest path's embed-input rule, re-hashed against the stored `embed_sha256` before any embed call;
  - `errors`: SQLSTATE → domain error. A 23505 is a refusal ONLY on `tenants_eval_of_tenant_id_key`; any other unique violation surfaces as a bug, not a 409;
  - `ports`, and `jobs` (the copy handler, and the 3-step re-embed: a short tx, the embed with no tx open and a heartbeat per batch, then a fenced tx).
- `packages/core/src/ragforge_core/apps/api/routes/eval_snapshots.py` (implements the stub; already in `ROUTERS`). Every route 404s before authentication unless `RAGFORGE_EVAL_API` is on (it reuses `deps.EvalApiEnabled`).
  - `POST /v1/eval/tenant` (admin key of the source): the bootstrap. In one transaction it creates the eval tenant, its user and its key (eval+chat+read). It returns `TenantProvisioned` with `cache-control: no-store`, and a second call gets 409.
  - `POST /v1/eval/snapshots` (admin key of the SOURCE): enqueues the copy in the source tenant and calls `begin_eval_snapshot`, all or nothing → 202 `EvalSnapshot` (digests None). Progress is on the existing `GET /v1/jobs/{id}`.
  - `GET /v1/eval/snapshots/{id}`: the eval key, or the source's admin key acting on its own pair only. Anyone else gets 404, and a key with neither scope gets 403.
  - `POST /v1/eval/snapshots/{id}/reembed {space_id}`: 404 for an unknown space, 422 for a retired one, 409 while the copy is pending.
- Worker registration (ADDITIVE) in `apps/worker/main.py`: `register_eval_snapshot_handlers(handlers, eval_snapshot_deps(container))` right after `register_handlers`, plus `SqlEvalSnapshotStore`, the Postgres side of the port.
- Tests in `packages/core/tests/module_eval_snap/` (62): `test_ev_snap_copy.py` (definers as the real login roles), `test_ev_snap_api.py` (routes + the job handlers against the test Postgres), `test_ev_snap_sql_reference.py` (hypothesis: SQL digests == Python reference, SQL uuid5 == `uuid.uuid5`), and `test_ev_snap_units.py` (pure rules and the handlers with fake ports). `ev_snap_support.py` and `conftest.py` are helpers.

Every test the task required is present:
- copy into a non-eval tenant, into another owner's eval tenant, or from a collection the caller can't see is refused;
- no ingestion job is enqueued;
- the embeddings are identical;
- the digests are stable across two copies, tenants and installs, and move on a text edit;
- deleting or reconnecting the live connector leaves the snapshot;
- the owner's keys never list the eval tenant's collections or documents;
- exclusions are honoured at copy time, and a later exclusion purges the page (that owner only).

Added as well:
- only rf_worker can execute the copy and re-embed, and only rf_api can execute begin and provision;
- the copy is idempotent;
- a source still indexing is refused;
- the pairing CHECKs hold;
- the re-embed feeds exactly the ingest texts, then flips back to the old space with zero embed calls;
- a drifted input is refused before any embed;
- rf_backup reads the whole snapshot (ADR-0014).

## 2. Integration steps
1. Nothing to wire: the router is already in `routes/__init__.py`, the handlers are registered in `_main`, and migration 0012 applies with the rest. `make nuke && make up` in dev, since the schema version moves to 12.
2. **ADR-0013 docs (I can't edit docs/):** add the six functions from the table above to its definer table.
   - "GRANT EXECUTE to the admin role only" is implemented as follows. Postgres has no admin role (admin is an API scope). Only an `admin`-scope key can reach `begin` (rf_api), and only the job role can run the copy (rf_worker). The copy refuses unless a pending row created by `begin` exists for that exact source collection and snapshot id.
3. **Optional cleanups:**
   - Move `SqlEvalSnapshotStore` unchanged to `adapters/db/eval_snapshot.py`. I own no adapters path; it imports only sqlalchemy/domain types, so the move is mechanical.
   - Extract a public `rag.chunker.embed_input(title, heading_path, display_text) -> str` from `Chunker._finish`. Services can't import rag (sibling layers), so `services/eval_snapshot/embed_input.py` mirrors the rule. `test_the_embed_input_is_the_chunkers_own_on_every_demo_chunk` pins the mirror to the real chunker on all 30 D0 chunks, and the job re-hashes every input against `embed_sha256` anyway: a divergence fails the job (`dead`) and never writes a mis-keyed vector. After extracting it, inject it from the worker: add an `embed_input` callable field to `EvalSnapshotDeps`, wired in `eval_snapshot_deps`.
4. **Backups and nuke (ADR-0014, r4 P2):** nothing to change today. The eval tenant lives in the same tables and volumes, so `make nuke` (`down --volumes`) wipes it, and `rf_backup` (`pg_read_all_data` + `backup_read` policies) reads it (tested). When the backup script is written, it must NOT filter by `tenants.kind`, and the restore drill should check that `rf.eval_snapshots` digests are equal to `rf._eval_snapshot_digests(tenant, collection)` after a restore.

## 3. Needed outside my paths
- **SDK + CLI bootstrap (locked SDK, so the integrator decides):** `RagforgeClient` has no method for `POST /v1/eval/tenant`. Proposed:
  - `async def eval_tenant_create(self) -> TenantProvisioned` (empty JSON body);
  - `ragforge-eval tenant create --key-file <owner admin key>`, which writes `api_key.secret` to a new 0600 key file (the hidden-input rule: never printed, never logged) and records `tenant.id` in the private target;
  - a Makefile target `eval-tenant` that runs it (`! make eval-tenant` style).
  - Then `ragforge-eval snapshot create` (already in `evals/cli.py`) works with the owner's admin key, and runs use the eval key.
- **DESIGN / ADR-0017 sync:** the snapshot collection name is `eval-snapshot-<YYYY-MM-DD>-<first 8 of id>`, so two snapshots on one day don't collide.

## 4. What the integrator must test live (`make up`, `RAGFORGE_EVAL_API=1`)
1. With an owner admin key: `POST /v1/eval/tenant` → 201 with the key (`no-store`); a second call → 409.
2. `POST /v1/eval/snapshots {source_collection_id: <the Notion collection>}` → 202. Watch `/v1/jobs/{job_id}` reach `done` with page/chunk/vector counts. `GET /v1/eval/snapshots/{id}` (eval key) shows both digests. Snapshotting the same unchanged collection a second time gives the same content and index digests.
3. Owner UI, `make ask` and MCP show no duplicate pages. With the eval key, `/v1/query` over the snapshot returns hits (the vectors serve as copied).
4. Exclude one page in the owner tenant. The snapshot's content digest changes, the page is gone from the eval tenant, and `/v1/eval/resolve` reports it `missing`.
5. Reconnect the Notion connector (delete + connect). The snapshot is untouched (same digests).
6. `POST .../reembed {space_id: <a new space>}`: the job embeds each distinct chunk text once, the content digest is unchanged and the index digest changes. Re-embedding back to the old space is instant, with the old index digest.
7. With the flag off, every `/v1/eval/*` route is 404 without auth.

## 5. Checks run (2026-10-21)
- `uv run pytest packages/core/tests/module_eval_snap packages/core/tests/test_db_foundation.py packages/core/tests/test_migrate_concurrency.py packages/core/tests/module_c2 packages/core/tests/module_worker packages/core/tests/test_catalog_invariants.py packages/core/tests/test_api_health.py -q`: **199 passed**.
- `ruff check` and `ruff format --check` are clean on my paths. `uv run mypy packages/core/src` is clean (168 files), and so is `MYPYPATH=packages/core/tests/module_eval_snap uv run mypy packages/core/tests/module_eval_snap/test_*.py`. `uv run lint-imports`: 6 kept, 0 broken.
- Mutation check on the migration, each mutation killed by the named tests:
  - dropping the `eval_of_tenant_id = source` guard (another-owner test);
  - the trigger moved to AFTER UPDATE (purge test);
  - dropping the exclusion filter plus inverting the vector-copy predicate (9 tests).

## 6. Known gaps
- None in the server path that I know of. `collections.active_space_id` is NOT NULL (0001), so an empty source still gets a ready snapshot with a space.
- The purge is irreversible by design (frozen snapshot). Lifting an exclusion needs a new snapshot.
