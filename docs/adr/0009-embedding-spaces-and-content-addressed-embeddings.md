# ADR-0009: Embedding spaces, content-addressed embeddings, and search functions

**Status:** Accepted · 2026-10-16 · Revised in rounds 3 and 4

## Context
1. A fixed vector column ties the schema to one model.
2. Under RLS, prepared generic plans can't match a partial index (`space_id = $1`), and non-leakproof `@@` can block GIN.
3. Filtered HNSW under-returns. `relaxed_order` is unsorted. `max_scan_tuples` defaults to 20k.
4. A filter on a column the HNSW scan can't see (collection, when it lived only on chunks) starves results.
5. Embeddings are written before their chunk transaction, so a GC that runs in between could delete them.
6. During blue/green, syncs that embed only into the active space leave gaps in the new one.

## Decision
- **Spaces:** `embedding_spaces(id, provider, model, dim, doc_prefix, query_prefix, distance, status building|active|retired)`; `collections.active_space_id`.
- **Embeddings:** `embeddings(tenant_id, collection_id, space_id, embed_sha256, embedding ext.vector, created_at)`. Tenant **and collection** are on the row, so the HNSW scan's filter sees both.
  - The hash covers the embedded text (including the header). Unchanged text keeps its embedding when ordinals move; only new or changed text is embedded.
  - Cross-page duplicates are collapsed at retrieval by `display_sha256`.
- **Index:** a partial expression HNSW per space, `((embedding::ext.vector(<dim>)) ext.vector_cosine_ops) WHERE space_id = <id>`, built `CONCURRENTLY`.
- **`search_vectors(space, q, collections, k)`:** SECURITY DEFINER, owned by `rf_system`.
  - The tenant comes from the setting and is applied as an explicit predicate. `EXECUTE` dynamic SQL means a custom plan every time; only the integer space ID and dim are spliced in, everything else is passed with `USING`.
  - **Exact path:** used if the cached count (`collection_space_stats`, maintained by the sync and GC jobs, never `count()` per query) for (tenant, collections, space) is < `EXACT_THRESHOLD` (default 20k; tuned in M3 against the `/query` p95 **including TOAST detoast cost**, ~3 KB per vector), or if the space's HNSW index isn't `indisvalid` yet.
    - It's a separate function, `search_vectors_exact()`, declared with `SET enable_indexscan = off` as a **function attribute**, so the setting is restored on exit and can't leak into later statements.
    - The btree filter runs as a bitmap scan, then distance ordering with `LIMIT k`. Vectors are never materialized into work_mem.
  - **HNSW path** otherwise: `ef_search = 100`, `iterative_scan = relaxed_order`, `max_scan_tuples = 100000`, filter on `tenant_id AND collection_id = ANY($c)`, over-fetch 2×, then re-sort in a materialized CTE.
  - Join to chunks on `(tenant_id, collection_id, embed_sha256)` with `DISTINCT ON (chunk)`.
- **Multi-space queries:** when the requested collections have different active spaces, embed the query once per space, search each, and fuse with RRF (rank-based, so scores never need to be comparable across spaces).
- **GC:** `gc_embeddings()` deletes unreferenced rows whose last touch is **older than 24 h**. Touches go to a narrow side table, `embedding_touches(tenant_id, collection_id, space_id, embed_sha256, touched_at)`, **not** to `embeddings`. Updating the vector row would be non-HOT and would insert a fresh HNSW entry plus a dead tuple on every sync. Before embedding, every sync **upserts touches for all the hashes it will reference** and reads which ones already exist in `embeddings`. It embeds only the missing hashes, and inserts them with `ON CONFLICT DO NOTHING` (the touch row already protects them). Retired spaces: 7 days after retirement, a maintenance job drops their index and deletes their embeddings and touches. The 24 h grace is far longer than any sync, so in-flight syncs are protected. A metric counts orphan hits, and a repair job re-enqueues affected pages.
- **Blue/green:** while `collections.building_space_id` is set, every sync also embeds into it (dual write). Backfill covers older content. The flip requires a **coverage check** (every chunk has an embedding in the new space) plus evals. Flipping `active_space_id` back is the instant rollback. Automation is Advanced; MVP/Core use the runbook.

## Consequences
- Deterministic index use, verified in tests with `auto_explain` (`log_nested_statements = on`, `log_analyze = on`) capturing the plans **inside** the functions, called as `rf_api` 6+ times, for both paths.
- Explicit-predicate isolation inside the function, verified by the cross-tenant test.
- Storage grows with the number of collections that hold identical text. Accepted.
