# Module B (Retrieval) handoff

Status: done (review rounds 1 and 2 fixes applied). 104 module tests pass. `ruff check`, `ruff format --check`, `mypy --strict` (sources and tests) and `lint-imports` are clean. Catalog and foundation tests pass at head. The full suite passed when I ran it (`uv run pytest packages -q`: 921 passed).

## What exists

| Path | What |
|---|---|
| `rag/tokens.py` | `count_tokens()`: the only token counter (approximation; see below) |
| `rag/chunker.py` | `HeadingChunker` implements `Chunker` (DESIGN 7.2) |
| `rag/fusion.py`, `rag/mmr.py`, `rag/packing.py` | RRF k=60, collapse by `display_sha256`, MMR λ=0.7, `pack()` → `PackedContext` |
| `rag/retriever.py` | `SearchBackend` Protocol, `HybridRetriever` implements `Retriever`, `rerank_and_select()` |
| `adapters/db/search.py` | `PgSearchBackend` (satisfies `SearchBackend` structurally), `build_space_index()`, `drop_retired_space_index()`, `vector_literal()` |
| `adapters/db/index.py` | `PgDocumentIndex` implements `DocumentIndex` |
| `adapters/llm/embedder_ollama.py` | `OllamaEmbedder` implements `Embedder`; `EmbeddingFailed` (502) |
| `adapters/llm/embedder_registry.py` | `OllamaEmbedderRegistry` implements `EmbedderRegistry`; `load_embedding_spaces(tx)` |
| `adapters/rerank/flashrank.py` | `FlashRankReranker` implements `Reranker` |
| slot `0003_retrieval.py` | tsv trigger + GIN, `search_vectors`, `search_vectors_exact`, `search_keywords`, `gc_embeddings`, internal helpers, default-space HNSW built `CONCURRENTLY` |

## Phase 2 wiring (bootstrap.py)
```python
http = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60)
async with tm.system() as tx:
    spaces = await load_embedding_spaces(tx)
registry  = OllamaEmbedderRegistry(spaces, http, num_ctx=settings.embed_num_ctx,
                                   keep_alive=settings.ollama_keep_alive)
retriever = HybridRetriever(PgSearchBackend(exact_threshold=20_000))
reranker  = FlashRankReranker(cache_dir="/opt/models/flashrank",   # baked; never downloads
                              token_counter=rag.tokens.count_tokens)  # REQUIRED (adapters can't import rag)
index     = PgDocumentIndex()
```
**Retrieval flow for D (important):** `Retriever.retrieve()` returns **fused and collapsed candidates only**. Reranking is CPU model work (about 250 ms), and PHASE1 forbids model work inside a transaction, so `retrieve()` doesn't rerank. The full flow is:
1. Embed the query once per space, with no transaction open. Get the spaces from `spaces_for()` in a short transaction first.
2. In a short transaction, call `retrieve(..., k=RERANK_TOP)` (20).
3. Close the transaction, then call `rag.retriever.rerank_and_select(query, candidates, reranker, final_n=5|8)`.
4. Call `pack()`.

The grade node reads `hits[0].rerank_score` (ADR-0006). It's `None` when the reranker model is unavailable **or an inference call failed** (logged as `reranker_failed`, exception type only). D must handle `None`, for example by treating the answer as "can't grade" and skipping abstention.

If a blue/green space flip lands between `spaces_for()` and `retrieve()`, `retrieve()` no longer raises. It searches the active spaces that have a query vector, ignores vectors for spaces that are no longer active, logs `retrieve_space_without_vector`, and still runs keyword search over every collection. D doesn't need a retry.

## Requests for the integrator
1. **Bake the FlashRank model** into the `server` image, for example `RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/opt/models/flashrank')"`. Without it the reranker degrades gracefully: fused order, `rerank_score=None`, and one `reranker_unavailable` warning. TinyBERT-L-2 is the default because of the 250 ms budget. `ms-marco-MiniLM-L-12-v2` is the quality option, to be decided by the M3 evals.
2. **Settings** (config.py): `rerank_model`, `rerank_cache_dir`, `rerank_top_n` (20), `vector_exact_threshold` (20000), `embed_batch_size` (32). Every adapter already takes these as constructor arguments. `rf.search_vectors` clamps the threshold to [0, 100000] server-side, so values above 100000 have no effect.
3. **No new dependencies.** It uses `httpx` (core), `flashrank` (`ingest` extra; imported lazily, so a missing install also degrades gracefully), and `respx`/`hypothesis` (dev).
4. **Worker jobs:**
   - `gc_embeddings` as a daily `maintenance` job: run `SELECT rf.gc_embeddings(5000)` in a system transaction, **commit after every call**, and loop **until it returns 0**. The return value is the number of rows processed (deleted embeddings and touches, plus re-armed touches), not just deletions. Each call is bounded to about `p_batch` rows, and the loop always terminates.
   - A `ragforge-migrate` sub-step that calls `adapters.db.search.build_space_index(conn, space_id)` (autocommit, `SET ROLE rf_system`) for every non-retired space with dim ≤ 2000. This covers new spaces; 0003 builds the existing ones.
   - In the same migrator step, call `adapters.db.search.drop_retired_space_index(conn, space_id)` for every retired space. It runs `DROP INDEX CONCURRENTLY`, and returns False and does nothing for a space that is not retired or that a collection still uses. GC no longer drops indexes, because a plain `DROP INDEX` held ACCESS EXCLUSIVE on `rf.embeddings` until the GC transaction committed.
5. **Tokenizer:** `count_tokens` is a documented conservative approximation of BERT WordPiece:
   - CJK ideographs, kana, Thai, Lao, Khmer and Myanmar cost 1 token per character.
   - A Hangul syllable costs its NFD jamo count (2 or 3).
   - Each other punctuation character costs 1 token, and any other word costs `1 + (len-1)//6`.

   The first version charged a whole CJK run as one word and under-counted it about 5x (review finding). The chunker now also splits sentences at `。！？；` and windows unspaced runs by **tokens**, not characters. Hypothesis tests over CJK, kana, Hangul, Thai and mixed text check that every non-code, non-table chunk is ≤ 512 by this counter. That is a bound under the approximation, not under the real tokenizer. To make it exact: bake the nomic/BERT `tokenizer.json` (vocab) into the image and replace the body of `rag/tokens.count_tokens` only. `tokenizers` is already installed as a FlashRank dependency. Rare scripts outside the listed ranges still use the word rule.
6. **ADR-0013 amendment:** these internal functions were added, all with `REVOKE ALL FROM PUBLIC` and no grants:
   - `rf.vector_candidates_hnsw`: SECURITY DEFINER, pinned search_path, never granted;
   - `rf.vector_space_operator` and `rf.vector_sql`: INVOKER, only called from inside the definers;
   - `rf.chunks_tsv_refresh`: INVOKER trigger.

   - `rf.collections_on_space(tenant, space, collections) -> uuid[]`: INVOKER SQL function, pinned search_path, only called from inside the vector definers.

   `search_vectors` also has a 5th argument, `p_exact_threshold bigint DEFAULT 20000`. It is clamped to [0, 100000], so an rf_api caller can at most force an exact scan over 100k vectors. Both paths are tenant-isolated.
7. **CHECK on embedding_spaces** (0001 owner / integrator, in a later expand migration): `CHECK ((status = 'retired') = (retired_at IS NOT NULL))`. Until then, `gc_embeddings` stamps `retired_at = now()` on a retired space that has none, which starts its 7-day window. Without the stamp, `retired_at < now() - 7 days` is never true for NULL and the space would never be purged. I did not use `coalesce(retired_at, created_at)`, because it would purge immediately and lose the rollback window.
8. **Runbook (blue/green):** a collection's vectors in its previous space are kept for the instant rollback, and GC keeps re-arming them while its chunks still use the hash (see below). Once a flip is final, the runbook should either retire the old space (if no collection uses it any more) or, in Advanced, run a per-collection release step. That step does not exist yet; it would delete `(collection, old space)` embeddings, touches and stats in batches. **Evals against a building space** (before the flip) cannot go through `rf.search_vectors` any more, because it only serves a collection from its active space. The Advanced flip automation needs its own ungranted eval/coverage function.
9. **M3 tuning note (EXACT_THRESHOLD):** `search_vectors_exact` has `SET enable_indexscan = off` as a function attribute (ADR-0009), and the attribute covers every statement in the function. The liveness `EXISTS` (chunks_embed_sha) and the chunks/documents joins therefore run as bitmap probes or hash joins, not plain index scans. This is correct but slower per candidate. Measure the function as a whole when tuning the threshold against the `/query` p95.
10. **GIN pending list:** `chunks_tsv_gin` uses the default `fastupdate`. Right after a large sync, the pending list makes the planner cost every GIN probe like a seq scan, until (auto)vacuum merges it. Keep autovacuum on for `rf.chunks`. If keyword p95 spikes after bulk syncs, set `gin_pending_list_limit` on the index, or run `gin_clean_pending_list` at the end of a large sync.

## Design decisions worth reviewing
- **Each space serves only its own collections (round 2, P1):** `SearchBackend.active_spaces` returns `{space_id: [collections whose active_space_id is that space]}`, and `retrieve()` passes only that subset to `vector_search`. `spaces_for()` still returns the sorted space ids. As defence in depth, every vector definer (`search_vectors`, `search_vectors_exact`, `vector_candidates_hnsw`) narrows `p_collection_ids` through `rf.collections_on_space` before it searches or reads `collection_space_stats`. So even a direct rf_api call cannot be served:
  - a collection's leftover vectors in its previous space;
  - its unverified building-space vectors;
  - the leftovers' stats, which would otherwise skew the exact/HNSW choice.

  Tests cover two collections on different active spaces with leftover and building vectors that sit nearest to the query, on both paths and through the full retriever. They check that each collection is fused from exactly one list, and that the plan stays exact when only the leftovers' stats are large.
- **GC and leftovers in non-current spaces:** I chose to keep re-arming them. The `referenced` check ignores the space on purpose: ADR-0009 makes "flip `active_space_id` back" the instant rollback, and that only works if the old vectors survive. They cost storage only, since search never serves them. They are removed when the space is retired (7 days later) or by the runbook step in request 8. Making GC delete them after 24 h would shrink the rollback window to about a day after the last sync, without telling anyone.
- **Keyword search plan (round 2):** `search_keywords` now runs one `EXECUTE` per distinct `fts_config`, with the tsquery passed in `USING`, so every call gets a custom plan that knows the actual term. A test with 20k chunks over 200 documents captures the plan through auto_explain on the 6th call as rf_api and finds `Bitmap Index Scan on chunks_tsv_gin`. Each config contributes at most k hits, and the union is re-ranked. `p_query` is cut to 1000 characters before `websearch_to_tsquery`.
- **Reranker query cap (round 2):** the query is cut to `MAX_QUERY_TOKENS = 64` (binary search over prefixes with the injected counter) before the passage budget is computed. The budget is therefore at least 445 tokens and never negative.
- **Ollama errors (round 2):** the `error` field is mapped to a fixed category ("model not found", "input exceeds the model context", "out of memory", "model failed to load", "invalid request", or "unrecognised error") and is never copied, because some Ollama errors quote the input.
- **HNSW determinism (found by experiment on PG17):**
  - With a selective tenant or collection filter, the planner picks the PK btree plus a sort over HNSW. PG17 also carries the CTE's index order upward, so a "re-sort" over the same query is optimised away, even though `relaxed_order` output isn't sorted.
  - Fix: the HNSW scan lives in `vector_candidates_hnsw`. It pins `enable_seqscan`, `enable_bitmapscan` and `enable_sort` to off as **function attributes**, which are restored on exit. `search_vectors` then re-sorts its output in a `MATERIALIZED` CTE over a function scan, where the sort is real.
  - Tests check both behaviours with auto_explain: the partial index is used on the 6th call as rf_api, and iterative scan is what yields k rows at 1% selectivity.
- **Dynamic SQL:** only `{DIM}`, `{SPACE}` (integers read from `embedding_spaces`) and `{OP}` (one of `<=>`, `<->`, `<#>` from a closed CASE on the CHECK-constrained `distance`) are spliced, through `rf.vector_sql`. The SQL uses token replacement instead of `format()` because psycopg parses `%` in parameterless driver SQL, and that broke the migration.
- **Liveness before LIMIT:** both vector paths check that a chunk of a non-deleted document references the embedding **before** `ORDER BY … LIMIT 2k`. Without it, tombstoned pages (7 days) and old chunk versions (24 h GC grace) crowd live hits out of the candidate pool (review finding). On the HNSW path it is a filter that `iterative_scan` scans past. A test puts the 60 nearest vectors on archived or orphaned rows and checks that both paths return k live hits, the exact path the true live top k, with the HNSW index still used.
- **Isolation is double:** the definer search functions return only `(chunk_id, score)`. Hydration (title, text, URL) joins `chunks`/`documents` in the caller's query under RLS.
- **GC vs sync race:**
  - GC locks victim touches with `FOR UPDATE SKIP LOCKED` and deletes the touch and the embedding in one statement. `touch_and_find_missing` upserts touches, then reads existence in a **new statement**.
  - Both interleavings are tested: sync first, then the vector is kept; GC first, then the sync blocks, sees the hash missing and re-embeds.
  - Embeddings that have **no** touch row are never GC'd. Every writer touches before inserting, and deleting untouched rows could race a first touch.
- **GC cost:** a stale touch whose hash a chunk still references is **re-armed** (`touched_at = now()`) in the same statement that deletes the unreferenced ones. Each stale touch is visited once per 24 h. Batches no longer re-walk every old live touch. Re-arming only extends a hash's life, and the grace now means "24 h since GC last saw it referenced". Cost: one narrow-row update per live touch per day.
- **Retired-space purge:**
  - It only runs for spaces retired more than 7 days ago and not referenced by any collection as active or building.
  - Embeddings, then touches, then stats rows are deleted in `p_batch`-bounded steps through the PK prefix (collections, then rows). The step budget is shared across all retired spaces in one call.
  - The empty HNSW index is dropped by the migrator step `drop_retired_space_index()` (`CONCURRENTLY`), not by GC.
- **`upsert_document`:**
  - It runs `SELECT … FOR UPDATE`, then decides between inserted, updated, unchanged and stale. A first-insert race is handled by `ON CONFLICT DO NOTHING` plus one locked retry, and the UPDATEs repeat the `COALESCE(fetched_at,'-infinity') < new` guard.
  - It checks that `record.id == ids.document_id(current_tenant, …)`.
  - **Tombstones are sticky against older fetches:** `updated` and `unchanged` clear `deleted_at` only when `fetched_at > deleted_at`, meaning the fetch started after the tombstone. A sync that fetched before reconcile's tombstone and commits after it still records its content and `fetched_at` (monotonic), but the page stays deleted and keeps its 7-day purge clock. The outcome is still `updated`/`unchanged`, so C's flow is unchanged. This is tested on both paths.
- **`replace_chunks`:** upsert on `chunk_id(doc, ordinal)` with an `IS DISTINCT FROM` guard (identical rows aren't rewritten, which is tested via `xmin`), then trims `ordinal >= n`.
- **Chunker interpretations:**
  - When small sections merge, the merged chunk takes the **first** section's `heading_path`.
  - Heading text is included in `display_text`.
  - A code block over 512 tokens stays one over-long chunk; this is the only case where the max can be exceeded.
  - Overlap is whole trailing sentences (or a word suffix), never code or tables. `Chunk.metadata["overlap_chars"]` marks the overlap prefix.
- **MMR:** similarity is word-Jaccard, because `RankedChunk` carries no vector. Relevance is the rerank score, falling back to the fused score, scaled by its max (min-max scaling made novelty unable to win in small pools).
- **tsv:**
  - Fields: title (from documents) and heading path at weight A, body at weight B, using the collection's `fts_config`.
  - Changing a collection's `fts_config` needs a resync.
  - `search_keywords` runs one `websearch_to_tsquery` per distinct config and skips stopword-only queries.
- **Hot spot:** `collection_space_stats` is incremented inside the fenced transaction. Concurrent syncs into one collection serialise on that row until commit. That's acceptable at design load; a sharded counter can replace it later if needed.

- **Reranker passage budget:** the "title › heading path" header is prepended only when `query + header + chunk + 3 special tokens ≤ 512`. Otherwise the chunk is scored without the header, so its tail is never truncated by the header.

## Notes for other modules
- **C:** run step 1 (touches) in its own committed short transaction before embedding. This is what protects hashes from GC.
- **Tests:** module_b helpers are imported as `from module_b.b_support import …`, which resolves for both pytest and mypy. A bare `import b_support` doesn't resolve under mypy.
