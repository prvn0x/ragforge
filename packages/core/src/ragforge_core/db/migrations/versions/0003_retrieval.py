"""Module B (retrieval): chunks.tsv trigger + GIN, search_vectors, search_vectors_exact,
search_keywords, gc_embeddings, per-space HNSW index build, embedding_touches GC index.

OWNED BY MODULE B. Only module B edits it.

Shape of the vector search (ADR-0009):
  search_vectors(space, q, collections, k, exact_threshold)      granted: rf_api, rf_worker
    |- collections narrowed to those whose ACTIVE space is `space` (rf.collections_on_space);
    |  exact_threshold clamped to [0, 100000]
    |- cached count (collection_space_stats) < threshold, or the space's HNSW index missing /
    |  not yet valid  -> search_vectors_exact()                    granted: rf_api, rf_worker
    |                    (function attribute SET enable_indexscan = off: bitmap btree filter
    |                     + exact distance sort; restored on exit, never leaks)
    '- otherwise      -> vector_candidates_hnsw() (internal, not granted) over-fetches 2k with
                         ef_search=100 / iterative_scan=relaxed_order / max_scan_tuples=100000,
                         then search_vectors re-sorts them in a MATERIALIZED CTE.
  The HNSW scan lives in its own function for two reasons found by experiment on PG17:
    1. with a selective tenant/collection filter the planner prefers the PK btree + sort, so
       the HNSW function pins enable_seqscan/bitmapscan/sort = off (as function attributes)
       to make index use deterministic;
    2. PG17 propagates the CTE's index ordering, so a re-sort over the same query is optimised
       away; relaxed_order output is NOT sorted, so the re-sort must run over a function scan,
       which has no ordering to propagate.
  Liveness (a chunk of a non-deleted document still references the embedding) is checked
  BEFORE the LIMIT on both paths: orphaned and tombstoned embeddings linger for days (7-day
  tombstone window + 24 h GC grace) and, being near-duplicates of their live replacements,
  would otherwise fill the 2k candidate pool and starve the result. On the HNSW path the check
  is a filter that iterative_scan keeps scanning past, which is exactly what it is for.
  Dynamic SQL splices only integers read from rf.embedding_spaces (space id, dim) plus one of
  three constant operator tokens chosen by a closed CASE on the CHECK-constrained `distance`.
  Everything else is passed with USING. The tenant always comes from rf.current_tenant() and
  is applied as an explicit predicate on embeddings, chunks and documents (the owner bypasses
  RLS, so these predicates ARE the isolation).

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from alembic import op

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

# pgvector HNSW supports at most 2000 dimensions for `vector`; larger spaces use the exact path.
HNSW_MAX_DIM = 2000
OPCLASS = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}


def upgrade() -> None:
    # Index builds first: CONCURRENTLY commits whatever ran before it, so doing it first means an
    # interrupted run leaves at most an index behind (cleaned up below on retry). Every other
    # statement is idempotent (CREATE OR REPLACE / IF NOT EXISTS) for the same reason.
    _build_space_indexes()
    run_sql(_TSV_SQL)
    run_sql(_SEARCH_SQL)
    run_sql(_GC_SQL)
    run_sql(_GRANTS_SQL)
    # Module B: add DDL above this line. Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")


def _build_space_indexes() -> None:
    """Partial expression HNSW per non-retired space (ADR-0009), built CONCURRENTLY so a live
    database keeps accepting embedding writes. Later spaces are built by the migrator-run step
    (adapters.db.search.build_space_index; Phase 2 wires it into ragforge-migrate)."""
    spaces = (
        op.get_bind()
        .exec_driver_sql(
            "SELECT id, dim, distance FROM rf.embedding_spaces "
            "WHERE status <> 'retired' ORDER BY id"
        )
        .fetchall()
    )
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        for space_id, dim, distance in spaces:
            if int(dim) > HNSW_MAX_DIM:
                continue
            name = f"embeddings_hnsw_space_{int(space_id)}"
            invalid = bind.exec_driver_sql(
                "SELECT NOT i.indisvalid FROM pg_index i "  # noqa: S608 - int() only
                f"WHERE i.indexrelid = to_regclass('rf.{name}')"
            ).scalar()
            if invalid:  # leftover of an interrupted CONCURRENTLY build
                bind.exec_driver_sql(f"DROP INDEX CONCURRENTLY IF EXISTS rf.{name}")
            bind.exec_driver_sql(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON rf.embeddings "
                f"USING hnsw ((embedding::ext.vector({int(dim)})) ext.{OPCLASS[str(distance)]}) "
                f"WHERE space_id = {int(space_id)}"
            )


_TSV_SQL = """
-- Keyword index. Title + heading path weigh A, body B. INVOKER (not definer): it runs as the
-- writing role under RLS, so it can only read the collection/document of the row's own tenant.
-- A collection's fts_config change needs a resync to re-index existing chunks.
CREATE OR REPLACE FUNCTION rf.chunks_tsv_refresh() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = pg_catalog, rf, pg_temp
AS $$
DECLARE
    v_cfg   regconfig;
    v_title text;
BEGIN
    SELECT c.fts_config INTO v_cfg
      FROM rf.collections c WHERE c.id = NEW.collection_id AND c.tenant_id = NEW.tenant_id;
    SELECT d.title INTO v_title
      FROM rf.documents d WHERE d.id = NEW.document_id AND d.tenant_id = NEW.tenant_id;
    v_cfg := coalesce(v_cfg, 'simple'::regconfig);
    NEW.tsv := setweight(to_tsvector(v_cfg, coalesce(v_title, '')), 'A')
            || setweight(to_tsvector(v_cfg, array_to_string(NEW.heading_path, ' ')), 'A')
            || setweight(to_tsvector(v_cfg, NEW.display_text), 'B');
    RETURN NEW;
END
$$;
REVOKE ALL ON FUNCTION rf.chunks_tsv_refresh() FROM PUBLIC;

CREATE OR REPLACE TRIGGER chunks_tsv
    BEFORE INSERT OR UPDATE OF display_text, heading_path, embed_sha256, collection_id, document_id
    ON rf.chunks FOR EACH ROW EXECUTE FUNCTION rf.chunks_tsv_refresh();

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON rf.chunks USING gin (tsv);
-- gc_embeddings scans stale touches oldest-first.
CREATE INDEX IF NOT EXISTS embedding_touches_touched_at ON rf.embedding_touches (touched_at);
"""

_SEARCH_SQL = r"""
-- ------------------------------------------------------------------ dynamic SQL builder
-- The ONLY splicing point: integers (dim, space id) and an operator token from a closed set.
-- Token replacement instead of format(): psycopg parses percent signs in parameterless SQL.
CREATE OR REPLACE FUNCTION rf.vector_sql(p_template text, p_dim integer, p_op text, p_space_id integer)
    RETURNS text
    LANGUAGE plpgsql IMMUTABLE
    SET search_path = pg_catalog, pg_temp
AS $$
BEGIN
    IF p_op NOT IN ('<=>', '<->', '<#>') OR p_dim IS NULL OR p_space_id IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'invalid vector_sql arguments', ERRCODE = '22023';
    END IF;
    RETURN replace(replace(replace(p_template, '{DIM}', p_dim::text), '{OP}', p_op),
                   '{SPACE}', p_space_id::text);
END
$$;
REVOKE ALL ON FUNCTION rf.vector_sql(text, integer, text, integer) FROM PUBLIC;

-- ------------------------------------------------------------------ shared argument checks
-- Returns the space's (dim, operator token). The operator comes from a closed CASE over the
-- CHECK-constrained distance column, never from free text. INVOKER and not granted: it is only
-- called from inside the definer functions below (so it runs as their owner).
CREATE OR REPLACE FUNCTION rf.vector_space_operator(p_space_id integer, p_query ext.vector, p_k integer,
                                                     OUT dim integer, OUT op text)
    LANGUAGE plpgsql STABLE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_distance text;
BEGIN
    IF p_k IS NULL OR p_k < 1 OR p_k > 1000 THEN
        RAISE EXCEPTION USING MESSAGE = 'k must be between 1 and 1000', ERRCODE = '22023';
    END IF;
    SELECT s.dim, s.distance INTO dim, v_distance FROM rf.embedding_spaces s WHERE s.id = p_space_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING MESSAGE = 'unknown embedding space ' || p_space_id, ERRCODE = '22023';
    END IF;
    IF p_query IS NULL OR ext.vector_dims(p_query) <> dim THEN
        RAISE EXCEPTION USING MESSAGE = 'query vector must have ' || dim || ' dimensions',
                              ERRCODE = '22023';
    END IF;
    op := CASE v_distance WHEN 'cosine' THEN '<=>' WHEN 'l2' THEN '<->' WHEN 'ip' THEN '<#>' END;
    IF op IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'unsupported distance ' || v_distance, ERRCODE = '22023';
    END IF;
END
$$;
REVOKE ALL ON FUNCTION rf.vector_space_operator(integer, ext.vector, integer) FROM PUBLIC;

-- ------------------------------------------------------------------ collections on a space
-- The requested collections (of the CURRENT tenant) whose ACTIVE space is p_space_id. Every
-- vector search runs over this subset only, never over p_collection_ids directly: a collection
-- keeps vectors in spaces it no longer (or not yet) serves from -- its previous space after a
-- blue/green flip (kept for the instant rollback) and its building space during backfill. Those
-- must never be served (a building space is unverified until the coverage check; an old space
-- is stale), and must not double a collection's RRF weight by appearing in two space lists.
-- INVOKER and not granted: only called from inside the definers below.
CREATE OR REPLACE FUNCTION rf.collections_on_space(p_tenant uuid, p_space_id integer,
                                                    p_collection_ids uuid[])
    RETURNS uuid[]
    LANGUAGE sql STABLE
    SET search_path = pg_catalog, rf, pg_temp
AS $$
    SELECT coalesce(array_agg(c.id ORDER BY c.id), '{}'::uuid[])
      FROM rf.collections c
     WHERE c.tenant_id = p_tenant AND c.id = ANY (p_collection_ids)
       AND c.active_space_id = p_space_id
$$;
REVOKE ALL ON FUNCTION rf.collections_on_space(uuid, integer, uuid[]) FROM PUBLIC;

-- ------------------------------------------------------------------ exact path
-- enable_indexscan = off is a function attribute, so it also applies to the liveness EXISTS and
-- the chunks/documents joins below: they run as bitmap probes or hash joins instead of plain
-- index scans. Correct, but slower per candidate; the M3 tuning of EXACT_THRESHOLD measures
-- this function as a whole, so the cost is inside the number it tunes (see handoff B).
CREATE OR REPLACE FUNCTION rf.search_vectors_exact(p_space_id integer, p_query ext.vector,
                                                    p_collection_ids uuid[], p_k integer)
    RETURNS TABLE (chunk_id uuid, distance double precision)
    LANGUAGE plpgsql STABLE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
    SET enable_indexscan = off
AS $$
DECLARE
    v_tenant uuid := rf.current_tenant();
    v_dim    integer;
    v_op     text;
    v_colls  uuid[];
BEGIN
    IF v_tenant IS NULL OR coalesce(cardinality(p_collection_ids), 0) = 0 THEN
        RETURN;
    END IF;
    SELECT s.dim, s.op INTO v_dim, v_op FROM rf.vector_space_operator(p_space_id, p_query, p_k) s;
    v_colls := rf.collections_on_space(v_tenant, p_space_id, p_collection_ids);
    IF cardinality(v_colls) = 0 THEN
        RETURN;
    END IF;
    RETURN QUERY EXECUTE rf.vector_sql(
        $q$
        WITH cand AS MATERIALIZED (
            SELECT e.collection_id, e.embed_sha256,
                   ((e.embedding::ext.vector({DIM})) OPERATOR(ext.{OP}) $1::ext.vector({DIM}))::float8 AS distance
              FROM rf.embeddings e
             WHERE e.space_id = {SPACE} AND e.tenant_id = $2 AND e.collection_id = ANY ($3)
               AND EXISTS (SELECT 1 FROM rf.chunks c
                             JOIN rf.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id
                                                AND d.deleted_at IS NULL
                            WHERE c.tenant_id = $2 AND c.collection_id = e.collection_id
                              AND c.embed_sha256 = e.embed_sha256)
             ORDER BY 3
             LIMIT $4 * 2
        ), hits AS (
            SELECT DISTINCT ON (c.id) c.id AS chunk_id, cand.distance
              FROM cand
              JOIN rf.chunks c ON c.tenant_id = $2 AND c.collection_id = cand.collection_id
                              AND c.embed_sha256 = cand.embed_sha256
              JOIN rf.documents d ON d.tenant_id = $2 AND d.id = c.document_id AND d.deleted_at IS NULL
             ORDER BY c.id, cand.distance
        )
        SELECT h.chunk_id, h.distance FROM hits h ORDER BY h.distance, h.chunk_id LIMIT $4
        $q$, v_dim, v_op, p_space_id)
    USING p_query, v_tenant, v_colls, p_k;
END
$$;
REVOKE ALL ON FUNCTION rf.search_vectors_exact(integer, ext.vector, uuid[], integer) FROM PUBLIC;

-- ------------------------------------------------------------------ HNSW candidates (internal)
CREATE OR REPLACE FUNCTION rf.vector_candidates_hnsw(p_space_id integer, p_query ext.vector,
                                                      p_collection_ids uuid[], p_limit integer)
    RETURNS TABLE (collection_id uuid, embed_sha256 bytea, distance double precision)
    LANGUAGE plpgsql STABLE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
    SET enable_seqscan = off
    SET enable_bitmapscan = off
    SET enable_sort = off
    SET hnsw.ef_search = 100
    SET hnsw.iterative_scan = relaxed_order
    SET hnsw.max_scan_tuples = 100000
AS $$
DECLARE
    v_tenant uuid := rf.current_tenant();
    v_dim    integer;
    v_op     text;
    v_colls  uuid[];
BEGIN
    IF v_tenant IS NULL OR coalesce(cardinality(p_collection_ids), 0) = 0 THEN
        RETURN;
    END IF;
    SELECT s.dim, s.op INTO v_dim, v_op
      FROM rf.vector_space_operator(p_space_id, p_query, least(p_limit, 1000)) s;
    v_colls := rf.collections_on_space(v_tenant, p_space_id, p_collection_ids);
    IF cardinality(v_colls) = 0 THEN
        RETURN;
    END IF;
    RETURN QUERY EXECUTE rf.vector_sql(
        $q$
        SELECT e.collection_id, e.embed_sha256,
               ((e.embedding::ext.vector({DIM})) OPERATOR(ext.{OP}) $1::ext.vector({DIM}))::float8
          FROM rf.embeddings e
         WHERE e.space_id = {SPACE} AND e.tenant_id = $2 AND e.collection_id = ANY ($3)
           AND EXISTS (SELECT 1 FROM rf.chunks c
                         JOIN rf.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id
                                            AND d.deleted_at IS NULL
                        WHERE c.tenant_id = $2 AND c.collection_id = e.collection_id
                          AND c.embed_sha256 = e.embed_sha256)
         ORDER BY (e.embedding::ext.vector({DIM})) OPERATOR(ext.{OP}) $1::ext.vector({DIM})
         LIMIT $4
        $q$, v_dim, v_op, p_space_id)
    USING p_query, v_tenant, v_colls, p_limit;
END
$$;
REVOKE ALL ON FUNCTION rf.vector_candidates_hnsw(integer, ext.vector, uuid[], integer) FROM PUBLIC;

-- ------------------------------------------------------------------ dispatcher
-- p_exact_threshold is the adapter's tuning knob (settings.vector_exact_threshold), but the
-- function is granted to rf_api, so it is clamped to [0, 100000]: a caller can at most force an
-- exact scan over 100k vectors, never over an arbitrarily large collection.
CREATE OR REPLACE FUNCTION rf.search_vectors(p_space_id integer, p_query ext.vector,
                                              p_collection_ids uuid[], p_k integer,
                                              p_exact_threshold bigint DEFAULT 20000)
    RETURNS TABLE (chunk_id uuid, distance double precision)
    LANGUAGE plpgsql STABLE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
#variable_conflict use_column
DECLARE
    v_tenant    uuid := rf.current_tenant();
    v_threshold bigint := least(greatest(coalesce(p_exact_threshold, 20000), 0), 100000);
    v_colls     uuid[];
    v_count     bigint;
    v_index_ok  boolean;
BEGIN
    IF v_tenant IS NULL OR coalesce(cardinality(p_collection_ids), 0) = 0 THEN
        RETURN;
    END IF;
    PERFORM rf.vector_space_operator(p_space_id, p_query, p_k);
    v_colls := rf.collections_on_space(v_tenant, p_space_id, p_collection_ids);
    IF cardinality(v_colls) = 0 THEN
        RETURN;
    END IF;
    -- Cached count maintained by the index writer and GC; never count(*) per query. Only the
    -- collections served from this space count: leftovers of other collections would skew it.
    SELECT coalesce(sum(s.vector_count), 0) INTO v_count
      FROM rf.collection_space_stats s
     WHERE s.tenant_id = v_tenant AND s.collection_id = ANY (v_colls)
       AND s.space_id = p_space_id;
    SELECT coalesce(bool_and(i.indisvalid AND i.indisready), false) INTO v_index_ok
      FROM pg_catalog.pg_index i
     WHERE i.indexrelid = pg_catalog.to_regclass('rf.embeddings_hnsw_space_' || p_space_id);

    IF v_count < v_threshold OR NOT v_index_ok THEN
        RETURN QUERY
            SELECT x.chunk_id, x.distance
              FROM rf.search_vectors_exact(p_space_id, p_query, v_colls, p_k) x;
        RETURN;
    END IF;

    RETURN QUERY
        WITH cand AS MATERIALIZED (
            SELECT v.collection_id, v.embed_sha256, v.distance
              FROM rf.vector_candidates_hnsw(p_space_id, p_query, v_colls, p_k * 2) v
        ), hits AS (
            SELECT DISTINCT ON (c.id) c.id AS hit_id, cand.distance AS hit_distance
              FROM cand
              JOIN rf.chunks c ON c.tenant_id = v_tenant AND c.collection_id = cand.collection_id
                              AND c.embed_sha256 = cand.embed_sha256
              JOIN rf.documents d ON d.tenant_id = v_tenant AND d.id = c.document_id
                                 AND d.deleted_at IS NULL
             ORDER BY c.id, cand.distance
        )
        SELECT h.hit_id, h.hit_distance FROM hits h ORDER BY h.hit_distance, h.hit_id LIMIT p_k;
END
$$;
REVOKE ALL ON FUNCTION rf.search_vectors(integer, ext.vector, uuid[], integer, bigint) FROM PUBLIC;

-- ------------------------------------------------------------------ keyword search
-- One websearch_to_tsquery per DISTINCT fts_config among the requested collections, each
-- applied only to the collections that use that config. Each config runs as its own EXECUTE
-- (static text, everything in USING) so the planner sees the actual tsquery: a custom plan
-- estimates @@ from the tsv statistics and picks the chunks_tsv_gin bitmap scan for selective
-- terms, instead of a generic plan over an opaque CTE value that walks the whole collection
-- through chunks_tenant_collection. Each config contributes at most k hits; the union is then
-- re-ranked. The query text is capped at 1000 characters: websearch_to_tsquery and the GIN
-- probe cost grow with it, and no real question is longer.
CREATE OR REPLACE FUNCTION rf.search_keywords(p_query text, p_collection_ids uuid[], p_k integer)
    RETURNS TABLE (chunk_id uuid, rank real)
    LANGUAGE plpgsql STABLE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_tenant uuid := rf.current_tenant();
    v_query  text := left(p_query, 1000);
    v_cfg    regconfig;
    v_colls  uuid[];
    v_tsq    tsquery;
    v_ids    uuid[] := '{}';
    v_ranks  real[] := '{}';
    v_pids   uuid[];
    v_pranks real[];
BEGIN
    IF p_k IS NULL OR p_k < 1 OR p_k > 1000 THEN
        RAISE EXCEPTION USING MESSAGE = 'k must be between 1 and 1000', ERRCODE = '22023';
    END IF;
    IF v_tenant IS NULL OR coalesce(cardinality(p_collection_ids), 0) = 0
       OR coalesce(btrim(v_query), '') = '' THEN
        RETURN;
    END IF;
    FOR v_cfg, v_colls IN
        SELECT col.fts_config, array_agg(col.id)
          FROM rf.collections col
         WHERE col.tenant_id = v_tenant AND col.id = ANY (p_collection_ids)
         GROUP BY col.fts_config
    LOOP
        v_tsq := websearch_to_tsquery(v_cfg, v_query);
        CONTINUE WHEN numnode(v_tsq) = 0;  -- stopwords only
        EXECUTE $q$
            SELECT array_agg(h.id ORDER BY h.r DESC, h.id), array_agg(h.r ORDER BY h.r DESC, h.id)
              FROM (SELECT c.id, ts_rank_cd(c.tsv, $1) AS r
                      FROM rf.chunks c
                      JOIN rf.documents d ON d.tenant_id = $2 AND d.id = c.document_id
                                         AND d.deleted_at IS NULL
                     WHERE c.tenant_id = $2 AND c.collection_id = ANY ($3) AND c.tsv @@ $1
                     ORDER BY 2 DESC, 1
                     LIMIT $4) h
        $q$ INTO v_pids, v_pranks USING v_tsq, v_tenant, v_colls, p_k;
        v_ids := v_ids || coalesce(v_pids, '{}'::uuid[]);
        v_ranks := v_ranks || coalesce(v_pranks, '{}'::real[]);
    END LOOP;
    RETURN QUERY
        SELECT u.id, u.r FROM unnest(v_ids, v_ranks) AS u(id, r) ORDER BY u.r DESC, u.id LIMIT p_k;
END
$$;
REVOKE ALL ON FUNCTION rf.search_keywords(text, uuid[], integer) FROM PUBLIC;
"""

_GC_SQL = r"""
-- Cross-tenant maintenance (worker). Each call does at most ~p_batch units of work and returns
-- how many rows it processed; the worker loops (one committed transaction per call) until it
-- returns 0.
--
-- 1. Stale touches (touched_at older than 24 h), oldest first, at most p_batch:
--    * still referenced by a chunk -> RE-ARMED (touched_at = now()). Without this, every call
--      would re-walk all old-but-live touches (most of the table) before reaching a victim,
--      making each batch O(live touches). Re-arming only extends a hash's life: the grace now
--      means "24 h since GC last saw it referenced", which is the safer reading anyway.
--    * unreferenced -> the touch and the embedding are deleted in one statement.
--    The referenced check ignores the space on purpose: a collection's vectors in its PREVIOUS
--    space (after a blue/green flip) stay re-armed while its chunks still use the hash, because
--    flipping active_space_id back is the ADR-0009 instant rollback. They cost storage only:
--    search serves a collection from its active space alone (rf.collections_on_space), and
--    they are purged with the space once it is retired.
-- 2. Spaces retired more than 7 days ago and referenced by no collection: their embeddings,
--    then their touches, then their stats rows, all in p_batch-sized steps. The (empty) HNSW
--    index is NOT dropped here: DROP INDEX would hold ACCESS EXCLUSIVE on rf.embeddings until
--    the caller commits, blocking all vector search and sync writes. The migrator-run
--    adapters.db.search.drop_retired_space_index() drops it CONCURRENTLY instead.
--
-- Concurrency with a sync (DocumentIndex.touch_and_find_missing upserts touches, THEN reads
-- which hashes exist): GC locks each stale touch row (SKIP LOCKED) and deletes it in the same
-- statement as the embedding. A touch that commits first makes the row fail the re-checked
-- `touched_at` predicate, so it is not a victim; a touch that arrives later blocks on GC's row
-- lock, then re-inserts the touch and (new statement, new snapshot) sees the embedding gone and
-- re-embeds it. Either order leaves no chunk pointing at a deleted embedding.
-- Embeddings with NO touch row are never deleted here: every writer touches before inserting,
-- and deleting untouched rows could race a concurrent first touch.
CREATE OR REPLACE FUNCTION rf.gc_embeddings(p_batch integer DEFAULT 5000)
    RETURNS bigint
    LANGUAGE plpgsql VOLATILE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
    SET lock_timeout = '5s'
AS $$
DECLARE
    v_processed bigint := 0;
    v_budget    bigint := p_batch;
    v_n         bigint;
    v_space     integer;
BEGIN
    IF p_batch IS NULL OR p_batch < 1 OR p_batch > 100000 THEN
        RAISE EXCEPTION USING MESSAGE = 'batch must be between 1 and 100000', ERRCODE = '22023';
    END IF;
    -- A runbook that sets status = 'retired' without retired_at would otherwise never be
    -- purged (NULL < x is not true). Stamping it now starts the 7-day rollback window instead
    -- of purging at once (coalesce with created_at would delete the vectors immediately).
    UPDATE rf.embedding_spaces s SET retired_at = now()
     WHERE s.status = 'retired' AND s.retired_at IS NULL;

    WITH stale AS (
        SELECT t.tenant_id, t.collection_id, t.space_id, t.embed_sha256,
               EXISTS (SELECT 1 FROM rf.chunks c
                        WHERE c.tenant_id = t.tenant_id AND c.collection_id = t.collection_id
                          AND c.embed_sha256 = t.embed_sha256) AS referenced
          FROM rf.embedding_touches t
         WHERE t.touched_at < now() - interval '24 hours'
         ORDER BY t.touched_at
         LIMIT p_batch
           FOR UPDATE OF t SKIP LOCKED
    ), rearmed AS (
        UPDATE rf.embedding_touches t
           SET touched_at = now()
          FROM stale v
         WHERE v.referenced
           AND t.tenant_id = v.tenant_id AND t.collection_id = v.collection_id
           AND t.space_id = v.space_id AND t.embed_sha256 = v.embed_sha256
        RETURNING 1
    ), untouched AS (
        DELETE FROM rf.embedding_touches t
         USING stale v
         WHERE NOT v.referenced
           AND t.tenant_id = v.tenant_id AND t.collection_id = v.collection_id
           AND t.space_id = v.space_id AND t.embed_sha256 = v.embed_sha256
        RETURNING t.tenant_id, t.collection_id, t.space_id, t.embed_sha256
    ), gone AS (
        DELETE FROM rf.embeddings e
         USING untouched u
         WHERE e.tenant_id = u.tenant_id AND e.collection_id = u.collection_id
           AND e.space_id = u.space_id AND e.embed_sha256 = u.embed_sha256
        RETURNING e.collection_id, e.space_id
    ), counts AS (
        SELECT g.collection_id, g.space_id, count(*) AS n FROM gone g GROUP BY 1, 2
    ), stats AS (
        UPDATE rf.collection_space_stats s
           SET vector_count = greatest(0, s.vector_count - c.n), updated_at = now()
          FROM counts c
         WHERE s.collection_id = c.collection_id AND s.space_id = c.space_id
        RETURNING 1
    )
    SELECT (SELECT count(*) FROM rearmed) + (SELECT count(*) FROM untouched) INTO v_n;
    v_processed := v_processed + v_n;

    FOR v_space IN
        SELECT s.id FROM rf.embedding_spaces s
         WHERE s.status = 'retired' AND s.retired_at < now() - interval '7 days'
           AND NOT EXISTS (SELECT 1 FROM rf.collections c
                            WHERE c.active_space_id = s.id OR c.building_space_id = s.id)
         ORDER BY s.id
    LOOP
        EXIT WHEN v_budget <= 0;
        -- Walk collections so each delete uses the PK prefix (tenant, collection, space)
        -- instead of a full scan for space_id.
        DELETE FROM rf.embeddings e
         WHERE e.ctid = ANY (ARRAY(
               SELECT e2.ctid FROM rf.collections c
                 JOIN rf.embeddings e2 ON e2.tenant_id = c.tenant_id AND e2.collection_id = c.id
                                      AND e2.space_id = v_space
                LIMIT v_budget));
        GET DIAGNOSTICS v_n = ROW_COUNT;
        v_processed := v_processed + v_n;
        v_budget := v_budget - v_n;
        CONTINUE WHEN v_budget <= 0;
        DELETE FROM rf.embedding_touches t
         WHERE t.ctid = ANY (ARRAY(
               SELECT t2.ctid FROM rf.collections c
                 JOIN rf.embedding_touches t2 ON t2.tenant_id = c.tenant_id
                                             AND t2.collection_id = c.id AND t2.space_id = v_space
                LIMIT v_budget));
        GET DIAGNOSTICS v_n = ROW_COUNT;
        v_processed := v_processed + v_n;
        v_budget := v_budget - v_n;
        CONTINUE WHEN v_budget <= 0;
        DELETE FROM rf.collection_space_stats s WHERE s.space_id = v_space;
    END LOOP;
    RETURN v_processed;
END
$$;
REVOKE ALL ON FUNCTION rf.gc_embeddings(integer) FROM PUBLIC;
"""

_GRANTS_SQL = """
GRANT EXECUTE ON FUNCTION rf.search_vectors(integer, ext.vector, uuid[], integer, bigint),
                          rf.search_vectors_exact(integer, ext.vector, uuid[], integer),
                          rf.search_keywords(text, uuid[], integer)
      TO rf_api, rf_worker;
GRANT EXECUTE ON FUNCTION rf.gc_embeddings(integer) TO rf_worker;
"""
