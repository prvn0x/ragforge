"""Keyword rank: distinct query terms covered first, then cover density (Phase 2b review).

0010 ranked the OR query by `ts_rank_cd` alone, which counts every occurrence: a chunk saying
"node" eight times scored 0.80 against 0.40 for "consistent hashing places virtual nodes on a
ring", which covers all four terms once (measured; normalisation 1|32 keeps the order). For a
question the second chunk is the answer, so 0010's docstring claim ("ranked by coverage") did not
hold.

The rank is now `matched + d / (1 + d)`, where `matched` is how many DISTINCT query lexemes the
chunk contains (exact lexeme equality, from `tsvector_to_array`) and `d` is `ts_rank_cd`, mapped
into [0, 1) so it only breaks ties between chunks covering the same number of terms. Title and
heading weights still act through `d`. Everything else is 0010 unchanged: the OR of at most 32
lexemes, quoting, tenant scope, SECURITY DEFINER with pinned search_path, grants (CREATE OR
REPLACE keeps them).

Cost: one pass over each matching chunk's lexemes. The OR already ranks every matching chunk of
the tenant's collections, which is fine for a personal corpus and is the thing to revisit (a
term-frequency cutoff or a BM25 index) before multi-user scale.

Revision ID: 0011
Revises: 0010
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

MAX_TERMS = 32


def upgrade() -> None:
    run_sql(
        """
        CREATE OR REPLACE FUNCTION rf.search_keywords(
            p_query text, p_collection_ids uuid[], p_k integer)
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
            v_terms  text[];
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
                -- OR of the query's own lexemes, first-appearance order, capped.
                SELECT array_agg(t.lexeme ORDER BY t.first_pos), string_agg(
                           '''' || replace(replace(t.lexeme, '\\', '\\\\'), '''', '\\''') || '''',
                           ' | ' ORDER BY t.first_pos)::tsquery
                  INTO v_terms, v_tsq
                  FROM (SELECT u.lexeme, u.positions[1] AS first_pos
                          FROM unnest(to_tsvector(v_cfg, v_query)) AS u
                         ORDER BY u.positions[1], u.lexeme
                         LIMIT %(max_terms)s) t;
                CONTINUE WHEN v_tsq IS NULL;  -- stopwords only
                EXECUTE $q$
                    SELECT array_agg(h.id ORDER BY h.r DESC, h.id),
                           array_agg(h.r ORDER BY h.r DESC, h.id)
                      FROM (SELECT c.id,
                                   ((SELECT count(*) FROM unnest(tsvector_to_array(c.tsv)) l
                                     WHERE l = ANY ($5))
                                   + ts_rank_cd(c.tsv, $1) / (1 + ts_rank_cd(c.tsv, $1)))::real AS r
                              FROM rf.chunks c
                              JOIN rf.documents d ON d.tenant_id = $2 AND d.id = c.document_id
                                                 AND d.deleted_at IS NULL
                             WHERE c.tenant_id = $2 AND c.collection_id = ANY ($3)
                               AND c.tsv @@ $1
                             ORDER BY 2 DESC, 1
                             LIMIT $4) h
                $q$ INTO v_pids, v_pranks USING v_tsq, v_tenant, v_colls, p_k, v_terms;
                v_ids := v_ids || coalesce(v_pids, '{}'::uuid[]);
                v_ranks := v_ranks || coalesce(v_pranks, '{}'::real[]);
            END LOOP;
            RETURN QUERY
                SELECT u.id, u.r FROM unnest(v_ids, v_ranks) AS u(id, r)
                 ORDER BY u.r DESC, u.id LIMIT p_k;
        END
        $$;
        """.replace("%(max_terms)s", str(MAX_TERMS))
    )
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
