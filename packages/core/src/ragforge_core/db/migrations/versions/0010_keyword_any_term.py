"""Keyword search matches ANY query term, ranked by coverage (Phase 2b eval finding).

`websearch_to_tsquery` ANDs every term, which suits a search box and fails a question: on the
owner's notes, "Why does consistent hashing use virtual nodes?" matched nothing on the
Consistent Hashing page, because its virtual-node chunk never says "use", while a Cassandra
chunk that happened to contain every word won the keyword arm. Questions carry filler words the
answer does not, so the AND form throws away exactly the chunks it should find.

The query is now the OR of its own lexemes, and `ts_rank_cd` (cover density) ranks a chunk
higher the more terms it covers and the closer together they are. On the same notes that chunk
goes from no match to the top keyword hit. Measured in `evals/private` (B2b → B3).

Details:
  * Lexemes come from `to_tsvector(cfg, query)`, so the collection's own parser, stemmer and
    stopword list apply exactly as before. Websearch operators ("quoted phrase", -negation, or)
    are no longer interpreted: users ask questions, and a negated term inside an OR would match
    almost every row.
  * At most 32 distinct lexemes, in order of first appearance: an OR's cost grows with the rows
    each term matches, and no real question has more content words. The 1000-character cap
    stays.
  * Each lexeme is quoted for tsquery input (backslash and quote escaped) and cast with
    `::tsquery`, which does not re-parse or re-stem, so no lexeme can inject an operator.

Revision ID: 0010
Revises: 0009
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0010"
down_revision = "0009"
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
                SELECT string_agg(
                           '''' || replace(replace(t.lexeme, '\\', '\\\\'), '''', '\\''') || '''',
                           ' | ' ORDER BY t.first_pos)::tsquery
                  INTO v_tsq
                  FROM (SELECT u.lexeme, u.positions[1] AS first_pos
                          FROM unnest(to_tsvector(v_cfg, v_query)) AS u
                         ORDER BY u.positions[1], u.lexeme
                         LIMIT %(max_terms)s) t;
                CONTINUE WHEN v_tsq IS NULL;  -- stopwords only
                EXECUTE $q$
                    SELECT array_agg(h.id ORDER BY h.r DESC, h.id),
                           array_agg(h.r ORDER BY h.r DESC, h.id)
                      FROM (SELECT c.id, ts_rank_cd(c.tsv, $1) AS r
                              FROM rf.chunks c
                              JOIN rf.documents d ON d.tenant_id = $2 AND d.id = c.document_id
                                                 AND d.deleted_at IS NULL
                             WHERE c.tenant_id = $2 AND c.collection_id = ANY ($3)
                               AND c.tsv @@ $1
                             ORDER BY 2 DESC, 1
                             LIMIT $4) h
                $q$ INTO v_pids, v_pranks USING v_tsq, v_tenant, v_colls, p_k;
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
