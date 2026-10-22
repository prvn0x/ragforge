"""SQL side of retrieval: satisfies rag.retriever.SearchBackend structurally (adapters may not
import rag), plus the migrator-run per-space HNSW index build.

Isolation is layered: the SECURITY DEFINER search functions apply the tenant from
rf.current_tenant() as an explicit predicate and return only (chunk_id, score); hydration then
joins chunks/documents in THIS query as the calling role, so RLS filters it a second time. A bug
in either layer alone cannot leak another tenant's text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import psycopg
from psycopg import sql
from sqlalchemy import text

from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Tx

EXACT_THRESHOLD = 20_000  # ADR-0009 default; tuned in M3 against /query p95
HNSW_MAX_DIM = 2000  # pgvector HNSW limit for the `vector` type
_OPCLASS = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}

# Hydration runs as the calling role, so RLS filters chunks/documents a second time.
_VECTOR_SEARCH = text(
    """
    WITH hits AS (
        SELECT v.chunk_id, -v.distance AS score
          FROM rf.search_vectors(:space_id, CAST(:query AS ext.vector),
                                 CAST(:collections AS uuid[]), :k, :threshold) v
    )
    SELECT c.id, c.document_id, c.collection_id, c.anchor, d.title, c.heading_path,
           c.display_text, c.display_sha256, d.external_url, s.score
      FROM hits s
      JOIN rf.chunks c ON c.id = s.chunk_id
      JOIN rf.documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
     ORDER BY s.score DESC, c.id
    """
)
_KEYWORD_SEARCH = text(
    """
    WITH hits AS (
        SELECT k.chunk_id, k.rank::float8 AS score
          FROM rf.search_keywords(:query, CAST(:collections AS uuid[]), :k) k
    )
    SELECT c.id, c.document_id, c.collection_id, c.anchor, d.title, c.heading_path,
           c.display_text, c.display_sha256, d.external_url, s.score
      FROM hits s
      JOIN rf.chunks c ON c.id = s.chunk_id
      JOIN rf.documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
     ORDER BY s.score DESC, c.id
    """
)
_ACTIVE_SPACES = text(
    """
    SELECT c.active_space_id, array_agg(c.id ORDER BY c.id)
      FROM rf.collections c
      JOIN rf.embedding_spaces s ON s.id = c.active_space_id
     WHERE c.tenant_id = rf.current_tenant() AND c.id = ANY (CAST(:collections AS uuid[]))
       AND s.status = 'active'
     GROUP BY 1
     ORDER BY 1
    """
)


class PgSearchBackend:
    """Postgres implementation of the retriever's SearchBackend. Stateless; share one."""

    def __init__(self, *, exact_threshold: int = EXACT_THRESHOLD) -> None:
        if exact_threshold < 0:
            raise ValueError("exact_threshold must be >= 0")
        self._exact_threshold = exact_threshold

    async def active_spaces(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[int, Sequence[UUID]]:
        """Active space -> the collections served from it (rf.search_vectors narrows to the
        same set server-side; this keeps each call's argument honest and its stats exact)."""
        result = await connection(tx).execute(_ACTIVE_SPACES, {"collections": list(collection_ids)})
        return {int(row[0]): [UUID(str(c)) for c in row[1]] for row in result}

    async def vector_search(
        self,
        tx: Tx,
        space_id: int,
        query_vector: Sequence[float],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> Sequence[RankedChunk]:
        result = await connection(tx).execute(
            _VECTOR_SEARCH,
            {
                "space_id": int(space_id),
                "query": vector_literal(query_vector),
                "collections": list(collection_ids),
                "k": int(k),
                "threshold": self._exact_threshold,
            },
        )
        return [_ranked(row) for row in result]

    async def keyword_search(
        self, tx: Tx, query: str, collection_ids: Sequence[UUID], k: int
    ) -> Sequence[RankedChunk]:
        result = await connection(tx).execute(
            _KEYWORD_SEARCH, {"query": query, "collections": list(collection_ids), "k": int(k)}
        )
        return [_ranked(row) for row in result]


def vector_literal(values: Sequence[float]) -> str:
    """pgvector text form. repr() round-trips float64 exactly; NaN/inf are rejected here with a
    clear error instead of a pgvector parse error deep inside a definer function."""
    if not values:
        raise ValueError("empty vector")
    parts: list[str] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("vector contains NaN or infinity")
        parts.append(repr(number))
    return "[" + ",".join(parts) + "]"


def _ranked(row: Any) -> RankedChunk:
    return RankedChunk(
        chunk_id=row[0],
        document_id=row[1],
        collection_id=row[2],
        anchor=row[3],
        title=row[4],
        heading_path=tuple(row[5] or ()),
        display_text=row[6],
        display_sha256=bytes(row[7]),
        external_url=row[8],
        fused_score=float(row[9]),
    )


# ---------------------------------------------------------------------- space index build
def space_index_name(space_id: int) -> str:
    """Naming contract shared with migration 0003 and rf.search_vectors (index validity check)."""
    return f"embeddings_hnsw_space_{int(space_id)}"


class IndexBuildInProgress(RuntimeError):
    """An invalid space index may still be under construction by another session; dropping it
    would cancel that build's work. The caller retries on its next run."""


# Sessions that may be building an index on rf.embeddings: holders of its
# ShareUpdateExclusiveLock (a CONCURRENTLY build holds it throughout), MINUS the routine holders
# of the same lock: autovacuum workers and manual VACUUM / ANALYZE. Those are matched by pid,
# which pg_stat_activity and the progress views show for every role (other columns of other
# roles' rows may be NULL, which is why the create-index view alone is not authoritative).
POSSIBLE_INDEX_BUILDERS = """
    SELECT l.pid
      FROM pg_locks l
     WHERE l.relation = 'rf.embeddings'::regclass
       AND l.mode = 'ShareUpdateExclusiveLock'
       AND l.granted AND l.pid <> pg_backend_pid()
       AND NOT EXISTS (SELECT 1 FROM pg_stat_activity a
                        WHERE a.pid = l.pid AND a.backend_type = 'autovacuum worker')
       AND NOT EXISTS (SELECT 1 FROM pg_stat_progress_vacuum v WHERE v.pid = l.pid)
       AND NOT EXISTS (SELECT 1 FROM pg_stat_progress_analyze z WHERE z.pid = l.pid)
"""
_INDEX_STATE = sql.SQL(
    """
    SELECT i.indisvalid,
           EXISTS (SELECT 1 FROM pg_stat_progress_create_index p
                    WHERE p.index_relid = i.indexrelid)
           OR EXISTS ({builders})
      FROM pg_index i
     WHERE i.indexrelid = to_regclass(%s)
"""
).format(builders=sql.SQL(POSSIBLE_INDEX_BUILDERS))


def build_space_index(conn: psycopg.Connection[Any], space_id: int) -> bool:
    """Build (or finish) the partial expression HNSW index of one space, CONCURRENTLY.

    Migrator-run step (PHASE1: CREATE INDEX CONCURRENTLY cannot run inside a function or a
    transaction): `conn` must be in autocommit mode and its current role must own rf.embeddings
    (the migrator after `SET ROLE rf_system`). Until the index is valid, rf.search_vectors
    serves the space through the exact path, so building late is safe, only slower.
    Returns False when the space cannot have an HNSW index (dim > 2000): it stays exact-only.

    A valid index is left alone (no DDL, so no wait on concurrent writers). An INVALID index is
    either the leftover of an interrupted build (dropped and rebuilt) or a build still running
    in another session. Raises `IndexBuildInProgress` instead of dropping when that is possible:
    the progress view shows the build (rows of other roles can be hidden from us, so this is
    backed by pg_locks: a CONCURRENTLY build holds a session ShareUpdateExclusiveLock on
    rf.embeddings for its whole duration). Autovacuum and VACUUM/ANALYZE hold the same lock
    routinely and are excluded (`POSSIBLE_INDEX_BUILDERS`): counting them would leave an invalid
    leftover, which every insert still maintains, in place until a later run.
    """
    if not conn.autocommit:
        raise ValueError("build_space_index needs an autocommit connection (CONCURRENTLY)")
    row = conn.execute(
        "SELECT dim, distance FROM rf.embedding_spaces WHERE id = %s", (int(space_id),)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown embedding space {space_id}")
    dim, distance = int(row[0]), str(row[1])
    if dim > HNSW_MAX_DIM:
        return False
    name = space_index_name(space_id)
    state = conn.execute(_INDEX_STATE, (f"rf.{name}",)).fetchone()
    if state is not None:
        valid, busy = bool(state[0]), bool(state[1])
        if valid:
            return True
        if busy:
            raise IndexBuildInProgress(f"space {int(space_id)}: index build may be in progress")
        conn.execute(
            sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS rf.{}").format(sql.Identifier(name))
        )
    conn.execute(
        sql.SQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON rf.embeddings "
            "USING hnsw ((embedding::ext.vector({dim})) ext.{opclass}) WHERE space_id = {space}"
        ).format(
            name=sql.Identifier(name),
            dim=sql.Literal(dim),
            opclass=sql.Identifier(_OPCLASS[distance]),
            space=sql.Literal(int(space_id)),
        )
    )
    return True


def drop_retired_space_index(conn: psycopg.Connection[Any], space_id: int) -> bool:
    """Drop the HNSW index of a retired space, CONCURRENTLY.

    Migrator-run step, same connection contract as `build_space_index` (autocommit, current
    role owns rf.embeddings). It lives here and not in rf.gc_embeddings because a plain
    DROP INDEX inside the GC transaction would hold ACCESS EXCLUSIVE on rf.embeddings until the
    worker commits, stalling every vector search and sync write meanwhile.
    Returns False (and drops nothing) unless the space is retired and no collection uses it as
    active or building space. Dropping before GC has purged the rows is harmless: retired
    spaces are never searched (active_spaces filters on status), and GC deletes through the PK.
    """
    if not conn.autocommit:
        raise ValueError("drop_retired_space_index needs an autocommit connection (CONCURRENTLY)")
    row = conn.execute(
        """
        SELECT s.status = 'retired'
               AND NOT EXISTS (SELECT 1 FROM rf.collections c
                                WHERE c.active_space_id = s.id OR c.building_space_id = s.id)
          FROM rf.embedding_spaces s WHERE s.id = %s
        """,
        (int(space_id),),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown embedding space {space_id}")
    if not row[0]:
        return False
    conn.execute(
        sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS rf.{}").format(
            sql.Identifier(space_index_name(space_id))
        )
    )
    return True
