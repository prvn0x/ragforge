"""Module B test helpers (seeding as superuser, plan capture). Imported by module_b tests only."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg import sql

from ragforge_core.adapters.db.search import build_space_index, vector_literal
from ragforge_core.domain import ids

Conn = psycopg.Connection[tuple[object, ...]]
DEFAULT_SPACE = 1
DEFAULT_DIM = 768


def sha(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def unit_vector(seed: int | str, dim: int = DEFAULT_DIM) -> list[float]:
    rng = random.Random(str(seed))  # noqa: S311 - deterministic test data
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def near(base: Sequence[float], seed: int | str, noise: float = 0.05) -> list[float]:
    """A vector close to `base` (cosine distance small)."""
    rng = random.Random(str(seed))  # noqa: S311
    return [b + rng.uniform(-noise, noise) for b in base]


def add_space(
    superuser: Conn, dim: int, *, distance: str = "cosine", status: str = "active"
) -> int:
    """A fresh embedding space (unique model name) so tests never share vectors or indexes."""
    row = superuser.execute(
        "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status) "
        "VALUES ('ollama', %s, %s, %s, %s) RETURNING id",
        (f"test-{uuid4().hex[:10]}", dim, distance, status),
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


def build_index_as_owner(migrator: Conn, space_id: int) -> None:
    """The migrator-run build step, exactly as production would run it (owner = rf_system)."""
    migrator.autocommit = True
    migrator.execute("SET ROLE rf_system")
    assert build_space_index(migrator, space_id) is True


def use_space(superuser: Conn, collection: UUID, space_id: int) -> None:
    superuser.execute(
        "UPDATE rf.collections SET active_space_id = %s WHERE id = %s", (space_id, collection)
    )


def add_document(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    *,
    title: str = "Doc",
    url: str | None = None,
    deleted: bool = False,
) -> UUID:
    upload = uuid4()
    superuser.execute(
        "INSERT INTO rf.uploads (id, tenant_id, collection_id, blob_key, filename, sha256, "
        "size_bytes) VALUES (%s, %s, %s, %s, 'f.md', %s, 1)",
        (upload, tenant, collection, f"blob-{upload}", sha(str(upload))),
    )
    external_id = f"file-{upload}"
    doc = ids.document_id(tenant, collection, None, external_id)
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
        "external_id, external_url, title, fetched_at, deleted_at) "
        "VALUES (%s, %s, %s, %s, 'file', %s, %s, %s, now(), %s)",
        (doc, tenant, collection, upload, external_id, url, title, now() if deleted else None),
    )
    return doc


def add_chunk(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    document: UUID,
    ordinal: int,
    display_text: str,
    *,
    embed_key: str | None = None,
    heading_path: Sequence[str] = (),
) -> tuple[UUID, bytes]:
    """Returns (chunk_id, embed_sha256). `embed_key` lets several chunks share one embedding."""
    embed_sha = sha(embed_key or f"{document}:{ordinal}:{display_text}")
    chunk = ids.chunk_id(document, ordinal)
    superuser.execute(
        "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal, "
        "heading_path, display_text, display_sha256, embed_sha256, token_count) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)",
        (
            chunk,
            tenant,
            collection,
            document,
            f"a{ordinal}",
            ordinal,
            list(heading_path),
            display_text,
            sha(display_text),
            embed_sha,
        ),
    )
    return chunk, embed_sha


def add_embedding(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    embed_sha: bytes,
    vector: Sequence[float],
    *,
    space_id: int = DEFAULT_SPACE,
    touched_hours_ago: float | None = 0.0,
    created_hours_ago: float = 0.0,
) -> None:
    superuser.execute(
        "INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding, "
        "created_at) VALUES (%s, %s, %s, %s, %s::ext.vector, now() - make_interval(secs => %s))",
        (tenant, collection, space_id, embed_sha, vector_literal(vector), created_hours_ago * 3600),
    )
    if touched_hours_ago is not None:
        superuser.execute(
            "INSERT INTO rf.embedding_touches (tenant_id, collection_id, space_id, embed_sha256, "
            "touched_at) VALUES (%s, %s, %s, %s, now() - make_interval(secs => %s)) "
            "ON CONFLICT (tenant_id, collection_id, space_id, embed_sha256) "
            "DO UPDATE SET touched_at = excluded.touched_at",
            (tenant, collection, space_id, embed_sha, touched_hours_ago * 3600),
        )


def set_stats(superuser: Conn, tenant: UUID, collection: UUID, space_id: int, count: int) -> None:
    superuser.execute(
        "INSERT INTO rf.collection_space_stats (tenant_id, collection_id, space_id, vector_count) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (collection_id, space_id) "
        "DO UPDATE SET vector_count = excluded.vector_count",
        (tenant, collection, space_id, count),
    )


def now() -> datetime:
    return datetime.now(UTC)


def as_role(conn: Conn, role: str, tenant: UUID | None) -> None:
    """Switch a superuser session to an app role (RLS applies) inside an open transaction."""
    conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
    if tenant is not None:
        conn.execute("SELECT set_config('app.tenant_id', %s, false)", (str(tenant),))


@contextmanager
def captured_plans(conn: Conn) -> Iterator[list[str]]:
    """Capture auto_explain output of every statement, including those nested inside the
    SECURITY DEFINER functions (log_nested_statements), as client NOTICEs. `conn` must be a
    superuser session (auto_explain settings are superuser-only); callers then SET ROLE."""
    plans: list[str] = []

    def _collect(diag: psycopg.errors.Diagnostic) -> None:
        if diag.message_primary and "Query Text" in diag.message_primary:
            plans.append(diag.message_primary)

    conn.execute("LOAD 'auto_explain'")
    for setting, value in (
        ("auto_explain.log_min_duration", "0"),
        ("auto_explain.log_nested_statements", "on"),
        ("auto_explain.log_analyze", "on"),
        ("auto_explain.log_level", "notice"),
        ("client_min_messages", "notice"),
    ):
        conn.execute(
            sql.SQL("SET {} = {}").format(sql.Identifier(*setting.split(".")), sql.Literal(value))
        )
    conn.add_notice_handler(_collect)
    try:
        yield plans
    finally:
        conn.remove_notice_handler(_collect)


def call_search(
    conn: Conn,
    space_id: int,
    query: Sequence[float],
    collections: Sequence[UUID],
    k: int,
    **kw: Any,
) -> list[tuple[UUID, float]]:
    threshold = kw.get("threshold", 20000)
    rows = conn.execute(
        "SELECT chunk_id, distance FROM rf.search_vectors(%s, %s::ext.vector, %s::uuid[], %s, %s)",
        (space_id, vector_literal(query), list(collections), k, threshold),
    ).fetchall()
    return [(UUID(str(r[0])), float(str(r[1]))) for r in rows]
