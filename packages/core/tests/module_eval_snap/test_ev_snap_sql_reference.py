"""The SQL digests and ids of migration 0012 against their Python definitions, on random data
(hypothesis): `rf._eval_snapshot_digests` == services.eval_snapshot.digests, and
`rf._uuid5` == uuid.uuid5 (so snapshot ids are exactly domain.ids')."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid4, uuid5

import psycopg
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ragforge_core.domain import ids
from ragforge_core.services.eval_snapshot.digests import (
    SnapshotChunk,
    SnapshotPage,
    content_digest,
    index_digest,
)

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
DB_EXAMPLES = settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

# Postgres text cannot hold NUL; everything else (any plane, combining marks, separators) can.
pg_text = st.text(st.characters(exclude_characters="\x00", exclude_categories=["Cs"]),
                  max_size=10)  # fmt: skip
headings = st.lists(pg_text, max_size=3).map(tuple)
# few distinct texts, so duplicate (heading, text) pairs and their occurrences really happen
chunk_specs = st.lists(st.tuples(st.sampled_from([(), ("A",), ("A", "B")]) | headings,
                                 st.sampled_from(["same", "other", ""]) | pg_text),
                       max_size=5)  # fmt: skip
# Notion page ids are unique per collection (the documents key); two FILES may share a
# filename, which the digests break ties for by the page's own encoding.
notion_pages = st.lists(st.tuples(st.just("notion"), pg_text.filter(bool), chunk_specs),
                        max_size=3, unique_by=lambda page: page[1])  # fmt: skip
file_pages = st.lists(st.tuples(st.just("file"), st.sampled_from(["a.md", "b.md"]) | pg_text,
                                chunk_specs), max_size=3)  # fmt: skip
page_specs = st.tuples(notion_pages, file_pages).map(lambda both: [*both[0], *both[1]])


@DB_EXAMPLES
@given(st.uuids(), st.text(st.characters(exclude_characters="\x00",
                                         exclude_categories=["Cs"])))  # fmt: skip
def test_sql_uuid5_is_python_uuid5(superuser: Conn, namespace: UUID, name: str) -> None:
    row = superuser.execute("SELECT rf._uuid5(%s, %s)", (namespace, name)).fetchone()
    assert row == (uuid5(namespace, name),)


def test_sql_ids_are_the_domain_ids(superuser: Conn) -> None:
    tenant, collection, page = uuid4(), uuid4(), "page/é"
    doc = superuser.execute(
        "SELECT rf._uuid5(%s, concat_ws(chr(31), 'doc', %s::text, %s::text, 'upload', %s::text))",
        (ids.RAGFORGE_NAMESPACE, tenant, collection, page),
    ).fetchone()
    assert doc == (ids.document_id(tenant, collection, None, page),)
    chunk = superuser.execute(
        "SELECT rf._uuid5(%s, concat_ws(chr(31), 'chunk', %s::text, 7::text))",
        (ids.RAGFORGE_NAMESPACE, doc[0] if doc else None),
    ).fetchone()
    assert doc is not None
    assert chunk == (ids.chunk_id(UUID(str(doc[0])), 7),)


def _insert(superuser: Conn, spec: list[tuple[str, str, list[tuple[tuple[str, ...], str]]]]
            ) -> tuple[UUID, UUID, list[SnapshotPage]]:  # fmt: skip
    tenant, collection, upload = uuid4(), uuid4(), uuid4()
    ex = superuser.execute
    ex("INSERT INTO rf.tenants (id, name, kind) VALUES (%s, %s, 'personal')",
       (tenant, f"t-{tenant.hex[:6]}"))  # fmt: skip
    ex("INSERT INTO rf.collections (id, tenant_id, name, active_space_id) VALUES (%s, %s, 'c', 1)",
       (collection, tenant))  # fmt: skip
    ex("INSERT INTO rf.uploads (id, tenant_id, collection_id, blob_key, filename, sha256, "
       "size_bytes) VALUES (%s, %s, %s, %s, 'f', %s, 0)",
       (upload, tenant, collection, f"k/{upload}", hashlib.sha256(b"x").digest()))  # fmt: skip
    pages: list[SnapshotPage] = []
    for n, (kind, key, raw_chunks) in enumerate(spec):
        doc, external = uuid4(), f"ext-{n}"
        metadata = {"filename": key} if kind == "file" else {}
        ex("INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
           "external_id, title, metadata) VALUES (%s, %s, %s, %s, %s, %s, 'T', %s)",
           (doc, tenant, collection, upload, kind, key if kind == "notion" else external,
            json.dumps(metadata)))  # fmt: skip
        chunks = []
        for ordinal, (heading, text) in enumerate(raw_chunks):
            display = hashlib.sha256(text.encode()).digest()
            embed = hashlib.sha256(f"{ordinal}{text}".encode()).digest()
            ex("INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,"
               " heading_path, display_text, display_sha256, embed_sha256, token_count) "
               "VALUES (%s, %s, %s, %s, 'a', %s, %s, %s, %s, %s, 1)",
               (uuid4(), tenant, collection, doc, ordinal, list(heading), text, display,
                embed))  # fmt: skip
            chunks.append(SnapshotChunk(ordinal, heading, text, display, embed))
        pages.append(SnapshotPage(kind, key, tuple(chunks)))
    return tenant, collection, pages


@DB_EXAMPLES
@given(page_specs)
def test_sql_digests_equal_the_reference(
    superuser: Conn, spec: list[tuple[str, str, list[tuple[tuple[str, ...], str]]]]
) -> None:
    tenant, collection, pages = _insert(superuser, spec)
    row = superuser.execute(
        "SELECT * FROM rf._eval_snapshot_digests(%s, %s)", (tenant, collection)
    ).fetchone()
    assert row == (content_digest(pages), index_digest(1, pages))
