"""Slot 0003 search functions against real Postgres + pgvector (ADR-0009, ADR-0013).

Proves: the CONCURRENTLY build + checksum hook; exact vs HNSW path selection (cached stats,
index validity) with plans captured by auto_explain INSIDE the definer functions after 6+ calls
as rf_api; k results under a 1% selective filter (and that the iterative scan is what makes it
so); re-sorted HNSW output; tombstones; cross-tenant isolation of every function.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import psycopg
import pytest

from module_b.b_support import (
    add_chunk,
    add_document,
    add_embedding,
    add_space,
    as_role,
    build_index_as_owner,
    call_search,
    captured_plans,
    near,
    set_stats,
    sha,
    unit_vector,
    use_space,
)
from pgtest import connect
from ragforge_core.adapters.db.search import vector_literal
from ragforge_core.config import Settings
from ragforge_core.db.migrate import migration_checksums

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
DIM = 8
N_COLLECTIONS = 100
PER_COLLECTION = 100


# ---------------------------------------------------------------------- migration mechanics
def test_concurrent_index_build_kept_the_checksum_hook_working(superuser: Conn) -> None:
    row = superuser.execute(
        "SELECT sha256 FROM rf.migration_checksums WHERE revision = '0003'"
    ).fetchone()
    assert row is not None, "autocommit_block broke the on_version_apply checksum hook"
    assert row[0] == migration_checksums()["0003"]
    assert superuser.execute("SELECT version_num FROM rf.alembic_version").fetchone() is not None


def test_default_space_has_a_valid_partial_expression_hnsw_index(superuser: Conn) -> None:
    row = superuser.execute(
        """
        SELECT i.indisvalid, pg_get_indexdef(i.indexrelid), pg_get_userbyid(c.relowner)
          FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
         WHERE i.indexrelid = to_regclass('rf.embeddings_hnsw_space_1')
        """
    ).fetchone()
    assert row is not None
    valid, definition, owner = row
    assert valid is True
    assert owner == "rf_system"
    assert "USING hnsw (((embedding)::ext.vector(768)) ext.vector_cosine_ops)" in str(definition)
    assert "WHERE (space_id = 1)" in str(definition)


def test_tsv_trigger_uses_the_collection_fts_config(
    superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, english = make_tenant()
    simple = uuid4()
    superuser.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id, fts_config) "
        "VALUES (%s, %s, 'simple-coll', 1, 'simple')",
        (simple, tenant),
    )
    for coll in (english, simple):
        doc = add_document(superuser, tenant, coll, title="Running Guide")
        add_chunk(
            superuser, tenant, coll, doc, 0, "The runners were running", heading_path=["Tips"]
        )
    rows: dict[object, object] = {
        r[0]: r[1]
        for r in superuser.execute(
            "SELECT collection_id, tsv::text FROM rf.chunks WHERE collection_id = ANY(%s)",
            ([english, simple],),
        ).fetchall()
    }
    assert "'run':" in str(rows[english])
    assert "'running'" not in str(rows[english])
    assert "'running':" in str(rows[simple])  # simple config: no stemming
    assert "'tip':" in str(rows[english])
    assert "'guid':" in str(rows[english])


# ---------------------------------------------------------------------- HNSW world (module)
@dataclass(frozen=True)
class World:
    space: int
    tenant: UUID
    target: UUID  # 1% of the tenant's vectors in this space
    others: list[UUID]
    other_tenant: UUID
    other_coll: UUID
    query: list[float]


def _tenant(conn: Conn, name: str) -> UUID:
    tenant = uuid4()
    conn.execute(
        "INSERT INTO rf.tenants (id, name, kind) VALUES (%s, %s, 'personal')", (tenant, name)
    )
    return tenant


def _collections(conn: Conn, tenant: UUID, space: int, n: int) -> list[UUID]:
    colls = [uuid4() for _ in range(n)]
    for i, coll in enumerate(colls):
        conn.execute(
            "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) "
            "VALUES (%s, %s, %s, %s)",
            (coll, tenant, f"c{i}-{coll.hex[:6]}", space),
        )
    return colls


def _seed_vectors(
    conn: Conn, tenant: UUID, colls: list[UUID], space: int, with_chunks: bool
) -> None:
    """PER_COLLECTION random unit vectors per collection (server-side, seeded)."""
    conn.execute(
        """
        INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding)
        SELECT %(t)s, c.id, %(s)s, sha256(convert_to(c.id::text || ':' || i, 'UTF8')),
               (SELECT array_agg(random() - 0.5)::float4[]::ext.vector
                  FROM generate_series(1, %(dim)s) WHERE i > 0)
          FROM unnest(%(colls)s::uuid[]) AS c(id), generate_series(1, %(n)s) AS i
        """,
        {"t": tenant, "s": space, "colls": colls, "n": PER_COLLECTION, "dim": DIM},
    )
    if not with_chunks:
        return
    for coll in colls:
        doc = add_document(conn, tenant, coll, title=f"Doc {coll.hex[:4]}")
        conn.execute(
            """
            INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,
                                   display_text, display_sha256, embed_sha256, token_count)
            SELECT gen_random_uuid(), %(t)s, %(c)s, %(d)s, 'a' || i, i, 'chunk ' || i,
                   sha256(convert_to(%(c)s::text || 'display' || i, 'UTF8')),
                   sha256(convert_to(%(c)s::text || ':' || i, 'UTF8')), 2
              FROM generate_series(1, %(n)s) AS i
            """,
            {"t": tenant, "c": coll, "d": doc, "n": PER_COLLECTION},
        )


@pytest.fixture(scope="module")
def world(settings: Settings) -> Iterator[World]:
    with connect(settings, "superuser") as su, connect(settings, "migrator") as migrator:
        su.autocommit = True
        su.execute("SELECT setseed(0.42)")
        space = add_space(su, DIM)
        tenant = _tenant(su, "hnsw-a")
        colls = _collections(su, tenant, space, N_COLLECTIONS)
        _seed_vectors(su, tenant, colls[:1], space, with_chunks=True)
        _seed_vectors(su, tenant, colls[1:], space, with_chunks=False)
        other_tenant = _tenant(su, "hnsw-b")
        (other_coll,) = _collections(su, other_tenant, space, 1)
        su.execute(  # tenant B holds COPIES of tenant A's target vectors: nearest neighbours
            """
            INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding)
            SELECT %s, %s, space_id, embed_sha256, embedding FROM rf.embeddings
             WHERE collection_id = %s
            """,
            (other_tenant, other_coll, colls[0]),
        )
        _seed_vectors(su, other_tenant, [other_coll], space, with_chunks=True)
        su.execute("ANALYZE rf.embeddings")
        build_index_as_owner(migrator, space)
        yield World(
            space, tenant, colls[0], colls[1:], other_tenant, other_coll, unit_vector("q", DIM)
        )


def _api(settings: Settings, tenant: UUID | None) -> Conn:
    conn = connect(settings, "api")
    if tenant is not None:
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
    return conn


def _six_calls_plans(
    session: Conn, world: World, collections: list[UUID], threshold: int = 20000
) -> list[str]:
    with captured_plans(session) as plans:
        as_role(session, "rf_api", world.tenant)
        for _ in range(6):
            plans.clear()
            assert call_search(
                session, world.space, world.query, collections, 10, threshold=threshold
            )
        return list(plans)


# ---------------------------------------------------------------------- path selection
def test_hnsw_path_uses_the_partial_index_after_six_calls(
    world: World, superuser: Conn, b_session: Conn
) -> None:
    set_stats(superuser, world.tenant, world.target, world.space, 25_000)
    try:
        plans = "\n".join(_six_calls_plans(b_session, world, [world.target]))
    finally:
        set_stats(superuser, world.tenant, world.target, world.space, PER_COLLECTION)
    assert f"Index Scan using embeddings_hnsw_space_{world.space} on embeddings" in plans
    assert "Bitmap Heap Scan on embeddings" not in plans


def test_small_collection_uses_exact_path_bitmap_scan(
    world: World, superuser: Conn, b_session: Conn
) -> None:
    set_stats(superuser, world.tenant, world.target, world.space, 19_999)
    plans = "\n".join(_six_calls_plans(b_session, world, [world.target]))
    assert "Bitmap Heap Scan on embeddings" in plans
    assert "using embeddings_hnsw_space" not in plans


def test_invalid_index_falls_back_to_exact_path(
    world: World, superuser: Conn, b_session: Conn
) -> None:
    index = f"rf.embeddings_hnsw_space_{world.space}"
    set_stats(superuser, world.tenant, world.target, world.space, 25_000)
    superuser.execute(
        "UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (index,)
    )
    try:
        plans = "\n".join(_six_calls_plans(b_session, world, [world.target]))
    finally:
        superuser.execute(
            "UPDATE pg_index SET indisvalid = true WHERE indexrelid = %s::regclass", (index,)
        )
        set_stats(superuser, world.tenant, world.target, world.space, PER_COLLECTION)
    assert "using embeddings_hnsw_space" not in plans
    assert "Bitmap Heap Scan on embeddings" in plans


def test_space_without_index_uses_exact_path(
    superuser: Conn, make_tenant: MakeTenant, b_session: Conn
) -> None:
    tenant, _, coll = make_tenant()
    space = add_space(superuser, DIM)
    use_space(superuser, coll, space)
    doc = add_document(superuser, tenant, coll)
    _, embed = add_chunk(superuser, tenant, coll, doc, 0, "only chunk")
    add_embedding(superuser, tenant, coll, embed, unit_vector(1, DIM), space_id=space)
    set_stats(superuser, tenant, coll, space, 1_000_000)
    w = World(space, tenant, coll, [], tenant, coll, unit_vector(2, DIM))
    plans = "\n".join(_six_calls_plans(b_session, w, [coll]))
    assert "Bitmap Heap Scan on embeddings" in plans
    assert "using embeddings_hnsw_space" not in plans


# ---------------------------------------------------------------------- HNSW behaviour
def test_k_results_under_a_one_percent_selective_filter(world: World, settings: Settings) -> None:
    with _api(settings, world.tenant) as api:
        hnsw = call_search(api, world.space, world.query, [world.target], 10, threshold=0)
        exact = call_search(api, world.space, world.query, [world.target], 10, threshold=10**12)
    assert len(hnsw) == 10, "iterative scan must keep scanning until k rows pass the filter"
    assert [d for _, d in hnsw] == sorted(d for _, d in hnsw), "re-sort of relaxed_order output"
    recall = len({c for c, _ in hnsw} & {c for c, _ in exact}) / 10
    assert recall >= 0.8


def test_without_iterative_scan_the_same_filter_under_returns(
    world: World, b_session: Conn
) -> None:
    """Contrast: the setting the function pins is what makes the previous test pass."""
    b_session.execute("SET LOCAL hnsw.iterative_scan = off")
    b_session.execute("SET LOCAL hnsw.ef_search = 100")
    b_session.execute("SET LOCAL enable_seqscan = off")
    b_session.execute("SET LOCAL enable_bitmapscan = off")
    b_session.execute("SET LOCAL enable_sort = off")
    rows = b_session.execute(
        f"""
        SELECT 1 FROM rf.embeddings e
         WHERE e.space_id = {world.space} AND e.tenant_id = %s AND e.collection_id = %s
         ORDER BY (e.embedding::ext.vector({DIM})) OPERATOR(ext.<=>) %s::ext.vector({DIM})
         LIMIT 10
        """,
        (world.tenant, world.target, vector_literal(world.query)),
    ).fetchall()
    assert len(rows) < 10


def test_hnsw_and_exact_results_are_scoped_to_the_tenant(world: World, settings: Settings) -> None:
    """Tenant B holds exact copies of tenant A's vectors (equally near) plus its own chunks."""
    with _api(settings, world.other_tenant) as api:
        for threshold in (0, 10**12):  # HNSW, exact
            assert (
                call_search(api, world.space, world.query, [world.target], 10, threshold=threshold)
                == []
            )
            own = call_search(api, world.space, world.query, [world.target, world.other_coll], 10,
                              threshold=threshold)  # fmt: skip
            owners = api.execute(
                "SELECT DISTINCT tenant_id FROM rf.chunks WHERE id = ANY(%s)",
                ([c for c, _ in own],),
            ).fetchall()
            assert len(own) == 10
            assert owners == [(world.other_tenant,)]
    with _api(settings, None) as api:  # no tenant set: fail closed
        assert call_search(api, world.space, world.query, [world.target], 10, threshold=0) == []


def test_internal_helpers_are_not_executable_by_app_roles(settings: Settings, world: World) -> None:
    query = vector_literal(world.query)
    calls: list[tuple[str, tuple[object, ...]]] = [
        (
            "SELECT * FROM rf.vector_candidates_hnsw(%s, %s::ext.vector, %s::uuid[], 10)",
            (world.space, query, [world.target]),
        ),
        ("SELECT * FROM rf.vector_space_operator(%s, %s::ext.vector, 10)", (world.space, query)),
    ]
    for role in ("api", "worker"):
        for statement, params in calls:
            with (
                connect(settings, role) as conn,
                pytest.raises(psycopg.errors.InsufficientPrivilege),
            ):
                conn.execute(statement, params)
        with connect(settings, role) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT rf.vector_sql('x', 1, '<=>', 1)")
    with connect(settings, "api") as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("SELECT rf.gc_embeddings()")


def test_argument_validation(world: World, settings: Settings) -> None:
    with _api(settings, world.tenant) as api, pytest.raises(psycopg.errors.InvalidParameterValue):
        call_search(api, world.space, [0.1] * (DIM + 1), [world.target], 10)
    with _api(settings, world.tenant) as api, pytest.raises(psycopg.errors.InvalidParameterValue):
        call_search(api, world.space, world.query, [world.target], 0)
    with _api(settings, world.tenant) as api, pytest.raises(psycopg.errors.InvalidParameterValue):
        call_search(api, 999_999, world.query, [world.target], 5)


# ---------------------------------------------------------------------- dead vectors
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_tombstoned_and_orphaned_vectors_never_crowd_out_live_hits(
    superuser: Conn, make_tenant: MakeTenant, b_migrator: Conn, b_session: Conn,
    settings: Settings,
) -> None:  # fmt: skip
    """Review finding: the k*2 nearest embeddings were taken BEFORE checking that a live chunk
    references them. Here the 60 nearest vectors belong to tombstoned documents (an archived
    handbook) or to no chunk at all (an edited chunk's old version, inside the GC grace), and
    both paths must still return k live hits, the exact path the TRUE live top k."""
    tenant, _, coll = make_tenant()
    space = add_space(superuser, DIM)
    use_space(superuser, coll, space)
    query = unit_vector("dead-q", DIM)
    live: dict[UUID, list[float]] = {}
    for i in range(50):
        doc = add_document(superuser, tenant, coll, title=f"Live {i}")
        chunk, embed = add_chunk(superuser, tenant, coll, doc, 0, f"live {i}")
        live[chunk] = unit_vector(f"live-{i}", DIM)
        add_embedding(superuser, tenant, coll, embed, live[chunk], space_id=space)
    for i in range(30):
        doc = add_document(superuser, tenant, coll, title=f"Archived {i}", deleted=True)
        _, embed = add_chunk(superuser, tenant, coll, doc, 0, f"archived {i}")
        add_embedding(superuser, tenant, coll, embed, near(query, f"t{i}", 0.01), space_id=space)
    for i in range(30):
        add_embedding(
            superuser, tenant, coll, sha(f"orphan-{i}"), near(query, f"o{i}", 0.01), space_id=space
        )
    build_index_as_owner(b_migrator, space)
    expected = sorted(live, key=lambda c: (_cosine(live[c], query), c))[:10]
    with _api(settings, tenant) as api:
        exact = call_search(api, space, query, [coll], 10, threshold=10**12)
        hnsw = call_search(api, space, query, [coll], 10, threshold=0)
    assert [c for c, _ in exact] == expected
    assert len(hnsw) == 10
    assert {c for c, _ in hnsw} <= set(live)
    assert len({c for c, _ in hnsw} & set(expected)) >= 8
    with captured_plans(b_session) as plans:  # prove the HNSW result came from the index
        as_role(b_session, "rf_api", tenant)
        call_search(b_session, space, query, [coll], 10, threshold=0)
    assert f"Index Scan using embeddings_hnsw_space_{space} on embeddings" in "\n".join(plans)


# ---------------------------------------------------------------------- small-world semantics
def test_exact_function_isolation_tombstones_and_shared_embeddings(
    superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    tenant, _, coll = make_tenant()
    intruder, _, intruder_coll = make_tenant()
    live = add_document(superuser, tenant, coll, title="Live")
    dead = add_document(superuser, tenant, coll, title="Dead", deleted=True)
    base = unit_vector("base")
    c1, e1 = add_chunk(superuser, tenant, coll, live, 0, "first", embed_key="shared")
    c2, _ = add_chunk(superuser, tenant, coll, live, 1, "second copy", embed_key="shared")
    _, e3 = add_chunk(superuser, tenant, coll, dead, 0, "tombstoned")
    add_embedding(superuser, tenant, coll, e1, near(base, 1))
    add_embedding(superuser, tenant, coll, e3, near(base, 2, noise=0.0))  # the closest vector
    statement = "SELECT chunk_id FROM rf.search_vectors_exact(1, %s::ext.vector, %s::uuid[], 5)"
    with _api(settings, tenant) as api:
        rows = {r[0] for r in api.execute(statement, (vector_literal(base), [coll])).fetchall()}
    assert rows == {c1, c2}, "both chunks of one embedding; tombstoned document excluded"
    with _api(settings, intruder) as api:
        assert (
            api.execute(statement, (vector_literal(base), [coll, intruder_coll])).fetchall() == []
        )


def test_search_keywords_per_fts_config_and_isolation(
    superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    tenant, _, english = make_tenant()
    other, _, other_coll = make_tenant()
    simple = uuid4()
    superuser.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id, fts_config) "
        "VALUES (%s, %s, 'kw-simple', 1, 'simple')",
        (simple, tenant),
    )
    d_en = add_document(superuser, tenant, english)
    d_si = add_document(superuser, tenant, simple)
    d_dead = add_document(superuser, tenant, english, deleted=True)
    en, _ = add_chunk(superuser, tenant, english, d_en, 0, "the deployment runs nightly")
    si_hit, _ = add_chunk(superuser, tenant, simple, d_si, 0, "cron running nightly")
    add_chunk(
        superuser, tenant, simple, d_si, 1, "the deployment runs nightly"
    )  # 'runs' != 'running'
    add_chunk(superuser, tenant, english, d_dead, 0, "running deleted page")
    d_other = add_document(superuser, other, other_coll)
    add_chunk(superuser, other, other_coll, d_other, 0, "running for the other tenant")
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    with _api(settings, tenant) as api:
        hits = {
            r[0]
            for r in api.execute(statement, ("running", [english, simple, other_coll])).fetchall()
        }
        assert api.execute(statement, ("the and of", [english])).fetchall() == []  # stopwords
        assert api.execute(statement, ("   ", [english])).fetchall() == []
    assert hits == {en, si_hit}
    with _api(settings, other) as api:
        assert api.execute(statement, ("nightly", [english, simple])).fetchall() == []
    with _api(settings, None) as api:
        assert api.execute(statement, ("nightly", [english])).fetchall() == []


def test_search_keywords_matches_any_term_ranked_by_coverage(
    superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    """0010: a question carries filler words its answer does not. The AND form missed the
    Consistent Hashing vnode chunk (no "use") and returned a chunk that had every word."""
    tenant, _, english = make_tenant()
    doc = add_document(superuser, tenant, english)
    best, _ = add_chunk(
        superuser, tenant, english, doc, 0, "virtual nodes spread consistent hashing"
    )
    partial, _ = add_chunk(superuser, tenant, english, doc, 1, "nodes join the ring")
    add_chunk(superuser, tenant, english, doc, 2, "unrelated gardening text")
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    with _api(settings, tenant) as api:
        hits = [
            r[0]
            for r in api.execute(
                statement, ("Why use virtual nodes in consistent hashing?", [english])
            )
        ]
        # Websearch operators are plain words now: no negation, no injected tsquery syntax.
        for odd in ("-virtual nodes", '"virtual" | !nodes', "o'neil \\ & nodes", "nodes:*"):
            assert partial in {r[0] for r in api.execute(statement, (odd, [english]))}, odd
    assert hits[:2] == [best, partial]


def test_search_keywords_ranks_distinct_terms_above_repetition(
    superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    """0011 (review P2): by ts_rank_cd alone a chunk repeating one term outranked one covering
    every term once (0.80 vs 0.40, measured). Distinct terms matched now come first."""
    tenant, _, english = make_tenant()
    doc = add_document(superuser, tenant, english)
    repeat, _ = add_chunk(
        superuser, tenant, english, doc, 0, "node node node node node node node node topology"
    )
    covers, _ = add_chunk(
        superuser, tenant, english, doc, 1, "consistent hashing places virtual nodes on a ring"
    )
    two, _ = add_chunk(superuser, tenant, english, doc, 2, "virtual node pools")
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    with _api(settings, tenant) as api:
        q = "Why does consistent hashing use virtual nodes?"
        assert [r[0] for r in api.execute(statement, (q, [english]))] == [covers, two, repeat]


def test_search_keywords_caps_the_or_at_32_terms(
    superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    tenant, _, english = make_tenant()
    doc = add_document(superuser, tenant, english)
    add_chunk(superuser, tenant, english, doc, 0, "zebracorn")
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    filler = " ".join(f"q{i:02d}z" for i in range(32))  # 32 distinct, unstemmed lexemes
    with _api(settings, tenant) as api:
        assert api.execute(statement, (f"zebracorn {filler}", [english])).fetchall() != []
        assert api.execute(statement, (f"{filler} zebracorn", [english])).fetchall() == []


# ---------------------------------------------------------------------- spaces vs collections
def test_definers_serve_a_collection_only_from_its_active_space(
    superuser: Conn, make_tenant: MakeTenant, b_migrator: Conn, b_session: Conn,
    settings: Settings,
) -> None:  # fmt: skip
    """Defence in depth for the P1 finding, at the definer level (a direct rf_api call cannot
    bypass the retriever's per-space split): `flipped` left its nearest vectors in `space`,
    `building` is backfilling into it. Neither is served from `space`, on either path, and
    their stats do not push `stays` onto the HNSW path."""
    tenant, _, stays = make_tenant()
    other = add_space(superuser, DIM)
    space = add_space(superuser, DIM)
    use_space(superuser, stays, space)
    flipped, building = uuid4(), uuid4()
    for coll, name in ((flipped, "flipped"), (building, "building")):
        superuser.execute(
            "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) "
            "VALUES (%s, %s, %s, %s)",
            (coll, tenant, name, other),
        )
    superuser.execute(
        "UPDATE rf.collections SET building_space_id = %s WHERE id = %s", (space, building)
    )
    query = unit_vector("spaces-q", DIM)
    served: set[UUID] = set()
    for i in range(20):
        doc = add_document(superuser, tenant, stays, title=f"Stays {i}")
        chunk, embed = add_chunk(superuser, tenant, stays, doc, 0, f"stays {i}")
        add_embedding(superuser, tenant, stays, embed, unit_vector(f"s{i}", DIM), space_id=space)
        served.add(chunk)
    for coll in (flipped, building):
        for i in range(20):
            doc = add_document(superuser, tenant, coll, title=f"{coll} {i}")
            _, embed = add_chunk(superuser, tenant, coll, doc, 0, f"near {i}")
            add_embedding(superuser, tenant, coll, embed, near(query, f"{coll}{i}", 0.01),
                          space_id=space)  # fmt: skip
        set_stats(superuser, tenant, coll, space, 1_000_000)
    set_stats(superuser, tenant, stays, space, 20)
    build_index_as_owner(b_migrator, space)
    everything = [stays, flipped, building]
    with _api(settings, tenant) as api:
        for threshold in (0, 20_000):  # HNSW, exact
            hits = call_search(api, space, query, everything, 10, threshold=threshold)
            assert len(hits) == 10
            assert {c for c, _ in hits} <= served
        exact = api.execute(
            "SELECT chunk_id FROM rf.search_vectors_exact(%s, %s::ext.vector, %s::uuid[], 10)",
            (space, vector_literal(query), [flipped, building]),
        ).fetchall()
        assert exact == []
    w = World(space, tenant, stays, [], tenant, stays, query)
    plans = "\n".join(_six_calls_plans(b_session, w, everything))
    assert "Bitmap Heap Scan on embeddings" in plans, "only stays' 20 vectors are counted"
    assert "using embeddings_hnsw_space" not in plans


def test_caller_threshold_is_clamped_so_rf_api_cannot_force_a_huge_exact_scan(
    world: World, superuser: Conn, b_session: Conn
) -> None:
    """Review finding: p_exact_threshold is caller-controlled on a function granted to rf_api.
    It is clamped to 100000, so a collection above that always takes the HNSW path."""
    set_stats(superuser, world.tenant, world.target, world.space, 150_000)
    try:
        plans = "\n".join(_six_calls_plans(b_session, world, [world.target], threshold=10**12))
    finally:
        set_stats(superuser, world.tenant, world.target, world.space, PER_COLLECTION)
    assert f"Index Scan using embeddings_hnsw_space_{world.space} on embeddings" in plans
    assert "Bitmap Heap Scan on embeddings" not in plans


# ---------------------------------------------------------------------- keyword plans
KEYWORD_ROWS = 20_000


@pytest.fixture(scope="module")
def keyword_world(settings: Settings) -> Iterator[tuple[UUID, UUID, set[UUID]]]:
    """One collection with KEYWORD_ROWS chunks over 200 documents; five mention a rare term."""
    with connect(settings, "superuser") as su:
        su.autocommit = True
        tenant = _tenant(su, "kw-large")
        (coll,) = _collections(su, tenant, 1, 1)
        docs = [add_document(su, tenant, coll, title=f"Manual {i}") for i in range(200)]
        su.execute(
            """
            INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,
                                   display_text, display_sha256, embed_sha256, token_count)
            SELECT gen_random_uuid(), %(t)s, %(c)s, (%(d)s::uuid[])[1 + i %% 200], 'k' || i, i,
                   CASE WHEN i %% 4000 = 0 THEN 'configure the zebracorn relay ' || i
                        ELSE 'routine maintenance notes for server ' || (i %% 50) END,
                   sha256(convert_to('kwd' || i, 'UTF8')), sha256(convert_to('kwe' || i, 'UTF8')), 5
              FROM generate_series(1, %(n)s) AS i
            """,
            {"t": tenant, "c": coll, "d": docs, "n": KEYWORD_ROWS},
        )
        # VACUUM, not just ANALYZE: it merges GIN's fastupdate pending list (the steady state
        # autovacuum keeps). A fresh 20k-row pending list makes every GIN probe look as costly
        # as a seq scan, which is a property of the bulk load, not of the query.
        su.execute("VACUUM ANALYZE rf.documents")
        su.execute("VACUUM ANALYZE rf.chunks")
        rare = {
            UUID(str(r[0]))
            for r in su.execute(
                "SELECT id FROM rf.chunks WHERE collection_id = %s AND display_text LIKE %s",
                (coll, "%zebracorn%"),
            ).fetchall()
        }
        yield tenant, coll, rare


def test_keyword_search_uses_the_gin_index_on_a_large_collection(
    keyword_world: tuple[UUID, UUID, set[UUID]], b_session: Conn
) -> None:
    """Review finding: nothing proved GIN use. The tsquery is now a parameter of a per-config
    EXECUTE (custom plan), so a selective term is found through chunks_tsv_gin, not by walking
    all 20k rows of the collection."""
    tenant, coll, rare = keyword_world
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    with captured_plans(b_session) as plans:
        as_role(b_session, "rf_api", tenant)
        for _ in range(6):
            plans.clear()
            hits = {r[0] for r in b_session.execute(statement, ("zebracorn", [coll])).fetchall()}
            assert hits == rare
        captured = "\n".join(plans)
    assert "Bitmap Index Scan on chunks_tsv_gin" in captured


def test_keyword_query_text_is_capped(
    keyword_world: tuple[UUID, UUID, set[UUID]], settings: Settings
) -> None:
    """A 100k-character query costs no more than its first 1000 characters: the rare term
    placed past the cap is not searched."""
    tenant, coll, _ = keyword_world
    statement = "SELECT chunk_id FROM rf.search_keywords(%s, %s::uuid[], 10)"
    with _api(settings, tenant) as api:
        assert api.execute(statement, ("x" * 1001 + " zebracorn", [coll])).fetchall() == []
        padded = "zebracorn " + "or qqq " * 15_000  # OR keeps the rare term matching
        assert len(api.execute(statement, (padded, [coll])).fetchall()) == 5
