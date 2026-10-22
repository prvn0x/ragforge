"""Slot 0004 definer functions against a real Postgres (ADR-0007, ADR-0013):
take_tokens (global token bucket), route_notion_webhook, purge_tombstones.

Each function gets: behaviour, its grant boundary (who may / may not EXECUTE, no PUBLIC) and a
cross-tenant test proving it reaches exactly the rows it is named for, while the same role
going around it is still confined by RLS."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
import respx
from psycopg import errors
from sqlalchemy.ext.asyncio import AsyncEngine

from pgtest import connect
from ragforge_core.adapters.connectors.notion import NotionClient
from ragforge_core.adapters.connectors.notion.client import NOTION_API_BASE
from ragforge_core.adapters.connectors.notion.db import purge_tombstones, route_notion_webhook
from ragforge_core.adapters.connectors.ratelimit import (
    PostgresRateLimiter,
    integration_key,
    limiter_key,
    tokens_for_delay,
)
from ragforge_core.config import Settings
from ragforge_core.db.engine import make_engine, system_transaction
from ragforge_core.domain.errors import UpstreamRateLimited

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]


def _key() -> bytes:
    return hashlib.sha256(os.urandom(16)).digest()


@pytest.fixture
def worker(settings: Settings) -> Iterator[Conn]:
    with connect(settings, "worker") as conn:
        conn.autocommit = True  # each call is its own short transaction, like the adapter
        yield conn


@pytest.fixture
def worker2(settings: Settings) -> Iterator[Conn]:
    with connect(settings, "worker") as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture
def api(settings: Settings) -> Iterator[Conn]:
    with connect(settings, "api") as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture
async def worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "worker")
    yield engine
    await engine.dispose()


def _int_result(conn: Conn, sql: str, params: tuple[object, ...]) -> int:
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, int)
    return value


def _take(conn: Conn, key: bytes, n: int = 1) -> int:
    return _int_result(conn, "SELECT rf.take_tokens(%s, %s)", (key, n))


def _penalize(conn: Conn, key: bytes, n: int) -> int:
    return _int_result(conn, "SELECT rf.penalize_tokens(%s, %s)", (key, n))


# ---------------------------------------------------------------------- take_tokens
def test_take_tokens_allows_a_burst_of_3_then_charges_debt(worker: Conn) -> None:
    key = _key()
    waits = [_take(worker, key) for _ in range(5)]
    assert waits[:3] == [0, 0, 0]
    assert 200 <= waits[3] <= 340  # 1 token of debt at 3/s ~= 333 ms
    assert 500 <= waits[4] <= 670
    assert waits[3] < waits[4]


def test_take_tokens_refills_with_wall_clock_time(worker: Conn) -> None:
    key = _key()
    for _ in range(3):
        _take(worker, key)
    time.sleep(0.45)  # ~1.35 tokens refill
    assert _take(worker, key) == 0
    assert _take(worker, key) > 0


def test_take_tokens_refill_is_capped_at_capacity(worker: Conn) -> None:
    key = _key()
    _take(worker, key)
    time.sleep(1.2)  # would be 3.6 + 2 tokens uncapped
    assert [_take(worker, key) for _ in range(3)] == [0, 0, 0]
    assert _take(worker, key) > 0


def test_take_tokens_is_global_across_connections(worker: Conn, worker2: Conn) -> None:
    """Two workers interleaving share ONE budget: per-connection buckets would give 0 waits
    for the first 3 calls on EACH connection."""
    key = _key()
    waits = [_take(conn, key) for conn in (worker, worker2) * 3]
    assert sorted(waits)[:3] == [0, 0, 0]
    assert waits.count(0) == 3
    assert 700 <= max(waits) <= 1050  # ~3 tokens of debt, minus refill between round trips


def test_keys_are_independent(worker: Conn) -> None:
    drained, other = _key(), _key()
    _take(worker, drained, 30)
    assert _take(worker, other) == 0


def test_retry_after_drain_makes_every_worker_wait(worker: Conn, worker2: Conn) -> None:
    key = _key()
    _take(worker, key, tokens_for_delay(2.0))  # worker 1 got a 429 with Retry-After: 2
    assert _take(worker2, key) >= 1000  # worker 2 waits it out too (4 tokens of debt)


@pytest.mark.parametrize(("key", "n"), [(b"x" * 31, 1), (b"x" * 32, 0), (b"x" * 32, 1801)])
def test_take_tokens_validates_input(worker: Conn, key: bytes, n: int) -> None:
    with pytest.raises(errors.InvalidParameterValue):
        _take(worker, key, n)


def test_take_tokens_grants(worker: Conn, api: Conn, superuser: Conn) -> None:
    with pytest.raises(errors.InsufficientPrivilege):
        _take(api, _key())
    with pytest.raises(errors.InsufficientPrivilege):
        worker.execute("SELECT * FROM rf.notion_rate")
    with pytest.raises(errors.InsufficientPrivilege):
        worker.execute("UPDATE rf.notion_rate SET tokens = 100")
    assert _public_can_execute(superuser, "take_tokens") is False


async def test_rate_limiter_is_global_across_concurrent_tasks_and_engines(
    settings: Settings, worker_engine: AsyncEngine
) -> None:
    """Two asyncio tasks on two separate engines (stand-ins for two worker processes) doing
    4 acquires each: 8 requests at 3/s with a burst of 3 cannot finish in under ~5/3 s."""
    other_engine = make_engine(settings, "worker")
    token = f"secret-{uuid4()}"
    try:
        limiters = [PostgresRateLimiter(worker_engine), PostgresRateLimiter(other_engine)]

        async def burst(limiter: PostgresRateLimiter) -> None:
            for _ in range(4):
                await limiter.acquire(token)

        start = time.monotonic()
        await asyncio.gather(*(burst(lim) for lim in limiters))
        elapsed = time.monotonic() - start
    finally:
        await other_engine.dispose()
    assert elapsed >= 1.55, f"8 requests finished in {elapsed:.2f}s: budget not shared"
    assert elapsed < 6


async def test_rate_limiter_splits_huge_drains(worker_engine: AsyncEngine) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    limiter = PostgresRateLimiter(worker_engine, sleep=fake_sleep, max_wait_s=None)
    await limiter.acquire(f"secret-{uuid4()}", 2000)  # > per-call cap of 1800
    assert len(slept) == 1
    assert 660 <= slept[0] <= 670
    with pytest.raises(ValueError, match="tokens"):
        await limiter.acquire("k", 0)


# ---------------------------------------------------------------------- penalize_tokens
def test_penalty_is_idempotent_per_window_not_additive(worker: Conn, worker2: Conn) -> None:
    """Three workers report the same `Retry-After: 2`: the integration waits ~2 s once, not
    ~6 s (debt is clamped to the penalty, never added to it)."""
    key = _key()
    n = tokens_for_delay(2.0)
    waits = [_penalize(conn, key, n) for conn in (worker, worker2, worker)]
    assert all(1900 <= w <= 2000 for w in waits), waits
    assert 2000 <= _take(worker2, key) <= 2400  # 2 s penalty + 1 token (~333 ms)


def test_penalty_extends_but_never_shortens_the_window(worker: Conn) -> None:
    key = _key()
    _penalize(worker, key, tokens_for_delay(10.0))
    assert _penalize(worker, key, tokens_for_delay(1.0)) >= 9900  # shorter one: no effect
    assert _penalize(worker, key, tokens_for_delay(20.0)) >= 19900  # longer one: extends


def test_penalty_on_existing_debt_does_not_forgive_it(worker: Conn) -> None:
    key = _key()
    _take(worker, key, 30)  # ~9 s of debt already queued by other callers
    assert _penalize(worker, key, tokens_for_delay(1.0)) >= 8900


@pytest.mark.parametrize(("key", "n"), [(b"x" * 31, 1), (b"x" * 32, 0), (b"x" * 32, 1801)])
def test_penalize_tokens_validates_input(worker: Conn, key: bytes, n: int) -> None:
    with pytest.raises(errors.InvalidParameterValue):
        _penalize(worker, key, n)


def test_penalize_tokens_grants(api: Conn, superuser: Conn) -> None:
    with pytest.raises(errors.InsufficientPrivilege):
        _penalize(api, _key(), 1)
    assert _public_can_execute(superuser, "penalize_tokens") is False


async def test_long_retry_after_on_one_worker_stops_every_worker(
    worker_engine: AsyncEngine, worker: Conn
) -> None:
    """Worker A gets `Retry-After: 120`: it gives its job back (no pinned slot) AND worker B,
    on another connection, must wait ~120 s before its next request (raw SQL view)."""
    token = f"secret-{uuid4()}"
    limiter = PostgresRateLimiter(worker_engine)
    limited = httpx.Response(429, json={"code": "rate_limited"}, headers={"Retry-After": "120"})
    with respx.mock(base_url=NOTION_API_BASE, assert_all_mocked=True) as router:
        route = router.get("/pages/x").mock(return_value=limited)
        async with NotionClient(token, limiter) as client:
            with pytest.raises(UpstreamRateLimited) as info:
                await client.request("GET", "/pages/x")
    assert info.value.retry_after_s == 120
    assert route.call_count == 1
    wait_b = _take(worker, integration_key(token))
    assert 119_000 <= wait_b <= 121_000


async def test_other_workers_requeue_instead_of_sleeping_through_a_long_penalty(
    settings: Settings, worker_engine: AsyncEngine
) -> None:
    """Worker A (engine 1) gets `Retry-After: 120`. Worker B (engine 2, its own limiter) must
    NOT sleep 2 minutes holding its job slot: its acquire raises UpstreamRateLimited at once,
    with the remaining wait, exactly like worker A. Short debts are still slept through."""
    token = f"secret-{uuid4()}"
    limited = httpx.Response(429, json={"code": "rate_limited"}, headers={"Retry-After": "120"})
    with respx.mock(base_url=NOTION_API_BASE, assert_all_mocked=True) as router:
        router.get("/pages/x").mock(return_value=limited)
        async with NotionClient(token, PostgresRateLimiter(worker_engine)) as client:
            with pytest.raises(UpstreamRateLimited):
                await client.request("GET", "/pages/x")

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    other_engine = make_engine(settings, "worker")
    try:
        worker_b = PostgresRateLimiter(other_engine, sleep=fake_sleep)
        with pytest.raises(UpstreamRateLimited) as info:
            await worker_b.acquire(limiter_key(token))
        assert info.value.retry_after_s is not None
        assert 119.0 <= info.value.retry_after_s <= 121.0
        assert slept == []

        fresh = PostgresRateLimiter(other_engine, sleep=fake_sleep)
        key = limiter_key(f"secret-{uuid4()}")
        for _ in range(4):  # burst of 3, then ~333 ms of debt: slept, not raised
            await fresh.acquire(key)
        assert len(slept) == 1
        assert 0 < slept[0] <= 0.34
    finally:
        await other_engine.dispose()


def test_limiter_key_and_raw_token_share_one_bucket(worker: Conn) -> None:
    """`limiter_key(token)` (what the client passes) and the token itself land on the same
    row, keyed by sha256(token) as ADR-0007 specifies."""
    token = f"secret-{uuid4()}"
    assert integration_key(limiter_key(token)) == hashlib.sha256(token.encode()).digest()
    assert integration_key("sha256:not-hex") == hashlib.sha256(b"sha256:not-hex").digest()
    for _ in range(3):
        assert _take(worker, integration_key(limiter_key(token))) == 0
    assert _take(worker, hashlib.sha256(token.encode()).digest()) > 0  # same bucket, now empty


# ---------------------------------------------------------------------- route_notion_webhook
def _connector(
    su: Conn, tenant: UUID, coll: UUID, config: dict[str, str], status: str = "active"
) -> UUID:
    row = su.execute(
        "INSERT INTO rf.connectors (tenant_id, kind, target_collection_id, display_name, config,"
        " status) VALUES (%s, 'notion', %s, 'n', %s, %s) RETURNING id",
        (tenant, coll, json.dumps(config), status),
    ).fetchone()
    assert row is not None
    return UUID(str(row[0]))


def test_route_notion_webhook_crosses_tenants_only_for_matching_active_connectors(
    superuser: Conn, api: Conn, worker: Conn, make_tenant: MakeTenant
) -> None:
    ws, bot = f"ws-{uuid4()}", f"bot-{uuid4()}"
    (ta, _, ca), (tb, _, cb), (tc, _, cc), (td, _, cd) = (make_tenant() for _ in range(4))
    a = _connector(superuser, ta, ca, {"workspace_id": ws, "integration_id": bot})
    b = _connector(superuser, tb, cb, {"workspace_id": ws, "integration_id": bot})
    _connector(superuser, tc, cc, {"workspace_id": f"other-{ws}", "integration_id": bot})
    _connector(superuser, td, cd, {"workspace_id": ws, "integration_id": bot}, status="paused")
    _connector(superuser, tc, cc, {"workspace_id": ws, "integration_id": f"other-{bot}"})

    rows = api.execute("SELECT * FROM rf.route_notion_webhook(%s, %s)", (bot, ws)).fetchall()
    assert sorted((UUID(str(t)), UUID(str(c))) for t, c in rows) == sorted([(ta, a), (tb, b)])
    # The same role going around the function sees nothing without a tenant (RLS fail-closed).
    assert api.execute("SELECT count(*) FROM rf.connectors").fetchone() == (0,)
    assert api.execute("SELECT * FROM rf.route_notion_webhook('', %s)", (ws,)).fetchall() == []
    assert api.execute("SELECT * FROM rf.route_notion_webhook(%s, NULL)", (bot,)).fetchall() == []
    with pytest.raises(errors.InsufficientPrivilege):
        worker.execute("SELECT * FROM rf.route_notion_webhook(%s, %s)", (bot, ws))
    assert _public_can_execute(superuser, "route_notion_webhook") is False


async def test_route_adapter(
    superuser: Conn, api_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    ws, bot = f"ws-{uuid4()}", f"bot-{uuid4()}"
    conn_id = _connector(superuser, tenant, coll, {"workspace_id": ws, "integration_id": bot})
    async with system_transaction(api_engine) as tx:
        assert await route_notion_webhook(tx, bot, ws) == [(tenant, conn_id)]


# ---------------------------------------------------------------------- purge_tombstones
def _document(su: Conn, tenant: UUID, coll: UUID, deleted_days_ago: float | None) -> UUID:
    connector = _connector(su, tenant, coll, {})
    doc = uuid4()
    su.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind,"
        " external_id, title, deleted_at) VALUES (%s, %s, %s, %s, 'notion', %s, 't',"
        " CASE WHEN %s::float8 IS NULL THEN NULL"
        "      ELSE now() - make_interval(secs => %s::float8 * 86400) END)",
        (doc, tenant, coll, connector, str(uuid4()), deleted_days_ago, deleted_days_ago),
    )
    sha = hashlib.sha256(doc.bytes).digest()
    su.execute(
        "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,"
        " display_text, display_sha256, embed_sha256, token_count)"
        " VALUES (%s, %s, %s, %s, 'a', 0, 'x', %s, %s, 1)",
        (uuid4(), tenant, coll, doc, sha, sha),
    )
    return doc


def _exists(su: Conn, doc: UUID) -> bool:
    row = su.execute("SELECT count(*) FROM rf.documents WHERE id = %s", (doc,)).fetchone()
    return row == (1,)


def _purge(conn: Conn, limit: int = 1000) -> int:
    return _int_result(conn, "SELECT rf.purge_tombstones(%s)", (limit,))


def _purge_all(conn: Conn) -> int:
    total = 0
    while (n := _purge(conn)) > 0:
        total += n
    return total


def test_purge_tombstones_hard_deletes_only_week_old_tombstones_across_tenants(
    superuser: Conn, worker: Conn, api: Conn, make_tenant: MakeTenant
) -> None:
    _purge_all(worker)  # start from a clean slate (shared session database)
    (ta, _, ca), (tb, _, cb), (tc, _, cc) = (make_tenant() for _ in range(3))
    old_a = _document(superuser, ta, ca, 8)
    old_b = _document(superuser, tb, cb, 7.01)
    recent = _document(superuser, tc, cc, 6.9)
    live = _document(superuser, tc, cc, None)

    # Without the function, the worker cannot even see these rows (no tenant set).
    assert worker.execute("DELETE FROM rf.documents WHERE id = %s", (old_a,)).rowcount == 0

    assert _purge(worker) == 2
    assert not _exists(superuser, old_a)
    assert not _exists(superuser, old_b)
    assert _exists(superuser, recent)
    assert _exists(superuser, live)
    assert superuser.execute(
        "SELECT count(*) FROM rf.chunks WHERE document_id IN (%s, %s)", (old_a, old_b)
    ).fetchone() == (0,)  # chunks cascade
    with pytest.raises(errors.InsufficientPrivilege):
        _purge(api)
    assert _public_can_execute(superuser, "purge_tombstones") is False


def test_purge_tombstones_is_batched(
    superuser: Conn, worker: Conn, make_tenant: MakeTenant
) -> None:
    _purge_all(worker)
    tenant, _, coll = make_tenant()
    for _ in range(3):
        _document(superuser, tenant, coll, 10)
    assert _purge(worker, 2) == 2
    assert _purge(worker, 2) == 1
    assert _purge(worker, 2) == 0
    for bad in (0, 10001):
        with pytest.raises(errors.InvalidParameterValue):
            _purge(worker, bad)


def test_purge_skips_a_document_being_resurrected_concurrently(
    settings: Settings, superuser: Conn, worker: Conn, make_tenant: MakeTenant
) -> None:
    """A sync holding the row lock (about to clear deleted_at) is neither blocked nor raced:
    purge skips the row, and the resurrected document survives."""
    _purge_all(worker)
    tenant, _, coll = make_tenant()
    doc = _document(superuser, tenant, coll, 30)
    with connect(settings, "worker") as sync:
        sync.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
        sync.execute("SELECT 1 FROM rf.documents WHERE id = %s FOR UPDATE", (doc,))
        worker.execute("SET statement_timeout = '5s'")
        assert _purge(worker) == 0  # SKIP LOCKED: returns at once
        sync.execute("UPDATE rf.documents SET deleted_at = NULL WHERE id = %s", (doc,))
        sync.commit()
    assert _purge(worker) == 0
    assert _exists(superuser, doc)


async def test_purge_adapter(
    superuser: Conn, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    async with system_transaction(worker_engine) as tx:
        while await purge_tombstones(tx) > 0:
            pass
    tenant, _, coll = make_tenant()
    doc = _document(superuser, tenant, coll, 9)
    async with system_transaction(worker_engine) as tx:
        assert await purge_tombstones(tx, limit=10) == 1
    assert not _exists(superuser, doc)


# ---------------------------------------------------------------------- helpers
def _public_can_execute(su: Conn, name: str) -> bool:
    rows = su.execute(
        """
        SELECT p.proacl::text FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'rf' AND p.proname = %s
        """,
        (name,),
    ).fetchall()
    assert rows, f"rf.{name} does not exist"
    return any(acl is None or "{=" in str(acl) or ",=" in str(acl) for (acl,) in rows)
