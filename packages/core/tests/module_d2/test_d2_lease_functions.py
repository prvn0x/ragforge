"""Slot 0005 definer functions: assert_run_lease, heartbeat_run, chat_active (ADR-0013).

Each function gets a cross-tenant test, a privilege test, and a test of the guarantee it exists
for (fencing, no revival of expired leases, boolean-only cross-tenant visibility).
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import errors

from pgtest import connect
from ragforge_core.config import Settings

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeConversation = Callable[[], tuple[UUID, UUID]]

LEASE_LOST = "RFL01"
DEFINER_FUNCTIONS = {
    "assert_run_lease(uuid,uuid)": {"rf_api"},
    "heartbeat_run(uuid,uuid,integer)": {"rf_api"},
    "chat_active()": {"rf_worker"},
}


def _hold(superuser: Conn, conversation: UUID, *, seconds: int = 60) -> UUID:
    run = uuid4()
    superuser.execute(
        "UPDATE rf.conversations SET active_run_id = %s, "
        "run_expires_at = clock_timestamp() + make_interval(secs => %s) WHERE id = %s",
        (run, seconds, conversation),
    )
    return run


def _expire(superuser: Conn, conversation: UUID) -> None:
    superuser.execute(
        "UPDATE rf.conversations SET run_expires_at = clock_timestamp() - interval '1 second' "
        "WHERE id = %s",
        (conversation,),
    )


def _as_api(settings: Settings, tenant: UUID | None = None) -> Conn:
    conn = connect(settings, "api")
    if tenant is not None:
        conn.execute("SELECT set_config('app.tenant_id', %s, false)", (str(tenant),))
    return conn


def _assert_lease(conn: Conn, conversation: UUID, run: UUID) -> None:
    conn.execute("SELECT rf.assert_run_lease(%s, %s)", (conversation, run))


# ---------------------------------------------------------------------- privileges
@pytest.mark.parametrize(("signature", "allowed"), DEFINER_FUNCTIONS.items())
def test_execute_granted_only_to_the_calling_role(
    superuser: Conn, signature: str, allowed: set[str]
) -> None:
    for role in ("rf_api", "rf_worker", "rf_metrics", "rf_backup"):
        (can,) = superuser.execute(  # type: ignore[misc]
            "SELECT has_function_privilege(%s, %s, 'EXECUTE')", (role, f"rf.{signature}")
        ).fetchone()
        assert can is (role in allowed), f"{role} EXECUTE on {signature}"


@pytest.mark.parametrize("signature", DEFINER_FUNCTIONS)
def test_definer_is_owned_by_rf_system_with_pinned_path(superuser: Conn, signature: str) -> None:
    row = superuser.execute(
        "SELECT pg_get_userbyid(proowner), prosecdef, proconfig FROM pg_proc "
        "WHERE oid = %s::regprocedure",
        (f"rf.{signature}",),
    ).fetchone()
    assert row == ("rf_system", True, ["search_path=pg_catalog, rf, ext, pg_temp"])


# ---------------------------------------------------------------------- assert_run_lease
def test_assert_run_lease_accepts_the_live_holder(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    tenant, conversation = make_conversation()
    run = _hold(superuser, conversation)
    with _as_api(settings, tenant) as conn:
        _assert_lease(conn, conversation, run)


def test_lease_functions_fail_closed_without_a_tenant(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    """ADR-0013: an unset app.tenant_id is NULL and matches nothing. Even the live holder's own
    run id is rejected, so a path that forgets set_config fails instead of crossing tenants."""
    _tenant, conversation = make_conversation()
    run = _hold(superuser, conversation, seconds=10)
    before = superuser.execute(
        "SELECT run_expires_at FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone()
    with _as_api(settings) as conn, pytest.raises(psycopg.Error) as exc:
        _assert_lease(conn, conversation, run)
    assert exc.value.sqlstate == LEASE_LOST
    with _as_api(settings) as conn:
        assert _heartbeat(conn, conversation, run) is False
        conn.commit()
    after = superuser.execute(
        "SELECT run_expires_at FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone()
    assert before == after


@pytest.mark.parametrize("case", ["wrong_run", "expired", "released", "missing"])
def test_assert_run_lease_rejects_non_holders(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation, case: str
) -> None:
    tenant, conversation = make_conversation()
    run = _hold(superuser, conversation)
    if case == "wrong_run":
        run = uuid4()
    elif case == "expired":
        _expire(superuser, conversation)
    elif case == "released":
        superuser.execute(
            "UPDATE rf.conversations SET active_run_id = NULL, run_expires_at = NULL WHERE id = %s",
            (conversation,),
        )
    else:
        conversation = uuid4()
    with _as_api(settings, tenant) as conn, pytest.raises(psycopg.Error) as exc:
        _assert_lease(conn, conversation, run)
    assert exc.value.sqlstate == LEASE_LOST


def test_assert_run_lease_is_tenant_bound(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    """Cross-tenant: the holder's own run id is rejected when bound to another tenant, and the
    foreign caller never takes the row lock (the owner can still lock it with NOWAIT)."""
    _tenant_a, conversation = make_conversation()
    tenant_b, _ = make_conversation()
    run = _hold(superuser, conversation)
    with _as_api(settings, tenant_b) as conn:
        with pytest.raises(psycopg.Error) as exc:
            _assert_lease(conn, conversation, run)
        assert exc.value.sqlstate == LEASE_LOST
        # The failed transaction is still open here: a row lock taken before the RAISE would
        # still be held, so NOWAIT proves none was taken.
        superuser.execute(
            "SELECT 1 FROM rf.conversations WHERE id = %s FOR UPDATE NOWAIT", (conversation,)
        )
        conn.rollback()


def test_assert_run_lease_holds_the_row_lock_until_commit(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    """The fence and the write commit atomically: no takeover can interleave."""
    tenant, conversation = make_conversation()
    run = _hold(superuser, conversation)
    with _as_api(settings, tenant) as conn:
        _assert_lease(conn, conversation, run)  # transaction stays open (autocommit off)
        with pytest.raises(errors.LockNotAvailable):
            superuser.execute(
                "SELECT 1 FROM rf.conversations WHERE id = %s FOR UPDATE NOWAIT", (conversation,)
            )
        conn.commit()
    superuser.execute(
        "SELECT 1 FROM rf.conversations WHERE id = %s FOR UPDATE NOWAIT", (conversation,)
    )


# ---------------------------------------------------------------------- heartbeat_run
def _heartbeat(conn: Conn, conversation: UUID, run: UUID, ttl: int = 60) -> bool:
    row = conn.execute("SELECT rf.heartbeat_run(%s, %s, %s)", (conversation, run, ttl)).fetchone()
    assert row is not None
    return bool(row[0])


def test_heartbeat_run_extends_only_a_live_lease(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    tenant, conversation = make_conversation()
    run = _hold(superuser, conversation, seconds=10)
    with _as_api(settings, tenant) as conn:
        assert _heartbeat(conn, conversation, run, ttl=300) is True
        conn.commit()
    (remaining,) = superuser.execute(  # type: ignore[misc]
        "SELECT extract(epoch FROM run_expires_at - clock_timestamp()) FROM rf.conversations "
        "WHERE id = %s",
        (conversation,),
    ).fetchone()
    assert 250 < float(str(remaining)) <= 300
    with _as_api(settings, tenant) as conn:
        assert _heartbeat(conn, conversation, uuid4()) is False
    _expire(superuser, conversation)
    with _as_api(settings, tenant) as conn:
        assert _heartbeat(conn, conversation, run) is False, "expired leases are never revived"


def test_heartbeat_run_is_tenant_bound(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    _tenant_a, conversation = make_conversation()
    tenant_b, _ = make_conversation()
    run = _hold(superuser, conversation, seconds=10)
    before = superuser.execute(
        "SELECT run_expires_at FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone()
    with _as_api(settings, tenant_b) as conn:
        assert _heartbeat(conn, conversation, run) is False
        conn.commit()
    after = superuser.execute(
        "SELECT run_expires_at FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone()
    assert before == after


@pytest.mark.parametrize("ttl", [0, 4, 601])
def test_heartbeat_run_rejects_unbounded_ttl(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation, ttl: int
) -> None:
    tenant, conversation = make_conversation()
    run = _hold(superuser, conversation)
    with _as_api(settings, tenant) as conn, pytest.raises(errors.InvalidParameterValue):
        _heartbeat(conn, conversation, run, ttl=ttl)


# ---------------------------------------------------------------------- chat_active
def _chat_active(settings: Settings) -> bool:
    with connect(settings, "worker") as conn:
        row = conn.execute("SELECT rf.chat_active()").fetchone()
    assert row is not None
    return bool(row[0])


def test_chat_active_sees_a_live_lease_of_any_tenant(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    """Cross-tenant by design (the worker has no tenant), but it returns one boolean only.
    Only the True direction is asserted on committed data: other tests' leases may be live."""
    _tenant, conversation = make_conversation()
    _hold(superuser, conversation)
    assert _chat_active(settings) is True


def test_chat_active_ignores_expired_and_released_leases(
    settings: Settings, make_conversation: MakeConversation
) -> None:
    """The False direction needs "no other live lease", which a shared session database cannot
    promise. So it runs in ONE rolled-back transaction: clear every lease there (invisible to
    every other session), then call the function as rf_worker in that same transaction."""
    _tenant, conversation = make_conversation()
    with connect(settings, "superuser") as conn:  # autocommit off: one transaction, rolled back
        conn.execute(
            "UPDATE rf.conversations SET active_run_id = NULL, run_expires_at = NULL "
            "WHERE active_run_id IS NOT NULL"
        )
        assert _chat_active_in_tx(conn) is False
        _hold(conn, conversation)
        assert _chat_active_in_tx(conn) is True
        _expire(conn, conversation)
        assert _chat_active_in_tx(conn) is False, "an expired lease is not chat activity"
        conn.rollback()


def _chat_active_in_tx(conn: Conn) -> bool:
    conn.execute("SET LOCAL ROLE rf_worker")
    try:
        row = conn.execute("SELECT rf.chat_active()").fetchone()
    finally:
        conn.execute("RESET ROLE")
    assert row is not None
    return bool(row[0])


def test_worker_cannot_see_conversations_without_a_tenant(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    _tenant, conversation = make_conversation()
    _hold(superuser, conversation)
    with connect(settings, "worker") as conn:
        assert conn.execute("SELECT count(*) FROM rf.conversations").fetchone() == (0,)


async def test_chat_active_adapter_runs_in_a_worker_system_transaction(
    settings: Settings, superuser: Conn, make_conversation: MakeConversation
) -> None:
    from ragforge_core.adapters.db.conversations import chat_active
    from ragforge_core.db.engine import make_engine, system_transaction

    _tenant, conversation = make_conversation()
    _hold(superuser, conversation)  # True is the only shared-database-safe direction
    engine = make_engine(settings, "worker")
    try:
        async with system_transaction(engine) as tx:
            assert await chat_active(tx) is True
    finally:
        await engine.dispose()
