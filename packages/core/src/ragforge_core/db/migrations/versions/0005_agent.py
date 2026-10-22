"""Module D (agent): conversation run leases (assert_run_lease, heartbeat_run, chat_active) and the
LangGraph checkpoint + store schema.

OWNED BY MODULE D. Only module D edits it.

Run leases (DESIGN section 8, Flow B): `conversations.active_run_id/run_expires_at` is the lease.
Taking it is a CAS UPDATE under RLS (adapters/db/conversations.py). The functions here exist for
the LangGraph pool path (the fence must run inside the saver's own write transaction) and for
the worker, which must see leases across tenants:

  assert_run_lease(conversation, run)   api     row-locks the conversation; raises RFL01 unless
                                                run still holds an unexpired lease (fencing)
  heartbeat_run(conversation, run, ttl) api     extends a live lease; false = lost
  chat_active()                         worker  any live lease anywhere (throttles embedding);
                                                returns one boolean, never tenant data

Tenant binding (fail closed, ADR-0013): both lease functions only ever consider the row of the
tenant in `app.tenant_id`. An unset tenant is NULL, which matches no row, so assert_run_lease
raises RFL01 and heartbeat_run returns false. Every caller (the saver, SqlConversationRuns)
sets the tenant transaction-locally; a code path that forgets to must fail, not work across
tenants on the strength of a run id (run ids reach clients in InterruptEvent).

Every function is SECURITY DEFINER owned by rf_system and follows ADR-0013 (pinned search_path
with pg_temp LAST, REVOKE ALL FROM PUBLIC, explicit GRANT to exactly the calling role).
`clock_timestamp()` (not `now()`) is used for every lease comparison: `now()` is the transaction
start, which is stale for a caller that waited on the row lock and would let an expired lease
pass the fence.

LangGraph tables (ADR-0008): created here from the INSTALLED library's own migration SQL, so the
library's `setup()` is never needed at runtime (app roles cannot run DDL). The applied prefix of
each library MIGRATIONS list is pinned by count AND by a sha256 fingerprint: if a library upgrade
changes those statements this migration fails loudly instead of silently diverging, and the new
statements go into a NEW revision. `CREATE INDEX CONCURRENTLY` cannot run inside a transaction;
the tables are empty at creation, so the plain (transactional) form builds the identical index.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ragforge_core.db.migrations.helpers import bump_schema_version, classify, run_sql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

# Pinned: langgraph-checkpoint-postgres 3.2.0 (langgraph 1.2.15, langgraph-checkpoint 4.3.0).
CHECKPOINT_MIGRATIONS_APPLIED = 10
CHECKPOINT_MIGRATIONS_SHA256 = "8d6189c8273a44977c803d2d8769091a75c23c5f290eb309229eb1e39e06e8fc"
STORE_MIGRATIONS_APPLIED = 4
STORE_MIGRATIONS_SHA256 = "ffc09083a25179bf111c480619265ed90442dde61becf94a25c3f1d823eee193"

LANGGRAPH_CLASSES = {
    "checkpoints": "authoritative",
    "checkpoint_blobs": "authoritative",
    "checkpoint_writes": "authoritative",
    "checkpoint_migrations": "system",
    "store": "authoritative",
    "store_migrations": "system",
}

ASSERT_RUN_LEASE = """
CREATE FUNCTION rf.assert_run_lease(p_conversation_id uuid, p_run_id uuid)
    RETURNS void
    LANGUAGE plpgsql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_run     uuid;
    v_expires timestamptz;
BEGIN
    -- The tenant predicate is part of the WHERE (not a later check) so a caller bound to another
    -- tenant never even takes the row lock on this conversation.
    SELECT active_run_id, run_expires_at INTO v_run, v_expires
        FROM rf.conversations
        WHERE id = p_conversation_id
          AND tenant_id = rf.current_tenant()
        FOR UPDATE;
    -- One message and SQLSTATE for every failure: missing, foreign, released, taken over or
    -- expired all mean "discard your work", and distinguishing them would leak existence.
    IF NOT FOUND
       OR p_run_id IS NULL
       OR v_run IS DISTINCT FROM p_run_id
       OR v_expires <= clock_timestamp() THEN
        RAISE EXCEPTION 'run lease lost' USING ERRCODE = 'RFL01';
    END IF;
END;
$$;
REVOKE ALL ON FUNCTION rf.assert_run_lease(uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.assert_run_lease(uuid, uuid) TO rf_api;
"""

HEARTBEAT_RUN = """
CREATE FUNCTION rf.heartbeat_run(p_conversation_id uuid, p_run_id uuid, p_ttl_seconds integer)
    RETURNS boolean
    LANGUAGE plpgsql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
BEGIN
    IF p_ttl_seconds IS NULL OR p_ttl_seconds < 5 OR p_ttl_seconds > 600 THEN
        RAISE EXCEPTION 'heartbeat_run: ttl must be between 5 and 600 seconds'
            USING ERRCODE = '22023';
    END IF;
    -- An expired lease is never revived: once expired, another run may already have taken
    -- over (or be about to), and the old holder must stop.
    UPDATE rf.conversations
        SET run_expires_at = clock_timestamp() + make_interval(secs => p_ttl_seconds)
        WHERE id = p_conversation_id
          AND active_run_id = p_run_id
          AND run_expires_at > clock_timestamp()
          AND tenant_id = rf.current_tenant();
    RETURN FOUND;
END;
$$;
REVOKE ALL ON FUNCTION rf.heartbeat_run(uuid, uuid, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.heartbeat_run(uuid, uuid, integer) TO rf_api;
"""

CHAT_ACTIVE = """
CREATE FUNCTION rf.chat_active()
    RETURNS boolean
    LANGUAGE sql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    SELECT EXISTS (
        SELECT 1 FROM rf.conversations
        WHERE active_run_id IS NOT NULL AND run_expires_at > clock_timestamp()
    );
$$;
REVOKE ALL ON FUNCTION rf.chat_active() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.chat_active() TO rf_worker;

-- chat_active() is polled by every worker; keep it an index probe over live leases only.
CREATE INDEX conversations_live_runs ON rf.conversations (run_expires_at)
    WHERE active_run_id IS NOT NULL;
"""

# LangGraph rows carry no tenant_id (they are outside RLS), so tenant/user/conversation deletes
# would orphan them. These INVOKER triggers keep the "reached only via conversations" invariant
# true for deletes too. Cascaded deletes (tenant -> users -> conversations) run as the table
# owner, plain deletes as rf_api, which has DELETE on these tables.
CLEANUP_TRIGGERS = """
CREATE FUNCTION rf.drop_conversation_checkpoints()
    RETURNS trigger
    LANGUAGE plpgsql
    SECURITY INVOKER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
BEGIN
    DELETE FROM rf.checkpoint_writes WHERE thread_id = OLD.id::text;
    DELETE FROM rf.checkpoint_blobs WHERE thread_id = OLD.id::text;
    DELETE FROM rf.checkpoints WHERE thread_id = OLD.id::text;
    RETURN OLD;
END;
$$;
REVOKE ALL ON FUNCTION rf.drop_conversation_checkpoints() FROM PUBLIC;
CREATE TRIGGER conversations_drop_checkpoints
    AFTER DELETE ON rf.conversations
    FOR EACH ROW EXECUTE FUNCTION rf.drop_conversation_checkpoints();

-- Store namespaces always start with (tenant_id, user_id) (DESIGN section 8, Memory); the
-- library encodes a namespace as its labels joined with '.'. "Starts with <prefix>." is the
-- byte-wise range ['<prefix>.', '<prefix>/') ('/' follows '.'), written with the
-- text_pattern_ops operators so the library's store_prefix_idx serves it whatever the collation.
-- (No LIKE: run_sql goes through psycopg, which reads a percent sign as a placeholder.)
CREATE FUNCTION rf.drop_user_store_items()
    RETURNS trigger
    LANGUAGE plpgsql
    SECURITY INVOKER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_prefix text := OLD.tenant_id::text || '.' || OLD.id::text;
BEGIN
    DELETE FROM rf.store
        WHERE prefix = v_prefix
           OR (prefix ~>=~ (v_prefix || '.') AND prefix ~<~ (v_prefix || '/'));
    RETURN OLD;
END;
$$;
REVOKE ALL ON FUNCTION rf.drop_user_store_items() FROM PUBLIC;
CREATE TRIGGER users_drop_store_items
    AFTER DELETE ON rf.users
    FOR EACH ROW EXECUTE FUNCTION rf.drop_user_store_items();
"""

# rf_api runs the agent (saver + store); the *_migrations tables are readable so it can verify
# the schema at startup. rf_worker gets NOTHING here: these rows hold full conversation content
# with no tenant_id and no RLS, so a blanket worker grant would be a new cross-tenant read path.
# When summarize_conversation lands it reads through a tenant-scoped path (FencedPostgresSaver
# with the tenant set + the conversations lookup under RLS), with an ADR-0013 amendment first.
LANGGRAPH_GRANTS = """
GRANT SELECT, INSERT, UPDATE, DELETE ON
    rf.checkpoints, rf.checkpoint_blobs, rf.checkpoint_writes, rf.store
    TO rf_api;
GRANT SELECT ON rf.checkpoint_migrations, rf.store_migrations TO rf_api;
"""


def _pinned_prefix(statements: Sequence[str], count: int, sha256: str, name: str) -> list[str]:
    """The first `count` library statements, verified against the pinned fingerprint."""
    prefix = list(statements[:count])
    digest = hashlib.sha256("\x1e".join(prefix).encode()).hexdigest()
    if len(prefix) != count or digest != sha256:
        raise RuntimeError(
            f"{name}: installed LangGraph migration SQL differs from the version pinned in 0005. "
            "Do not edit 0005; apply new library statements in a new revision and re-run the "
            "saver contract tests."
        )
    return prefix


def _transactional(statement: str) -> str:
    """Drop CONCURRENTLY (illegal in a transaction; equivalent on a just-created empty table)."""
    return statement.replace("CREATE INDEX CONCURRENTLY", "CREATE INDEX")


def _apply_library_migrations(statements: Sequence[str], version_table: str) -> None:
    for version, statement in enumerate(statements):
        run_sql(_transactional(statement))
        # Record it exactly as the library's setup() would, so setup() is a no-op if ever run.
        run_sql(f"INSERT INTO rf.{version_table} (v) VALUES ({int(version)})")  # noqa: S608 - fixed table name, int()


def _create_langgraph_schema() -> None:
    # Imported here, not at module level: Alembic imports every revision file, and only this
    # one needs the `agent` extra.
    from langgraph.checkpoint.postgres.base import MIGRATIONS as CHECKPOINT_MIGRATIONS
    from langgraph.store.postgres.base import MIGRATIONS as STORE_MIGRATIONS

    checkpoint = _pinned_prefix(
        CHECKPOINT_MIGRATIONS,
        CHECKPOINT_MIGRATIONS_APPLIED,
        CHECKPOINT_MIGRATIONS_SHA256,
        "checkpoint",
    )
    store = _pinned_prefix(
        STORE_MIGRATIONS, STORE_MIGRATIONS_APPLIED, STORE_MIGRATIONS_SHA256, "store"
    )
    # The library SQL is unqualified. Pin the target schema for this migration's transaction
    # only (SET LOCAL), so nothing can land in another schema whatever the session default.
    run_sql("SET LOCAL search_path = rf, pg_catalog")
    # checkpoint MIGRATIONS[0] creates its own version table; the store's setup() creates
    # store_migrations outside its list, so do the same here.
    _apply_library_migrations(checkpoint, "checkpoint_migrations")
    run_sql("CREATE TABLE rf.store_migrations (v INTEGER PRIMARY KEY)")
    _apply_library_migrations(store, "store_migrations")
    run_sql("SET LOCAL search_path = rf, ext")
    for table, data_class in LANGGRAPH_CLASSES.items():
        classify(table, data_class)


def upgrade() -> None:
    run_sql(ASSERT_RUN_LEASE)
    run_sql(HEARTBEAT_RUN)
    run_sql(CHAT_ACTIVE)
    _create_langgraph_schema()
    run_sql(CLEANUP_TRIGGERS)
    run_sql(LANGGRAPH_GRANTS)
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
