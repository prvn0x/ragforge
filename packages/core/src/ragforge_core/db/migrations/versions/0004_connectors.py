"""Module C (connectors): take_tokens + penalize_tokens (global Notion rate limit),
route_notion_webhook, purge_tombstones.

OWNED BY MODULE C. Only module C edits it.

Every function here is SECURITY DEFINER, owned by rf_system (the migration runs as it), and
follows ADR-0013: pinned `search_path` with pg_temp LAST, REVOKE ALL FROM PUBLIC, and an
explicit GRANT to exactly the role that calls it (DESIGN section 6.1).

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# Token bucket shared by every worker for one Notion integration (ADR-0007). Notion's documented
# average is ~3 requests/s; a capacity of 3 allows a one-second burst and no more.
#
# The bucket is a DEBT bucket: the caller always takes its tokens and the balance may go
# negative; the function returns how long the caller must wait before using them. That gives a
# strict global rate in ONE round trip (no retry polling, no thundering herd) and lets a
# Retry-After be applied by taking `ceil(retry_after * rate)` tokens, which "drains the bucket"
# for every other worker too. A caller that is cancelled after taking tokens only wastes budget
# (errs on the slow side), never exceeds it. `p_n` is capped so one call cannot block a key for
# more than 10 minutes.
TAKE_TOKENS = """
CREATE FUNCTION rf.take_tokens(p_integration_key bytea, p_n integer)
    RETURNS integer
    LANGUAGE plpgsql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    c_capacity CONSTANT real := 3;
    c_rate     CONSTANT real := 3;      -- tokens per second
    c_max_n    CONSTANT integer := 1800; -- 10 minutes of budget
    v_tokens   real;
    v_last     timestamptz;
    v_now      timestamptz;
BEGIN
    IF p_integration_key IS NULL OR length(p_integration_key) <> 32 THEN
        RAISE EXCEPTION 'take_tokens: integration_key must be a 32-byte sha256 digest'
            USING ERRCODE = '22023';
    END IF;
    IF p_n IS NULL OR p_n < 1 OR p_n > c_max_n THEN
        -- Never a percent sign in this SQL: run_sql goes through the driver's paramstyle.
        RAISE EXCEPTION 'take_tokens: n must be between 1 and 1800'
            USING ERRCODE = '22023';
    END IF;

    -- Create the bucket full on first use; concurrent first callers race harmlessly here.
    INSERT INTO rf.notion_rate (integration_key, tokens, refilled_at)
        VALUES (p_integration_key, c_capacity, clock_timestamp())
        ON CONFLICT (integration_key) DO NOTHING;

    -- The row lock serializes every caller of this key across all processes.
    SELECT tokens, refilled_at INTO v_tokens, v_last
        FROM rf.notion_rate WHERE integration_key = p_integration_key
        FOR UPDATE;

    -- clock_timestamp() AFTER the lock: now() is the transaction start, which is stale for a
    -- caller that waited on the lock (it would refill too little and, worse, could move
    -- refilled_at backwards). GREATEST guards against a backwards wall-clock step.
    v_now := GREATEST(clock_timestamp(), v_last);
    v_tokens := LEAST(
        c_capacity,
        v_tokens + c_rate * EXTRACT(EPOCH FROM (v_now - v_last))::real
    ) - p_n;

    UPDATE rf.notion_rate
        SET tokens = v_tokens, refilled_at = v_now
        WHERE integration_key = p_integration_key;

    IF v_tokens >= 0 THEN
        RETURN 0;
    END IF;
    RETURN ceil(-v_tokens / c_rate * 1000)::integer;
END;
$$;
REVOKE ALL ON FUNCTION rf.take_tokens(bytea, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.take_tokens(bytea, integer) TO rf_worker;
"""

# Applying a 429 Retry-After to the shared bucket (ADR-0007 "Retry-After drains the bucket").
#
# Why not take_tokens(ceil(retry_after * rate)): debt ADDS UP. When N in-flight requests on the
# same token all get `Retry-After: 2`, each would add 6 tokens of debt and the integration would
# stall for ~2N seconds instead of 2. Notion's Retry-After is a deadline, not a cost, so this
# function sets the balance to AT MOST -p_n (`LEAST`): idempotent per window, the bucket waits
# max(retry_after) once, however many workers report the same 429. A longer Retry-After still
# extends the window; a shorter one never shortens it.
#
# It does not make the caller wait: it returns the resulting wait in ms for information. The
# caller either retries (its next take_tokens pays the wait) or gives the job back to the queue
# when the wait is long. Either way every OTHER worker of the integration backs off too.
PENALIZE_TOKENS = """
CREATE FUNCTION rf.penalize_tokens(p_integration_key bytea, p_n integer)
    RETURNS integer
    LANGUAGE plpgsql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    c_capacity CONSTANT real := 3;
    c_rate     CONSTANT real := 3;
    c_max_n    CONSTANT integer := 1800;
    v_tokens   real;
    v_last     timestamptz;
    v_now      timestamptz;
BEGIN
    IF p_integration_key IS NULL OR length(p_integration_key) <> 32 THEN
        RAISE EXCEPTION 'penalize_tokens: integration_key must be a 32-byte sha256 digest'
            USING ERRCODE = '22023';
    END IF;
    IF p_n IS NULL OR p_n < 1 OR p_n > c_max_n THEN
        RAISE EXCEPTION 'penalize_tokens: n must be between 1 and 1800'
            USING ERRCODE = '22023';
    END IF;

    INSERT INTO rf.notion_rate (integration_key, tokens, refilled_at)
        VALUES (p_integration_key, c_capacity, clock_timestamp())
        ON CONFLICT (integration_key) DO NOTHING;

    SELECT tokens, refilled_at INTO v_tokens, v_last
        FROM rf.notion_rate WHERE integration_key = p_integration_key
        FOR UPDATE;

    -- Same refill arithmetic as take_tokens, then clamp DOWN to the penalty (never add to it).
    v_now := GREATEST(clock_timestamp(), v_last);
    v_tokens := LEAST(
        c_capacity,
        v_tokens + c_rate * EXTRACT(EPOCH FROM (v_now - v_last))::real,
        -p_n::real
    );

    UPDATE rf.notion_rate
        SET tokens = v_tokens, refilled_at = v_now
        WHERE integration_key = p_integration_key;

    RETURN ceil(-v_tokens / c_rate * 1000)::integer;
END;
$$;
REVOKE ALL ON FUNCTION rf.penalize_tokens(bytea, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.penalize_tokens(bytea, integer) TO rf_worker;
"""

# Webhook routing (ADR-0007 [A]). The API verifies the signature first, then needs to know which
# tenants' connectors a Notion (integration, workspace) pair belongs to: inherently cross-tenant,
# because the request carries no tenant. Returns ids only (never config or ciphertext), and only
# for active connectors. The same workspace connected by two tenants legitimately routes to both.
# The connector config keys are owned by module C: config->>'integration_id' (Notion bot id)
# and config->>'workspace_id'.
ROUTE_NOTION_WEBHOOK = """
CREATE INDEX connectors_notion_route
    ON rf.connectors ((config->>'workspace_id'), (config->>'integration_id'))
    WHERE kind = 'notion';

CREATE FUNCTION rf.route_notion_webhook(p_integration_id text, p_workspace_id text)
    RETURNS TABLE (tenant_id uuid, connector_id uuid)
    LANGUAGE sql
    STABLE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    SELECT c.tenant_id, c.id
    FROM rf.connectors c
    WHERE c.kind = 'notion'
      AND c.status = 'active'
      AND p_integration_id IS NOT NULL AND p_integration_id <> ''
      AND p_workspace_id IS NOT NULL AND p_workspace_id <> ''
      AND c.config->>'workspace_id' = p_workspace_id
      AND c.config->>'integration_id' = p_integration_id
    ORDER BY c.tenant_id, c.id
$$;
REVOKE ALL ON FUNCTION rf.route_notion_webhook(text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.route_notion_webhook(text, text) TO rf_api;
"""

# Hard delete of documents tombstoned more than 7 days ago (ADR-0007 reconcile step 3), across
# all tenants (a worker maintenance job runs under the SYSTEM tenant). Chunks and document_links
# cascade; embeddings are content-addressed and collected by gc_embeddings (module B).
#
# Bounded batches keep the transaction short; the caller loops until it returns < p_limit.
# SKIP LOCKED: a row a sync is resurrecting right now is left for the next run, and the outer
# `deleted_at < cutoff` is re-checked by the DELETE after any wait, so a document that was
# un-tombstoned concurrently is never deleted.
PURGE_TOMBSTONES = """
CREATE INDEX documents_tombstoned ON rf.documents (deleted_at) WHERE deleted_at IS NOT NULL;

CREATE FUNCTION rf.purge_tombstones(p_limit integer DEFAULT 1000)
    RETURNS integer
    LANGUAGE plpgsql
    VOLATILE
    SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_cutoff  timestamptz := now() - interval '7 days';
    v_deleted integer;
BEGIN
    IF p_limit IS NULL OR p_limit < 1 OR p_limit > 10000 THEN
        RAISE EXCEPTION 'purge_tombstones: limit must be between 1 and 10000'
            USING ERRCODE = '22023';
    END IF;
    WITH victims AS (
        SELECT d.id FROM rf.documents d
        WHERE d.deleted_at IS NOT NULL AND d.deleted_at < v_cutoff
        ORDER BY d.deleted_at
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    )
    DELETE FROM rf.documents d
        USING victims v
        WHERE d.id = v.id AND d.deleted_at IS NOT NULL AND d.deleted_at < v_cutoff;
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$;
REVOKE ALL ON FUNCTION rf.purge_tombstones(integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.purge_tombstones(integer) TO rf_worker;
"""


def upgrade() -> None:
    run_sql(TAKE_TOKENS)
    run_sql(PENALIZE_TOKENS)
    run_sql(ROUTE_NOTION_WEBHOOK)
    run_sql(PURGE_TOMBSTONES)
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
