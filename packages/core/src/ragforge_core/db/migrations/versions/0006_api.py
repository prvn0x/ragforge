"""Module E (api/safety): authenticate_api_key, provision_tenant, MCP key marker, and grants.

OWNED BY MODULE E. Both functions are named cross-tenant paths in ADR-0013 and follow its rules:
pinned search_path (pg_temp last), REVOKE ALL FROM PUBLIC, explicit GRANT EXECUTE, and a
cross-tenant test (packages/core/tests/module_e2/test_e2_db_functions.py).

Additive change to a 0001 table (allowed by docs/PHASE1.md): the new nullable
`api_keys.client_kind` column marks MCP keys, and a CHECK makes it impossible to store an MCP key
that carries `approvals:write` or `admin` (DESIGN section 10: an agent must never be able to
approve its own destructive actions). Existing rows are NULL (= ordinary user key) and pass.
Also additive: indexes backing the API's keyset pages on documents and audit_log.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql(
        """
        -- ------------------------------------------------------------ MCP key marker
        ALTER TABLE rf.api_keys
            ADD COLUMN client_kind text CHECK (client_kind IN ('mcp')),
            ADD CONSTRAINT api_keys_mcp_never_privileged CHECK (
                client_kind IS DISTINCT FROM 'mcp'
                OR NOT (scopes && ARRAY['approvals:write', 'admin']::text[])
            );
        CREATE INDEX api_keys_tenant_created ON rf.api_keys (tenant_id, created_at DESC, id DESC);

        -- ------------------------------------------------------------ keyset-page indexes
        -- Additive. Each matches one API keyset page exactly (tenant predicate + ORDER BY), so a
        -- page is an index range scan instead of a sort of the tenant's (or table's) rows:
        --   GET /v1/documents: live documents, created_at DESC, id DESC
        --   GET /v1/audit:     id DESC (the 0001 index orders by created_at, not id)
        CREATE INDEX documents_live_tenant_created
            ON rf.documents (tenant_id, created_at DESC, id DESC) WHERE deleted_at IS NULL;
        CREATE INDEX audit_log_tenant_id_desc ON rf.audit_log (tenant_id, id DESC);

        -- ------------------------------------------------------------ authenticate_api_key
        -- Cross-tenant by nature: the caller does not know its tenant before authenticating.
        -- The comparison is constant-time: every call XOR-accumulates all 32 bytes, including
        -- for an unknown prefix (against a fixed dummy digest) and for revoked keys, so response
        -- time does not reveal how many leading bytes matched or whether a prefix exists.
        CREATE FUNCTION rf.authenticate_api_key(p_prefix text, p_sha256 bytea)
            RETURNS TABLE (api_key_id uuid, tenant_id uuid, user_id uuid, scopes text[],
                           client_kind text)
            LANGUAGE plpgsql VOLATILE SECURITY DEFINER
            SET search_path = pg_catalog, rf, ext, pg_temp
        AS $fn$
        #variable_conflict use_column
        DECLARE
            k        rf.api_keys;  -- the table row type; this SQL must contain no percent sign
            expected bytea;
            diff     integer := 0;
            i        integer;
        BEGIN
            IF p_prefix IS NULL OR p_sha256 IS NULL OR length(p_sha256) <> 32 THEN
                RETURN;
            END IF;

            SELECT a.* INTO k FROM rf.api_keys a WHERE a.prefix = p_prefix;
            expected := coalesce(k.key_sha256, decode(repeat('00', 32), 'hex'));

            FOR i IN 0..31 LOOP
                diff := diff | (get_byte(expected, i) # get_byte(p_sha256, i));
            END LOOP;

            IF k.id IS NULL OR diff <> 0 OR k.revoked_at IS NOT NULL
               OR k.tenant_id = '00000000-0000-0000-0000-000000000000'::uuid THEN
                RETURN;
            END IF;

            -- Throttled, non-blocking usage stamp: at most one write per key per minute, and a
            -- concurrent request that holds the row lock is skipped instead of waited on, so a
            -- busy key never serializes authentication.
            UPDATE rf.api_keys a SET last_used_at = now()
             WHERE a.id = (SELECT b.id FROM rf.api_keys b
                            WHERE b.id = k.id
                              AND (b.last_used_at IS NULL
                                   OR b.last_used_at < now() - interval '60 seconds')
                            FOR UPDATE SKIP LOCKED);

            RETURN QUERY SELECT
                k.id,
                k.tenant_id,
                k.user_id,
                CASE WHEN k.client_kind = 'mcp'
                     THEN ARRAY(SELECT s FROM unnest(k.scopes) AS s
                                 WHERE s NOT IN ('approvals:write', 'admin'))
                     ELSE k.scopes END,
                k.client_kind;
        END
        $fn$;
        REVOKE ALL ON FUNCTION rf.authenticate_api_key(text, bytea) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.authenticate_api_key(text, bytea) TO rf_api;

        -- ------------------------------------------------------------ provision_tenant
        -- Cross-tenant: creates a tenant the caller is not (yet) scoped to. Admin scope is
        -- enforced by the API/CLI; the function enforces the data rules (no 'system' kind, team
        -- tenants only with recorded visibility consent, ADR-0013) and writes the audit row in
        -- the same transaction.
        CREATE FUNCTION rf.provision_tenant(
                p_name text, p_kind text, p_owner_display_name text, p_consent boolean)
            RETURNS TABLE (tenant_id uuid, user_id uuid)
            LANGUAGE plpgsql VOLATILE SECURITY DEFINER
            SET search_path = pg_catalog, rf, ext, pg_temp
        AS $fn$
        #variable_conflict use_column
        DECLARE
            t     uuid;
            u     uuid;
            nm    text := btrim(coalesce(p_name, ''));
            v_owner text := btrim(coalesce(p_owner_display_name, ''));
        BEGIN
            IF p_kind IS NULL OR p_kind NOT IN ('personal', 'team') THEN
                RAISE EXCEPTION 'tenant kind must be personal or team' USING ERRCODE = '22023';
            END IF;
            IF p_kind = 'team' AND p_consent IS NOT TRUE THEN
                RAISE EXCEPTION 'team tenants require recorded team-visibility consent'
                    USING ERRCODE = '22023';
            END IF;
            IF length(nm) NOT BETWEEN 1 AND 200 OR length(v_owner) NOT BETWEEN 1 AND 200 THEN
                RAISE EXCEPTION 'name and owner display name must be 1..200 characters'
                    USING ERRCODE = '22023';
            END IF;

            INSERT INTO rf.tenants (name, kind, team_visibility_consent_at)
                VALUES (nm, p_kind, CASE WHEN p_kind = 'team' THEN now() END)
                RETURNING id INTO t;
            INSERT INTO rf.users (tenant_id, display_name) VALUES (t, v_owner) RETURNING id INTO u;
            INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
                VALUES (t, 'system:provision_tenant', 'tenant.provisioned', t::text,
                        jsonb_build_object('kind', p_kind,
                                           'team_visibility_consent', p_kind = 'team'));
            RETURN QUERY SELECT t, u;
        END
        $fn$;
        REVOKE ALL ON FUNCTION rf.provision_tenant(text, text, text, boolean) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.provision_tenant(text, text, text, boolean) TO rf_api;
        """
    )
    # Module E: add DDL above this line. Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
