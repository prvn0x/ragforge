"""Base schema: the shared data contract for all Phase 1 modules (DESIGN section 6).

Module-owned objects (functions, triggers, vector indexes) come in 0002-0006.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import classify, run_sql, tenant_table

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TENANT_TABLES = [
    "users", "api_keys", "collections", "connectors", "sync_cursors", "exclusions", "uploads",
    "documents", "document_links", "chunks", "embeddings", "embedding_touches",
    "collection_space_stats", "jobs", "jobs_history", "audit_log", "conversations",
    "conversation_summaries", "pending_approvals", "usage_events",
]  # fmt: skip

CLASSES = {
    "alembic_version": "system", "schema_meta": "authoritative", "migration_checksums": "system", "tenants": "authoritative",
    "users": "authoritative", "api_keys": "authoritative", "embedding_spaces": "authoritative",
    "collections": "authoritative", "connectors": "authoritative", "sync_cursors": "derived",
    "exclusions": "authoritative", "uploads": "authoritative", "documents": "derived",
    "document_links": "derived", "chunks": "derived", "embeddings": "derived",
    "embedding_touches": "derived", "collection_space_stats": "derived",
    "notion_rate": "system", "connector_inflight": "derived", "jobs": "derived",
    "jobs_history": "derived", "audit_log": "authoritative", "conversations": "authoritative",
    "conversation_summaries": "authoritative", "pending_approvals": "authoritative",
    "usage_events": "authoritative", "eval_runs": "authoritative",
}  # fmt: skip


def upgrade() -> None:
    run_sql(
        """
        -- ------------------------------------------------------------ tenancy primitive
        CREATE FUNCTION rf.current_tenant() RETURNS uuid
            LANGUAGE sql STABLE PARALLEL SAFE
            AS $$ SELECT NULLIF(current_setting('app.tenant_id', true), '')::uuid $$;
        REVOKE ALL ON FUNCTION rf.current_tenant() FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION rf.current_tenant() TO rf_api, rf_worker, rf_backup;

        -- ------------------------------------------------------------ schema compatibility
        CREATE TABLE rf.schema_meta (
            singleton        boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            schema_version   integer NOT NULL,
            min_code_version integer NOT NULL
        );
        INSERT INTO rf.schema_meta VALUES (true, 1, 0);

        -- Checksums of applied migration files: editing an already-applied revision is
        -- detected by ragforge-migrate instead of being silently skipped.
        CREATE TABLE rf.migration_checksums (
            revision   text PRIMARY KEY,
            sha256     text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        );

        -- ------------------------------------------------------------ tenants & identity
        CREATE TABLE rf.tenants (
            id                         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name                       text NOT NULL,
            kind                       text NOT NULL CHECK (kind IN ('personal', 'team', 'system')),
            team_visibility_consent_at timestamptz,
            created_at                 timestamptz NOT NULL DEFAULT now(),
            CHECK (kind <> 'team' OR team_visibility_consent_at IS NOT NULL)
        );
        INSERT INTO rf.tenants (id, name, kind)
            VALUES ('00000000-0000-0000-0000-000000000000', 'SYSTEM', 'system');
        ALTER TABLE rf.tenants ENABLE ROW LEVEL SECURITY;
        CREATE POLICY tenant_isolation ON rf.tenants USING (id = rf.current_tenant());
        CREATE POLICY backup_read ON rf.tenants FOR SELECT TO rf_backup USING (true);

        CREATE TABLE rf.users (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id    uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            display_name text NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, id)
        );

        CREATE TABLE rf.api_keys (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id    uuid NOT NULL,
            user_id      uuid NOT NULL,
            name         text NOT NULL,
            prefix       text NOT NULL UNIQUE,
            key_sha256   bytea NOT NULL CHECK (length(key_sha256) = 32),
            scopes       text[] NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            last_used_at timestamptz,
            revoked_at   timestamptz,
            FOREIGN KEY (tenant_id, user_id) REFERENCES rf.users (tenant_id, id) ON DELETE CASCADE
        );

        -- ------------------------------------------------------------ embedding spaces (global)
        CREATE TABLE rf.embedding_spaces (
            id           integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            provider     text NOT NULL,
            model        text NOT NULL,
            dim          integer NOT NULL CHECK (dim BETWEEN 1 AND 16000),
            doc_prefix   text NOT NULL DEFAULT '',
            query_prefix text NOT NULL DEFAULT '',
            distance     text NOT NULL CHECK (distance IN ('cosine', 'l2', 'ip')),
            status       text NOT NULL CHECK (status IN ('building', 'active', 'retired')),
            created_at   timestamptz NOT NULL DEFAULT now(),
            retired_at   timestamptz,
            UNIQUE (provider, model, dim, doc_prefix, query_prefix, distance)
        );
        INSERT INTO rf.embedding_spaces (provider, model, dim, doc_prefix, query_prefix, distance, status)
            VALUES ('ollama', 'nomic-embed-text', 768, 'search_document: ', 'search_query: ', 'cosine', 'active');

        -- ------------------------------------------------------------ collections & sources
        CREATE TABLE rf.collections (
            id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id         uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            name              text NOT NULL,
            active_space_id   integer NOT NULL REFERENCES rf.embedding_spaces,
            building_space_id integer REFERENCES rf.embedding_spaces,
            fts_config        regconfig NOT NULL DEFAULT 'english',
            tau               real,
            created_at        timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, name),
            UNIQUE (tenant_id, id),
            CHECK (building_space_id IS DISTINCT FROM active_space_id)
        );

        CREATE TABLE rf.connectors (
            id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id            uuid NOT NULL,
            kind                 text NOT NULL CHECK (kind IN ('notion')),
            target_collection_id uuid NOT NULL,
            display_name         text NOT NULL,
            config               jsonb NOT NULL DEFAULT '{}',
            secret_ciphertext    bytea,
            wrapped_dek          bytea,
            status               text NOT NULL DEFAULT 'active'
                                 CHECK (status IN ('active', 'paused', 'error', 'deleting')),
            last_error           text,
            next_poll_at         timestamptz NOT NULL DEFAULT now(),
            created_at           timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, target_collection_id) REFERENCES rf.collections (tenant_id, id)
                ON DELETE CASCADE
        );

        CREATE TABLE rf.sync_cursors (
            connector_id uuid NOT NULL,
            tenant_id    uuid NOT NULL,
            scope        text NOT NULL,
            cursor       text NOT NULL,
            updated_at   timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connector_id, scope),
            FOREIGN KEY (tenant_id, connector_id) REFERENCES rf.connectors (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.exclusions (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id   uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            source_kind text NOT NULL CHECK (source_kind IN ('notion', 'file')),
            external_id text NOT NULL,
            reason      text,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, source_kind, external_id)
        );

        CREATE TABLE rf.uploads (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id     uuid NOT NULL,
            collection_id uuid NOT NULL,
            blob_key      text NOT NULL UNIQUE,
            filename      text NOT NULL,
            sha256        bytea NOT NULL CHECK (length(sha256) = 32),
            size_bytes    bigint NOT NULL CHECK (size_bytes >= 0),
            created_at    timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id) ON DELETE CASCADE
        );

        -- ------------------------------------------------------------ documents & chunks
        CREATE TABLE rf.documents (
            id                  uuid PRIMARY KEY,  -- uuidv5(tenant, collection, connector|'upload', external_id)
            tenant_id           uuid NOT NULL,
            collection_id       uuid NOT NULL,
            connector_id        uuid,
            upload_id           uuid,
            source_kind         text NOT NULL CHECK (source_kind IN ('notion', 'file')),
            external_id         text NOT NULL,
            external_url        text,
            title               text NOT NULL,
            external_updated_at timestamptz,
            fetched_at          timestamptz,
            page_sha256         bytea,
            synced_source_ids   text[] NOT NULL DEFAULT '{}',
            metadata            jsonb NOT NULL DEFAULT '{}',
            deleted_at          timestamptz,
            created_at          timestamptz NOT NULL DEFAULT now(),
            updated_at          timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, id),
            UNIQUE NULLS NOT DISTINCT (tenant_id, collection_id, connector_id, external_id),
            FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, connector_id) REFERENCES rf.connectors (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, upload_id) REFERENCES rf.uploads (tenant_id, id) ON DELETE CASCADE,
            CHECK ((connector_id IS NULL) <> (upload_id IS NULL))
        );
        CREATE INDEX documents_tenant_collection ON rf.documents (tenant_id, collection_id);

        CREATE TABLE rf.document_links (
            tenant_id        uuid NOT NULL,
            from_document_id uuid NOT NULL,
            to_external_id   text NOT NULL,
            PRIMARY KEY (from_document_id, to_external_id),
            FOREIGN KEY (tenant_id, from_document_id) REFERENCES rf.documents (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.chunks (
            id             uuid PRIMARY KEY,  -- uuidv5(document_id, ordinal)
            tenant_id      uuid NOT NULL,
            collection_id  uuid NOT NULL,
            document_id    uuid NOT NULL,
            anchor         text NOT NULL,
            ordinal        integer NOT NULL CHECK (ordinal >= 0),
            heading_path   text[] NOT NULL DEFAULT '{}',
            display_text   text NOT NULL,
            display_sha256 bytea NOT NULL CHECK (length(display_sha256) = 32),
            embed_sha256   bytea NOT NULL CHECK (length(embed_sha256) = 32),
            token_count    integer NOT NULL CHECK (token_count >= 0),
            metadata       jsonb NOT NULL DEFAULT '{}',
            tsv            tsvector,  -- maintained by trigger (module B, 0003)
            UNIQUE (document_id, ordinal),
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, document_id) REFERENCES rf.documents (tenant_id, id) ON DELETE CASCADE
        );
        CREATE INDEX chunks_tenant_collection ON rf.chunks (tenant_id, collection_id);
        CREATE INDEX chunks_embed_sha ON rf.chunks (tenant_id, collection_id, embed_sha256);

        CREATE TABLE rf.embeddings (
            tenant_id     uuid NOT NULL,
            collection_id uuid NOT NULL,
            space_id      integer NOT NULL REFERENCES rf.embedding_spaces,
            embed_sha256  bytea NOT NULL CHECK (length(embed_sha256) = 32),
            embedding     ext.vector NOT NULL,  -- dim fixed per space; HNSW per space (module B)
            created_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, collection_id, space_id, embed_sha256),
            FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.embedding_touches (
            tenant_id     uuid NOT NULL,
            collection_id uuid NOT NULL,
            space_id      integer NOT NULL,
            embed_sha256  bytea NOT NULL CHECK (length(embed_sha256) = 32),
            touched_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, collection_id, space_id, embed_sha256),
            FOREIGN KEY (space_id) REFERENCES rf.embedding_spaces,
            FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.collection_space_stats (
            tenant_id     uuid NOT NULL,
            collection_id uuid NOT NULL,
            space_id      integer NOT NULL REFERENCES rf.embedding_spaces,
            vector_count  bigint NOT NULL DEFAULT 0,
            updated_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (collection_id, space_id),
            FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id) ON DELETE CASCADE
        );

        -- ------------------------------------------------------------ rate limiting (system; definer-only)
        CREATE TABLE rf.notion_rate (
            integration_key bytea PRIMARY KEY CHECK (length(integration_key) = 32),
            tokens          real NOT NULL,
            refilled_at     timestamptz NOT NULL
        );
        CREATE TABLE rf.connector_inflight (
            connector_id uuid PRIMARY KEY REFERENCES rf.connectors ON DELETE CASCADE,
            running      integer NOT NULL DEFAULT 0 CHECK (running >= 0)
        );

        -- ------------------------------------------------------------ jobs (module A owns functions)
        CREATE TABLE rf.jobs (
            id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id        uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            connector_id     uuid,
            kind             text NOT NULL,
            priority         smallint NOT NULL CHECK (priority IN (0, 10, 20, 30, 40)),
            payload          jsonb NOT NULL DEFAULT '{}',
            dedupe_key       text NOT NULL,
            status           text NOT NULL DEFAULT 'queued'
                             CHECK (status IN ('queued', 'running', 'done', 'failed', 'dead', 'superseded')),
            attempts         integer NOT NULL DEFAULT 0,
            max_attempts     integer NOT NULL DEFAULT 8,
            run_after        timestamptz NOT NULL DEFAULT now(),
            enqueued_at      timestamptz NOT NULL DEFAULT now(),
            leased_by        text,
            lease_version    integer NOT NULL DEFAULT 0,
            lease_expires_at timestamptz,
            last_error       text,
            progress         jsonb NOT NULL DEFAULT '{}',
            finished_at      timestamptz,
            CHECK ((status = 'running') = (lease_expires_at IS NOT NULL)),
            -- SET NULL (connector_id) only: a delete_connector job must survive the delete it performs.
            FOREIGN KEY (tenant_id, connector_id) REFERENCES rf.connectors (tenant_id, id)
                ON DELETE SET NULL (connector_id)
        );
        CREATE INDEX jobs_claim ON rf.jobs (priority, enqueued_at) WHERE status = 'queued';
        CREATE UNIQUE INDEX jobs_dedupe ON rf.jobs (tenant_id, dedupe_key) WHERE status = 'queued';
        CREATE INDEX jobs_leases ON rf.jobs (lease_expires_at) WHERE status = 'running';

        CREATE TABLE rf.jobs_history (
            LIKE rf.jobs INCLUDING DEFAULTS INCLUDING CONSTRAINTS
        ) PARTITION BY RANGE (finished_at);
        ALTER TABLE rf.jobs_history ALTER COLUMN finished_at SET NOT NULL;
        CREATE TABLE rf.jobs_history_default PARTITION OF rf.jobs_history DEFAULT;
        COMMENT ON TABLE rf.jobs_history_default IS 'class: derived';
        CREATE INDEX jobs_history_id ON rf.jobs_history (id);

        -- ------------------------------------------------------------ audit, conversations, usage
        CREATE TABLE rf.audit_log (
            id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id  uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            actor      text NOT NULL,
            action     text NOT NULL,
            target     text,
            details    jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX audit_log_tenant_time ON rf.audit_log (tenant_id, created_at DESC);

        CREATE TABLE rf.conversations (
            id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),  -- = LangGraph thread_id
            tenant_id          uuid NOT NULL,
            user_id            uuid NOT NULL,
            collection_ids     uuid[] NOT NULL,
            title              text,
            active_run_id      uuid,
            run_expires_at     timestamptz,
            head_checkpoint_id text,
            created_at         timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, user_id) REFERENCES rf.users (tenant_id, id) ON DELETE CASCADE,
            CHECK ((active_run_id IS NULL) = (run_expires_at IS NULL))
        );

        CREATE TABLE rf.conversation_summaries (
            conversation_id uuid PRIMARY KEY,
            tenant_id       uuid NOT NULL,
            upto_message_id text NOT NULL,
            summary         text NOT NULL,
            updated_at      timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES rf.conversations (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.pending_approvals (
            id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       uuid NOT NULL,
            conversation_id uuid NOT NULL,
            run_id          uuid NOT NULL,
            tool_call_id    text NOT NULL,
            tool            text NOT NULL,
            args            jsonb NOT NULL,
            status          text NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
            created_at      timestamptz NOT NULL DEFAULT now(),
            decided_at      timestamptz,
            UNIQUE (conversation_id, tool_call_id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES rf.conversations (tenant_id, id) ON DELETE CASCADE
        );

        CREATE TABLE rf.usage_events (
            id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id         uuid NOT NULL REFERENCES rf.tenants ON DELETE CASCADE,
            kind              text NOT NULL,
            provider          text NOT NULL,
            model             text NOT NULL,
            prompt_version    text,
            prompt_tokens     integer NOT NULL DEFAULT 0,
            completion_tokens integer NOT NULL DEFAULT 0,
            latency_ms        integer,
            created_at        timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE rf.eval_runs (
            id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            git_sha        text,
            dataset        text NOT NULL,
            space_id       integer REFERENCES rf.embedding_spaces,
            model          text,
            prompt_version text,
            metrics        jsonb NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now()
        );
        """
    )

    for table in TENANT_TABLES:
        tenant_table(table)
    for table, data_class in CLASSES.items():
        classify(table, data_class)

    # ---------------------------------------------------------------- grants (least privilege)
    # Writes to jobs/notion_rate/connector_inflight happen only via module A/C definer functions.
    run_sql(
        """
        GRANT SELECT ON rf.schema_meta, rf.embedding_spaces TO rf_api, rf_worker, rf_metrics;
        GRANT SELECT ON rf.tenants TO rf_api, rf_worker;

        GRANT SELECT, INSERT, UPDATE, DELETE ON
            rf.users, rf.api_keys, rf.collections, rf.connectors, rf.exclusions, rf.uploads,
            rf.conversations, rf.pending_approvals
            TO rf_api;
        GRANT SELECT, UPDATE, DELETE ON rf.documents TO rf_api;  -- UPDATE: tombstone (deleted_at)
        GRANT SELECT ON rf.chunks, rf.document_links, rf.collection_space_stats,
            rf.conversation_summaries, rf.jobs, rf.jobs_history TO rf_api;
        GRANT SELECT, INSERT ON rf.audit_log, rf.usage_events TO rf_api;
        GRANT SELECT ON rf.eval_runs TO rf_api;

        GRANT SELECT ON rf.users, rf.collections, rf.exclusions, rf.uploads, rf.conversations,
            rf.pending_approvals, rf.jobs, rf.jobs_history TO rf_worker;
        GRANT SELECT, UPDATE, DELETE ON rf.connectors TO rf_worker;  -- DELETE: delete_connector job
        GRANT DELETE ON rf.uploads TO rf_worker;
        GRANT SELECT, INSERT, UPDATE, DELETE ON
            rf.sync_cursors, rf.documents, rf.document_links, rf.chunks, rf.embeddings,
            rf.embedding_touches, rf.collection_space_stats, rf.conversation_summaries
            TO rf_worker;
        GRANT SELECT, INSERT ON rf.audit_log, rf.usage_events TO rf_worker;
        """
    )


def downgrade() -> None:
    raise NotImplementedError("0001 is the base schema; restore from backup instead.")
