"""ADR-0017 §1 (r3 P0-C, r4 P1/P2): the paired eval tenant and the frozen eval snapshot.

OWNED BY EV-SNAP. Additive (expand only):

- `tenants.kind` gains 'eval'; `tenants.eval_of_tenant_id` pairs an eval tenant with the ONE
  personal/team tenant whose notes it may hold (unique, set only at bootstrap, CHECKed both
  ways, ON DELETE CASCADE: deleting the owner's tenant deletes the full copy of their notes).
- `rf.eval_snapshots` (authoritative): one row per snapshot collection in the eval tenant, with
  its synthetic `uploads` row, the job that last changed it and the two digests.
- Named cross-tenant paths (ADR-0013 rules: pinned search_path with pg_temp last, REVOKE ALL
  FROM PUBLIC, explicit grants, a cross-tenant test each in tests/module_eval_snap):

  | Function                                   | Kind     | Grant     |
  |--------------------------------------------|----------|-----------|
  | provision_eval_tenant(owner)               | definer  | rf_api    |
  | paired_eval_tenant()                       | definer  | rf_api, rf_worker |
  | begin_eval_snapshot(src, target, id, job)  | definer  | rf_api    |
  | copy_collection_to_eval(src, target, id)   | definer  | rf_worker |
  | finish_eval_reembed(collection, space)     | definer  | rf_worker |
  | _purge_eval_snapshot_pages() (trigger)     | definer  | none      |
  | _eval_copy_guard, _eval_snapshot_digests, _eval_snapshot_lock_key, _eval_page_key,
  |   _netstring, _uuid5                       | invoker  | none (called by the definers) |

  The SOURCE tenant is always `rf.current_tenant()` (the calling session's app.tenant_id),
  never a parameter; the target must be `kind = 'eval' AND eval_of_tenant_id = source`.
  "Admin only" is enforced in two places: the route requires the `admin` scope of the source
  tenant to enqueue a copy, and only the job role (rf_worker) can EXECUTE the copy itself.
  copy_collection_to_eval takes the snapshot id as a third argument: it is the idempotency key
  that makes a retried job a no-op instead of a second copy.

Concurrency: copies take a SHARED transaction advisory lock on the eval tenant, the exclusion
purge an EXCLUSIVE one, so an exclusion that commits while a copy runs is either seen by the
copy's exclusion filter or purges the copied page right after the copy commits (READ COMMITTED:
each statement after the lock wait takes a new snapshot). No percent signs anywhere below:
run_sql goes through psycopg's paramstyle.

Revision ID: 0012
Revises: 0011
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, classify, run_sql, tenant_table

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

# domain.ids.RAGFORGE_NAMESPACE, spelled out in the SQL below so this revision never depends on
# other code (a test asserts the two stay equal).
RAGFORGE_NAMESPACE = "6f1c2a9e-3b7d-5c44-9a0e-2f8d1b7c4e53"

_TABLES = """
ALTER TABLE rf.tenants DROP CONSTRAINT tenants_kind_check;
ALTER TABLE rf.tenants ADD CONSTRAINT tenants_kind_check
    CHECK (kind IN ('personal', 'team', 'system', 'eval'));
ALTER TABLE rf.tenants
    ADD COLUMN eval_of_tenant_id uuid UNIQUE REFERENCES rf.tenants (id) ON DELETE CASCADE,
    ADD CONSTRAINT tenants_eval_pairing CHECK ((kind = 'eval') = (eval_of_tenant_id IS NOT NULL)),
    ADD CONSTRAINT tenants_eval_not_self CHECK (eval_of_tenant_id IS DISTINCT FROM id);

CREATE TABLE rf.eval_snapshots (
    collection_id        uuid PRIMARY KEY,
    tenant_id            uuid NOT NULL,
    upload_id            uuid NOT NULL,
    source_collection_id uuid NOT NULL,  -- no FK: the source may be deleted, the snapshot stays
    job_id               uuid NOT NULL,  -- the job that last changed (or will change) it
    space_id             integer REFERENCES rf.embedding_spaces,
    pages                integer NOT NULL DEFAULT 0 CHECK (pages >= 0),
    chunks               integer NOT NULL DEFAULT 0 CHECK (chunks >= 0),
    vectors              integer NOT NULL DEFAULT 0 CHECK (vectors >= 0),
    links                integer NOT NULL DEFAULT 0 CHECK (links >= 0),
    excluded             integer NOT NULL DEFAULT 0 CHECK (excluded >= 0),
    purged_pages         integer NOT NULL DEFAULT 0 CHECK (purged_pages >= 0),
    content_digest       text CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    index_digest         text CHECK (index_digest ~ '^[0-9a-f]{64}$'),
    created_at           timestamptz NOT NULL DEFAULT now(),
    ready_at             timestamptz,
    purged_at            timestamptz,
    UNIQUE (tenant_id, collection_id),
    UNIQUE (tenant_id, upload_id),
    FOREIGN KEY (tenant_id, collection_id) REFERENCES rf.collections (tenant_id, id)
        ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, upload_id) REFERENCES rf.uploads (tenant_id, id) ON DELETE CASCADE,
    CHECK ((ready_at IS NULL) = (content_digest IS NULL)),
    CHECK ((content_digest IS NULL) = (index_digest IS NULL)),
    CHECK ((ready_at IS NULL) = (space_id IS NULL))
);
"""

_HELPERS = """
-- uuidv5 exactly as Python's uuid.uuid5 (RFC 4122 4.3): sha1(namespace bytes || name), version
-- nibble 5, variant 10. Used so snapshot ids equal domain.ids.document_id/chunk_id (frozen).
CREATE FUNCTION rf._uuid5(p_namespace uuid, p_name text) RETURNS uuid
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    SELECT encode(set_byte(set_byte(h, 6, (get_byte(h, 6) & 15) | 80),
                           8, (get_byte(h, 8) & 63) | 128), 'hex')::uuid
      FROM (SELECT substring(ext.digest(decode(replace(p_namespace::text, '-', ''), 'hex')
                                        || convert_to(p_name, 'UTF8'), 'sha1')
                             FROM 1 FOR 16) AS h) s
$$;
REVOKE ALL ON FUNCTION rf._uuid5(uuid, text) FROM PUBLIC;

-- "<utf-8 byte length>:<text>,": the unambiguous field encoding of both digests
-- (services/eval_snapshot/digests.py is the reference implementation; a test pins the two).
CREATE FUNCTION rf._netstring(p_value text) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$ SELECT octet_length(p_value)::text || ':' || p_value || ',' $$;
REVOKE ALL ON FUNCTION rf._netstring(text) FROM PUBLIC;

-- ADR-0017 principle 6: notion -> the page id; file -> metadata.filename (the upload's
-- external_id is a random id per install). Same rule as /v1/eval/resolve.
CREATE FUNCTION rf._eval_page_key(p_source_kind text, p_external_id text, p_metadata jsonb)
    RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    SELECT CASE WHEN p_source_kind = 'file'
                THEN coalesce(p_metadata ->> 'filename', p_external_id)
                ELSE p_external_id END
$$;
REVOKE ALL ON FUNCTION rf._eval_page_key(text, text, jsonb) FROM PUBLIC;

CREATE FUNCTION rf._eval_snapshot_lock_key(p_eval_tenant uuid) RETURNS bigint
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$ SELECT hashtextextended('rf.eval_snapshot:' || p_eval_tenant::text, 0) $$;
REVOKE ALL ON FUNCTION rf._eval_snapshot_lock_key(uuid) FROM PUBLIC;

-- Both digests of one snapshot collection, from what is committed now. INVOKER and never
-- granted: only the definers below call it, as the owner, so the explicit tenant predicates
-- ARE the isolation.
--   content: sorted pages (kind, key, part), part = kind, key, chunk count, display texts in
--            ordinal order. Equal notes give equal digests across copies, tenants and installs.
--   index:   the active space id, then every chunk's sorted (chunk_key with occurrence,
--            embed_sha256). Changes on a re-embed; the gate only records it.
CREATE FUNCTION rf._eval_snapshot_digests(p_tenant uuid, p_collection uuid)
    RETURNS TABLE (content_digest text, index_digest text)
    LANGUAGE sql STABLE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    WITH pages AS (
        SELECT d.source_kind AS kind,
               rf._eval_page_key(d.source_kind, d.external_id, d.metadata) AS key,
               rf._netstring(d.source_kind)
                 || rf._netstring(rf._eval_page_key(d.source_kind, d.external_id, d.metadata))
                 || rf._netstring(count(c.id)::text)
                 || coalesce(string_agg(rf._netstring(c.display_text), '' ORDER BY c.ordinal),
                             '') AS part
          FROM rf.documents d
          LEFT JOIN rf.chunks c ON c.tenant_id = p_tenant AND c.document_id = d.id
         WHERE d.tenant_id = p_tenant AND d.collection_id = p_collection
           AND d.deleted_at IS NULL
         GROUP BY d.id
    ), keyed AS (
        SELECT d.source_kind AS kind,
               rf._eval_page_key(d.source_kind, d.external_id, d.metadata) AS key,
               c.heading_path, c.display_sha256, c.embed_sha256,
               row_number() OVER (PARTITION BY c.document_id, c.heading_path, c.display_sha256
                                  ORDER BY c.ordinal) - 1 AS occurrence
          FROM rf.chunks c
          JOIN rf.documents d ON d.tenant_id = p_tenant AND d.id = c.document_id
                             AND d.deleted_at IS NULL
         WHERE c.tenant_id = p_tenant AND c.collection_id = p_collection
    ), entries AS (
        SELECT rf._netstring(k.kind) || rf._netstring(k.key)
                 || rf._netstring(cardinality(k.heading_path)::text)
                 || coalesce((SELECT string_agg(rf._netstring(h.v), '' ORDER BY h.i)
                                FROM unnest(k.heading_path) WITH ORDINALITY AS h(v, i)), '')
                 || rf._netstring(encode(k.display_sha256, 'hex'))
                 || rf._netstring(k.occurrence::text)
                 || rf._netstring(encode(k.embed_sha256, 'hex')) AS part
          FROM keyed k
    )
    SELECT
        encode(sha256(convert_to(
            rf._netstring('ragforge.eval_snapshot.content.v1')
            || coalesce((SELECT string_agg(p.part, '' ORDER BY p.kind COLLATE "C",
                                           p.key COLLATE "C", p.part COLLATE "C")
                           FROM pages p), ''),
            'UTF8')), 'hex'),
        encode(sha256(convert_to(
            rf._netstring('ragforge.eval_snapshot.index.v1')
            || rf._netstring(coalesce((SELECT col.active_space_id::text FROM rf.collections col
                                        WHERE col.tenant_id = p_tenant
                                          AND col.id = p_collection), ''))
            || coalesce((SELECT string_agg(e.part, '' ORDER BY e.part COLLATE "C")
                           FROM entries e), ''),
            'UTF8')), 'hex')
$$;
REVOKE ALL ON FUNCTION rf._eval_snapshot_digests(uuid, uuid) FROM PUBLIC;

-- The r4 P1 guard, shared by begin and copy. Returns the source tenant (the session's).
-- A collection the caller cannot see and one that does not exist are the same error (P0002),
-- so the guard does not reveal other tenants' ids.
CREATE FUNCTION rf._eval_copy_guard(p_source_collection_id uuid, p_target_tenant_id uuid)
    RETURNS uuid
    LANGUAGE plpgsql STABLE
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_source uuid := rf.current_tenant();
BEGIN
    IF v_source IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'eval snapshots require a tenant session',
            ERRCODE = '42501';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM rf.tenants t
                    WHERE t.id = v_source AND t.kind IN ('personal', 'team')) THEN
        RAISE EXCEPTION USING MESSAGE = 'only a personal or team tenant can be snapshotted',
            ERRCODE = '42501';
    END IF;
    IF p_target_tenant_id IS NULL OR NOT EXISTS (
           SELECT 1 FROM rf.tenants t
            WHERE t.id = p_target_tenant_id AND t.kind = 'eval'
              AND t.eval_of_tenant_id = v_source) THEN
        RAISE EXCEPTION USING MESSAGE = 'the target is not this tenant''s eval tenant',
            ERRCODE = '42501';
    END IF;
    IF p_source_collection_id IS NULL OR NOT EXISTS (
           SELECT 1 FROM rf.collections c
            WHERE c.tenant_id = v_source AND c.id = p_source_collection_id) THEN
        RAISE EXCEPTION USING MESSAGE = 'source collection not found', ERRCODE = 'P0002';
    END IF;
    RETURN v_source;
END
$$;
REVOKE ALL ON FUNCTION rf._eval_copy_guard(uuid, uuid) FROM PUBLIC;
"""

_PROVISION = """
-- Bootstrap: the CALLER's tenant gets its one eval tenant (+ its owner user). The route/CLI
-- then mints the eval key (eval + chat + read) under the new tenant's RLS, in the same tx.
CREATE FUNCTION rf.provision_eval_tenant(p_owner_display_name text)
    RETURNS TABLE (tenant_id uuid, user_id uuid)
    LANGUAGE plpgsql VOLATILE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
#variable_conflict use_column
DECLARE
    v_source uuid := rf.current_tenant();
    v_name   text;
    v_owner  text := btrim(coalesce(p_owner_display_name, ''));
    t        uuid;
    u        uuid;
BEGIN
    SELECT s.name INTO v_name FROM rf.tenants s
     WHERE s.id = v_source AND s.kind IN ('personal', 'team');
    IF v_name IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'only a personal or team tenant can have an eval tenant',
            ERRCODE = '42501';
    END IF;
    IF length(v_owner) NOT BETWEEN 1 AND 200 THEN
        RAISE EXCEPTION USING MESSAGE = 'owner display name must be 1..200 characters',
            ERRCODE = '22023';
    END IF;
    -- A second eval tenant violates the UNIQUE on eval_of_tenant_id (23505): exactly one pair.
    INSERT INTO rf.tenants (name, kind, eval_of_tenant_id)
        VALUES (left('eval: ' || v_name, 200), 'eval', v_source)
        RETURNING id INTO t;
    INSERT INTO rf.users (tenant_id, display_name) VALUES (t, v_owner) RETURNING id INTO u;
    INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
        VALUES (t, 'system:provision_eval_tenant', 'tenant.provisioned', t::text,
                jsonb_build_object('kind', 'eval', 'eval_of_tenant_id', v_source));
    RETURN QUERY SELECT t, u;
END
$$;
REVOKE ALL ON FUNCTION rf.provision_eval_tenant(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.provision_eval_tenant(text) TO rf_api;

-- The caller's paired eval tenant, or NULL. Ids only.
CREATE FUNCTION rf.paired_eval_tenant() RETURNS uuid
    LANGUAGE sql STABLE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
    SELECT t.id FROM rf.tenants t
     WHERE t.kind = 'eval' AND t.eval_of_tenant_id = rf.current_tenant()
$$;
REVOKE ALL ON FUNCTION rf.paired_eval_tenant() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.paired_eval_tenant() TO rf_api, rf_worker;
"""

_BEGIN = """
-- The request: an empty snapshot collection + its synthetic upload (no blob: never processed,
-- never re-ingested) + the pending eval_snapshots row, in the eval tenant, all bound to the copy
-- job the caller enqueued in ITS OWN tenant in the same transaction.
CREATE FUNCTION rf.begin_eval_snapshot(
        p_source_collection_id uuid, p_target_tenant_id uuid, p_snapshot_id uuid,
        p_job_id uuid)
    RETURNS text
    LANGUAGE plpgsql VOLATILE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_source uuid := rf._eval_copy_guard(p_source_collection_id, p_target_tenant_id);
    src      rf.collections;
    v_name   text;
    v_upload uuid;
BEGIN
    IF p_snapshot_id IS NULL OR p_job_id IS NULL OR NOT EXISTS (
           SELECT 1 FROM rf.jobs j
            WHERE j.id = p_job_id AND j.tenant_id = v_source
              AND j.kind = 'eval_snapshot_copy') THEN
        RAISE EXCEPTION USING MESSAGE = 'a queued eval_snapshot_copy job of this tenant is required',
            ERRCODE = '22023';
    END IF;
    SELECT * INTO src FROM rf.collections c
     WHERE c.tenant_id = v_source AND c.id = p_source_collection_id;
    v_name := 'eval-snapshot-' || to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD') || '-'
              || left(p_snapshot_id::text, 8);
    INSERT INTO rf.collections (id, tenant_id, name, active_space_id, fts_config, tau)
        VALUES (p_snapshot_id, p_target_tenant_id, v_name, src.active_space_id, src.fts_config,
                src.tau);
    INSERT INTO rf.uploads (tenant_id, collection_id, blob_key, filename, sha256, size_bytes)
        VALUES (p_target_tenant_id, p_snapshot_id, 'eval-snapshot/' || p_snapshot_id::text,
                v_name, sha256(convert_to('rf.eval_snapshot:' || p_snapshot_id::text, 'UTF8')),
                0)
        RETURNING id INTO v_upload;
    INSERT INTO rf.eval_snapshots (collection_id, tenant_id, upload_id, source_collection_id,
                                   job_id)
        VALUES (p_snapshot_id, p_target_tenant_id, v_upload, p_source_collection_id, p_job_id);
    INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
        VALUES (p_target_tenant_id, 'system:begin_eval_snapshot', 'eval_snapshot.requested',
                p_snapshot_id::text, jsonb_build_object('job_id', p_job_id));
    RETURN v_name;
END
$$;
REVOKE ALL ON FUNCTION rf.begin_eval_snapshot(uuid, uuid, uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.begin_eval_snapshot(uuid, uuid, uuid, uuid) TO rf_api;
"""

_COPY = """
-- The copy, as one statement per table inside the job's fenced transaction (no model call, so
-- one transaction is correct and makes the digest describe exactly what committed). Copies the
-- live, NOT excluded documents (one per page key: DISTINCT ON keeps the lowest id when two
-- connectors hold the same page), their chunks, their vectors in the source's ACTIVE space (no
-- re-embed), embedding touches and stats (so GC and the exact/HNSW switch behave as for any
-- collection), and their links (minus links to excluded pages). A chunk without a vector means
-- the source is still indexing: refuse (55000, retryable) rather than freeze a hole into the
-- baseline. Idempotent on p_snapshot_id: a retry after the copy committed returns its counts.
CREATE FUNCTION rf.copy_collection_to_eval(
        p_source_collection_id uuid, p_target_tenant_id uuid, p_snapshot_id uuid)
    RETURNS TABLE (copied_pages integer, copied_chunks integer, copied_vectors integer,
                   copied_links integer, skipped_excluded integer, content_digest_hex text,
                   index_digest_hex text)
    LANGUAGE plpgsql VOLATILE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
#variable_conflict use_column
DECLARE
    v_source   uuid := rf._eval_copy_guard(p_source_collection_id, p_target_tenant_id);
    v_ns       uuid := '6f1c2a9e-3b7d-5c44-9a0e-2f8d1b7c4e53'::uuid;
    s          rf.eval_snapshots;
    v_space    integer;
    v_docs     uuid[];
    v_pages    integer;
    v_chunks   integer;
    v_vectors  integer;
    v_links    integer;
    v_excluded integer;
    v_missing  integer;
    v_content  text;
    v_index    text;
BEGIN
    PERFORM pg_advisory_xact_lock_shared(rf._eval_snapshot_lock_key(p_target_tenant_id));
    SELECT * INTO s FROM rf.eval_snapshots e
     WHERE e.tenant_id = p_target_tenant_id AND e.collection_id = p_snapshot_id
       FOR UPDATE;
    IF s.collection_id IS NULL OR s.source_collection_id <> p_source_collection_id THEN
        RAISE EXCEPTION USING MESSAGE = 'no such eval snapshot for this source collection',
            ERRCODE = 'P0002';
    END IF;
    IF s.ready_at IS NOT NULL THEN
        RETURN QUERY SELECT s.pages, s.chunks, s.vectors, s.links, s.excluded,
                            s.content_digest, s.index_digest;
        RETURN;
    END IF;

    SELECT c.active_space_id INTO v_space FROM rf.collections c
     WHERE c.tenant_id = v_source AND c.id = p_source_collection_id;
    SELECT count(*)::integer INTO v_excluded FROM rf.documents d
     WHERE d.tenant_id = v_source AND d.collection_id = p_source_collection_id
       AND d.deleted_at IS NULL
       AND EXISTS (SELECT 1 FROM rf.exclusions x
                    WHERE x.tenant_id = v_source AND x.source_kind = d.source_kind
                      AND x.external_id = d.external_id);
    SELECT coalesce(array_agg(t.id ORDER BY t.id), '{}'::uuid[]) INTO v_docs
      FROM (SELECT DISTINCT ON (d.source_kind, d.external_id) d.id
              FROM rf.documents d
             WHERE d.tenant_id = v_source AND d.collection_id = p_source_collection_id
               AND d.deleted_at IS NULL
               AND NOT EXISTS (SELECT 1 FROM rf.exclusions x
                                WHERE x.tenant_id = v_source AND x.source_kind = d.source_kind
                                  AND x.external_id = d.external_id)
             ORDER BY d.source_kind, d.external_id, d.id) t;

    -- Upload-owned, so the live connector's ON DELETE CASCADE can never reach the copy.
    INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, upload_id, source_kind,
                              external_id, external_url, title, external_updated_at,
                              fetched_at, page_sha256, synced_source_ids, metadata)
    SELECT rf._uuid5(v_ns, concat_ws(chr(31), 'doc', p_target_tenant_id::text,
                                     p_snapshot_id::text, 'upload', d.external_id)),
           p_target_tenant_id, p_snapshot_id, NULL, s.upload_id, d.source_kind, d.external_id,
           d.external_url, d.title, d.external_updated_at, d.fetched_at, d.page_sha256,
           d.synced_source_ids, d.metadata
      FROM rf.documents d
     WHERE d.tenant_id = v_source AND d.id = ANY (v_docs);
    GET DIAGNOSTICS v_pages = ROW_COUNT;

    INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,
                           heading_path, display_text, display_sha256, embed_sha256,
                           token_count, metadata)
    SELECT rf._uuid5(v_ns, concat_ws(chr(31), 'chunk', nd.id::text, c.ordinal::text)),
           p_target_tenant_id, p_snapshot_id, nd.id, c.anchor, c.ordinal, c.heading_path,
           c.display_text, c.display_sha256, c.embed_sha256, c.token_count, c.metadata
      FROM rf.chunks c
      JOIN rf.documents sd ON sd.tenant_id = v_source AND sd.id = c.document_id
      JOIN rf.documents nd ON nd.tenant_id = p_target_tenant_id
                          AND nd.collection_id = p_snapshot_id
                          AND nd.source_kind = sd.source_kind
                          AND nd.external_id = sd.external_id
     WHERE c.tenant_id = v_source AND c.document_id = ANY (v_docs);
    GET DIAGNOSTICS v_chunks = ROW_COUNT;

    INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding)
    SELECT p_target_tenant_id, p_snapshot_id, e.space_id, e.embed_sha256, e.embedding
      FROM rf.embeddings e
     WHERE e.tenant_id = v_source AND e.collection_id = p_source_collection_id
       AND e.space_id = v_space
       AND e.embed_sha256 IN (SELECT c.embed_sha256 FROM rf.chunks c
                               WHERE c.tenant_id = p_target_tenant_id
                                 AND c.collection_id = p_snapshot_id);
    GET DIAGNOSTICS v_vectors = ROW_COUNT;

    SELECT count(*)::integer INTO v_missing
      FROM (SELECT DISTINCT c.embed_sha256 FROM rf.chunks c
             WHERE c.tenant_id = p_target_tenant_id AND c.collection_id = p_snapshot_id) h
     WHERE NOT EXISTS (SELECT 1 FROM rf.embeddings e
                        WHERE e.tenant_id = p_target_tenant_id
                          AND e.collection_id = p_snapshot_id
                          AND e.space_id = v_space AND e.embed_sha256 = h.embed_sha256);
    IF v_missing > 0 THEN
        RAISE EXCEPTION USING MESSAGE = 'the source collection is still indexing ('
            || v_missing::text || ' chunk texts have no vector yet); the job retries',
            ERRCODE = '55000';
    END IF;

    INSERT INTO rf.embedding_touches (tenant_id, collection_id, space_id, embed_sha256)
    SELECT e.tenant_id, e.collection_id, e.space_id, e.embed_sha256 FROM rf.embeddings e
     WHERE e.tenant_id = p_target_tenant_id AND e.collection_id = p_snapshot_id;
    INSERT INTO rf.collection_space_stats (tenant_id, collection_id, space_id, vector_count)
        VALUES (p_target_tenant_id, p_snapshot_id, v_space, v_vectors);

    INSERT INTO rf.document_links (tenant_id, from_document_id, to_external_id)
    SELECT p_target_tenant_id, nd.id, l.to_external_id
      FROM rf.document_links l
      JOIN rf.documents sd ON sd.tenant_id = v_source AND sd.id = l.from_document_id
      JOIN rf.documents nd ON nd.tenant_id = p_target_tenant_id
                          AND nd.collection_id = p_snapshot_id
                          AND nd.source_kind = sd.source_kind
                          AND nd.external_id = sd.external_id
     WHERE l.tenant_id = v_source AND l.from_document_id = ANY (v_docs)
       AND NOT EXISTS (SELECT 1 FROM rf.exclusions x
                        WHERE x.tenant_id = v_source AND x.source_kind = sd.source_kind
                          AND x.external_id = l.to_external_id);
    GET DIAGNOSTICS v_links = ROW_COUNT;

    UPDATE rf.collections c SET active_space_id = v_space, building_space_id = NULL
     WHERE c.tenant_id = p_target_tenant_id AND c.id = p_snapshot_id;
    SELECT g.content_digest, g.index_digest INTO v_content, v_index
      FROM rf._eval_snapshot_digests(p_target_tenant_id, p_snapshot_id) g;
    UPDATE rf.eval_snapshots e
       SET space_id = v_space, pages = v_pages, chunks = v_chunks, vectors = v_vectors,
           links = v_links, excluded = v_excluded, content_digest = v_content,
           index_digest = v_index, ready_at = now()
     WHERE e.collection_id = p_snapshot_id;
    INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
    SELECT t.tenant, 'system:copy_collection_to_eval', 'eval_snapshot.copied',
           p_snapshot_id::text,
           jsonb_build_object('pages', v_pages, 'chunks', v_chunks, 'vectors', v_vectors,
                              'links', v_links, 'excluded', v_excluded, 'space_id', v_space)
      FROM (VALUES (v_source), (p_target_tenant_id)) AS t(tenant);
    RETURN QUERY SELECT v_pages, v_chunks, v_vectors, v_links, v_excluded, v_content, v_index;
END
$$;
REVOKE ALL ON FUNCTION rf.copy_collection_to_eval(uuid, uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.copy_collection_to_eval(uuid, uuid, uuid) TO rf_worker;
"""

_REEMBED = """
-- The re-embed's final, fenced step (the job inserted the new space's vectors just before, in
-- the same transaction). Refuses unless EVERY chunk has a vector in the new space, so the flip
-- is never partial; flips the snapshot's active space; drops vectors no chunk references any
-- more (an exclusion purge may have removed pages meanwhile); records the new index digest.
-- Tenant = the session's (the eval tenant); vectors of the old space are kept, so re-embedding
-- back to it is an instant flip (nothing missing to embed).
CREATE FUNCTION rf.finish_eval_reembed(p_collection_id uuid, p_space_id integer)
    RETURNS TABLE (content_digest_hex text, index_digest_hex text, vector_count integer)
    LANGUAGE plpgsql VOLATILE SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
#variable_conflict use_column
DECLARE
    v_tenant  uuid := rf.current_tenant();
    s         rf.eval_snapshots;
    v_status  text;
    v_missing integer;
    v_count   integer;
    v_content text;
    v_index   text;
BEGIN
    IF v_tenant IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'eval snapshots require a tenant session',
            ERRCODE = '42501';
    END IF;
    PERFORM pg_advisory_xact_lock_shared(rf._eval_snapshot_lock_key(v_tenant));
    SELECT * INTO s FROM rf.eval_snapshots e
     WHERE e.tenant_id = v_tenant AND e.collection_id = p_collection_id
       FOR UPDATE;
    IF s.collection_id IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'eval snapshot not found', ERRCODE = 'P0002';
    END IF;
    IF s.ready_at IS NULL THEN
        RAISE EXCEPTION USING MESSAGE = 'the snapshot is still being copied', ERRCODE = '55000';
    END IF;
    SELECT sp.status INTO v_status FROM rf.embedding_spaces sp WHERE sp.id = p_space_id;
    IF v_status IS NULL OR v_status = 'retired' THEN
        RAISE EXCEPTION USING MESSAGE = 'embedding space not found or retired',
            ERRCODE = '22023';
    END IF;

    SELECT count(*)::integer INTO v_missing
      FROM (SELECT DISTINCT c.embed_sha256 FROM rf.chunks c
             WHERE c.tenant_id = v_tenant AND c.collection_id = p_collection_id) h
     WHERE NOT EXISTS (SELECT 1 FROM rf.embeddings e
                        WHERE e.tenant_id = v_tenant AND e.collection_id = p_collection_id
                          AND e.space_id = p_space_id AND e.embed_sha256 = h.embed_sha256);
    IF v_missing > 0 THEN
        RAISE EXCEPTION USING MESSAGE = v_missing::text
            || ' chunk texts have no vector in the new space', ERRCODE = '55000';
    END IF;

    DELETE FROM rf.embedding_touches t
     WHERE t.tenant_id = v_tenant AND t.collection_id = p_collection_id
       AND NOT EXISTS (SELECT 1 FROM rf.chunks c
                        WHERE c.tenant_id = v_tenant AND c.collection_id = p_collection_id
                          AND c.embed_sha256 = t.embed_sha256);
    DELETE FROM rf.embeddings e
     WHERE e.tenant_id = v_tenant AND e.collection_id = p_collection_id
       AND NOT EXISTS (SELECT 1 FROM rf.chunks c
                        WHERE c.tenant_id = v_tenant AND c.collection_id = p_collection_id
                          AND c.embed_sha256 = e.embed_sha256);
    SELECT count(*)::integer INTO v_count FROM rf.embeddings e
     WHERE e.tenant_id = v_tenant AND e.collection_id = p_collection_id
       AND e.space_id = p_space_id;
    INSERT INTO rf.collection_space_stats AS st (tenant_id, collection_id, space_id, vector_count)
        VALUES (v_tenant, p_collection_id, p_space_id, v_count)
        ON CONFLICT (collection_id, space_id)
        DO UPDATE SET vector_count = excluded.vector_count, updated_at = now();

    UPDATE rf.collections c SET active_space_id = p_space_id, building_space_id = NULL
     WHERE c.tenant_id = v_tenant AND c.id = p_collection_id;
    SELECT g.content_digest, g.index_digest INTO v_content, v_index
      FROM rf._eval_snapshot_digests(v_tenant, p_collection_id) g;
    UPDATE rf.eval_snapshots e
       SET space_id = p_space_id, vectors = v_count, content_digest = v_content,
           index_digest = v_index
     WHERE e.collection_id = p_collection_id;
    INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
        VALUES (v_tenant, 'system:finish_eval_reembed', 'eval_snapshot.reembedded',
                p_collection_id::text,
                jsonb_build_object('from_space_id', s.space_id, 'to_space_id', p_space_id,
                                   'vectors', v_count));
    RETURN QUERY SELECT v_content, v_index, v_count;
END
$$;
REVOKE ALL ON FUNCTION rf.finish_eval_reembed(uuid, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rf.finish_eval_reembed(uuid, integer) TO rf_worker;
"""

_PURGE = """
-- r4 P2 privacy: a later exclusion on the SOURCE tenant hard-deletes the matching snapshot
-- pages in its eval tenant in the same transaction as the exclusion (chunks and links cascade),
-- drops the vectors no remaining chunk references, and re-records both digests, so a run on
-- the purged snapshot can never be compared with a baseline that still had the page (the gate
-- requires equal content digests; resolve reports the page `missing`). Not undone when the
-- exclusion is forgotten: a snapshot is frozen, and a refresh is an explicit new snapshot.
-- Definer trigger function: no EXECUTE grants (like 0008's outcome triggers).
CREATE FUNCTION rf._purge_eval_snapshot_pages() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path = pg_catalog, rf, ext, pg_temp
AS $$
DECLARE
    v_eval    uuid;
    r         record;
    v_n       integer;
    v_content text;
    v_index   text;
BEGIN
    SELECT t.id INTO v_eval FROM rf.tenants t
     WHERE t.kind = 'eval' AND t.eval_of_tenant_id = NEW.tenant_id;
    IF v_eval IS NULL THEN
        RETURN NULL;
    END IF;
    PERFORM pg_advisory_xact_lock(rf._eval_snapshot_lock_key(v_eval));
    FOR r IN SELECT s.collection_id, s.upload_id, s.ready_at FROM rf.eval_snapshots s
              WHERE s.tenant_id = v_eval ORDER BY s.collection_id
    LOOP
        DELETE FROM rf.documents d
         WHERE d.tenant_id = v_eval AND d.collection_id = r.collection_id
           AND d.upload_id = r.upload_id AND d.source_kind = NEW.source_kind
           AND d.external_id = NEW.external_id;
        GET DIAGNOSTICS v_n = ROW_COUNT;
        CONTINUE WHEN v_n = 0;
        DELETE FROM rf.embedding_touches t
         WHERE t.tenant_id = v_eval AND t.collection_id = r.collection_id
           AND NOT EXISTS (SELECT 1 FROM rf.chunks c
                            WHERE c.tenant_id = v_eval AND c.collection_id = r.collection_id
                              AND c.embed_sha256 = t.embed_sha256);
        DELETE FROM rf.embeddings e
         WHERE e.tenant_id = v_eval AND e.collection_id = r.collection_id
           AND NOT EXISTS (SELECT 1 FROM rf.chunks c
                            WHERE c.tenant_id = v_eval AND c.collection_id = r.collection_id
                              AND c.embed_sha256 = e.embed_sha256);
        UPDATE rf.collection_space_stats st
           SET vector_count = (SELECT count(*) FROM rf.embeddings e
                                WHERE e.tenant_id = v_eval AND e.collection_id = r.collection_id
                                  AND e.space_id = st.space_id),
               updated_at = now()
         WHERE st.tenant_id = v_eval AND st.collection_id = r.collection_id;
        SELECT g.content_digest, g.index_digest INTO v_content, v_index
          FROM rf._eval_snapshot_digests(v_eval, r.collection_id) g;
        UPDATE rf.eval_snapshots s
           SET purged_pages = s.purged_pages + v_n, purged_at = now(),
               pages = greatest(s.pages - v_n, 0),
               chunks = (SELECT count(*) FROM rf.chunks c
                          WHERE c.tenant_id = v_eval AND c.collection_id = r.collection_id),
               vectors = (SELECT count(*) FROM rf.embeddings e
                           WHERE e.tenant_id = v_eval AND e.collection_id = r.collection_id
                             AND e.space_id = s.space_id),
               content_digest = CASE WHEN r.ready_at IS NULL THEN NULL ELSE v_content END,
               index_digest = CASE WHEN r.ready_at IS NULL THEN NULL ELSE v_index END
         WHERE s.collection_id = r.collection_id;
        INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
            VALUES (v_eval, 'system:purge_eval_snapshot_pages', 'eval_snapshot.purged',
                    r.collection_id::text,
                    jsonb_build_object('source_kind', NEW.source_kind, 'pages', v_n));
    END LOOP;
    RETURN NULL;
END
$$;
REVOKE ALL ON FUNCTION rf._purge_eval_snapshot_pages() FROM PUBLIC;
CREATE TRIGGER exclusions_purge_eval_snapshots
    AFTER INSERT ON rf.exclusions
    FOR EACH ROW EXECUTE FUNCTION rf._purge_eval_snapshot_pages();
"""

_GRANTS = """
GRANT SELECT ON rf.eval_snapshots TO rf_api, rf_worker;
-- The reembed route records its job on the row (under the eval tenant's RLS); nothing else.
GRANT UPDATE (job_id) ON rf.eval_snapshots TO rf_api;
"""


def upgrade() -> None:
    run_sql(_TABLES)
    tenant_table("eval_snapshots")
    classify("eval_snapshots", "authoritative")
    run_sql(_HELPERS)
    run_sql(_PROVISION)
    run_sql(_BEGIN)
    run_sql(_COPY)
    run_sql(_REEMBED)
    run_sql(_PURGE)
    run_sql(_GRANTS)
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
