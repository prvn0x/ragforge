"""Why a document is tombstoned (Phase 2a review round 4, P1: undo of a Notion delete).

A tombstone has two very different causes, and they need different revival rules:

  excluded        the user (API DELETE /documents) or an approved agent tool removed it. The
                  content is intact; undo (forgetting the exclusion) must bring it back at once.
  source_deleted  sync/reconcile confirmed the item is gone at the source. Only a fetch that
                  STARTED after the tombstone may revive it (handoff B's sticky-tombstone rule),
                  so a sync that fetched before the deletion and committed after cannot
                  resurrect the owner's deleted content.

Before this slot both shared the sticky rule, so an immediate undo of a Notion delete failed: the
re-fetch's fetched_at is the Notion Date header minus 60 s, earlier than the tombstone, and the
page silently stayed deleted while the API answered 204.

`deleted_reason` is normalised by a BEFORE trigger: clearing deleted_at clears the reason, and
a tombstone written without one is 'source_deleted' (the safe, sticky default), so every writer
that does not opt in keeps the old behaviour. Writers that mean "excluded" set it explicitly.
The CHECK then holds for every writer, including ones that only touch deleted_at.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql(
        """
        ALTER TABLE rf.documents
            ADD COLUMN deleted_reason text
                CHECK (deleted_reason IN ('excluded', 'source_deleted'));
        UPDATE rf.documents SET deleted_reason = 'source_deleted'
         WHERE deleted_at IS NOT NULL;
        ALTER TABLE rf.documents ADD CONSTRAINT documents_deleted_reason_iff_deleted
            CHECK ((deleted_at IS NULL) = (deleted_reason IS NULL));

        -- INVOKER trigger function: no grants, pinned search_path, never callable directly.
        CREATE FUNCTION rf._documents_deleted_reason() RETURNS trigger
            LANGUAGE plpgsql
            SET search_path = pg_catalog, rf, pg_temp
        AS $fn$
        BEGIN
            IF NEW.deleted_at IS NULL THEN
                NEW.deleted_reason := NULL;
            ELSIF NEW.deleted_reason IS NULL THEN
                NEW.deleted_reason := 'source_deleted';
            END IF;
            RETURN NEW;
        END
        $fn$;
        REVOKE ALL ON FUNCTION rf._documents_deleted_reason() FROM PUBLIC;
        CREATE TRIGGER documents_deleted_reason
            BEFORE INSERT OR UPDATE OF deleted_at, deleted_reason ON rf.documents
            FOR EACH ROW EXECUTE FUNCTION rf._documents_deleted_reason();
        """
    )
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
