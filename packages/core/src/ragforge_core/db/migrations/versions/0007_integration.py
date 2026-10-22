"""Phase 2 integration: the additive DDL the Phase 1 handoffs asked the integrator for.

- `pending_approvals.executed_at` (handoff D1, request 2): the execution claim. An approved
  destructive tool call runs exactly once across LangGraph node re-runs, because the claim
  (`UPDATE ... SET executed_at = now() WHERE executed_at IS NULL RETURNING 1`) commits in the
  same transaction as the side effect. rf_api already has UPDATE on the table.
- `embedding_spaces`: `status = 'retired'` iff `retired_at` is set (handoff B, request 7), so the
  7-day purge window of a retired space always has a start. Rows that predate the rule are
  stamped first (the same thing gc_embeddings does lazily).

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

from ragforge_core.db.migrations.helpers import bump_schema_version, run_sql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql(
        """
        ALTER TABLE rf.pending_approvals ADD COLUMN executed_at timestamptz;
        ALTER TABLE rf.pending_approvals ADD CONSTRAINT pending_approvals_executed_only_approved
            CHECK (executed_at IS NULL OR status = 'approved');

        UPDATE rf.embedding_spaces SET retired_at = now()
         WHERE status = 'retired' AND retired_at IS NULL;
        UPDATE rf.embedding_spaces SET retired_at = NULL
         WHERE status <> 'retired' AND retired_at IS NOT NULL;
        ALTER TABLE rf.embedding_spaces ADD CONSTRAINT embedding_spaces_retired_at_iff_retired
            CHECK ((status = 'retired') = (retired_at IS NOT NULL));
        """
    )
    # Keep bump_schema_version() last.
    bump_schema_version()


def downgrade() -> None:
    raise NotImplementedError("forward-only; expand/contract (ADR-0002)")
