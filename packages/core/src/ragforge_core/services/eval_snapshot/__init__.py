"""The eval tenant and the frozen eval snapshot (ADR-0017 §1 snapshot row, r3 P0-C, r4).

A snapshot is a copy of one of the owner's collections in the owner's PAIRED eval tenant, so
gates and baselines compare code on fixed notes while the daily UI, `make ask` and MCP never
see duplicate pages. The copy itself is SQL (migration 0012, `rf.copy_collection_to_eval`):
the source tenant is the session's, the target must be its paired eval tenant, and it runs as
one transaction so the recorded digests describe exactly what committed.

Modules:
    policy       job kinds, payload models, the eval key's scopes (pure)
    digests      reference implementation of the content/index digests (pure; pins the SQL)
    embed_input  the ingest path's embed-input rule, re-checked against embed_sha256 (pure)
    errors       the refusals (SQLSTATE -> domain error) shared by the route and the worker
    ports        what the job handlers need from storage
    jobs         the copy and re-embed job handlers
"""

from ragforge_core.services.eval_snapshot.errors import (
    EvalTenantExists,
    SnapshotNotReady,
    refusal_for,
)
from ragforge_core.services.eval_snapshot.jobs import (
    EvalSnapshotDeps,
    run_copy_job,
    run_reembed_job,
)
from ragforge_core.services.eval_snapshot.policy import (
    EVAL_KEY_NAME,
    EVAL_KEY_SCOPES,
    EVAL_SNAPSHOT_KINDS,
    KIND_SNAPSHOT_COPY,
    KIND_SNAPSHOT_REEMBED,
    MAX_RUNTIME_S,
    copy_job,
    reembed_job,
)
from ragforge_core.services.eval_snapshot.ports import (
    CopyOutcome,
    EmbedSource,
    ReembedOutcome,
    SnapshotStore,
)

__all__ = [
    "EVAL_KEY_NAME",
    "EVAL_KEY_SCOPES",
    "EVAL_SNAPSHOT_KINDS",
    "KIND_SNAPSHOT_COPY",
    "KIND_SNAPSHOT_REEMBED",
    "MAX_RUNTIME_S",
    "CopyOutcome",
    "EmbedSource",
    "EvalSnapshotDeps",
    "EvalTenantExists",
    "ReembedOutcome",
    "SnapshotNotReady",
    "SnapshotStore",
    "copy_job",
    "reembed_job",
    "refusal_for",
    "run_copy_job",
    "run_reembed_job",
]
