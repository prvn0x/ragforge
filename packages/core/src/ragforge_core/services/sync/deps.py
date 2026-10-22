"""What the sync handlers are wired with (bootstrap builds it in Phase 2).

`PermanentJobError` is re-exported from services.jobs (one definition for the runtime and sync).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragforge_core.domain.ports import (
    BlobStore,
    Chunker,
    DocumentIndex,
    DocumentParser,
    EmbedderRegistry,
    JobQueue,
    SecretBox,
    TransactionManager,
)
from ragforge_core.services.jobs.registry import PermanentJobError
from ragforge_core.services.sync.policy import SyncPolicy
from ragforge_core.services.sync.ports import ConnectorFactory, SyncStore

__all__ = ["PermanentJobError", "SyncDeps"]


@dataclass(frozen=True, slots=True)
class SyncDeps:
    tx: TransactionManager
    queue: JobQueue
    store: SyncStore
    index: DocumentIndex
    chunker: Chunker
    embedders: EmbedderRegistry
    secrets: SecretBox
    connectors: ConnectorFactory
    blobs: BlobStore
    parser: DocumentParser
    policy: SyncPolicy = field(default_factory=SyncPolicy)
    actor: str = "worker:sync"
