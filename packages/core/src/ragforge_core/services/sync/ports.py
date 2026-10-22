"""Module-local ports of the sync service (module C2).

`SyncStore` is the persistence the sync handlers need beyond the frozen domain ports: connector
rows (including the sealed token), cursors, exclusions, uploads, the document facts used by the
freshness rule, and audit rows. The SQL implementation lives in `adapters/connectors/store.py`.

Adapters may import only domain types, so every signature here uses domain types and
primitives, and the row types are read-only Protocols that the adapter's own dataclasses
satisfy structurally (the adapter never imports this module).

Every method runs inside the caller's tenant `tx` (`TransactionManager.tenant`); the tenant is
ambient (RLS), never a parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from ragforge_core.domain.models import SealedSecret, SourceKind
from ragforge_core.domain.ports import SourceConnector, Tx


# ----------------------------------------------------------------- row views (read-only)
class ConnectorState(Protocol):
    @property
    def id(self) -> UUID: ...
    @property
    def kind(self) -> str: ...
    @property
    def target_collection_id(self) -> UUID: ...
    @property
    def config(self) -> Mapping[str, Any]: ...
    @property
    def status(self) -> str: ...
    @property
    def sealed(self) -> SealedSecret | None:
        """None once the secret was destroyed (DELETE /connectors) or never stored."""
        ...


class UploadState(Protocol):
    @property
    def id(self) -> UUID: ...
    @property
    def collection_id(self) -> UUID: ...
    @property
    def blob_key(self) -> str: ...
    @property
    def filename(self) -> str: ...
    @property
    def sha256(self) -> bytes: ...
    @property
    def size_bytes(self) -> int: ...
    @property
    def created_at(self) -> datetime: ...


class KnownDocument(Protocol):
    """A live (not tombstoned) document of a connector, as reconcile needs it."""

    @property
    def id(self) -> UUID: ...
    @property
    def external_id(self) -> str: ...
    @property
    def fetched_at(self) -> datetime | None: ...
    @property
    def synced_source_ids(self) -> Sequence[str]: ...


# ----------------------------------------------------------------- persistence
@runtime_checkable
class SyncStore(Protocol):
    # connectors -----------------------------------------------------------------
    async def load_connector(self, tx: Tx, connector_id: UUID) -> ConnectorState | None: ...

    async def mark_connector_synced(self, tx: Tx, connector_id: UUID, next_poll_in_s: int) -> None:
        """Clear last_error and schedule the next poll (only while status is 'active')."""
        ...

    async def mark_connector_error(self, tx: Tx, connector_id: UUID, message: str) -> None:
        """status='error' + last_error, only from 'active' (never overrides paused/deleting)."""
        ...

    async def destroy_secret(self, tx: Tx, connector_id: UUID) -> None:
        """NULL the ciphertext and wrapped DEK and set status 'deleting' (idempotent)."""
        ...

    async def delete_connector_documents(self, tx: Tx, connector_id: UUID, limit: int) -> int:
        """Delete up to `limit` documents (chunks/links cascade). Returns rows deleted."""
        ...

    async def delete_connector(self, tx: Tx, connector_id: UUID) -> bool: ...

    # cursors ----------------------------------------------------------------------
    async def read_cursor(self, tx: Tx, connector_id: UUID, scope: str) -> datetime | None: ...

    async def advance_cursor(
        self, tx: Tx, connector_id: UUID, scope: str, value: datetime
    ) -> datetime:
        """Monotonic: stored = GREATEST(stored, value). Returns the stored value."""
        ...

    # exclusions -------------------------------------------------------------------
    async def excluded_ids(
        self, tx: Tx, source_kind: SourceKind, external_ids: Sequence[str]
    ) -> set[str]: ...

    async def lock_source_item(self, tx: Tx, source_kind: SourceKind, external_id: str) -> None:
        """Transaction-scoped lock serializing an exclusion write against a sync commit of the
        same (tenant, source_kind, external_id), so an excluded page cannot be resurrected by a
        sync whose fenced transaction raced the exclusion."""
        ...

    async def add_exclusion(
        self, tx: Tx, source_kind: SourceKind, external_id: str, reason: str | None
    ) -> None:
        """Idempotent; takes `lock_source_item` first (used by the API's DELETE /documents)."""
        ...

    # documents (read-only facts for the freshness rule and reconcile) -------------
    async def fetched_at_by_external_id(
        self, tx: Tx, connector_id: UUID, external_ids: Sequence[str]
    ) -> dict[str, datetime | None]:
        """Live documents of the connector among `external_ids` -> stored fetched_at."""
        ...

    async def live_documents(self, tx: Tx, connector_id: UUID) -> Sequence[KnownDocument]: ...

    # uploads -----------------------------------------------------------------------
    async def insert_upload(
        self,
        tx: Tx,
        *,
        upload_id: UUID,
        collection_id: UUID,
        blob_key: str,
        filename: str,
        sha256: bytes,
        size_bytes: int,
    ) -> None: ...

    async def load_upload(self, tx: Tx, upload_id: UUID) -> UploadState | None: ...

    # audit -------------------------------------------------------------------------
    async def audit(
        self, tx: Tx, *, actor: str, action: str, target: str | None, details: Mapping[str, Any]
    ) -> None: ...


# ----------------------------------------------------------------- source connectors
@runtime_checkable
class ConnectorFactory(Protocol):
    """Builds a live SourceConnector from a decrypted token (module C1 implements it).

    An async context manager so the connector can own and close its HTTP client. The token is
    plaintext: implementations must never log it or put it in exception messages.
    """

    def open(
        self, *, connector_id: UUID, kind: str, token: bytes, config: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[SourceConnector]: ...
