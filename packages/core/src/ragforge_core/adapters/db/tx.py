"""The one place adapters turn the opaque domain `Tx` into a SQLAlchemy connection."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_core.domain.ports import Tx


def connection(tx: Tx) -> AsyncConnection:
    if not isinstance(tx, AsyncConnection):
        raise TypeError(f"expected an AsyncConnection from TransactionManager, got {type(tx)!r}")
    return tx
