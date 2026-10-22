"""Notion connector (ADR-0007, DESIGN Flow A and section 7.1).

Wiring (Phase 2 / sync service):
    connectors = NotionConnectorFactory(PostgresRateLimiter(worker_engine))
    async with connectors.open(connector_id=..., kind="notion", token=..., config=...) as source:
        ...
"""

from ragforge_core.adapters.connectors.notion.client import (
    NOTION_VERSION,
    NotionClient,
    NotionResponse,
)
from ragforge_core.adapters.connectors.notion.connector import NotionConnector, NotionPageGone
from ragforge_core.adapters.connectors.notion.errors import (
    NotionApiError,
    NotionError,
    NotionNotFound,
    NotionProtocolError,
    NotionTransportError,
)
from ragforge_core.adapters.connectors.notion.factory import NotionConnectorFactory
from ragforge_core.adapters.connectors.notion.ids import anchor_id, canonical_id, deep_link

__all__ = [
    "NOTION_VERSION",
    "NotionApiError",
    "NotionClient",
    "NotionConnector",
    "NotionConnectorFactory",
    "NotionError",
    "NotionNotFound",
    "NotionPageGone",
    "NotionProtocolError",
    "NotionResponse",
    "NotionTransportError",
    "anchor_id",
    "canonical_id",
    "deep_link",
]
