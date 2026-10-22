"""LOCKED seam (ADR-0018 revision 2): which prompt a model call belongs to.

The ChatModel port carries no prompt identity, and adapters know only provider and model. The
node or route that BUILDS a prompt sets `prompt_name` (name@version) around its model call;
`InstrumentedChatModel` (OBS-METRICS) wraps the port in bootstrap and reads it. The default is
`catalog.UNKNOWN_PROMPT`, and a test asserts no chat/verifier/interview/extras path leaves it.
`surface` likewise tells owner traffic from eval/demo traffic (derived from the key's tenant
kind / route by the API; never a tenant id).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ragforge_core.observability.catalog import UNKNOWN_PROMPT

prompt_name: ContextVar[str] = ContextVar("ragforge_prompt_name", default=UNKNOWN_PROMPT)
surface: ContextVar[str] = ContextVar("ragforge_surface", default="other")


@contextmanager
def for_prompt(name: str) -> Iterator[None]:
    """Mark the model calls inside the block as belonging to prompt `name` (name@version)."""
    token = prompt_name.set(name)
    try:
        yield
    finally:
        prompt_name.reset(token)
