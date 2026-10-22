"""Chat-model factory: the one place that maps `RAGFORGE_LLM_PROVIDER` to a ChatModel adapter.

Why a factory rather than an `if` in bootstrap: the choice is configuration-driven and tested
here in isolation (no database, no agent), and bootstrap stays a pure wiring list. Embeddings are
NOT switched by this setting: they are pinned per embedding space (ADR-0009).
"""

from __future__ import annotations

from typing import Protocol, assert_never

import httpx

from ragforge_core.adapters.llm.chat_gemini import GeminiChatModel
from ragforge_core.adapters.llm.chat_ollama import OllamaChatModel
from ragforge_core.config import LLMProvider, Settings
from ragforge_core.domain.ports import ChatModel


class ManagedChatModel(ChatModel, Protocol):
    """A ChatModel that owns resources (its HTTP pool): the composition root closes it."""

    async def aclose(self) -> None: ...


def build_chat_model(
    settings: Settings, client: httpx.AsyncClient | None = None
) -> ManagedChatModel:
    """Build the configured chat adapter. `client` (tests, or a shared pool) must already point
    at the chosen provider's base URL; without one the adapter opens and owns its own.

    Fails fast (no network call): an unusable configuration raises here, at startup, instead
    of on the first chat turn. Settings already reject Gemini without a key; the adapter checks
    again because Settings can be built with validation skipped (`model_construct`)."""
    provider = settings.llm_provider
    if provider is LLMProvider.ollama:
        return OllamaChatModel.from_settings(settings, client)
    if provider is LLMProvider.gemini:
        return GeminiChatModel.from_settings(settings, client)
    assert_never(provider)
