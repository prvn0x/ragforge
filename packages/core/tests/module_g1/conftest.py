"""Module G1 fixtures: a Gemini adapter over a respx-mocked API, and an environment with no
LLM settings leaking in from the developer's shell (Settings reads RAGFORGE_* env vars)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from module_g1.g1_support import BASE, CANARY_KEY, MODEL
from ragforge_core.adapters.llm.chat_gemini import GeminiChatModel

_LLM_ENV = (
    "RAGFORGE_LLM_PROVIDER",
    "RAGFORGE_GEMINI_API_KEY",
    "RAGFORGE_GEMINI_MODEL",
    "RAGFORGE_GEMINI_THINKING_BUDGET",
    "RAGFORGE_CHAT_MODEL",
    "RAGFORGE_OLLAMA_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=BASE) as http:
        yield http


@pytest.fixture
def model(client: httpx.AsyncClient) -> GeminiChatModel:
    return GeminiChatModel(client, api_key=SecretStr(CANARY_KEY), model=MODEL)
