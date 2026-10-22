"""Provider selection (build_chat_model) and the fail-fast Gemini settings."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from module_g1.g1_support import CANARY_KEY
from ragforge_core.adapters.llm.chat_gemini import GeminiChatModel, ModelConfigurationError
from ragforge_core.adapters.llm.chat_ollama import OllamaChatModel
from ragforge_core.adapters.llm.provider import build_chat_model
from ragforge_core.config import LLMProvider, Settings
from ragforge_core.domain.ports import ChatModel

DB_SECRET = "db-password-canary-g1"


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- settings
def test_defaults_select_ollama_and_frozen_gemini_defaults() -> None:
    settings = _settings()
    assert settings.llm_provider is LLMProvider.ollama
    assert settings.gemini_thinking_budget == 0
    assert settings.gemini_api_key.get_secret_value() == ""
    assert settings.gemini_model


def test_provider_values_are_exactly_ollama_and_gemini() -> None:
    assert {p.value for p in LLMProvider} == {"ollama", "gemini"}
    with pytest.raises(ValidationError):
        _settings(llm_provider="openai")


def test_env_vars_use_the_frozen_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGFORGE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("RAGFORGE_GEMINI_API_KEY", CANARY_KEY)
    monkeypatch.setenv("RAGFORGE_GEMINI_MODEL", "gemini-custom")
    monkeypatch.setenv("RAGFORGE_GEMINI_THINKING_BUDGET", "256")
    settings = _settings()
    assert settings.llm_provider is LLMProvider.gemini
    assert settings.gemini_api_key.get_secret_value() == CANARY_KEY
    assert settings.gemini_model == "gemini-custom"
    assert settings.gemini_thinking_budget == 256
    assert CANARY_KEY not in repr(settings)
    assert CANARY_KEY not in str(settings.model_dump())


@pytest.mark.parametrize("key", [None, "", "   "])
def test_gemini_without_a_key_fails_fast_with_a_clear_message(key: str | None) -> None:
    overrides: dict[str, object] = {"llm_provider": "gemini", "db_api_password": DB_SECRET}
    if key is not None:
        overrides["gemini_api_key"] = key
    with pytest.raises(ValidationError) as caught:
        _settings(**overrides)
    message = str(caught.value)
    assert "RAGFORGE_GEMINI_API_KEY must be set when RAGFORGE_LLM_PROVIDER=gemini" in message
    assert DB_SECRET[:4] not in message  # the error must not echo other settings' secrets


def test_ollama_does_not_need_a_gemini_key() -> None:
    assert _settings(llm_provider="ollama").gemini_api_key.get_secret_value() == ""


@pytest.mark.parametrize("model", ["../x", "a/b", "m?key=1", "", "x y"])
def test_model_names_that_could_alter_the_url_are_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        _settings(gemini_model=model)


@pytest.mark.parametrize("budget", [-2, 40_000])
def test_thinking_budget_is_bounded(budget: int) -> None:
    with pytest.raises(ValidationError):
        _settings(gemini_thinking_budget=budget)


# ---------------------------------------------------------------------- factory
async def test_factory_builds_ollama_by_default() -> None:
    chat = build_chat_model(_settings())
    try:
        assert isinstance(chat, OllamaChatModel)
        assert isinstance(chat, ChatModel)
    finally:
        await chat.aclose()


async def test_factory_builds_gemini_when_selected() -> None:
    settings = _settings(
        llm_provider="gemini",
        gemini_api_key=CANARY_KEY,
        gemini_model="gemini-x",
        gemini_thinking_budget=128,
    )
    chat = build_chat_model(settings)
    try:
        assert isinstance(chat, GeminiChatModel)
        assert (chat.provider, chat.model) == ("gemini", "gemini-x")
    finally:
        await chat.aclose()


async def test_factory_uses_a_given_client_and_does_not_close_it() -> None:
    settings = _settings(llm_provider="gemini", gemini_api_key=CANARY_KEY)
    async with httpx.AsyncClient(base_url="https://gemini.test") as client:
        chat = build_chat_model(settings, client)
        await chat.aclose()
        assert not client.is_closed


def test_factory_rechecks_the_key_when_validation_was_skipped() -> None:
    unchecked = Settings.model_construct(
        llm_provider=LLMProvider.gemini,
        gemini_api_key=SecretStr(""),
        gemini_model="gemini-x",
        gemini_thinking_budget=0,
    )
    with pytest.raises(ModelConfigurationError, match="not configured"):
        build_chat_model(unchecked)
