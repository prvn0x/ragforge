"""One real call to the Gemini API. SKIPPED unless RAGFORGE_GEMINI_API_KEY is set in the
environment (never read from .env, so a normal test run makes no network call).

Run: RAGFORGE_GEMINI_API_KEY=... uv run pytest packages/core/tests/module_g1/test_g1_live_smoke.py
The key is wrapped in SecretStr at import and never appears in assertions or output.
"""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from ragforge_core.adapters.llm.chat_gemini import GeminiChatModel
from ragforge_core.config import LLMProvider, Settings
from ragforge_core.domain.models import (
    ChatMessage,
    Role,
    TextDelta,
    ToolCall,
    ToolSpec,
    UsageReport,
)

_KEY = SecretStr(os.environ.get("RAGFORGE_GEMINI_API_KEY", ""))
_MODEL = os.environ.get("RAGFORGE_GEMINI_MODEL", "")

pytestmark = pytest.mark.skipif(
    not _KEY.get_secret_value().strip(), reason="RAGFORGE_GEMINI_API_KEY not set"
)


def _settings() -> Settings:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.gemini, gemini_api_key=_KEY)
    # model_copy skips validation; the adapter re-validates the model name itself.
    return settings.model_copy(update={"gemini_model": _MODEL}) if _MODEL else settings


async def test_live_stream_tool_call_and_json() -> None:
    chat = GeminiChatModel.from_settings(_settings())
    try:
        events = [e async for e in chat.stream("Answer in one word.", [_ask("Capital of France?")])]
        assert any(isinstance(e, TextDelta) for e in events)
        assert isinstance(events[-1], UsageReport)
        assert events[-1].usage.prompt_tokens > 0

        tool = ToolSpec(
            "search_notes",
            "Search the user's notes.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        prompt = "Always call search_notes before answering."
        calls = [e async for e in chat.stream(prompt, [_ask("What is Kafka?")], [tool])]
        tool_calls = tuple(e for e in calls if isinstance(e, ToolCall))
        assert tool_calls

        # Second turn through a FRESH adapter (no cached thought signature), as after a restart
        # or a tool approval resumed on another replica: proves the functionResponse shape and
        # that the placeholder signature is accepted by the real API.
        history = [
            _ask("What is Kafka?"),
            ChatMessage(Role.assistant, "", tool_calls=tool_calls),
            *(ChatMessage(Role.tool, "Kafka is a log.", tool_call_id=c.id) for c in tool_calls),
        ]
        resumed = GeminiChatModel.from_settings(_settings())
        try:
            answer = [e async for e in resumed.stream(prompt, history, [tool])]
        finally:
            await resumed.aclose()
        assert isinstance(answer[-1], UsageReport)

        schema: dict[str, object] = {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["answer"],
        }
        payload, usage = await chat.complete_json("Reply as JSON.", [_ask("Is 2 > 1?")], schema)
        assert payload.get("answer") in {"yes", "no"}
        assert usage.provider == "gemini"
    finally:
        await chat.aclose()


async def test_live_dynamic_thinking_replays_a_real_signature() -> None:
    """thinking_budget=-1: the model thinks, so its functionCall carries a real thoughtSignature,
    which the SAME adapter must replay (from its cache) for the second turn to be accepted. Also
    exercises the stretched read timeout on a thinking model."""
    settings = _settings().model_copy(update={"gemini_thinking_budget": -1})
    chat = GeminiChatModel.from_settings(settings)
    try:
        tool = ToolSpec(
            "search_notes",
            "Search the user's notes.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        prompt = "Always call search_notes before answering."
        calls = [e async for e in chat.stream(prompt, [_ask("Compare Kafka and Redis.")], [tool])]
        tool_calls = tuple(e for e in calls if isinstance(e, ToolCall))
        assert tool_calls
        history = [
            _ask("Compare Kafka and Redis."),
            ChatMessage(Role.assistant, "", tool_calls=tool_calls),
            *(ChatMessage(Role.tool, "Both are fast.", tool_call_id=c.id) for c in tool_calls),
        ]
        answer = [e async for e in chat.stream(prompt, history, [tool])]
        assert isinstance(answer[-1], UsageReport)
    finally:
        await chat.aclose()


def _ask(text: str) -> ChatMessage:
    return ChatMessage(Role.user, text)
