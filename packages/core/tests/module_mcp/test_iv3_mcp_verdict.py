"""IV3-UI: the MCP `ask` warning for an unverified turn says WHY honestly (live failure,
2026-10-19): a refusal or uncited answer had nothing from the notes to check, so "verifier off for
this model" would be false. A cited unverified answer keeps the old text."""

from __future__ import annotations

from uuid import uuid4

from mcp_support import cite, connected, mock_api, sse, structured

from ragforge_contracts.sse import (
    ChatEvent,
    CitationEvent,
    DoneEvent,
    TokenEvent,
    VerificationEvent,
)
from ragforge_mcp.turn import NOTHING_TO_CHECK_WARNING

DONE = DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="answer@11")


async def test_an_uncited_unverified_refusal_says_nothing_to_check() -> None:
    refusal: list[ChatEvent] = [
        TokenEvent(text="Your notes don't cover this."),
        VerificationEvent(status="unverified"),
        DONE,
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(refusal))
        async with connected() as client:
            result = structured(await client.call_tool("ask", {"question": "q"}))
    assert result["warning"] == NOTHING_TO_CHECK_WARNING
    assert "verifier off" not in result["warning"]


async def test_a_cited_unverified_answer_keeps_the_verifier_off_text() -> None:
    cited: list[ChatEvent] = [
        TokenEvent(text="Consumers poll \\[S1\\]."),
        CitationEvent(citation=cite(1)),
        VerificationEvent(status="unverified"),
        DONE,
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(cited))
        async with connected() as client:
            result = structured(await client.call_tool("ask", {"question": "q"}))
    assert "verifier off for this model" in result["warning"]
    assert "Nothing to check" not in result["warning"]


async def test_an_uncited_but_flagged_unverified_answer_keeps_the_verifier_off_text() -> None:
    """Untrusted verifier: inferred citations dropped, every sentence flagged. The answer made
    claims, so telling the calling model "no answer came from the user's notes" would be false."""
    flagged: list[ChatEvent] = [
        TokenEvent(text="Kafka elects leaders with Raft."),
        VerificationEvent(
            status="unverified", unsupported_claims=["Kafka elects leaders with Raft."]
        ),
        DONE,
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(flagged))
        async with connected() as client:
            result = structured(await client.call_tool("ask", {"question": "q"}))
    assert "verifier off for this model" in result["warning"]
    assert "Nothing to check" not in result["warning"]
    assert result["unsupported_claims"]


async def test_a_verified_or_partial_uncited_turn_is_not_called_nothing_to_check() -> None:
    """The rule is `unverified` AND uncited only: any other verdict keeps its own warning."""
    partial: list[ChatEvent] = [TokenEvent(text="x"), VerificationEvent(status="partial"), DONE]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(partial))
        async with connected() as client:
            result = structured(await client.call_tool("ask", {"question": "q"}))
    assert "Verification is partial" in result["warning"]
