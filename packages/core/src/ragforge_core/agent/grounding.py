"""Ground check (ADR-0011 rule 5): one JSON-schema LLM call, one repair retry, malformed -> partial.

WHY malformed maps to `partial` and not `verified`/`unsupported`: a verifier that cannot answer
has proven nothing. `verified` would be a false badge (the headline metric is the false-"verified"
rate, DESIGN section 2), and `unsupported` would throw away a possibly good answer and trigger a
regeneration because of a formatting glitch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

from ragforge_core.agent.fencing import fence_answer
from ragforge_core.domain.models import (
    ChatMessage,
    ChatTurnUsage,
    Role,
    Verification,
)
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability.prompt_context import for_prompt
from ragforge_core.prompts import get_prompt, render

GROUND_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "unsupported_claims"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "partially_supported", "unsupported"],
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string", "maxLength": 2000},
            "maxItems": 20,
        },
    },
}

_VERDICTS: Final = {
    "supported": Verification.verified,
    "partially_supported": Verification.partial,
    "unsupported": Verification.unsupported,
}


class MalformedVerdict(ValueError):
    """The model's JSON did not match GROUND_SCHEMA."""


@dataclass(frozen=True, slots=True)
class GroundResult:
    status: Verification
    unsupported_claims: tuple[str, ...]
    usage: tuple[ChatTurnUsage, ...]
    # False when the verifier never produced a valid verdict (status is then the `partial`
    # fallback). Server-inferred citations need a conclusive pass (ADR-0015): "the verifier
    # could not answer" must never be read as "the verifier agreed".
    conclusive: bool = True
    # False when the verdict cannot be placed in the answer: a flagged claim was not an exact
    # substring (so it was discarded), or "partially_supported" came with no usable claim. Both
    # are the NORMAL output of a 3B verifier that paraphrases. The verdict then says "something
    # is unsupported" without saying what, so no sentence can be cleared by it (ADR-0015).
    located: bool = True


def parse_verdict(
    payload: dict[str, Any], answer: str, also: str = ""
) -> tuple[Verification, tuple[str, ...]]:
    """Validate the payload by hand (the adapter's schema mode is best effort on local models).

    Claims are kept only if they are verbatim substrings of the answer (or of `also`, the same
    answer without the server's inferred markers, which a verifier may copy either way): the
    contract highlights them by exact match, so a paraphrased claim would point at nothing. A
    "supported" verdict that still lists verbatim claims is downgraded to partial (the model
    contradicted itself; the conservative reading wins).
    """
    if set(payload) != {"verdict", "unsupported_claims"}:
        raise MalformedVerdict(f"expected keys verdict, unsupported_claims; got {sorted(payload)}")
    verdict, claims = payload["verdict"], payload["unsupported_claims"]
    if not isinstance(verdict, str) or verdict not in _VERDICTS:
        raise MalformedVerdict(f"verdict must be one of {sorted(_VERDICTS)}")
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise MalformedVerdict("unsupported_claims must be a list of strings")
    texts = [t for t in (answer, also) if t]
    verbatim = tuple(
        dict.fromkeys(c.strip() for c in claims if c.strip() and any(c.strip() in t for t in texts))
    )
    status = _VERDICTS[verdict]
    if status is Verification.verified and verbatim:
        status = Verification.partial
    return status, verbatim


def is_located(payload: dict[str, Any], status: Verification, claims: tuple[str, ...]) -> bool:
    """See GroundResult.located. `payload` already passed parse_verdict."""
    flagged = [c for c in payload["unsupported_claims"] if c.strip()]
    if len(claims) < len(dict.fromkeys(c.strip() for c in flagged)):
        return False  # at least one flagged claim was paraphrased and discarded
    return status is not Verification.partial or bool(claims)


async def ground_check(chat: ChatModel, sources: str, answer: str, also: str = "") -> GroundResult:
    """`answer` is what the verifier reads; `also` is an equivalent text claims may be quoted
    from (see parse_verdict). Every caller (chat, interview, eval verify) is labelled
    `ground_check@<v>` here, once, for its metrics (ADR-0018 R2-3)."""
    with for_prompt(get_prompt("ground_check").tag):
        return await _ground_check(chat, sources, answer, also)


async def _ground_check(chat: ChatModel, sources: str, answer: str, also: str) -> GroundResult:
    system = render("ground_check")
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.user, content=f"{sources}\n\n{fence_answer(answer)}")
    ]
    usage: list[ChatTurnUsage] = []
    for attempt in range(2):
        try:
            payload, used = await chat.complete_json(system, messages, GROUND_SCHEMA)
        except ValueError as exc:  # adapter could not parse JSON at all (json errors subclass it)
            error, raw = str(exc), ""
        else:
            usage.append(used)
            try:
                status, claims = parse_verdict(payload, answer, also)
            except MalformedVerdict as exc:
                error, raw = str(exc), json.dumps(payload)[:4000]
            else:
                located = is_located(payload, status, claims)
                return GroundResult(status, claims, tuple(usage), located=located)
        if attempt == 0:  # the one repair retry: show the model what was wrong
            if raw:
                messages.append(ChatMessage(role=Role.assistant, content=raw))
            messages.append(
                ChatMessage(
                    role=Role.user,
                    content=f"That reply did not match the JSON schema ({error}). "
                    "Reply again with JSON matching the schema exactly.",
                )
            )
    return GroundResult(Verification.partial, (), tuple(usage), conclusive=False)
