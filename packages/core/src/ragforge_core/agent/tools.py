"""Agent tools (DESIGN section 8): specs, strict argument validation and execution.

Security properties enforced here, independent of what the model does:
- Arguments are validated against closed schemas (`extra="forbid"`): a model that adds
  `tenant_id`/`user_id` gets an error, and there is no argument through which a tenant or user
  could be chosen. Every tool runs in `ctx.tenant_tx()` and the facts namespace of the Principal.
- Destructiveness is a property of the tool REGISTRY, never of the call: `forget_fact` and
  `remove_from_index` always go through the approval node.
- `remember_fact` is quote-gated: the stored fact IS a >= 20-char verbatim quote of the user's
  latest message, so retrieved documents (or the model) cannot plant lasting memories.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ragforge_core.agent import retrieval
from ragforge_core.agent.context import AgentContext
from ragforge_core.domain.models import RankedChunk, ToolCall, ToolSpec
from ragforge_core.domain.ports import Tx

MIN_QUOTE_CHARS: Final = 20
FACT_REF_CHARS: Final = 12  # short fact id shown to the model: sha256 prefix (48 bits)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchNotesArgs(_Args):
    query: Annotated[str, Field(min_length=1, max_length=500)]


class GetPageArgs(_Args):
    document_id: UUID


class RemoveFromIndexArgs(_Args):
    document_id: UUID


class ListSourcesArgs(_Args):
    pass


class SyncStatusArgs(_Args):
    pass


class RememberFactArgs(_Args):
    user_quote: Annotated[str, Field(min_length=1, max_length=2000)]
    label: Annotated[str | None, Field(max_length=80)] = None


class ForgetFactArgs(_Args):
    """`fact_id` is the short id shown next to each fact in the model's <facts> block (or a full
    sha256 key). It is only ever resolved inside the Principal's own facts namespace."""

    fact_id: Annotated[str, Field(pattern=rf"^(?:[0-9a-f]{{{FACT_REF_CHARS}}}|[0-9a-f]{{64}})$")]

    @field_validator("fact_id", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        """Models echo the id the way it was displayed (`[fact:ab12...]`); accept that form."""
        if not isinstance(value, str):
            return value
        return value.strip().strip("[]").strip().lower().removeprefix("fact:").strip()


@dataclass(frozen=True, slots=True)
class ToolDef:
    spec: ToolSpec
    args: type[_Args]


def _tool(name: str, description: str, args: type[_Args], *, destructive: bool = False) -> ToolDef:
    schema = args.model_json_schema()
    schema.pop("title", None)
    return ToolDef(ToolSpec(name, description, schema, destructive=destructive), args)


REGISTRY: Final[Mapping[str, ToolDef]] = {
    t.spec.name: t
    for t in (
        _tool(
            "search_notes", "Search the user's notes again with a different query.", SearchNotesArgs
        ),
        _tool("get_page", "Read more of one page (document_id from a source).", GetPageArgs),
        _tool(
            "list_sources",
            "List the sources (connectors, uploads) feeding the notes.",
            ListSourcesArgs,
        ),
        _tool("sync_status", "Show sync status and errors of the user's sources.", SyncStatusArgs),
        _tool(
            "remember_fact",
            "Remember something the USER explicitly asked you to remember. user_quote must be "
            "the user's exact words from their latest message (at least 20 characters).",
            RememberFactArgs,
        ),
        _tool(
            "forget_fact",
            "Forget a remembered fact (needs the user's approval). fact_id is the id shown "
            "next to the fact, e.g. 3f2a9c0d1b7e for [fact:3f2a9c0d1b7e].",
            ForgetFactArgs,
            destructive=True,
        ),
        _tool(
            "remove_from_index",
            "Exclude a page from the index in every collection that holds a copy "
            "(needs the user's approval).",
            RemoveFromIndexArgs,
            destructive=True,
        ),
    )
}

TOOL_SPECS: Final[tuple[ToolSpec, ...]] = tuple(t.spec for t in REGISTRY.values())


def is_destructive(name: str) -> bool:
    tool = REGISTRY.get(name)
    return tool is not None and tool.spec.destructive


def parse_args(call: ToolCall) -> _Args | str:
    """Validated arguments, or an error message for the model (never raises on model input)."""
    tool = REGISTRY.get(call.name)
    if tool is None:
        return f"Unknown tool {call.name!r}."
    try:
        return tool.args.model_validate(call.arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
        )
        return f"Invalid arguments for {call.name}: {problems}"


def quote_gate(quote: str, latest_user_message: str) -> str | None:
    """None if the quote may be stored, else the reason it may not (deterministic, no LLM)."""
    candidate = quote.strip()
    if len(candidate) < MIN_QUOTE_CHARS:
        return f"the quote must be at least {MIN_QUOTE_CHARS} characters of the user's words"
    if candidate not in latest_user_message:
        return "the quote must appear verbatim in the user's latest message"
    return None


def fact_id(quote: str) -> str:
    """Store key: content-addressed, so remembering the same quote twice is one fact."""
    return hashlib.sha256(quote.strip().encode("utf-8")).hexdigest()


def fact_ref(key: str) -> str:
    """The short id the model sees for a stored fact (and passes back to forget_fact)."""
    return key[:FACT_REF_CHARS]


def resolve_fact_key(ref: str, shown_keys: Sequence[str]) -> str | None:
    """Map a forget_fact id to a store key.

    A short id resolves only against the facts this turn SHOWED the model (so a guessed prefix
    cannot reach anything else), and only when it is unambiguous. A full 64-hex key is taken as
    is: it can only address the Principal's own namespace, where the delete happens.
    """
    if len(ref) == 64:
        return ref
    matches = {k for k in shown_keys if fact_ref(k) == ref}
    return matches.pop() if len(matches) == 1 else None


@dataclass(slots=True)
class ToolOutcome:
    text: str  # untrusted body for the model (fenced by the caller)
    hits: list[RankedChunk] = field(default_factory=list)  # retrieval results to label + emit


def _json(rows: Sequence[Mapping[str, object]]) -> str:
    return json.dumps([dict(r) for r in rows], default=str, ensure_ascii=False)


async def run_safe(
    ctx: AgentContext,
    args: _Args,
    *,
    collection_ids: Sequence[UUID],
    user_message: str,
) -> ToolOutcome:
    """Execute a non-destructive tool. Dispatch is on the validated argument type: each tool has
    its own argument class, bound 1:1 to its name by REGISTRY."""
    deps = ctx.deps
    match args:
        case SearchNotesArgs(query=query):
            found = await retrieval.retrieve(ctx, query, collection_ids)
            return ToolOutcome(f"{len(found.hits)} results for the query.", found.hits)
        case GetPageArgs(document_id=document_id):
            async with ctx.tenant_tx() as tx:
                chunks = await deps.knowledge.get_page(
                    tx, document_id, collection_ids, ctx.settings.page_chunk_limit
                )
            if not chunks:
                return ToolOutcome("No such page in these collections.")
            return ToolOutcome(f"{len(chunks)} sections of the page.", list(chunks))
        case RememberFactArgs(user_quote=quote, label=label):
            return ToolOutcome(await _remember(ctx, quote, label, user_message))
        case ListSourcesArgs():
            async with ctx.tenant_tx() as tx:
                rows = await deps.knowledge.list_sources(tx, collection_ids)
            return ToolOutcome(_json(rows))
        case SyncStatusArgs():
            async with ctx.tenant_tx() as tx:
                rows = await deps.knowledge.sync_status(tx, collection_ids)
            return ToolOutcome(_json(rows))
    raise TypeError(f"{type(args).__name__} is not a safe tool")


async def _remember(ctx: AgentContext, quote: str, label: str | None, user_message: str) -> str:
    reason = quote_gate(quote, user_message)
    if reason is not None:
        return f"Not remembered: {reason}."
    fact = quote.strip()
    await ctx.deps.store.aput(
        ctx.facts_namespace,
        fact_id(fact),
        {"fact": fact, "label": label, "conversation_id": str(ctx.conversation_id)},
        index=False,
    )
    return "Remembered."


async def run_destructive(
    ctx: AgentContext,
    tx: Tx,
    args: _Args,
    collection_ids: Sequence[UUID],
    shown_fact_keys: Sequence[str] = (),
) -> str:
    """Execute an APPROVED destructive tool inside the caller's tx (the one holding the
    execution claim), so a failure rolls back the claim and the effect together."""
    match args:
        case RemoveFromIndexArgs(document_id=document_id):
            removed = await ctx.deps.knowledge.remove_from_index(
                tx, document_id, collection_ids, ctx.actor
            )
            return "Page removed from the index." if removed else "No such page."
        case ForgetFactArgs(fact_id=ref):
            key = resolve_fact_key(ref, shown_fact_keys)
            if key is None:
                return "No such fact."
            # The store has its own connection; delete is idempotent, so if the claim's commit
            # fails after this, a re-run deletes nothing new (at-least-once, same outcome).
            # WHY the store I/O stays inside the claim tx (it holds one pending_approvals row
            # lock for two single-row store round trips): claiming in a separate, committed tx
            # first would make a crash before the delete lose it for good (at-most-once, with
            # "already performed" reported), and deleting before the claim would run the effect
            # before the claim's status='approved' check. The lock is uncontended: only the
            # holder of this conversation's run lease (this run) ever touches these rows.
            existing = await ctx.deps.store.aget(ctx.facts_namespace, key)
            if existing is None:
                return "No such fact."
            await ctx.deps.store.adelete(ctx.facts_namespace, key)
            return "Fact forgotten."
    raise TypeError(f"{type(args).__name__} is not a destructive tool")
