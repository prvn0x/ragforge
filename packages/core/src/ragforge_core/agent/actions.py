"""Explicit actions: which tools (if any) a turn may bind (ADR-0015).

The answer path binds NO tools (retrieve -> answer). Measured on llama3.2:3b (NEXT.md, Phase 2b):
binding tools on every turn took correct answers from 46% to 15% and doubled p50 latency,
because a small model calls tools it does not need and passes them garbage (its own schema, a
`[S1]` label as a document id). Tools exist for ACTIONS the user explicitly asks for, so the
decision "is this turn an action?" is made here, deterministically, from the user's own raw
message (never from retrieved text or model output), and a matched action binds ONLY its tools.

Two ways to ask, both deterministic:
- a command at the start of the message: `/remember <words>`, `/forget <fact id>`,
  `/remove <page>`, `/sources`, `/sync` (the reliable path, documented for clients). A command
  whose arguments are fixed by the command itself (`server_call`) does not even ask the model
  for the tool call: the server builds it, so the documented path is deterministic end to end;
- a narrow imperative phrasing at the start of the message ("remember that ...", "please forget
  the fact ...", "remove the Kafka basics page from my index", "what are my sources", "what is
  my sync status?", "when did Notion last sync?").

WHY narrow phrasings and not a classifier call: a classifier is another model call per turn (the
ADR's latency rule) and is itself a small model's judgment. The DESTRUCTIVE phrasings are the
narrowest, because a false positive there is not free: it binds a destructive tool, switches the
prompt to "perform an action" and invites the model to propose de-indexing the user's most
relevant page. The owner's corpus is databases, Kafka and system design, where "delete a key
from the index" or "remove the leader from the ISR" are ordinary QUESTIONS. So a destructive
phrasing needs an explicit object (a quoted title, or "... page/note") AND "from my/the index"
ending the message, is never a question (no "?" anywhere, no "can you ..." form), and everything
else must use the command. "Forget" must name a fact. The same "no question" rule holds for the
bare remember phrasing, because remember_fact WRITES long-term memory. Every safety property
still holds when detection fires wrongly (destructive tools need human approval, remember_fact
is quote-gated against this same message, arguments are validated); a false negative answers
the message as a question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ragforge_core.agent.settings import AgentSettings, ToolMode
from ragforge_core.agent.tools import REGISTRY, TOOL_SPECS
from ragforge_core.domain.models import ToolCall, ToolSpec


class Action(StrEnum):
    remember = "remember"
    forget = "forget"
    remove_page = "remove_page"
    sources = "sources"
    sync = "sync"


ACTION_TOOLS: Final[dict[Action, tuple[str, ...]]] = {
    Action.remember: ("remember_fact",),
    Action.forget: ("forget_fact",),
    Action.remove_page: ("remove_from_index",),
    Action.sources: ("list_sources",),
    Action.sync: ("sync_status",),
}

_COMMANDS: Final[dict[str, Action]] = {
    "remember": Action.remember,
    "forget": Action.forget,
    "remove": Action.remove_page,
    "sources": Action.sources,
    "sync": Action.sync,
}
# The command word must END at whitespace, ":" or the end of the message (review round 5): with
# `\b`, "/sources/foo is a dir" was the sources command and "/remove/tmp" a removal of "/tmp".
_COMMAND: Final = re.compile(
    r"\A\s*/(?P<name>[a-z]+)(?=[\s:]|\Z)[ \t:]*(?P<rest>.*)\Z", re.DOTALL | re.IGNORECASE
)
_POLITE: Final = r"\A\s*(?:(?:hey|ok|okay)[,!]?\s+)?(?:(?:please|pls|kindly)[,]?\s+)?"
_CAN_YOU: Final = r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
# The page a destructive phrasing names: a quoted title, or words ending in page(s)/note(s).
# "Delete the page table entry from the index" has no such object right before "from".
# NOT "doc"/"document" (review round 4): in a search/DB corpus "delete stale documents from the
# search index" is an ordinary question about Elasticsearch, not about the user's own notes.
_PAGE: Final = (
    r"(?:\"[^\"\n]{1,200}\"|'[^'\n]{1,200}'|\u201c[^\u201d\n]{1,200}\u201d"
    r"|[^.?!\n\"]{0,200}?\b(?:page|note)s?)"
)
# "... from my/the index" must END the message (review round 4): "remove deleted notes from the
# index in Elasticsearch" goes on to name somebody else's index.
_FROM_INDEX: Final = (
    r"\s+(?:from|out\s+of)\s+(?:the|my)\s+(?:(?:search\s+)?index|search|knowledge\s+base)"
    r"\s*[.!]?\s*\Z"
)
_MINE: Final = r"(?:notion|my\s+(?:notes|sources|notion(?:\s+(?:pages|workspace))?))"
_REMEMBER: Final = (
    r"(?:(?:remember|keep\s+in\s+mind|note\s+down)|(?:don'?t|do\s+not)\s+forget)\s*(?:that\b|:)"
)
# Read-only phrasings may be questions ("what are my sources?"): a false positive binds a tool
# that changes nothing. A WRITE is different (review round 4): "Remember that Redis is single
# threaded?" is a recall question, and binding remember_fact there lets a small model store a
# fact the user never asked to keep. So a remember phrasing with "?" counts only in the explicit
# request form "can/could/would/will you remember that ...?"; the bare form must not ask.
_REMEMBER_REQUEST: Final = r"(?:can|could|would|will)\s+you\s+(?:please\s+)?" + _REMEMBER
_READ_ONLY_PHRASES: Final[tuple[tuple[Action, str], ...]] = (
    # Anchored at the end like sync (review round 4): "show me my sources of truth for Kafka
    # consumer lag" is a corpus question, "what are my sources?" is not.
    (
        Action.sources,
        r"(?:(?:list|show)(?:\s+me)?(?:\s+all)?\s+my|(?:what|which)\s+(?:are\s+)?my)\s+"
        r"(?:sources|connectors|data\s+sources)\s*[?.!]?\s*\Z",
    ),
    # Sync is anchored to the user's OWN sources (review round 2): "sync status of replicas in
    # Kafka" and "when did the follower last sync with the leader?" are replication questions
    # about the corpus. So "sync status" must end the message or name my notes/sources/Notion,
    # and "last sync" needs Notion, "my notes" or "my sources" as its subject.
    (
        Action.sync,
        rf"(?:what(?:'s|\s+is)\s+(?:the\s+|my\s+)?)?(?:notion\s+)?sync(?:ing)?\s+status"
        rf"(?:\s+(?:of|for)\s+{_MINE})?\s*[?.!]?\s*\Z",
    ),
    (Action.sync, rf"(?:when\s+did|has)\s+{_MINE}\s+last\s+sync"),
)
# Forget names a FACT (review round 4): "forget what I said, explain Raft" and "forget about
# consistency, ..." are conversational, not a request to delete a stored fact. The word "fact"
# must appear in the same clause ("forget the fact that ...", "forget about the Postgres fact").
_DESTRUCTIVE_PHRASES: Final[tuple[tuple[Action, str], ...]] = (
    (Action.forget, r"forget\s+(?:\[?fact:|(?:about\s+)?[^.?!,;:\n]{0,80}?\bfacts?\b)"),
    (Action.remove_page, rf"(?:remove|exclude|delete|drop|unindex)\s+{_PAGE}{_FROM_INDEX}"),
    (Action.remove_page, rf"stop\s+indexing\s+{_PAGE}\s*[.!]?\s*\Z"),
)
# A remember phrasing followed by a SECOND request (review round 5): "keep in mind that Postgres
# uses MVCC, how does vacuum work" has no "?", yet its second clause is a question. Treating the
# turn as a remember action would answer it under action_reply@1 from tool results only, so the
# question would silently never be answered from the notes. Such a message is a question (a
# false negative: the user can still say `/remember ...`). Only interrogative and request words
# that open a clause count; relative clauses ("..., which means ...", "..., when traffic is
# low") and lists ("A, B and C") stay part of the fact.
_SECOND_ASK: Final = re.compile(
    r"(?:[,;:.]|\n|\s-{1,2}\s)\s*(?:(?:and|but|so|also|now|then|btw|anyway)[,]?\s+)?"
    r"(?:how|what|why|does|did|is|are|can|could|should|would|explain|describe|compare|tell\s+me"
    r"|show\s+me|summari[sz]e|give\s+me)\b",
    re.IGNORECASE,
)
# (action, may the message contain "?", pattern). Destructive actions and the bare remember
# form are never inferred from a question (`detect`).
_PHRASES: Final[tuple[tuple[Action, bool, re.Pattern[str]], ...]] = (
    (Action.remember, True, re.compile(_POLITE + _REMEMBER_REQUEST, re.IGNORECASE)),
    (Action.remember, False, re.compile(_POLITE + _CAN_YOU + _REMEMBER, re.IGNORECASE)),
    *(
        (action, True, re.compile(_POLITE + _CAN_YOU + body, re.IGNORECASE))
        for action, body in _READ_ONLY_PHRASES
    ),
    *(
        (action, False, re.compile(_POLITE + body, re.IGNORECASE))
        for action, body in _DESTRUCTIVE_PHRASES
    ),
)


@dataclass(frozen=True, slots=True)
class Intent:
    action: Action | None  # None: a question (retrieve -> answer, no tools)
    query: str  # retrieval text: the message, minus a leading command word
    command: bool = False  # asked with a slash command (not a phrasing)
    argument: str = ""  # the command's argument, verbatim ("" for a phrasing or no argument)


def detect(message: str) -> Intent:
    """Classify the user's raw message. Commands win over phrasings; an unknown `/word` is a
    question like any other text (it must not silently become an action)."""
    command = _COMMAND.match(message)
    if command is not None and command["name"].lower() in _COMMANDS:
        rest = command["rest"].strip()
        return Intent(_COMMANDS[command["name"].lower()], rest or message.strip(), True, rest)
    asks = "?" in message
    for action, question_ok, pattern in _PHRASES:
        if asks and not question_ok:
            continue  # a write the user did not plainly request is never inferred from a question
        match = pattern.search(message)
        if match is None:
            continue
        if action is Action.remember and _SECOND_ASK.search(message, match.end()):
            return Intent(None, message)  # "remember that X, how does Y work": a question
        return Intent(action, message)
    return Intent(None, message)


# The tool argument a command's text fills, for commands that need no choice by the model.
_ARGUMENT: Final[dict[Action, str | None]] = {
    Action.remember: "user_quote",
    Action.forget: "fact_id",
    Action.sources: None,
    Action.sync: None,
}


def server_call(intent: Intent, call_id: str) -> ToolCall | None:
    """The tool call a slash command implies, built by the SERVER (no model involved), or None.

    WHY: the command form is the documented reliable path, so it must not depend on a 3B model
    copying "I prefer Java" into user_quote correctly. `/remove <page>` is excluded on purpose:
    choosing WHICH retrieved page the words name is a judgment the model makes from the shown
    sources (and a human approves). Server-built calls are still validated by the tool's closed
    schema, quote-gated (remember) and approval-gated (forget), exactly like a model's call."""
    if not intent.command or intent.action not in _ARGUMENT:
        return None
    key = _ARGUMENT[intent.action]
    if key is not None and not intent.argument:
        return None  # "/remember" alone: nothing to build; the model asks what to remember
    (name,) = ACTION_TOOLS[intent.action]
    return ToolCall(id=call_id, name=name, arguments={} if key is None else {key: intent.argument})


def tools_for_turn(settings: AgentSettings, action: str, tool_rounds: int) -> tuple[ToolSpec, ...]:
    """The tool specs to bind on this generate step.

    - `max_tool_rounds` stays the hard cap for every mode (0 disables tools entirely).
    - mode `always` (the pre-ADR-0015 behaviour, kept as an explicit override for evaluation):
      every tool, up to `max_tool_rounds` rounds.
    - mode `actions` (default): nothing on a question; on an explicit action ONLY that action's
      tools, for ONE round. One round because the user asked for one action: after its result
      the model writes the reply with tools unbound, which also stops a small model from
      looping on the same call.
    """
    if tool_rounds >= settings.max_tool_rounds:
        return ()
    if settings.tool_mode is ToolMode.always:
        return TOOL_SPECS
    if not action or tool_rounds >= 1:
        return ()
    try:
        names = ACTION_TOOLS[Action(action)]
    except ValueError:  # a checkpoint written by a build with another action set: bind nothing
        return ()
    return tuple(REGISTRY[name].spec for name in names)
