"""The LangGraph wiring (DESIGN section 8). Routing functions are pure and bounded:

- rewrite at most `max_rewrites` times (ADR-0006), then abstain;
- generate -> tools -> generate at most `max_tool_rounds` times, then tools are unbound (and by
  default tools are bound only on an explicit action, for one round: ADR-0015);
- at most ONE strict regeneration, then abstain.

so every turn terminates in a bounded number of model calls regardless of model output.
"""

from __future__ import annotations

from typing import Final, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragforge_core.agent import actions, nodes
from ragforge_core.agent.context import AgentContext
from ragforge_core.agent.settings import AgentSettings
from ragforge_core.agent.state import AgentState
from ragforge_core.domain.models import Verification

# Longest legal path is ~30 node steps (2 retrievals, 3 tool rounds with an approval, strict
# regeneration, abstain); the limit only guards against a routing bug.
RECURSION_LIMIT: Final = 64

AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def after_memory(
    settings: AgentSettings, state: AgentState
) -> Literal["contextualize", "retrieve", "generate"]:
    """Only a follow-up QUESTION is rewritten into a standalone query. An explicit action
    (ADR-0015) skips it on every turn: its query is fixed by the user's own words (the command
    argument or the phrasing), an LLM rewrite would replace exactly the text that names the page
    to remove, and the action path promises one model call less, not one more.

    A command whose tool call the SERVER builds (`/sync`, `/sources`, `/forget <id>`,
    `/remember <words>`) skips retrieval too (review round 3): no page can change the call, the
    reply reports a tool result and is never page-attributed, so embedding and reranking the
    command text would only add latency and a meaningless TurnSources. Only when that call will
    actually be made: with tools disabled (`max_tool_rounds=0`) the turn is a question again."""
    if _server_call_ready(settings, state):
        return "generate"
    if state.get("action"):
        return "retrieve"
    return "contextualize" if state.get("messages") else "retrieve"


def _server_call_ready(settings: AgentSettings, state: AgentState) -> bool:
    calls = state.get("command_calls", [])
    bound = {t.name for t in actions.tools_for_turn(settings, state.get("action", ""), 0)}
    return bool(calls) and all(c["name"] in bound for c in calls)


def after_grade(
    settings: AgentSettings, state: AgentState
) -> Literal["generate", "rewrite", "abstain"]:
    """An explicit action (ADR-0015) always reaches `generate`, where its tool is bound: the
    user asked for something to be DONE, and "remember that I prefer Java" must not abstain
    with "not in your notes" just because no note matches it."""
    if state.get("grade") == "ok" or state.get("action"):
        return "generate"
    return "rewrite" if state.get("rewrites", 0) < settings.max_rewrites else "abstain"


def after_generate(state: AgentState) -> Literal["run_tools", "verify_citations"]:
    return "run_tools" if state.get("pending_calls") else "verify_citations"


def after_tools(state: AgentState) -> Literal["approval", "generate"]:
    return "approval" if state.get("approval_calls") else "generate"


def after_citations(state: AgentState) -> Literal["decide", "ground_check"]:
    """Only a `pending` draft needs the verifier: `unsupported` (rules 1-2) goes to the
    regeneration decision, and a pure refusal (`unverified`, ADR-0015) makes no claim to check."""
    pending = state.get("verification") == Verification.pending.value
    return "ground_check" if pending else "decide"


def after_decide(state: AgentState) -> Literal["finalize", "generate_strict", "abstain"]:
    if state.get("verification") != Verification.unsupported.value:
        return "finalize"
    return "abstain" if state.get("regenerated", False) else "generate_strict"


def build_graph(
    settings: AgentSettings, checkpointer: BaseCheckpointSaver[str] | None
) -> AgentGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    for node in (
        nodes.load_memory,
        nodes.contextualize,
        nodes.retrieve,
        nodes.grade,
        nodes.rewrite,
        nodes.abstain,
        nodes.generate,
        nodes.run_tools,
        nodes.approval,
        nodes.execute,
        nodes.verify_citations,
        nodes.ground_check,
        nodes.decide,
        nodes.generate_strict,
        nodes.finalize,
        nodes.extras,
    ):
        graph.add_node(node.__name__, nodes.guarded(node, check_lease=node not in nodes.PURE_NODES))

    graph.add_edge(START, "load_memory")
    graph.add_conditional_edges("load_memory", lambda s: after_memory(settings, s))
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", lambda s: after_grade(settings, s))
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("generate", after_generate)
    graph.add_conditional_edges("run_tools", after_tools)
    graph.add_edge("approval", "execute")
    graph.add_edge("execute", "generate")
    graph.add_conditional_edges("verify_citations", after_citations)
    graph.add_edge("ground_check", "decide")
    graph.add_conditional_edges("decide", after_decide)
    graph.add_edge("generate_strict", "verify_citations")
    graph.add_edge("abstain", "finalize")
    # ADR-0016: after verification and outside it; only routed when an extra will be made, so a
    # turn without one pays nothing (not even the node's lease heartbeat)
    graph.add_conditional_edges(
        "finalize", lambda s: "extras" if settings.extras and nodes.extras_kind(s) else END
    )
    graph.add_edge("extras", END)
    return graph.compile(checkpointer=checkpointer)
