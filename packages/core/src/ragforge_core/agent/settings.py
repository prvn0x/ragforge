"""Agent tunables. Plain values injected into the runner (config.py is the integrator's file;
docs/handoff/D1.md requests the matching `RAGFORGE_AGENT_*` settings)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# Grade threshold used when a collection has no tuned `collections.tau` (ADR-0006). FlashRank's
# cross-encoder emits a sigmoid relevance in [0, 1]; 0.35 is the provisional value until the
# eval harness tunes it per collection and records it in eval_runs.
DEFAULT_TAU: Final = 0.35


class ToolMode(StrEnum):
    """Which tools a turn may bind (ADR-0015). `actions` (default): none on a question, only the
    matched action's tools on an explicit action. `always`: every tool on every turn (the
    pre-ADR-0015 behaviour, kept only as an evaluation override; it measured 15% vs 46%)."""

    actions = "actions"
    always = "always"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    default_tau: float = DEFAULT_TAU
    retrieval_k: int = 5  # top-5 local / top-8 hosted (DESIGN section 7.3)
    rerank_top: int = 20  # fused candidates fed to the cross-encoder (ADR-0006)
    max_rewrites: int = 1  # ADR-0006: below tau -> ONE rewrite -> still below -> abstain
    max_tool_rounds: int = 3  # hard cap on generate->tools loops per turn (0 = no tools at all)
    tool_mode: ToolMode = ToolMode.actions  # ADR-0015: retrieve -> answer, tools for actions
    history_messages: int = 12  # committed messages sent to the model (context-window trim)
    memory_facts: int = 20  # long-term facts shown (fenced, user-side) to the model
    page_chunk_limit: int = 20  # get_page returns at most this many chunks
    closest_pages: int = 3  # pages listed in an abstain answer
    heartbeat_interval_s: float = 15.0  # DESIGN section 8: lease heartbeat every 15 s
    strict_citations: bool = False  # strict = one citation per sentence even on the 1st draft
    verifier_trusted: bool = True  # False: the model failed the verifier SLO -> "unverified"
    # ADR-0016: add a labelled "Not from your notes" incident (or code, on request) when the
    # verified answer has none. Never part of the verified answer. Off here (one more model call
    # per turn); the app turns it on through `agent_extras` (default true).
    extras: bool = False
    summarize_after_messages: int = 20  # enqueue summarize_conversation beyond this many
