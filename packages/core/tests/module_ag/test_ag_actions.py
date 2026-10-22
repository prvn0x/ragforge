"""Deterministic action detection and the tools a turn may bind (ADR-0015)."""

from __future__ import annotations

import pytest

from ragforge_core.agent import AgentSettings, ToolMode
from ragforge_core.agent.actions import (
    ACTION_TOOLS,
    Action,
    detect,
    server_call,
    tools_for_turn,
)
from ragforge_core.agent.tools import REGISTRY, TOOL_SPECS
from ragforge_core.domain.models import ToolCall


@pytest.mark.parametrize(
    ("message", "action", "query"),
    [
        ("/remember I prefer Java code examples", Action.remember, "I prefer Java code examples"),
        ("/forget 3f2a9c0d1b7e", Action.forget, "3f2a9c0d1b7e"),
        ("/remove Kafka basics", Action.remove_page, "Kafka basics"),
        ("/sources", Action.sources, "/sources"),
        ("/SYNC", Action.sync, "/SYNC"),
        ("Remember that I prefer Java examples.", Action.remember, None),
        ("please remember: my exam is on Friday", Action.remember, None),
        ("Can you keep in mind that I use Postgres 16?", Action.remember, None),
        ("Don't forget that I prefer short answers", Action.remember, None),
        ("forget about the Postgres fact", Action.forget, None),
        ("Please forget the fact that I use Postgres 16", Action.forget, None),
        ("forget [fact:3f2a9c0d1b7e]", Action.forget, None),
        ("Can you remember that I use Postgres 16?", Action.remember, None),  # a request form
        ("Please remove the Kafka basics page from my index", Action.remove_page, None),
        ("exclude 'Old notes' from the search index", Action.remove_page, None),
        ('Delete "B-tree internals" from my index.', Action.remove_page, None),
        ("Drop my interview notes from the knowledge base", Action.remove_page, None),
        ("Stop indexing the drafts page", Action.remove_page, None),
        ("what are my sources?", Action.sources, None),
        ("List my connectors", Action.sources, None),
        ("What's the sync status?", Action.sync, None),
        ("when did Notion last sync?", Action.sync, None),
        ("sync status", Action.sync, None),
        ("What is the sync status of my notes?", Action.sync, None),
        ("When did my notes last sync?", Action.sync, None),
        # review round 5: the command word ends at whitespace, ":" or the end
        ("/remove: Kafka basics", Action.remove_page, "Kafka basics"),
        ("/sync\n", Action.sync, "/sync"),
        # a remember fact may contain clauses that are not a second request
        ("Remember that I prefer Java, not Kotlin", Action.remember, None),
        ("remember that Postgres is MVCC, which means readers never block", Action.remember, None),
        ("remember that A, B and C are my teams", Action.remember, None),
    ],
)
def test_explicit_actions_are_detected(message: str, action: Action, query: str | None) -> None:
    intent = detect(message)
    assert intent.action is action
    assert intent.query == (query if query is not None else message)


@pytest.mark.parametrize(
    "question",
    [
        "How does Redis remember keys after a restart?",
        "Remember the CAP theorem?",
        "What does Kafka do when a consumer forgets to commit?",
        "How do I remove a node from a Redis cluster?",
        "Remove the duplicates from a Kafka topic?",
        "Which sources of truth does the saga pattern use?",
        "What is the difference between sync and async replication?",
        "/explain kafka partitions",  # unknown command: a question
        "Explain the outbox pattern. Also, remember that ...",  # not at the start
        # The owner's corpus is DB / Kafka / system design: "index" is a core noun there, so
        # none of these may bind the destructive remove_from_index (review round 1, P1).
        "Can you delete a key from the index in a B-tree?",
        "Drop duplicate rows from the index?",
        "Delete the old offsets from the index segment when compacting? explain",
        "Delete the page table entry from the index",
        "Remove the leader from the ISR when it lags",
        "Remove the stopwords before vectorizing text",
        "remove the handbook",  # no "from my index": use /remove
        "Remove it",
        "Stop indexing during bulk loads to speed up inserts",
        "Can you remove the Kafka basics page from my index?",  # destructive: never a question
        "Remove the Kafka basics page from my index?",
        "Forget about the Postgres fact?",
        "Could you forget that I use Postgres 16",  # destructive: no "can you" form
        # replication questions about the corpus are not the user's sync status (round 2)
        "Sync status of replicas in Kafka",
        "what is the sync status of a Kafka follower",
        "When did the follower last sync with the leader?",
        "Did the replica last sync before the failover?",
        "What is the sync status between primary and standby in Postgres?",
        # review round 4: corpus questions that still matched
        "Delete stale documents from the search index",
        "Remove deleted documents from the index in Elasticsearch",
        "Remove deleted notes from the index in Elasticsearch",  # someone else's index
        "forget what I said, explain Raft",
        "Forget about consistency, what does Cassandra do for availability",
        "show me my sources of truth for Kafka consumer lag",
        "which are my sources of inconsistency in a distributed cache",
        # a recall QUESTION must never bind the long-term-memory write
        "Remember that Redis is single threaded?",
        "Don't forget that Kafka retries are not idempotent by default?",
        # review round 5: a pasted path is not a command
        "/sources/foo is a dir",
        "/remove/tmp",
        "/sync/state is stale",
        "/remember.txt holds my notes",
        # review round 5: a remember phrasing followed by a second request is a question
        "keep in mind that Postgres uses MVCC, how does vacuum work",
        "Remember that I use Postgres 16; explain vacuum",
        "remember that Kafka is my queue. What is a consumer group",
        "Note down that we use Redis, and what does SCAN cost",
    ],
)
def test_questions_are_not_actions(question: str) -> None:
    assert detect(question).action is None


def test_commands_keep_their_argument_verbatim() -> None:
    intent = detect("/remember  I prefer Java code examples ")
    assert intent.command
    assert intent.argument == "I prefer Java code examples"
    assert not detect("Remember that I prefer Java examples.").command


@pytest.mark.parametrize(
    ("message", "name", "arguments"),
    [
        (
            "/remember I prefer Java code examples",
            "remember_fact",
            {"user_quote": "I prefer Java code examples"},
        ),
        ("/forget 3f2a9c0d1b7e", "forget_fact", {"fact_id": "3f2a9c0d1b7e"}),
        ("/sources", "list_sources", {}),
        ("/sync", "sync_status", {}),
    ],
)
def test_fixed_argument_commands_are_built_by_the_server(
    message: str, name: str, arguments: dict[str, str]
) -> None:
    assert server_call(detect(message), "c1") == ToolCall(id="c1", name=name, arguments=arguments)
    assert REGISTRY[name].args.model_validate(arguments)  # a valid call, not a guess


@pytest.mark.parametrize(
    "message",
    [
        "/remove Kafka basics",  # choosing the page is the model's (approved) judgment
        "/remember",  # nothing to remember
        "/forget",
        "Remember that I prefer Java examples.",  # a phrasing: the model fills the arguments
        "How does Kafka store records?",
    ],
)
def test_other_turns_get_no_server_call(message: str) -> None:
    assert server_call(detect(message), "c1") is None


def test_a_question_binds_no_tools() -> None:
    assert tools_for_turn(AgentSettings(), "", 0) == ()


@pytest.mark.parametrize("action", list(Action))
def test_an_action_binds_only_its_tools_for_one_round(action: Action) -> None:
    bound = tools_for_turn(AgentSettings(), action.value, 0)
    assert tuple(t.name for t in bound) == ACTION_TOOLS[action]
    assert tools_for_turn(AgentSettings(), action.value, 1) == ()


def test_destructive_actions_keep_their_registry_flag() -> None:
    (remove,) = tools_for_turn(AgentSettings(), Action.remove_page.value, 0)
    (forget,) = tools_for_turn(AgentSettings(), Action.forget.value, 0)
    assert remove.destructive
    assert forget.destructive


def test_max_tool_rounds_zero_disables_even_actions() -> None:
    assert tools_for_turn(AgentSettings(max_tool_rounds=0), Action.remember.value, 0) == ()


def test_always_mode_is_the_old_behaviour_override() -> None:
    always = AgentSettings(tool_mode=ToolMode.always, max_tool_rounds=2)
    assert tools_for_turn(always, "", 0) == TOOL_SPECS
    assert tools_for_turn(always, "", 1) == TOOL_SPECS
    assert tools_for_turn(always, "", 2) == ()


def test_an_unknown_checkpointed_action_binds_nothing() -> None:
    assert tools_for_turn(AgentSettings(), "drop_everything", 0) == ()
