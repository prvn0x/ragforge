"""No model call is ever recorded as `prompt="unknown"` (ADR-0018 R2-3).

Two complementary proofs:
- behaviour: real agent turns (the D1 in-memory harness: the production graph and nodes)
  through an `InstrumentedChatModel` reach every chat prompt (answer, actions, action_reply,
  rewrite, contextualize, answer_strict, ground_check, extras_*), and each call carries its own
  tag; interview calls are labelled by `ask_json` with the prompt they were built from;
- structure: an AST scan of agent/** and apps/** proves every model call site sits inside a
  `with for_prompt(...)` block, so a NEW call site that forgets one fails here, not on the board.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import ragforge_core
from module_d1.d1_fakes import (
    COLLECTION,
    FakeReranker,
    Harness,
    Stream,
    call,
    collect,
    make_harness,
    make_hit,
    principal,
    say,
    verdict,
)
from module_obs.obs_metrics_support import Delta, ScriptedChat
from ragforge_core.agent import AgentDeps, AgentSettings, LangGraphAgentRunner
from ragforge_core.agent.interview.prompts import (
    GENERAL_GRADE_PROMPT,
    GENERAL_QUESTION_PROMPT,
    GRADE_PROMPT,
    QUESTION_PROMPT,
    Prompt,
)
from ragforge_core.agent.interview.replies import ModelReplyInvalid, QuestionReply, ask_json
from ragforge_core.observability.llm import InstrumentedChatModel
from ragforge_core.prompts import get_prompt

REQUESTS = "ragforge_llm_requests_total"
SRC = Path(ragforge_core.__file__).parent
SETTINGS = AgentSettings(heartbeat_interval_s=3600, extras=True)


def tag(name: str) -> str:
    return get_prompt(name).tag


def instrumented(script: list[Any], hits: list[Any] | None = None) -> Harness:
    """The D1 harness with its scripted model behind the production decorator."""
    h = make_harness(script, results=[hits] if hits is not None else None, settings=SETTINGS)
    deps = AgentDeps(
        chat=InstrumentedChatModel(h.chat),
        retriever=h.retriever,
        reranker=FakeReranker(h.txm),
        embedders=h.embedders,
        runs=h.runs,
        tx=h.txm,
        jobs=h.jobs,
        conversations=h.conversations,
        knowledge=h.knowledge,
        store=h.store,
        settings=SETTINGS,
    )
    h.runner = LangGraphAgentRunner(deps, InMemorySaver())
    return h


def prompts_called(delta: Delta) -> list[str]:
    """The prompt label of every model call that happened, one entry per call."""
    out: list[str] = []
    for (name, labels), increase in delta.changes().items():
        if name == REQUESTS:
            out += [dict(labels)["prompt"]] * int(increase)
    return sorted(out)


async def _turn(h: Harness, message: str, conversation: Any = None) -> list[Any]:
    return await collect(h.runner.run_turn(principal(), conversation, [COLLECTION], message))


# ---------------------------------------------------------------------- chat turns
async def test_an_answered_turn_labels_answer_and_ground_check() -> None:
    h = instrumented([say("Fact one holds [S1]."), verdict()])
    delta = Delta((REQUESTS,))
    await _turn(h, "what is fact one?")
    assert prompts_called(delta) == sorted([tag("answer"), tag("ground_check")])


async def test_a_follow_up_labels_contextualize() -> None:
    h = instrumented([say("Fact one holds [S1]."), verdict()])
    done = (await _turn(h, "what is fact one?"))[-1]
    h.script(say("What is fact two?"), say("Fact two holds [S2]."), verdict())
    delta = Delta((REQUESTS,))
    await _turn(h, "and the other?", done.conversation_id)
    assert tag("contextualize") in prompts_called(delta)


async def test_a_weak_grade_labels_rewrite_and_abstains_without_a_model_call() -> None:
    weak = [make_hit(1, 0.05)]
    h = instrumented([say("alternative query")], hits=weak)
    delta = Delta((REQUESTS,))
    await _turn(h, "unknown topic?")
    assert prompts_called(delta) == [tag("rewrite")]  # abstain is server text: no call


async def test_an_invalid_citation_labels_the_strict_regeneration() -> None:
    h = instrumented([say("Fact one is false [S9]."), say("Fact one is true [S1]."), verdict()])
    delta = Delta((REQUESTS,))
    await _turn(h, "fact one?")
    assert prompts_called(delta) == sorted(
        [tag("answer"), tag("answer_strict"), tag("ground_check")]
    )


async def test_an_action_labels_actions_then_action_reply() -> None:
    quote = "I prefer Java code examples in answers"
    h = instrumented(
        [say("", call("remember_fact", user_quote=quote)), say("Saved [T1]."), verdict()]
    )
    delta = Delta((REQUESTS,))
    await _turn(h, f"Remember that {quote}")
    assert prompts_called(delta) == sorted(
        [tag("actions"), tag("action_reply"), tag("ground_check")]
    )


@pytest.mark.parametrize(
    ("message", "extra", "kind"),
    [
        (
            "How does Kafka store records?",
            say("Common failure pattern: a rebalance loop."),
            "incident",
        ),
        (
            "Show the code for a Kafka producer",
            say("```java\nproducer.send(record);\n```\nSends one record."),
            "code",
        ),
    ],
)
async def test_extras_are_labelled_by_their_kind(message: str, extra: Stream, kind: str) -> None:
    kafka = make_hit(1, title="Kafka basics", text="Kafka stores records in partitions.")
    answer = "**In simple terms:** Kafka keeps messages in partitions [S1]."
    h = instrumented([say(answer), verdict(), extra], hits=[kafka])
    delta = Delta((REQUESTS,))
    await _turn(h, message)
    assert tag(f"extras_{kind}") in prompts_called(delta)


# ---------------------------------------------------------------------- interview
@pytest.mark.parametrize(
    "name", [QUESTION_PROMPT, GRADE_PROMPT, GENERAL_QUESTION_PROMPT, GENERAL_GRADE_PROMPT]
)
async def test_interview_calls_carry_the_tag_of_the_prompt_they_were_built_from(name: str) -> None:
    prompt = Prompt(tag(name), "system", "user")
    reply: dict[str, object] = {"question": "q", "topic": "t", "expected_points": ["a"]}
    chat = InstrumentedChatModel(ScriptedChat(payload=reply))
    delta = Delta((REQUESTS,))
    await ask_json(chat, prompt, QuestionReply, chunk_ids=())
    assert prompts_called(delta) == [tag(name)]


async def test_a_repaired_interview_reply_labels_both_attempts() -> None:
    prompt = Prompt(tag(QUESTION_PROMPT), "system", "user")
    chat = InstrumentedChatModel(ScriptedChat(payload={"nonsense": 1}))
    delta = Delta((REQUESTS,))
    with pytest.raises(ModelReplyInvalid):
        await ask_json(chat, prompt, QuestionReply, chunk_ids=())
    assert prompts_called(delta) == [tag(QUESTION_PROMPT)] * 2


# ---------------------------------------------------------------------- structure
# Calls that reach the model. `_ground_check` / `_ask_json` are the bodies their public
# wrappers run inside `for_prompt`, so their CALL SITES are checked like any model call.
MODEL_CALLS = {"consume_stream", "complete_json", "stream", "_ground_check", "_ask_json"}
# Functions whose body IS the plumbing (the label is set by whoever calls them).
PLUMBING = {
    ("agent/llm.py", "consume_stream"),
    ("agent/eval_verify.py", "stream"),
    ("agent/eval_verify.py", "complete_json"),
    ("agent/grounding.py", "_ground_check"),
    ("agent/interview/replies.py", "_ask_json"),
}


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_for_prompt(item: ast.withitem) -> bool:
    expr = item.context_expr
    return isinstance(expr, ast.Call) and _called_name(expr) == "for_prompt"


def _unlabelled(tree: ast.AST, rel: str) -> Iterator[str]:
    def visit(node: ast.AST, function: str, labelled: bool) -> Iterator[str]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function, labelled = node.name, False
        if isinstance(node, ast.With | ast.AsyncWith) and any(map(_is_for_prompt, node.items)):
            labelled = True
        if (
            isinstance(node, ast.Call)
            and _called_name(node) in MODEL_CALLS
            and _is_model_stream(node)
            and not labelled
            and (rel, function) not in PLUMBING
        ):
            yield f"{rel}:{node.lineno} {function}() calls {_called_name(node)} outside for_prompt"
        for child in ast.iter_child_nodes(node):
            yield from visit(child, function, labelled)

    yield from visit(tree, "<module>", False)


def _sources() -> Iterator[tuple[str, ast.AST]]:
    for package in ("agent", "apps"):
        for path in sorted((SRC / package).rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            yield rel, ast.parse(path.read_text(encoding="utf-8"))


def _is_model_stream(node: ast.Call) -> bool:
    """`.stream(` of the ChatModel port, not Starlette's `request.stream()` (no arguments)."""
    return _called_name(node) != "stream" or bool(node.args)


def test_every_model_call_site_is_inside_for_prompt() -> None:
    problems = [problem for rel, tree in _sources() for problem in _unlabelled(tree, rel)]
    assert problems == [], "\n".join(problems)


def test_the_scan_finds_the_call_sites_it_guards() -> None:
    """The guard is only as good as its matcher: it must SEE the known call sites."""
    seen = 0
    for _, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in MODEL_CALLS:
                seen += _is_model_stream(node)
    assert seen >= 10  # contextualize, rewrite, generate, strict, extras, ground, ask_json, ...


def test_the_scan_flags_a_call_outside_for_prompt() -> None:
    bad = ast.parse("async def node(chat):\n    await chat.complete_json('s', [], {})\n")
    good = ast.parse(
        "async def node(chat):\n    with for_prompt('x@1'):\n"
        "        await chat.complete_json('s', [], {})\n"
    )
    assert list(_unlabelled(bad, "agent/x.py")) != []
    assert list(_unlabelled(good, "agent/x.py")) == []
