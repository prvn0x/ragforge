"""The interview prompts, built as (system, user) pairs. Pure: an eval harness calls these with
fixed notes and gets byte-identical prompts.

Trust boundary (ADR-0011 rule 7): the SYSTEM prompt holds only server-owned text: the template,
the level rubric, the kind and difficulty rules below. The level is a closed enum (never request
text), so nothing a user typed reaches a system prompt. Everything a user or a page wrote is DATA
in the USER message, inside a fence:
- notes: `<source>` blocks via the chat path's own `fence_hits` (fence-like tags neutralized);
- the topic, the client-sent question, the client-sent expected points, the user's answer and the
  questions already asked (general mode): `inert` turns EVERY `<` into `&lt;`, so no tag at all
  can be forged from inside them. That is stricter than `fencing.neutralize` (which only knows the
  chat path's tag names) and costs nothing: models read `&lt;` as `<`, and these texts are graded,
  never rendered from here.
Fencing lowers the odds that a model obeys injected text; it is not the guarantee. The
guarantees are structural: the reply is parsed strictly (`replies.py`), the score is bounded by
the schema, and every statement that claims to come from the notes is verified (`verify.py`).

Levels (owner decision 2026-10-19): the level changes BOTH what is asked and how it is graded.
Each level names its focus, what it ADDS over the level below (a disjoint delta) and what a 3, 6,
8 and 10 look like (`LEVELS`); difficulty is one step WITHIN the level (`difficulty_rule`):
"harder" must not ask for the next level's delta and "easier" must still ask for this level's own
delta. The two ends are therefore the SAME delta, once forbidden and once required, so
harder-SDE-2 and easier-Senior-SDE-1 cannot ask the same thing. The cumulative focus is never used
as a limit: foci overlap (several levels mention failure modes), and a limit quoting one would
forbid part of the level's own focus. The rubric is generic: it names no company's ladder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ragforge_core.agent.fencing import fence_hits
from ragforge_core.agent.interview.model import Difficulty, Kind, Level
from ragforge_core.agent.state import hit_to_state
from ragforge_core.domain.models import RankedChunk
from ragforge_core.prompts import get_prompt, render

QUESTION_PROMPT: Final = "interview_question"
GRADE_PROMPT: Final = "interview_grade"
GENERAL_QUESTION_PROMPT: Final = "interview_general_question"
GENERAL_GRADE_PROMPT: Final = "interview_general_grade"

# General mode: at most this many earlier questions, each cut to this many characters, go to
# the model (design review: bounded prompt size whatever the client sends).
MAX_EXCLUDED_QUESTIONS: Final = 15
MAX_EXCLUDED_CHARS: Final = 300


@dataclass(frozen=True, slots=True)
class LevelRubric:
    name: str  # how the prompts name the bar
    focus: str  # what a question at this level asks for (cumulative: may repeat lower levels)
    adds: str  # what this level asks for that the level below does not (disjoint, no period)
    aspects: tuple[str, ...]  # the key terms of `adds`, lower case (tests prove disjointness)
    anchors: tuple[str, str, str, str]  # what a 10, 8, 6 and 3 look like


LEVELS: Final[dict[Level, LevelRubric]] = {
    Level.sde_1: LevelRubric(
        "SDE-1",
        "what it is, why it exists and its basic correct usage.",
        "the basics: what it is, why it exists and its correct basic usage",
        ("what it is", "why it exists", "basic usage"),
        (
            "correct definition, why it exists and correct basic usage, with a small example or "
            "consequence; nothing wrong.",
            "the correct what and why; the usage is vague or has one small gap.",
            "the correct what, but the why or the usage is missing.",
            "fragments: some right terms, but the idea itself is missing or partly wrong.",
        ),
    ),
    Level.sde_2: LevelRubric(
        "SDE-2",
        "how it works inside, step by step, and its common failure modes and how they show.",
        "the internal mechanism, step by step, and its common failure modes and how they show",
        ("internal mechanism", "step by step", "failure mode"),
        (
            "the internal mechanism step by step, plus the common failure modes and their "
            "symptoms; nothing wrong.",
            "the mechanism is right; one common failure mode is missing or thin.",
            "the correct what and why, but the mechanism is hand-waved.",
            "fragments: some right terms, but the mechanism is missing or partly wrong.",
        ),
    ),
    Level.senior_sde_1: LevelRubric(
        "Senior SDE-1",
        "the trade-offs, the failure modes, the why behind the design, and production judgement "
        "(what you would choose, monitor or change, and why).",
        "the trade-offs between options and production judgement (what you would choose, "
        "monitor or change, and why)",
        ("trade-off", "production judgement", "monitor"),
        (
            "the mechanism and the why, the main trade-offs, the failure modes and how they show "
            "in production, and a justified choice.",
            "the correct core with the why and the main trade-off; misses a secondary failure "
            "mode or the production angle.",
            "the correct core but shallow: what or how without the trade-offs or failure modes.",
            "fragments: some right terms, but the core idea is missing or partly wrong.",
        ),
    ),
    Level.senior_sde_2: LevelRubric(
        "Senior SDE-2",
        "design across several systems, operating it at scale, and migrating or rolling it out "
        "safely.",
        "design across several systems, operating it at scale, and safe migration or rollout",
        ("several systems", "scale", "migration", "rollout"),
        (
            "places it in a multi-system design, covers its limits at scale and how to operate "
            "them, and gives a safe migration or rollout path.",
            "sound cross-system reasoning; the scale or the migration part is thin.",
            "correct single-system trade-offs only, nothing across systems or at scale.",
            "fragments: some right terms, but the design reasoning is missing or partly wrong.",
        ),
    ),
    Level.staff: LevelRubric(
        "Staff",
        "architecture across teams, cost, risk, long-term evolution, and when NOT to use it.",
        "architecture across teams, cost, risk, long-term evolution, and when NOT to use it",
        ("teams", "cost", "risk", "evolution", "when not"),
        (
            "architecture-level judgement: cost and risk, the impact across teams and owners, "
            "how the design evolves, and when not to use it.",
            "strong architecture reasoning; one of cost, risk or long-term evolution is missing.",
            "solid senior trade-offs without the architecture, cost or evolution view.",
            "fragments: some right terms, but no architecture-level reasoning.",
        ),
    ),
}
_ORDER: Final = tuple(Level)

KIND_RULES: Final[dict[Kind, str]] = {
    Kind.concept: "concept. Ask the candidate to explain an idea from the notes and why it "
    "matters: what problem it solves, what it costs, when it is the wrong choice.",
    Kind.scenario: "scenario. Describe a realistic production situation built from a failure "
    "mode, symptom, limit or risk that the notes describe, and ask the candidate to diagnose it "
    "or say what they would do. Every fact in the scenario must come from the notes.",
    Kind.design: "design. Ask a system-design or trade-off question: choosing between options "
    "the notes compare, or designing around the limits and guarantees they describe.",
    Kind.code: "code. Ask about the code in the notes: what it does, a bug or edge case in it, "
    "how to change it, or the low-level design behind it. If you show code, copy at most 12 "
    "lines of it exactly, in one fenced block.",
}

# General mode has no notes: its own rules (the notes-mode ones say "from the notes").
GENERAL_KIND_RULES: Final[dict[Kind, str]] = {
    Kind.concept: "concept. Ask the candidate to explain an idea of the topic and why it "
    "matters: what problem it solves, what it costs, when it is the wrong choice.",
    Kind.scenario: "scenario. Describe a realistic production situation built from a common "
    "failure pattern of the topic (no company, incident, date or person) and ask the candidate "
    "to diagnose it or say what they would do.",
    Kind.design: "design. Ask a system-design or trade-off question about the topic: choosing "
    "between well-known options, or designing around its limits and guarantees.",
    Kind.code: "code. Ask about a short piece of code or a low-level design on the topic: what "
    "it does, a bug or edge case in it, or how to change it. If you show code, write at most 12 "
    "lines in one fenced block.",
}


def difficulty_rule(level: Level, difficulty: Difficulty) -> str:
    """One step WITHIN the level, bounded by the neighbours' DELTAS (`LevelRubric.adds`), never
    their cumulative focus: "easier" still requires what this level adds over the one below,
    "harder" never asks for what the next level adds. So harder-L forbids exactly what
    easier-(L+1) requires, and no rule forbids part of its own level's focus."""
    at = _ORDER.index(level)
    rubric = LEVELS[level]
    above = LEVELS[_ORDER[at + 1]] if at + 1 < len(_ORDER) else None
    if difficulty is Difficulty.easier:
        floor = (
            f" It still asks for what the {rubric.name} bar adds over the bar below: {rubric.adds}."
            if at > 0
            else ""
        )
        return f"the entry end of the {rubric.name} bar: one focused idea, asked directly.{floor}"
    if difficulty is Difficulty.harder:
        ceiling = (
            f" Do not ask for what the {above.name} bar adds: {above.adds}."
            if above
            else " It is the hardest question of the bar."
        )
        return (
            f"the top end of the {rubric.name} bar: combine two ideas, or push on an edge case, "
            f"within this bar's focus.{ceiling}"
        )
    return f"the middle of the {rubric.name} bar."


def level_anchors(level: Level) -> str:
    ten, eight, six, three = LEVELS[level].anchors
    return "\n".join(
        f"- {score}: {text}" for score, text in ((10, ten), (8, eight), (6, six), (3, three))
    )


@dataclass(frozen=True, slots=True)
class Prompt:
    tag: str  # "name@version", logged with every call (and recorded by evals)
    system: str
    user: str


def inert(text: str) -> str:
    """Untrusted text that must stay inside its fence: no `<` survives, so no tag is formed."""
    return text.replace("<", "&lt;")


def _block(tag: str, text: str) -> str:
    return f"<{tag}>\n{inert(text)}\n</{tag}>"


def fence_notes(notes: Sequence[RankedChunk]) -> str:
    """The notes labelled S1..Sn in the given order, exactly as the chat path packs sources."""
    return fence_hits([hit_to_state(n) for n in notes], list(range(1, len(notes) + 1)))


def _rubric(level: Level) -> dict[str, str]:
    rubric = LEVELS[level]
    return {"level_name": rubric.name, "level_focus": rubric.focus}


def question_prompt(
    notes: Sequence[RankedChunk],
    *,
    kind: Kind,
    difficulty: Difficulty,
    topic: str | None,
    level: Level,
) -> Prompt:
    system = render(
        QUESTION_PROMPT,
        **_rubric(level),
        kind_rule=KIND_RULES[kind],
        difficulty_rule=difficulty_rule(level, difficulty),
    )
    parts = [fence_notes(notes)]
    if topic:
        parts.append(_block("topic", topic))
    return Prompt(get_prompt(QUESTION_PROMPT).tag, system, "\n\n".join(parts))


def _key(expected_points: Sequence[str]) -> str:
    return "\n".join(f"- {point}" for point in expected_points) or "(none given)"


def grade_prompt(
    notes: Sequence[RankedChunk],
    *,
    question: str,
    expected_points: Sequence[str],
    answer: str,
    level: Level,
) -> Prompt:
    user = "\n\n".join(
        [
            fence_notes(notes),
            _block("question", question),
            _block("answer_key", _key(expected_points)),
            _block("answer_candidate", answer),
        ]
    )
    system = render(GRADE_PROMPT, **_rubric(level), level_anchors=level_anchors(level))
    return Prompt(get_prompt(GRADE_PROMPT).tag, system, user)


def recent_questions(asked: Sequence[str]) -> tuple[str, ...]:
    """The client's earlier questions as the model sees them: the LAST `MAX_EXCLUDED_QUESTIONS`,
    each one line cut to `MAX_EXCLUDED_CHARS` (blank ones dropped). The latest ones matter most
    for "do not repeat yourself", and the cap bounds the prompt whatever the client sends."""
    flat = (" ".join(q.split())[:MAX_EXCLUDED_CHARS] for q in asked)
    return tuple(q for q in flat if q)[-MAX_EXCLUDED_QUESTIONS:]


def general_question_prompt(
    *,
    topic: str,
    kind: Kind,
    difficulty: Difficulty,
    level: Level,
    asked_before: Sequence[str] = (),
) -> Prompt:
    """No notes at all: the topic and the (already bounded, `recent_questions`) earlier
    questions, both fenced as data."""
    system = render(
        GENERAL_QUESTION_PROMPT,
        **_rubric(level),
        kind_rule=GENERAL_KIND_RULES[kind],
        difficulty_rule=difficulty_rule(level, difficulty),
    )
    parts = [_block("topic", topic)]
    if asked_before:
        parts.append(_block("asked_before", "\n".join(f"- {q}" for q in asked_before)))
    return Prompt(get_prompt(GENERAL_QUESTION_PROMPT).tag, system, "\n\n".join(parts))


def general_grade_prompt(
    *, question: str, expected_points: Sequence[str], answer: str, level: Level
) -> Prompt:
    user = "\n\n".join(
        [
            _block("question", question),
            _block("answer_key", _key(expected_points)),
            _block("answer_candidate", answer),
        ]
    )
    system = render(GENERAL_GRADE_PROMPT, **_rubric(level), level_anchors=level_anchors(level))
    return Prompt(get_prompt(GENERAL_GRADE_PROMPT).tag, system, user)
