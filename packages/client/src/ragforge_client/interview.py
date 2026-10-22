"""Interview mode, client side: deterministic phrase/command routing and the session state.

Shared by the UI and `make ask`, so both clients start, steer, pause and end an interview the
same way. Pure: no I/O. The server is stateless per call (`interview_question` /
`interview_grade`); the SESSION lives here: the topics, the difficulty, which question kind comes
next, the chunks already asked (sent back as `exclude_chunk_ids`, so nothing repeats), and the
history the summary is built from.

Why deterministic (owner decision, NEXT.md "Interview mode"): the mode is chosen EXPLICITLY, by a
toggle or a fixed phrase, never inferred by a model. Live, "ask me interview questions" went
through Q&A and "where's the question?" became a new search; a phrase table is testable both ways.

Starting: "interview me [on X]", "quiz me [on X]", "ask me [some] [interview] questions [on|about
X]", `/interview [X]`. X may be a LIST ("Kafka, Redis", "Kafka and Redis"): the session then
alternates topics per question (the API takes one topic per request). A quoted X ("interview me
on "publish and subscribe"") is one topic. No X (or "anything") means mixed topics.

Routing while the mode is on (`route`), first match wins (design review, binding):
1. a session command, matched against the WHOLE message (so an answer that merely contains
   "again" or "stop" is still an answer): repeat, skip, next, harder, easier, next topic, stop.
   The same polite frame as start phrases is allowed around it ("ok stop", "next question
   please", "can you repeat that"): a command misread as an answer would cost a grading call,
   score low, step the level down and add a false weak topic;
2. `/answer [text]`: always an answer (the escape hatch for rule 4);
3. a start phrase ("interview me on Redis"): same session, new topic(s);
4. a NORMAL QUESTION leaves the mode (owner decision). Deterministic rule: the message ends in
   "?" (or the full-width U+FF1F) AND its first word is an interrogative opener (what, why, how,
   when, where, which, who, whom, whose, explain, can, could, should, would, will, is, are, was,
   were, does, do, did) or it starts with "tell me". A HEDGED answer ("Is it because the ISR
   shrinks and the leader can't commit?") matches too, and leaves. That is why leaving is
   REVERSIBLE: the session is only SUSPENDED (pending question and history kept), and the reply
   offers "↩ That was my answer — grade it" (UI) / `/answer` (terminal), which grades the very
   message that left. The cost of a wrong guess is one Q&A turn and one click;
5. anything else is the answer to the pending question. With NO question pending (it was graded,
   or fetching the next one failed) there is nothing to grade, so the text leaves too.

While SUSPENDED (`route_suspended`), only `/answer [text]`, `/resume`, `/stop`, `/general` and
start phrases touch the interview; everything else is ordinary Q&A.

Phase 3c (owner decisions 2026-10-19):
- LEVEL: the bar a session is asked and graded at (SDE-1 .. Staff, default Senior SDE-1). Set by
  the UI dropdown or a phrase ("... at staff level", "... as a senior SDE 2", `--level staff`;
  `parse_level` is the one deterministic table). It is FIXED for a session, so a session's
  average means one bar: a start phrase naming another level ends this session and starts a new
  one (`Do.restart`). Difficulty (easier/standard/harder) stays a fine step WITHIN the level.
- GENERAL-KNOWLEDGE FALLBACK: when the server answers 409 `INTERVIEW_NO_NOTES` or
  `INTERVIEW_EXHAUSTED` for a TOPIC, the session records an `Offer` for that topic and the
  client offers it; it never switches by itself. Accepting (`General`: a button, `/general
  [topic]`, or "y" right after the offer when no question is pending) puts the topic back as a
  general-knowledge topic: its questions are model knowledge, not from the notes, never
  verified. A mixed session (no topic) is never offered the fallback. General and notes scores
  are never averaged together (`Summary`).


After a grade the two clients differ ON PURPOSE: the terminal fetches the next question at once
(the grade is already printed above the prompt), while the UI waits for "Next question" so the
grade card can be read. In the UI, free text typed after a grade (no question pending) therefore
follows rule 5 and goes to Q&A, suspending the interview; "Back to the interview" resumes it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import cycle, islice
from uuid import UUID

from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    Citation,
    InterviewDifficulty,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewKind,
    InterviewLevel,
    InterviewQuestion,
    InterviewQuestionRequest,
    InterviewSource,
    Problem,
)

MAX_TOPIC_CHARS = 200  # InterviewQuestionRequest.topic
MAX_TOPICS = 8  # a longer list is clipped: one sitting can't cover more anyway
MAX_ANSWER_CHARS = 8000  # InterviewGradeRequest.answer
MAX_QUESTION_CHARS = 2000  # InterviewGradeRequest.question
MAX_POINT_CHARS = 500  # InterviewGradeRequest.expected_points items
MAX_POINTS = 20  # InterviewGradeRequest.expected_points
MAX_CHUNKS = 10  # InterviewGradeRequest.chunk_ids
MAX_EXCLUDED = 500  # InterviewQuestionRequest.exclude_chunk_ids
# Read timeout for the two interview calls, shared by the UI and `make ask`. They are plain
# request/response, but each runs several model calls server-side (a grade is grading + a model
# answer + its verification) and sends nothing until done, so the SDK's 30 s default would time
# out a healthy server, and the server would still finish and charge the interview budget.
INTERVIEW_TIMEOUT_S = 120.0
WEAK_SCORE = 6  # a graded answer at or below this marks its topic as weak in the summary
MAX_EXCLUDED_QUESTIONS = 20  # InterviewQuestionRequest.exclude_questions
MAX_EXCLUDED_QUESTION_CHARS = 1000  # ... and each item
DEFAULT_LEVEL = InterviewLevel.senior_sde_1  # owner decision: the default bar

# The one wording for general-knowledge content, shared by the UI and `make ask` so it reads the
# same everywhere (owner decision: a clear, consistent label; never styled like notes content).
GENERAL_LABEL = "General knowledge · not verified"
GENERAL_NOTE = "Written by the model, not from your notes. Nothing here was checked against them."
GENERAL_GRADE_LABEL = (
    "General knowledge · not verified · graded by the model against its own answer key; "
    "not checked against your notes or any source"
)
# A notes question whose expected points all failed the server's notes check comes with none
# (Phase 3c): the grade judges against the notes alone. Shown only where the key would be.
NO_KEY_NOTE = "No checked answer key for this question; graded against your notes"

LEVEL_NAMES: dict[str, str] = {
    InterviewLevel.sde_1: "SDE-1",
    InterviewLevel.sde_2: "SDE-2",
    InterviewLevel.senior_sde_1: "Senior SDE-1",
    InterviewLevel.senior_sde_2: "Senior SDE-2",
    InterviewLevel.staff: "Staff",
}

# One question of each kind in turn, so a session covers all four (owner decision).
KIND_ROTATION: tuple[InterviewKind, ...] = (
    InterviewKind.concept,
    InterviewKind.scenario,
    InterviewKind.design,
    InterviewKind.code,
)
_LEVELS: tuple[InterviewDifficulty, ...] = (
    InterviewDifficulty.easier,
    InterviewDifficulty.standard,
    InterviewDifficulty.harder,
)


def is_exhausted(problem: Problem) -> bool:
    """The question endpoint's "nothing new left on this topic". Decided by the Problem TYPE,
    never by the status code alone: a 404 can also mean an older server without the endpoint."""
    return problem.type == INTERVIEW_EXHAUSTED


class Fallback(StrEnum):
    """Why the notes gave no question on a topic: the two 409s where the general-knowledge
    fallback may be OFFERED."""

    no_notes = "no_notes"  # nothing in the notes matches the topic at all
    exhausted = "exhausted"  # every matching note was already asked this session


def fallback_reason(problem: Problem) -> Fallback | None:
    """The 409 a client may offer the fallback on, by Problem TYPE (never the status alone: a
    plain 409 is something else); None for every other problem."""
    if problem.type == INTERVIEW_NO_NOTES:
        return Fallback.no_notes
    if problem.type == INTERVIEW_EXHAUSTED:
        return Fallback.exhausted
    return None


# ------------------------------------------------------------------ levels
_LEVEL_NUMBERS = {"1": 1, "i": 1, "one": 1, "2": 2, "ii": 2, "two": 2}
_LEVEL_ROLE = r"sde|swe|software\s+(?:development\s+)?engineer|engineer"
_LEVEL_FORM = re.compile(
    rf"(?P<staff>staff)(?:\s+(?:{_LEVEL_ROLE}))?"
    rf"|(?:(?P<senior>senior|sr)\s*)?(?P<role>{_LEVEL_ROLE})?\s*(?P<n>1|ii|i|one|2|two)?"
)


def parse_level(text: str) -> InterviewLevel | None:
    """A level as people type it, or None. Deterministic table (owner decision), tolerant of
    spelling: "Staff", "staff engineer", "SDE2", "sde-2", "SDE II", "senior SDE 2", "Sr. SDE-1",
    "senior" (= Senior SDE-1), the enum values ("senior_sde_2"), an article before it and
    "level" after it. A bare number or "sde" alone is not a level (no guessing)."""
    spaced = re.sub(r"[-_./]", " ", text.casefold())
    spaced = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", spaced)  # "sde2" -> "sde 2"
    words = spaced.split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    if words and words[-1] == "level":
        words = words[:-1]
    match = _LEVEL_FORM.fullmatch(" ".join(words)) if words else None
    if match is None:
        return None
    if match["staff"]:
        return InterviewLevel.staff
    number = _LEVEL_NUMBERS.get(match["n"] or "")
    if match["senior"]:
        return InterviewLevel.senior_sde_2 if number == 2 else InterviewLevel.senior_sde_1
    if match["role"] and number is not None:
        return InterviewLevel.sde_1 if number == 1 else InterviewLevel.sde_2
    return None


def level_name(level: InterviewLevel | str) -> str:
    """Display name of a level; one this client doesn't know (newer server) as its text."""
    return LEVEL_NAMES.get(level, str(level))


def level_choices() -> str:
    return ", ".join(LEVEL_NAMES.values())


# ------------------------------------------------------------------ routing
@dataclass(frozen=True, slots=True)
class Start:
    """Start (or switch) an interview. Empty `topics` = mixed topics across all the notes.
    `level` None: not named (a new session uses `DEFAULT_LEVEL`; a running one keeps its own).
    `unknown_level`: the text after `--level`, or a level-like "at principal level" clause, that
    `parse_level` didn't know (the client says so and goes on at the session's level)."""

    topics: tuple[str, ...] = ()
    level: InterviewLevel | None = None
    unknown_level: str | None = None


class Command(StrEnum):
    repeat = "repeat"
    skip = "skip"
    next = "next"  # the next question (after a grade); with one pending it skips it
    harder = "harder"
    easier = "easier"
    next_topic = "next topic"
    stop = "stop"
    resume = "resume"  # back into a suspended interview; in the mode it repeats the question


@dataclass(frozen=True, slots=True)
class Run:
    command: Command


@dataclass(frozen=True, slots=True)
class Answer:
    """Grade `text`. None (a bare `/answer`): the message that suspended the session, if any."""

    text: str | None


@dataclass(frozen=True, slots=True)
class Leave:
    """Suspend the mode; `message` goes to normal Q&A."""

    message: str


@dataclass(frozen=True, slots=True)
class General:
    """Accept the general-knowledge offer for `topic` (None: the newest open offer). Only ever
    from an explicit human act: a button, `/general [topic]`, or "y" while the session waits on
    its offers (`InterviewSession.offering`)."""

    topic: str | None = None


type Route = Start | Run | Answer | Leave | General

# Trailing punctuation dropped before matching. Not quotes: a quoted topic must keep its closing
# quote so `split_topics` sees it whole ("interview me on "publish and subscribe".").
_TRAILING = " \t\n.!?…,;:"
_QUOTES = "\"'\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}"
_POLITE = r"(?:(?:please|ok|okay|now|so|hey|(?:can|could|would|will)\s+you)[\s,]+)*"
# Topic words that mean "no particular topic".
_ANY_TOPIC = frozenset(
    {"anything", "everything", "all", "all topics", "any topic", "random", "random topics", "mixed"}
)
_START = re.compile(
    rf"""^{_POLITE}(?:
        /interview(?:\s+(?P<command_topic>.+?))?
      # "interview me as a staff engineer on Kafka": a level BEFORE the topic (a level after it
      # is taken off first, `_take_level`); a non-level here ("as a friend") is no start
      | (?:interview|quiz)\s+me(?:\s+(?:as|at)\s+(?P<me_level>.+?))?
            (?:\s+(?:on|about)\s+(?P<me_topic>.+?))?
      # live failure (owner typed "INTERVIEW MODE" and got a Q&A refusal): "interview mode [on
      # X]", "interview mode on", "start [an|the] interview [mode] [on X]"; a bare "interview"
      # or "interview mode is ..." stays a normal message
      | (?:(?:start|begin|enter)\s+(?:an?\s+|the\s+)?interview(?:\s+mode)?
           | interview\s+mode(?:\s+on)?)
            (?:\s+(?:on|about|for)\s+(?P<mode_topic>.+?))?
      | (?:ask|give)\s+me\s+(?:some\s+|a\s+few\s+|more\s+|\d+\s+)?(?:interview\s+)?questions
            (?:\s+(?:on|about)\s+(?P<ask_topic>.+?))?
      # topic first: "ask me Kafka interview questions" (and the owner's own word order,
      # "ask me java questions interview"); the ask/give prefix keeps "what are Kafka interview
      # questions?" a normal question
      | (?:ask|give)\s+me\s+(?:some\s+|a\s+few\s+|more\s+|\d+\s+)?(?P<pre_topic>.+?)
            \s+(?:interview\s+questions|questions\s+(?:for\s+(?:an?\s+)?)?interview)
    )(?:[\s,]+please)?$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
# "Kafka, Redis", "Kafka and Redis", "Kafka, Redis, and Postgres", "Kafka & Redis", "Kafka; Redis"
_TOPIC_SEPARATOR = re.compile(r"\s*(?:,\s*and\s+|,|;|\s&\s|\s+and\s+)\s*", re.IGNORECASE)

_COMMANDS: dict[str, Command] = {
    phrase: command
    for command, phrases in {
        Command.repeat: (
            "repeat",
            "/repeat",
            "repeat the question",
            "repeat that",
            "again",
            "say again",
            "say that again",
            "where's the question",
            "where is the question",
            "what was the question",
            "what's the question",
            "what is the question",
        ),
        Command.skip: ("skip", "/skip", "skip it", "skip this", "skip this one", "pass"),
        Command.next: (
            *("next", "/next", "next question", "next one", "another", "another one"),
            *("another question", "one more", "continue"),
        ),
        Command.harder: (
            "harder",
            "/harder",
            "make it harder",
            "harder question",
            "more difficult",
        ),
        Command.easier: ("easier", "/easier", "make it easier", "easier question", "simpler"),
        Command.next_topic: (
            "next topic",
            "/topic",
            "/next-topic",
            "change topic",
            "change the topic",
            "new topic",
            "different topic",
            "switch topic",
            "another topic",
        ),
        Command.stop: (
            "stop",
            "/stop",
            "stop it",
            "stop now",
            "stop the quiz",
            "stop quiz",
            "end the quiz",
            "end quiz",
            "quit the quiz",
            "end",
            "end interview",
            "end the interview",
            "stop interview",
            "stop the interview",
            "exit",
            "quit",
            "finish",
            "/end",
        ),
        Command.resume: ("/resume", "resume", "resume interview", "resume the interview"),
    }.items()
    for phrase in phrases
}
# While suspended only explicit, slash-or-"resume" commands count: "stop" or "again" typed in a
# normal conversation is a chat message, not an order to the paused interview.
_SUSPENDED_COMMANDS = frozenset({"/resume", "resume interview", "resume the interview", "/stop"})
# The polite frame around a command: the start phrases' `_POLITE` prefix and a trailing "please".
_COMMAND_FRAME = re.compile(rf"^{_POLITE}(?P<core>.+?)(?:[\s,]+please)?$", re.DOTALL)
_ANSWER = re.compile(r"^/answer(?:\s+(?P<text>.*))?$", re.IGNORECASE | re.DOTALL)
_OPENERS = frozenset(
    {
        *("what", "why", "how", "when", "where", "which", "who", "whom", "whose", "explain"),
        *("can", "could", "should", "would", "will", "is", "are", "was", "were"),
        *("does", "do", "did"),
    }
)


def _straight(message: str) -> str:
    """Phone keyboards send U+2019 in "where's"; CJK keyboards send the full-width
    question mark (U+FF1F)."""
    return message.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'").replace(
        "\N{FULLWIDTH QUESTION MARK}", "?"
    )


def _flat(message: str) -> str:
    """Single-spaced, straight quotes, trailing punctuation off."""
    return " ".join(_straight(message).split()).rstrip(_TRAILING)


def _normal(message: str) -> str:
    return _flat(message).casefold()


def split_topics(raw: str) -> tuple[str, ...]:
    """The topic list of a start phrase: separated by commas, semicolons, "and" or "&";
    de-duplicated (case-insensitive), each clipped to the API limit, at most `MAX_TOPICS`.
    A fully quoted topic is ONE topic ("publish and subscribe"). Empty or "anything" = mixed."""
    text = raw.strip().rstrip(_TRAILING).strip()
    if len(text) > 1 and text[0] in _QUOTES and text[-1] in _QUOTES:
        parts = [text[1:-1]]
    else:
        parts = _TOPIC_SEPARATOR.split(text)
    topics: dict[str, str] = {}
    for part in parts:
        topic = part.strip().strip(_QUOTES).strip()[:MAX_TOPIC_CHARS]
        if topic and topic.casefold() not in _ANY_TOPIC:
            topics.setdefault(topic.casefold(), topic)
    return tuple(topics.values())[:MAX_TOPICS]


# Leading filler before a topic-first phrase ("the top 10 Kafka interview questions"): left in,
# "the top 10 Kafka" matches no page title.
_FILLER = re.compile(
    r"^(?:(?:the|a|an|about|some|top|\d+|most|commonly|common|frequently|important|good|basic|"
    r"advanced|hard|tough|tricky|asked)(?:\s+|$))+",
    re.IGNORECASE,
)
# "give me the answers to / an overview of / a list of Kafka interview questions" asks for
# content about interview questions, not to be interviewed: no start.
_NOT_AN_INTERVIEW = re.compile(
    r"^(?:the\s+|an?\s+)?(?:answers?|overview|list|summary|explanation)\b", re.IGNORECASE
)


# `/interview kafka --level staff` (also `--level=staff`; a missing value is an unknown level)
_LEVEL_FLAG = re.compile(r"\s+--level(?:(?:\s*=\s*|\s+)(?P<level>.*))?$", re.IGNORECASE)
# "... at staff level", "... as a senior SDE 2", "... for SDE-2": the clause STARTS here
_LEVEL_LEAD = re.compile(r"[\s,]+(?:at|as|for)\s+", re.IGNORECASE)
# A clause that is MEANT as a level the table doesn't know: "at principal level", "at L5",
# "for a junior engineer", "at sde". Kept narrow (a level-like head word, or one or two words
# then "level", never "the ... level") so "caching at the application level" stays a topic,
# and never across "on/about" (a level BEFORE the topic is `_START`'s `me_level`).
_LEVEL_LIKE = re.compile(
    r"(?:an?\s+)?(?:"
    r"(?:principal|junior|jr|distinguished|intern|senior|sr|staff|sde|swe|l\d+|e\d+|ic\d+)"
    r"(?:\s+(?!(?:on|about)\b)[\w.-]+){0,3}"  # "as a staff engineer ON Kafka": the topic
    r"|(?!the\b)[\w-]+(?:\s+[\w-]+)?\s+level)",
    re.IGNORECASE,
)
# The polite trailing "please" (`_START` allows it): off BEFORE the level clause is looked for,
# or "Kafka at staff level, please" keeps "at staff level" in the topic.
_TRAILING_PLEASE = re.compile(r"[\s,]+please$", re.IGNORECASE)


def _take_level(text: str) -> tuple[str, InterviewLevel | None, str | None]:
    """(text without its trailing level clause, the level, the unknown level text).

    A trailing "at/as/for <X>" is a level clause only when ALL of X is a level (`parse_level`),
    so a topic that merely contains "at" or "as" ("data at rest", "Kafka as a log") is kept
    whole. The leftmost such clause wins ("data at rest at staff level" -> "data at rest").
    Failing that, a clause that LOOKS like a level (`_LEVEL_LIKE`: "at principal level") is an
    unknown level: dropped from the topic and reported, never silently searched as topic text
    while the session runs at the default bar."""
    flag = _LEVEL_FLAG.search(text)
    if flag is not None:
        raw = (flag.group("level") or "").strip()
        level = parse_level(raw)
        return text[: flag.start()], level, None if level is not None else raw
    leads = list(_LEVEL_LEAD.finditer(text))
    for lead in leads:
        level = parse_level(text[lead.end() :])
        if level is not None:
            return text[: lead.start()], level, None
    for lead in leads:
        clause = text[lead.end() :]
        if _LEVEL_LIKE.fullmatch(clause):
            return text[: lead.start()], None, clause
    return text, None, None


def detect_start(message: str) -> Start | None:
    """A start phrase (the WHOLE message, case-insensitive), or None. Never a model's guess."""
    flat = _flat(message)
    polite = _TRAILING_PLEASE.sub("", flat).rstrip(_TRAILING)
    stripped, level, unknown = _take_level(polite)
    match = _START.match(stripped)
    if match is None:
        if stripped == polite:
            return None
        # the "level" clause was really part of a message that is no start phrase without it
        match, level, unknown = _START.match(flat), None, None
        if match is None:
            return None
    pre = match.group("pre_topic")
    if pre and _NOT_AN_INTERVIEW.match(pre):
        return None
    if match.group("me_level") is not None:
        level = parse_level(match.group("me_level"))
        if level is None:  # "interview me as a friend on X" is not a start (as before 3c)
            return None
    raw = (
        match.group("command_topic")
        or match.group("me_topic")
        or match.group("ask_topic")
        or match.group("mode_topic")
        or (_FILLER.sub("", pre) if pre else None)
    )
    return Start(topics=split_topics(raw or ""), level=level, unknown_level=unknown)


def _command_key(message: str) -> str:
    """The message as a command lookup key: normalised, polite frame removed ("ok stop" ->
    "stop", "next question please" -> "next question")."""
    normal = _normal(message)
    framed = _COMMAND_FRAME.match(normal)
    core = framed.group("core").rstrip(_TRAILING) if framed else normal
    return core if core in _COMMANDS else normal


def detect_command(message: str) -> Command | None:
    """A session command: the whole message (inside the polite frame) is one fixed phrase."""
    return _COMMANDS.get(_command_key(message))


def is_normal_question(message: str) -> bool:
    """Rule 4 of the module docstring: ends in "?" (or U+FF1F) and opens like a question."""
    text = _straight(message).strip()
    if not text.endswith("?"):
        return False
    words = _normal(text).split()
    if not words:
        return False
    return words[0] in _OPENERS or words[:2] == ["tell", "me"]


def _forced_answer(message: str) -> Answer | None:
    forced = _ANSWER.match(message.strip())
    if forced is None:
        return None
    return Answer((forced.group("text") or "").strip() or None)


_GENERAL = re.compile(r"^/general(?:\s+(?P<topic>.+))?$", re.IGNORECASE | re.DOTALL)
_YES = frozenset({"y", "yes"})


def detect_general(message: str, *, offering: bool = False) -> General | None:
    """`/general [topic]` always; a bare "y"/"yes" only while `offering` (the session waits on
    its offers: no topic left, NO question pending, so "y" can't be someone's answer)."""
    text = _flat(message)
    explicit = _GENERAL.match(text)
    if explicit is not None:
        topic = (explicit.group("topic") or "").strip().strip(_QUOTES).strip()
        return General(topic or None)
    if offering and text.casefold() in _YES:
        return General()
    return None


def route(message: str, *, pending: bool, offering: bool = False) -> Route:
    """What a message means while the mode is on (module docstring, rules 1-5; `/general` and
    the offer's "y" come right after the commands)."""
    command = detect_command(message)
    if command is not None:
        return Run(command)
    general = detect_general(message, offering=offering)
    if general is not None:
        return general
    forced = _forced_answer(message)
    if forced is not None:
        return forced
    start = detect_start(message)
    if start is not None:
        return start
    if is_normal_question(message) or not pending:
        return Leave(message)
    return Answer(message.strip())


def route_suspended(message: str) -> Route | None:
    """What a message means while the interview is SUSPENDED; None = ordinary Q&A."""
    key = _command_key(message)
    if key in _SUSPENDED_COMMANDS:
        return Run(_COMMANDS[key])
    return detect_general(message) or _forced_answer(message) or detect_start(message)


# ------------------------------------------------------------------ session
def is_general(question: InterviewQuestion | InterviewGrade) -> bool:
    """General-knowledge content (not from the notes, never verified). Anything but an explicit
    `notes` source counts: a source this client doesn't know (newer server) must never be shown
    as notes content or with a verified badge."""
    return question.source != InterviewSource.notes


def general_grade(grade: InterviewGrade, question: InterviewQuestion | None) -> bool:
    """A grade is shown as general knowledge when the grade OR the question it grades says so:
    a general question's grade must never be drawn as notes content, whatever a (buggy or
    newer) server labels the grade."""
    return is_general(grade) or (question is not None and is_general(question))


def _key(topic: str) -> str:
    return topic.casefold()


@dataclass(frozen=True, slots=True)
class Asked:
    """One question of the session and how it ended: graded, or abandoned (skipped).
    `requested_topic`: the SESSION topic it was asked for ("Redis"), which may differ from the
    server's `question.topic` ("Redis eviction"); None for a mixed question."""

    question: InterviewQuestion
    answer: str | None = None
    grade: InterviewGrade | None = None
    requested_topic: str | None = None

    @property
    def score(self) -> int | None:
        return self.grade.score if self.grade is not None else None

    @property
    def general(self) -> bool:
        return is_general(self.question)


@dataclass(frozen=True, slots=True)
class Page:
    title: str
    url: str | None


@dataclass(frozen=True, slots=True)
class WeakTopic:
    topic: str
    kind: str
    score: int
    pages: tuple[Page, ...]  # the notes to revisit


@dataclass(frozen=True, slots=True)
class WeakGeneral:
    """A weak general-knowledge answer: there are no pages to revisit, so the advice is to add
    notes on `topic` (the session topic the owner typed)."""

    topic: str
    kind: str
    score: int


@dataclass(frozen=True, slots=True)
class Offer:
    """An open general-knowledge offer for `topic` (the notes had nothing, or nothing left)."""

    topic: str
    reason: Fallback


@dataclass(frozen=True, slots=True)
class Summary:
    """The session's outcome. Notes-based and general-knowledge scores are kept APART (owner
    decision): `average` is over notes-based grades only, `general_average` over general ones;
    they are never averaged together."""

    asked: int
    answered: int  # every graded answer, notes and general
    skipped: int
    average: float | None  # notes-based grades only; None when none was graded
    weak: tuple[WeakTopic, ...]  # notes-based, with pages to revisit
    exhausted: tuple[str, ...]  # topics with no unasked notes left (all covered)
    level: InterviewLevel | str = DEFAULT_LEVEL
    topics: tuple[str, ...] = ()  # the topics the session was about (empty: mixed)
    notes_answered: int = 0
    general_answered: int = 0
    general_average: float | None = None
    weak_general: tuple[WeakGeneral, ...] = ()


class NoQuestion(Exception):
    """An answer arrived with no question pending."""


class Do(StrEnum):
    """What the client must do after `InterviewSession.apply` (the session is already updated)."""

    ask = "ask"  # fetch the next question (`question_request`)
    grade = "grade"  # grade `Step.text` (`grade_request`)
    show = "show"  # show the pending question again
    end = "end"  # the interview is over: show `summary()`
    chat = "chat"  # the session is suspended: send `Step.text` to normal Q&A
    nothing = "nothing"  # only show `Step.notice`
    # a start phrase named ANOTHER level: end this session (show its summary), then begin a new
    # one from the same `Start` (the level is fixed per session; this one keeps its level, and a
    # pending question is recorded as skipped, as on stop)
    restart = "restart"


@dataclass(frozen=True, slots=True)
class Step:
    do: Do
    text: str = ""
    notice: str | None = None


@dataclass(frozen=True, slots=True)
class Exhausted:
    """Outcome of `InterviewSession.exhausted`: `ended` = nothing is left to ask from and nothing
    to offer; `offer` = the general-knowledge offer this made (the client shows it)."""

    notice: str
    ended: bool
    offer: Offer | None = None


def _clip_kind(kind: InterviewKind | str) -> InterviewKind | None:
    return kind if isinstance(kind, InterviewKind) else None


def _request_source(question: InterviewQuestion) -> InterviewSource:
    """The source to echo on grade. An unknown source (newer server) can't pass the strict
    request enum: no notes ids means it can only be graded as general (never as notes)."""
    if isinstance(question.source, InterviewSource):
        return question.source
    return InterviewSource.notes if question.chunk_ids else InterviewSource.general


@dataclass
class InterviewSession:
    """The client-held interview. Mutated only through its methods.

    `topics` = the topics still asked from (alternated, `upcoming_topic`); empty with no `offers`
    = mixed. `general_topics` (casefolded) are topics the owner moved to general knowledge.
    `offers` = topics the notes had nothing (left) on, waiting for the owner's choice; a session
    with no topics left but open offers is WAITING, never mixed (a mixed session never offers).
    `paused` = suspended by a normal question; `left_message` is that question, kept so "That
    was my answer — grade it" can grade it. `unsent` = an answer whose grading failed
    (`grade_failed`): a bare `/answer` sends it again."""

    topics: list[str] = field(default_factory=list)
    difficulty: InterviewDifficulty = InterviewDifficulty.standard
    level: InterviewLevel = DEFAULT_LEVEL
    current: InterviewQuestion | None = None
    current_topic: str | None = None  # the session topic `current` was asked for
    history: list[Asked] = field(default_factory=list)
    asked_chunk_ids: list[UUID] = field(default_factory=list)
    next_kind: InterviewKind = KIND_ROTATION[0]
    turn: int = 0  # index into `topics` of the next request
    exhausted_topics: list[str] = field(default_factory=list)
    general_topics: set[str] = field(default_factory=set)
    offers: list[Offer] = field(default_factory=list)
    named_topics: list[str] = field(default_factory=list)  # every topic named, for the summary
    paused: bool = False
    left_message: str | None = None
    unsent: str | None = None
    # The collections the pending question was asked from: the grade must use the same ones
    # (the token binds them; the UI selection may change while the user is answering).
    question_collections: list[UUID] = field(default_factory=list)
    # Set by `question_request`, committed by `asked` only when the question really arrived: a
    # failed fetch must not move the pending question's collections (re-review P2).
    requested_collections: list[UUID] = field(default_factory=list)
    requested_topic: str | None = None

    @classmethod
    def begin(cls, start: Start) -> InterviewSession:
        return cls(
            topics=list(start.topics),
            level=start.level or DEFAULT_LEVEL,
            named_topics=list(start.topics),
        )

    @property
    def upcoming_topic(self) -> str | None:
        """The topic the next request asks about (None = mixed, or WAITING on offers)."""
        return self.topics[self.turn % len(self.topics)] if self.topics else None

    @property
    def upcoming_general(self) -> bool:
        topic = self.upcoming_topic
        return topic is not None and _key(topic) in self.general_topics

    @property
    def waiting(self) -> bool:
        """No topic left to ask from, but general-knowledge offers are open."""
        return not self.topics and bool(self.offers)

    @property
    def offering(self) -> bool:
        """A bare "y" accepts the newest offer only while the session WAITS on its offers (no
        topic left to ask, so no question is or will be pending and "y" can't be an answer).

        Not merely "offers open, nothing pending": the UI sits idle after every grade until
        "Next question", so that rule let a "y" typed questions later (meaning "yes, go on")
        switch to general knowledge on an old offer. While other topics remain, the offer is
        taken by its button or `/general <topic>`: the same armed rule the terminal prints
        (`ask.offer_line`)."""
        return self.waiting and self.current is None

    @property
    def number(self) -> int:
        """1-based number of the pending (or next) question, for display."""
        return len(self.history) + 1

    # ------------------------------------------------------------ requests
    def question_request(self, collection_ids: Sequence[UUID]) -> InterviewQuestionRequest:
        """The next question: every kind is allowed, the one due next listed FIRST (the server
        takes the first requested kind the note fits; the rotation then follows whatever kind it
        did ask, see `asked`). The level is ALWAYS sent (the session's, fixed). A general topic
        sends `source=general` and the session's earlier general questions on it, so the model
        does not repeat itself (it has no notes to exclude)."""
        start = KIND_ROTATION.index(self.next_kind)
        kinds = list(islice(cycle(KIND_ROTATION), start, start + len(KIND_ROTATION)))
        self.requested_collections = list(collection_ids)
        topic = self.upcoming_topic
        self.requested_topic = topic
        general = self.upcoming_general
        return InterviewQuestionRequest(
            collection_ids=list(collection_ids),
            topic=topic,
            kinds=kinds,
            difficulty=self.difficulty,
            exclude_chunk_ids=list(self.asked_chunk_ids),
            level=self.level,
            source=InterviewSource.general if general else InterviewSource.notes,
            exclude_questions=self._asked_general(topic) if general and topic else [],
        )

    def _asked_general(self, topic: str) -> list[str]:
        """The newest `MAX_EXCLUDED_QUESTIONS` general questions asked for `topic` (the
        contract's caps), each clipped to the item limit."""
        asked = [
            a.question.question
            for a in self.history
            if a.general
            and a.requested_topic is not None
            and _key(a.requested_topic) == _key(topic)
        ]
        return [q[:MAX_EXCLUDED_QUESTION_CHARS] for q in asked[-MAX_EXCLUDED_QUESTIONS:]]

    def grade_request(self, collection_ids: Sequence[UUID], answer: str) -> InterviewGradeRequest:
        """Raises `NoQuestion` with nothing pending and ValueError for an empty or over-long
        answer (callers say so; the request would only fail validation server-side).

        The server's own question fields are echoed back, clipped to the request limits: a
        slightly over-long question must not make the answer ungradable. Level, source and the
        token are ALWAYS the question's own (design review: the token binds them, and a session
        level must never re-grade a question asked at another). A kind, difficulty or level
        this client doesn't know (a newer server) can't be echoed through the strict request
        enums, so it falls back to the session's (handoff IV-UI.md, contract note). A general
        question sends no chunk ids (the contract requires none).

        A general question with no `question_token` (a server bug: the contract requires one)
        raises a plain ValueError saying so, checked FIRST: otherwise the request's pydantic
        error reached the owner as validation noise, and retyping the answer could never help."""
        question = self.current
        if question is None:
            raise NoQuestion
        source = _request_source(question)
        if source is InterviewSource.general and not question.question_token:
            raise ValueError(
                "this general-knowledge question came without its server token, so it can't be "
                "graded; skip it"
            )
        text = answer.strip()
        if not text:
            raise ValueError("the answer is empty")
        if len(text) > MAX_ANSWER_CHARS:
            raise ValueError(f"the answer is {len(text)} characters; the limit is 8000")
        difficulty = question.difficulty
        return InterviewGradeRequest(
            collection_ids=self.question_collections or list(collection_ids),
            question=question.question[:MAX_QUESTION_CHARS],
            kind=_clip_kind(question.kind) or self._previous_kind(),
            difficulty=difficulty
            if isinstance(difficulty, InterviewDifficulty)
            else self.difficulty,
            chunk_ids=[] if source is InterviewSource.general else question.chunk_ids[:MAX_CHUNKS],
            expected_points=[p[:MAX_POINT_CHARS] for p in question.expected_points[:MAX_POINTS]],
            answer=text,
            question_token=question.question_token,
            level=question.level if isinstance(question.level, InterviewLevel) else self.level,
            source=source,
        )

    def _previous_kind(self) -> InterviewKind:
        """The kind requested first for the pending question (rotation already moved past it)."""
        return KIND_ROTATION[(KIND_ROTATION.index(self.next_kind) - 1) % len(KIND_ROTATION)]

    # ------------------------------------------------------------ transitions
    def asked(self, question: InterviewQuestion) -> None:
        """The reply to the last `question_request` arrived. One still pending is recorded as
        skipped. Its chunks are never asked again this session: `asked_chunk_ids` keeps the
        newest `MAX_EXCLUDED` (the API cap), dropping the OLDEST when full; only a session of
        well over 100 questions could see an early note again, which beats a request the API
        rejects."""
        self.skip()
        self.current = question
        self.current_topic = self.requested_topic
        self.question_collections = list(self.requested_collections)
        known = set(self.asked_chunk_ids)
        self.asked_chunk_ids += [c for c in question.chunk_ids if c not in known]
        del self.asked_chunk_ids[:-MAX_EXCLUDED]
        kind = _clip_kind(question.kind)
        # an unknown kind (newer server) still advances the rotation from where it was
        base = (
            KIND_ROTATION.index(kind) if kind is not None else KIND_ROTATION.index(self.next_kind)
        )
        self.next_kind = KIND_ROTATION[(base + 1) % len(KIND_ROTATION)]
        if self.topics:
            self.turn = (self.turn + 1) % len(self.topics)

    def graded(self, answer: str, grade: InterviewGrade) -> None:
        """Record the grade and adapt: a near-perfect answer (9+) steps the difficulty up, a
        weak one (3 or less) down, always WITHIN the session's level. The owner's explicit
        harder/easier still win: they set the difficulty directly and the next grade adapts
        from there."""
        if self.current is None:
            raise NoQuestion
        self.history.append(Asked(self.current, answer.strip(), grade, self.current_topic))
        self.current = None
        self.left_message = None
        self.unsent = None
        if grade.score >= 9:
            self.harder()
        elif grade.score <= 3:
            self.easier()

    def grade_failed(self, answer: str) -> None:
        """Grading `answer` failed (network, rate limit): keep it for a bare `/answer`."""
        self.unsent = answer.strip() or None

    def skip(self) -> bool:
        """Abandon the pending question (kept in the history as skipped). False: none pending.
        An unsent answer belonged to it, so it goes too."""
        self.unsent = None
        if self.current is None:
            return False
        self.history.append(Asked(self.current, requested_topic=self.current_topic))
        self.current = None
        return True

    def harder(self) -> bool:
        """One difficulty step up (within the level); False when already at the top."""
        return self._step(+1)

    def easier(self) -> bool:
        return self._step(-1)

    def next_topic(self) -> str:
        """Move away from the current topic; returns a notice saying where to.

        - several topics: the next one in the list (the turn already alternates, so this skips
          one ahead of the pending question's topic); notes and general topics alike, each
          question keeps its own label;
        - one fixed topic: cleared to MIXED, because the client has no other topic to pick and
          mixed questions across all the notes are the honest move away. A mixed session never
          offers the general fallback, so open offers are dropped;
        - already mixed: just move on (the excluded chunks prevent repeats).
        The pending question, if any, is recorded as skipped."""
        self.skip()
        if len(self.topics) > 1:
            return f"topic: {self.upcoming_topic}"
        self.topics = []
        self.offers = []
        self.turn = 0
        return "topic: mixed, across all your notes"

    def switch(self, topics: Sequence[str]) -> None:
        """A start phrase during the session: same session (history, no-repeat, level), new
        topics, asked from the notes first (a general topic is always a fresh opt-in)."""
        self.skip()
        self.topics = list(topics)
        self.turn = 0
        self.offers = []
        self.general_topics = set()
        known = {_key(t) for t in self.named_topics}
        self.named_topics += [t for t in topics if _key(t) not in known]

    def exhausted(self, problem: Problem | None = None) -> Exhausted:
        """The server had no question on `upcoming_topic`: its notes have nothing on it
        (`INTERVIEW_NO_NOTES`) or nothing left (`INTERVIEW_EXHAUSTED`, also the default for
        `problem` None). The topic is dropped from the rotation and, for a notes topic, an
        `Offer` is opened: the client OFFERS the general-knowledge fallback, never switches.
        A mixed session is never offered it and ends; so does a session left with neither
        topics nor offers."""
        reason = (fallback_reason(problem) if problem is not None else None) or Fallback.exhausted
        topic = self.upcoming_topic
        if topic is None:
            notice = (
                "Your notes have nothing to ask from."
                if reason is Fallback.no_notes
                else "No new questions are left in your notes."
            )
            return Exhausted(notice, ended=True)
        self.topics.remove(topic)
        offer: Offer | None = None
        if _key(topic) in self.general_topics:
            self.general_topics.discard(_key(topic))
            notice = f'No more general-knowledge questions on "{topic}".'
        else:
            if reason is Fallback.exhausted:
                self.exhausted_topics.append(topic)
            offer = Offer(topic, reason)
            self.offers = [o for o in self.offers if _key(o.topic) != _key(topic)] + [offer]
            notice = gone_notice(offer)
        if self.topics:
            self.turn %= len(self.topics)
            return Exhausted(f"{notice} Continuing with {', '.join(self.topics)}.", False, offer)
        return Exhausted(notice, ended=not self.offers, offer=offer)

    def accept_general(self, topic: str | None) -> str | None:
        """Move an OFFERED topic to general knowledge (the newest offer for None). It goes back
        into the rotation as the very next topic. None: no such offer is open (a topic the notes
        still cover is never moved: the fallback is offered only where the notes ran out)."""
        offer = self._offer(topic)
        if offer is None:
            return None
        self.offers.remove(offer)
        self.general_topics.add(_key(offer.topic))
        index = self.turn % len(self.topics) if self.topics else 0
        self.topics.insert(index, offer.topic)
        self.turn = index
        return offer.topic

    def _offer(self, topic: str | None) -> Offer | None:
        if topic is None:
            return self.offers[-1] if self.offers else None
        return next((o for o in self.offers if _key(o.topic) == _key(topic)), None)

    def suspend(self, message: str) -> None:
        """A normal question left the mode: keep everything, remember the message (it may have
        been a hedged answer, see the module docstring)."""
        self.paused = True
        self.left_message = message.strip() or None

    def chatted(self) -> None:
        """Another Q&A message while suspended: the one that left is no longer the last word,
        so "grade it" no longer applies to it."""
        self.left_message = None

    def apply(self, step: Route) -> Step:
        """Apply one routed message; any route but `Leave` resumes a suspended session. A
        request for a question while WAITING on offers asks nothing (the notes have nothing
        left on the topics): it says so instead."""
        result = self._apply(step)
        if result.do is Do.ask and self.waiting:
            return Step(Do.nothing, notice=_join(result.notice, WAITING_NOTICE))
        return result

    def _apply(self, step: Route) -> Step:
        if isinstance(step, Leave):
            self.suspend(step.message)
            return Step(Do.chat, step.message)
        left, self.paused = self.left_message, False
        self.left_message = None
        match step:
            case Start(level=level) if level is not None and level is not self.level:
                self.skip()  # the session ends here (like stop): its summary counts it
                return Step(Do.restart, notice=restart_notice(level))
            case Start(topics=topics):
                self.switch(topics)
                unknown = unknown_level_notice(step, self.level)
                return Step(Do.ask, notice=_join(f"topic: {describe_topics(topics)}", unknown))
            case General(topic=topic):
                return self._general(topic)
            case Run(command=Command.stop):
                self.skip()
                return Step(Do.end)
            case Run(command=Command.repeat | Command.resume):
                return self._show()
            case Run(command=Command.harder | Command.easier as command):
                return self._difficulty(command)
            case Run(command=Command.next_topic):
                return Step(Do.ask, notice=self.next_topic())
            case Run():  # skip, next
                self.skip()
                return Step(Do.ask)
            case Answer(text=None):
                again = left or self.unsent
                return Step(Do.grade, again) if again and self.current else self._show()
            case Answer(text=str() as text):
                if self.current is None:
                    return Step(Do.nothing, notice=_NOTHING_PENDING)
                return Step(Do.grade, text)
        raise AssertionError(step)  # pragma: no cover - the match is exhaustive

    def _general(self, topic: str | None) -> Step:
        """Accept an offer: an explicit human choice. A pending notes question is skipped (the
        owner asked for the general one now)."""
        accepted = self.accept_general(topic)
        if accepted is None:
            return Step(Do.nothing, notice=NO_OFFER_NOTICE)
        self.skip()
        return Step(Do.ask, notice=f"topic: {accepted} · {GENERAL_LABEL} (not from your notes)")

    def _show(self) -> Step:
        if self.current is None:
            return Step(Do.nothing, notice=_NOTHING_PENDING)
        return Step(Do.show)

    def _difficulty(self, command: Command) -> Step:
        moved = self.harder() if command is Command.harder else self.easier()
        if not moved:
            edge = "hardest" if command is Command.harder else "easiest"
            return Step(Do.nothing, notice=f"Already at the {edge} difficulty for this level.")
        self.skip()
        within = level_name(self.level)
        return Step(
            Do.ask, notice=f"difficulty: {difficulty_label(self.difficulty)} (within {within})"
        )

    def _step(self, by: int) -> bool:
        index = _LEVELS.index(self.difficulty) + by
        if not 0 <= index < len(_LEVELS):
            return False
        self.difficulty = _LEVELS[index]
        return True

    @property
    def last(self) -> Asked | None:
        return self.history[-1] if self.history else None

    # ------------------------------------------------------------ summary
    def summary(self) -> Summary:
        """What was asked, how it went, and the weak topics (score <= 6), notes and general
        knowledge APART: a weak notes topic lists the pages to revisit (the grade's citations,
        its model answer's sources, else the question's); a weak general topic has no pages,
        only the advice to add notes on it."""
        graded = [(a, a.grade) for a in self.history if a.grade is not None]
        notes = [(a, g) for a, g in graded if not a.general]
        general = [(a, g) for a, g in graded if a.general]
        weak = tuple(
            WeakTopic(
                topic=a.question.topic,
                kind=str(a.question.kind),
                score=g.score,
                pages=_pages(g.citations or a.question.citations),
            )
            for a, g in notes
            if g.score <= WEAK_SCORE
        )
        weak_general = tuple(
            WeakGeneral(a.requested_topic or a.question.topic, str(a.question.kind), g.score)
            for a, g in general
            if g.score <= WEAK_SCORE
        )
        return Summary(
            asked=len(self.history),
            answered=len(graded),
            skipped=len(self.history) - len(graded),
            average=_mean([g.score for _, g in notes]),
            weak=weak,
            exhausted=tuple(self.exhausted_topics),
            level=self.level,
            topics=tuple(self.named_topics),
            notes_answered=len(notes),
            general_answered=len(general),
            general_average=_mean([g.score for _, g in general]),
            weak_general=weak_general,
        )


_NOTHING_PENDING = 'No question is pending: say "next" for one.'
WAITING_NOTICE = (
    "Your notes have no more questions on these topics: take a general-knowledge offer, "
    "or stop for the summary."
)
NO_OFFER_NOTICE = (
    "No general-knowledge offer is open: it is offered only for a topic your notes have "
    "nothing (left) on."
)


def _join(first: str | None, second: str | None) -> str | None:
    parts = [p for p in (first, second) if p]
    return " ".join(parts) or None


def _mean(scores: Sequence[int]) -> float | None:
    return sum(scores) / len(scores) if scores else None


def gone_notice(offer: Offer) -> str:
    """Why the notes gave nothing on the offer's topic (owner wording)."""
    if offer.reason is Fallback.no_notes:
        return f'Your notes have nothing on "{offer.topic}".'
    return f'All your "{offer.topic}" notes are asked.'


def offer_label(topic: str) -> str:
    """The one-click offer, per topic (owner wording, with the topic named so a topic list's
    offers are told apart)."""
    return f'Interview me on "{topic}" from general knowledge (not from your notes, not verified)'


def restart_notice(level: InterviewLevel) -> str:
    return (
        f"Level changed to {level_name(level)}: that starts a new interview (a session is asked "
        "and graded at one level)."
    )


def unknown_level_notice(start: Start, level: InterviewLevel) -> str | None:
    """The notice for a level (`--level` value or level-like clause) `parse_level` didn't know,
    or None."""
    if start.unknown_level is None:
        return None
    shown = start.unknown_level or "(nothing)"
    return f'Unknown level "{shown}": using {level_name(level)}. Levels: {level_choices()}.'


def _pages(citations: Sequence[Citation]) -> tuple[Page, ...]:
    """One entry per page (a page cited under several headings is one page to revisit)."""
    seen: dict[UUID, Page] = {}
    for c in citations:
        seen.setdefault(c.document_id, Page(c.title, str(c.url) if c.url else None))
    return tuple(seen.values())


def describe_topics(topics: Sequence[str]) -> str:
    return ", ".join(topics) if topics else "mixed, across all your notes"


def difficulty_label(difficulty: InterviewDifficulty | str) -> str:
    """Display name of a difficulty: a step WITHIN the level (Phase 3c; it used to stand for
    the Senior SDE-1 bar). One this client doesn't know (newer server) is its plain text."""
    return str(difficulty.value if isinstance(difficulty, InterviewDifficulty) else difficulty)


def question_line(question: InterviewQuestion) -> str:
    """Level, and the difficulty when it is not standard: "Staff", "Staff · harder"."""
    level = level_name(question.level)
    if question.difficulty == InterviewDifficulty.standard:
        return level
    return f"{level} · {difficulty_label(question.difficulty)}"


def summary_headline(summary: Summary) -> str:
    """One plain line: "Staff · Kafka · avg 6.5". General-knowledge scores get their own,
    labelled average, never folded into the notes one."""
    topics = ", ".join(summary.topics) if summary.topics else "mixed topics"
    parts = [level_name(summary.level), topics]
    both = summary.average is not None and summary.general_average is not None
    if summary.average is not None:
        parts.append(f"avg {summary.average:.1f}" + (" from your notes" if both else ""))
    if summary.general_average is not None:
        parts.append(f"general knowledge avg {summary.general_average:.1f} (not verified)")
    if summary.average is None and summary.general_average is None:
        parts.append("none graded")
    return " · ".join(parts)
