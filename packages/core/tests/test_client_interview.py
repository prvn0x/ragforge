"""`ragforge_client.interview`: the deterministic start/command/answer routing (pinned both ways)
and the client-held session (topic alternation, kind rotation, no repeats, difficulty, exhaustion,
suspend/resume, summary)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from ragforge_client.interview import (
    KIND_ROTATION,
    MAX_EXCLUDED,
    MAX_EXCLUDED_QUESTION_CHARS,
    MAX_EXCLUDED_QUESTIONS,
    Answer,
    Command,
    Do,
    Fallback,
    General,
    InterviewSession,
    Leave,
    NoQuestion,
    Offer,
    Route,
    Run,
    Start,
    Step,
    WeakGeneral,
    describe_topics,
    detect_command,
    detect_start,
    difficulty_label,
    fallback_reason,
    is_exhausted,
    is_general,
    is_normal_question,
    level_name,
    parse_level,
    question_line,
    route,
    route_suspended,
    split_topics,
    summary_headline,
)
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    Citation,
    InterviewDifficulty,
    InterviewGrade,
    InterviewKind,
    InterviewLevel,
    InterviewQuestion,
    InterviewSource,
    Problem,
)

COLLECTIONS = [uuid4()]
LQ, RQ = "\N{LEFT DOUBLE QUOTATION MARK}", "\N{RIGHT DOUBLE QUOTATION MARK}"


def _citation(title: str, document_id: UUID | None = None) -> Citation:
    return Citation(
        anchor="block-1",
        document_id=document_id or uuid4(),
        title=title,
        heading_path=["ISR"],
        url=f"https://www.notion.so/{title.replace(' ', '-')}",
        snippet="...",
        label=1,
    )


def question(
    kind: InterviewKind | str = InterviewKind.concept,
    topic: str = "Kafka ISR",
    chunks: int = 2,
    citations: list[Citation] | None = None,
    difficulty: InterviewDifficulty | str = InterviewDifficulty.standard,
    token: str | None = None,
) -> InterviewQuestion:
    return InterviewQuestion(
        question=f"Why does {topic} matter\\?",
        kind=kind,
        topic=topic,
        difficulty=difficulty,
        chunk_ids=[uuid4() for _ in range(chunks)],
        expected_points=["the leader waits for the ISR", "min.insync.replicas"],
        citations=citations if citations is not None else [_citation("Kafka basics")],
        question_token=token,
    )


def grade(score: int, citations: list[Citation] | None = None) -> InterviewGrade:
    return InterviewGrade(
        score=score,
        summary="ok",
        right=[],
        missed=[],
        misconceptions=[],
        model_answer="The ISR \\[S1\\].",
        citations=citations if citations is not None else [],
        verification="verified",
    )


# ------------------------------------------------------------------ start phrases
@pytest.mark.parametrize(
    ("message", "topics"),
    [
        ("interview me", ()),
        ("Interview me.", ()),
        ("interview me on Kafka", ("Kafka",)),
        ("Interview me about Kafka consumer groups!", ("Kafka consumer groups",)),
        ("interview me on Kafka?", ("Kafka",)),
        (
            "interview me on Kafka\N{FULLWIDTH QUESTION MARK}",
            ("Kafka",),
        ),  # full-width question mark
        ("quiz me", ()),
        ("Quiz me on Redis please", ("Redis",)),
        ("ask me interview questions", ()),
        ("ask me some questions", ()),
        ("ask me some interview questions on system design?", ("system design",)),
        ("Ask me 5 questions about Postgres MVCC", ("Postgres MVCC",)),
        ("can you ask me a few interview questions about Raft?", ("Raft",)),
        ("please, quiz me on redis", ("redis",)),
        ("/interview", ()),
        ("/interview kafka streams", ("kafka streams",)),
        ("/INTERVIEW Kafka", ("Kafka",)),
        ("interview me on anything", ()),
        ("quiz me on all topics", ()),
        ("  interview   me   on   Kafka  ", ("Kafka",)),
        # topic LISTS (design review): the session alternates them per question
        ("interview me on Kafka, Redis", ("Kafka", "Redis")),
        ("interview me on Kafka and Redis", ("Kafka", "Redis")),
        ("/interview kafka, redis", ("kafka", "redis")),
        ("quiz me on Kafka, Redis, and Postgres", ("Kafka", "Redis", "Postgres")),
        ("quiz me on Kafka & Redis", ("Kafka", "Redis")),
        ("quiz me on Kafka, kafka, Redis", ("Kafka", "Redis")),  # de-duplicated
        ('interview me on "publish and subscribe".', ("publish and subscribe",)),
        (f"interview me on {LQ}Kafka and Redis{RQ}", ("Kafka and Redis",)),
        # topic before "interview questions" (how the owner phrases it)
        ("ask me Kafka interview questions", ("Kafka",)),
        ("ask me some Java interview questions", ("Java",)),
        ("give me 5 Kafka and Redis interview questions", ("Kafka", "Redis")),
        ("ask me java questions interview", ("java",)),
        ("ask me java questions for an interview", ("java",)),
        ("give me interview questions on Redis", ("Redis",)),
        ("give me the top 10 Kafka interview questions", ("Kafka",)),  # filler stripped
        ("ask me some commonly asked Redis interview questions", ("Redis",)),
    ],
)
def test_start_phrases_and_their_topics(message: str, topics: tuple[str, ...]) -> None:
    assert detect_start(message) == Start(topics=topics)


@pytest.mark.parametrize(
    "message",
    [
        "what is an interview?",
        "how do I prepare for a Kafka interview",
        "questions about Kafka ISR?",
        "act as a principal engineer interviewing me",  # owner decision: never inferred
        "what are common Kafka interview questions?",
        "ask me",
        "interview",
        "/interviewer",
        "interview me later, first explain ISR",
        "I had an interview me and my friend",
        "quiz",
        "the interviewer asked me questions on Kafka",
        "don't interview me",
        "what are good Kafka interview questions?",
        "Kafka interview questions",  # a search, not an explicit request to be interviewed
        "ask me Kafka questions",  # no "interview": could be anything
    ],
)
def test_ordinary_messages_never_start_the_mode(message: str) -> None:
    assert detect_start(message) is None


def test_topics_are_clipped_to_the_request_limit_and_the_list_is_bounded() -> None:
    start = detect_start("/interview " + "k" * 300)
    assert start is not None
    assert [len(t) for t in start.topics] == [200]
    many = split_topics(", ".join(f"t{i}" for i in range(20)))
    assert len(many) == 8


# ------------------------------------------------------------------ commands and routing
@pytest.mark.parametrize(
    ("message", "command"),
    [
        ("repeat", Command.repeat),
        ("where's the question?", Command.repeat),
        ("Where\N{RIGHT SINGLE QUOTATION MARK}s the question?", Command.repeat),
        ("where's the question\N{FULLWIDTH QUESTION MARK}", Command.repeat),
        ("what was the question", Command.repeat),
        ("again", Command.repeat),
        ("Skip", Command.skip),
        ("/skip", Command.skip),
        ("next", Command.next),
        ("next question", Command.next),
        ("harder", Command.harder),
        ("make it harder!", Command.harder),
        ("easier", Command.easier),
        ("next topic", Command.next_topic),
        ("change topic", Command.next_topic),
        ("stop", Command.stop),
        ("End interview.", Command.stop),
        ("exit", Command.stop),
        ("/stop", Command.stop),
        ("/resume", Command.resume),
        # the polite frame (round-1 review): misread as answers these would be graded ~0-3,
        # step the level down and add a false weak topic
        ("Next question please", Command.next),
        ("next one", Command.next),
        ("again please", Command.repeat),
        ("can you repeat that?", Command.repeat),
        ("Stop it", Command.stop),
        ("ok stop", Command.stop),
        ("OK, stop.", Command.stop),
        ("please stop", Command.stop),
        ("stop the quiz", Command.stop),
        ("okay, make it harder please", Command.harder),
        ("now skip, please", Command.skip),
    ],
)
def test_session_commands(message: str, command: Command) -> None:
    assert detect_command(message) == command
    assert route(message, pending=True) == Run(command)


@pytest.mark.parametrize(
    "message",
    [
        "the leader stops accepting writes again",  # contains "again" and "stop", still prose
        "skip the ISR check and the data is lost",
        "it gets harder to rebalance",
        # the polite frame only wraps a WHOLE command: these are still answers
        "ok so the consumer stops polling",
        "please note the next question of durability is acks",
        "stop the producer, then restart the broker",
        "again, the leader waits for the ISR please",
    ],
)
def test_a_command_word_inside_an_answer_is_still_the_answer(message: str) -> None:
    assert detect_command(message) is None
    assert route(message, pending=True) == Answer(message)


HEDGED = "Is it because the ISR shrinks and the leader can't commit?"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # rule 4: a normal question leaves (suspends) the mode (owner decision) ...
        ("What is ISR in Kafka?", Leave("What is ISR in Kafka?")),
        ("how does Raft elect a leader?", Leave("how does Raft elect a leader?")),
        ("tell me about consumer lag?", Leave("tell me about consumer lag?")),
        ("explain MVCC?", Leave("explain MVCC?")),
        (
            "Why does Kafka need ZooKeeper\N{FULLWIDTH QUESTION MARK}",
            Leave("Why does Kafka need ZooKeeper\N{FULLWIDTH QUESTION MARK}"),
        ),
        # ... a HEDGED answer looks the same and suspends too: "grade it" recovers it (below)
        (HEDGED, Leave(HEDGED)),
        # rule 5: everything else is the answer
        ("The leader waits for every ISR member", Answer("The leader waits for every ISR member")),
        ("ISR shrank?", Answer("ISR shrank?")),  # no interrogative opener
        ("what the leader does is wait", Answer("what the leader does is wait")),  # no "?"
        ("Because of acks=all? Not sure", Answer("Because of acks=all? Not sure")),
        ("  I don't know  ", Answer("I don't know")),
        # rule 2: /answer forces an answer, even one phrased as a question
        (f"/answer {HEDGED}", Answer(HEDGED)),
        ("/ANSWER  acks=all", Answer("acks=all")),
        ("/answer", Answer(None)),
        # rule 3: a start phrase restarts with the new topic(s)
        ("interview me on Redis", Start(("Redis",))),
        ("quiz me on Kafka and Redis", Start(("Kafka", "Redis"))),
        # rule 1: commands win over the question rule
        ("where's the question?", Run(Command.repeat)),
        ("what was the question?", Run(Command.repeat)),
    ],
)
def test_routing_while_a_question_is_pending(message: str, expected: Route) -> None:
    assert route(message, pending=True) == expected


def test_with_nothing_pending_free_text_goes_to_q_and_a() -> None:
    assert route("The leader waits", pending=False) == Leave("The leader waits")
    assert route("skip", pending=False) == Run(Command.skip)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("/answer", Answer(None)),
        ("/answer it is acks=all", Answer("it is acks=all")),
        ("/resume", Run(Command.resume)),
        ("resume interview", Run(Command.resume)),
        ("/stop", Run(Command.stop)),
        ("interview me on Redis", Start(("Redis",))),
        # ordinary words are ordinary Q&A while suspended
        ("stop", None),
        ("again", None),
        ("resume", None),
        ("What is ISR?", None),
        ("The leader waits", None),
    ],
)
def test_routing_while_suspended(message: str, expected: Route | None) -> None:
    assert route_suspended(message) == expected


def test_normal_question_rule_needs_both_the_mark_and_the_opener() -> None:
    assert is_normal_question("Why?")
    assert is_normal_question("why\N{FULLWIDTH QUESTION MARK}")
    assert not is_normal_question("Why")
    assert not is_normal_question("?")
    assert not is_normal_question("Telling me?")


def test_exhaustion_is_decided_by_problem_type_never_by_status() -> None:
    assert is_exhausted(Problem(type=INTERVIEW_EXHAUSTED, title="x", status=409))
    assert not is_exhausted(Problem(title="Conflict", status=409))
    assert not is_exhausted(Problem(title="Not Found", status=404))


# ------------------------------------------------------------------ session: requests
def test_requests_rotate_through_all_four_kinds_and_never_repeat_chunks() -> None:
    session = InterviewSession(topics=["Kafka"])
    seen_kinds: list[InterviewKind] = []
    asked: list[UUID] = []
    for _ in range(5):
        request = session.question_request(COLLECTIONS)
        assert request.topic == "Kafka"
        assert set(request.kinds) == set(InterviewKind)  # every kind allowed ...
        assert request.exclude_chunk_ids == asked  # ... nothing already asked
        q = question(kind=request.kinds[0])  # a server that honours the preference
        seen_kinds.append(request.kinds[0])
        session.asked(q)
        asked += q.chunk_ids
    assert seen_kinds == [*KIND_ROTATION, KIND_ROTATION[0]]


def test_rotation_follows_the_kind_the_server_actually_asked() -> None:
    session = InterviewSession()
    session.asked(question(kind=InterviewKind.design))
    assert session.question_request(COLLECTIONS).kinds[0] is InterviewKind.code


def test_an_unknown_kind_from_a_newer_server_still_advances_and_still_grades() -> None:
    session = InterviewSession()
    session.asked(question(kind="puzzle", difficulty="expert"))
    assert session.question_request(COLLECTIONS).kinds[0] is InterviewKind.scenario
    request = session.grade_request(COLLECTIONS, "an answer")
    assert request.kind is InterviewKind.concept  # the kind requested for it
    assert request.difficulty is InterviewDifficulty.standard


def test_a_topic_list_alternates_one_topic_per_request() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis")))
    topics = []
    for _ in range(4):
        topics.append(session.question_request(COLLECTIONS).topic)
        session.asked(question())
    assert topics == ["Kafka", "Redis", "Kafka", "Redis"]


def test_mixed_topics_send_no_topic() -> None:
    assert InterviewSession.begin(Start()).question_request(COLLECTIONS).topic is None


def test_excluded_chunks_are_deduplicated_and_capped_dropping_the_oldest() -> None:
    session = InterviewSession()
    shared = uuid4()
    first = question(chunks=1)
    session.asked(first.model_copy(update={"chunk_ids": [shared, *first.chunk_ids]}))
    for _ in range(MAX_EXCLUDED):
        q = question(chunks=1)
        session.asked(q.model_copy(update={"chunk_ids": [shared, *q.chunk_ids]}))
    excluded = session.question_request(COLLECTIONS).exclude_chunk_ids
    assert len(excluded) == MAX_EXCLUDED
    assert excluded[-1] == session.asked_chunk_ids[-1]
    assert first.chunk_ids[0] not in excluded  # the oldest went
    assert excluded.count(shared) == 1  # never listed twice


def test_grade_request_echoes_the_question_token_and_validates_the_answer() -> None:
    session = InterviewSession()
    with pytest.raises(NoQuestion):
        session.grade_request(COLLECTIONS, "x")
    q = question(token="signed.token")
    session.asked(q)
    request = session.grade_request(COLLECTIONS, "  The ISR  ")
    assert request.answer == "The ISR"
    assert (request.chunk_ids, request.expected_points) == (q.chunk_ids, q.expected_points)
    assert request.question_token == "signed.token"
    with pytest.raises(ValueError, match="empty"):
        session.grade_request(COLLECTIONS, "   ")
    with pytest.raises(ValueError, match="8000"):
        session.grade_request(COLLECTIONS, "x" * 8001)


def test_the_grade_uses_the_collections_the_question_was_asked_from() -> None:
    """The UI selection may change while the user answers; the token binds the question's
    collections, so grading with the new selection would always fail (final review P2)."""
    session = InterviewSession()
    session.question_request(COLLECTIONS)
    session.asked(question())
    other = [uuid4()]
    assert session.grade_request(other, "an answer").collection_ids == list(COLLECTIONS)


def test_no_token_from_an_older_server_sends_none() -> None:
    session = InterviewSession()
    session.asked(question())
    assert session.grade_request(COLLECTIONS, "a").question_token is None


def test_an_over_long_question_is_clipped_so_the_answer_stays_gradable() -> None:
    session = InterviewSession()
    long = question().model_copy(
        update={
            "question": "q" * 2500,
            "expected_points": ["p" * 600] * 25,
            "chunk_ids": [uuid4() for _ in range(12)],
        }
    )
    session.asked(long)
    request = session.grade_request(COLLECTIONS, "answer")
    assert len(request.question) == 2000
    assert len(request.expected_points) == 20
    assert all(len(p) == 500 for p in request.expected_points)
    assert len(request.chunk_ids) == 10


# ------------------------------------------------------------------ session: transitions
def test_difficulty_steps_and_stops_at_the_edges() -> None:
    session = InterviewSession()
    levels = []
    assert session.harder()
    levels.append(session.difficulty)
    assert not session.harder()
    assert session.easier()
    assert session.easier()
    levels.append(session.difficulty)
    assert not session.easier()
    assert levels == [InterviewDifficulty.harder, InterviewDifficulty.easier]
    assert session.question_request(COLLECTIONS).difficulty is InterviewDifficulty.easier


@pytest.mark.parametrize(
    ("score", "after"),
    [
        (10, InterviewDifficulty.harder),
        (9, InterviewDifficulty.harder),
        (6, InterviewDifficulty.standard),
        (4, InterviewDifficulty.standard),
        (3, InterviewDifficulty.easier),
        (0, InterviewDifficulty.easier),
    ],
)
def test_the_next_question_adapts_to_the_score(score: int, after: InterviewDifficulty) -> None:
    session = InterviewSession()
    session.asked(question())
    session.graded("answer", grade(score))
    assert session.difficulty is after
    assert session.current is None


def test_graded_without_a_question_is_refused() -> None:
    with pytest.raises(NoQuestion):
        InterviewSession().graded("a", grade(5))


def test_a_question_replaced_before_grading_counts_as_skipped() -> None:
    session = InterviewSession(topics=["Kafka"])
    session.asked(question())
    session.asked(question())  # e.g. after "harder"
    assert [a.grade for a in session.history] == [None]
    assert session.skip()
    assert not session.skip()


def test_next_topic_clears_a_fixed_topic_to_mixed() -> None:
    session = InterviewSession(topics=["Kafka"])
    session.asked(question())
    assert "mixed" in session.next_topic()
    assert session.upcoming_topic is None
    assert session.current is None
    assert len(session.history) == 1
    session.next_topic()  # already mixed: just moves on
    assert session.upcoming_topic is None


def test_next_topic_with_a_list_moves_to_the_next_one() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis", "Raft")))
    session.asked(question(topic="Kafka"))  # the pending question is about Kafka
    assert session.next_topic() == "topic: Redis"
    assert session.question_request(COLLECTIONS).topic == "Redis"
    assert session.topics == ["Kafka", "Redis", "Raft"]  # the list itself is kept


def test_a_start_phrase_mid_session_switches_topics_but_keeps_history_and_no_repeat() -> None:
    session = InterviewSession.begin(Start(("Kafka",)))
    q = question()
    session.asked(q)
    step = session.apply(Start(("Redis", "Raft")))
    assert step.do is Do.ask
    request = session.question_request(COLLECTIONS)
    assert request.topic == "Redis"
    assert set(q.chunk_ids) <= set(request.exclude_chunk_ids)
    assert len(session.history) == 1


def test_an_exhausted_topic_is_dropped_offered_and_the_rest_continue() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis", "Raft")))
    session.asked(question())  # Kafka asked; Redis is next
    outcome = session.exhausted()
    assert not outcome.ended
    assert "Redis" in outcome.notice
    assert outcome.offer == Offer("Redis", Fallback.exhausted)
    assert session.topics == ["Kafka", "Raft"]
    assert session.question_request(COLLECTIONS).topic == "Raft"
    assert session.exhausted().ended is False  # Raft gone, Kafka left
    assert session.question_request(COLLECTIONS).topic == "Kafka"
    final = session.exhausted()
    # every topic ran out, but the offers are open: the session WAITS (never switches itself)
    assert not final.ended
    assert session.waiting
    assert [o.topic for o in session.offers] == ["Redis", "Raft", "Kafka"]
    assert session.summary().exhausted == ("Redis", "Raft", "Kafka")


def test_mixed_exhaustion_ends_the_interview() -> None:
    assert InterviewSession().exhausted().ended


# ------------------------------------------------------------------ session: apply
def test_apply_maps_each_route_to_what_the_client_must_do() -> None:
    session = InterviewSession.begin(Start(("Kafka",)))
    assert session.apply(Run(Command.repeat)).do is Do.nothing  # nothing pending yet
    session.asked(question())
    assert session.apply(Run(Command.repeat)).do is Do.show
    assert session.apply(Answer("acks=all")) == session.apply(Answer("acks=all"))
    assert session.apply(Answer("acks=all")).do is Do.grade
    harder = session.apply(Run(Command.harder))
    assert harder.do is Do.ask
    assert harder.notice == "difficulty: harder (within Senior SDE-1)"
    assert session.current is None  # the pending question was skipped for a harder one
    assert session.apply(Run(Command.harder)).do is Do.nothing  # already at the top
    session.asked(question())
    assert session.apply(Run(Command.skip)).do is Do.ask
    assert session.apply(Answer("late")).do is Do.nothing  # nothing to grade
    assert session.apply(Run(Command.stop)).do is Do.end


def test_a_normal_question_suspends_and_grade_it_recovers_the_hedged_answer() -> None:
    session = InterviewSession.begin(Start(("Kafka",)))
    q = question()
    session.asked(q)
    step = session.apply(route(HEDGED, pending=True))
    assert (step.do, step.text) == (Do.chat, HEDGED)
    assert session.paused
    assert session.current is q
    # "↩ That was my answer — grade it" / a bare /answer grades the message that left
    back = route_suspended("/answer")
    assert back is not None
    graded = session.apply(back)
    assert (graded.do, graded.text) == (Do.grade, HEDGED)
    assert not session.paused


def test_another_q_and_a_message_while_suspended_forgets_the_left_message() -> None:
    session = InterviewSession()
    session.asked(question())
    session.apply(Leave("What is ISR?"))
    session.chatted()
    step = session.apply(Answer(None))
    assert step.do is Do.show  # nothing to grade any more: show the question again
    assert not session.paused


def test_a_failed_grade_keeps_the_answer_for_a_bare_answer_until_skipped() -> None:
    session = InterviewSession()
    session.asked(question())
    session.apply(Leave(HEDGED))
    assert session.apply(Answer(None)) == Step(Do.grade, HEDGED)
    assert session.left_message is None  # cleared before grading ...
    session.grade_failed(HEDGED)  # ... so the failure keeps it
    assert session.apply(Answer(None)) == Step(Do.grade, HEDGED)
    session.graded(HEDGED, grade(7))
    assert session.unsent is None
    session.asked(question())
    session.grade_failed("acks=all")
    session.skip()
    assert session.unsent is None  # it belonged to the skipped question


def test_resume_shows_the_pending_question_again() -> None:
    session = InterviewSession()
    session.asked(question())
    session.apply(Leave("What is ISR?"))
    assert session.apply(Run(Command.resume)).do is Do.show
    assert not session.paused
    assert session.left_message is None


# ------------------------------------------------------------------ summary
def test_summary_lists_weak_topics_with_their_pages_once_each() -> None:
    session = InterviewSession()
    page = uuid4()
    kafka = [_citation("Kafka basics", page), _citation("Kafka basics", page)]
    session.asked(question(topic="Kafka ISR", kind=InterviewKind.concept))
    session.graded("a", grade(4, citations=kafka))
    session.asked(question(topic="Redis", kind=InterviewKind.scenario))
    session.graded("b", grade(9))
    session.asked(question(topic="Raft", citations=[_citation("Raft notes")]))
    session.graded("c", grade(6, citations=[]))  # no grade citations: the question's pages
    session.asked(question(topic="MVCC"))
    session.skip()
    summary = session.summary()
    assert (summary.asked, summary.answered, summary.skipped) == (4, 3, 1)
    assert summary.average == pytest.approx(19 / 3)
    assert [(w.topic, w.kind, w.score) for w in summary.weak] == [
        ("Kafka ISR", "concept", 4),
        ("Raft", "concept", 6),
    ]
    assert [p.title for p in summary.weak[0].pages] == ["Kafka basics"]
    assert summary.weak[0].pages[0].url == "https://www.notion.so/Kafka-basics"
    assert [p.title for p in summary.weak[1].pages] == ["Raft notes"]


def test_an_empty_session_summary_has_no_average() -> None:
    summary = InterviewSession().summary()
    assert (summary.asked, summary.average, summary.weak) == (0, None, ())


def test_labels() -> None:
    assert difficulty_label(InterviewDifficulty.standard) == "standard"  # within the level
    assert difficulty_label(InterviewDifficulty.harder) == "harder"
    assert difficulty_label("expert") == "expert"  # unknown level: plain text
    assert describe_topics([]) == "mixed, across all your notes"
    assert describe_topics(["Kafka", "Redis"]) == "Kafka, Redis"


@pytest.mark.parametrize(
    ("message", "topics"),
    [
        ("ask me common interview questions", ()),  # only filler: mixed, never topic "common"
        ("give me the top 10 interview questions", ()),
        ("ask me hard interview questions", ()),
        ("ask me about Kafka interview questions", ("Kafka",)),
        ("give me best practices interview questions", ("best practices",)),
    ],
)
def test_filler_alone_means_mixed_topics(message: str, topics: tuple[str, ...]) -> None:
    assert detect_start(message) == Start(topics=topics)


@pytest.mark.parametrize(
    "message",
    [
        "give me the answers to Kafka interview questions",
        "give me an overview of Kafka interview questions",
        "give me a list of Redis interview questions",
    ],
)
def test_asking_about_interview_questions_is_not_an_interview(message: str) -> None:
    assert detect_start(message) is None


def test_a_failed_fetch_does_not_move_the_pending_questions_collections() -> None:
    session = InterviewSession()
    session.question_request(COLLECTIONS)
    session.asked(question())
    session.question_request([uuid4()])  # the next fetch fails (429, timeout): no `asked`
    assert session.grade_request([uuid4()], "an answer").collection_ids == list(COLLECTIONS)


# ================================================================== Phase 3c (IV3-UI)
def _general_question(
    topic: str = "Redis",
    text: str = "How does Redis evict keys\\?",
    token: str = "gtok",  # noqa: S107
) -> InterviewQuestion:
    return InterviewQuestion(
        question=text,
        kind=InterviewKind.concept,
        topic=topic,
        difficulty=InterviewDifficulty.standard,
        chunk_ids=[],
        expected_points=["LRU / LFU policies", "maxmemory"],
        citations=[],
        question_token=token,
        level=InterviewLevel.staff,
        source=InterviewSource.general,
    )


def _general_grade(score: int) -> InterviewGrade:
    return grade(score, citations=[]).model_copy(
        update={"source": InterviewSource.general, "verification": "unverified"}
    )


NO_NOTES = Problem(type=INTERVIEW_NO_NOTES, title="No notes", status=409)
EXHAUSTED = Problem(type=INTERVIEW_EXHAUSTED, title="Exhausted", status=409)


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Staff", InterviewLevel.staff),
        ("staff engineer", InterviewLevel.staff),
        ("staff level", InterviewLevel.staff),
        ("SDE1", InterviewLevel.sde_1),
        ("sde-1", InterviewLevel.sde_1),
        ("SDE I", InterviewLevel.sde_1),
        ("SDE2", InterviewLevel.sde_2),
        ("sde-2", InterviewLevel.sde_2),
        ("sde ii", InterviewLevel.sde_2),
        ("software engineer 2", InterviewLevel.sde_2),
        ("senior", InterviewLevel.senior_sde_1),
        ("Senior SDE-1", InterviewLevel.senior_sde_1),
        ("Sr. SDE 1", InterviewLevel.senior_sde_1),
        ("senior sde 2", InterviewLevel.senior_sde_2),
        ("senior sde ii", InterviewLevel.senior_sde_2),
        ("a senior SDE 2", InterviewLevel.senior_sde_2),
        ("senior_sde_2", InterviewLevel.senior_sde_2),  # the enum value
        ("sde_1", InterviewLevel.sde_1),
        # not levels: no guessing
        ("sde", None),
        ("2", None),
        ("principal", None),
        ("junior", None),
        ("staff 2", None),
        ("rest", None),
        ("", None),
    ],
)
def test_level_table(text: str, level: InterviewLevel | None) -> None:
    assert parse_level(text) is level


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("interview me on Kafka at staff level", Start(("Kafka",), InterviewLevel.staff)),
        ("interview me on Kafka at the staff level", Start(("Kafka",), InterviewLevel.staff)),
        ("quiz me on Kafka as a senior SDE 2", Start(("Kafka",), InterviewLevel.senior_sde_2)),
        ("interview me on Redis as an SDE2", Start(("Redis",), InterviewLevel.sde_2)),
        ("/interview kafka --level staff", Start(("kafka",), InterviewLevel.staff)),
        ("/interview kafka --level=sde-2", Start(("kafka",), InterviewLevel.sde_2)),
        ("/interview --level senior sde ii", Start((), InterviewLevel.senior_sde_2)),
        ("interview me at staff level", Start((), InterviewLevel.staff)),
        ("interview me as a staff engineer on Kafka", Start(("Kafka",), InterviewLevel.staff)),
        ("ask me Kafka interview questions for SDE 2", Start(("Kafka",), InterviewLevel.sde_2)),
        # a topic that merely contains at/as/for stays whole; the leftmost LEVEL clause wins
        ("interview me on data at rest", Start(("data at rest",))),
        (
            "interview me on data at rest at staff level",
            Start(("data at rest",), InterviewLevel.staff),
        ),
        ("quiz me about Kafka as a log", Start(("Kafka as a log",))),
        # an unknown --level is reported, never silently a topic
        ("/interview kafka --level wizard", Start(("kafka",), None, "wizard")),
        ("/interview kafka --level", Start(("kafka",), None, "")),
        # the polite trailing "please" never hides the level clause (review round 1)
        ("interview me on Kafka at staff level please", Start(("Kafka",), InterviewLevel.staff)),
        (
            "interview me on Kafka at staff level, please.",
            Start(("Kafka",), InterviewLevel.staff),
        ),
        ("/interview please", Start(())),
        # a level the table doesn't know is reported and kept out of the topic (review round 1)
        ("interview me on Kafka at principal level", Start(("Kafka",), None, "principal level")),
        ("interview me on Kafka at L5", Start(("Kafka",), None, "L5")),
        ("interview me on Kafka at sde", Start(("Kafka",), None, "sde")),
        (
            "interview me on data at rest at principal level, please",
            Start(("data at rest",), None, "principal level"),
        ),
        # ... but a topic that merely ends in "level" or holds a level-ish word stays whole
        (
            "interview me on caching at the application level",
            Start(("caching at the application level",)),
        ),
        (
            "interview me on leader election for e2e tests",
            Start(("leader election for e2e tests",)),
        ),
        ("interview me on Kafka internals", Start(("Kafka internals",))),
    ],
)
def test_start_phrases_with_a_level(message: str, expected: Start) -> None:
    assert detect_start(message) == expected


@pytest.mark.parametrize(
    ("message", "topics"),
    [
        ("INTERVIEW MODE", ()),  # the live failure: it went to Q&A and got a refusal
        ("interview mode", ()),
        ("interview mode on", ()),
        ("Interview mode on Kafka", ("Kafka",)),
        ("interview mode about Kafka, Redis", ("Kafka", "Redis")),
        ("start interview", ()),
        ("start interview on Kafka", ("Kafka",)),
        ("start an interview", ()),
        ("Start an interview on Redis.", ("Redis",)),
        ("start the interview about Raft", ("Raft",)),
        ("start interview mode on Kafka", ("Kafka",)),
        ("please start an interview on Kafka", ("Kafka",)),
    ],
)
def test_new_start_phrases(message: str, topics: tuple[str, ...]) -> None:
    assert detect_start(message) == Start(topics=topics)


@pytest.mark.parametrize(
    "message",
    [
        "what is interview mode?",
        "interview mode is broken",
        "how does interview mode work",
        "start",
        "start the interviewer",
        "start interview prep notes",
        "interview me as a friend",  # "as a <not a level>": no start (as before 3c)
        "interview me at work on Kafka",
        "what is kafka at staff level?",
        "explain Kafka as a senior SDE 2 would",
    ],
)
def test_new_negatives_never_start(message: str) -> None:
    assert detect_start(message) is None


def test_the_level_is_sent_on_both_requests_and_the_question_level_is_echoed() -> None:
    session = InterviewSession.begin(Start(("Kafka",), InterviewLevel.staff))
    request = session.question_request(COLLECTIONS)
    assert (request.level, request.source) == (InterviewLevel.staff, InterviewSource.notes)
    assert request.exclude_questions == []
    q = question(token="tok").model_copy(update={"level": InterviewLevel.sde_2})
    session.asked(q)
    graded = session.grade_request(COLLECTIONS, "acks=all")
    # the QUESTION's level (the token binds it), never the session's re-applied
    assert (graded.level, graded.source, graded.question_token) == (
        InterviewLevel.sde_2,
        InterviewSource.notes,
        "tok",
    )


def test_a_new_session_defaults_to_senior_sde_1_and_an_unknown_level_falls_back() -> None:
    session = InterviewSession.begin(Start(("Kafka",)))
    assert session.level is InterviewLevel.senior_sde_1
    session.asked(question().model_copy(update={"level": "principal"}))  # newer server
    assert session.grade_request(COLLECTIONS, "a").level is InterviewLevel.senior_sde_1


def test_the_level_is_fixed_per_session_another_level_restarts() -> None:
    session = InterviewSession.begin(Start(("Kafka",), InterviewLevel.sde_1))
    session.asked(question())
    step = session.apply(Start(("Kafka",), InterviewLevel.staff))
    assert step.do is Do.restart
    assert step.notice is not None
    assert "starts a new interview" in step.notice
    assert "Staff" in step.notice
    # this session ends (the client shows its summary): the pending question counts as skipped
    assert session.current is None
    assert session.summary().skipped == 1
    assert session.level is InterviewLevel.sde_1
    # the same level, or none named, is a topic switch in the SAME session
    assert session.apply(Start(("Redis",), InterviewLevel.sde_1)).do is Do.ask
    assert session.apply(Start(("Raft",))).do is Do.ask
    assert session.level is InterviewLevel.sde_1
    assert session.topics == ["Raft"]


def test_an_unknown_level_mid_session_says_so_and_keeps_the_level() -> None:
    session = InterviewSession.begin(Start(("Kafka",), InterviewLevel.staff))
    step = session.apply(Start(("Redis",), None, "wizard"))
    assert step.do is Do.ask
    assert step.notice is not None
    assert 'Unknown level "wizard": using Staff' in step.notice
    assert session.level is InterviewLevel.staff


def test_difficulty_is_a_step_within_the_level() -> None:
    session = InterviewSession.begin(Start((), InterviewLevel.staff))
    step = session.apply(Run(Command.harder))
    assert step.notice == "difficulty: harder (within Staff)"
    request = session.question_request(COLLECTIONS)
    assert (request.level, request.difficulty) == (InterviewLevel.staff, InterviewDifficulty.harder)


# ------------------------------------------------------------------ fallback offers
@pytest.mark.parametrize(
    ("problem", "reason", "notice"),
    [
        (NO_NOTES, Fallback.no_notes, 'Your notes have nothing on "Redis".'),
        (EXHAUSTED, Fallback.exhausted, 'All your "Redis" notes are asked.'),
    ],
)
def test_both_409_types_open_an_offer_and_never_switch(
    problem: Problem, reason: Fallback, notice: str
) -> None:
    session = InterviewSession.begin(Start(("Redis",)))
    outcome = session.exhausted(problem)
    assert outcome.notice == notice
    assert outcome.offer == Offer("Redis", reason)
    assert not outcome.ended
    assert session.waiting
    assert session.general_topics == set()  # NOT switched: only a human choice does that
    assert session.apply(Run(Command.next)).do is Do.nothing  # nothing to ask from the notes
    assert fallback_reason(problem) is reason


def test_no_notes_is_not_counted_as_fully_covered() -> None:
    session = InterviewSession.begin(Start(("Redis", "Kafka")))
    session.exhausted(NO_NOTES)
    assert session.summary().exhausted == ()


def test_other_problems_are_not_fallbacks() -> None:
    assert fallback_reason(Problem(title="Conflict", status=409)) is None
    assert fallback_reason(Problem(title="Not Found", status=404)) is None


def test_a_mixed_session_is_never_offered_the_fallback() -> None:
    for problem in (NO_NOTES, EXHAUSTED):
        session = InterviewSession()
        outcome = session.exhausted(problem)
        assert outcome.ended
        assert outcome.offer is None
        assert session.offers == []


def test_accepting_an_offer_asks_general_next_with_its_earlier_general_questions() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis"), InterviewLevel.staff))
    session.asked(question(topic="Kafka ISR"))  # Kafka; Redis next
    session.exhausted(NO_NOTES)  # Redis: no notes -> offer; Kafka continues
    assert session.question_request(COLLECTIONS).source is InterviewSource.notes
    step = session.apply(General("redis"))  # case-insensitive
    assert step.do is Do.ask
    assert step.notice is not None
    assert "General knowledge · not verified" in step.notice
    request = session.question_request(COLLECTIONS)
    assert (request.topic, request.source, request.level) == (
        "Redis",
        InterviewSource.general,
        InterviewLevel.staff,
    )
    assert request.exclude_questions == []
    first = _general_question(text="Q one\\?")
    session.asked(first)
    session.graded("lru", _general_grade(7))
    # Kafka (notes) comes next; then Redis again, excluding the earlier general question
    kafka = session.question_request(COLLECTIONS)
    assert (kafka.topic, kafka.source, kafka.exclude_questions) == (
        "Kafka",
        InterviewSource.notes,
        [],
    )
    session.asked(question(topic="Kafka acks"))
    session.skip()
    again = session.question_request(COLLECTIONS)
    assert (again.topic, again.source) == ("Redis", InterviewSource.general)
    assert again.exclude_questions == ["Q one\\?"]


def test_exclude_questions_are_per_topic_capped_and_clipped() -> None:
    session = InterviewSession.begin(Start(("Redis",)))
    session.exhausted(NO_NOTES)
    session.apply(General())
    for i in range(25):
        session.question_request(COLLECTIONS)
        session.asked(_general_question(text=f"Q{i} " + "x" * 1200))
    session.skip()
    request = session.question_request(COLLECTIONS)
    assert len(request.exclude_questions) == MAX_EXCLUDED_QUESTIONS
    assert request.exclude_questions[0].startswith("Q5 ")  # the newest 20 kept
    assert all(len(q) == MAX_EXCLUDED_QUESTION_CHARS for q in request.exclude_questions)


def test_a_general_grade_request_has_no_chunks_and_echoes_source_level_token() -> None:
    session = InterviewSession.begin(Start(("Redis",)))
    session.exhausted(NO_NOTES)
    session.apply(General())
    session.question_request(COLLECTIONS)
    session.asked(_general_question())
    request = session.grade_request(COLLECTIONS, "LRU")
    assert request.chunk_ids == []
    assert (request.source, request.level, request.question_token) == (
        InterviewSource.general,
        InterviewLevel.staff,
        "gtok",
    )
    assert request.expected_points == ["LRU / LFU policies", "maxmemory"]


def test_an_unknown_source_is_never_treated_as_notes() -> None:
    odd = _general_question().model_copy(update={"source": "web"})
    assert is_general(odd)
    session = InterviewSession()
    session.asked(odd)
    assert session.grade_request(COLLECTIONS, "a").source is InterviewSource.general


def test_a_general_question_without_its_token_says_so_instead_of_a_validation_error() -> None:
    session = InterviewSession()
    session.asked(_general_question().model_copy(update={"question_token": None}))
    with pytest.raises(ValueError, match="came without its server token") as raised:
        session.grade_request(COLLECTIONS, "LRU")
    assert "validation error" not in str(raised.value)
    with pytest.raises(ValueError, match="server token"):  # checked before the answer itself
        session.grade_request(COLLECTIONS, "")


def test_an_empty_answer_key_still_grades() -> None:
    session = InterviewSession()
    session.asked(question().model_copy(update={"expected_points": []}))
    request = session.grade_request(COLLECTIONS, "acks=all")
    assert request.expected_points == []
    assert request.source is InterviewSource.notes


def _offering(session: InterviewSession) -> bool:
    return session.offering


def test_y_takes_the_offer_only_while_the_session_waits_on_it() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis")))
    session.asked(question())  # Kafka pending
    session.exhausted(NO_NOTES)  # Redis offered, Kafka question still pending
    assert not session.offering
    assert route("y", pending=True, offering=session.offering) == Answer("y")
    assert route("/general", pending=True, offering=session.offering) == General()
    assert route("/general Redis", pending=True) == General("Redis")
    session.skip()
    # nothing pending, but Kafka is still asked: an OLD offer is not taken by a stray "y"
    assert not _offering(session)
    assert route("y", pending=False, offering=session.offering) == Leave("y")
    assert route("/general Redis", pending=False) == General("Redis")
    session.exhausted(NO_NOTES)  # Kafka gone too: the session now waits on its offers
    assert _offering(session)  # (a call: mypy would keep the narrowed property)
    assert route("y", pending=False, offering=True) == General()
    assert route("Yes.", pending=False, offering=True) == General()
    assert route("y", pending=False, offering=False) == Leave("y")
    assert route_suspended("/general") == General()


def test_general_without_an_open_offer_is_refused() -> None:
    session = InterviewSession.begin(Start(("Kafka",)))
    step = session.apply(General("Kafka"))  # the notes still cover Kafka: never moved
    assert step.do is Do.nothing
    assert session.general_topics == set()
    session.exhausted(NO_NOTES)
    assert session.apply(General("Postgres")).do is Do.nothing  # not offered


def test_next_topic_moves_between_notes_and_general_topics() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis")))
    session.question_request(COLLECTIONS)
    session.asked(question())
    session.exhausted(NO_NOTES)  # Redis offered
    session.apply(General("Redis"))
    assert session.upcoming_general
    session.question_request(COLLECTIONS)
    session.asked(_general_question())
    assert session.apply(Run(Command.next_topic)).do is Do.ask
    assert session.question_request(COLLECTIONS).source is InterviewSource.notes


def test_a_start_phrase_clears_offers_and_general_topics() -> None:
    session = InterviewSession.begin(Start(("Redis",)))
    session.exhausted(NO_NOTES)
    session.apply(General())
    session.apply(Start(("Redis", "Kafka")))
    assert session.offers == []
    assert session.question_request(COLLECTIONS).source is InterviewSource.notes


def test_going_mixed_drops_offers_so_mixed_never_offers() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis")))
    session.asked(question())
    session.exhausted(NO_NOTES)  # Redis offered; topics == [Kafka]
    session.apply(Run(Command.next_topic))  # one topic -> mixed
    assert session.offers == []
    assert not session.waiting
    request = session.question_request(COLLECTIONS)
    assert (request.topic, request.source) == (None, InterviewSource.notes)


def test_an_exhausted_general_topic_is_dropped_without_a_new_offer() -> None:
    session = InterviewSession.begin(Start(("Redis",)))
    session.exhausted(NO_NOTES)
    session.apply(General())
    outcome = session.exhausted(EXHAUSTED)
    assert outcome.offer is None
    assert outcome.ended


# ------------------------------------------------------------------ summary split
def test_summary_keeps_notes_and_general_scores_apart() -> None:
    session = InterviewSession.begin(Start(("Kafka", "Redis"), InterviewLevel.staff))
    session.question_request(COLLECTIONS)
    session.asked(question(topic="Kafka ISR"))
    session.graded("a", grade(5))
    session.exhausted(NO_NOTES)  # Redis
    session.apply(General("Redis"))
    session.question_request(COLLECTIONS)
    session.asked(_general_question(topic="Redis eviction"))
    session.graded("b", _general_grade(4))
    session.question_request(COLLECTIONS)
    session.asked(question(topic="Kafka acks"))
    session.graded("c", grade(8))
    summary = session.summary()
    assert summary.average == pytest.approx(6.5)  # notes only: (5 + 8) / 2, never with the 4
    assert summary.general_average == pytest.approx(4.0)
    assert (summary.notes_answered, summary.general_answered, summary.answered) == (2, 1, 3)
    assert [w.topic for w in summary.weak] == ["Kafka ISR"]  # notes: pages to revisit
    assert summary.weak_general == (WeakGeneral("Redis", "concept", 4),)  # the session topic
    assert summary_headline(summary) == (
        "Staff · Kafka, Redis · avg 6.5 from your notes · general knowledge avg 4.0 (not verified)"
    )


def test_summary_headline_like_the_owner_wrote_it() -> None:
    session = InterviewSession.begin(Start(("Kafka",), InterviewLevel.staff))
    for score in (6, 7):
        session.asked(question())
        session.graded("a", grade(score))
    assert summary_headline(session.summary()) == "Staff · Kafka · avg 6.5"
    assert summary_headline(InterviewSession().summary()) == (
        "Senior SDE-1 · mixed topics · none graded"
    )


def test_level_names_and_question_line() -> None:
    assert [level_name(level) for level in InterviewLevel] == [
        "SDE-1",
        "SDE-2",
        "Senior SDE-1",
        "Senior SDE-2",
        "Staff",
    ]
    assert level_name("principal") == "principal"
    harder = question(difficulty=InterviewDifficulty.harder)
    assert question_line(harder.model_copy(update={"level": InterviewLevel.staff})) == (
        "Staff · harder"
    )
    assert question_line(question()) == "Senior SDE-1"
