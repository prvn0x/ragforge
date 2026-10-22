"""EV-RUN: the runners' pure parts (pacing, classification, page ranking, the E-V reduction,
the deterministic question checks), with property tests where an invariant exists."""

from __future__ import annotations

from uuid import UUID, uuid5

import httpx
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from test_ev_run_fakes import NS, api_error, load_verifier

from ragforge_client.evals.runners import (
    Pacer,
    RetryPolicy,
    StreamProblem,
    answer_leak,
    checked_texts,
    classify,
    cluster_order,
    pages_in_rank_order,
    references_code,
    verify_output,
    with_retries,
)
from ragforge_client.evals.schema import ErrorClass, ShownRef, Span
from ragforge_contracts.api import (
    EvalChunk,
    EvalChunksResponse,
    EvalCoverageUnit,
    EvalVerifierClaim,
    EvalVerifyResponse,
    PageKey,
    Problem,
    StageHit,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


# ------------------------------------------------------------------ pacing and retries
async def test_pacer_spaces_requests_evenly() -> None:
    clock = Clock()
    pacer = Pacer(rpm=30, clock=clock, sleep=clock.sleep)  # one request per 2 s
    await pacer.wait()
    clock.now += 0.5
    await pacer.wait()
    clock.now += 5.0
    await pacer.wait()
    assert clock.slept == [1.5]


async def test_no_rpm_means_no_pacing() -> None:
    clock = Clock()
    pacer = Pacer(rpm=None, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        await pacer.wait()
    assert clock.slept == []


@given(st.integers(min_value=100, max_value=599))
@example(429)
@example(428)
@example(499)
@example(500)
@example(599)
def test_only_429_and_5xx_are_transient(status: int) -> None:
    transient = status == 429 or status >= 500
    for exc in (api_error(status), StreamProblem(Problem(title="x", status=status))):
        error = classify(exc)
        assert (error.cls is ErrorClass.provider_transient) is transient
        assert error.status == status


def test_transport_failures_and_timeouts_are_ours() -> None:
    assert classify(httpx.ConnectError("refused")).cls is ErrorClass.ours
    assert classify(TimeoutError()).cls is ErrorClass.ours
    assert classify(TimeoutError()).message == "timeout"


@given(st.integers(min_value=0, max_value=4), st.integers(min_value=0, max_value=6))
async def test_retries_never_exceed_the_policy(retries: int, failures: int) -> None:
    clock = Clock()
    pacer = Pacer(clock=clock, sleep=clock.sleep)
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise api_error(503)
        return "ok"

    result, error, used = await with_retries(flaky, RetryPolicy(retries, backoff_s=1.0), pacer)
    assert used == attempts == min(failures, retries) + 1
    assert (result == "ok") is (failures <= retries)
    assert (error is None) is (failures <= retries)
    assert clock.slept == [2.0**i for i in range(used - 1)]


async def test_unexpected_exceptions_propagate() -> None:
    async def bug() -> str:
        raise KeyError("a bug of ours must crash loudly, not become a row")

    with pytest.raises(KeyError):
        await with_retries(bug, RetryPolicy(), Pacer())


# ------------------------------------------------------------------ E-R page ranking
_PAGES = [f"file:p{i}.md" for i in range(6)]


@st.composite
def stage_hits(draw: st.DrawFn) -> tuple[list[StageHit], dict[UUID, str]]:
    n = draw(st.integers(min_value=0, max_value=15))
    pages = draw(st.lists(st.sampled_from(_PAGES), min_size=n, max_size=n))
    ranks = draw(st.permutations(list(range(1, n + 1))))
    hits = [
        StageHit(chunk_id=uuid5(NS, f"c{i}"), document_id=uuid5(NS, p), rank=r, score=0.0)
        for i, (p, r) in enumerate(zip(pages, ranks, strict=True))
    ]
    return hits, {h.chunk_id: p for h, p in zip(hits, pages, strict=True)}


@given(stage_hits())
def test_pages_are_unique_and_ordered_by_their_best_rank(
    case: tuple[list[StageHit], dict[UUID, str]],
) -> None:
    hits, page_of = case
    pages = pages_in_rank_order(hits, page_of.__getitem__)
    assert len(pages) == len(set(pages)) == len(set(page_of.values()))
    best = {p: min(h.rank for h in hits if page_of[h.chunk_id] == p) for p in pages}
    assert [best[p] for p in pages] == sorted(best.values())


def test_rank_ties_are_broken_by_chunk_id() -> None:
    a, b = UUID(int=1), UUID(int=2)
    hits = [
        StageHit(chunk_id=b, document_id=b, rank=1, score=0.0),
        StageHit(chunk_id=a, document_id=a, rank=1, score=0.0),
    ]
    assert pages_in_rank_order(hits, {a: "file:a.md", b: "file:b.md"}.__getitem__) == [
        "file:a.md",
        "file:b.md",
    ]


# ------------------------------------------------------------------ E-V reduction
def _response(**fields: object) -> EvalVerifyResponse:
    base: dict[str, object] = {
        "status": "partial",
        "claims": [],
        "coverage_units": [],
        "verifier_claims": [],
        "verifier_ran": True,
    }
    return EvalVerifyResponse.model_validate(base | fields)


def test_located_and_unlocated_verifier_claims_are_split() -> None:
    out = verify_output(
        _response(
            verifier_claims=[
                EvalVerifierClaim(text="a", start=3, end=9),
                EvalVerifierClaim(text="b"),
                EvalVerifierClaim(text="c", start=4, end=4),  # empty: not a location
            ],
            coverage_units=[EvalCoverageUnit(text="x", start=0, end=2, kind="residue")],
            verifier_verdict="partial",
        )
    )
    assert out.verifier_spans == [Span(start=3, end=9)]
    assert out.unlocated_verifier_claims == 2
    assert out.coverage_spans == [Span(start=0, end=2)]
    assert out.verifier_objected


@pytest.mark.parametrize(
    ("verdict", "claims", "objected"),
    [
        ("verified", [], False),
        (None, [], False),  # the verifier did not run
        ("unsupported", [], True),  # an objection with nothing to place (rule 4a)
        ("verified", [EvalVerifierClaim(text="a")], True),
    ],
)
def test_an_objection_is_a_non_verified_verdict_or_any_claim(
    verdict: str | None, claims: list[EvalVerifierClaim], objected: bool
) -> None:
    out = verify_output(_response(verifier_verdict=verdict, verifier_claims=claims))
    assert out.verifier_objected is objected


def test_cluster_order_puts_each_clean_source_first() -> None:
    items = cluster_order(load_verifier().items)
    seen_clean: set[str] = set()
    for item in items:
        if item.mutation is None:
            seen_clean.add(item.id)
        else:
            assert item.source_id in seen_clean
    assert len(items) == len(load_verifier().items)


# ------------------------------------------------------------------ E-I question checks
def test_answer_leak_flags_a_question_that_states_its_answer() -> None:
    points = ["Cooperative sticky moves only the partitions that change owner"]
    assert answer_leak(
        "Why does cooperative sticky move only partitions that change owner?", points
    )
    assert not answer_leak("How does the cooperative assignor reduce rebalance pauses?", points)
    assert not answer_leak("Anything?", [])


def test_a_code_question_must_point_at_the_code() -> None:
    notes = ["Text\n\n```python\ndef charge(idempotency_key):\n    store.put_if_absent(key)\n```\n"]
    assert references_code("What does put_if_absent guard against?", notes)
    assert references_code("What does `charge` return twice?", notes)
    assert not references_code("What does the function guard against?", notes)
    assert not references_code("Why store the key?", notes)  # a plain word in a block


# ------------------------------------------------------------------ the judge sees what was shown
def _chunk(cid: UUID, text: str, sha: str) -> EvalChunk:
    return EvalChunk(
        chunk_id=cid,
        document_id=uuid5(NS, "doc"),
        page_key=PageKey(source_kind="file", key="a.md"),
        heading_path=[],
        ordinal=0,
        display_text=text,
        display_sha256=sha,
    )


def test_checked_texts_follow_the_labels_and_refuse_any_drift() -> None:
    import hashlib

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    a, b = uuid5(NS, "a"), uuid5(NS, "b")
    refs = [
        ShownRef(label=2, chunk_id=str(b), display_sha256=sha("bee")),
        ShownRef(label=1, chunk_id=str(a), display_sha256=sha("ay")),
    ]
    ok = EvalChunksResponse(chunks=[_chunk(a, "ay", sha("ay")), _chunk(b, "bee", sha("bee"))])
    assert checked_texts(refs, ok) == ("ay", "bee")
    # the server's digest still matches the run, but the text it returned does not
    lying = EvalChunksResponse(chunks=[_chunk(a, "ay!", sha("ay")), _chunk(b, "bee", sha("bee"))])
    assert checked_texts(refs, lying) is None
    # the text changed since the run (a new digest), or the chunk is gone
    moved = EvalChunksResponse(chunks=[_chunk(a, "ay2", sha("ay2")), _chunk(b, "bee", sha("bee"))])
    assert checked_texts(refs, moved) is None
    assert checked_texts(refs, EvalChunksResponse(chunks=[_chunk(a, "ay", sha("ay"))])) is None
