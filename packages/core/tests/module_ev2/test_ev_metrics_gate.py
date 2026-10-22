"""EV-METRICS gate scenarios, built from the fixture runs: A/A passes, a synthetic drop blocks,
incomparable runs are not evaluable, transient errors are excluded, warn-only surfaces never
block, every check prints n/method/MDE, and an override is recorded (never hidden)."""

from __future__ import annotations

from dataclasses import replace

from test_ev_metrics_support import datasets, fixture_run, renamed, with_rows

from ragforge_client.evals import gate, metrics
from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    AnswerOutput,
    ErrorClass,
    EvalKind,
    GateOutcome,
    GateVerdict,
    GradeOutput,
    RetrievalConfig,
    RetrievalOutput,
    RunError,
    RunRow,
    SnapshotRef,
    Timings,
    VerifierSet,
    VerifyOutput,
    digest,
)


def _check(verdict: GateVerdict, name: str) -> GateOutcome:
    return next(c.outcome for c in verdict.checks if c.name == name)


def _aa(run: RunFile) -> list[RunFile]:
    return [renamed(run, "aa-1"), renamed(run, "aa-2")]


# ------------------------------------------------------------------ synthetic arms
def _demote_relevant(row: RunRow, to: int = 8) -> RunRow:
    """Move every relevant page of a need=any question below rank `to` in the reranked stage."""
    qs = datasets().questions
    assert qs is not None
    q = next(q for q in qs.questions if q.id == row.item_id)
    out = row.output
    if not q.answerable or q.need.value != "any" or not isinstance(out, RetrievalOutput):
        return row
    ranked = out.stages["reranked"]
    rest = [p for p in ranked if p not in q.relevance]
    rel = [p for p in ranked if p in q.relevance]
    stages = {**out.stages, "reranked": rest[: to - 1] + rel + rest[to - 1 :]}
    return row.model_copy(update={"output": out.model_copy(update={"stages": stages})})


def _graded_generator(correct_ids: set[str] | None = None, run_id: str = "gen") -> RunFile:
    """The generator fixture with a deterministic `correct` grade (all answered rows correct,
    or only `correct_ids`)."""
    run = fixture_run("generator")

    def change(r: RunRow) -> RunRow:
        assert isinstance(r.output, AnswerOutput)
        ok = not r.output.abstained if correct_ids is None else r.item_id in correct_ids
        return r.model_copy(update={"grades": {"correct": ok}})

    return with_rows(run, change, run_id)


def _slo_output(r: RunRow, ds: Datasets) -> RunRow:
    """Every clean item verified with no objection, every planted item flipped on its span."""
    assert ds.verifier is not None
    item = next(i for i in ds.verifier.items if i.id == r.item_id)
    if item.span is None:
        out = VerifyOutput(
            status="verified",
            verifier_ran=True,
            verifier_objected=False,
            coverage_spans=[],
            verifier_spans=[],
            unlocated_verifier_claims=0,
        )
    else:
        out = VerifyOutput(
            status="partial",
            verifier_ran=True,
            verifier_objected=True,
            coverage_spans=[],
            verifier_spans=[item.span],
            unlocated_verifier_claims=0,
        )
    return r.model_copy(update={"output": out})


def _verifier_variant(
    *, times: int = 1, sources: set[str] | None = None, run_id: str = "ver"
) -> tuple[RunFile, Datasets]:
    """A synthetic verifier set built from D0: only `sources` (all when None), each source's
    items replicated `times` under new ids, with a run that meets the SLO on every item. The
    header's dataset digest is recomputed so the metric layer accepts the pair."""
    base, ds = fixture_run("verifier"), datasets()
    assert ds.verifier is not None
    keep = [i for i in ds.verifier.items if sources is None or i.source_id in sources]
    items = [
        i.model_copy(update={"id": f"{i.id}~{t}", "source_id": f"{i.source_id}~{t}"})
        for t in range(times)
        for i in keep
    ]
    vset = VerifierSet(name=f"demo-x{times}", items=items)
    new_ds = Datasets(questions=ds.questions, verifier=vset, interview=ds.interview)
    by_id = {r.item_id: r for r in base.rows}
    rows = [
        _slo_output(by_id[i.id.split("~")[0]].model_copy(update={"item_id": i.id}), new_ds)
        for i in items
    ]
    ref = base.header.dataset.model_copy(
        update={"name": vset.name, "digest": digest(vset.model_dump(mode="json"))}
    )
    header = base.header.model_copy(update={"run_id": run_id, "dataset": ref})
    return replace(base, header=header, rows=rows), new_ds


def _slo_verifier(run_id: str = "ver") -> tuple[RunFile, Datasets]:
    """A verifier run that meets the SLO with room to spare: D0 x3 (24 clean, 36 planted), so
    0/24 false rejects has an exact upper bound of 0.14 and 36/36 flips a lower bound of 0.90."""
    return _verifier_variant(times=3, run_id=run_id)


# ------------------------------------------------------------------ A/A acceptance
def test_aa_pair_passes_on_retrieval() -> None:
    run = fixture_run("retrieval")
    verdict = gate.evaluate(run, renamed(run, "cand"), _aa(run), datasets())
    assert verdict.outcome is GateOutcome.pass_
    assert _check(verdict, "aa.retrieval_identical") is GateOutcome.pass_


def test_aa_pair_passes_on_chat_when_graded() -> None:
    run = _graded_generator()
    verdict = gate.evaluate(run, renamed(run, "cand"), _aa(run), datasets())
    assert verdict.outcome is GateOutcome.pass_, gate.format_verdict(verdict)


def test_aa_pair_passes_on_a_verifier_that_meets_the_slo() -> None:
    run, ds = _slo_verifier()
    verdict = gate.evaluate(run, renamed(run, "cand"), _aa(run), ds)
    assert verdict.outcome is GateOutcome.pass_, gate.format_verdict(verdict)


def test_a_noisy_aa_pair_passes_on_retrieval() -> None:
    """Realistic A/A noise (one question retrieved differently) must not block: neither the
    paired tests nor a floor may fire. The ADR expects identical retrieval with chunk_id
    tie-breaks, so the difference itself is surfaced as a warn, which still passes."""
    a1 = fixture_run("retrieval")
    first = next(r.item_id for r in a1.rows if _demote_relevant(r) != r and _first_rank(r) <= 5)
    a2 = with_rows(a1, lambda r: _demote_relevant(r) if r.item_id == first else r, "aa-2")
    verdict = gate.evaluate(a1, a2, [a1, a2], datasets())
    assert gate.passes(verdict), gate.format_verdict(verdict)
    not_pass = {c.name for c in verdict.checks if c.outcome is not GateOutcome.pass_}
    assert not_pass == {"aa.retrieval_identical"}, gate.format_verdict(verdict)


def test_a_noisy_aa_pair_passes_on_chat() -> None:
    """Two of the generator's correct grades flip between two runs of one build: PASS."""
    a1 = _graded_generator()
    answered = sorted(
        r.item_id for r in a1.rows if isinstance(r.output, AnswerOutput) and not r.output.abstained
    )
    a2 = _graded_generator(correct_ids=set(answered[2:]), run_id="aa-2")
    verdict = gate.evaluate(a1, a2, [a1, a2], datasets())
    assert verdict.outcome is GateOutcome.pass_, gate.format_verdict(verdict)
    correct = next(c for c in verdict.checks if c.name == "chat.correct")
    assert correct.effect is not None
    assert correct.effect < 0  # the noise is real, it is just not significant


def test_aa_pair_passes_on_interview() -> None:
    run = fixture_run("interview")
    verdict = gate.evaluate(run, renamed(run, "cand"), _aa(run), datasets())
    assert verdict.outcome is GateOutcome.pass_


def test_the_fixture_verifier_fails_only_the_absolute_slo_not_the_regression_test() -> None:
    """At n = 10 eligible items the badge-flip CI low is 0.555 (Clopper-Pearson-widened) < 0.75
    and the false-reject upper bound 0.53 > 0.20: the SLO is a property of the verifier model AND
    of the evidence, so D0 can never certify a model and even A/A blocks; the paired test
    itself passes."""
    run = fixture_run("verifier")
    verdict = gate.evaluate(run, renamed(run, "cand"), _aa(run), datasets())
    assert verdict.outcome is GateOutcome.block
    assert _check(verdict, "verifier.planted.badge_flipped") is GateOutcome.pass_
    assert _check(verdict, "slo.verifier.badge_flip_recall.ci_low") is GateOutcome.block


# ------------------------------------------------------------------ regressions block
def test_a_synthetic_retrieval_drop_blocks() -> None:
    base = fixture_run("retrieval")
    cand = with_rows(base, _demote_relevant, "dropped")
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert verdict.outcome is GateOutcome.block
    hit = next(c for c in verdict.checks if c.name == "retrieval.reranked.hit@5")
    assert hit.outcome is GateOutcome.block
    assert hit.effect is not None
    assert hit.effect < -0.02
    assert hit.p_value is not None
    assert hit.p_value < 0.025
    assert "Holm-adjusted" in hit.note


def test_a_one_question_drop_passes_without_a_configured_floor() -> None:
    """r1 P1: the floor is absolute (frozen.json), never the baseline's value. One demoted
    question (8.3 points at n = 12, p = 1) passes the test and, with no floor, the floor too."""
    base = fixture_run("retrieval")
    first = next(r.item_id for r in base.rows if _demote_relevant(r) != r and _first_rank(r) <= 5)
    cand = with_rows(base, lambda r: _demote_relevant(r) if r.item_id == first else r, "one")
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert verdict.outcome is GateOutcome.pass_, gate.format_verdict(verdict)
    floor = next(c for c in verdict.checks if c.name == "floor.retrieval.reranked.hit@5")
    assert floor.outcome is GateOutcome.pass_
    assert "no absolute floor configured" in floor.note


def test_a_configured_floor_blocks_below_floor_minus_3_and_only_there() -> None:
    base = fixture_run("retrieval")
    first = next(r.item_id for r in base.rows if _demote_relevant(r) != r and _first_rank(r) <= 5)
    cand = with_rows(base, lambda r: _demote_relevant(r) if r.item_id == first else r, "one")
    hit = metrics.metric_map(metrics.summarize(cand, datasets(), [], []))[gate.RETRIEVAL_HIT]
    assert hit.value is not None

    def floor_outcome(floor: float) -> GateOutcome:
        v = gate.evaluate(base, cand, _aa(base), datasets(), floors={gate.RETRIEVAL_HIT: floor})
        assert _check(v, "retrieval.reranked.hit@5") is GateOutcome.pass_  # the test never fires
        return _check(v, "floor.retrieval.reranked.hit@5")

    assert floor_outcome(hit.value + 0.029) is GateOutcome.pass_  # within the 3-point margin
    assert floor_outcome(hit.value + 0.031) is GateOutcome.block
    assert floor_outcome(0.5) is GateOutcome.pass_


def _first_rank(row: RunRow) -> int:
    qs = datasets().questions
    assert qs is not None
    assert isinstance(row.output, RetrievalOutput)
    q = next(q for q in qs.questions if q.id == row.item_id)
    ranked = row.output.stages["reranked"]
    return next((i + 1 for i, p in enumerate(ranked) if p in q.relevance), 99)


def test_an_improvement_never_blocks() -> None:
    base = with_rows(fixture_run("retrieval"), _demote_relevant, "worse-baseline")
    cand = renamed(fixture_run("retrieval"), "better")
    verdict = gate.evaluate(base, cand, [], datasets())
    assert _check(verdict, "retrieval.reranked.hit@5") is GateOutcome.pass_


def test_a_chat_correctness_drop_blocks() -> None:
    base = _graded_generator()
    cand = _graded_generator(correct_ids=set(), run_id="all-wrong")
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert _check(verdict, "chat.correct") is GateOutcome.block
    assert verdict.outcome is GateOutcome.block


def test_an_error_of_our_own_is_a_hard_floor_on_chat() -> None:
    base = _graded_generator()
    err = RunError(cls=ErrorClass.ours, message="KeyError in grader")
    cand = with_rows(
        base,
        lambda r: (
            r.model_copy(update={"output": None, "error": err})
            if r.item_id == "kafka-ordering"
            else r
        ),
        "one-error",
    )
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert _check(verdict, "errors.ours") is GateOutcome.block


def test_planted_errors_no_longer_caught_block_the_verifier() -> None:
    base, ds = _slo_verifier()

    def miss(r: RunRow) -> RunRow:
        if r.item_id.startswith("p-") and isinstance(r.output, VerifyOutput):
            return r.model_copy(
                update={
                    "output": r.output.model_copy(
                        update={
                            "status": "verified",
                            "verifier_spans": [],
                            "verifier_objected": False,
                        }
                    )
                }
            )
        return r

    verdict = gate.evaluate(base, with_rows(base, miss, "blind"), _aa(base), ds)
    assert _check(verdict, "verifier.planted.badge_flipped") is GateOutcome.block
    assert _check(verdict, "slo.verifier.badge_flip_recall.ci_low") is GateOutcome.block


def test_the_slo_fails_on_a_perfect_verifier_at_tiny_n() -> None:
    """r1 P1: 2 sources (3 planted, 2 clean), all flipped and none rejected. A percentile
    bootstrap says [1, 1] and [0, 0]; the exact bounds (CP low 0.29, CP high 0.84) fail both."""
    run, ds = _verifier_variant(sources={"c-durability", "c-ordering"})
    verdict = gate.evaluate(run, renamed(run, "cand"), [], ds)
    flip = next(c for c in verdict.checks if c.name == "slo.verifier.badge_flip_recall.ci_low")
    reject = next(
        c for c in verdict.checks if c.name == "slo.verifier.false_reject.verifier_caused.ci_high"
    )
    assert flip.n == 3
    assert reject.n == 2
    assert flip.outcome is GateOutcome.block
    assert reject.outcome is GateOutcome.block
    assert flip.effect is not None
    assert abs(flip.effect - 0.2924017738) < 1e-6  # CP low of 3/3
    assert reject.effect is not None
    assert abs(reject.effect - 0.8418861170) < 1e-6  # CP high of 0/2
    assert "clopper_pearson" in flip.method


def test_the_chat_and_verifier_verdicts_compose() -> None:
    """ADR §5: on a chat change the E-V hard floor (a verifier-kind verdict) is ANDed in."""
    chat = _graded_generator()
    chat_v = gate.evaluate(chat, renamed(chat, "c"), _aa(chat), datasets())
    good, ds = _slo_verifier()
    good_v = gate.evaluate(good, renamed(good, "c"), _aa(good), ds)
    tiny, tds = _verifier_variant(sources={"c-durability", "c-ordering"})
    tiny_v = gate.evaluate(tiny, renamed(tiny, "c"), [], tds)
    assert gate.all_pass([chat_v, good_v])
    assert not gate.all_pass([chat_v, tiny_v])
    assert not gate.all_pass([])


# ------------------------------------------------------------------ warn-only surfaces
def test_interview_regressions_only_warn() -> None:
    base = fixture_run("interview")

    def flatten(r: RunRow) -> RunRow:
        assert isinstance(r.output, GradeOutput)
        return r.model_copy(
            update={"output": r.output.model_copy(update={"score": 5, "misconceptions_kept": []})}
        )

    verdict = gate.evaluate(base, with_rows(base, flatten, "flat"), _aa(base), datasets())
    assert verdict.outcome is GateOutcome.warn
    assert _check(verdict, "interview.ranking") is GateOutcome.warn
    assert all(c.outcome is not GateOutcome.block for c in verdict.checks)


# ------------------------------------------------------------------ not evaluable
def test_a_content_digest_mismatch_is_not_evaluable() -> None:
    base = fixture_run("retrieval")
    snap = SnapshotRef(collection_id="c", content_digest="a" * 64, index_digest="i" * 64)
    a = renamed(base, "a", snapshot=snap)
    b = renamed(base, "b", snapshot=snap.model_copy(update={"content_digest": "b" * 64}))
    verdict = gate.evaluate(a, b, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    assert "content digest" in verdict.checks[0].note
    assert not gate.passes(verdict.model_copy(update={"override": "ship it"}))


def test_a_reembed_is_evaluable_but_a_chunker_change_is_not() -> None:
    base = fixture_run("retrieval")
    snap = SnapshotRef(collection_id="c", content_digest="a" * 64, index_digest="i" * 64)
    a = renamed(base, "a", snapshot=snap)
    reembed = renamed(base, "b", snapshot=snap.model_copy(update={"index_digest": "j" * 64}))
    assert gate.evaluate(a, reembed, [], datasets()).outcome is not GateOutcome.not_evaluable
    assert base.header.retrieval is not None
    chunker: RetrievalConfig = base.header.retrieval.model_copy(
        update={"chunker_version": "heading-v2"}
    )
    rechunk = renamed(a, "c", retrieval=chunker)
    verdict = gate.evaluate(a, rechunk, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    assert "chunker" in verdict.checks[0].note


def test_retrieval_config_in_only_one_arm_is_not_evaluable() -> None:
    base = fixture_run("retrieval")
    verdict = gate.evaluate(base, renamed(base, "no-config", retrieval=None), [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    assert "retrieval config in only one arm" in verdict.checks[0].note


def test_different_eval_kinds_or_datasets_are_not_evaluable() -> None:
    verdict = gate.evaluate(fixture_run("retrieval"), fixture_run("generator"), [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    one_snapshot = renamed(
        fixture_run("retrieval"),
        "s",
        snapshot=SnapshotRef(collection_id="c", content_digest="a" * 64, index_digest="i" * 64),
    )
    verdict = gate.evaluate(fixture_run("retrieval"), one_snapshot, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable


# ------------------------------------------------------------------ hygiene, MDE, override
def test_transient_errors_are_excluded_from_both_arms() -> None:
    """A 429 on every item the candidate would have lost must not read as a regression."""
    base = fixture_run("retrieval")
    err = RunError(cls=ErrorClass.provider_transient, status=429, message="rate limited")
    lost = sorted(r.item_id for r in base.rows if _demote_relevant(r) != r and _first_rank(r) <= 5)
    lost = lost[:8]  # 8 one-way losses: p = 2/256, Holm x2 = 0.0156 < 0.025 -> a real block
    assert len(lost) == 8
    storm = with_rows(
        base,
        lambda r: r.model_copy(update={"output": None, "error": err}) if r.item_id in lost else r,
        "storm",
    )
    real = with_rows(base, lambda r: _demote_relevant(r) if r.item_id in lost else r, "real")
    assert (
        _check(gate.evaluate(base, real, _aa(base), datasets()), "retrieval.reranked.hit@5")
        is GateOutcome.block
    )  # the control: a real loss
    # 8 of 12 lost to a 429 storm is UNMEASURED (integrator fix of review P1): never a pass
    verdict = gate.evaluate(base, storm, _aa(base), datasets())
    hit = next(c for c in verdict.checks if c.name == "retrieval.reranked.hit@5")
    assert hit.outcome is GateOutcome.not_evaluable
    assert not gate.passes(verdict)
    assert f"{len(lost)} provider-transient item(s) excluded" in hit.note
    assert _check(verdict, "errors.ours") is GateOutcome.pass_
    # a small loss (2 of 12, under the 20% bound) is excluded from both arms and still measured
    few = with_rows(
        base,
        lambda r: (
            r.model_copy(update={"output": None, "error": err}) if r.item_id in lost[:2] else r
        ),
        "few",
    )
    small = next(
        c
        for c in gate.evaluate(base, few, _aa(base), datasets()).checks
        if c.name == "retrieval.reranked.hit@5"
    )
    assert small.outcome is GateOutcome.pass_
    assert small.n == 12 - 2


def test_an_all_transient_candidate_never_passes() -> None:
    base = fixture_run("generator")
    err = RunError(cls=ErrorClass.provider_transient, status=503, message="unavailable")
    cand = with_rows(base, lambda r: r.model_copy(update={"output": None, "error": err}), "outage")
    assert not gate.passes(gate.evaluate(base, cand, _aa(base), datasets()))


def test_an_error_of_ours_hidden_by_a_successful_resume_still_blocks() -> None:
    """Attempt 1 fails with our own error, attempt 2 (a resume) succeeds: the flaky bug of ours
    must still hit the errors.ours hard floor (integrator fix of review P1)."""
    base = _graded_generator()
    err = RunError(cls=ErrorClass.ours, message="KeyError in grader")
    first = next(r for r in base.rows if r.item_id == "kafka-ordering")
    failed = first.model_copy(update={"output": None, "error": err, "attempt": 1})
    retried = first.model_copy(update={"attempt": 2})
    cand = replace(
        renamed(base, "resumed"),
        rows=[failed if r.item_id == "kafka-ordering" else r for r in base.rows] + [retried],
    )
    assert metrics.select_rows(cand).errors_ours == 0  # the metric uses the latest attempt
    assert metrics.select_rows(cand).ours_any_attempt == 1
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert _check(verdict, "errors.ours") is GateOutcome.block


def test_items_stale_in_one_arm_are_excluded_and_counted() -> None:
    base = fixture_run("retrieval")
    lost = sorted(r.item_id for r in base.rows if _demote_relevant(r) != r and _first_rank(r) <= 5)
    lost = lost[:8]
    cand = with_rows(
        base,
        lambda r: (
            _demote_relevant(r).model_copy(update={"stale": True}) if r.item_id in lost else r
        ),
        "stale",
    )
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    hit = next(c for c in verdict.checks if c.name == "retrieval.reranked.hit@5")
    assert hit.outcome is GateOutcome.pass_  # the losses are on stale rows only
    assert hit.n == 12 - len(lost)
    assert f"{len(lost)} stale item(s) excluded" in hit.note


def _latency_run(rows: list[RunRow]) -> RunFile:
    base = fixture_run("retrieval")
    return replace(
        base,
        header=base.header.model_copy(update={"run_id": "lat", "eval_kind": EvalKind.latency}),
        rows=rows,
    )


def test_latency_keeps_the_order_asked_after_a_resume_and_keeps_a_zero_total() -> None:
    """A transient first attempt retried at the END of the file keeps its first position, so it
    is still cold; a total of 0.0 is a value, not a missing one."""
    base = fixture_run("retrieval")
    err = RunError(cls=ErrorClass.provider_transient, status=429, message="rate limited")
    rows = [
        r.model_copy(update={"timings": Timings(ttft_s=0.1, total_s=float(i))})
        for i, r in enumerate(base.rows[:6])
    ]
    first = rows[0].model_copy(update={"output": None, "error": err})
    retry = rows[0].model_copy(update={"attempt": 2, "timings": Timings(ttft_s=9.0, total_s=9.0)})
    run = _latency_run([first, *rows[1:], retry])
    cold, warm = metrics.cold_and_warm_rows(run)
    assert [r.item_id for r in cold] == [rows[0].item_id, rows[1].item_id]
    assert cold[0].attempt == 2
    assert [r.item_id for r in warm] == [r.item_id for r in rows[2:]]
    zero = _latency_run(
        [rows[0], rows[1], rows[2].model_copy(update={"timings": Timings(total_s=0.0)}), *rows[3:]]
    )
    m = metrics.metric_map(metrics.summarize(zero, datasets(), [], []))
    assert m["latency.total.p50"].n == 4
    verdict = gate.evaluate(zero, renamed(zero, "c"), [], datasets())
    assert next(c for c in verdict.checks if c.name == "latency.total_s").n == 4


def test_every_check_prints_n_method_and_mde() -> None:
    base = fixture_run("retrieval")
    verdict = gate.evaluate(base, with_rows(base, _demote_relevant, "d"), _aa(base), datasets())
    for c in verdict.checks:
        assert c.method
        assert c.mde is not None or "MDE" in c.note, c.name
    text = gate.format_verdict(verdict)
    assert "n=12" in text
    assert "MDE=" in text
    assert "exact McNemar" in text


def test_mde_comes_from_the_aa_discordance() -> None:
    base = fixture_run("retrieval")
    noisy = with_rows(base, _demote_relevant, "aa-noisy")  # 9 of 12 hit@5 outcomes flip
    quiet = gate.evaluate(base, renamed(base, "c"), _aa(base), datasets())
    loud = gate.evaluate(base, renamed(base, "c"), [base, noisy], datasets())
    q = next(c for c in quiet.checks if c.name == "retrieval.reranked.hit@5")
    lo = next(c for c in loud.checks if c.name == "retrieval.reranked.hit@5")
    assert q.mde is not None
    assert lo.mde is not None
    assert lo.mde > q.mde  # 0.70 from discordance 0.75 vs the 7/12 discreteness floor
    assert "A/A discordance 0.750" in lo.note
    mrr = next(c for c in quiet.checks if c.name == "retrieval.reranked.mrr")
    assert mrr.mde is None
    assert "MDE n/a: A/A identical" in mrr.note
    missing = gate.evaluate(base, renamed(base, "c"), [], datasets())
    m = next(c for c in missing.checks if c.name == "retrieval.reranked.hit@5")
    assert m.mde is None
    assert "no comparable A/A pair" in m.note


def test_an_override_is_recorded_and_honoured_only_for_a_block() -> None:
    base = fixture_run("retrieval")
    cand = with_rows(base, _demote_relevant, "dropped")
    cand = renamed(cand, "dropped", override="known regression, ticket filed")
    verdict = gate.evaluate(base, cand, _aa(base), datasets())
    assert verdict.outcome is GateOutcome.block  # the evidence is never rewritten
    assert verdict.override == "known regression, ticket filed"
    assert gate.passes(verdict)
    assert "override: known regression" in gate.format_verdict(verdict)
    assert not gate.passes(verdict.model_copy(update={"override": None}))


def test_the_gate_is_deterministic() -> None:
    base = fixture_run("retrieval")
    cand = with_rows(base, _demote_relevant, "d")
    assert gate.evaluate(base, cand, _aa(base), datasets()) == gate.evaluate(
        base, cand, _aa(base), datasets()
    )
