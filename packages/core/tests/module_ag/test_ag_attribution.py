"""Server-side attribution and the status of a mixed answer (ADR-0015), as pure functions.

The coverage guarantee itself (residue, allowlist, the property and the review attacks) is in
test_ag_coverage.py."""

from __future__ import annotations

import time

import pytest

from module_d1.d1_fakes import make_hit
from ragforge_core.agent.attribution import (
    Attribution,
    Inferred,
    annotate,
    attribute,
    content_tokens,
    inferred_labels,
    negated,
    no_provenance,
    settle,
)
from ragforge_core.agent.coverage import valid_labels
from ragforge_core.agent.state import hit_to_state
from ragforge_core.agent.structure import Code, structure
from ragforge_core.domain.models import Verification

KAFKA = hit_to_state(
    make_hit(
        1,
        title="Kafka basics",
        text="Kafka stores records in partitions. Each partition is an ordered, immutable log. "
        "Consumers track their position with offsets.",
    )
)
REDIS = hit_to_state(
    make_hit(
        2,
        title="Redis cluster",
        text="Redis Cluster splits the keyspace into 16384 hash slots across master nodes.",
    )
)
HITS = [KAFKA, REDIS]
VALID = valid_labels([1, 2], 0)


def _attribute(answer: str, shown: tuple[int, ...] = (1, 2)) -> Attribution:
    return attribute(answer, HITS, shown, valid_labels(shown, 0))


def test_correct_uncited_sentences_are_attributed_to_the_source_that_states_them() -> None:
    answer = (
        "Kafka stores records in partitions. Each partition is an ordered, immutable log. "
        "Redis Cluster uses 16384 hash slots."
    )
    found = _attribute(answer)
    assert [(i.label, i.sentence) for i in found.inferred] == [
        (1, "Kafka stores records in partitions."),
        (1, "Each partition is an ordered, immutable log."),
        (2, "Redis Cluster uses 16384 hash slots."),
    ]
    assert found.uncited == ()
    assert not found.covered
    assert not found.refusal_only


@pytest.mark.parametrize(
    "fabricated",
    [
        "Kafka was invented by NASA in 1850.",  # shares one topic word only
        "Redis Cluster splits the keyspace into 999 hash slots.",  # drifted number
        "Zookeeper elects the controller broker using Raft consensus.",  # unrelated claim
    ],
)
def test_a_fabricated_sentence_gets_no_source(fabricated: str) -> None:
    found = _attribute(f"Kafka stores records in partitions. {fabricated}")
    assert [i.sentence for i in found.inferred] == ["Kafka stores records in partitions."]
    assert found.uncited == (fabricated,)


@pytest.mark.parametrize(
    "negated",
    [
        "Kafka does not store records in partitions.",
        "Kafka never stores records in partitions.",
        "Kafka doesn\u2019t store records in partitions.",
        # blind review round 2 of the parser rewrite (P2)
        "Kafka doesnt store records in partitions.",
        "Kafka does n-o-t store records in partitions.",
        "Kafka is incapable of storing records in partitions.",
    ],
)
def test_a_negation_the_source_lacks_gets_no_source(negated: str) -> None:
    """Review attack: every content word of the negated sentence is in its source, so lexical
    overlap alone lent it an inferred citation and a verifier miss made it `verified`."""
    assert _attribute(negated).uncited == (negated,)


@pytest.mark.parametrize(
    "plain", ["The Kafka client stores records.", "An event agent can hint at it.", "U.S. law"]
)
def test_words_ending_in_nt_or_dotted_letters_are_not_negations(plain: str) -> None:
    """The apostrophe-less contractions are a closed list: "client", "event" and "hint" end in
    "nt" without negating anything; "U.S." joined is "us", not a negation."""
    assert not negated(plain)


def test_a_dozen_is_a_number() -> None:
    """ "a dozen brokers" must pass the number check like "12 brokers" (and fail against 3)."""
    found = _attribute("The Kafka cluster has a dozen brokers.")
    assert found.uncited == ("The Kafka cluster has a dozen brokers.",)


def test_a_negation_the_source_states_is_attributed() -> None:
    page = hit_to_state(make_hit(1, text="Kafka does not delete records after consumption."))
    answer = "Kafka does not delete records after consumption."
    assert [i.label for i in attribute(answer, [page], [1], VALID).inferred] == [1]


def test_a_recombined_number_gets_no_source() -> None:
    """Review round 4: membership let "3 partitions" borrow the 3 of "3 brokers". The number
    must stand next to one of its neighbours in the source too."""
    brokers = hit_to_state(
        make_hit(
            1, title="Kafka", text="Kafka stores records in partitions. The cluster has 3 brokers."
        )
    )
    recombined = "Kafka stores records in 3 partitions."
    assert attribute(recombined, [brokers], [1], VALID).uncited == (recombined,)
    honest = "The Kafka cluster has 3 brokers."
    assert [i.label for i in attribute(honest, [brokers], [1], VALID).inferred] == [1]


def test_a_number_is_matched_next_to_its_noun_across_stopwords() -> None:
    assert [i.label for i in _attribute("Redis Cluster uses 16384 hash slots.").inferred] == [2]


def test_only_shown_labels_can_be_inferred() -> None:
    """REDIS supports the sentence, but the model was only shown S1: no inferred S2."""
    found = _attribute("Redis Cluster splits the keyspace into 16384 hash slots.", (1,))
    assert found.inferred == ()
    assert found.uncited == ("Redis Cluster splits the keyspace into 16384 hash slots.",)


def test_model_cited_units_are_not_attributed_and_count_as_covered() -> None:
    found = _attribute("Kafka stores records in partitions [S2].")
    assert found.inferred == ()
    assert found.uncited == ()
    assert found.covered


def test_an_invalid_label_covers_nothing() -> None:
    """check_citations rejects it first (rule 1); attribution never counts it as coverage."""
    assert not _attribute("Kafka stores records in partitions [S9].").covered


def test_label_markers_never_count_as_evidence() -> None:
    assert content_tokens("Partitions [S1] [s2] are logs") == ["partition", "log"]


def test_the_tokenizer_is_unicode_aware() -> None:
    cjk = "Kafka \u5c06\u8bb0\u5f55\u5b58\u50a8\u5728\u5206\u533a\u4e2d\u3002"
    assert content_tokens("Die Gr\u00f6\u00dfe") == ["die", "gr\u00f6sse"]
    assert content_tokens(cjk) == ["kafka", cjk[6:-1]]


def test_a_list_marker_is_not_evidence_and_does_not_block_attribution() -> None:
    found = _attribute("2) Kafka stores records in partitions.")
    assert [(i.label, i.sentence) for i in found.inferred] == [
        (1, "Kafka stores records in partitions.")
    ]


def test_a_code_block_is_attributed_only_when_whole_and_exact() -> None:
    """Code is covered only as a WHOLE fenced block or WHOLE inline span of the page, exactly.
    (Earlier builds matched prose lines and prompts token-wise; four review rounds found
    bypasses, so excerpts, prompts and re-spacing no longer count.)"""
    cli = hit_to_state(
        make_hit(
            1,
            title="Kafka CLI",
            text="Create one with:\n```bash\nkafka-topics --create --topic x\n```\n"
            "Or run `kafka-topics --list`.",
        )
    )
    quoted = "```\nkafka-topics --create --topic x\n```"
    found = attribute(quoted, [cli], [1], VALID)
    assert [(i.label, i.sentence) for i in found.inferred] == [(1, quoted)]
    span = "```\nkafka-topics --list\n```"  # equals a whole inline code span of the page
    assert [i.label for i in attribute(span, [cli], [1], VALID).inferred] == [1]
    for altered in (
        "```\nkafka-topics --topic --create x\n```",  # every word on the page, command is not
        "```\nkafka-topics  --create --topic x\n```",  # re-spaced
        "```\n$ kafka-topics --list\n```",  # prompt added
    ):
        assert attribute(altered, [cli], [1], VALID).uncited == (altered,)


@pytest.mark.parametrize(
    "block",
    [
        "```\nsudo rm -rf /\n```",  # a token-aligned PREFIX of the source line (round 3, P2)
        "```\nrm -rf /tmp/cache\n```",  # a suffix: "sudo" dropped
        "```\nsudo rm -rf /tmp/cache --no-preserve-root\n```",  # an extension
        "```\nkafka-topics --list\nsudo rm -rf /tmp/cache\n```",  # the page's lines reordered
        "```\nsudo rm -rf /tmp\n```",  # truncated at a path separator
    ],
)
def test_code_quotes_only_whole_source_lines_in_order(block: str) -> None:
    """Review of the redesign, round 3 (P2): for code a prefix is a different command."""
    page = hit_to_state(
        make_hit(
            1,
            title="Ops",
            text="```\nsudo rm -rf /tmp/cache\nkafka-topics --list\n```\n# rm -rf /data",
        )
    )
    assert attribute(block, [page], [1], VALID).inferred == ()
    whole = "```\nsudo rm -rf /tmp/cache\nkafka-topics --list\n```"
    assert [i.label for i in attribute(whole, [page], [1], VALID).inferred] == [1]


def test_a_commented_out_source_line_never_attributes_the_live_command() -> None:
    page = hit_to_state(make_hit(1, title="Ops", text="# rm -rf /data --force"))
    assert attribute("```\nrm -rf /data --force\n```", [page], [1], VALID).inferred == ()


def test_attribution_is_cpu_cheap() -> None:
    """ADR-0015 latency rule: attribution of a long answer over k=8 sources is well under the
    cost of a single token from a local model (~20-50 ms)."""
    hits = [hit_to_state(make_hit(n, text=KAFKA["display_text"] * 20)) for n in range(1, 9)]
    answer = " ".join(f"Kafka stores records in partitions number {n}." for n in range(40))
    timings = []
    for _ in range(3):  # best of 3: a swap stall on a loaded laptop is not attribution cost
        start = time.perf_counter()
        attribute(answer, hits, range(1, 9), valid_labels(range(1, 9), 0))
        timings.append(time.perf_counter() - start)
    assert min(timings) < 0.05  # measured ~3 ms on an M1 Pro


# ---------------------------------------------------------------- settle (status rules)
A = "Kafka stores records in partitions. Each partition is an ordered, immutable log."
S1, S2 = "Kafka stores records in partitions.", "Each partition is an ordered, immutable log."
BOTH = Attribution((Inferred(1, S1), Inferred(1, S2)), (), False, False)


def _settle(
    attribution: Attribution = BOTH,
    *,
    verdict: Verification = Verification.verified,
    conclusive: bool = True,
    located: bool = True,
    claims: tuple[str, ...] = (),
    tools_ran: bool = False,
    answer: str = A,
) -> tuple[Verification, list[str], tuple[str, ...]]:
    out = settle(
        answer,
        verdict=verdict,
        conclusive=conclusive,
        located=located,
        verifier_claims=claims,
        valid=valid_labels([1, 2], 1 if tools_ran else 0),
        attribution=attribution,
        tools_ran=tools_ran,
    )
    return out.status, [i.sentence for i in out.inferred], out.unsupported_claims


def test_settle_verified_when_the_residue_is_empty() -> None:
    assert _settle() == (Verification.verified, [S1, S2], ())


def test_settle_drops_an_inferred_citation_the_verifier_flags_claim_level() -> None:
    status, kept, claims = _settle(verdict=Verification.partial, claims=("immutable log",))
    assert (status, kept) == (Verification.partial, [S1])
    assert claims == ("immutable log", S2)


def test_settle_downgrades_verified_when_residue_is_left() -> None:
    """The verifier missed a fabrication; the uncovered sentence is residue: `partial`."""
    fabricated = "Kafka was invented by NASA in 1850."
    answer = f"{S1} {fabricated}"
    found = Attribution((Inferred(1, S1),), (fabricated,), False, False)
    status, kept, claims = _settle(found, answer=answer)
    assert (status, kept, claims) == (Verification.partial, [S1], (fabricated,))


def test_settle_inconclusive_verifier_shows_no_citation_but_keeps_the_answer() -> None:
    """A malformed verifier proved nothing: no provenance is shown and every unit is
    highlighted, but a lexically supported answer is not thrown away for a formatting glitch
    (ADR-0015 rule 4a). Its badge is `unverified` ("shown, not checked")."""
    status, kept, claims = _settle(verdict=Verification.partial, conclusive=False)
    assert (status, kept) == (Verification.unverified, [])
    assert claims == (S1, S2)


def test_settle_partial_verdict_that_cannot_be_located_clears_no_sentence() -> None:
    """Review round 1, P1: "something is unsupported, I won't say where" must not leave a
    server-made citation on the sentence it meant."""
    status, kept, claims = _settle(verdict=Verification.partial, located=False)
    assert (status, kept) == (Verification.unverified, [])
    assert claims == (S1, S2)


def test_settle_located_partial_verdict_keeps_the_unflagged_citation() -> None:
    status, kept, _ = _settle(verdict=Verification.partial, claims=(S2,))
    assert (status, kept) == (Verification.partial, [S1])


def test_settle_unlocated_supported_verdict_is_never_verified() -> None:
    cited = "Kafka stores records in partitions [S1]."
    found = Attribution((), (), True, False)
    status, kept, _ = _settle(found, located=False, answer=cited)
    assert (status, kept) == (Verification.partial, [])


def test_settle_inconclusive_verifier_with_model_citations_stays_partial() -> None:
    cited = "Kafka stores records in partitions [S1]."
    found = Attribution((), (), True, False)
    status, _, _ = _settle(found, verdict=Verification.partial, conclusive=False, answer=cited)
    assert status is Verification.partial


def test_settle_unsupported_verdict_drops_every_inferred_citation() -> None:
    status, kept, claims = _settle(verdict=Verification.unsupported, claims=(S1,))
    assert (status, kept, claims) == (Verification.unsupported, [], (S1,))


def test_settle_everything_rejected_means_no_provenance() -> None:
    status, kept, _ = _settle(verdict=Verification.partial, claims=(S1, S2))
    assert (status, kept) == (Verification.unsupported, [])


def test_settle_a_pure_refusal_is_unverified_whatever_the_verdict() -> None:
    refusal = "Your notes don't cover this."
    found = _attribute(refusal)
    assert found.refusal_only
    for verdict in Verification:
        status, kept, claims = _settle(found, verdict=verdict, answer=refusal)
        assert (status, kept, claims) == (Verification.unverified, [], ())


def test_settle_tool_turns_are_exempt_from_the_no_provenance_rule() -> None:
    reply = "I will remember that you prefer Java examples."
    nothing = Attribution((), (reply,), False, False)
    status, _, claims = _settle(nothing, tools_ran=True, answer=reply)
    assert (status, claims) == (Verification.partial, (reply,))
    assert no_provenance(nothing, tools_ran=False)
    assert not no_provenance(nothing, tools_ran=True)
    cited = f"{reply[:-1]} [T1]."
    status, _, _ = _settle(Attribution((), (), True, False), tools_ran=True, answer=cited)
    assert status is Verification.verified


def test_inferred_labels_are_deduplicated_and_never_repeat_a_model_citation() -> None:
    inferred = [Inferred(2, "a"), Inferred(1, "b"), Inferred(2, "c"), Inferred(3, "d")]
    assert inferred_labels(inferred, model_cited=[3]) == [2, 1]


# ---------------------------------------------------------------- one label parser (round 2)
@pytest.mark.parametrize("marker", ["[S2-S9]", "[S2, S10000]", "[1, S2]"])
def test_a_malformed_label_does_not_hide_a_sentence_from_attribution(marker: str) -> None:
    """Review round 2, P1: a malformed label is not a label (citations' canonical parser), so
    the drifted sentence is scored, has no support, and stays residue."""
    drifted = f"Redis Cluster splits the keyspace into 999 hash slots {marker}."
    out = _attribute(f"Kafka stores records in partitions [S1]. {drifted}")
    assert out.inferred == ()
    assert out.uncited == (drifted,)


# ---------------------------------------------------------------- the verifier's view
def test_annotate_writes_inferred_pairings_and_maps_claims_back_verbatim() -> None:
    answer = "Kafka stores records in partitions. Redis has 16384 hash slots! Done"
    view = annotate(
        answer,
        (
            Inferred(1, "Kafka stores records in partitions."),
            Inferred(2, "Redis has 16384 hash slots!"),
        ),
    )
    assert view.text == (
        "Kafka stores records in partitions [S1]. Redis has 16384 hash slots [S2]! Done"
    )
    for quoted, expected in [
        ("Redis has 16384 hash slots [S2]!", "Redis has 16384 hash slots!"),
        (
            "Kafka stores records in partitions [S1]. Redis",
            "Kafka stores records in partitions. Redis",
        ),
        ("hash slots [S", "hash slots"),  # a quote cut inside a marker clamps to its start
        ("[S2]", ""),  # nothing but a marker names no client text
    ]:
        mapped = view.to_answer(quoted)
        assert mapped == expected
        assert mapped in answer  # always a verbatim substring of what the client holds


def test_annotate_puts_a_code_pairing_on_its_own_line() -> None:
    block = "```\nkafka-topics --list\n```"
    view = annotate(f"Run:\n{block}\nDone.", (Inferred(1, block),))
    assert view.text == f"Run:\n{block}\n[S1]\nDone."


def test_annotate_without_inferred_citations_is_the_answer() -> None:
    view = annotate("Kafka stores records in partitions [S1].", ())
    assert view.text == view.answer
    assert view.to_answer("partitions [S1]") == "partitions [S1]"


# ---------------------------------------------------------------- code is matched verbatim
_RM_SOURCE = "Clear the cache with:\n```\nrm -rf ./cache\n```"
_LOOP_SOURCE = (
    "Prune stale files:\n```python\nfor f in files:\n    if is_stale(f):\n        os.remove(f)\n```"
)


def _hit_with(text: str) -> dict[str, object]:
    return {"title": "Ops", "heading_path": [], "display_text": text, "external_url": None}


def _block(markdown: str) -> Code:
    """The one code element of `markdown`, as the CommonMark parse finds it."""
    (code,) = structure(markdown).code
    return code


@pytest.mark.parametrize(
    ("source", "answer_code"),
    [
        (_RM_SOURCE, "rm -rf . /cache"),  # whitespace inserted: deletes the current directory
        ("```\nrm -rf /tmp\n```", "rm -rf / tmp"),
        (_LOOP_SOURCE, "for f in files:\n    os.remove(f)"),  # guard line dropped
        (_LOOP_SOURCE, "for f in files:\nif is_stale(f):\nos.remove(f)"),  # indentation flattened
        (_LOOP_SOURCE, "if is_stale(f):\n    for f in files:\n        os.remove(f)"),  # reordered
    ],
)
def test_altered_code_never_borrows_provenance(source: str, answer_code: str) -> None:
    """Review of the redesign, P1 (third code bypass): tolerant code matching let destructive
    variants of a source command be badged verified. Code must match the page verbatim."""
    from ragforge_core.agent.attribution import _source, verbatim_source

    block = _block(f"```\n{answer_code}\n```")
    assert verbatim_source(block, [_source(1, _hit_with(source))]) is None  # type: ignore[arg-type]


def test_verbatim_code_still_attributes() -> None:
    from ragforge_core.agent.attribution import _source, verbatim_source

    src = _source(1, _hit_with(_LOOP_SOURCE))  # type: ignore[arg-type]
    exact = _block("```python\nfor f in files:\n    if is_stale(f):\n        os.remove(f)\n```")
    assert verbatim_source(exact, [src]) == 1
    indented = "```\n  for f in files:\n      if is_stale(f):\n          os.remove(f)\n```"
    assert verbatim_source(_block(indented), [src]) == 1  # uniform dedent only
    prompt = _block("```\n$ rm -rf ./cache\n```")  # a prompt changes the text: not the block
    assert verbatim_source(prompt, [_source(1, _hit_with(_RM_SOURCE))]) is None  # type: ignore[arg-type]
    # the same block nested in a quote or a list item is the same code element (parser content)
    for nested in (
        "> ```python\n> for f in files:\n>     if is_stale(f):\n>         os.remove(f)\n> ```",
        "- x\n\n  ```python\n  for f in files:\n      if is_stale(f):\n"
        "          os.remove(f)\n  ```",
        "    for f in files:\n        if is_stale(f):\n            os.remove(f)",
    ):
        assert verbatim_source(_block(nested), [src]) == 1, nested
    # a fence whose info string is not a language tag may carry prose: never covered
    tagged = _block("```NASA built Kafka\nfor f in files:\n```")
    assert verbatim_source(tagged, [_source(1, _hit_with("```\nfor f in files:\n```"))]) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------- Phase 2b: inline prose quotes
_PROSE_PAGE = (
    "Followers fall out of the ISR after replica.lag.time.max.ms, 30 seconds by default.\n\n"
    "The default policy is noeviction; allkeys-lru suits a cache.\n\n"
    "Clean with:\n```\nrm -rf /tmp/build/cache --dry-run\n```"
)


def _span(text: str) -> Code:
    (code,) = structure(f"Use `{text}` here.").code
    return code


@pytest.mark.parametrize("quoted", ["replica.lag.time.max.ms", "noeviction", "allkeys-lru"])
def test_an_inline_span_stated_only_in_prose_is_never_backed(quoted: str) -> None:
    """The prose-quote exception was REMOVED (owner decision after review round 9: eight rounds
    found a P1 in it). Code is backed only by a whole code element of the cited page."""
    from ragforge_core.agent.attribution import _source, code_matches

    assert not code_matches(_span(quoted), _source(1, _hit_with(_PROSE_PAGE)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "quoted",
    [
        "allkeys",  # stops inside "allkeys-lru"
        "replica.lag.time",  # stops inside the dotted name
        "lag.time.max",  # starts and stops inside it
        "rm -rf /tmp/build",  # trims a code-block command (and is not prose)
        "rm -rf /tmp/build/cache",  # drops --dry-run: code blocks keep the whole-element rule
        "noeviction allkeys-lru",  # words the page never states together
        "Noeviction",  # case differs
    ],
)
def test_an_inline_span_that_is_not_a_whole_token_prose_quote_stays_unbacked(
    quoted: str,
) -> None:
    from ragforge_core.agent.attribution import _source, code_matches

    assert not code_matches(_span(quoted), _source(1, _hit_with(_PROSE_PAGE)))  # type: ignore[arg-type]


def test_a_code_block_never_quotes_prose() -> None:
    from ragforge_core.agent.attribution import _source, code_matches

    block = _block("```\nnoeviction\n```")
    assert not code_matches(block, _source(1, _hit_with(_PROSE_PAGE)))  # type: ignore[arg-type]


def test_a_prose_quote_must_be_in_the_page_its_label_cites() -> None:
    """S1 holds `noeviction` as code; S2 does not. Citing S2 leaves the span unbacked."""
    hits = [
        hit_to_state(make_hit(1, title="Redis", text="The default policy is `noeviction`.")),
        hit_to_state(make_hit(2, title="Redis ops", text="Watch memory usage.")),
    ]
    valid = valid_labels([1, 2], 0)
    ok = attribute("The default is `noeviction` [S1].", hits, [1, 2], valid)
    wrong = attribute("The default is `noeviction` [S2].", hits, [1, 2], valid)
    assert ok.code_unbacked == ()
    assert wrong.code_unbacked == ("The default is `noeviction` [S2].",)


@pytest.mark.parametrize(
    "page",
    [
        "Clean the build cache with rm -rf /`var/cache/app` on the build box.",  # code span
        "Clean the build cache with rm -rf /[var/cache](http://x) on the build box.",  # link
        "Clean the build cache with rm -rf /https://example.com/x on the build box.",  # URL
        "Clean the build cache with rm -rf /![img](x.png) on the build box.",  # image
        "Clean the build cache with rm -rf /<b>var</b> on the build box.",  # inline HTML
        "Clean the build cache with rm -rf /**var/cache** on the build box.",  # emphasis
    ],
)
def test_a_prose_quote_never_ends_where_a_hidden_element_starts(page: str) -> None:
    """Review round 2, P1: dropped atoms joined the text around them, so "rm -rf /" looked
    like whole tokens of the page and was covered (even inferred: server-side provenance)."""
    from ragforge_core.agent.attribution import _source, code_matches

    assert not code_matches(_span("rm -rf /"), _source(1, _hit_with(page)))  # type: ignore[arg-type]
    hits = [hit_to_state(make_hit(1, title="Ops", text=page))]
    answer = "Clean the build cache with `rm -rf /` on the build box."
    for draft in (answer, answer[:-1] + " [S1]."):
        found = attribute(draft, hits, [1], valid_labels([1], 0))
        out = settle(
            draft,
            verdict=Verification.verified,
            conclusive=True,
            located=True,
            verifier_claims=(),
            valid=valid_labels([1], 0),
            attribution=found,
            tools_ran=False,
        )
        assert out.status is not Verification.verified, (page, draft)


@pytest.mark.parametrize(
    ("page", "quoted"),
    [
        ("Undo it with git reset --hard HEAD~1 first.", "git reset --hard HEAD"),
        ("Pin it: pip install requests==2.0 today.", "pip install requests"),
        ("Run rm -rf build* to clean.", "rm -rf build"),
        ("Use cache\u200bwarm to start.", "cache"),  # zero-width space continues the token
    ],
)
def test_a_prose_quote_cannot_trim_a_token(page: str, quoted: str) -> None:
    """Review round 2, P2: only whitespace and sentence punctuation end a quoted run."""
    from ragforge_core.agent.attribution import _source, code_matches

    assert not code_matches(_span(quoted), _source(1, _hit_with(page)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("page", "quoted"),
    [
        ('Set "noeviction" first.', "noeviction"),
        ("(see replica.lag.time.max.ms).", "replica.lag.time.max.ms"),
        ("Pick allkeys-lru: it suits caches.", "allkeys-lru"),
        ("It is noeviction", "noeviction"),
    ],
)
def test_prose_in_quotes_or_brackets_never_backs_code_either(page: str, quoted: str) -> None:
    from ragforge_core.agent.attribution import _source, code_matches

    assert not code_matches(_span(quoted), _source(1, _hit_with(page)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "heading",
    [
        "Cleanup <!-- dropdb_prod -->",
        'Cleanup <span title=" dropdb_prod ">x</span>',
        "Cleanup `dropdb_prod`",
    ],
)
def test_heading_markup_is_parsed_like_the_body(heading: str) -> None:
    """Review round 3, P2: raw heading Markdown counted as prose, so text a rendered page never
    shows (a comment, an attribute) could be quoted."""
    from ragforge_core.agent.attribution import _source, code_matches

    hit = {"title": "Ops", "heading_path": [heading], "display_text": "x", "external_url": None}
    assert not code_matches(_span("dropdb_prod"), _source(1, hit))  # type: ignore[arg-type]
    plain = {**hit, "heading_path": ["Run dropdb_prod only in staging"]}
    assert not code_matches(_span("dropdb_prod"), _source(1, plain))  # type: ignore[arg-type]


# ---------------------------------------------------------------- review round 7
@pytest.mark.parametrize(
    "page",
    [
        "| step | status |\n|---|---|\n| ~~rm -rf / \\| tee log~~ | removed |",  # struck cell, \|
        "~~&#114;&#109;~~ was the old way.",  # struck entities
        "Old way: <del> rmall </del> do not.",  # HTML-wrapped text
        "Old way: <span hidden> rmall </span> do not.",
        "Clean with rm\\\nall is the database volume.",  # a hard break joins two lines
        "Clean with rm\nall is the database volume.",  # a soft break too
    ],
)
def test_round7_contexts_never_read_as_prose(page: str) -> None:
    from ragforge_core.agent.attribution import _source, code_matches

    src = _source(1, _hit_with(page))  # type: ignore[arg-type]
    for quoted in ("rm", "rmall", "tee"):
        if quoted == "rm" and "&#114;" not in page and "rm -rf" not in page:
            continue
        assert not code_matches(_span(quoted), src), (page, quoted)


@pytest.mark.parametrize("quoted", ["rm -rf /", "git reset --hard HEAD~1", "slot = CRC16(key)"])
def test_a_multi_token_span_never_quotes_prose(quoted: str) -> None:
    """Review round 7: prose quotes are ONE token. Multi-word commands need a whole code element."""
    from ragforge_core.agent.attribution import _source, code_matches

    page = (
        "Never run rm -rf / here. Undo with git reset --hard HEAD~1 first. "
        "Use slot = CRC16(key) mod 16384."
    )
    assert not code_matches(_span(quoted), _source(1, _hit_with(page)))  # type: ignore[arg-type]
