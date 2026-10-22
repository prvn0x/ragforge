"""Integrator gate for the DOT rule (ADR-0015 "Dotted names"): the two live false flags of
2026-10-19 go through the REAL chat path (check_citations strict + attribute + settle), cited,
and come back verified; an uncited merge attempt still flags."""

from __future__ import annotations

import pytest

from module_d1.d1_fakes import (
    COLLECTION,
    Harness,
    collect,
    make_harness,
    make_hit,
    of_type,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import AgentSettings
from ragforge_core.domain.models import RankedChunk, TurnEvent, TurnVerification, Verification

SPRING = make_hit(
    1,
    title="Spring Boot",
    text="To change a log level at runtime, make a POST request to "
    "/actuator/loggers/{logger.name} with a JSON body. Logs switch to DEBUG with no restart.",
)
KAFKA = make_hit(
    2,
    title="Kafka Producer Internals",
    text="Setting max.in.flight.requests.per.connection to 1 provides strictest ordering.",
)
LOGGERS = "You can make a POST request to /actuator/loggers/{logger.name} with a JSON body [S1]."
ORDERING = "Setting max.in.flight.requests.per.connection to 1 provides strictest ordering [S2]."


def _harness(*hits: RankedChunk, strict: bool = False) -> Harness:
    return make_harness(
        results=[list(hits or (SPRING, KAFKA))],
        settings=AgentSettings(heartbeat_interval_s=3600, strict_citations=strict),
    )


async def _verification(h: Harness) -> TurnVerification:
    events: list[TurnEvent] = await collect(
        h.runner.run_turn(principal(), None, [COLLECTION], "How do I change a log level?")
    )
    (ver,) = of_type(events, TurnVerification)
    return ver


@pytest.mark.parametrize("strict", [False, True])
async def test_both_live_sentences_are_verified_through_the_chat_path(strict: bool) -> None:
    h = _harness(strict=strict)
    h.script(say(f"{LOGGERS} {ORDERING}"), verdict())
    ver = await _verification(h)
    assert ver.status is Verification.verified
    assert ver.unsupported_claims == ()


async def test_a_dotted_run_the_notes_do_not_hold_still_cuts_and_flags() -> None:
    """ "fast.and.it" is in no shown source: the dot still ends a sentence, so the uncited
    second sentence riding on the cited first one is flagged."""
    h = _harness()
    rider = "Kafka was built by NASA in 1850 and is fast."
    attack = rider[:-1] + ".and.it" + " provides strictest ordering [S2]."
    h.script(say(attack), verdict())
    ver = await _verification(h)
    assert ver.status is not Verification.verified
    assert any("NASA" in c for c in ver.unsupported_claims)
