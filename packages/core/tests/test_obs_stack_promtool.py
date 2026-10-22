"""The alert rules and the scrape config, checked by the real Prometheus tooling (ADR-0018 R2-4):
`promtool check config`, `check rules`, the rule unit tests (firing AND quiet cases), the real
parser over every dashboard query, and mutation checks proving the unit tests catch the design
mistakes they exist for. Runs the image compose pins, so the binary that evaluates the rules in
the stack is the one tested; skips cleanly without Docker."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs

from test_obs_stack_support import (
    RULES_TEST_YML,
    RULES_YML,
    Promtool,
    alert_rules,
    dashboard,
    expanded_queries,
    load_yaml,
    observability_files,
    require_promtool,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def promtool() -> Promtool:
    return require_promtool()


def test_scrape_config_is_valid_and_loads_only_the_rules_file(promtool: Promtool) -> None:
    rc, out = promtool.run(observability_files(), "check", "config", "prometheus.yml")

    assert rc == 0, out
    # rules/tests/*.yml must never be loaded as rules (the glob is not recursive).
    assert "SUCCESS: 1 rule files found" in out


def test_rules_file_is_valid(promtool: Promtool) -> None:
    rc, out = promtool.run(observability_files(), "check", "rules", "rules/ragforge.rules.yml")

    assert rc == 0, out
    assert f"SUCCESS: {len(alert_rules())} rules found" in out


def test_rule_unit_tests_pass(promtool: Promtool) -> None:
    rc, out = promtool.run(
        observability_files(), "test", "rules", "rules/tests/ragforge.rules.test.yml"
    )

    assert rc == 0, out


def test_every_alert_has_a_firing_and_a_quiet_case() -> None:
    firing: set[str] = set()
    quiet: set[str] = set()
    for case in load_yaml(RULES_TEST_YML)["tests"]:
        for check in case["alert_rule_test"]:
            (firing if check["exp_alerts"] else quiet).add(check["alertname"])
    alerts = {rule["alert"] for rule in alert_rules()}

    assert alerts - firing == set(), "alerts never shown firing"
    assert alerts - quiet == set(), "alerts never shown quiet"


def test_every_dashboard_query_parses_with_the_real_promql_parser(promtool: Promtool) -> None:
    # A rules file with one recording rule per dashboard query: `check rules` runs Prometheus'
    # own parser on each, after Grafana's variables are substituted.
    queries = expanded_queries(dashboard())
    rules = [{"record": f"obs_stack:dashboard:q{i}", "expr": e} for i, (_, e) in enumerate(queries)]
    files = {"dashboard.rules.yml": yaml.safe_dump({"groups": [{"name": "d", "rules": rules}]})}

    rc, out = promtool.run({k: v.encode() for k, v in files.items()}, "check", "rules", *files)

    assert rc == 0, out
    assert f"SUCCESS: {len(queries)} rules found" in out


# --------------------------------------------------------------------------- mutation checks

Mutation = Callable[[dict[str, Any]], None]


# the queue alert's "a READY job exists" guard, as YAML folds it into the expression
_READY_GUARD = (
    " and on (priority) max by (priority)"
    ' (ragforge_jobs_ready{priority=~"interactive|webhook"}) > 0'
)


def _expr(old: str, new: str) -> Mutation:
    def apply(rule: dict[str, Any]) -> None:
        assert old in rule["expr"], f"mutation target {old!r} not in {rule['alert']}"
        rule["expr"] = rule["expr"].replace(old, new)

    return apply


def _drop_for(rule: dict[str, Any]) -> None:
    del rule["for"]


def _for(duration: str) -> Mutation:
    def apply(rule: dict[str, Any]) -> None:
        assert "for" in rule, f"{rule['alert']} has no `for` to change"
        rule["for"] = duration

    return apply


MUTATIONS: list[tuple[str, str, Mutation]] = [
    # The one-user guards (ratio AND >= 3 bad events AND >= 20 events in 6 h), both sides.
    ("no volume guard", "RagforgeChatLatencyHostedHigh", _expr("[6h])) >= 20", "[6h])) >= 0")),
    ("volume guard one too high", "RagforgeChatLatencyHostedHigh",
     _expr("[6h])) >= 20", "[6h])) >= 21")),
    ("no slow-turn floor", "RagforgeChatLatencyHostedHigh", _expr(") >= 3", ") >= 0")),
    ("slow-turn floor of 2", "RagforgeChatLatencyHostedHigh", _expr(") >= 3", ") >= 2")),
    ("slow-turn floor one too high", "RagforgeChatLatencyHostedHigh", _expr(") >= 3", ") >= 4")),
    ("hosted alert on the unmet 5 s SLO edge", "RagforgeChatLatencyHostedHigh",
     _expr('le="8.0"', 'le="5.0"')),
    ("no 5xx floor", "RagforgeHttp5xxHigh", _expr("[6h])) >= 3", "[6h])) >= 0")),
    ("5xx floor one too high", "RagforgeHttp5xxHigh", _expr("[6h])) >= 3", "[6h])) >= 4")),
    ("5xx volume guard lowered", "RagforgeHttp5xxHigh", _expr("[6h])) >= 20", "[6h])) >= 5")),
    # Every R2-4 number is pinned from both sides by a boundary case at real volume.
    ("slow-share budget doubled", "RagforgeChatLatencyHostedHigh", _expr("> 0.05", "> 0.10")),
    ("slow-share budget tightened", "RagforgeChatLatencyHostedHigh", _expr("> 0.05", "> 0.03")),
    ("5xx budget raised", "RagforgeHttp5xxHigh", _expr("> 0.02", "> 0.03")),
    ("5xx budget tightened", "RagforgeHttp5xxHigh", _expr("> 0.02", "> 0.015")),
    ("verified floor lowered", "RagforgeVerifiedShareLow", _expr("< 0.5", "< 0.45")),
    ("verified floor raised", "RagforgeVerifiedShareLow", _expr("< 0.5", "< 0.55")),
    ("no drift grace", "RagforgeVerifiedShareLow", _drop_for),
    ("local SLO on the hosted edge", "RagforgeChatLatencyLocalHigh",
     _expr('le="12.0"', 'le="5.0"')),
    ("no zero numerator", "RagforgeVerifiedShareLow", _expr(" or vector(0)", "")),
    ("guard of 5 per window, not per hour", "RagforgeVerifiedShareLow", _expr(">= 30", ">= 5")),
    ("newborn series trusted on a fresh Prometheus", "RagforgeLlmQuotaExhausted",
     _expr("(up offset 10m)", "(up)")),
    ("increase() only (misses the first refusal)", "RagforgeLlmQuotaExhausted",
     _expr('outcome="quota_exhausted"} offset 10m', 'outcome="quota_exhausted"} offset 1s')),
    ("threshold too low", "RagforgeLlmRateLimited", _expr("> 5", "> 2")),
    ("threshold 5 counted as over 5", "RagforgeLlmRateLimited", _expr("> 5", ">= 5")),
    ("threshold one too high", "RagforgeLlmRateLimited", _expr("> 5", "> 6")),
    ("disconnects counted", "RagforgeHttp5xxHigh", _expr('{status_class!="disconnected"}', "")),
    ("no wake-from-sleep grace", "RagforgeSyncStale", _drop_for),
    ("backfill not exempt", "RagforgeInteractiveQueueStuck",
     _expr('priority=~"interactive|webhook"', 'priority=~".+"')),
    ("a job in retry backoff pages", "RagforgeInteractiveQueueStuck",
     _expr(_READY_GUARD, "")),
    ("absolute dead count", "RagforgeDeadJobsRising",
     _expr("min_over_time(max(ragforge_jobs_dead)[15m:1m])", "0")),
    ("two-point comparison (misses a death after a requeue)", "RagforgeDeadJobsRising",
     _expr("min_over_time(max(ragforge_jobs_dead)[15m:1m])", "max(ragforge_jobs_dead offset 15m)")),
    ("missing gauges blamed on a down API", "RagforgeDbGaugesMissing",
     _expr(' and on () (up{job="ragforge-api"} == 1)', "")),
    ("no grace for one failed scrape", "RagforgeDbGaugesMissing", _drop_for),
    ("only up == 0", "RagforgeWorkerTargetsMissing",
     _expr('absent(up{job="ragforge-worker"})', 'up{job="ragforge-worker"} == 0')),
    ("no grace for a worker restart", "RagforgeWorkerTargetsMissing", _drop_for),
    ("grace shorter than one restart", "RagforgeTargetDown", _for("30s")),
]  # fmt: skip


def _rules_text(alert: str | None = None, mutate: Mutation | None = None) -> str:
    """The rules file re-serialised, with one rule optionally mutated."""
    loaded = yaml.safe_load(RULES_YML.read_text(encoding="utf-8"))
    for group in loaded["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == alert and mutate is not None:
                mutate(rule)
    dumped: str = yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True)
    return dumped


def test_re_serialised_rules_still_pass(promtool: Promtool) -> None:
    # The control for the mutations: re-dumping the YAML alone must not break anything, so each
    # mutation's failure below is caused by the mutation.
    files = observability_files(_rules_text())

    rc, out = promtool.run(files, "test", "rules", "rules/tests/ragforge.rules.test.yml")

    assert rc == 0, out


@pytest.mark.parametrize(
    ("why", "alert", "mutate"), MUTATIONS, ids=[f"{a}: {w}" for w, a, _ in MUTATIONS]
)
def test_the_unit_tests_catch_a_broken_rule(
    promtool: Promtool, why: str, alert: str, mutate: Mutation
) -> None:
    files = observability_files(_rules_text(alert, mutate))

    rc, out = promtool.run(files, "test", "rules", "rules/tests/ragforge.rules.test.yml")

    assert rc != 0, f"the unit tests did not notice: {why}"
    assert f"alertname: {alert}" in out, out
