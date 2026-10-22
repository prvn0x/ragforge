"""Every metric the dashboard and the alert rules query is in the LOCKED catalogue, with labels that
metric really has and label values it can really take (ADR-0018 R2-1). This is the seam test: a
renamed metric, a typo'd label or a value the instrumentation never emits would otherwise show up
only as an empty panel or an alert that can never fire."""

from __future__ import annotations

import pytest

from ragforge_core.config import LLMProvider
from ragforge_core.domain.models import JobPriority
from ragforge_core.observability import catalog
from ragforge_core.observability.catalog import Metric
from test_obs_stack_support import (
    PROMETHEUS_YML,
    Matcher,
    Selector,
    alert_rules,
    dashboard,
    dashboard_queries,
    expand,
    expanded_queries,
    load_yaml,
    parse,
    variable_values,
)

HISTOGRAM_SUFFIXES = ("_bucket", "_count", "_sum")
TARGET_LABELS = {"job", "instance"}  # attached by Prometheus to every scraped series
BUILTIN_SERIES = {"up"}  # Prometheus' own per-target series; the ONLY non-catalogue name allowed

# The catalogue locks the allowed values but not which tuple belongs to which label; this is that
# map. A label absent here (route, model, prompt, kind, ...) is bounded by the instrumentation and
# has no closed set to check against.
VALUES_BY_LABEL: dict[str, tuple[str, ...]] = {
    "surface": catalog.SURFACES,
    "status_class": catalog.STATUS_CLASSES,
    "reason": catalog.CLAIM_REASONS,
    "stage": catalog.RETRIEVAL_STAGES,
    "provider": tuple(p.value for p in LLMProvider),
    "priority": tuple(p.name for p in JobPriority),
}
VALUES_BY_METRIC_LABEL: dict[tuple[str, str], tuple[str, ...]] = {
    ("ragforge_chat_turns_total", "outcome"): catalog.TURN_OUTCOMES,
    ("ragforge_llm_requests_total", "outcome"): catalog.LLM_OUTCOMES,
    ("ragforge_llm_request_duration_seconds", "outcome"): catalog.LLM_OUTCOMES,
    ("ragforge_embed_requests_total", "outcome"): catalog.EMBED_OUTCOMES,
    ("ragforge_jobs_total", "outcome"): catalog.JOB_OUTCOMES,
    ("ragforge_verification_total", "status"): catalog.VERIFICATION_STATUSES,
    ("ragforge_notion_requests_total", "status"): catalog.NOTION_STATUS,
}


def _job_names() -> tuple[str, ...]:
    return tuple(job["job_name"] for job in load_yaml(PROMETHEUS_YML)["scrape_configs"])


def resolve(series: str) -> tuple[Metric | None, str]:
    """(catalogue metric, suffix) for an exposed series name; (None, "") if unknown."""
    metrics = catalog.by_name()
    if series in metrics:
        return metrics[series], ""
    for suffix in HISTOGRAM_SUFFIXES:
        base = series.removesuffix(suffix)
        if base != series and base in metrics and metrics[base].kind == "histogram":
            return metrics[base], suffix
    return None, ""


def _literal_values(m: Matcher) -> list[str]:
    """The values a matcher names; regex matchers must be plain `a|b` alternations."""
    if m.op in ("=", "!="):
        return [m.value]
    literal = bool(m.value) and all(c.isalnum() or c in "_|-" for c in m.value)
    assert literal, f"{m.label}{m.op}{m.value!r}: only literal alternations can be checked"
    return m.value.split("|")


def selector_problems(sel: Selector, expanded: dict[str, list[str]] | None = None) -> list[str]:
    """Why `sel` does not match the catalogue (empty list = fine).

    `expanded` maps a dashboard variable to its selectable values, so `surface=~"$surface"` is
    checked against every value the dropdown can send.
    """
    if sel.metric in BUILTIN_SERIES:
        metric, suffix, labels = None, "", TARGET_LABELS
    else:
        metric, suffix = resolve(sel.metric)
        if metric is None:
            return [f"{sel.metric}: not in catalog.CATALOG"]
        labels = set(metric.labels) | TARGET_LABELS | ({"le"} if suffix == "_bucket" else set())
    problems: list[str] = []
    for m in sel.matchers:
        if m.label not in labels:
            problems.append(f"{sel.metric}: has no label {m.label!r}")
            continue
        values = (expanded or {}).get(m.value.lstrip("$")) if m.value.startswith("$") else None
        for value in values if values is not None else _literal_values(m):
            problems += _value_problems(sel.metric, metric, m.label, value)
    return problems


def _value_problems(series: str, metric: Metric | None, label: str, value: str) -> list[str]:
    if label == "le":
        assert metric is not None
        assert metric.buckets is not None
        edges = {str(float(edge)) for edge in metric.buckets}  # prometheus_client renders 5.0
        return [] if value in edges else [f"{series}: le={value!r} is not a bucket edge"]
    if label == "job":
        return [] if value in _job_names() else [f"{series}: no scrape job {value!r}"]
    allowed = VALUES_BY_METRIC_LABEL.get((metric.name, label)) if metric else None
    allowed = allowed or VALUES_BY_LABEL.get(label)
    if allowed is not None and value not in allowed:
        return [f"{series}: {label}={value!r} is not one of {allowed}"]
    return []


# Range functions whose result is only meaningful for one metric type. rate() on a gauge, or
# delta() on a counter, returns plausible-looking numbers that are wrong: no error ever shows.
COUNTER_FUNCTIONS = frozenset({"rate", "irate", "increase", "resets"})
GAUGE_FUNCTIONS = frozenset({"delta", "idelta", "deriv", "predict_linear", "holt_winters"})


def function_problems(sel: Selector) -> list[str]:
    """Why the functions around `sel` do not fit its metric type (empty list = fine).

    The INNERMOST call is the one applied to the selector's range; histogram_quantile must have
    `_bucket` series somewhere beneath it, so it is checked over the whole enclosing chain.
    """
    if sel.metric in BUILTIN_SERIES:
        kind, suffix = "gauge", ""
    else:
        metric, suffix = resolve(sel.metric)
        if metric is None:
            return []  # already reported by selector_problems
        kind = metric.kind
    counter_like = kind == "counter" or (kind == "histogram" and suffix != "")
    innermost = sel.calls[-1] if sel.calls else None
    problems: list[str] = []
    if innermost in COUNTER_FUNCTIONS and not counter_like:
        problems.append(f"{innermost}() over {sel.metric}, a {kind}: counters only")
    if innermost in GAUGE_FUNCTIONS and kind != "gauge":
        problems.append(f"{innermost}() over {sel.metric}, a {kind}: gauges only")
    if "histogram_quantile" in sel.calls and suffix != "_bucket":
        problems.append(f"histogram_quantile over {sel.metric}: needs a histogram's _bucket")
    return problems


def grouping_problems(expr: str) -> list[str]:
    """Grouping labels (by/on/...) must exist on some metric of the expression: `sum by (provder)`
    silently collapses every series into one."""
    parsed = parse(expr)
    known = set(TARGET_LABELS) | {"le"}
    for sel in parsed.selectors:
        metric, _ = resolve(sel.metric)
        known |= set(metric.labels) if metric else set()
    return [f"grouping by unknown label {label!r}" for label in parsed.grouping_labels - known]


def all_problems(expr: str, expanded: dict[str, list[str]] | None = None) -> list[str]:
    parsed = parse(expr)
    assert parsed.selectors, f"no metric in {expr!r}"
    problems = [p for sel in parsed.selectors for p in selector_problems(sel, expanded)]
    problems += [p for sel in parsed.selectors for p in function_problems(sel)]
    return problems + grouping_problems(expr)


def _dashboard_exprs() -> list[tuple[str, str]]:
    """Raw dashboard queries. `$window`/`$__range` sit inside `[...]`, which the reader skips, and
    `$surface` is a matcher value resolved per dropdown choice, so nothing needs expanding."""
    return dashboard_queries(dashboard())


# --------------------------------------------------------------------------- the extractor


def test_parse_reads_names_matchers_and_grouping() -> None:
    parsed = parse(
        'sum by (le, provider) (rate(a_bucket{surface="chat", le="5.0"}[1h] offset 10m))'
        ' / on (job) group_left() b unless absent(up{job=~"x|y"}) > 0.05 and c{d!="e"} # f g'
    )

    assert parsed.selectors == (
        Selector(
            "a_bucket",
            (Matcher("surface", "=", "chat"), Matcher("le", "=", "5.0")),
            ("sum", "rate"),
        ),
        Selector("b", ()),
        Selector("up", (Matcher("job", "=~", "x|y"),), ("absent",)),
        Selector("c", (Matcher("d", "!=", "e"),)),
    )
    assert parsed.grouping_labels == {"le", "provider", "job"}


def test_parse_records_the_calls_around_each_selector() -> None:
    parsed = parse(
        "histogram_quantile(0.95, sum(rate(a_bucket[5m])) by (le))"
        " > (min_over_time(max(b)[15m:1m]) or vector(0)) + c"
    )

    assert [(s.metric, s.calls) for s in parsed.selectors] == [
        ("a_bucket", ("histogram_quantile", "sum", "rate")),
        ("b", ("min_over_time", "max")),
        ("c", ()),
    ]
    with pytest.raises(ValueError, match="unclosed"):
        parse("rate(x[5m]")
    with pytest.raises(ValueError, match="unbalanced"):
        parse("x)")


def test_parse_is_strict_about_what_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="without a metric name"):
        parse('{__name__="x"}')
    with pytest.raises(ValueError, match="cannot read label matchers"):
        parse("x{label=unquoted}")
    with pytest.raises(ValueError, match="unexpected '@'"):
        parse("x @ 123")


def test_every_dashboard_variable_is_defined() -> None:
    assert len(expanded_queries(dashboard())) == len(dashboard_queries(dashboard()))


def test_expand_fails_on_an_undefined_variable() -> None:
    assert expand("rate(x[$window]) ${a}", {"window": "1h", "a": "b"}) == "rate(x[1h]) b"
    with pytest.raises(KeyError):
        expand("rate(x[$typo])", {"window": "1h"})


@pytest.mark.parametrize(
    ("expr", "fragment"),
    [
        ("ragforge_chat_turns_totl", "not in catalog"),
        ('ragforge_chat_turns_total{tenant="t1"}', "no label 'tenant'"),
        ('ragforge_chat_turns_total{outcome="answerd"}', "'answerd' is not one of"),
        ('ragforge_llm_requests_total{outcome!="okay"}', "'okay' is not one of"),
        ('ragforge_chat_turn_duration_seconds_bucket{le="4.0"}', "not a bucket edge"),
        ('ragforge_chat_turns_total_bucket{le="5.0"}', "not in catalog"),  # counter: no buckets
        ('ragforge_jobs_ready{priority=~"interactive|urgent"}', "'urgent' is not one of"),
        ('up{job="ragforge-wrker"}', "no scrape job"),
        ("sum by (provder) (ragforge_chat_turns_total)", "unknown label 'provder'"),
        ("rate(ragforge_jobs_dead[1h])", "rate() over ragforge_jobs_dead, a gauge"),
        ("sum(increase(ragforge_build_info[1h]))", "increase() over ragforge_build_info"),
        ("delta(ragforge_jobs_total[1h])", "delta() over ragforge_jobs_total, a counter"),
        ("rate(ragforge_chat_turn_duration_seconds[5m])", "a histogram: counters only"),
        ("histogram_quantile(0.95, ragforge_chat_turn_duration_seconds_count)", "needs a"),
        ("histogram_quantile(0.9, sum by (le) (rate(ragforge_jobs_total[5m])))", "needs a"),
    ],
)
def test_the_checker_rejects_each_kind_of_drift(expr: str, fragment: str) -> None:
    assert any(fragment in problem for problem in all_problems(expr)), all_problems(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "rate(ragforge_jobs_total[5m])",
        "increase(ragforge_chat_turn_duration_seconds_count[1h])",
        "max(ragforge_jobs_dead) > min_over_time(max(ragforge_jobs_dead)[15m:1m])",
        "deriv(ragforge_jobs_ready[10m])",
        "histogram_quantile(0.9, sum by (le) (rate(ragforge_job_duration_seconds_bucket[5m])))",
    ],
)
def test_the_checker_accepts_functions_that_fit_the_type(expr: str) -> None:
    assert all_problems(expr) == []


def test_the_checker_resolves_dashboard_variables_to_their_choices() -> None:
    expr = 'ragforge_chat_turns_total{surface=~"$surface"}'

    assert all_problems(expr, {"surface": ["chat", "eval"]}) == []
    assert "'staff' is not one of" in all_problems(expr, {"surface": ["chat", "staff"]})[0]


# --------------------------------------------------------------------------- the artefacts


@pytest.mark.parametrize("rule", alert_rules(), ids=lambda rule: rule["alert"])
def test_every_alert_rule_queries_only_catalogue_metrics(rule: dict[str, str]) -> None:
    assert all_problems(rule["expr"]) == []


@pytest.mark.parametrize(("title", "expr"), list(_dashboard_exprs()), ids=lambda v: str(v)[:60])
def test_every_dashboard_query_uses_only_catalogue_metrics(title: str, expr: str) -> None:
    expanded = variable_values(dashboard())

    assert all_problems(expr, expanded) == [], title


def test_the_dashboard_covers_every_catalogue_metric() -> None:
    used = {
        (resolve(sel.metric)[0] or Metric(sel.metric, "gauge", "")).name
        for _, expr in _dashboard_exprs()
        for sel in parse(expr).selectors
    }

    assert {m.name for m in catalog.CATALOG} - used == set()


def test_owner_signals_never_include_eval_or_demo_by_default() -> None:
    surface = next(v for v in dashboard()["templating"]["list"] if v["name"] == "surface")

    assert surface["current"]["value"] == ["chat"]
    for rule in alert_rules():
        for sel in parse(rule["expr"]).selectors:
            metric, _ = resolve(sel.metric)
            if metric is not None and "surface" in metric.labels:
                surfaces = [m for m in sel.matchers if m.label == "surface"]
                assert surfaces == [Matcher("surface", "=", "chat")], rule["alert"]
