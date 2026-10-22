"""Shared helpers for the OBS-STACK tests (no tests here; the file name only follows the module's
owned-path prefix). Two things live here because several test files need exactly the same answer:

- `selectors()`: a small PromQL selector extractor. Prometheus' parser is Go-only, so the
  always-on tests read metric names and label matchers with this, and the Docker tests separately
  prove every expression parses with the real `promtool`. Its own unit tests pin what it sees.
- `promtool()`: runs the pinned prom/prometheus image's promtool with the files streamed in on
  stdin as a tar (no bind mount: nothing from the host is shared, which also works on Rancher
  Desktop where only some host paths are visible to the VM) and with no network.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs

REPO_ROOT = Path(__file__).resolve().parents[3]
OBS_DIR = REPO_ROOT / "docker" / "observability"
PROMETHEUS_YML = OBS_DIR / "prometheus.yml"
RULES_YML = OBS_DIR / "rules" / "ragforge.rules.yml"
RULES_TEST_YML = OBS_DIR / "rules" / "tests" / "ragforge.rules.test.yml"
DASHBOARD_JSON = OBS_DIR / "grafana" / "dashboards" / "ragforge.json"
PROVISIONING = OBS_DIR / "grafana" / "provisioning"
COMPOSE_SNIPPET = OBS_DIR / "compose.obs.snippet.yaml"
COMPOSE_YAML = REPO_ROOT / "compose.yaml"
OBS_SERVICES = frozenset({"prometheus", "grafana"})

# Values Grafana substitutes before a query reaches Prometheus. Any plausible value works for the
# checks: only names, labels and syntax are inspected.
GRAFANA_BUILTINS = {"__range": "1d", "__rate_interval": "1m", "__interval": "15s"}

_IDENT = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_NUMBER_OR_DURATION = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:[a-z]+\d*)*")
_MATCHER = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=~|!~|!=|=)\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)')
_AGGREGATIONS = {
    "sum", "min", "max", "avg", "group", "stddev", "stdvar", "count", "count_values",
    "bottomk", "topk", "quantile", "limitk", "limit_ratio",
}  # fmt: skip
_GROUPING = {"by", "without", "on", "ignoring", "group_left", "group_right"}
_KEYWORDS = {"and", "or", "unless", "offset", "bool", "inf", "nan", "atan2"}


@dataclass(frozen=True, slots=True)
class Matcher:
    label: str
    op: str
    value: str


@dataclass(frozen=True, slots=True)
class Selector:
    """One vector selector: `metric{label op "value", ...}`, with the functions and aggregations
    that enclose it, outermost first (`calls`), so a caller can check that e.g. rate() is only
    ever applied to a counter."""

    metric: str
    matchers: tuple[Matcher, ...]
    calls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Parsed:
    selectors: tuple[Selector, ...]
    grouping_labels: frozenset[str]  # every label named in by/without/on/ignoring/group_*


def _closing(text: str, start: int, opener: str, closer: str) -> int:
    """Index just past the bracket that closes text[start] (strings skipped)."""
    depth, i = 0, start
    while i < len(text):
        ch = text[i]
        if ch == '"':
            i = _end_of_string(text, i)
            continue
        depth += (ch == opener) - (ch == closer)
        i += 1
        if depth == 0:
            return i
    raise ValueError(f"unbalanced {opener!r} at {start} in {text!r}")


def _end_of_string(text: str, start: int) -> int:
    i = start + 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _matchers(body: str) -> tuple[Matcher, ...]:
    found, pos = [], 0
    while body[pos:].strip():
        m = _MATCHER.match(body, pos)
        if m is None:
            raise ValueError(f"cannot read label matchers {body!r}")
        found.append(Matcher(m.group(1), m.group(2), m.group(3)))
        pos = m.end()
    return tuple(found)


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def parse(expr: str) -> Parsed:
    """Extract the vector selectors and grouping labels of a PromQL expression.

    Deliberately strict: anything it cannot classify raises, so a construct this reader does not
    understand fails the test loudly instead of hiding a metric name.
    """
    selectors: list[Selector] = []
    grouping: set[str] = set()
    # One entry per open "(": the call it belongs to, or None for a plain grouping paren.
    # `pending` is the function/aggregation named just before its "(" (an aggregation's
    # `by (...)` clause may sit in between; it is consumed whole below).
    open_calls: list[str | None] = []
    pending: str | None = None
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            open_calls.append(pending)
            pending, i = None, i + 1
        elif ch == ")":
            if not open_calls:
                raise ValueError(f"unbalanced ')' at {i} in {expr!r}")
            open_calls.pop()
            i += 1
        elif ch.isspace() or ch in "+-*/^%<>=!,":
            i += 1
        elif ch == "#":  # a PromQL comment runs to the end of the line
            newline = expr.find("\n", i)
            i = len(expr) if newline == -1 else newline
        elif ch == '"':
            i = _end_of_string(expr, i)
        elif ch == "[":  # range or subquery: durations only
            i = _closing(expr, i, "[", "]")
        elif ch == "{":
            raise ValueError(f"selector without a metric name in {expr!r}")
        elif ch.isdigit():
            m = _NUMBER_OR_DURATION.match(expr, i)
            assert m is not None
            i = m.end()
        elif (ident := _IDENT.match(expr, i)) is not None:
            name, i = ident.group(), ident.end()
            nxt = _skip_space(expr, i)
            if name in _GROUPING:
                end = _closing(expr, nxt, "(", ")")
                grouping.update(_IDENT.findall(expr[nxt + 1 : end - 1]))
                i = end
            elif name in _KEYWORDS:
                continue
            elif name in _AGGREGATIONS or (nxt < len(expr) and expr[nxt] == "("):
                pending = name  # aggregation (`sum by (x) (...)`) or function call
            else:
                calls = tuple(call for call in open_calls if call is not None)
                body: tuple[Matcher, ...] = ()
                if nxt < len(expr) and expr[nxt] == "{":
                    end = _closing(expr, nxt, "{", "}")
                    body, i = _matchers(expr[nxt + 1 : end - 1]), end
                selectors.append(Selector(name, body, calls))
        else:
            raise ValueError(f"unexpected {ch!r} at {i} in {expr!r}")
    if open_calls:
        raise ValueError(f"unclosed '(' in {expr!r}")
    return Parsed(tuple(selectors), frozenset(grouping))


def expand(expr: str, variables: Mapping[str, str]) -> str:
    """Substitute Grafana `$var` / `${var}` the way Grafana does before querying."""

    def value(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name not in variables:
            raise KeyError(f"undefined dashboard variable ${name} in {expr!r}")
        return variables[name]

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", value, expr)


# --------------------------------------------------------------------------- loading


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dashboard() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))
    return loaded


def alert_rules() -> list[dict[str, Any]]:
    return [rule for group in load_yaml(RULES_YML)["groups"] for rule in group["rules"]]


def panels(board: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Every panel, including any nested inside a collapsed row."""
    for panel in board["panels"]:
        yield panel
        yield from panel.get("panels", [])


def variable_values(board: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each custom dashboard variable's selectable values."""
    return {
        var["name"]: [v.strip() for v in var["query"].split(",")]
        for var in board["templating"]["list"]
    }


def dashboard_queries(board: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(panel title, raw expr) for every Prometheus query on the board."""
    return [
        (panel["title"], target["expr"])
        for panel in panels(board)
        for target in panel.get("targets", [])
    ]


def expanded_queries(board: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Dashboard queries with every variable set to its first value (for the real parser)."""
    variables = {name: values[0] for name, values in variable_values(board).items()}
    variables.update(GRAFANA_BUILTINS)
    return [(title, expand(expr, variables)) for title, expr in dashboard_queries(board)]


def merged_into_compose_yaml() -> bool:
    """True once the integrator has pasted the snippet's services into compose.yaml."""
    return set(load_yaml(COMPOSE_YAML)["services"]) >= OBS_SERVICES


def obs_compose() -> dict[str, Any]:
    """`{"services", "volumes"}` of the obs stack, from the file that actually runs it:
    compose.yaml once merged (so the tests guard the real stack), the snippet until then."""
    if not merged_into_compose_yaml():
        snippet: dict[str, Any] = load_yaml(COMPOSE_SNIPPET)
        return snippet
    compose = load_yaml(COMPOSE_YAML)
    services = {name: compose["services"][name] for name in OBS_SERVICES}
    return {"services": services, "volumes": compose.get("volumes") or {}}


def prometheus_image() -> str:
    """The image compose runs; the tests run promtool from exactly the same digest."""
    image: str = obs_compose()["services"]["prometheus"]["image"]
    return image


# --------------------------------------------------------------------------- promtool


def _tar(files: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def observability_files(rules_text: str | None = None) -> dict[str, bytes]:
    """The tree promtool needs, laid out as compose mounts it (rules optionally replaced)."""
    rules = rules_text if rules_text is not None else RULES_YML.read_text(encoding="utf-8")
    return {
        "prometheus.yml": PROMETHEUS_YML.read_bytes(),
        "rules/ragforge.rules.yml": rules.encode(),
        "rules/tests/ragforge.rules.test.yml": RULES_TEST_YML.read_bytes(),
    }


def _docker(docker: str, *args: str, stdin: bytes = b"", timeout: int = 60) -> tuple[int, str]:
    """(returncode, stdout+stderr) of one docker CLI call."""
    proc = subprocess.run(  # noqa: S603 - the resolved docker binary, fixed argv, no shell
        [docker, *args], input=stdin, capture_output=True, timeout=timeout
    )
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


@dataclass(frozen=True, slots=True)
class Promtool:
    """promtool from the pinned image, in a throwaway container with no network and no mounts."""

    docker: str
    image: str

    def run(self, files: Mapping[str, bytes], *args: str) -> tuple[int, str]:
        """`promtool <args>` in a working dir holding `files`: (returncode, output)."""
        script = 'mkdir -p /tmp/w && cd /tmp/w && tar -x && exec promtool "$@"'
        run = ["run", "-i", "--rm", "--network", "none", "--entrypoint", "sh", self.image]
        return _docker(self.docker, *run, "-c", script, "sh", *args, stdin=_tar(files), timeout=120)


def require_promtool() -> Promtool:
    """Skip cleanly when Docker is unusable; pull the pinned image once otherwise."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not installed")
    if _docker(docker, "info", timeout=30)[0] != 0:
        pytest.skip("docker daemon not reachable")
    image = prometheus_image()
    present = _docker(docker, "image", "inspect", image, timeout=30)[0] == 0
    if not present and _docker(docker, "pull", "-q", image, timeout=300)[0] != 0:
        pytest.skip(f"cannot pull {image} (offline?)")
    return Promtool(docker, image)
