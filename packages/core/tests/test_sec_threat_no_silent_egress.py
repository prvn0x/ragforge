"""Egress that would happen without any URL in our code (ADR-0018 §Security 4, R2-8).

Three libraries can phone home on their own: LangSmith (installed with LangChain, and as a pytest
plugin) sends traces when its env says so, FlashRank downloads its model from the internet when
the model directory is missing, and onnxruntime (under FlashRank) links Microsoft's 1DS telemetry
client. All three are pinned off here, along with the compose-level facts SECURITY.md relies on:
no service gets tracing or proxy variables, and none mounts the Docker socket.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import os
import re
import subprocess
import sys
import types
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Final

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs
from langsmith import utils as langsmith_utils

import egress_guard
from ragforge_core.adapters.rerank.flashrank import (
    DEFAULT_MODEL,
    ORT_TELEMETRY_ENV,
    FlashRankReranker,
    RerankerUnavailable,
)
from test_sec_threat_egress_inventory import shipped_files

REPO: Final = Path(__file__).resolve().parents[3]
SOURCES: Final = sorted(REPO.glob("packages/*/src/**/*.py"))
# LangSmith reads LANGSMITH_<X> then LANGCHAIN_<X>; any of them in a service env is a finding.
TRACING_PREFIXES: Final = ("LANGSMITH_", "LANGCHAIN_")
PROXY_VARS: Final = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"})
# A proxy variable SET outside Python (Dockerfile ENV, a Makefile/shell export, compose): any case,
# httpx honours http_proxy too. Python code may name them (judge.py documents trust_env).
# Two shapes: `NAME=value` / `NAME: value`, and the legacy Dockerfile `ENV NAME value`.
_PROXY_SET: Final = re.compile(
    r"\b(?:https?|all)_proxy\b\s*[=:]|^\s*ENV\s+(?:https?|all)_proxy\s",
    re.IGNORECASE | re.MULTILINE,
)
# Where a Docker/Rancher Desktop socket lives. Mounting the socket itself, or ANY directory
# above it (`/var/run:/var/run`, `/run:/host/run`, `~/.rd:/rd`), hands a container the socket.
DOCKER_SOCKETS: Final = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "~/.rd/docker.sock",
    "~/.docker/run/docker.sock",
    "${HOME}/.rd/docker.sock",
    "$HOME/.rd/docker.sock",
    "${HOME}/.docker/run/docker.sock",
    "$HOME/.docker/run/docker.sock",
)


# ---------------------------------------------------------------- LangSmith


@pytest.fixture
def tracing_env(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> Iterator[None]:
    """A shell with tracing ON and a (synthetic) key. Every variable the pin touches is first
    recorded by monkeypatch, so undo restores the env exactly. langsmith's global switch is then
    put back as the session had it: pinned off again when the egress_guard plugin is active
    (its session-wide pin must survive this test), else "follow the env"."""
    for name in (*egress_guard.TRACING_OFF_ENV, *egress_guard.TRACING_CREDENTIAL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-synthetic-not-a-key")
    egress_guard.reset_langsmith(enabled=None)
    yield
    monkeypatch.undo()
    if pytestconfig.pluginmanager.has_plugin("egress_guard"):
        egress_guard.pin_langsmith_off()
    else:
        egress_guard.reset_langsmith(enabled=None)


def test_pin_langsmith_off_beats_a_shell_with_tracing_on(tracing_env: None) -> None:
    assert langsmith_utils.tracing_is_enabled() is True  # precondition: the env really is on
    egress_guard.pin_langsmith_off()
    assert langsmith_utils.tracing_is_enabled() is False
    for name in egress_guard.TRACING_CREDENTIAL_ENV:
        assert name not in os.environ, name


def test_langchain_sees_tracing_off_after_the_pin(tracing_env: None) -> None:
    """LangChain asks langsmith (langchain_core.tracers.context), so the pin covers the agent's
    LangGraph/LangChain calls too, not only direct langsmith use."""
    context = importlib.import_module("langchain_core.tracers.context")
    egress_guard.pin_langsmith_off()
    assert context._tracing_v2_is_enabled() is False


def _shipped_texts() -> Iterator[tuple[str, str]]:
    """(repo-relative path, text) of everything the inventory scans: packages/*/src, scripts,
    docker (the Dockerfile's ENV reaches every container), compose*.yaml and the Makefile (its
    exports reach `make ask` and the CLIs)."""
    for path in shipped_files():
        try:
            yield path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def tracing_offenders(files: Iterable[tuple[str, str]]) -> list[str]:
    """Files that name a LangSmith/LangChain tracing variable or use langsmith itself."""
    words = (*TRACING_PREFIXES, "langsmith")
    return [where for where, text in files if any(word in text for word in words)]


def proxy_offenders(files: Iterable[tuple[str, str]]) -> list[str]:
    """Non-Python files that SET a proxy variable (Python may name one: judge.py documents it)."""
    return [where for where, text in files if not where.endswith(".py") and _PROXY_SET.search(text)]


def test_shipped_files_never_turn_tracing_on() -> None:
    """Nothing shipped names a LangSmith/LangChain tracing variable or uses langsmith itself,
    so tracing can only start from the owner's own shell, which the tests pin off."""
    assert tracing_offenders(_shipped_texts()) == []


def test_shipped_files_outside_python_never_set_a_proxy() -> None:
    """A proxy set in the Dockerfile, Makefile, a script or compose would route Notion/Gemini
    traffic (tokens, note text) through a third party."""
    assert proxy_offenders(_shipped_texts()) == []


def test_the_tracing_and_proxy_scans_have_teeth() -> None:
    """The two scans above pass vacuously on a clean tree; this proves they would catch the
    shapes that matter, in the files the review named (Dockerfile ENV, Makefile export)."""
    assert {"docker/Dockerfile", "Makefile", "scripts/bootstrap.sh"} <= dict(
        _shipped_texts()
    ).keys()
    planted = [
        ("docker/Dockerfile", "ENV LANGCHAIN_TRACING_V2=true\nENV HTTPS_PROXY=http://p:3128"),
        ("Makefile", "export LANGSMITH_API_KEY\nexport http_proxy=x"),
        ("docker/legacy.Dockerfile", "FROM x\n  ENV HTTPS_PROXY http://p:3128\n"),
        ("compose.obs.yaml", "    environment:\n      ALL_PROXY: x"),
        ("packages/x/src/x/judge.py", "# never honours HTTP_PROXY / ALL_PROXY"),
        ("scripts/x.sh", "uv run ragforge-ask"),
    ]
    assert tracing_offenders(planted) == ["docker/Dockerfile", "Makefile"]
    assert proxy_offenders(planted) == [
        "docker/Dockerfile", "Makefile", "docker/legacy.Dockerfile", "compose.obs.yaml",
    ]  # fmt: skip


# ---------------------------------------------------------------- compose


def _services(root: Path = REPO) -> list[tuple[str, dict[str, Any]]]:
    """("file:service", definition) for every service of every compose*.yaml, each file on its
    own: merging them by name would let a later file's `api` hide what an earlier one sets."""
    services: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("compose*.yaml")):
        document = yaml.safe_load(path.read_text()) or {}
        for name, service in (document.get("services") or {}).items():
            services.append((f"{path.name}:{name}", service or {}))
    return services


def test_a_later_compose_file_cannot_hide_an_earlier_service(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n  api:\n    environment: {LANGSMITH_TRACING: 'true'}\n"
    )
    (tmp_path / "compose.obs.yaml").write_text("services:\n  api:\n    image: x\n")
    found = dict(_services(tmp_path))
    assert set(found) == {"compose.yaml:api", "compose.obs.yaml:api"}
    assert "LANGSMITH_TRACING" in _env_names(found["compose.yaml:api"])


def _env_names(service: dict[str, Any]) -> set[str]:
    """Names in `environment:` (mapping or list form; YAML merge keys are already resolved)."""
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k) for k in env}
    return {str(item).split("=", 1)[0] for item in env}


def test_compose_services_get_no_tracing_or_proxy_env() -> None:
    services = _services()
    assert {"compose.yaml:api", "compose.yaml:worker"} <= {n for n, _ in services}  # real file
    for name, service in services:
        env = _env_names(service)
        assert not {e for e in env if e.upper().startswith(TRACING_PREFIXES)}, name
        assert not {e for e in env if e.upper() in PROXY_VARS}, name
        assert "env_file" not in service, name  # env_file would also hand over all of .env


def _mount_sources(service: dict[str, Any]) -> list[str]:
    """The host side of every volume: short form `src:dst[:mode]` or long form `source:`."""
    sources = []
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            sources.append(str(volume.get("source", "")))
        else:
            sources.append(str(volume).split(":", 1)[0])
    return sources


def exposes_docker_socket(source: str) -> bool:
    """A mount source that IS a known socket path or a directory above one (`/var/run`, `/run`,
    `~/.rd`, even `/`), or names `docker.sock` anywhere. Named volumes (`pgdata`) never match."""
    if "docker.sock" in source:
        return True
    root = source.rstrip("/") or "/"
    if root in ("/", "~", "${HOME}", "$HOME"):
        return True
    return any(sock.startswith(root + "/") for sock in DOCKER_SOCKETS)


def socket_offenders(services: Iterable[tuple[str, dict[str, Any]]]) -> list[str]:
    return [
        name
        for name, service in services
        if any(map(exposes_docker_socket, _mount_sources(service)))
    ]


def test_no_compose_service_mounts_the_docker_socket() -> None:
    """Whoever holds the socket can `docker inspect` every container and read its env secrets
    (SECURITY.md §Docker): no service of ours may hold it. Trivy scans tarballs for that reason."""
    assert socket_offenders(_services()) == []


def test_the_socket_scan_catches_parent_directory_mounts(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n"
        "  a: {volumes: ['/var/run:/var/run']}\n"
        "  b: {volumes: ['/run/:/host/run:ro']}\n"
        "  c: {volumes: [{type: bind, source: ~/.rd, target: /rd}]}\n"
        "  d: {volumes: ['/var/run/docker.sock:/var/run/docker.sock']}\n"
        "  e: {volumes: ['${HOME}/.docker:/d']}\n"
        "  f: {volumes: ['/:/host']}\n"
        "  ok: {volumes: ['pgdata:/var/lib/postgresql/data', './docker/init:/init:ro',"
        " '/var/lib/x:/x']}\n"
    )
    assert socket_offenders(_services(tmp_path)) == [
        "compose.yaml:a", "compose.yaml:b", "compose.yaml:c", "compose.yaml:d",
        "compose.yaml:e", "compose.yaml:f",
    ]  # fmt: skip


# ---------------------------------------------------------------- FlashRank


def test_reranker_never_downloads_by_default() -> None:
    default = inspect.signature(FlashRankReranker).parameters["allow_download"].default
    assert default is False


def _calls_passing_allow_download() -> Iterator[tuple[str, ast.expr]]:
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "allow_download":
                        yield str(path.relative_to(REPO)), keyword.value


def test_no_shipped_code_enables_the_download() -> None:
    """Bootstrap builds the reranker without the flag (baked model, read-only image); any call
    that passes it must pass a literal False, never a setting that could flip at runtime."""
    passed = list(_calls_passing_allow_download())
    for where, value in passed:
        assert isinstance(value, ast.Constant), where
        assert value.value is False, where


def test_missing_model_fails_closed_without_touching_flashrank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an empty cache dir the adapter refuses BEFORE importing flashrank, whose Ranker
    would otherwise fetch the model from the internet. Run under the egress guard as well."""
    imported: list[str] = []
    real_import = importlib.import_module

    def spy(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    reranker = FlashRankReranker(cache_dir=tmp_path, token_counter=len)
    try:
        with egress_guard.guarded() as guard, pytest.raises(RerankerUnavailable):
            asyncio.run(reranker.load())
    finally:
        reranker.close()
    assert "flashrank" not in imported
    assert guard.refused == []


# ---------------------------------------------------------------- onnxruntime telemetry


def _fake_flashrank_stack(
    monkeypatch: pytest.MonkeyPatch, *, with_switch: bool = True
) -> list[str]:
    """Stand-ins for onnxruntime and flashrank that record the order of events: whether the
    telemetry switch ran, and what the environment said when the Ranker (which creates the
    first InferenceSession) was built."""
    events: list[str] = []
    onnxruntime = types.ModuleType("onnxruntime")
    if with_switch:
        onnxruntime.disable_telemetry_events = lambda: events.append("disable")  # type: ignore[attr-defined]
    flashrank = types.ModuleType("flashrank")

    class Ranker:
        def __init__(self, **_: object) -> None:
            events.append(f"session {ORT_TELEMETRY_ENV}={os.environ.get(ORT_TELEMETRY_ENV)}")

        def rerank(self, request: object) -> list[dict[str, Any]]:
            return []

    flashrank.Ranker = Ranker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    monkeypatch.setitem(sys.modules, "flashrank", flashrank)
    monkeypatch.setenv(ORT_TELEMETRY_ENV, "0")  # an environment that says "on"
    return events


def _load_with_a_baked_model(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_MODEL).mkdir()
    reranker = FlashRankReranker(cache_dir=tmp_path, token_counter=len)
    try:
        asyncio.run(reranker.load())
    finally:
        reranker.close()


def test_the_reranker_turns_onnxruntime_telemetry_off_before_the_first_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _fake_flashrank_stack(monkeypatch)
    _load_with_a_baked_model(tmp_path)
    assert events == ["disable", f"session {ORT_TELEMETRY_ENV}=1"]


def test_an_onnxruntime_without_the_switch_still_loads_with_the_env_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _fake_flashrank_stack(monkeypatch, with_switch=False)
    _load_with_a_baked_model(tmp_path)
    assert events == [f"session {ORT_TELEMETRY_ENV}=1"]


def test_importing_the_adapter_forces_the_env_switch_on() -> None:
    """The env switch must be in place before ANY import of onnxruntime in the process, not only
    the adapter's own: a fresh interpreter whose environment says "0" reads "1" after import."""
    code = (
        "import os, ragforge_core.adapters.rerank.flashrank as f; "
        "print(os.environ[f.ORT_TELEMETRY_ENV])"
    )
    env = {**os.environ, ORT_TELEMETRY_ENV: "0"}
    out = subprocess.run(  # noqa: S603 - this interpreter, a fixed argv, no shell
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip() == "1"


def test_the_locked_onnxruntime_has_both_switches() -> None:
    """The adapter tolerates a missing function (so a future onnxruntime cannot crash startup);
    this pins that the LOCKED one really has it, and that its native library reads the env
    switch, so neither safeguard is silently a no-op today."""
    onnxruntime = importlib.import_module("onnxruntime")
    assert callable(getattr(onnxruntime, "disable_telemetry_events", None))
    assert onnxruntime.__file__ is not None
    capi = Path(onnxruntime.__file__).parent / "capi"
    natives = [p for p in capi.iterdir() if p.name.startswith("libonnxruntime.")]
    assert natives, f"no libonnxruntime under {capi}"
    assert all(ORT_TELEMETRY_ENV.encode() in p.read_bytes() for p in natives)


def test_the_image_sets_the_env_switch_for_every_entrypoint() -> None:
    dockerfile = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    runtime = dockerfile.split("AS runtime", 1)[1].split("\nFROM ", 1)[0]
    assert re.search(rf"^\s*{ORT_TELEMETRY_ENV}=1\s*$", runtime, re.MULTILINE), runtime
