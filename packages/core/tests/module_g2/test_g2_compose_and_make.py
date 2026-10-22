"""Static guarantees on compose.yaml, the Makefile and .env.example: the Gemini key reaches only
the two services that call the model, and `make up` asks only on a first interactive run."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs
from g2_support import FAKE_KEY, REPO_ROOT, load_setup_provider

sp = load_setup_provider()

LLM_VARS = {
    "RAGFORGE_LLM_PROVIDER",
    "RAGFORGE_GEMINI_API_KEY",
    "RAGFORGE_GEMINI_MODEL",
    "RAGFORGE_GEMINI_THINKING_BUDGET",
}


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    # safe_load resolves anchors and `<<` merges the way compose does.
    loaded: dict[str, Any] = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text())
    return loaded


def _environment(service: dict[str, Any]) -> dict[str, Any]:
    env = service.get("environment") or {}
    if isinstance(env, list):  # the `- KEY=value` form
        return dict(item.split("=", 1) if "=" in item else (item, None) for item in env)
    assert isinstance(env, dict)
    return env


def test_only_api_and_worker_see_provider_settings(compose: dict[str, Any]) -> None:
    exposed = {
        name
        for name, service in compose["services"].items()
        if LLM_VARS & set(_environment(service))
    }

    assert exposed == {"api", "worker"}
    for name in ("api", "worker"):
        assert set(_environment(compose["services"][name])) >= LLM_VARS


def test_key_is_interpolated_from_env_not_hardcoded(compose: dict[str, Any]) -> None:
    for name in ("api", "worker"):
        value = _environment(compose["services"][name])["RAGFORGE_GEMINI_API_KEY"]
        assert value == "${RAGFORGE_GEMINI_API_KEY:-}"


def test_no_service_loads_the_whole_env_file(compose: dict[str, Any]) -> None:
    # env_file would hand a service every secret in .env, Gemini key included.
    for name, service in compose["services"].items():
        assert "env_file" not in service, name


def test_key_variable_is_referenced_in_exactly_one_place() -> None:
    # Catches the key being wired into another service by any YAML spelling the parse misses.
    text = (REPO_ROOT / "compose.yaml").read_text()
    assigned = [
        line
        for line in text.splitlines()
        if "RAGFORGE_GEMINI_API_KEY" in line and not line.lstrip().startswith("#")
    ]

    assert len(assigned) == 1
    assert compose_anchor_users(text, "llm-env") == 2


def compose_anchor_users(text: str, anchor: str) -> int:
    return len(re.findall(rf"\*{re.escape(anchor)}\b", text))


def test_env_example_holds_no_key_value() -> None:
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if line.startswith("RAGFORGE_GEMINI_API_KEY="):
            assert line == "RAGFORGE_GEMINI_API_KEY="


def test_env_example_leaves_every_provider_variable_unset() -> None:
    # `cp .env.example .env` must keep the first-run question: the `make up` gate asks only while
    # RAGFORGE_LLM_PROVIDER is absent, so no provider line may be an active assignment.
    values = sp.read_env_values((REPO_ROOT / ".env.example").read_text())

    assert not set(values) & LLM_VARS


def test_uncommenting_the_env_example_provider_lines_gives_a_config_that_starts(
    tmp_path: Path,
) -> None:
    # The inline notes on those lines must not become part of the values (compose drops ` # ...`),
    # and the result must pass the same check doctor runs, without any network call.
    example = (REPO_ROOT / ".env.example").read_text()
    provider_lines = [
        line[2:] for line in example.splitlines() if line[2:].partition("=")[0] in LLM_VARS
    ]
    env = tmp_path / ".env"
    env.write_text("\n".join(provider_lines) + "\n")

    values = sp.read_env_values(env.read_text())
    assert values == {
        "RAGFORGE_LLM_PROVIDER": "ollama",
        "RAGFORGE_GEMINI_API_KEY": "",
        "RAGFORGE_GEMINI_MODEL": sp.DEFAULT_GEMINI_MODEL,
        "RAGFORGE_GEMINI_THINKING_BUDGET": "0",
    }
    assert sp._config_problem(values["RAGFORGE_GEMINI_MODEL"], "0") == ""
    assert sp.main(["--check", "--env-file", str(env)], environ={}) == 0


def _recipe(target: str) -> str:
    makefile = (REPO_ROOT / "Makefile").read_text()
    match = re.search(rf"^{target}:.*\n((?:\t.*\n)+)", makefile, re.MULTILINE)
    assert match, target
    return match.group(1)


def test_make_up_runs_setup_only_through_the_if_unset_gate() -> None:
    recipe = _recipe("up")

    assert "scripts/setup_provider.py --if-unset" in recipe
    assert recipe.index("--if-unset") < recipe.index("up --build")


def test_make_setup_runs_the_interactive_menu() -> None:
    assert "python3 scripts/setup_provider.py" in _recipe("setup")


# ------------------------------------------------- the real compose binary (integration, read-only)
def _compose_config(env_file: Path) -> dict[str, Any]:
    """`docker compose config` resolves interpolation and merges without creating anything.

    The shell environment is reduced to PATH/HOME so no exported RAGFORGE_* leaks in, and every
    `${VAR}` compose.yaml references gets a fake value so required variables resolve.
    """
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not installed")
    env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG")
        if k in os.environ
    }
    result = subprocess.run(  # noqa: S603 - fixed binary, read-only `config` subcommand
        [
            docker,
            "compose",
            "--env-file",
            str(env_file),
            "-f",
            str(REPO_ROOT / "compose.yaml"),
            "-p",
            "ragforge-g2-config-probe",
            "--profile",
            "*",
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )
    if (
        result.returncode != 0
        and "compose" in result.stderr
        and "is not a docker command" in result.stderr
    ):
        pytest.skip("docker compose v2 not available")
    assert result.returncode == 0, result.stderr[-500:]
    loaded: dict[str, Any] = json.loads(result.stdout)
    return loaded


@pytest.mark.integration
def test_real_compose_gives_the_key_to_api_and_worker_only_and_reads_env_like_setup(
    tmp_path: Path,
) -> None:
    # The end-to-end proof the static tests approximate: compose itself resolves the four values
    # (written with export, quotes and inline comments) into api and worker and into nothing else.
    referenced = set(
        re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", (REPO_ROOT / "compose.yaml").read_text())
    )
    filler = "".join(f"{name}=fake-value-for-g2\n" for name in sorted(referenced - LLM_VARS))
    managed = (
        "RAGFORGE_LLM_PROVIDER=gemini   # inline note\n"
        f'export RAGFORGE_GEMINI_API_KEY="{FAKE_KEY}"  # personal\n'
        "RAGFORGE_GEMINI_MODEL=gemini-2.5-flash  # fast\n"
        "RAGFORGE_GEMINI_THINKING_BUDGET=0   # -1..32768; 0 = thinking off\n"
    )
    env_file = tmp_path / "probe.env"
    env_file.write_text(filler + managed)

    config = _compose_config(env_file)

    expected = sp.read_env_values(managed)
    for name, service in config["services"].items():
        resolved = {k: v for k, v in (service.get("environment") or {}).items() if k in LLM_VARS}
        if name in ("api", "worker"):
            assert resolved == expected, name
        else:
            assert resolved == {}, name
    # Nowhere else either (labels, commands, build args, healthchecks). Top-level `x-` extension
    # fields are YAML anchors only: compose never hands them to a container.
    assert sorted(_paths_holding(config, FAKE_KEY)) == [
        "services.api.environment.RAGFORGE_GEMINI_API_KEY",
        "services.worker.environment.RAGFORGE_GEMINI_API_KEY",
        "x-llm-env.RAGFORGE_GEMINI_API_KEY",
    ]


def _paths_holding(node: object, needle: str, path: str = "") -> list[str]:
    """Dotted paths of every string in the resolved config that contains `needle`."""
    if isinstance(node, dict):
        return [
            hit
            for key, value in node.items()
            for hit in _paths_holding(value, needle, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(node, list):
        return [hit for i, v in enumerate(node) for hit in _paths_holding(v, needle, f"{path}.{i}")]
    return [path] if isinstance(node, str) and needle in node else []
