"""SEC-SCAN: .gitignore and .dockerignore keep every sensitive path out of git and image builds.

ADR-0018 Security 3 + R2-8. The paths are checked with the real matching rules, not by grepping
for a line: `pathspec`'s GitIgnoreSpec implements git's wildmatch (negation, anchoring, `**`),
and docker-py's PatternMatcher implements the Docker CLI's .dockerignore rules, where `*.key`
only matches at the context ROOT and a nested key needs `**/*.key`. A grep would pass both.

The eval and client key directories live outside the repo by design; the test pins that too,
so a future default under the checkout would fail here instead of leaking a key into git.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from pathspec import GitIgnoreSpec

from ragforge_client.ask import DEFAULT_KEY_DIR
from ragforge_client.evals.cli import DEFAULT_EVAL_KEY_DIR

REPO: Final = Path(__file__).resolve().parents[3]

# (case id, path): every path holds, or could hold, a secret or the owner's private data, and
# BOTH .gitignore and .dockerignore must cover it (SEC-SCAN handoff step 1, applied).
_SENSITIVE: Final[list[tuple[str, str]]] = [
    ("dotenv", ".env"),
    ("dotenv-nested", "packages/ui/.env"),
    ("dotenv-variant", ".env.local"),
    ("dotenv-variant-nested", "packages/ui/.env.production"),
    ("key-at-root", "owner.key"),
    ("key-nested", "packages/core/eval/demo/owner.key"),
    ("pem-at-root", "server.pem"),
    ("pem-nested", "docker/tls/server.pem"),
    ("eval-private", "evals/private/questions.toml"),
    ("eval-private-runs", "evals/private/runs/a.jsonl"),
    ("eval-demo-runs", "evals/demo/runs/2026-10-22/answers.jsonl"),
    ("eval-any-dataset-runs", "evals/held_out/runs/r.jsonl"),
]


def _cases(_ignore_file: str) -> list[object]:
    return [pytest.param(path, id=case_id) for case_id, path in _SENSITIVE]


# Files that must stay visible: dropping them silently would break the repo or the build.
MUST_COMMIT: Final = [
    ".env.example",
    ".gitleaks.toml",
    "scripts/security/ignores.toml",
    "evals/demo/questions.toml",
    "packages/core/src/ragforge_core/services/auth/keys.py",
]
MUST_BUILD: Final = ["pyproject.toml", "uv.lock", "packages/core/pyproject.toml"]


def _lines(name: str) -> list[str]:
    return (REPO / name).read_text(encoding="utf-8").splitlines()


def _gitignored() -> Callable[[str], bool]:
    spec = GitIgnoreSpec.from_lines(_lines(".gitignore"))
    return lambda path: spec.match_file(path)


def _dockerignored() -> Callable[[str], bool]:
    # docker-py (via testcontainers) ships no type hints, and the repo's own `docker/` directory
    # shadows the package for mypy's path-based lookup; a runtime import sidesteps both.
    pattern_matcher = importlib.import_module("docker.utils.build").PatternMatcher
    patterns = [p.strip() for p in _lines(".dockerignore")]
    matcher = pattern_matcher([p for p in patterns if p and not p.startswith("#")])
    return lambda path: bool(matcher.matches(path))


@pytest.mark.parametrize("path", _cases(".gitignore"))
def test_gitignore_covers_sensitive_path(path: str) -> None:
    assert _gitignored()(path), f".gitignore does not cover {path}"


@pytest.mark.parametrize("path", _cases(".dockerignore"))
def test_dockerignore_keeps_sensitive_path_out_of_the_build_context(path: str) -> None:
    assert _dockerignored()(path), f".dockerignore does not exclude {path}"


@pytest.mark.parametrize("path", MUST_COMMIT)
def test_gitignore_does_not_swallow_committed_files(path: str) -> None:
    # The `.env.*` line needs a `!.env.example` exception, or the template disappears from git.
    assert not _gitignored()(path), f".gitignore hides {path}, which must be committed"


@pytest.mark.parametrize("path", MUST_BUILD)
def test_dockerignore_keeps_the_build_inputs(path: str) -> None:
    assert not _dockerignored()(path), f".dockerignore drops {path}, which the build needs"


@pytest.mark.parametrize("key_dir", [DEFAULT_KEY_DIR, DEFAULT_EVAL_KEY_DIR], ids=["client", "eval"])
def test_key_directories_live_outside_the_repo(key_dir: Path) -> None:
    resolved = key_dir.expanduser().resolve()
    assert not resolved.is_relative_to(REPO), f"{key_dir} resolves inside the checkout"


def test_matchers_are_not_vacuous() -> None:
    """Guard the guard: both matchers must reject an ordinary source path, or every assertion
    above would pass against an ignore-everything file."""
    source = "packages/core/src/ragforge_core/config.py"
    assert not _gitignored()(source)
    assert not _dockerignored()(source)
