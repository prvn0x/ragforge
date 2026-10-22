"""The pre-publish gate (scripts/security/prepublish.py, Phase 5 R3-4), run in a temp git repo
with planted files. Token-shaped strings are built at runtime so this file never trips the gate.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "security" / "prepublish.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepublish", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepublish"] = module
    spec.loader.exec_module(module)
    return module


gate = _load()

RF_KEY = "rf_" + "a1" * 8 + "_" + "Q" * 43
NTN = "ntn_" + "x" * 30
GEMINI = "AIza" + "B" * 35
REAL_ID = "5" * 8 + "-1111-4222-8333-" + "4" * 12


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True, env=env)  # noqa: S603
    subprocess.run([git, "add", "-A", "-f"], cwd=tmp_path, check=True, env=env)  # noqa: S603
    return tmp_path


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - our own committed script, no shell
        [sys.executable, str(SCRIPT), "--no-live", *args],
        cwd=repo, capture_output=True, text=True, check=False,
    )  # fmt: skip


def test_a_clean_tree_passes(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        {
            "README.md": "Run `make up`. Roles: rf_api, rf_system. The text /Users/ is fine.\n",
            ".env.example": "RAGFORGE_NOTION_TOKEN=\n",
            "evals/demo/runs/r.summary.json": "{}\n",
            "evals/demo/runs/r.header.json": "{}\n",
            "packages/x/test_ids.py": f'ID = "{REAL_ID}"  # code fixtures may hold uuids\n',
            "docs/x.md": "placeholder 00000000-0000-4000-8000-000000000000\n",
        },
    )
    result = _run(repo)
    assert result.returncode == 0, result.stdout
    assert "0 findings" in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "keys/owner.key",
        "certs/x.pem",
        "evals/private/questions.toml",
        "evals/demo/runs/r.jsonl",
        "evals/demo/runs/target.json",
        ".idea/workspace.xml",
    ],
)
def test_a_forbidden_path_fails(tmp_path: Path, path: str) -> None:
    result = _run(_repo(tmp_path, {path: "x\n", "README.md": "ok\n"}))
    assert result.returncode == 1
    assert path in result.stdout


@pytest.mark.parametrize("secret", [RF_KEY, NTN, GEMINI, "secret_" + "c" * 43])
def test_a_token_anywhere_fails_and_is_never_printed(tmp_path: Path, secret: str) -> None:
    result = _run(_repo(tmp_path, {"packages/a.py": f'X = "{secret}"\n'}))
    assert result.returncode == 1
    assert "packages/a.py:1" in result.stdout
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("docs/NEXT.md", f"tenant {REAL_ID}"),
        ("evals/results.toml", f'collection = "{REAL_ID}"'),
        ("docs/a.md", "see /Users/someone/Desktop/x"),
        ("README.md", "log in /home/someone/.config"),
    ],
)
def test_ids_and_home_paths_in_reader_files_fail(tmp_path: Path, name: str, line: str) -> None:
    result = _run(_repo(tmp_path, {name: line + "\n"}))
    assert result.returncode == 1
    assert f"{name}:1" in result.stdout
    assert REAL_ID not in result.stdout


def test_live_ids_are_found_in_any_file_and_never_printed() -> None:
    findings = gate.scan(["packages/a.py"], lambda _: f'X = "{REAL_ID}"\n', frozenset({REAL_ID}))
    assert [f.rule for f in findings] == ["a live id or key prefix from this database"]
    assert REAL_ID not in findings[0].render()


def test_binary_files_are_skipped_but_their_path_is_checked() -> None:
    findings = gate.scan(["x.key"], lambda _: None, frozenset())
    assert [f.line for f in findings] == [0]


def test_outside_a_git_repo_it_refuses_to_run(tmp_path: Path) -> None:
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    result = subprocess.run(  # noqa: S603 - our own committed script, no shell
        [sys.executable, str(SCRIPT), "--no-live"],
        cwd=tmp_path, capture_output=True, text=True, check=False, env=env,
    )  # fmt: skip
    assert result.returncode == 2
    assert "git init" in result.stdout


def test_nothing_staged_refuses_to_run(tmp_path: Path) -> None:
    result = _run(_repo(tmp_path, {}))
    assert result.returncode == 2


def test_a_canary_token_with_the_gitleaks_stopword_is_exempt() -> None:
    canary = "AIza" + "leakcanary" + "C" * 25
    assert gate.content_findings("packages/a.py", f'K = "{canary}"', frozenset()) == []
    both = f'K = "{canary}" "{GEMINI}"'
    assert [f.rule for f in gate.content_findings("packages/a.py", both, frozenset())] == [
        "gemini-key"
    ]


def _private(tmp_path: Path, allow: str = "") -> dict[str, frozenset[str]]:
    d = tmp_path / "private"
    d.mkdir()
    (d / "questions.toml").write_text(
        '[[questions]]\nid = "acme-widget-q"\npages = ["Acme_Widget_Ops_Notes", "Kafka"]\n'
        '[[questions]]\nid = "q1"\n[questions.relevance]\n"Widget Team Runbook" = 2\n',
        encoding="utf-8",
    )
    if allow:
        (d / "prepublish-allow.txt").write_text(allow, encoding="utf-8")
    terms: dict[str, frozenset[str]] = gate.private_terms(d)
    return terms


def test_private_ids_and_distinctive_titles_are_collected(tmp_path: Path) -> None:
    terms = _private(tmp_path)
    assert terms["a private question id"] == {"acme-widget-q"}  # "q1" is too short to matter
    # A one-word title ("Kafka") is ordinary text; multi-word and underscored titles count.
    assert terms["a private note title"] == {"Acme_Widget_Ops_Notes", "Widget Team Runbook"}


def test_private_terms_are_found_as_whole_tokens_and_never_printed(tmp_path: Path) -> None:
    terms = _private(tmp_path)
    text = "missed `acme-widget-q`\ncited Widget Team Runbook\nacme-widget-qx is another id\n"
    findings = gate.content_findings("docs/NEXT.md", text, frozenset(), terms)
    assert [(f.line, f.rule) for f in findings] == [
        (1, "a private question id"),
        (2, "a private note title"),
    ]
    assert all("widget" not in f.render().lower() for f in findings)


def test_the_local_allow_list_skips_generic_titles(tmp_path: Path) -> None:
    terms = _private(tmp_path, allow="Widget Team Runbook\n")
    assert terms["a private note title"] == {"Acme_Widget_Ops_Notes"}


def test_no_private_set_means_no_private_terms(tmp_path: Path) -> None:
    assert all(not t for t in gate.private_terms(tmp_path / "absent").values())
