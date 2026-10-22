"""SEC-SCAN: the custom gitleaks rules catch every RAGForge token shape and spare the fixtures.

ADR-0018 Security 3 + R2-8. Two layers:

1. A Python model of the parts of gitleaks' matching our custom rules use (regex, secretGroup,
   entropy floor, keyword prefilter, global stopwords), driven by `.gitleaks.toml` itself.
   Fast, no Docker: it fails on the edit that breaks a rule.
2. One integration test pipes the same samples through the real, pinned gitleaks binary
   (`gitleaks stdin`), so the model cannot drift from Go's RE2 semantics unnoticed.

Every token-shaped sample is assembled at runtime from a seeded generator: the source text of
this file never contains a complete token, so `make secrets-scan` stays clean over it (a test
below proves that too).
"""

from __future__ import annotations

import json
import math
import random
import re
import shlex
import shutil
import string
import subprocess
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pytest

REPO: Final = Path(__file__).resolve().parents[3]
CONFIG: Final = REPO / ".gitleaks.toml"
TESTS: Final = REPO / "packages/core/tests"
# Must equal GITLEAKS_IMAGE in scripts/security/lib.sh (pinned by version AND digest).
GITLEAKS_IMAGE: Final = (
    "zricethezav/gitleaks:v8.30.1"
    "@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f"
)

ALNUM: Final = string.ascii_letters + string.digits
URLSAFE: Final = ALNUM + "-_"
CUSTOM_RULES: Final = frozenset(
    {"ragforge-api-key", "notion-token-ntn", "notion-token-secret", "gemini-api-key"}
)


# ---------------------------------------------------------------- the model


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    regex: re.Pattern[str]
    secret_group: int
    entropy: float
    keywords: tuple[str, ...]


def _shannon(text: str) -> float:
    """Bits per character, as gitleaks computes it for the `entropy` floor."""
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def _config() -> Mapping[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _rules() -> dict[str, Rule]:
    rules = cast(list[dict[str, object]], _config()["rules"])
    return {
        str(r["id"]): Rule(
            id=str(r["id"]),
            regex=re.compile(str(r["regex"])),
            secret_group=cast(int, r.get("secretGroup", 0)),
            entropy=cast(float, r.get("entropy", 0.0)),
            keywords=tuple(cast(list[str], r.get("keywords", []))),
        )
        for r in rules
    }


def _global_stopwords() -> list[str]:
    """Stopwords of the global allowlists that apply to every rule (no paths, no targetRules)."""
    allowlists = cast(list[dict[str, object]], _config().get("allowlists", []))
    return [
        word.lower()
        for al in allowlists
        if "paths" not in al and "targetRules" not in al
        for word in cast(list[str], al.get("stopwords", []))
    ]


def _path_allowlist() -> list[re.Pattern[str]]:
    allowlists = cast(list[dict[str, object]], _config().get("allowlists", []))
    return [
        re.compile(p)
        for al in allowlists
        if "targetRules" not in al and "regexes" not in al
        for p in cast(list[str], al.get("paths", []))
    ]


def scan(text: str, *, stopwords: bool = True) -> list[str]:
    """Rule ids the custom rules report for `text` (one per match)."""
    words = _global_stopwords() if stopwords else []
    hits: list[str] = []
    for rule in _rules().values():
        if rule.keywords and not any(k.lower() in text.lower() for k in rule.keywords):
            continue
        for match in rule.regex.finditer(text):
            secret = match.group(rule.secret_group)
            if _shannon(secret) < rule.entropy or any(w in secret.lower() for w in words):
                continue
            hits.append(rule.id)
    return hits


# ---------------------------------------------------------------- synthetic samples
# Seeded, so a failure reproduces. Not a CSPRNG on purpose: these are throwaway test strings.
_RNG: Final = random.Random(20261022)  # noqa: S311


def _rand(n: int, alphabet: str = ALNUM) -> str:
    return "".join(_RNG.choice(alphabet) for _ in range(n))


def rf_key() -> str:
    return "rf" + "_" + _rand(16, "0123456789abcdef") + "_" + _rand(43, URLSAFE)


def notion_ntn() -> str:
    return "ntn" + "_" + _rand(11, string.digits) + _rand(35)


def notion_secret() -> str:
    return "secret" + "_" + _rand(43)


def gemini_key() -> str:
    return "AI" + "za" + _rand(35, URLSAFE)


SAMPLES: Final[dict[str, Callable[[], str]]] = {
    "ragforge-api-key": rf_key,
    "notion-token-ntn": notion_ntn,
    "notion-token-secret": notion_secret,
    "gemini-api-key": gemini_key,
}

# Where a pasted token really ends up: a dotenv line, JSON, a shell export, a URL, prose.
CONTEXTS: Final = [
    "{}",
    "RAGFORGE_KEY={}\n",
    '{{"token": "{}"}}',
    "export NOTION_TOKEN='{}'",
    "https://example.invalid/cb?key={}&x=1",
    "the key `{}` was pasted into chat",
]


def test_every_custom_rule_has_a_sample() -> None:
    assert set(_rules()) >= CUSTOM_RULES
    assert set(SAMPLES) == CUSTOM_RULES


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("rule_id", sorted(CUSTOM_RULES))
def test_rule_detects_its_token_shape_in_context(rule_id: str, context: str) -> None:
    assert scan(context.format(SAMPLES[rule_id]())) == [rule_id]


@pytest.mark.parametrize(
    ("rule_id", "mangle"),
    [
        ("ragforge-api-key", lambda t: t[:-1]),  # secret one char short
        ("ragforge-api-key", lambda t: t + "Z"),  # 44 chars: not a prefix match of a longer run
        ("ragforge-api-key", lambda t: "rf_ABCDEF0123456789" + t[19:]),  # prefix is lower hex
        ("notion-token-ntn", lambda t: t[:30]),  # too short to be a token
        ("notion-token-secret", lambda t: t[:-1]),
        ("notion-token-secret", lambda t: t + "Q"),
        ("gemini-api-key", lambda t: t[:-1]),
        ("gemini-api-key", lambda t: t + "-"),
    ],
)
def test_rule_rejects_near_misses(rule_id: str, mangle: Callable[[str], str]) -> None:
    assert scan(mangle(SAMPLES[rule_id]())) == []


@pytest.mark.parametrize(
    "placeholder",
    [
        "AI" + "za" + "X" * 35,
        "ntn" + "_" + "x" * 46,
        "secret" + "_" + "0" * 43,
        "rf_" + "0" * 16 + "_" + "a" * 43,
    ],
    ids=["gemini", "ntn", "secret", "rf"],
)
def test_low_entropy_placeholders_do_not_fire(placeholder: str) -> None:
    # Docs write placeholders like these; the entropy floor (not a path exception) spares them.
    assert scan(placeholder) == []


# The repo's deliberately key-shaped fixtures: (file, why it must not fire).
FIXTURE_FILES: Final = [
    "module_c1/c1_support.py",  # ntn_TEST... token: underscores break the shape
    "module_c2/c2_fakes.py",  # secret_notion_token_value
    "module_g1/g1_support.py",  # AIzaCANARY-...: 32 chars, not 35
    "module_g1/test_g1_review_hardening.py",  # AIza with smart quotes, tabs, spaces
    "module_g2/g2_support.py",  # AIzaSyG2-...leakcanary: exact shape, spared by the stopword
    "module_e2/test_e2_auth_unit.py",  # rf_ keys built by concatenation
    "module_e2/test_e2_sdk.py",
    "module_ev2/test_ev_judge_ports.py",  # AIza-test-secret-key
    "module_ui/test_ui_errors_config.py",  # rf_live_secret
    "test_secretbox.py",  # secret_notion_token
]


@pytest.mark.parametrize("relpath", FIXTURE_FILES)
def test_known_fixtures_do_not_fire(relpath: str) -> None:
    assert scan((TESTS / relpath).read_text(encoding="utf-8")) == []


def test_stopword_is_load_bearing_for_the_g2_canary() -> None:
    """The g2 FAKE_KEY has the exact Gemini shape: without the `leakcanary` stopword it fires.
    Proves the allowlist entry is needed (not dead config) and that the rule really matches it.
    (Its sibling OTHER_KEY is 34 chars after `AIza`, one short, so it never matches.)"""
    text = (TESTS / "module_g2/g2_support.py").read_text(encoding="utf-8")
    assert scan(text, stopwords=False) == ["gemini-api-key"]


def test_this_file_plants_no_complete_token() -> None:
    assert scan(Path(__file__).read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------- scan scope (path allowlist)


@pytest.mark.parametrize(
    "path",
    [
        ".venv/lib/python3.12/site-packages/x.py",
        "packages/core/src/ragforge_core/__pycache__/x.pyc",
        "node_modules/a/index.js",
        ".mypy_cache/3.12/x.json",
        ".ruff_cache/x",
        ".pytest_cache/v/x",
        ".import_linter_cache/x",
        ".hypothesis/examples/x",
        "evals/demo/runs/2026-10-22/answers.jsonl",
        "evals/private/runs/a.jsonl",
    ],
)
def test_scope_skips_generated_paths(path: str) -> None:
    assert any(p.search(path) for p in _path_allowlist()), f"{path} would be scanned"


@pytest.mark.parametrize(
    "path",
    [
        "docs/handoff/SEC-SCAN.md",
        "docs/workflows/phase4b-ops-build.js",
        "docs/adr/0018-ops-hardening-security-observability.md",
        "evals/demo/questions.toml",
        "evals/demo/fixtures/labels.jsonl",
        "packages/core/tests/module_g2/g2_support.py",
        "packages/core/src/ragforge_core/config.py",
        "scripts/bootstrap.sh",
        ".env.example",
        ".idea/dataSources.xml",
        "compose.yaml",
    ],
)
def test_scope_keeps_authored_paths(path: str) -> None:
    assert not any(p.search(path) for p in _path_allowlist()), f"{path} would be skipped"


# ---------------------------------------------------------------- the real binary


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=20)  # noqa: S607
    return probe.returncode == 0


# The container sees only the config file (read-only) and stdin: no repo, no network.
_GITLEAKS_RUN: Final = (
    "docker run --rm -i --network none --read-only --cap-drop ALL"
    " --security-opt no-new-privileges:true -v {config}:/cfg/gitleaks.toml:ro {image}"
    " stdin --config /cfg/gitleaks.toml --redact --no-banner --exit-code 0 --log-level error"
    " --report-format json --report-path -"
)


def _gitleaks_stdin(text: str) -> list[str]:
    """Rule ids the pinned gitleaks reports for `text`."""
    argv = shlex.split(_GITLEAKS_RUN.format(config=shlex.quote(str(CONFIG)), image=GITLEAKS_IMAGE))
    proc = subprocess.run(  # noqa: S603
        argv, input=text, capture_output=True, text=True, check=True, timeout=300
    )
    return [str(leak["RuleID"]) for leak in cast(list[dict[str, object]], json.loads(proc.stdout))]


def _lines() -> Iterator[tuple[str, str]]:
    for rule_id, make in sorted(SAMPLES.items()):
        for context in CONTEXTS:
            yield rule_id, context.format(make()).strip()


@pytest.mark.integration
def test_real_gitleaks_agrees_with_the_model() -> None:
    if not _docker_ready():
        pytest.skip("Docker is not reachable")
    cases = list(_lines())
    found = Counter(r for r in _gitleaks_stdin("\n".join(t for _, t in cases)) if r in CUSTOM_RULES)
    assert found == Counter(rule_id for rule_id, _ in cases)

    fixtures = "\n".join((TESTS / p).read_text(encoding="utf-8") for p in FIXTURE_FILES)
    assert [r for r in _gitleaks_stdin(fixtures) if r in CUSTOM_RULES] == []


def test_integration_test_uses_the_scripts_pinned_image() -> None:
    lib = (REPO / "scripts/security/lib.sh").read_text(encoding="utf-8")
    assert f'GITLEAKS_IMAGE="{GITLEAKS_IMAGE}"' in lib
