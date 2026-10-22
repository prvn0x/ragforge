"""SEC-SCAN: the scan gate (scripts/security/report.py) blocks, ignores and redacts as promised.

The gate is exercised as `make audit` runs it: a subprocess fed a synthetic JSON report, so the
test covers argument parsing, exit codes and the printed text (where a leak would show).
Synthetic reports only; no scanner runs here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

REPO: Final = Path(__file__).resolve().parents[3]
GATE: Final = REPO / "scripts/security/report.py"
TODAY: Final = "2026-10-22"
# Built at runtime so this file never holds a token-shaped literal (see the gitleaks rule test).
CANARY: Final = "CANARY" + "-must-never-print-" + "7f3a9c"


@dataclass(frozen=True)
class Run:
    code: int
    out: str
    err: str


def gate(tmp_path: Path, tool: str, report: object, ignores: str = "", *extra: str) -> Run:
    report_file = tmp_path / "report.json"
    report_file.write_text(report if isinstance(report, str) else json.dumps(report))
    ignore_file = tmp_path / "ignores.toml"
    ignore_file.write_text(ignores)
    args: Sequence[str] = [sys.executable, "-I", str(GATE), tool, str(report_file)]
    proc = subprocess.run(  # noqa: S603
        [*args, "--ignores", str(ignore_file), "--today", TODAY, *extra],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return Run(proc.returncode, proc.stdout, proc.stderr)


def trivy_vuln(cve: str, severity: str, pkg: str = "libfoo", fixed: str = "") -> object:
    vuln = {"VulnerabilityID": cve, "PkgName": pkg, "InstalledVersion": "1.0", "Severity": severity}
    if fixed:
        vuln["FixedVersion"] = fixed
    return {"SchemaVersion": 2, "Results": [{"Target": "img (debian)", "Vulnerabilities": [vuln]}]}


def ignore(
    cve: str, *, tool: str = "trivy", packages: str = "", review_by: str = "2027-01-31"
) -> str:
    scope = f"packages = [{packages}]\n" if packages else ""
    return (
        f'[[ignore]]\ntool = "{tool}"\nid = "{cve}"\n{scope}review_by = {review_by}\n'
        'reason = "no upstream fix; the package is never invoked by the application"\n'
    )


# ---------------------------------------------------------------- threshold and ignores


@pytest.mark.parametrize(
    ("severity", "code"), [("CRITICAL", 1), ("HIGH", 1), ("MEDIUM", 0), ("LOW", 0)]
)
def test_trivy_blocks_high_and_critical_only(tmp_path: Path, severity: str, code: int) -> None:
    run = gate(tmp_path, "trivy", trivy_vuln("CVE-2026-0001", severity, fixed="1.1"))
    assert run.code == code
    assert f"{severity} 1" in run.out
    assert ("BLOCKING" in run.out) == bool(code)


def test_justified_ignore_unblocks_and_is_reported(tmp_path: Path) -> None:
    run = gate(tmp_path, "trivy", trivy_vuln("CVE-2026-0001", "HIGH"), ignore("CVE-2026-0001"))
    assert run.code == 0
    assert "ignored  HIGH     CVE-2026-0001" in run.out
    assert "1 justified-ignored" in run.out


def test_ignore_scoped_to_packages_does_not_cover_another_package(tmp_path: Path) -> None:
    run = gate(
        tmp_path,
        "trivy",
        trivy_vuln("CVE-2026-0001", "HIGH", pkg="libbar"),
        ignore("CVE-2026-0001", packages='"libfoo"'),
    )
    assert run.code == 1
    assert "libbar@1.0" in run.out


def test_expired_ignore_fails_and_stops_hiding(tmp_path: Path) -> None:
    run = gate(
        tmp_path,
        "trivy",
        trivy_vuln("CVE-2026-0001", "HIGH"),
        ignore("CVE-2026-0001", review_by="2026-10-21"),
    )
    assert run.code == 1
    assert "EXPIRED" in run.out
    assert "BLOCKING HIGH     CVE-2026-0001" in run.out


def test_expired_ignore_fails_even_when_nothing_matches(tmp_path: Path) -> None:
    run = gate(
        tmp_path,
        "trivy",
        {"SchemaVersion": 2, "Results": []},
        ignore("CVE-1", review_by="2026-01-01"),
    )
    assert run.code == 1


def test_ignore_on_review_day_still_applies(tmp_path: Path) -> None:
    run = gate(
        tmp_path,
        "trivy",
        trivy_vuln("CVE-2026-0001", "HIGH"),
        ignore("CVE-2026-0001", review_by=TODAY),
    )
    assert run.code == 0


def test_stale_ignore_is_noted_not_failed(tmp_path: Path) -> None:
    run = gate(tmp_path, "trivy", {"SchemaVersion": 2, "Results": []}, ignore("CVE-2026-9999"))
    assert run.code == 0
    assert "1 ignore(s) matched nothing here: CVE-2026-9999" in run.out


def test_ignores_of_another_tool_do_not_apply(tmp_path: Path) -> None:
    pip_ignore = ignore("CVE-2026-0001", tool="pip-audit")
    run = gate(tmp_path, "trivy", trivy_vuln("CVE-2026-0001", "HIGH"), pip_ignore)
    assert run.code == 1


# ---------------------------------------------------------------- pip-audit


def pip_report(
    vuln_id: str = "PYSEC-2026-1", aliases: Sequence[str] = ("GHSA-aaaa-bbbb-cccc",)
) -> object:
    vuln = {"id": vuln_id, "fix_versions": ["2.0"], "aliases": list(aliases), "description": "x"}
    return {
        "dependencies": [
            {"name": "pkg", "version": "1.0", "vulns": [vuln]},
            {"name": "ok", "version": "1"},
        ]
    }


def test_pip_audit_blocks_every_advisory_and_names_the_fix(tmp_path: Path) -> None:
    run = gate(tmp_path, "pip-audit", pip_report())
    assert run.code == 1
    assert "PYSEC-2026-1" in run.out
    assert "pkg==1.0  (fix: 2.0)" in run.out


def test_pip_audit_ignore_matches_an_alias(tmp_path: Path) -> None:
    run = gate(tmp_path, "pip-audit", pip_report(), ignore("GHSA-aaaa-bbbb-cccc", tool="pip-audit"))
    assert run.code == 0


def test_pip_audit_clean_lock_passes(tmp_path: Path) -> None:
    assert gate(tmp_path, "pip-audit", {"dependencies": [{"name": "ok", "version": "1"}]}).code == 0


# ---------------------------------------------------------------- secrets never print


def test_gitleaks_finding_blocks_without_printing_the_secret(tmp_path: Path) -> None:
    leak = {
        "RuleID": "notion-token-ntn",
        "File": "docs/handoff/X.md",
        "StartLine": 12,
        "Secret": CANARY,
        "Match": f"token={CANARY}",
        "Fingerprint": "docs/handoff/X.md:notion-token-ntn:12",
    }
    run = gate(tmp_path, "gitleaks", [leak])
    assert run.code == 1
    assert "notion-token-ntn" in run.out
    assert "docs/handoff/X.md:12" in run.out
    assert CANARY not in run.out + run.err


def test_trivy_secret_blocks_at_any_severity_without_printing_it(tmp_path: Path) -> None:
    secret = {
        "RuleID": "generic-token",
        "Severity": "LOW",
        "StartLine": 3,
        "Title": "token",
        "Match": f"KEY={CANARY}",
        "Code": {"Lines": [{"Content": CANARY}]},
    }
    report = {"SchemaVersion": 2, "Results": [{"Target": "/app/settings.py", "Secrets": [secret]}]}
    run = gate(tmp_path, "trivy", report)
    assert run.code == 1
    assert "/app/settings.py:3" in run.out
    assert CANARY not in run.out + run.err


def test_gitleaks_cannot_be_ignored_from_the_ignore_file(tmp_path: Path) -> None:
    run = gate(tmp_path, "gitleaks", [], ignore("notion-token-ntn", tool="gitleaks"))
    assert run.code == 2
    assert ".gitleaks.toml" in run.err


# ---------------------------------------------------------------- unusable input is never "clean"


@pytest.mark.parametrize(
    ("tool", "report"),
    [
        ("trivy", ""),
        ("trivy", "not json"),
        ("trivy", {"SchemaVersion": 2, "Results": "oops"}),
        ("trivy", {"SchemaVersion": 2, "Results": [{"Vulnerabilities": []}]}),
        ("pip-audit", []),
        ("pip-audit", {"dependencies": [{"version": "1"}]}),
        ("gitleaks", {"leaks": []}),
        ("gitleaks", [{"RuleID": "x", "File": "a", "StartLine": "12"}]),
    ],
    ids=[
        "empty",
        "garbage",
        "results-not-list",
        "no-target",
        "pip-list",
        "pip-no-name",
        "gitleaks-object",
        "gitleaks-bad-line",
    ],
)
def test_malformed_report_is_an_error_not_a_pass(tmp_path: Path, tool: str, report: object) -> None:
    run = gate(tmp_path, tool, report)
    assert run.code == 2, run.out


def _entry(**fields: str) -> str:
    return "[[ignore]]\n" + "".join(f"{k} = {v}\n" for k, v in fields.items())


_REASON: Final = '"a long enough justification for the test"'


@pytest.mark.parametrize(
    "bad_ignore",
    [
        _entry(tool='"trivy"', id='"CVE-1"', review_by="2027-01-01", reason='"short"'),
        _entry(tool='"trivy"', id='"CVE-1"', reason=_REASON),
        _entry(tool='"trivy"', id='"CVE-1"', review_by='"2027-01-01"', reason=_REASON),
        _entry(tool='"grype"', id='"CVE-1"', review_by="2027-01-01", reason=_REASON),
        _entry(tool='"trivy"', review_by="2027-01-01", reason=_REASON),
        "[[ignore]\n",
    ],
    ids=["short-reason", "no-review-date", "date-as-string", "unknown-tool", "no-id", "bad-toml"],
)
def test_invalid_ignore_entry_is_an_error(tmp_path: Path, bad_ignore: str) -> None:
    run = gate(tmp_path, "trivy", {"SchemaVersion": 2, "Results": []}, bad_ignore)
    assert run.code == 2


def test_shipped_ignore_file_is_valid_and_not_expired() -> None:
    """The real ignores.toml parses, and no entry is past its review date today."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(GATE), "trivy", "-"],
        input=json.dumps({"SchemaVersion": 2, "Results": []}),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 expired" in proc.stdout
