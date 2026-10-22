"""Gate for the security scans (ADR-0018 Security 1-3, Revision 2 R2-8).

Reads ONE scanner's JSON report (pip-audit, Trivy or gitleaks), applies the justified ignore list
(`ignores.toml`) and prints a summary a human can act on. Exit codes:

  0  nothing blocking (below-threshold findings are still counted)
  1  a blocking finding that no valid ignore covers, or an ignore past its review date
  2  the report or the ignore file is unusable (a scanner crash must never read as "clean")

Why a separate gate instead of each tool's own `--exit-code`/`--ignore-vuln` flags: one policy
for all three tools (every ignore needs a reason and a review date, and expires loudly), and one
place that guarantees the output never contains a secret. gitleaks' and Trivy's secret findings
carry the matched text (`Secret`, `Match`, `Code`); this module never reads those fields, so a
redaction bug upstream cannot leak through here.

Stdlib only, so it runs under any Python 3.12 without touching the project environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

Tool = Literal["pip-audit", "trivy", "gitleaks"]
TOOLS: Final[tuple[Tool, ...]] = ("pip-audit", "trivy", "gitleaks")

# Trivy severities that fail the gate (ADR-0018: HIGH/CRITICAL are fixed or documented).
BLOCKING_SEVERITIES: Final = frozenset({"HIGH", "CRITICAL"})
SEVERITY_ORDER: Final = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")
MIN_REASON_CHARS: Final = 20
TRIVY_SCHEMA: Final = 2

EXIT_OK: Final = 0
EXIT_BLOCKED: Final = 1
EXIT_UNUSABLE: Final = 2


class ReportError(Exception):
    """The input cannot be trusted as a scan result (malformed JSON, wrong shape, bad ignore)."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One scanner finding, reduced to fields that are safe to print.

    `subject` names WHERE (package@version, or file:line); it never holds matched text.
    """

    tool: Tool
    id: str
    aliases: tuple[str, ...]
    severity: str
    package: str
    subject: str
    detail: str
    blocking: bool

    def ids(self) -> frozenset[str]:
        return frozenset((self.id, *self.aliases))


@dataclass(frozen=True, slots=True)
class Ignore:
    """A reviewed exception: which finding, why it is acceptable, and until when."""

    tool: Tool
    id: str
    packages: frozenset[str]
    reason: str
    review_by: dt.date

    def covers(self, finding: Finding) -> bool:
        return (
            finding.tool == self.tool
            and self.id in finding.ids()
            and (not self.packages or finding.package in self.packages)
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    blocking: tuple[Finding, ...]
    ignored: tuple[Finding, ...]
    below_threshold: tuple[Finding, ...]
    expired: tuple[Ignore, ...]
    unused: tuple[Ignore, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_BLOCKED if self.blocking or self.expired else EXIT_OK


# ---------------------------------------------------------------- JSON helpers
# Scanner JSON is untrusted input: every access is shape-checked so a schema change in a tool
# becomes a loud ReportError (exit 2), not a silently empty (= "clean") result.


def _obj(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ReportError(f"{where}: expected an object, got {type(value).__name__}")
    return cast(Mapping[str, object], value)


def _list(value: object, where: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReportError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _str(value: object, where: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise ReportError(f"{where}: expected a string, got {type(value).__name__}")
    return value


def _int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{where}: expected an integer, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------- parsers


def parse_pip_audit(doc: object) -> list[Finding]:
    """pip-audit `--format json`. PyPI advisories carry no severity, so every one blocks."""
    report = _obj(doc, "pip-audit")
    if "dependencies" not in report:
        raise ReportError("pip-audit: no `dependencies` key (not a pip-audit JSON report)")
    findings: list[Finding] = []
    for i, raw_dep in enumerate(_list(report["dependencies"], "dependencies")):
        dep = _obj(raw_dep, f"dependencies[{i}]")
        name = _str(dep.get("name"), f"dependencies[{i}].name")
        version = _str(dep.get("version"), f"dependencies[{i}].version", default="?")
        for j, raw_vuln in enumerate(_list(dep.get("vulns"), f"{name}.vulns")):
            vuln = _obj(raw_vuln, f"{name}.vulns[{j}]")
            fixes = [_str(v, f"{name}.fix_versions") for v in _list(vuln.get("fix_versions"), "")]
            aliases = [_str(a, f"{name}.aliases") for a in _list(vuln.get("aliases"), "")]
            findings.append(
                Finding(
                    tool="pip-audit",
                    id=_str(vuln.get("id"), f"{name}.vulns[{j}].id"),
                    aliases=tuple(aliases),
                    severity="UNKNOWN",
                    package=name,
                    subject=f"{name}=={version}",
                    detail=f"fix: {', '.join(fixes)}" if fixes else "no fixed version",
                    blocking=True,
                )
            )
    return findings


def _trivy_vulns(target: str, raw: object) -> Iterator[Finding]:
    for k, item in enumerate(_list(raw, f"{target}.Vulnerabilities")):
        vuln = _obj(item, f"{target}.Vulnerabilities[{k}]")
        severity = _str(vuln.get("Severity"), "Severity", default="UNKNOWN")
        pkg = _str(vuln.get("PkgName"), "PkgName")
        fixed = _str(vuln.get("FixedVersion"), "FixedVersion", default="")
        status = _str(vuln.get("Status"), "Status", default="")
        yield Finding(
            tool="trivy",
            id=_str(vuln.get("VulnerabilityID"), "VulnerabilityID"),
            aliases=(),
            severity=severity,
            package=pkg,
            subject=f"{pkg}@{_str(vuln.get('InstalledVersion'), 'InstalledVersion', default='?')}",
            detail=f"fix: {fixed}" if fixed else f"no fix ({status or 'unknown status'})",
            blocking=severity in BLOCKING_SEVERITIES,
        )


def _trivy_misconfigs(target: str, raw: object) -> Iterator[Finding]:
    for k, item in enumerate(_list(raw, f"{target}.Misconfigurations")):
        mis = _obj(item, f"{target}.Misconfigurations[{k}]")
        if _str(mis.get("Status"), "Status", default="FAIL") != "FAIL":
            continue
        severity = _str(mis.get("Severity"), "Severity", default="UNKNOWN")
        yield Finding(
            tool="trivy",
            id=_str(mis.get("ID"), "ID"),
            aliases=tuple(a for a in [_str(mis.get("AVDID"), "AVDID", default="")] if a),
            severity=severity,
            package=target,
            subject=target,
            detail=_str(mis.get("Title"), "Title", default="misconfiguration"),
            blocking=severity in BLOCKING_SEVERITIES,
        )


def _trivy_secrets(target: str, raw: object) -> Iterator[Finding]:
    # Deliberately reads only RuleID/Severity/StartLine: `Match` and `Code` hold file content.
    for k, item in enumerate(_list(raw, f"{target}.Secrets")):
        sec = _obj(item, f"{target}.Secrets[{k}]")
        line = sec.get("StartLine")
        rule = _str(sec.get("RuleID"), "RuleID")
        yield Finding(
            tool="trivy",
            id=rule,
            aliases=(),
            severity=_str(sec.get("Severity"), "Severity", default="UNKNOWN"),
            package=target,
            subject=f"{target}:{_int(line, 'StartLine')}" if line is not None else target,
            detail="secret (value withheld)",
            # Any secret baked into an image is blocking, whatever Trivy's severity says.
            blocking=True,
        )


def parse_trivy(doc: object) -> list[Finding]:
    """Trivy `--format json` (image or config scan): vulnerabilities, failed checks, secrets."""
    report = _obj(doc, "trivy")
    # Trivy omits `Results` when a scan found no targets, so presence of the schema marker (not
    # of Results) is what tells a real report from an empty or foreign file.
    if _int(report.get("SchemaVersion"), "trivy.SchemaVersion") != TRIVY_SCHEMA:
        raise ReportError(f"trivy: SchemaVersion {report.get('SchemaVersion')} != {TRIVY_SCHEMA}")
    findings: list[Finding] = []
    for i, raw_result in enumerate(_list(report.get("Results"), "Results")):
        result = _obj(raw_result, f"Results[{i}]")
        target = _str(result.get("Target"), f"Results[{i}].Target")
        findings += _trivy_vulns(target, result.get("Vulnerabilities"))
        findings += _trivy_misconfigs(target, result.get("Misconfigurations"))
        findings += _trivy_secrets(target, result.get("Secrets"))
    return findings


def parse_gitleaks(doc: object) -> list[Finding]:
    """gitleaks `--report-format json`: a list of leaks. Every leak blocks.

    There is no ignore path for gitleaks here on purpose: suppressions belong in the reviewed
    `.gitleaks.toml` allowlists, next to the rules they relax.
    """
    findings: list[Finding] = []
    for i, raw in enumerate(_list(doc, "gitleaks")):
        leak = _obj(raw, f"leaks[{i}]")
        path = _str(leak.get("File"), f"leaks[{i}].File")
        findings.append(
            Finding(
                tool="gitleaks",
                id=_str(leak.get("RuleID"), f"leaks[{i}].RuleID"),
                aliases=(),
                severity="HIGH",
                package=path,
                subject=f"{path}:{_int(leak.get('StartLine'), f'leaks[{i}].StartLine')}",
                detail="secret (value withheld)",
                blocking=True,
            )
        )
    return findings


PARSERS: Final = {"pip-audit": parse_pip_audit, "trivy": parse_trivy, "gitleaks": parse_gitleaks}


# ---------------------------------------------------------------- ignores


def _parse_ignore(raw: object, where: str) -> Ignore:
    entry = _obj(raw, where)
    tool = _str(entry.get("tool"), f"{where}.tool")
    if tool not in TOOLS:
        raise ReportError(f"{where}.tool: {tool!r} is not one of {', '.join(TOOLS)}")
    if tool == "gitleaks":
        raise ReportError(f"{where}: gitleaks findings are allowlisted in .gitleaks.toml only")
    reason = _str(entry.get("reason"), f"{where}.reason").strip()
    if len(reason) < MIN_REASON_CHARS:
        raise ReportError(f"{where}.reason: a justification of {MIN_REASON_CHARS}+ chars")
    review_by = entry.get("review_by")
    if not isinstance(review_by, dt.date) or isinstance(review_by, dt.datetime):
        raise ReportError(f"{where}.review_by: a TOML date (YYYY-MM-DD) is required")
    packages = [_str(p, f"{where}.packages") for p in _list(entry.get("packages"), where)]
    return Ignore(
        tool=cast(Tool, tool),
        id=_str(entry.get("id"), f"{where}.id"),
        packages=frozenset(packages),
        reason=reason,
        review_by=review_by,
    )


def load_ignores(path: Path) -> list[Ignore]:
    """Parse and validate the ignore file. A malformed entry is an error, never skipped."""
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReportError(f"{path}: {exc}") from exc
    return [
        _parse_ignore(raw, f"ignore[{i}]") for i, raw in enumerate(_list(doc.get("ignore"), ""))
    ]


def judge(findings: Sequence[Finding], ignores: Iterable[Ignore], today: dt.date) -> Verdict:
    """Split findings into blocking / ignored / below threshold.

    An expired ignore still fails the run (so it is re-reviewed, not forgotten), and it no longer
    hides anything: its findings are reported as blocking too.
    """
    live = [ig for ig in ignores if ig.review_by >= today]
    expired = tuple(ig for ig in ignores if ig.review_by < today)
    blocking: list[Finding] = []
    ignored: list[Finding] = []
    used: set[Ignore] = set()
    for finding in findings:
        if not finding.blocking:
            continue
        cover = next((ig for ig in live if ig.covers(finding)), None)
        if cover is None:
            blocking.append(finding)
        else:
            used.add(cover)
            ignored.append(finding)
    return Verdict(
        blocking=tuple(blocking),
        ignored=tuple(ignored),
        below_threshold=tuple(f for f in findings if not f.blocking),
        expired=expired,
        unused=tuple(ig for ig in live if ig not in used),
    )


# ---------------------------------------------------------------- output


def _severity_counts(findings: Iterable[Finding]) -> str:
    counts = Counter(f.severity for f in findings)
    parts = [f"{sev} {counts[sev]}" for sev in SEVERITY_ORDER if counts[sev]]
    return ", ".join(parts) if parts else "none"


def render(tool: Tool, label: str, findings: Sequence[Finding], verdict: Verdict) -> list[str]:
    """Human summary. Every line is built from `Finding`/`Ignore` fields, none of which hold
    matched secret text."""
    lines = [
        f"[{tool}] {label}: {len(findings)} finding(s) by severity: {_severity_counts(findings)}"
    ]
    lines += [
        f"  BLOCKING {f.severity:<8} {f.id:<22} {f.subject}  ({f.detail})" for f in verdict.blocking
    ]
    for ig in verdict.expired:
        lines.append(f"  EXPIRED  ignore {ig.id} (review_by {ig.review_by}): fix it or re-justify")
    # Ignored findings are grouped per advisory: one OS CVE often spans a dozen packages.
    grouped: dict[tuple[str, str], list[str]] = {}
    for f in verdict.ignored:
        grouped.setdefault((f.severity, f.id), []).append(f.package)
    lines += [
        f"  ignored  {sev:<8} {fid:<22} {len(pkgs)} package(s): {', '.join(sorted(set(pkgs)))}"
        for (sev, fid), pkgs in grouped.items()
    ]
    if verdict.unused:
        # One line: a config scan legitimately matches none of the image's OS-package ignores.
        stale = ", ".join(sorted({ig.id for ig in verdict.unused}))
        lines.append(f"  note     {len(verdict.unused)} ignore(s) matched nothing here: {stale}")
    lines.append(
        f"  => {len(verdict.blocking)} blocking, {len(verdict.ignored)} justified-ignored, "
        f"{len(verdict.below_threshold)} below threshold, {len(verdict.expired)} expired ignore(s)"
    )
    return lines


def _load_report(path: str) -> object:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"{path}: not a JSON report ({exc.__class__.__name__})") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tool", choices=TOOLS)
    parser.add_argument("report", help="JSON report path, or - for stdin")
    parser.add_argument("--label", default="", help="what was scanned (for the summary line)")
    parser.add_argument("--ignores", type=Path, default=Path(__file__).with_name("ignores.toml"))
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args(argv)
    tool = cast(Tool, args.tool)
    try:
        findings = PARSERS[tool](_load_report(args.report))
        ignores = [ig for ig in load_ignores(args.ignores) if ig.tool == tool]
    except ReportError as exc:
        print(f"[{tool}] ERROR: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE
    verdict = judge(findings, ignores, args.today)
    print("\n".join(render(tool, args.label or args.report, findings, verdict)))
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
