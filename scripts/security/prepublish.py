"""Pre-publish gate (Phase 5 R2-4 / R3-4): run once after `git init && git add -A`, before the
first commit. It checks exactly what is staged, so `.gitignore` has already been applied.

Three kinds of check, each reporting only `path:line  rule`, never the matched text:
1. Paths that must never be staged (env files, keys, private evals, raw eval runs).
2. Token shapes anywhere (RAGForge keys, Notion tokens, Gemini keys), plus UUIDs and home-folder
   paths in the files a reader sees (docs and eval data). Code and tests are exempt from the
   UUID rule: their fixed fixture ids are synthetic and too common to review one by one.
3. When the stack is running (`--live`, the default when `docker compose ps` shows postgres), the
   real tenant, collection and connector ids and API key prefixes, read from the database, must
   not appear in ANY staged file. This is the exact check; the UUID rule above is the broad one.

Exit 0 when clean, 1 on findings, 2 when it cannot run (not a git repo, nothing staged).
`make secrets-scan` (gitleaks) runs after this, from the Makefile target.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

# 1. Never staged. `.env.example` is the one env file that ships.
FORBIDDEN_PATHS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "evals/private/*",
    ".idea/*",
)
RUNS_DIR = re.compile(r"^evals/[^/]+/runs/")
RUNS_ALLOWED = ("*.summary.json", "*.header.json")
ALLOWED_ENV = ".env.example"

# 2. Content rules. The RAGForge shape is the exact one minted by services/auth/keys.py, so DB
# role names such as rf_api never match.
TOKEN_RULES: dict[str, re.Pattern[str]] = {
    "ragforge-api-key": re.compile(r"\brf_[0-9a-f]{16}_[A-Za-z0-9_-]{43}\b"),
    "notion-token": re.compile(r"\bntn_[A-Za-z0-9]{20,}"),
    "notion-legacy-token": re.compile(r"\bsecret_[A-Za-z0-9]{43}\b"),
    "gemini-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
}
# The same stopword as .gitleaks.toml: a planted test canary carries it inside the token itself
# (module_g2's FAKE_KEY), so both scanners exempt exactly that token and nothing else.
CANARY_STOPWORD = "leakcanary"
READER_RULES: dict[str, re.Pattern[str]] = {
    "uuid": re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    # A home folder with a user name after it; the bare text `/Users/` (naming the rule) is fine.
    "home-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
}
READER_FILES: tuple[str, ...] = ("*.md", "evals/*")
# Synthetic placeholders that may appear in docs (all-zero style ids).
ALLOWED_UUIDS = re.compile(r"^0{8}-0{4}-[0-9]0{3}-[0-9]0{3}-0{12}$")

# 4. The private eval set's own words: its question ids and the titles of the notes it cites,
# read from the LOCAL evals/private/*.toml (never staged). Only distinctive titles count (two or
# more words, or an underscore): a one-word title such as a technology name is ordinary text.
# `evals/private/prepublish-allow.txt` (git-ignored) lists generic titles to skip, one per line.
PRIVATE_DIR = Path("evals/private")
PRIVATE_ALLOW = PRIVATE_DIR / "prepublish-allow.txt"

LIVE_SQL = (
    "SELECT id::text FROM rf.tenants UNION ALL SELECT id::text FROM rf.collections "
    "UNION ALL SELECT id::text FROM rf.connectors UNION ALL SELECT prefix FROM rf.api_keys"
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int  # 0 = the path itself
    rule: str

    def render(self) -> str:
        where = self.path if self.line == 0 else f"{self.path}:{self.line}"
        return f"{where}  {self.rule}"


def path_findings(path: str) -> list[Finding]:
    """Rule 1: a path that must never be committed."""
    name = PurePosixPath(path).name
    if name == ALLOWED_ENV:
        return []
    for pattern in FORBIDDEN_PATHS:
        if fnmatch(path, pattern) or fnmatch(name, pattern):
            return [Finding(path, 0, f"forbidden path ({pattern})")]
    if RUNS_DIR.match(path) and not any(fnmatch(name, p) for p in RUNS_ALLOWED):
        return [Finding(path, 0, "raw eval run file (only *.summary.json / *.header.json ship)")]
    return []


def is_reader_file(path: str) -> bool:
    return any(fnmatch(path, p) for p in READER_FILES)


def _distinctive_title(title: str) -> bool:
    return len(title) >= 10 and ("_" in title or len(title.split()) >= 2)


def private_terms(private_dir: Path = PRIVATE_DIR) -> dict[str, frozenset[str]]:
    """Rule 4's terms, by kind. Empty when there is no private set (a fresh clone)."""
    ids: set[str] = set()
    titles: set[str] = set()
    for file in sorted(private_dir.glob("*.toml")):
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        questions = data.get("questions", [])
        for q in questions if isinstance(questions, list) else []:
            if not isinstance(q, dict):
                continue
            qid = q.get("id")
            if isinstance(qid, str) and len(qid) >= 6:
                ids.add(qid)
            pages = list(q.get("pages") or []) + list((q.get("relevance") or {}).keys())
            titles.update(t for t in pages if isinstance(t, str) and _distinctive_title(t))
    allow_file = private_dir / PRIVATE_ALLOW.name
    allowed = (
        set(allow_file.read_text(encoding="utf-8").splitlines()) if allow_file.exists() else set()
    )
    return {
        "a private question id": frozenset(ids - allowed),
        "a private note title": frozenset(titles - allowed),
    }


def content_findings(
    path: str,
    text: str,
    live_ids: frozenset[str],
    private: dict[str, frozenset[str]] | None = None,
) -> list[Finding]:
    """Rules 2 to 4 over one file's staged text."""
    out: list[Finding] = []
    reader = is_reader_file(path)
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in TOKEN_RULES.items():
            if any(CANARY_STOPWORD not in m.group(0) for m in pattern.finditer(line)):
                out.append(Finding(path, number, rule))
        if reader:
            if any(not ALLOWED_UUIDS.match(m) for m in READER_RULES["uuid"].findall(line)):
                out.append(Finding(path, number, "uuid"))
            if READER_RULES["home-path"].search(line):
                out.append(Finding(path, number, "home-path"))
        if any(value in line for value in live_ids):
            out.append(Finding(path, number, "a live id or key prefix from this database"))
        for rule, terms in (private or {}).items():
            if any(_has_term(line, term) for term in terms):
                out.append(Finding(path, number, rule))
    return out


def _has_term(line: str, term: str) -> bool:
    """`term` as a whole token (so the id `redis-x` does not match inside `redis-xyz`)."""
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", line) is not None


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(name)
    return path


def _git(args: Sequence[str]) -> bytes:
    # The resolved git binary, a fixed argv built here, no shell.
    return subprocess.run([_tool("git"), *args], check=True, capture_output=True).stdout  # noqa: S603


def staged_paths() -> list[str]:
    raw = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
    return [p for p in raw.decode("utf-8").split("\0") if p]


def staged_text(path: str) -> str | None:
    """The staged blob (what will be committed), or None for a binary file."""
    blob = _git(["show", f":{path}"])
    if b"\0" in blob[:8192]:
        return None
    return blob.decode("utf-8", errors="replace")


def live_ids() -> frozenset[str]:
    """Real ids and key prefixes from the running stack's database (read-only query)."""
    result = subprocess.run(  # noqa: S603 - the resolved docker binary, fixed argv, no shell
        [_tool("docker"), "compose", "exec", "-T", "postgres", "psql", "-U", "postgres", "-d",
         "ragforge", "-At", "-c", LIVE_SQL],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    values = (v.strip() for v in result.stdout.splitlines())
    # The all-zero SYSTEM tenant is a fixed constant of the schema (0001_base), not private.
    return frozenset(v for v in values if len(v) >= 12 and not ALLOWED_UUIDS.match(v))


def stack_running() -> bool:
    try:
        result = subprocess.run(  # noqa: S603 - the resolved docker binary, fixed argv, no shell
            [_tool("docker"), "compose", "ps", "-q", "postgres"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def scan(
    paths: Iterable[str],
    read: Callable[[str], str | None],
    ids: frozenset[str],
    private: dict[str, frozenset[str]] | None = None,
) -> list[Finding]:
    """All findings for `paths`; `read(path)` returns the staged text (None for binary)."""
    findings: list[Finding] = []
    for path in paths:
        findings += path_findings(path)
        text = read(path)
        if text is not None:
            findings += content_findings(path, text, ids, private)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="check the live database ids (default: when the stack is running)",
    )
    args = parser.parse_args(argv)
    try:
        paths = staged_paths()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("prepublish: not a git repository; run it after `git init && git add -A`")
        return 2
    if not paths:
        print("prepublish: nothing is staged; run `git add -A` first")
        return 2
    use_live = stack_running() if args.live is None else args.live
    try:
        ids = live_ids() if use_live else frozenset()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "prepublish: could not read the live ids (run it from the checkout whose stack is "
            "up, or set COMPOSE_PROJECT_NAME); pass --no-live to skip that check"
        )
        return 2
    private = private_terms()
    findings = scan(paths, staged_text, ids, private)
    for finding in findings:
        print(finding.render())
    live_note = f"{len(ids)} live ids checked" if use_live else "live ids NOT checked (stack down)"
    n_private = sum(len(t) for t in private.values())
    private_note = f"{n_private} private-set terms checked" if n_private else "no private set here"
    print(
        f"prepublish: {len(paths)} staged files, {len(findings)} findings; {live_note}; "
        f"{private_note}"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
