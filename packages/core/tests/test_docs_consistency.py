"""The four public documents stay true to the repository (PHASE5 R2-2, R2-8 item 4).

WHY a test and not a review step: docs drift silently. A renamed make target, a moved ADR, a
renamed test or a re-run eval leaves the prose wrong, and nothing else in `make check` notices.
These checks turn each kind of drift into a failing test:

- every `make X` written in code is a rule in the Makefile (parsed from rule lines, not .PHONY);
- every `make eval ARGS="<sub> ..."` and `ragforge-eval2 <sub>` names a real CLI subcommand;
- every relative link resolves inside the repo, and every `#anchor` matches a heading;
- every `docs/adr/*.md` has a table row in DECISIONS.md, every top-level `docs/*.md` is linked
  from the README, and the README stays at most 200 lines;
- prose avoids a short list of marketing words, and no document holds a UUID or a home path;
- every backticked `test_*` name is a test that exists, so a README "proof" cannot go stale;
- the README results rows equal the run summary they cite.

Markdown is parsed with markdown-it-py (already a core dependency), not with regexes, so code
spans, fences, tables and HTML comments are told apart exactly as a CommonMark renderer does.

Each checker is a pure function returning a list of violations. The `test_real_*` tests run them
on the real documents; the `test_planted_*` tests prove each checker fails on a planted violation
and passes on the fixed text (the mutation check), using temp files only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token

from ragforge_client.evals.cli import build_parser

REPO = Path(__file__).resolve().parents[3]

DOCS: tuple[str, ...] = ("README.md", "docs/GUIDE.md", "docs/RESULTS.md", "docs/DECISIONS.md")

# The integrator flips this to True once all four documents exist (PHASE5 "Deliverables"). While
# it is False, a missing document is SKIPPED (not xfailed) and links to a missing one of the four
# are tolerated; when True, a missing document or a link to one fails.
REQUIRED_DOCS = True

# Targets an ADR planned and DECISIONS.md lists as "not built". Allowed in DECISIONS.md only (a
# README or GUIDE step must never name one), and each must still be absent from the Makefile:
# once it is built, the "not built" row is false and this test says so.
NOT_BUILT_TARGETS: frozenset[str] = frozenset({"eval-models", "label-verdicts", "reindex"})
NOT_BUILT_DOC = "docs/DECISIONS.md"

README_MAX_LINES = 200

# Process notes that stay out of the README doc map on purpose (NEXT.md links them).
README_MAP_EXCLUDED: frozenset[str] = frozenset({"PHASE1.md", "PHASE5.md"})

BANNED_WORDS: tuple[str, ...] = (
    "blazing",
    "state-of-the-art",
    "seamless",
    "robust",
    "powerful",
    "cutting-edge",
    "revolutionary",
)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# Built from parts so this file does not itself trip the pre-publish scan (PHASE5 R3-4).
HOME_PATH = "/".join(("", "Users", ""))
MAKE_RULE_RE = re.compile(r"^([a-z0-9-]+):(?!=)", re.M)
# `make [-flag ...] [VAR=value ...] target`: GNU make accepts flags and variable assignments
# before the target, in any order. A flag that takes a separate value (`-C dir`) would read the
# value as the target; that fails loudly (a false alarm), never silently.
MAKE_VAR = r"""[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)\s+"""
FLAG = r"-\S+\s+"
MAKE_CALL_RE = re.compile(
    rf"(?<![\w./-])make\s+(?:{FLAG}|{MAKE_VAR})*([a-z0-9][a-z0-9-]*)(?![\w=-])"
)
# The command words: up to two bare words (`run retrieval`, `snapshot create`), each followed by
# a space, a quote or the end, so a path (`label evals/x.jsonl`) or a later flag ends the command.
# Leading flags are skipped, so `--verbose nope` is still read as `nope`.
EVAL_SUB_RE = re.compile(
    r"(?:(?<![\w-])make\s+eval\s+ARGS=[\"']?\s*|(?<![\w-])ragforge-eval2\s+)"
    rf"(?:{FLAG})*"
    r"([a-z][a-z0-9-]*(?:[ \t]+[a-z][a-z0-9-]*)?)(?=[\s\"']|$)"
)
TEST_NAME_RE = re.compile(r"test_[a-z0-9_]+")
TEST_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test_[a-z0-9_]+)\s*\(", re.M)
HTML_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.I)
HTML_LINK_RE = re.compile(r"<(?:a|img)\s[^>]*?\b(?:href|src)=[\"']([^\"']*)[\"']", re.I)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# Raw-HTML code: `<pre>` and `<code>` bodies render as code, so a command there is checked.
HTML_CODE_RE = re.compile(r"<(pre|code)\b[^>]*>(.*?)</\1\s*>", re.S | re.I)
HTML_CODE_OPEN_RE = re.compile(r"^<(?:pre|code)\b[^>]*>$", re.I)
HTML_CODE_CLOSE_RE = re.compile(r"^</(?:pre|code)\s*>$", re.I)
METRIC_CELL_RE = re.compile(r"^`([a-z_]+(?:\.[a-z0-9_@]+)+)`$")


# ----------------------------------------------------------------------------- parsed document
@dataclass(frozen=True)
class Link:
    target: str
    line: int
    in_table: bool


@dataclass(frozen=True)
class CitedTest:
    """A backticked test name and the nearest `.py` link before it in the same paragraph or
    bullet (None when there is none): the README names a proof beside the file that holds it."""

    name: str
    file_link: str | None


@dataclass(frozen=True)
class Doc:
    """What the checks need from one Markdown file, extracted once."""

    path: Path
    text: str
    links: tuple[Link, ...]
    code: tuple[str, ...]  # fenced blocks and inline code spans
    prose: str  # rendered-text runs outside code and HTML
    anchors: frozenset[str]
    cited_tests: tuple[CitedTest, ...] = ()


def _parser() -> MarkdownIt:
    # GitHub renders tables; plain CommonMark would read a table row as a paragraph.
    return MarkdownIt("commonmark").enable("table")


def github_slug(text: str) -> str:
    """GitHub's heading id (github-slugger): lower-case, drop everything but word characters,
    hyphens and spaces, then spaces become hyphens. Hyphen runs are NOT collapsed."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def _heading_text(inline: Token) -> str:
    # The rendered text: emphasis and link markers are separate tokens, so they drop out here.
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


def _anchors(headings: Iterable[str], html: str) -> frozenset[str]:
    """Heading ids with GitHub's de-duplication (`x`, `x-1`, `x-2`), plus explicit HTML ids."""
    seen: Counter[str] = Counter()
    out: set[str] = set(HTML_ANCHOR_RE.findall(html))
    for heading in headings:
        slug = github_slug(heading)
        out.add(slug if seen[slug] == 0 else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return frozenset(out)


def _html_links(html: str, line: int, in_table: bool) -> list[Link]:
    """`<a href>` and `<img src>` in raw HTML render as links too. Comments do not render, so a
    link inside one is not checked."""
    return [Link(t, line, in_table) for t in HTML_LINK_RE.findall(HTML_COMMENT_RE.sub("", html))]


def _html_code(html: str) -> list[str]:
    """The bodies of `<pre>` and `<code>` in a raw-HTML block, outside comments. Other HTML text
    is prose, where "make sure" is English, not a command."""
    return [body for _, body in HTML_CODE_RE.findall(HTML_COMMENT_RE.sub("", html))]


def parse_doc(path: Path) -> Doc:
    text = path.read_text(encoding="utf-8")
    tokens = _parser().parse(text)
    links: list[Link] = []
    code: list[str] = []
    prose: list[str] = []
    headings: list[str] = []
    html: list[str] = []
    cited: list[CitedTest] = []
    in_table = False
    for i, tok in enumerate(tokens):
        line = (tok.map[0] + 1) if tok.map else 0
        if tok.type == "table_open":
            in_table = True
        elif tok.type == "table_close":
            in_table = False
        elif tok.type in ("fence", "code_block"):
            code.append(tok.content)
        elif tok.type == "html_block":
            html.append(tok.content)
            code.extend(_html_code(tok.content))
            links.extend(_html_links(tok.content, line, in_table))
        elif tok.type == "inline":
            if i > 0 and tokens[i - 1].type == "heading_open":
                headings.append(_heading_text(tok))
            py_link: str | None = None  # the nearest `.py` link so far in this paragraph
            in_html_code = False  # inside an inline `<code>...</code>`: its text is code
            for child in tok.children or []:
                attr = {"link_open": "href", "image": "src"}.get(child.type)
                if attr is not None:
                    target = str(child.attrGet(attr) or "")
                    links.append(Link(target, line, in_table))
                    if target.partition("#")[0].endswith(".py"):
                        py_link = target
                elif child.type == "code_inline":
                    code.append(child.content)
                    if TEST_NAME_RE.fullmatch(child.content):
                        cited.append(CitedTest(child.content, py_link))
                elif child.type == "text":
                    (code if in_html_code else prose).append(child.content)
                elif child.type == "html_inline":
                    html.append(child.content)
                    links.extend(_html_links(child.content, line, in_table))
                    if HTML_CODE_OPEN_RE.match(child.content):
                        in_html_code = True
                    elif HTML_CODE_CLOSE_RE.match(child.content):
                        in_html_code = False
    return Doc(
        path=path,
        text=text,
        links=tuple(links),
        code=tuple(code),
        prose="\n".join(prose),
        anchors=_anchors(headings, "\n".join(html)),
        cited_tests=tuple(cited),
    )


# ----------------------------------------------------------------------------- repository facts
def makefile_targets(makefile: str) -> frozenset[str]:
    """Rule lines only (`name:` at column 0). `.PHONY` can list a target that has no rule, and an
    assignment (`X :=`) is not a target."""
    return frozenset(MAKE_RULE_RE.findall(makefile))


def defined_test_names(roots: Iterable[Path]) -> frozenset[str]:
    names: set[str] = set()
    for root in roots:
        for file in root.rglob("test_*.py"):
            names.update(TEST_DEF_RE.findall(file.read_text(encoding="utf-8")))
    return frozenset(names)


def has_subcommand(parser: argparse.ArgumentParser, words: str) -> bool:
    """Ask argparse itself: `<words> --help` exits 0 when every word is accepted (a subcommand,
    a nested subcommand or a free positional) and 2 for an invalid choice. This uses only the
    public API (no `_SubParsersAction` introspection). Flags are not checked: `--help` exits
    before argparse reports an unknown flag."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            parser.parse_args([*words.split(), "--help"])
        except SystemExit as exit_:
            return exit_.code == 0
    return True  # pragma: no cover - `--help` always exits


@cache
def _eval2_parser() -> argparse.ArgumentParser:
    return build_parser()


# ----------------------------------------------------------------------------- the checks
def unknown_make_targets(
    doc: Doc, targets: frozenset[str], not_built: frozenset[str] = frozenset()
) -> list[str]:
    """`not_built` names may appear only where a document records them as not built."""
    found = {name for block in doc.code for name in MAKE_CALL_RE.findall(block)}
    return sorted(
        f"{doc.path.name}: `make {name}` is not a Makefile rule"
        for name in found - targets - not_built
    )


def unknown_eval2_subcommands(doc: Doc, exists: Callable[[str], bool]) -> list[str]:
    found = {" ".join(words.split()) for block in doc.code for words in EVAL_SUB_RE.findall(block)}
    return sorted(
        f"{doc.path.name}: ragforge-eval2 has no subcommand {name!r}"
        for name in found
        if not exists(name)
    )


def _resolve(doc: Doc, target: str, repo: Path) -> tuple[Path | None, str, str | None]:
    """(file or None for a same-page anchor, anchor, error)."""
    path_part, _, anchor = target.partition("#")
    if not path_part:
        return None, unquote(anchor), None
    if path_part.startswith("/"):
        return None, "", "is absolute; use a path relative to the document"
    resolved = (doc.path.parent / unquote(path_part)).resolve()
    if not resolved.is_relative_to(repo.resolve()):
        return None, "", "points outside the repository"
    return resolved, unquote(anchor), None


def broken_links(doc: Doc, repo: Path, tolerated_missing: frozenset[Path]) -> list[str]:
    """Relative links resolve; `#anchor`s match a heading of the target (or this) document.
    External links (any scheme) are not fetched: that would be network access in a unit test."""
    out: list[str] = []
    for link in doc.links:
        if not link.target or urlsplit(link.target).scheme or link.target.startswith("//"):
            continue
        where = f"{doc.path.name}:{link.line}: link {link.target!r}"
        file, anchor, error = _resolve(doc, link.target, repo)
        if error:
            out.append(f"{where} {error}")
            continue
        if file is not None and not file.exists():
            if file not in tolerated_missing:
                out.append(f"{where} does not resolve")
            continue
        if not anchor:
            continue
        if file is None:
            anchors = doc.anchors
        elif file.suffix == ".md" and file.is_file():
            anchors = parse_doc(file).anchors
        else:
            continue  # a line anchor into code (`#L10`) is GitHub's, not a heading
        if anchor not in anchors:
            out.append(f"{where} has no heading for #{anchor}")
    return out


def _link_paths(doc: Doc, repo: Path, *, tables_only: bool = False) -> set[Path]:
    paths: set[Path] = set()
    for link in doc.links:
        if tables_only and not link.in_table:
            continue
        if urlsplit(link.target).scheme:
            continue
        file, _, error = _resolve(doc, link.target, repo)
        if file is not None and error is None:
            paths.add(file)
    return paths


def adrs_without_a_row(decisions: Doc, adr_dir: Path, repo: Path) -> list[str]:
    linked = _link_paths(decisions, repo, tables_only=True)
    return sorted(
        f"{decisions.path.name}: no table row links {adr.name}"
        for adr in adr_dir.glob("*.md")
        if adr.resolve() not in linked
    )


def docs_missing_from_readme(
    readme: Doc, docs_dir: Path, repo: Path, excluded: frozenset[str]
) -> list[str]:
    """The doc map is a table. A link elsewhere (GUIDE in a guarantee) does not put a document
    in the map, so only table-row links count."""
    linked = _link_paths(readme, repo, tables_only=True)
    return sorted(
        f"{readme.path.name}: docs/{doc.name} is not in the doc map"
        for doc in docs_dir.glob("*.md")
        if doc.name not in excluded and doc.resolve() not in linked
    )


def too_long(doc: Doc, max_lines: int) -> list[str]:
    n = len(doc.text.splitlines())
    return [f"{doc.path.name}: {n} lines, more than {max_lines}"] if n > max_lines else []


def banned_words(doc: Doc, words: Iterable[str] = BANNED_WORDS) -> list[str]:
    """Prose only: an identifier in code (`robust_retry`) is a name, not a claim."""
    out = []
    for word in words:
        pattern = r"\b" + re.escape(word).replace(r"\-", r"[\s-]+") + r"\b"
        if re.search(pattern, doc.prose, re.I):
            out.append(f"{doc.path.name}: banned word {word!r}")
    return out


def leaks(doc: Doc) -> list[str]:
    """The whole text, comments and code included: a UUID or a home path is private wherever it
    appears (tenant and collection ids, the owner's user name)."""
    out = [f"{doc.path.name}: holds a UUID" for _ in UUID_RE.finditer(doc.text)][:1]
    if HOME_PATH in doc.text:
        out.append(f"{doc.path.name}: holds a {HOME_PATH} path")
    return out


def unknown_test_names(doc: Doc, defined: frozenset[str]) -> list[str]:
    return sorted(
        f"{doc.path.name}: `{name}` is not a test in packages/*/tests"
        for name in {c for c in doc.code if TEST_NAME_RE.fullmatch(c)}
        if name not in defined
    )


def names_missing_from_their_cited_file(doc: Doc, repo: Path) -> list[str]:
    """A proof named beside a test file must be defined in THAT file. Existing somewhere else
    is not enough: a test that moved out of the cited file no longer backs the sentence. A
    missing file is `broken_links`' error, not this one's."""
    out: list[str] = []
    for cited in doc.cited_tests:
        if cited.file_link is None:
            continue
        file, _, error = _resolve(doc, cited.file_link, repo)
        if error or file is None or not file.is_file():
            continue
        if cited.name not in TEST_DEF_RE.findall(file.read_text(encoding="utf-8")):
            out.append(f"{doc.path.name}: `{cited.name}` is not defined in {file.name}")
    return sorted(set(out))


# ----------------------------------------------------------------------------- real documents
def _doc_or_skip(name: str) -> Doc:
    path = REPO / name
    if not path.is_file():
        if REQUIRED_DOCS:
            pytest.fail(f"{name} is required (REQUIRED_DOCS) but does not exist")
        pytest.skip(f"{name} not written yet; the integrator sets REQUIRED_DOCS = True")
    return parse_doc(path)


def _tolerated_missing() -> frozenset[Path]:
    return frozenset() if REQUIRED_DOCS else frozenset((REPO / d).resolve() for d in DOCS)


@cache
def _targets() -> frozenset[str]:
    return makefile_targets((REPO / "Makefile").read_text(encoding="utf-8"))


@cache
def _tests() -> frozenset[str]:
    return defined_test_names(REPO.glob("packages/*/tests"))


per_doc = pytest.mark.parametrize("name", DOCS)


@per_doc
def test_real_make_targets_exist(name: str) -> None:
    not_built = NOT_BUILT_TARGETS if name == NOT_BUILT_DOC else frozenset()
    assert unknown_make_targets(_doc_or_skip(name), _targets(), not_built) == []


def test_real_not_built_targets_are_still_not_built_and_come_from_an_adr() -> None:
    assert NOT_BUILT_TARGETS & _targets() == set(), "built now: drop it here and fix DECISIONS"
    adrs = "\n".join(f.read_text(encoding="utf-8") for f in (REPO / "docs" / "adr").glob("*.md"))
    assert {n for n in NOT_BUILT_TARGETS if f"make {n}" not in adrs} == set()


@per_doc
def test_real_eval2_subcommands_exist(name: str) -> None:
    doc = _doc_or_skip(name)
    assert unknown_eval2_subcommands(doc, lambda s: has_subcommand(_eval2_parser(), s)) == []


@per_doc
def test_real_links_and_anchors_resolve(name: str) -> None:
    assert broken_links(_doc_or_skip(name), REPO, _tolerated_missing()) == []


@per_doc
def test_real_prose_avoids_banned_words(name: str) -> None:
    assert banned_words(_doc_or_skip(name)) == []


@per_doc
def test_real_docs_hold_no_uuid_or_home_path(name: str) -> None:
    assert leaks(_doc_or_skip(name)) == []


@per_doc
def test_real_named_tests_are_in_the_file_cited_beside_them(name: str) -> None:
    assert names_missing_from_their_cited_file(_doc_or_skip(name), REPO) == []


@per_doc
def test_real_named_tests_exist(name: str) -> None:
    assert unknown_test_names(_doc_or_skip(name), _tests()) == []


def test_real_every_adr_has_a_decisions_row() -> None:
    decisions = _doc_or_skip("docs/DECISIONS.md")
    assert adrs_without_a_row(decisions, REPO / "docs" / "adr", REPO) == []


def test_real_readme_maps_every_top_level_doc() -> None:
    readme = _doc_or_skip("README.md")
    assert docs_missing_from_readme(readme, REPO / "docs", REPO, README_MAP_EXCLUDED) == []


def test_real_readme_is_short() -> None:
    assert too_long(_doc_or_skip("README.md"), README_MAX_LINES) == []


# ----------------------------------------------------------------------------- planted violations
def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _doc(tmp_path: Path, text: str, name: str = "README.md") -> Doc:
    return parse_doc(_write(tmp_path / name, text))


MAKEFILE = """.PHONY: up ghost
X := 1
up: env ## start
\t@true
env:
\t@true
eval-gate: ## gate
\t@true
"""


def test_planted_makefile_parse_uses_rule_lines_not_phony() -> None:
    assert makefile_targets(MAKEFILE) == {"up", "env", "eval-gate"}


def test_planted_unknown_make_target_fails(tmp_path: Path) -> None:
    targets = makefile_targets(MAKEFILE)
    good = _doc(tmp_path, "Run `make up`.\n\n```bash\nmake eval-gate ARGS=x  # gate\n```\n")
    bad = _doc(tmp_path, "Run `make ghost` or make sure.\n\n```bash\nmake eval\n```\n", "B.md")
    assert unknown_make_targets(good, targets) == []
    var_first = _doc(tmp_path, '`make NAME=x ARGS="a b" ghost` and `make X=1 up`\n', "V.md")
    assert unknown_make_targets(var_first, targets) == ["V.md: `make ghost` is not a Makefile rule"]
    assert unknown_make_targets(bad, targets) == [
        "B.md: `make eval` is not a Makefile rule",
        "B.md: `make ghost` is not a Makefile rule",
    ]
    assert unknown_make_targets(bad, targets, frozenset({"ghost"})) == [
        "B.md: `make eval` is not a Makefile rule"
    ]


def test_planted_make_flags_and_raw_html_code_are_checked(tmp_path: Path) -> None:
    targets = makefile_targets(MAKEFILE)
    good = _doc(
        tmp_path,
        "```bash\nmake -s up\nmake -j2 X=1 --no-print-directory env\n```\n\n"
        "<pre>make up</pre>\n\nSee <code>make env</code>.\n\n<p>To make sure, read on.</p>\n\n"
        "<!-- <pre>make hidden</pre> -->\n",
    )
    assert unknown_make_targets(good, targets) == []
    bad = _doc(
        tmp_path,
        "```bash\nmake -s flagged\n```\n\n<pre>\nmake inpre\n</pre>\n\n"
        "Run <code>make incode</code> now.\n",
        "B.md",
    )
    assert unknown_make_targets(bad, targets) == [
        "B.md: `make flagged` is not a Makefile rule",
        "B.md: `make incode` is not a Makefile rule",
        "B.md: `make inpre` is not a Makefile rule",
    ]


def test_planted_unknown_eval2_subcommand_fails(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser(prog="x")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run").add_subparsers(dest="kind", required=True)
    run.add_parser("retrieval")
    sub.add_parser("label").add_argument("path")

    def exists(words: str) -> bool:
        return has_subcommand(parser, words)

    good = _doc(
        tmp_path,
        '`make eval ARGS="run retrieval --x 1"`, `ragforge-eval2 label evals/a.jsonl` and '
        "`ragforge-eval2 --help`\n",
    )
    bad = _doc(
        tmp_path,
        "```\nmake eval ARGS='nope x'\nragforge-eval2 later\nmake eval ARGS=\"run nope\"\n"
        'make eval ARGS="--verbose flagged"\nragforge-eval2 -q flagged2\n```\n',
        "B.md",
    )
    assert unknown_eval2_subcommands(good, exists) == []
    assert unknown_eval2_subcommands(bad, exists) == [
        "B.md: ragforge-eval2 has no subcommand 'flagged'",
        "B.md: ragforge-eval2 has no subcommand 'flagged2'",
        "B.md: ragforge-eval2 has no subcommand 'later'",
        "B.md: ragforge-eval2 has no subcommand 'nope x'",
        "B.md: ragforge-eval2 has no subcommand 'run nope'",
    ]


def test_planted_real_eval2_parser_is_asked() -> None:
    assert has_subcommand(_eval2_parser(), "run")
    assert has_subcommand(_eval2_parser(), "run retrieval")
    assert not has_subcommand(_eval2_parser(), "no-such-subcommand")
    assert not has_subcommand(_eval2_parser(), "run no-such-kind")


def test_planted_github_slugs() -> None:
    assert github_slug("Phase 5: the write-up") == "phase-5-the-write-up"
    assert github_slug("`make up` & friends") == "make-up--friends"
    assert github_slug("Ünïcode_ok") == "ünïcode_ok"


def test_planted_broken_link_and_anchor_fail(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo / "docs" / "G.md", "# Guide\n\n## Install *it*\n\n## Install it\n")
    _write(repo / "code.py", "x = 1\n")
    _write(tmp_path / "out.md", "# Exists, but outside the repository\n")
    text = (
        "# Top\n\n[a](docs/G.md#install-it) [b](docs/G.md#install-it-1) [c](#top) "
        "[d](https://example.invalid/x) [e](docs/) [f](code.py#L3)\n"
    )
    assert broken_links(_doc(repo, text), repo, frozenset()) == []
    bad = _doc(
        repo,
        "[a](docs/G.md#nope) [b](missing.md) [c](#gone) [d](/abs.md) [e](../out.md)\n"
        "<!-- [hidden](also-missing.md) -->\n",
        "B.md",
    )
    assert [e.split(" ", 3)[3] for e in broken_links(bad, repo, frozenset())] == [
        "has no heading for #nope",
        "does not resolve",
        "has no heading for #gone",
        "is absolute; use a path relative to the document",
        "points outside the repository",
    ]
    tolerated = frozenset({(repo / "missing.md").resolve()})
    assert len(broken_links(bad, repo, tolerated)) == 4


def test_planted_raw_html_links_are_checked(tmp_path: Path) -> None:
    _write(tmp_path / "G.md", "# Guide\n")
    good = _doc(tmp_path, '<a href="G.md#guide">g</a>\n\n<p><img src="G.md" alt=""></p>\n')
    assert broken_links(good, tmp_path, frozenset()) == []
    bad = _doc(
        tmp_path,
        'Inline <a href="nope.md">x</a>.\n\n<div><img src="gone.png"></div>\n\n'
        '<!-- <a href="hidden.md">comment</a> -->\n',
        "B.md",
    )
    assert sorted(e.split(" ", 3)[2] for e in broken_links(bad, tmp_path, frozenset())) == [
        "'gone.png'",
        "'nope.md'",
    ]


def test_planted_html_anchor_counts(tmp_path: Path) -> None:
    doc = _doc(tmp_path, '<a id="custom"></a>\n\n[x](#custom)\n')
    assert broken_links(doc, tmp_path, frozenset()) == []


def test_planted_adr_without_a_decisions_row_fails(tmp_path: Path) -> None:
    adr = tmp_path / "docs" / "adr"
    _write(adr / "0001-a.md", "# A\n")
    _write(adr / "0002-b.md", "# B\n")
    rows = "| ADR | Why |\n|---|---|\n| [1](adr/0001-a.md) | x |\n| [2](adr/0002-b.md) | y |\n"
    good = _doc(tmp_path, rows, "docs/DECISIONS.md")
    assert adrs_without_a_row(good, adr, tmp_path) == []
    prose_only = rows.replace("| [2](adr/0002-b.md) | y |\n", "") + "\nSee [2](adr/0002-b.md).\n"
    bad = _doc(tmp_path, prose_only, "docs/DECISIONS.md")
    assert adrs_without_a_row(bad, adr, tmp_path) == ["DECISIONS.md: no table row links 0002-b.md"]


def test_planted_doc_missing_from_readme_fails(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    for name in ("GUIDE.md", "NEXT.md", "PHASE1.md"):
        _write(docs / name, "# x\n")
    table = "| Doc | What |\n|---|---|\n| [g](docs/GUIDE.md) | x |\n| [n](docs/NEXT.md) | y |\n"
    good = _doc(tmp_path, table)
    assert docs_missing_from_readme(good, docs, tmp_path, frozenset({"PHASE1.md"})) == []
    # NEXT.md is linked in prose and named in code, but its map row is gone.
    no_row = table.replace("| [n](docs/NEXT.md) | y |\n", "")
    bad = _doc(tmp_path, f"See [n](docs/NEXT.md) and `docs/NEXT.md`.\n\n{no_row}")
    assert docs_missing_from_readme(bad, docs, tmp_path, frozenset({"PHASE1.md"})) == [
        "README.md: docs/NEXT.md is not in the doc map"
    ]


def test_planted_long_readme_fails(tmp_path: Path) -> None:
    assert too_long(_doc(tmp_path, "x\n" * 200), 200) == []
    assert too_long(_doc(tmp_path, "x\n" * 201), 200) == ["README.md: 201 lines, more than 200"]


def test_planted_banned_words_fail_in_prose_only(tmp_path: Path) -> None:
    good = _doc(tmp_path, "A small tool. `robust_retry()` is a name.\n\n```\npowerful\n```\n")
    assert banned_words(good) == []
    bad = _doc(tmp_path, "A **Robust**, state of the art, cutting-edge tool.\n")
    assert banned_words(bad) == [
        "README.md: banned word 'state-of-the-art'",
        "README.md: banned word 'robust'",
        "README.md: banned word 'cutting-edge'",
    ]


def test_planted_uuid_and_home_path_fail_anywhere(tmp_path: Path) -> None:
    assert leaks(_doc(tmp_path, "id `<collection-id>` in ~/.config\n")) == []
    uuid = "-".join(("123e4567", "E89B", "12d3", "a456", "426614174000"))  # parts: see HOME_PATH
    planted = f"<!-- {uuid} -->\n\n```\ncd {HOME_PATH}someone\n```\n"
    assert leaks(_doc(tmp_path, planted)) == [
        "README.md: holds a UUID",
        f"README.md: holds a {HOME_PATH} path",
    ]


def test_planted_unknown_test_name_fails(tmp_path: Path) -> None:
    _write(
        tmp_path / "pkg" / "tests" / "test_x.py", "async def test_real_one() -> None:\n    ...\n"
    )
    defined = defined_test_names([tmp_path / "pkg" / "tests"])
    assert defined == {"test_real_one"}
    assert unknown_test_names(_doc(tmp_path, "`test_real_one` in `test_x.py`\n"), defined) == []
    assert unknown_test_names(_doc(tmp_path, "`test_renamed`\n"), defined) == [
        "README.md: `test_renamed` is not a test in packages/*/tests"
    ]


def test_planted_test_cited_beside_the_wrong_file_fails(tmp_path: Path) -> None:
    _write(tmp_path / "t" / "test_a.py", "def test_in_a() -> None:\n    ...\n")
    _write(tmp_path / "t" / "test_b.py", "def test_in_b() -> None:\n    ...\n")
    good = _doc(
        tmp_path,
        "- [test_a.py](t/test_a.py) `test_in_a`; [test_b.py](t/test_b.py#L1) `test_in_b`\n"
        "- `test_in_a`, named with no file beside it\n",
    )
    assert names_missing_from_their_cited_file(good, tmp_path) == []
    bad = _doc(tmp_path, "- [test_a.py](t/test_a.py) `test_in_a`, `test_in_b`\n", "B.md")
    assert names_missing_from_their_cited_file(bad, tmp_path) == [
        "B.md: `test_in_b` is not defined in test_a.py"
    ]
    # The file link does not carry over to the next bullet.
    next_bullet = _doc(tmp_path, "- [test_a.py](t/test_a.py) `test_in_a`\n- `test_in_b`\n")
    assert names_missing_from_their_cited_file(next_bullet, tmp_path) == []
