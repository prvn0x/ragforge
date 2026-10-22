"""Egress inventory (ADR-0018 §Security 4, R2-8): every outbound destination is listed here.

The claim in docs/SECURITY.md is "RAGForge's code talks to Notion, Gemini and the local Ollama,
and nothing else". This file makes it a test: it collects every URL literal in the shipped code
(packages/*/src, scripts, docker, compose and the Makefile) plus the URL-valued Settings
defaults, and compares the non-local hosts with `INVENTORY` EXACTLY. A new host, a known host
used from a new file, or a stale entry each fail, so adding an outbound destination is always a
reviewed edit of this table and of SECURITY.md §Egress together.

Any scheme counts (`postgresql://`, `grpc://`, ...), not only http. Local destinations
(loopback, the Mac's Ollama via host.docker.internal, dotless service names such as `postgres`
or `prometheus`) need no entry. A host a template builds at runtime (`http://{host}:{port}`)
carries no literal; it comes from Settings or CLI flags, which
`test_host_settings_default_to_this_machine` covers. A literal host with a template after it
(`https://x.org:{port}`, `https://x.org{path}`) IS reported, and a host only partly literal
(`https://api.{region}.x.org`) is reported as non-local: the scanner fails closed.
"""

from __future__ import annotations

import ipaddress
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

from egress_guard import is_local_host
from ragforge_client.evals.judge import JudgeConfig
from ragforge_core.config import Settings

REPO: Final = Path(__file__).resolve().parents[3]
SCAN_DIRS: Final = ("packages", "scripts", "docker")
SCAN_FILES: Final = ("Makefile",)
SKIP_PARTS: Final = frozenset({"tests", "__pycache__", "node_modules"})
TEXT_SUFFIXES: Final = frozenset(
    {".py", ".sh", ".sql", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".conf", ".txt", ""}
)

# A placeholder filled in at runtime: f-string/format `{x}`, shell `${X}` / `$X`, docs `<host>`.
_TEMPLATE: Final = r"(?:\{[^{}\s]*\}|\$\{[^{}\s]*\}|\$[A-Za-z_]\w*|<[A-Za-z_][\w-]*>)"
_TEMPLATE_RE: Final = re.compile(_TEMPLATE)
_TRAILING_TEMPLATES: Final = re.compile(rf"{_TEMPLATE}+$")
# scheme://authority, any scheme (RFC 3986). The authority runs over literal characters AND
# templates, so a template neither hides the literal host before it nor the one after a
# templated password (`postgresql://u:${PW}@db.example.org`); it stops at a path, quote or space.
# The lookbehind only stops a match from starting inside a word. It must NOT contain `-`, `+` or
# `.`: `${X:-https://h}` / `${X:+https://h}` (shell/compose defaults) put one right before the
# scheme, and a match that starts mid-scheme (`x.https://h`) still reports the right host.
_URL: Final = re.compile(
    rf"(?<![A-Za-z0-9])(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://"
    rf"(?P<authority>(?:{_TEMPLATE}|[A-Za-z0-9._~%:@\[\]-])*)"
)
# Our own MCP resource URIs (`ragforge://documents/{id}`): names, never a network destination.
_INERT_SCHEMES: Final = frozenset({"ragforge"})

Kind = Literal["egress", "build", "inert"]


@dataclass(frozen=True, slots=True)
class Destination:
    kind: Kind  # egress: the running system connects; build: image build only; inert: never
    why: str
    files: frozenset[str]


def _files(*paths: str) -> frozenset[str]:
    return frozenset(paths)


# THE inventory. Keep docs/SECURITY.md §Egress in step with it.
INVENTORY: Final[dict[str, Destination]] = {
    "api.notion.com": Destination(
        "egress", "Notion connector: the owner's pages, with the integration token (ADR-0007)",
        _files("packages/core/src/ragforge_core/adapters/connectors/notion/client.py"),
    ),
    "generativelanguage.googleapis.com": Destination(
        "egress",
        "Gemini, only when chosen: chat adapter (RAGFORGE_LLM_PROVIDER=gemini), the eval judge "
        "with --allow-third-party, and `make setup`'s one key check",
        _files(
            "packages/core/src/ragforge_core/adapters/llm/chat_gemini.py",
            "packages/client/src/ragforge_client/evals/judge.py",
            "scripts/setup_provider.py",
        ),
    ),
    "huggingface.co": Destination(
        "build", "reranker model, pinned commit + sha256 checksum, image build only",
        _files("docker/Dockerfile"),
    ),
    "www.notion.so": Destination(
        "inert", "deep links built from page URLs Notion returns; the owner's browser opens them",
        _files("packages/core/src/ragforge_core/adapters/connectors/notion/ids.py"),
    ),
    "schemas.openxmlformats.org": Destination(
        "inert", "the .docx XML namespace name, never fetched",
        _files("packages/core/src/ragforge_core/adapters/connectors/files/_sandbox.py"),
    ),
    "aistudio.google.com": Destination(
        "inert", "a link `make setup` prints for the owner to create a Gemini key",
        _files("scripts/setup_provider.py"),
    ),
}  # fmt: skip

# The Mac's host Ollama as the containers see it (Rancher/Docker Desktop resolve it; CLAUDE.md).
DOCKER_HOST_ALIAS: Final = "host.docker.internal"
# RFC 2606/6761 names reserved for documentation and tests; they cannot resolve to anything.
_RESERVED_SUFFIXES: Final = (".example", ".invalid", ".test", ".localhost")


def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def is_local(host: str) -> bool:
    """Loopback/unspecified IPs and `localhost` (the same rule the egress guard applies), the
    Docker host alias, reserved documentation names, and single-label names: a name without a
    dot is a compose/k8s service or /etc/hosts entry (`postgres`, `ollama`, `prometheus`), never
    an internet host, which keeps this test stable while other modules add compose services.
    A host that is only partly literal (`api{suffix}`) could become anything: never local."""
    if _TEMPLATE_RE.search(host):
        return False
    if is_local_host(host) or host == DOCKER_HOST_ALIAS or host.endswith(_RESERVED_SUFFIXES):
        return True
    return not is_ip_literal(host) and "." not in host


def shipped_files() -> Iterator[Path]:
    """Code and config that runs or builds the system. Tests are excluded (respx fixtures name
    real hosts without connecting), dot-dirs (caches) too; evals/ is never read (private data)."""
    yield from (REPO / name for name in SCAN_FILES)
    yield from sorted(REPO.glob("compose*.yaml"))
    for top in SCAN_DIRS:
        for path in sorted((REPO / top).rglob("*")):
            rel = path.relative_to(REPO).parts
            if any(p in SKIP_PARTS or p.startswith(".") or p.endswith(".egg-info") for p in rel):
                continue
            if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "Dockerfile"):
                yield path


def host_of_authority(authority: str) -> str | None:
    """The host part of `[userinfo@]host[:port]`, lowercased, or None when the host is empty or
    STARTS with a template (decided at runtime from configuration). Parsed by hand because
    urlsplit cannot take template characters.

    A template glued to the end of a complete dotted name (`x.org{path}`) is a path or port
    and is dropped, so the literal host is reported. Any other template inside the host
    (`api.{region}.x.org`, `api.{tld}`, `svc{n}`) is kept, and `is_local` treats a host with
    a template in it as non-local: the scan fails closed instead of guessing."""
    hostport = authority.rpartition("@")[2]
    if hostport.startswith("["):  # IPv6 literal: [::1]:8000
        host = hostport[1:].partition("]")[0]
    else:
        host = hostport.partition(":")[0]
    if not host or _TEMPLATE_RE.match(host):
        return None
    literal = _TRAILING_TEMPLATES.sub("", host)
    if "." in literal and not literal.endswith(".") and not _TEMPLATE_RE.search(literal):
        host = literal
    return host.lower()


def hosts_in(text: str) -> Iterator[str]:
    """Host names of the URL literals in `text` (any scheme but our inert `ragforge://`)."""
    for match in _URL.finditer(text):
        if match["scheme"].lower() in _INERT_SCHEMES:
            continue
        host = host_of_authority(match["authority"])
        if host:
            yield host


def non_local_hosts() -> dict[str, frozenset[str]]:
    found: defaultdict[str, set[str]] = defaultdict(set)
    for path in shipped_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for host in hosts_in(text):
            if not is_local(host):
                found[host].add(path.relative_to(REPO).as_posix())
    return {host: frozenset(files) for host, files in found.items()}


# ---------------------------------------------------------------- the scanner itself


def test_scanner_extracts_hosts_and_skips_templates() -> None:
    text = (
        'A = "https://api.notion.com/v1"; B = f"http://{host}:{port}"; '
        "C = 'http://[::1]:8000/x' D=http://user:pw@Example.ORG:8443/p "
        "E=https://huggingface.co/m/${COMMIT}.zip F=http://${HOST}/ G=wss://ws.example.net"
    )
    assert list(hosts_in(text)) == [
        "api.notion.com", "::1", "example.org", "huggingface.co", "ws.example.net",
    ]  # fmt: skip


def test_scanner_reports_a_literal_host_followed_by_a_template() -> None:
    """The common f-string shapes for a NEW host: a templated port or path after it. Skipping
    these would let `https://exfil.attacker.com:{port}` pass the inventory unseen."""
    for text in (
        'f"https://x.example.org:{port}/v1"',
        'f"https://x.example.org{path}"',
        '"https://x.example.org:${PORT}"',
        '"https://x.example.org$SUFFIX"',
        '"https://x.example.org:<port>/"',
        'f"https://user:{password}@x.example.org/"',
    ):
        assert list(hosts_in(text)) == ["x.example.org"], text


def test_scanner_sees_urls_inside_shell_default_interpolation() -> None:
    """compose.yaml already writes `${RAGFORGE_OLLAMA_BASE_URL:-http://host.docker.internal:11434}`;
    a new `${X:-https://vendor.example.com}` default must be reported like any other literal."""
    for text in (
        "X: ${RAGFORGE_X_URL:-https://x.example.org}",
        "X: ${RAGFORGE_X_URL:=https://x.example.org/v1}",
        "X: ${RAGFORGE_X_URL-https://x.example.org}",
        "X: ${RAGFORGE_X_URL:+https://x.example.org}",
        'X = "foo.https://x.example.org"',
    ):
        assert list(hosts_in(text)) == ["x.example.org"], text


def test_scanner_sees_the_compose_ollama_default() -> None:
    """The real compose file's interpolated default is scanned (and is local by rule)."""
    text = (REPO / "compose.yaml").read_text(encoding="utf-8")
    assert "host.docker.internal" in set(hosts_in(text))


def test_scanner_fails_closed_on_a_partly_templated_host() -> None:
    hosts = list(
        hosts_in('f"https://api.{region}.example.org/" f"http://svc{n}/" f"https://api.{tld}"')
    )
    assert hosts == ["api.{region}.example.org", "svc{n}", "api.{tld}"]
    assert not any(is_local(h) for h in hosts)


def test_scanner_sees_every_scheme() -> None:
    text = (
        '"postgresql://u@db.example.org/x" "postgresql+psycopg://u:pw@pg.example.net:5432/d" '
        '"grpc://g.example.com:443" "ftp://f.example.com/" "ragforge://documents/{id}" '
        "`scheme://` f'postgresql://{user}:{pw}@{host}/db'"
    )
    assert list(hosts_in(text)) == [
        "db.example.org", "pg.example.net", "g.example.com", "f.example.com",
    ]  # fmt: skip


def test_local_classification() -> None:
    for host in ("localhost", "127.0.0.1", "::1", "host.docker.internal", "postgres", "x.example"):
        assert is_local(host), host
    for host in ("api.notion.com", "10.0.0.5", "2001:db8::1", "evil.localhost.com", "ollama.ai"):
        assert not is_local(host), host


def test_scanner_sees_the_shipped_tree() -> None:
    """Guards the guard: if the walk silently found nothing (a moved directory, a wrong root),
    the inventory comparison below would pass vacuously."""
    scanned = {p.relative_to(REPO).as_posix() for p in shipped_files()}
    for must in (
        "packages/core/src/ragforge_core/adapters/connectors/notion/client.py",
        "packages/core/src/ragforge_core/config.py",
        "scripts/bootstrap.sh",
        "docker/Dockerfile",
        "compose.yaml",
        "Makefile",
    ):
        assert must in scanned, must
    assert not any("/tests/" in f or f.startswith("evals/") for f in scanned)


# ---------------------------------------------------------------- the inventory


def test_outbound_hosts_match_the_inventory_exactly() -> None:
    found = non_local_hosts()
    expected = {host: d.files for host, d in INVENTORY.items()}
    new = {h: sorted(f) for h, f in found.items() if h not in expected}
    stale = sorted(h for h in expected if h not in found)
    moved = {
        h: {"new files": sorted(found[h] - f), "gone": sorted(f - found[h])}
        for h, f in expected.items()
        if h in found and found[h] != f
    }
    assert not new, f"new outbound host(s): add to INVENTORY and docs/SECURITY.md: {new}"
    assert not moved, f"a listed host is used from different files now: {moved}"
    assert not stale, f"INVENTORY lists hosts no longer in the code; remove them: {stale}"


def test_runtime_egress_is_exactly_notion_and_gemini() -> None:
    """The one-line claim SECURITY.md makes, pinned on its own so it cannot drift by an
    innocent-looking INVENTORY edit that reclassifies a host."""
    runtime = {h for h, d in INVENTORY.items() if d.kind == "egress"}
    assert runtime == {"api.notion.com", "generativelanguage.googleapis.com"}


# A str setting with one of these name endings names a destination even without "://".
_HOST_FIELD_SUFFIXES: Final = ("_host", "_url", "_endpoint", "_addr", "_uri")


def _host_settings() -> dict[str, str]:
    """{field: host} for every str setting that is a URL or names a host by its field name."""
    hosts: dict[str, str] = {}
    for name, info in Settings.model_fields.items():
        default = info.default
        if not isinstance(default, str):
            continue
        if "://" in default:
            hosts[name] = urlsplit(default).hostname or ""
        elif name.endswith(_HOST_FIELD_SUFFIXES):
            hosts[name] = default
    return hosts


def test_host_settings_default_to_this_machine() -> None:
    """Every destination-valued setting (a URL, or a bare host like `db_host`) defaults to a
    local host, and the set is pinned: a new one (say a tracing endpoint or a remote
    `*_host`) must be added here deliberately."""
    hosts = _host_settings()
    assert set(hosts) == {"ollama_base_url", "db_host", "metrics_addr"}
    for name, host in hosts.items():
        assert host, name
        assert is_local(host), (name, host)


def test_eval_judge_is_local_and_unproxied_by_default() -> None:
    """ADR-0017 §4: the judge sees the owner's notes, so by default it is the local Ollama, a
    third party needs an explicit opt-in, and a local judge ignores HTTP(S)_PROXY (judge.py
    builds its client with trust_env=not is_local)."""
    config = JudgeConfig()
    assert config.provider == "ollama"
    assert config.is_local
    assert config.allow_third_party is False
    assert not JudgeConfig(provider="gemini").is_local
