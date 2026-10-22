"""The egress guard plugin (packages/core/tests/egress_guard.py; ADR-0018 R2-8).

Two levels: the guard itself in this process (what it refuses, what it lets through, that it
always restores the socket layer), and the plugin end to end in a child pytest, because the
property that matters (a test FAILS even when its code swallowed the refusal) only exists once
pytest's hooks run it. Every non-loopback address used here is TEST-NET-3 (203.0.113.0/24,
RFC 5737: documentation only, never routed), so a guard bug cannot reach a real host.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import textwrap
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
import pytest

import egress_guard
from egress_guard import EgressRefused, guarded, is_local_address, is_local_host

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
UNROUTABLE = "203.0.113.7"  # TEST-NET-3


@pytest.mark.parametrize(
    "host",
    ["localhost", "LOCALHOST", "127.0.0.1", "127.8.9.1", "::1", "[::1]", "::ffff:127.0.0.1",
     "0.0.0.0", "::", ""],  # noqa: S104 - a destination here, not a bind address
)  # fmt: skip
def test_loopback_and_unspecified_hosts_are_local(host: str) -> None:
    assert is_local_host(host)


@pytest.mark.parametrize(
    "host",
    [UNROUTABLE, "10.0.0.1", "192.168.1.10", "169.254.169.254", "::ffff:8.8.8.8", "2001:db8::1",
     "fe80::1%en0", "host.docker.internal", "api.notion.com", "localhost.attacker.example", None],
)  # fmt: skip
def test_everything_else_is_not_local(host: str | None) -> None:
    """Private ranges and the cloud metadata address are egress too: host.docker.internal is
    the Mac seen from the VM, never something a unit test needs."""
    assert not is_local_host(host)


def test_address_families() -> None:
    assert is_local_address(socket.AF_UNIX, "/var/run/docker.sock")
    assert is_local_address(socket.AF_INET, ("127.0.0.1", 5432))
    assert is_local_address(socket.AF_INET6, ("::1", 5432, 0, 0))
    assert is_local_address(socket.AF_INET, (b"localhost", 80))
    assert not is_local_address(socket.AF_INET, (UNROUTABLE, 443))
    assert not is_local_address(socket.AF_INET6, ("2001:db8::1", 443, 0, 0))
    assert not is_local_address(-1, ("127.0.0.1", 80))  # unknown family: fail closed


def test_loopback_connections_still_work() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server, guarded() as guard:
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        assert socket.getaddrinfo("localhost", port)
    assert guard.refused == []


def test_non_loopback_connect_is_refused_before_any_packet() -> None:
    """Refused at once (a real connect to TEST-NET would hang until the timeout)."""
    with guarded() as guard, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)
        started = time.monotonic()
        with pytest.raises(EgressRefused):
            sock.connect((UNROUTABLE, 443))
        assert time.monotonic() - started < 0.5
        with pytest.raises(EgressRefused):
            sock.connect_ex((UNROUTABLE, 443))
    assert len(guard.refused) == 2


def test_udp_sendto_and_name_lookups_are_refused() -> None:
    with guarded() as guard, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(EgressRefused):
            sock.sendto(b"x", (UNROUTABLE, 53))
        with pytest.raises(EgressRefused):
            sock.sendto(b"x", 0, (UNROUTABLE, 53))
        with pytest.raises(EgressRefused):
            socket.getaddrinfo("api.notion.com", 443)  # never resolved: DNS is egress too
    assert len(guard.refused) == 3


def test_sendmsg_and_the_gethostby_family_are_refused() -> None:
    """Separate C entry points that bypass connect/sendto/getaddrinfo: each needs its own wrap."""
    with guarded() as guard, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(EgressRefused):
            sock.sendmsg([b"x"], [], 0, (UNROUTABLE, 9))
        with pytest.raises(EgressRefused):
            socket.gethostbyname("api.notion.com")
        with pytest.raises(EgressRefused):
            socket.gethostbyname_ex("api.notion.com")
        with pytest.raises(EgressRefused):
            socket.gethostbyaddr(UNROUTABLE)
        with pytest.raises(EgressRefused):
            socket.getnameinfo((UNROUTABLE, 443), 0)
    assert len(guard.refused) == 5


def test_loopback_sendmsg_and_lookups_still_work() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client,
        guarded() as guard,
    ):
        server.bind(("127.0.0.1", 0))
        assert client.sendmsg([b"x"], [], 0, server.getsockname()) == 1
        client.connect(server.getsockname())
        assert client.sendmsg([b"y"]) == 1  # connected socket: no address to check
        assert socket.gethostbyname("localhost")
        assert socket.getnameinfo(("127.0.0.1", 80), socket.NI_NUMERICHOST)[0] == "127.0.0.1"
    assert guard.refused == []


def test_async_httpx_reaches_the_guard() -> None:
    """The real path: httpx -> anyio -> asyncio. The library wraps the refusal in its own
    ConnectError; the guard still recorded it, which is what fails the test in the plugin."""

    async def fetch() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get("https://api.notion.com/v1/users/me")

    with guarded() as guard, pytest.raises(httpx.ConnectError):
        asyncio.run(fetch())
    assert guard.refused
    assert "api.notion.com" in guard.refused[0]


def test_the_socket_layer_is_restored_even_when_the_block_raises() -> None:
    """Restored to exactly what was there before: the C methods when nothing else patched them,
    the plugin's own guard when this test itself runs under it (guards nest)."""
    names = ("connect", "connect_ex", "sendto", "sendmsg")
    lookups = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")
    before = {name: socket.socket.__dict__.get(name) for name in names}
    lookups_before = {name: getattr(socket, name) for name in lookups}

    def fail_inside_the_guard() -> None:
        with guarded():
            assert all(socket.socket.__dict__[name] is not before[name] for name in names)
            assert all(getattr(socket, name) is not lookups_before[name] for name in lookups)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fail_inside_the_guard()
    assert {name: socket.socket.__dict__.get(name) for name in names} == before
    assert {name: getattr(socket, name) for name in lookups} == lookups_before


def _root_pytest_ini() -> dict[str, Any]:
    with (REPO / "pyproject.toml").open("rb") as f:
        ini: dict[str, Any] = tomllib.load(f)["tool"]["pytest"]["ini_options"]
    return ini


def test_root_pyproject_registers_the_plugin(pytestconfig: pytest.Config) -> None:
    """Tripwire: once registered, removing the registration fails here instead of silently
    turning the guard and the LangSmith pin off for every test. Registered at the ROOT so a
    targeted run of another package (`pytest packages/contracts`) is guarded too."""
    ini = _root_pytest_ini()
    assert registers_the_plugin(ini.get("addopts"))
    assert "packages/core/tests" in ini.get("pythonpath", [])
    assert pytestconfig.pluginmanager.has_plugin("egress_guard")


def registers_the_plugin(addopts: object) -> bool:
    """`-p egress_guard` appears as an adjacent pair in addopts (a list, or one string), so
    other modules' options may sit before or after it (handoff step 1 says to append)."""
    if isinstance(addopts, str):
        words = addopts.split()
    elif isinstance(addopts, list):
        words = [str(word) for word in addopts]
    else:
        words = []
    return any(words[i : i + 2] == ADDOPTS for i in range(len(words) - 1))


def test_the_tripwire_accepts_appended_addopts() -> None:
    assert registers_the_plugin(["-ra", "-p", "egress_guard", "--strict-markers"])
    assert registers_the_plugin("-ra -p egress_guard")
    assert registers_the_plugin(ADDOPTS)
    for missing in (None, [], ["-p", "other", "egress_guard"], ["egress_guard", "-p"], "-p x"):
        assert not registers_the_plugin(missing), missing


# Until conftest registers the plugin (handoff step 1), its marker is unknown to this run.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
def test_opt_outs(request: pytest.FixtureRequest) -> None:
    item = request.node
    assert isinstance(item, pytest.Item)
    assert not egress_guard.guard_disabled(item, env={})
    assert egress_guard.guard_disabled(item, env={egress_guard.DISABLE_ENV: "off"})
    assert not egress_guard.guard_disabled(item, env={egress_guard.DISABLE_ENV: "on"})
    item.add_marker(egress_guard.MARKER)
    assert egress_guard.guard_disabled(item, env={})


# ---------------------------------------------------------------- the plugin, end to end

# The exact registration the integrator adds to the root pyproject.toml (handoff step 1); the
# child run below uses the same two keys, with pythonpath pointing at this directory.
ADDOPTS = ["-p", "egress_guard"]


def _pyproject(pythonpath: Path) -> str:
    addopts = ", ".join(f'"{opt}"' for opt in ADDOPTS)
    return (
        "[tool.pytest.ini_options]\n"
        f"addopts = [{addopts}]\n"
        f'pythonpath = ["{pythonpath.as_posix()}"]\n'
    )


INNER_TESTS = f'''
import os, socket
import pytest

UNROUTABLE = "{UNROUTABLE}"

@pytest.fixture
def leaky_fixture():
    try:
        socket.create_connection((UNROUTABLE, 443), timeout=1).close()
    except OSError:
        pass
    yield

def test_swallowed_refusal_still_fails():
    try:
        socket.create_connection((UNROUTABLE, 443), timeout=1).close()
    except OSError:
        pass  # a degrade path that hides the failure: the guard must not be fooled

def test_leaky_fixture_errors(leaky_fixture):
    pass

def test_loopback_passes():
    with socket.create_server(("127.0.0.1", 0)) as server:
        socket.create_connection(server.getsockname(), timeout=1).close()

@pytest.mark.allow_egress
def test_marked_test_is_not_guarded():
    assert "connect" not in socket.socket.__dict__

def test_guard_is_active():
    assert "connect" in socket.socket.__dict__

def test_langsmith_pinned_off():
    from langsmith import utils
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ
    assert utils.tracing_is_enabled() is False
'''


def _run_child(tmp_path: Path, extra_env: dict[str, str]) -> dict[str, str]:
    """Run INNER_TESTS in a child pytest whose root pyproject registers the plugin exactly as the
    handoff says. The tests sit in a package dir of their own with NO conftest, and only that
    dir is passed: what `pytest packages/contracts` does. Returns {test name: outcome} from the
    child's JUnit XML."""
    (tmp_path / "pyproject.toml").write_text(_pyproject(TESTS_DIR))
    package_tests = tmp_path / "packages" / "other" / "tests"
    package_tests.mkdir(parents=True)
    (package_tests / "test_inner.py").write_text(textwrap.dedent(INNER_TESTS))
    report = tmp_path / "report.xml"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        # A shell that had tracing on: the plugin must override it (synthetic values only).
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGSMITH_API_KEY": "lsv2-synthetic-not-a-key",
        **extra_env,
    }
    subprocess.run(  # noqa: S603 - fixed argv: this interpreter running pytest on tmp_path
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir",
         str(tmp_path), f"--junitxml={report}", str(package_tests)],
        cwd=tmp_path, env=env, capture_output=True, timeout=120, check=False,
    )  # fmt: skip
    outcomes: dict[str, str] = {}
    # S314: the XML is the report our own child pytest just wrote, not untrusted input.
    for case in ET.parse(report).getroot().iter("testcase"):  # noqa: S314
        kinds = {child.tag for child in case} & {"failure", "error", "skipped"}
        outcomes[case.attrib["name"]] = kinds.pop() if kinds else "passed"
    return outcomes


def test_plugin_fails_tests_that_leave_the_machine(tmp_path: Path) -> None:
    outcomes = _run_child(tmp_path, {})
    assert outcomes == {
        "test_swallowed_refusal_still_fails": "failure",
        "test_leaky_fixture_errors": "error",
        "test_loopback_passes": "passed",
        "test_marked_test_is_not_guarded": "passed",
        "test_guard_is_active": "passed",
        "test_langsmith_pinned_off": "passed",
    }


def test_plugin_can_be_switched_off_for_a_whole_run(tmp_path: Path) -> None:
    """RAGFORGE_TEST_EGRESS_GUARD=off leaves the socket layer alone (Docker on a remote host).
    The leaky tests then really try TEST-NET-3 and time out after 1 s, harmlessly."""
    outcomes = _run_child(tmp_path, {egress_guard.DISABLE_ENV: "off"})
    assert outcomes["test_guard_is_active"] == "failure"
    assert outcomes["test_swallowed_refusal_still_fails"] == "passed"
    assert outcomes["test_langsmith_pinned_off"] == "passed"  # tracing stays pinned regardless
