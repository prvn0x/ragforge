"""Pytest plugin: during a test, a socket may only reach this machine (ADR-0018 R2-8).

WHY: the egress audit says RAGForge talks to exactly Notion, Gemini and the host's Ollama. A unit
test that quietly reaches the internet breaks that claim in two ways: it may send synthetic (or
worse, real) note text somewhere, and it passes or fails on the network instead of the code. So
while a test's setup, call and teardown run, this plugin refuses every `connect`/`connect_ex`,
`sendto`/`sendmsg` and name lookup (`getaddrinfo`, `gethostbyname[_ex]`, `gethostbyaddr`,
`getnameinfo`) whose destination is not loopback, and it FAILS the test even when the code under
test caught the error (httpx turns it into a ConnectError that a degrade path may swallow).

What still works: respx/httpx MockTransport (no socket at all), TestClient (in process),
testcontainers (the Docker API is a unix socket and mapped ports are on localhost), uvicorn on
127.0.0.1 (with the default asyncio loop).

What it cannot see: it patches Python's `socket` module, so anything that bypasses it is
unguarded. That covers C/Rust network stacks, namely libpq (it only reaches the test Postgres
on localhost) and uvloop (installed through uvicorn[standard]: under `uvloop.run`,
`asyncio.open_connection` resolves and connects with nothing refused; no test runs a uvloop
loop today). It also covers child processes (a subprocess pytest, uv, curl or docker), a lookup
function bound by `from socket import gethostbyname` before the test started (the patch
replaces module attributes), and anything that runs at import or collection time.

Opt-outs, both deliberate and visible in review:
  - `@pytest.mark.allow_egress`: one test that legitimately needs the network (none today);
  - `RAGFORGE_TEST_EGRESS_GUARD=off`: the whole run, e.g. Docker on a remote TCP host.

It also pins LangSmith tracing OFF for the test process (`pin_langsmith_off`): `langsmith` is
installed as a pytest plugin, and a LANGSMITH_TRACING=true left in the owner's shell would
otherwise send traces of every LangChain call made by a test.

Registered for every pytest run in the repo by the root pyproject.toml
(`addopts = ["-p", "egress_guard"]`, `pythonpath = ["packages/core/tests"]`; see
docs/handoff/SEC-THREAT.md), so `pytest packages/contracts` alone is guarded too. Ad hoc:
`PYTHONPATH=packages/core/tests uv run pytest -p egress_guard ...`.
"""

from __future__ import annotations

import importlib
import ipaddress
import os
import socket
from collections.abc import Buffer, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

import pytest

MARKER: Final = "allow_egress"
DISABLE_ENV: Final = "RAGFORGE_TEST_EGRESS_GUARD"
_DISABLED_VALUES: Final = frozenset({"off", "0", "false", "no"})
_LOCAL_NAMES: Final = frozenset({"localhost", "localhost.", "ip6-localhost"})

# LangSmith reads LANGSMITH_* first, then LANGCHAIN_* (langsmith.utils.get_env_var).
TRACING_OFF_ENV: Final = {
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_TRACING_V2": "false",
    "LANGCHAIN_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}
TRACING_CREDENTIAL_ENV: Final = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
)


class EgressRefused(ConnectionRefusedError):
    """Raised instead of opening a non-loopback connection. An OSError subclass on purpose: the
    code under test sees an ordinary, immediate network failure instead of a hang."""


def _host_of(address: object) -> str | None:
    """The host part of a socket address: `(host, port)`, `(host, port, flow, scope)` or a bare
    host. Bytes are decoded."""
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return None
    return host


def is_local_host(host: str | None) -> bool:
    """True for loopback (127/8, ::1, IPv4-mapped loopback), the unspecified address (the
    kernel routes a connect to 0.0.0.0 / :: to this host) and the literal name `localhost`.
    Any other NAME is refused without resolving it: resolving would itself be DNS egress."""
    if host is None:
        return False
    host = host.strip("[]").split("%", 1)[0]  # IPv6 brackets, zone id (fe80::1%en0)
    if host == "":  # Python's spelling of INADDR_ANY
        return True
    if host.casefold() in _LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


def is_local_address(family: int, address: object) -> bool:
    """Whether a connect/sendto to `address` on a socket of `family` stays on this machine.
    Unix sockets always do; IP sockets only to a local host; any other family is refused
    (fail closed: nothing in this code base opens one)."""
    if family == getattr(socket, "AF_UNIX", object()):
        return True
    if family in (socket.AF_INET, socket.AF_INET6):
        return is_local_host(_host_of(address))
    return False


@dataclass
class Guard:
    """What one guarded block refused, so the plugin can fail the test afterwards."""

    refused: list[str] = field(default_factory=list)

    def refuse(self, what: str) -> EgressRefused:
        self.refused.append(what)
        return EgressRefused(
            f"egress_guard refused {what}: unit tests may only reach loopback "
            f"(@pytest.mark.{MARKER} opts one test out; ADR-0018 R2-8)"
        )

    def check(self, family: int, address: object) -> None:
        if not is_local_address(family, address):
            raise self.refuse(f"a connection to {address!r}")

    def check_lookup(self, host: object) -> None:
        """A name or reverse lookup: anything but a loopback name/address asks a resolver."""
        if not is_local_host(_host_of(host)):
            raise self.refuse(f"a name lookup of {host!r}")


def _patch_lookups(mp: pytest.MonkeyPatch, guard: Guard) -> None:
    """The resolver entry points. gethostbyname & co. are separate C functions that never go
    through getaddrinfo, so each is wrapped on its own."""
    getaddrinfo = socket.getaddrinfo
    gethostbyname = socket.gethostbyname
    gethostbyname_ex = socket.gethostbyname_ex
    gethostbyaddr = socket.gethostbyaddr
    getnameinfo = socket.getnameinfo

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None:  # None = the local wildcard / loopback, no lookup
            guard.check_lookup(host)
        return getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host: str) -> str:
        guard.check_lookup(host)
        return gethostbyname(host)

    def guarded_gethostbyname_ex(host: str) -> tuple[str, list[str], list[str]]:
        guard.check_lookup(host)
        return gethostbyname_ex(host)

    def guarded_gethostbyaddr(ip: str) -> tuple[str, list[str], list[str]]:
        guard.check_lookup(ip)
        return gethostbyaddr(ip)

    def guarded_getnameinfo(sockaddr: Any, flags: int) -> tuple[str, str]:
        guard.check_lookup(sockaddr)
        return getnameinfo(sockaddr, flags)

    mp.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    mp.setattr(socket, "gethostbyname", guarded_gethostbyname)
    mp.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    mp.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)
    mp.setattr(socket, "getnameinfo", guarded_getnameinfo)


@contextmanager
def guarded() -> Iterator[Guard]:
    """Patch the socket layer for the duration of the block. `pytest.MonkeyPatch` restores the
    originals (removing the class attributes, so the C methods show through again) even when
    the block raises."""
    guard = Guard()
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    sendto = socket.socket.sendto
    sendmsg = socket.socket.sendmsg

    def guarded_connect(self: socket.socket, address: Any) -> None:
        guard.check(self.family, address)
        connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        guard.check(self.family, address)
        return connect_ex(self, address)

    def guarded_sendto(self: socket.socket, data: Buffer, *args: Any) -> int:
        guard.check(self.family, args[-1])  # sendto(data, address) or (data, flags, address)
        return sendto(self, data, *args)

    def guarded_sendmsg(self: socket.socket, buffers: Any, *args: Any) -> int:
        # sendmsg(buffers[, ancdata[, flags[, address]]]): no address = the connected peer,
        # which connect() already checked.
        if len(args) >= 3 and args[2] is not None:
            guard.check(self.family, args[2])
        return sendmsg(self, buffers, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", guarded_connect)
        mp.setattr(socket.socket, "connect_ex", guarded_connect_ex)
        mp.setattr(socket.socket, "sendto", guarded_sendto)
        mp.setattr(socket.socket, "sendmsg", guarded_sendmsg)
        _patch_lookups(mp, guard)
        yield guard


def guard_disabled(item: pytest.Item, env: Mapping[str, str] = os.environ) -> bool:
    """A test opts out with the marker; a whole run with RAGFORGE_TEST_EGRESS_GUARD=off."""
    if item.get_closest_marker(MARKER) is not None:
        return True
    return env.get(DISABLE_ENV, "").strip().casefold() in _DISABLED_VALUES


def _langsmith() -> tuple[Any, Any] | None:
    """(langsmith, langsmith.utils), or None when it is not installed. Imported dynamically:
    the plugin must work without it, and `get_env_var` is an lru_cache behind typed overloads
    that hide `cache_clear` from mypy."""
    try:
        return importlib.import_module("langsmith"), importlib.import_module("langsmith.utils")
    except ImportError:
        return None


def reset_langsmith(*, enabled: bool | None) -> None:
    """Set langsmith's global tracing switch (None = follow the env) and forget the env values
    it cached, so the next `tracing_is_enabled()` reads the environment as it is now."""
    modules = _langsmith()
    if modules is None:  # not installed: nothing can trace
        return
    langsmith, utils = modules
    utils.get_env_var.cache_clear()
    langsmith.configure(enabled=enabled)


def pin_langsmith_off() -> None:
    """Force LangSmith tracing off and drop its credentials for this process.

    The env alone is not enough once langsmith is imported (its pytest plugin is, before any
    conftest): `get_env_var` is lru-cached, so the cache is cleared, and the global switch
    (`langsmith.configure(enabled=False)`) is set too; it outranks the environment."""
    os.environ.update(TRACING_OFF_ENV)
    for name in TRACING_CREDENTIAL_ENV:
        os.environ.pop(name, None)
    reset_langsmith(enabled=False)


def _guarded_phase(item: pytest.Item) -> Generator[None, object, object]:
    """Run one test phase under the guard, then fail it if anything was refused, even when the
    code under test swallowed the EgressRefused."""
    if guard_disabled(item):
        return (yield)
    with guarded() as guard:
        result = yield
    if guard.refused:
        pytest.fail(
            "egress_guard: this test tried to leave the machine: " + "; ".join(guard.refused),
            pytrace=False,
        )
    return result


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: this test may open non-loopback connections (egress_guard; ADR-0018 R2-8)",
    )
    pin_langsmith_off()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _guarded_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _guarded_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _guarded_phase(item))
