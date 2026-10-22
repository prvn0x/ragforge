"""Server wiring and secrets: the key never leaks, config rules, and the server really starts
(in-memory with the classic handshake, and as a real `ragforge-mcp` stdio subprocess)."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import httpx
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_support import (
    API,
    KEY,
    cite,
    connected,
    connector,
    everything,
    make_service,
    mock_api,
    page,
    problem,
    sse,
    structured,
    text_of,
)

from ragforge_contracts.api import Problem
from ragforge_contracts.sse import (
    ChatEvent,
    CitationEvent,
    ErrorEvent,
    ExtraEvent,
    TokenEvent,
    VerificationEvent,
)
from ragforge_mcp import server as server_module
from ragforge_mcp.config import ConfigError, is_loopback, load_config
from ragforge_mcp.errors import Redactor, display_url, redact_model
from ragforge_mcp.models import SyncResult, SyncSkipped
from ragforge_mcp.server import build_server, main, transport_security

TOOLS = {"ask", "search", "list_sources", "sync_now"}


def _key_file(path: Path, value: str = KEY, mode: int = 0o600) -> Path:
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


# ----------------------------------------------------------------- the key never leaks
async def test_key_never_reaches_tool_output_or_logs_even_when_the_api_echoes_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hostile or buggy API that reflects the bearer token into every error must still not
    get it to the agent or into this process's logs."""
    caplog.set_level(logging.DEBUG)
    echo = f"bad token {KEY}"
    cid = uuid4()
    with mock_api() as api:
        chat = api.post("/v1/chat").mock(
            return_value=sse([
                TokenEvent(text="x"),
                ErrorEvent(problem=Problem(title=echo, status=500, detail=echo)),
            ])
        )  # fmt: skip
        api.post("/v1/query").mock(return_value=problem(403, echo, echo))
        api.get("/v1/connectors").mock(
            return_value=httpx.Response(200, json=page([connector(cid, "N")]))
        )
        api.post(f"/v1/connectors/{cid}/sync").mock(return_value=problem(409, echo, echo))
        api.get("/v1/documents").mock(return_value=problem(422, echo, echo))
        async with connected() as client:
            results = [
                await client.call_tool("ask", {"question": "q"}),
                await client.call_tool("search", {"query": "q"}),
                await client.call_tool("sync_now", {}),
                await client.call_tool("list_sources", {}),
            ]
            with pytest.raises(Exception) as resource_error:  # noqa: PT011 - any error type
                await client.read_resource(f"ragforge://documents/{uuid4()}")
    assert [r.is_error for r in results] == [True, True, True, False]
    surfaced = [everything(r) for r in results] + [str(resource_error.value), caplog.text]
    for text in surfaced:
        assert KEY not in text
    assert "[redacted]" in text_of(results[0])  # the echo was there, and was removed
    # ...while the key WAS used, as a bearer token, on the wire only
    assert chat.calls.last.request.headers["authorization"] == f"Bearer {KEY}"


async def test_key_never_reaches_output_when_echoed_after_the_verdict() -> None:
    """Review r1 P1: an error after `verification` becomes a WARNING on a successful result, a
    path the first leak test did not cover. Also echoed into answer, citation and extra text."""
    echo = f"bad token {KEY}"
    events: list[ChatEvent] = [
        TokenEvent(text=f"answer {KEY}"),
        CitationEvent(citation=cite(1, f"title {KEY}")),
        VerificationEvent(status="verified"),
        ExtraEvent(kind="incident", text=f"extra {KEY}"),
        ErrorEvent(problem=Problem(title=echo, status=500, detail=echo)),
    ]
    with mock_api() as api:
        api.post("/v1/chat").mock(return_value=sse(events))
        async with connected() as client:
            result = await client.call_tool("ask", {"question": "q"})
    out = structured(result)
    assert "[redacted]" in out["warning"]
    assert KEY not in everything(result)


def test_redactor_also_removes_the_json_escaped_spelling() -> None:
    odd = 'k"ey\\with/quote'
    assert Redactor(odd)(json.dumps({"x": f"a {odd} b"})) == json.dumps({"x": "a [redacted] b"})


def test_redactor_and_config_reprs_hide_the_key(tmp_path: Path) -> None:
    assert KEY not in repr(Redactor(KEY))
    assert Redactor(KEY)(f"a {KEY} b") == "a [redacted] b"
    assert Redactor("")("unchanged") == "unchanged"
    config = load_config(["--key-file", str(_key_file(tmp_path / "k.key"))], {}, tmp_path)
    assert config.api_key.get_secret_value() == KEY
    assert KEY not in repr(config)
    assert KEY not in str(config)


def test_display_url_drops_credentials() -> None:
    assert display_url("http://user:pw@rf.test:8000/x") == "http://rf.test:8000/x"
    assert display_url("http://host:notaport") == "<invalid url>"


# ----------------------------------------------------------------- config
def test_key_sources_in_priority_order(tmp_path: Path) -> None:
    flag = _key_file(tmp_path / "flag.key", "from-flag-" + "x" * 20)
    env_file = _key_file(tmp_path / "env.key", "from-env-file-" + "x" * 20)
    keydir = tmp_path / "dir"
    keydir.mkdir()
    _key_file(keydir / "only.key", "from-dir-" + "x" * 20)

    def key(argv: list[str], env: dict[str, str]) -> str:
        return load_config(argv, env, keydir).api_key.get_secret_value()

    both = {"RAGFORGE_API_KEY": "from-env-" + "x" * 20, "RAGFORGE_KEY_FILE": str(env_file)}
    assert key(["--key-file", str(flag)], both).startswith("from-flag")
    assert key([], both).startswith("from-env-x")
    assert key([], {"RAGFORGE_KEY_FILE": str(env_file)}).startswith("from-env-file")
    assert key([], {}).startswith("from-dir")


@pytest.mark.parametrize("short", ["k", "dev", "x" * 19])
def test_config_refuses_a_key_too_short_to_redact_safely(tmp_path: Path, short: str) -> None:
    """Review r2: redacting "dev" or "k" would rewrite ordinary words in every tool output."""
    with pytest.raises(ConfigError, match="shorter than 20") as err:
        load_config([], {"RAGFORGE_API_KEY": short}, tmp_path)
    assert f"'{short}'" not in str(err.value)
    with pytest.raises(ConfigError, match="shorter than 20"):
        load_config(["--key-file", str(_key_file(tmp_path / "s.key", short))], {}, tmp_path)


def test_redaction_never_breaks_the_output_shape() -> None:
    """Review r2: replacing inside serialized JSON could hit field names. Values only now."""
    result = SyncResult(
        started=[],
        skipped=[SyncSkipped(connector_id=uuid4(), display_name="started", reason=f"x {KEY}")],
    )
    out = redact_model(result, Redactor("started"))  # a secret equal to a field name
    assert out.skipped[0].display_name == "[redacted]"
    assert out.started == []
    assert redact_model(result, Redactor(KEY)).skipped[0].reason == "x [redacted]"


def test_config_refuses_shared_key_files_without_echoing_the_key(tmp_path: Path) -> None:
    shared = _key_file(tmp_path / "shared.key", mode=0o644)
    with pytest.raises(ConfigError, match="chmod 600") as err:
        load_config(["--key-file", str(shared)], {}, tmp_path)
    assert KEY not in str(err.value)
    with pytest.raises(ConfigError, match="no API key"):
        load_config([], {}, tmp_path / "missing")


def test_config_urls_collections_and_transport(tmp_path: Path) -> None:
    env = {"RAGFORGE_API_KEY": KEY, "RAGFORGE_URL": "http://env:1", "RAGFORGE_API_URL": "http://x"}
    config = load_config(["--collection", "a", "--collection", "a", "--collection", "b"], env)
    assert config.api_url == "http://env:1"
    assert config.ui_url == "http://localhost:8501"
    assert config.collections == ("a", "b")
    assert (config.http, config.host) == (False, "127.0.0.1")
    assert load_config([], {"RAGFORGE_API_KEY": KEY}).api_url == "http://localhost:8000"
    assert load_config(["--http", "--port", "9100"], env).port == 9100
    too_many = [arg for i in range(21) for arg in ("--collection", f"c{i}")]
    with pytest.raises(ConfigError, match="at most 20"):
        load_config(too_many, env)


def test_http_refuses_a_non_loopback_host_unless_allowed() -> None:
    env = {"RAGFORGE_API_KEY": KEY}
    with pytest.raises(ConfigError, match="unencrypted"):
        load_config(["--http", "--host", "192.168.1.10"], env)
    assert (
        load_config(["--http", "--host", "192.168.1.10", "--allow-remote"], env).host
        == "192.168.1.10"
    )
    for host in ("127.0.0.1", "localhost", "::1", "127.0.0.2"):
        assert load_config(["--http", "--host", host], env).host == host
    assert not is_loopback("my-laptop.local")


def test_transport_security_is_explicit_and_only_adds_the_bound_host() -> None:
    local = transport_security("127.0.0.1", 8765)
    assert local.enable_dns_rebinding_protection
    assert local.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    remote = transport_security("10.0.0.5", 8765)
    assert remote.enable_dns_rebinding_protection
    assert remote.allowed_hosts[-1] == "10.0.0.5:8765"
    assert "*" not in remote.allowed_hosts


def test_the_default_directory_key_is_flagged_as_probably_human(tmp_path: Path) -> None:
    keydir = tmp_path / "dir"
    keydir.mkdir()
    _key_file(keydir / "owner.key")
    assert "owner's key" in (load_config([], {}, keydir).key_warning or "")
    flag = _key_file(tmp_path / "mcp.key")
    assert load_config(["--key-file", str(flag)], {}, keydir).key_warning is None
    assert load_config([], {"RAGFORGE_KEY_FILE": str(flag)}, keydir).key_warning is None
    assert load_config([], {"RAGFORGE_API_KEY": KEY}, keydir).key_warning is None


def test_main_prints_the_key_warning_on_stderr_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(["--key-file", str(_key_file(tmp_path / "k.key"))], {}, tmp_path)
    warned = dataclasses.replace(config, key_warning="probably the owner's key")
    monkeypatch.setattr(server_module, "load_config", lambda argv, env: warned)
    monkeypatch.setattr(anyio, "run", lambda fn, cfg: None)
    main([])
    out, err = capsys.readouterr()
    assert out == ""
    assert "warning: probably the owner's key" in err
    assert KEY not in err


def test_main_exits_2_on_bad_config_with_stdout_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RAGFORGE_API_KEY", raising=False)
    monkeypatch.delenv("RAGFORGE_KEY_FILE", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        main(["--key-file", str(_key_file(tmp_path / "k.key", mode=0o640))])
    assert exit_info.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""  # stdout is the MCP protocol channel on stdio
    assert "readable by others" in err
    assert KEY not in err


# ----------------------------------------------------------------- the server starts
async def test_in_memory_classic_handshake_lists_tools_and_instructions() -> None:
    service, rf = make_service()
    async with rf, Client(build_server(service), mode="legacy") as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert client.server_info is not None
        assert client.server_info.name == "ragforge"
        assert client.instructions is not None
        assert "Not from your notes" in client.instructions
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.description
        assert tool.output_schema is not None  # typed outputs
    assert tools["search"].input_schema["properties"]["k"]["maximum"] == 20
    assert tools["search"].annotations is not None
    assert tools["search"].annotations.read_only_hint is True
    assert tools["ask"].input_schema["required"] == ["question"]


async def test_stdio_entry_point_starts_and_reports_an_unreachable_api(tmp_path: Path) -> None:
    """The real `python -m ragforge_mcp.server` over stdio: it starts without the API (tools are
    listed), and a tool call fails cleanly when the API is down. Port 9 on localhost: refused."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ragforge_mcp.server"],
        env={"RAGFORGE_API_KEY": KEY, "RAGFORGE_URL": "http://127.0.0.1:9"},
    )
    errlog = tmp_path / "stderr.log"
    with errlog.open("w") as err:
        transport: Any = stdio_client(params, errlog=err)
        async with Client(transport) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("search", {"query": "x"})
    assert tools == TOOLS
    assert result.is_error
    assert "unreachable at http://127.0.0.1:9" in text_of(result)
    assert KEY not in errlog.read_text()
    assert API not in text_of(result)


def test_an_unreadable_key_file_is_a_clean_config_error(tmp_path: Path) -> None:
    """Re-review P2: a directory or a non-UTF-8 file raised a raw traceback."""
    directory = tmp_path / "a-dir.key"
    directory.mkdir(mode=0o700)
    binary = tmp_path / "bin.key"
    binary.write_bytes(b"\xff\xfe" + b"x" * 40)
    binary.chmod(0o600)
    for path in (directory, binary):
        with pytest.raises(ConfigError, match="cannot read the API key file"):
            load_config(["--key-file", str(path)], {}, tmp_path)
