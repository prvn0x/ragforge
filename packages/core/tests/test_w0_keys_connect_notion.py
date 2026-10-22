"""W0-KEYS: `make connect-notion` finds the RAGForge key like `make ask` and prompts only as a
last resort; the Notion token is always hidden input.

The script runs in-process with fakes for the API call, `input` and `getpass`. Temp HOME only.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from ragforge_client.ask import KeyFileError

REPO: Final = Path(__file__).resolve().parents[3]
# Built at runtime so this file never holds a token-shaped literal (see the gitleaks rule test).
KEYS: Final = {name: "CANARY-rf-" + name for name in ("flag", "env", "dir", "typed")}
NOTION_TOKEN: Final = "CANARY" + "-notion-" + "5d2a"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "w0_connect_notion", REPO / "scripts/connect_notion.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


@dataclass
class Fakes:
    """Stands in for the terminal and the API; records what the script asked and sent."""

    prompts: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, str, dict[str, object] | None]] = field(default_factory=list)
    collections: list[dict[str, object]] = field(
        default_factory=lambda: [{"id": "c-1", "name": "acme-demo"}]
    )
    inputs: list[str] = field(default_factory=list)

    def getpass(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        return NOTION_TOKEN if "Notion" in prompt else KEYS["typed"]

    def call(
        self, method: str, path: str, key: str, body: dict[str, object] | None = None
    ) -> object:
        self.calls.append((method, path, key, body))
        if (method, path) == ("GET", "/v1/collections"):
            return {"items": self.collections}
        if (method, path) == ("POST", "/v1/collections"):
            assert body is not None
            return {"id": "c-new", "name": body["name"]}
        return {"id": "k-1", "status": "active"}

    def key_used(self) -> str:
        return self.calls[0][2]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAGFORGE_KEY_FILE", raising=False)
    return tmp_path


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    fake = Fakes()
    monkeypatch.setattr(script.getpass, "getpass", fake.getpass)
    monkeypatch.setattr(script, "call", fake.call)

    def fake_input(prompt: str = "") -> str:
        fake.inputs.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    return fake


def write_key(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(value + "\n")
    path.chmod(0o600)
    return path


def key_dir(home: Path) -> Path:
    return home / ".config/ragforge"


def test_reads_the_single_key_file_and_prompts_only_for_the_notion_token(
    home: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    write_key(key_dir(home) / "owner.key", KEYS["dir"])

    script.main([])

    assert fakes.key_used() == KEYS["dir"]
    assert len(fakes.prompts) == 1
    assert "Notion" in fakes.prompts[0]
    post = fakes.calls[-1]
    assert post[:2] == ("POST", "/v1/connectors")
    assert post[3] is not None
    assert post[3]["token"] == NOTION_TOKEN  # the Notion token still comes from hidden input
    out, err = capsys.readouterr()
    assert KEYS["dir"] not in out + err
    assert NOTION_TOKEN not in out + err


def test_discovery_order_flag_then_env_then_dir(
    home: Path, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_key(key_dir(home) / "owner.key", KEYS["dir"])
    env_file = write_key(home / "elsewhere/env.key", KEYS["env"])
    flag_file = write_key(home / "elsewhere/flag.key", KEYS["flag"])
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(env_file))

    assert (
        script.load_key(str(flag_file), {"RAGFORGE_KEY_FILE": str(env_file)}, key_dir(home))
        == KEYS["flag"]
    )
    assert script.load_key(None, {"RAGFORGE_KEY_FILE": str(env_file)}, key_dir(home)) == KEYS["env"]
    assert script.load_key(None, {}, key_dir(home)) == KEYS["dir"]
    script.main(["--key-file", str(flag_file)])
    assert fakes.key_used() == KEYS["flag"]
    assert not any("RAGForge" in p for p in fakes.prompts)


def test_falls_back_to_the_hidden_prompt_only_when_no_key_file_exists(
    home: Path, fakes: Fakes
) -> None:
    (key_dir(home) / "mcp").mkdir(parents=True)  # subfolder keys are not candidates
    write_key(key_dir(home) / "mcp/mcp.key", "mcp")

    script.main([])

    assert fakes.key_used() == KEYS["typed"]
    assert [p for p in fakes.prompts if "RAGForge" in p] == [fakes.prompts[0]]


def test_no_config_folder_at_all_also_prompts(home: Path, fakes: Fakes) -> None:
    assert script.find_key_file(None, {}, key_dir(home)) is None
    assert script.load_key(None, {}, key_dir(home)) == KEYS["typed"]


@pytest.mark.parametrize(
    "case", ["missing flag file", "missing env file", "several keys", "loose mode"]
)
def test_a_named_or_ambiguous_key_is_an_error_not_a_prompt(
    home: Path, fakes: Fakes, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    explicit: str | None = None
    env: dict[str, str] = {}
    if case == "missing flag file":
        explicit = str(home / "nope.key")
    elif case == "missing env file":
        env = {"RAGFORGE_KEY_FILE": str(home / "nope.key")}
    elif case == "several keys":
        write_key(key_dir(home) / "a.key", "a")
        write_key(key_dir(home) / "b.key", "b")
    else:
        write_key(key_dir(home) / "owner.key", KEYS["dir"]).chmod(0o644)

    with pytest.raises(SystemExit) as exit_info:
        script.load_key(explicit, env, key_dir(home))

    assert str(exit_info.value.code).startswith("RAGForge key: ")
    assert fakes.prompts == []  # a typo or an ambiguity must not silently become a prompt


def test_find_key_file_returns_the_resolved_path(home: Path) -> None:
    only = write_key(key_dir(home) / "owner.key", KEYS["dir"])
    assert script.find_key_file(None, {}, key_dir(home)) == only
    write_key(key_dir(home) / "other.key", "x")
    with pytest.raises(KeyFileError):
        script.find_key_file(None, {}, key_dir(home))


def test_a_new_tenant_with_no_collection_gets_one_and_continues(
    home: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    """`make tenant` creates no collection; the quickstart must not stop here."""
    write_key(key_dir(home) / "owner.key", KEYS["dir"])
    fakes.collections = []

    script.main([])

    assert [c[:2] for c in fakes.calls] == [
        ("GET", "/v1/collections"),
        ("POST", "/v1/collections"),
        ("POST", "/v1/connectors"),
    ]
    assert fakes.calls[1][2] == KEYS["dir"]  # created with the key already loaded
    assert fakes.calls[1][3] == {"name": "notes"}  # the default name
    connector = fakes.calls[2][3]
    assert connector is not None
    assert connector["target_collection_id"] == "c-new"
    assert not any("Target collection" in p for p in fakes.inputs)  # nothing to choose from
    assert "Created collection notes (c-new)" in capsys.readouterr().out
