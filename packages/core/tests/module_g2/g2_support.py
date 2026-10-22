"""Module G2 test support: loads scripts/setup_provider.py (a stdlib script, not a package) and
provides a scripted console and a recording fake for urllib's opener, so no test touches a
terminal, the network, or the real .env."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "setup_provider.py"

# Shaped like an AI Studio key, distinctive enough that any leak is found by substring search.
FAKE_KEY = "AIzaSyG2-TESTONLY_leakcanary_0123456789"
OTHER_KEY = "AIzaSyG2-ROTATED_leakcanary_9876543210"


def load_setup_provider() -> ModuleType:
    """Import the script by path; registered in sys.modules so dataclasses resolve their module."""
    name = "ragforge_g2_setup_provider"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ScriptedConsole:
    """Answers prompts from fixed lists; `say` prints, so capsys sees everything shown."""

    answers: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    tty: bool = True
    events: list[str] = field(default_factory=list)

    def isatty(self) -> bool:
        return self.tty

    def ask(self, prompt: str) -> str:
        self.events.append(f"ask:{prompt}")
        if not self.answers:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return self.answers.pop(0)

    def secret(self, prompt: str) -> str:
        self.events.append(f"secret:{prompt}")
        if not self.secrets:
            raise AssertionError(f"unexpected secret prompt: {prompt}")
        return self.secrets.pop(0)

    def say(self, message: str) -> None:
        self.events.append(f"say:{message}")
        print(message)


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, *_: object) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@dataclass
class FakeOpener:
    """Stands in for urllib.request.urlopen: replays queued outcomes and records each request."""

    outcomes: list[object] = field(default_factory=list)
    requests: list[Any] = field(default_factory=list)

    def __call__(self, request: Any, timeout: float | None = None) -> _Response:
        assert timeout is not None, "every request must carry a timeout"
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("unexpected HTTP request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(json.dumps(outcome).encode())


def models_page(*names: str, methods: tuple[str, ...] = ("generateContent",)) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": f"models/{n}",
                "displayName": n.title(),
                "supportedGenerationMethods": list(methods),
            }
            for n in names
        ]
    }


def http_error(
    code: int, reason: str = "API_KEY_INVALID", echo: str = ""
) -> urllib.error.HTTPError:
    """An HTTPError with a Google-style body. `echo` lets a test plant text (like the key) in the
    body to prove the script never repeats free-form server text."""
    body = {
        "error": {
            "code": code,
            "message": f"API key not valid. {echo}",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": reason}],
        }
    }
    return urllib.error.HTTPError(
        "https://example.invalid", code, "error", Message(), io.BytesIO(json.dumps(body).encode())
    )
