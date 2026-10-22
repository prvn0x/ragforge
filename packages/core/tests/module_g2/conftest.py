"""Module G2 fixtures: no provider variable leaks in from the developer's shell.

setup_provider gives a shell-exported RAGFORGE_LLM_PROVIDER / RAGFORGE_GEMINI_* priority over .env
(compose's precedence), so a value exported where the suite runs would change what the tests
observe. Tests that exercise that precedence set the variables explicitly.
"""

from __future__ import annotations

import pytest

_MANAGED = (
    "RAGFORGE_LLM_PROVIDER",
    "RAGFORGE_GEMINI_API_KEY",
    "RAGFORGE_GEMINI_MODEL",
    "RAGFORGE_GEMINI_THINKING_BUDGET",
)


@pytest.fixture(autouse=True)
def _no_provider_vars_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MANAGED:
        monkeypatch.delenv(name, raising=False)
