"""Streamlit usage telemetry is a silent egress path (final review P1): pinned off on every
launch path from the repo, the make target AND a repo-root config file."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_make_ui_passes_the_flag() -> None:
    make = (ROOT / "Makefile").read_text()
    recipe = make[make.index("\nui:") : make.index("\n\n", make.index("\nui:"))]
    assert "--browser.gatherUsageStats false" in recipe


def test_the_repo_root_streamlit_config_turns_usage_stats_off() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["browser"]["gatherUsageStats"] is False
