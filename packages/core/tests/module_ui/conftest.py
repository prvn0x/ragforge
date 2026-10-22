"""UI test defaults: the page runs as `make ui` runs it, bound to loopback.

The app refuses to load a key when Streamlit listens beyond this machine (`exposure_refused`),
and Streamlit's own default is every interface. Tests that prove the refusal set the option
themselves; everything else gets the loopback address `make ui` passes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from streamlit import config

from ragforge_ui.config import ALLOW_REMOTE_ENV

_OPTIONS = ("server.address", "client.showErrorDetails")


@pytest.fixture(autouse=True)
def loopback_ui(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(ALLOW_REMOTE_ENV, raising=False)
    saved = {key: config.get_option(key) for key in _OPTIONS}
    config.set_option("server.address", "127.0.0.1")
    yield
    for key, value in saved.items():  # the app sets showErrorDetails; keep the process clean
        config.set_option(key, value)
