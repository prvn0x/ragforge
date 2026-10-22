"""RAGForge UI entry point (`make ui`), or by hand:

    uv run streamlit run packages/ui/src/ragforge_ui/app.py --server.address 127.0.0.1 \
        --client.showErrorDetails type [-- --key-file F]

A thin client of the public API (ADR-0005): everything goes through `ragforge_client`.
Without `--server.address 127.0.0.1` the page refuses to load the key (see
`config.exposure_refused`); set RAGFORGE_UI_ALLOW_REMOTE=1 only when something else guards it.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

from ragforge_client.ask import KeyFileError
from ragforge_ui.api import Backend, SdkBackend
from ragforge_ui.config import (
    ALLOW_REMOTE_ENV,
    exposure_refused,
    host_refused,
    load_config,
    origin_refused,
)
from ragforge_ui.eval_page import eval_page
from ragforge_ui.render import md_literal
from ragforge_ui.views import chat_page, is_busy, sources_page

# Tests put a fake `Backend` here (streamlit AppTest); the real app never sets it, so the key is
# read from its file on each run and never stored in session state.
BACKEND = "rf_backend"


def _header(name: str) -> str | None:
    """A header of this session's browser request; None outside one (tests, bare runs)."""
    try:
        value = st.context.headers.get(name)
    except (AttributeError, RuntimeError):
        return None
    return str(value) if value else None


def _host_header() -> str | None:
    """The Host the browser used for this session."""
    return _header("Host")


def _guards_refused() -> bool:
    """The exposure, Host and Origin checks, shown as errors. True = stop rendering.
    Checked before anything else, injected test backends included: no code path may hand a key
    (the daily key OR the eval key) to a page the whole network or another site can reach."""
    address = st.get_option("server.address")
    if exposure_refused(address, os.environ):
        where = md_literal(str(address)) if address else "every network interface"
        st.error(
            f"**This page is reachable from other machines** (listening on {where}). It acts "
            "with your API key, which can approve destructive actions, so it won't load the key. "
            "Restart with `--server.address 127.0.0.1` (make ui does), or set "
            f"`{ALLOW_REMOTE_ENV}=1` if a localhost-only port mapping or an authenticating proxy "
            "already guards it."
        )
        return True
    if host_refused(_host_header(), os.environ) or origin_refused(
        _header("Origin"), _host_header(), os.environ
    ):
        st.error(
            "**This page was opened by another web page or host name** than this UI on "
            "localhost or 127.0.0.1, which is how a malicious page would reach it (DNS "
            "rebinding, or another local site). It acts with your API key, so it won't load "
            "it. Open http://localhost:8501 directly."
        )
        return True
    return False


def _backend() -> Backend | None:
    if _guards_refused():
        return None
    injected = st.session_state.get(BACKEND)
    if injected is not None:
        backend: Backend = injected
        return backend
    try:
        config = load_config(sys.argv[1:], os.environ)
    except KeyFileError as exc:  # its message names the file, never the key
        st.error(f"**No usable API key.** {md_literal(str(exc))}")
        return None
    except (OSError, ValueError) as exc:
        # An unreadable key file: a directory, no permission, not UTF-8. Only the type is
        # shown: a decode error's message quotes bytes of the file, i.e. of the key.
        st.error(
            f"**Cannot read the API key file** ({type(exc).__name__}). Point --key-file or "
            "RAGFORGE_KEY_FILE at a readable UTF-8 text file holding only the key."
        )
        return None
    except SystemExit:  # argparse rejected a flag after `--` (it printed usage to the console)
        st.error("**Bad command-line flags.** Use: -- --key-file FILE --url URL")
        return None
    return SdkBackend(config.api_url, config.api_key)


def main() -> None:
    st.set_page_config(page_title="RAGForge", page_icon=":material/menu_book:")
    # Defence in depth: every known failure is caught and shown as a friendly message, but an
    # uncaught one would show its message and traceback in the browser by default ("full").
    # The exception type is enough on screen; the full trace still goes to the console.
    st.set_option("client.showErrorDetails", "type")
    st.sidebar.title("RAGForge")
    if _guards_refused():
        return
    # Switching page mid-answer would stop the stream before `done` (see views.BUSY).
    page = st.sidebar.radio("Page", ["Chat", "Sources", "Eval"], key="rf_page", disabled=is_busy())
    if page == "Eval":
        # The Eval page reads local run files and uses its own eval key (never the daily key),
        # so it opens even when the daily key is missing (EV-UI handoff); the guards ran above.
        eval_page()
        return
    backend = _backend()
    if backend is None:
        return
    st.sidebar.caption(f"API {backend.api_url}")
    if page == "Sources":
        sources_page(backend)
    else:
        chat_page(backend)


main()
