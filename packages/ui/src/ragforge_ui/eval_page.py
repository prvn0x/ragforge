"""The Eval page (ADR-0017 §6): runs, compare, drill-down, the blind label queue, calibration,
verifier and interview views, over LOCAL run and label files (`ragforge_client.evals`).

It needs no server API beyond `/v1/eval/chunks`, used only to show the notes a model was shown,
on demand, with the eval key (`eval_view.config`). Its only write is a D5 label, through the
label store's `LabelWriter` (atomic and locked; the CLI may write at the same time).

Tests inject `EvalPaths`, a `ChunkSource` and a label store through session state, exactly as
`app.BACKEND` injects the API: the real page never sets them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import streamlit as st

from ragforge_client.ask import KeyFileError
from ragforge_client.evals.interfaces import LabelReader, LabelWriter
from ragforge_ui.eval_view.config import (
    KEY_ENV,
    EvalConfig,
    EvalConfigError,
    EvalPaths,
    load_eval_config,
)
from ragforge_ui.eval_view.context import EvalContext, load_context
from ragforge_ui.eval_view.data import LabelAccess, label_store
from ragforge_ui.eval_view.drill import drill_view
from ragforge_ui.eval_view.labels import label_view
from ragforge_ui.eval_view.notes import ChunkSource, SdkChunkSource
from ragforge_ui.eval_view.quality import calibration_view, interview_view, verifier_view
from ragforge_ui.eval_view.runs import compare_view, runs_view
from ragforge_ui.render import md_literal

PATHS = "rf_eval_paths"
CHUNKS = "rf_eval_chunks"
LABELS = "rf_eval_labels"  # an object that is both a LabelReader and a LabelWriter
VIEW = "rf_eval_view"

VIEWS: dict[str, Callable[[EvalContext], None]] = {
    "Runs": runs_view,
    "Compare": compare_view,
    "Drill-down": drill_view,
    "Label queue": label_view,
    "Calibration": calibration_view,
    "Verifier": verifier_view,
    "Interview": interview_view,
}


def eval_page() -> None:
    st.header("Evals")
    try:
        config = load_eval_config(sys.argv[1:], os.environ)
    except EvalConfigError as exc:  # its message names the flag, never a value
        st.error(
            f"**Bad command-line flags** ({md_literal(str(exc))}). Use: -- --eval-runs DIR "
            "--eval-data DIR --eval-key-file FILE"
        )
        return
    injected: EvalPaths | None = st.session_state.get(PATHS)
    paths = injected or config.paths
    ctx = load_context(paths, _labels(paths), lambda: _chunk_source(config), config.api_url)
    for problem in ctx.problems:
        st.warning(md_literal(problem))
    if not ctx.entries:
        st.info(
            f"No eval runs in {md_literal(str(paths.runs))} yet. Run an eval (ragforge-eval) or "
            "point the page elsewhere with --eval-runs DIR."
        )
        return
    view = st.radio("View", list(VIEWS), horizontal=True, key=VIEW)
    VIEWS[view or "Runs"](ctx)


def _labels(paths: EvalPaths) -> LabelAccess:
    store = st.session_state.get(LABELS)
    if store is None:
        return label_store(paths.labels)
    reader: LabelReader = store
    writer: LabelWriter = store
    return LabelAccess(reader, writer)


def _chunk_source(config: EvalConfig) -> ChunkSource:
    injected: ChunkSource | None = st.session_state.get(CHUNKS)
    if injected is not None:
        return injected
    if config.key_file is None:
        raise KeyFileError(
            "no eval key: pass --eval-key-file FILE or set "
            f"{KEY_ENV} (the eval tenant's key, mode 600)"
        )
    return SdkChunkSource(config.api_url, config.key_file)
