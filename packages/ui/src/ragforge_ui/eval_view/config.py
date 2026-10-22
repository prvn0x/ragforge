"""Where the Eval page finds its files, and the eval key for fetching note text.

Flags after `--` (like the UI's `--key-file`), else env, else the defaults:

    --eval-runs DIR       $RAGFORGE_EVAL_RUNS      evals/private/runs
    --eval-data DIR       $RAGFORGE_EVAL_DATA      the runs dir's parent (evals/private)
    --eval-key-file FILE  $RAGFORGE_EVAL_KEY_FILE  none: shown notes cannot be fetched

The data dir holds `questions.toml` and, in the private layout, the sealed `heldout2.toml`
beside it (`datasets.load_datasets` reads both the same way).

Why a separate key: `/v1/eval/chunks` needs the dedicated eval key of the eval tenant (ADR-0017
§1), never the owner's daily key, so the UI's single-key autodiscovery is NOT reused here. The key
is read from its file only when a note is fetched, with the client's hidden-file rules
(`ragforge_client.ask.read_key`), and is never kept in session state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragforge_client.evals.datasets import HELDOUT2_FILE
from ragforge_ui.config import DEFAULT_URL

DEFAULT_RUNS = Path("evals/private/runs")
RUNS_ENV = "RAGFORGE_EVAL_RUNS"
DATA_ENV = "RAGFORGE_EVAL_DATA"
KEY_ENV = "RAGFORGE_EVAL_KEY_FILE"


@dataclass(frozen=True, slots=True)
class EvalPaths:
    """Every file the page reads. Explicit paths (not one root) so tests can point each at the
    committed golden fixtures, whose layout differs from a private eval directory."""

    runs: Path  # <run>.jsonl + <run>.summary.json + gate-<surface>.json (interfaces.py)
    questions: Path
    heldout2: Path  # the private layout's sealed held-out questions (may be absent)
    verifier: Path
    interview: Path
    labels: Path  # D5 labels.jsonl: the page's only write target
    judge_results: Path
    calibrations: Path

    @classmethod
    def under(cls, runs: Path, data: Path) -> EvalPaths:
        """The LOCKED file conventions (interfaces.py) plus the dataset file names of D0."""
        return cls(
            runs=runs,
            questions=data / "questions.toml",
            heldout2=data / HELDOUT2_FILE,
            verifier=data / "verifier.json",
            interview=data / "interview.json",
            labels=data / "labels.jsonl",
            judge_results=data / "judge-cache.jsonl",
            calibrations=data / "calibration.jsonl",
        )


@dataclass(frozen=True, slots=True)
class EvalConfig:
    paths: EvalPaths
    api_url: str
    key_file: Path | None


class EvalConfigError(ValueError):
    """A malformed eval flag (e.g. `--eval-runs` with no value). The message names the flag
    only, never a value, so the page can show it as is."""


def load_eval_config(argv: Sequence[str], env: Mapping[str, str]) -> EvalConfig:
    """The page's paths and eval key file. Raises EvalConfigError instead of exiting: argparse's
    default `SystemExit` would stop the Streamlit script with no message on the page."""
    parser = argparse.ArgumentParser(prog="ragforge-ui-eval", add_help=False, exit_on_error=False)
    parser.add_argument("--eval-runs")
    parser.add_argument("--eval-data")
    parser.add_argument("--eval-key-file")
    parser.add_argument("--url")
    try:
        # parse_known_args: the UI's own flags and Streamlit's share argv
        args, _ = parser.parse_known_args(list(argv))
    except argparse.ArgumentError as exc:
        raise EvalConfigError(f"bad value for {exc.argument_name or 'a flag'}") from None
    except SystemExit:  # the few argparse paths that still exit despite exit_on_error=False
        raise EvalConfigError("unparseable flags") from None
    runs = Path(args.eval_runs or env.get(RUNS_ENV) or DEFAULT_RUNS).expanduser()
    data = Path(args.eval_data or env.get(DATA_ENV) or runs.parent).expanduser()
    key = args.eval_key_file or env.get(KEY_ENV)
    url = args.url or env.get("RAGFORGE_URL") or env.get("RAGFORGE_API_URL") or DEFAULT_URL
    return EvalConfig(
        paths=EvalPaths.under(runs, data),
        api_url=url.rstrip("/"),
        key_file=Path(key).expanduser() if key else None,
    )
