"""Run file v2 I/O (LOCKED, integrator-owned): one JSONL per run, header first, footer last.

Runners write through `RunWriter` (append + fsync per row, so a crash keeps every finished row
and a resume can skip them); everyone else reads through `read_run`. Files are private (0600):
they hold answer text, never note text (ADR-0017 §7).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from ragforge_client.evals.schema import SCHEMA_VERSION, RunFooter, RunHeader, RunRow, digest


class RunFileError(ValueError):
    """Not a v2 run file, a newer version, or a header that differs from the one resumed."""


@dataclass(frozen=True)
class RunFile:
    path: Path
    header: RunHeader
    rows: list[RunRow]
    footer: RunFooter | None  # None: the run did not finish (resumable)

    @property
    def header_digest(self) -> str:
        return header_digest(self.header)


def header_digest(header: RunHeader) -> str:
    """The run's configuration identity: everything but its id, start time and override."""
    data = header.model_dump(mode="json", exclude={"run_id", "started_at", "override"})
    return digest(data)


def read_run(path: Path) -> RunFile:
    # "\n" only: str.splitlines() also splits on U+2028/U+2029/U+0085, which JSON writes raw
    # inside answer text (review P1, EV-JUDGE)
    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    if not lines:
        raise RunFileError(f"{path}: empty")
    first = json.loads(lines[0])
    if first.get("schema_version") != SCHEMA_VERSION or first.get("type") != "header":
        raise RunFileError(f"{path}: not a v{SCHEMA_VERSION} run file")
    header = RunHeader.model_validate(first)
    rows: list[RunRow] = []
    footer: RunFooter | None = None
    for line in lines[1:]:
        data = json.loads(line)
        if data.get("type") == "footer":
            footer = RunFooter.model_validate(data)
        else:
            rows.append(RunRow.model_validate(data))
    return RunFile(path, header, rows, footer)


def iter_runs(directory: Path) -> Iterator[RunFile]:
    """Every readable v2 run in `directory`, newest first; v1 and broken files are skipped."""
    for path in sorted(directory.glob("*.jsonl"), reverse=True):
        try:
            yield read_run(path)
        except (RunFileError, ValueError):
            continue


class RunWriter:
    """Append-only writer. `resume=True` reopens an unfinished run whose header is identical in
    configuration (`header_digest`) and returns the item ids already done in `done`."""

    def __init__(self, path: Path, header: RunHeader, *, resume: bool = False) -> None:
        self.path = path
        self.done: set[str] = set()
        self._rows = 0
        if resume and path.exists():
            existing = read_run(path)
            if existing.footer is not None:
                raise RunFileError(f"{path}: already finished")
            if header_digest(existing.header) != header_digest(header):
                raise RunFileError(f"{path}: configuration differs from the run being resumed")
            self.done = {r.item_id for r in existing.rows if r.error is None}
            self._rows = len(existing.rows)
            self._file: IO[str] = path.open("a", encoding="utf-8")
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._file = os.fdopen(fd, "w", encoding="utf-8")
            self._write(header.model_dump_json())

    def _write(self, line: str) -> None:
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def row(self, row: RunRow) -> None:
        self._write(row.model_dump_json())
        self._rows += 1

    def finish(self) -> None:
        self._write(RunFooter(finished_at=datetime.now(UTC), rows=self._rows).model_dump_json())

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._file.close()
