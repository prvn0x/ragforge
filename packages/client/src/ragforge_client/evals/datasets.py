"""Datasets of the eval layer (ADR-0017 §2): loading D0/D1/D3/D4, sealing `heldout2`, the
question<->page overlap flag, the D5 label store, the D3 builder for private runs and the
stratified calibration sample.

Everything here reads and writes local files only. The one network use is the D3 builder, which
fetches the shown note text through the SDK's `/v1/eval/chunks` (any object with that method), and
checks every digest before it trusts the text.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import re
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from pydantic import ValidationError

from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    SCHEMA_VERSION,
    AnswerOutput,
    EvalKind,
    InterviewSet,
    Label,
    LabelKind,
    Mutation,
    Question,
    QuestionSet,
    RunRow,
    Span,
    Split,
    VerifierItem,
    VerifierSet,
    digest,
    page_key_str,
)
from ragforge_contracts.api import ChunkKey, EvalChunk, EvalChunksResponse

__all__ = [
    "CALIBRATION_N",
    "OVERLAP_THRESHOLD",
    "CalibrationCandidate",
    "ChunkSource",
    "D3Build",
    "DatasetError",
    "LabelStore",
    "OverlapFlag",
    "SealError",
    "append_jsonl",
    "build_d3",
    "calibration_candidates",
    "calibration_sample",
    "check_validity",
    "content_lemmas",
    "item_digest_of",
    "jaccard",
    "load_datasets",
    "load_interview_set",
    "load_questions",
    "load_verifier_set",
    "newest_labels",
    "overlap_flags",
    "planted_span",
    "question_page_overlap",
    "seal_digest",
    "seal_questions",
    "verify_seal",
]

CALIBRATION_N = 50  # ADR-0017 §4: 50 owner-labelled E-G items, all criteria each
OVERLAP_THRESHOLD = 0.5  # ADR-0017 §2 D1 step 4: flag a question above this for rewording
SEAL_PREFIX = "sha256:"


class DatasetError(ValueError):
    """A dataset or label file that does not load: wrong version, bad shape, duplicate ids."""


class SealError(DatasetError):
    """`heldout2` was edited after sealing, or is required sealed and is not."""


# ------------------------------------------------------------------ item identity
def item_digest_of(
    question: str,
    answer: str,
    notes: Sequence[str],
    facts: Sequence[str] = (),
    level: str | None = None,
) -> str:
    """The digest of an item AS SHOWN (to the judge or to the owner in the label queue).

    One function for both, because calibration joins owner labels to judge results by this
    digest: if the queue and the judge hashed different fields, no label would ever match. Note
    ORDER is part of what was shown (S1 is not S2), so it is kept, not sorted.
    """
    return digest(
        {
            "question": question,
            "answer": answer,
            "notes": list(notes),
            "facts": list(facts),
            "level": level,
        }
    )


# ------------------------------------------------------------------ atomic, locked JSONL append
@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """An exclusive advisory lock on `<path>.lock`. A separate lock file, so the lock does not
    depend on the data file's inode (which a manual edit or restore can replace)."""
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory fds
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, lines: Sequence[str]) -> None:
    """Append `lines` under a lock (ADR-0017 D5) with ONE `O_APPEND` write, then fsync.

    O(new lines), not O(file): the judge cache grows by thousands of lines a run, and rewriting
    the whole file per put would be quadratic. The lock serialises writers (the CLI and the Eval
    page never interleave lines); a reader that races a write can see at most an unfinished LAST
    line, which `complete_lines` leaves for the next read. The file is private (0600): it may hold
    answer text.

    A torn tail (a writer that crashed or hit a full disk mid-line) is CUT before the append, never
    terminated: terminating it would turn the fragment into a complete, invalid line in the middle
    of the file, which every later read rejects (one crash would brick the owner's labels). The
    fragment was never a complete line, so no reader ever saw it as data."""
    if any("\n" in line for line in lines):
        raise ValueError("a JSONL line must not contain a newline")
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(line + "\n" for line in lines).encode()
    with _file_lock(path):
        created = not path.exists()
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = _cut_torn_tail(fd)
            try:
                written = os.write(fd, data)
                if written != len(data):  # a full disk mid-write
                    raise OSError(f"{path}: short write ({written} of {len(data)} bytes)")
            except OSError:
                os.ftruncate(fd, size)  # leave no fragment of OUR lines behind either
                raise
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            _fsync_dir(path.parent)


_TAIL_BLOCK = 4096


def _cut_torn_tail(fd: int) -> int:
    """Truncate the file to just after its last newline (to 0 if it has none) and return the new
    size. Reads backwards from the end in blocks, so a well-formed file costs one 1-byte read."""
    size = os.fstat(fd).st_size
    if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
        return size
    end = size
    while end > 0:
        start = max(0, end - _TAIL_BLOCK)
        newline = os.pread(fd, end - start, start).rfind(b"\n")
        if newline >= 0:
            keep = start + newline + 1
            break
        end = start
    else:
        keep = 0
    os.ftruncate(fd, keep)
    return keep


def complete_lines(data: bytes) -> tuple[list[str], int]:
    """The complete (newline-terminated) lines of `data` and how many bytes they span. An
    unterminated tail is a write in progress (or a crashed one): not yet a line."""
    end = data.rfind(b"\n") + 1
    # split on "\n" ONLY: str.splitlines() also splits on U+2028/U+2029/U+0085, which JSON
    # writes raw, so one pasted Notion line separator would brick the file (review P1)
    return data[:end].decode("utf-8").split("\n")[:-1], end


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    lines, _ = complete_lines(path.read_bytes())
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{n}: not JSON ({exc.msg})") from None
        if not isinstance(data, dict):
            raise DatasetError(f"{path}:{n}: not a JSON object")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(f"{path}:{n}: schema_version must be {SCHEMA_VERSION}")
        out.append(data)
    return out


# ------------------------------------------------------------------ D5 label store
class LabelStore:
    """`labels.jsonl` (ADR-0017 D5): append-only, one line per labelled item, newest wins.

    Implements LabelReader and LabelWriter. "Newest" is FILE order, not `labelled_at`: the lock
    serialises appends, so file order is the real order, whereas two machines' clocks can disagree.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, label: Label) -> None:
        append_jsonl(self.path, [label.model_dump_json()])

    def all(self) -> list[Label]:
        try:
            return [Label.model_validate(d) for d in _read_jsonl(self.path)]
        except ValidationError as exc:
            raise DatasetError(f"{self.path}: {exc.error_count()} invalid label field(s)") from None

    def latest(self) -> list[Label]:
        return newest_labels(self.all())


def newest_labels(labels: Sequence[Label]) -> list[Label]:
    """The last label per (item_digest, kind) in sequence order (file order for a LabelStore),
    in the order those last labels appear. Idempotent: `newest_labels(store.latest())` is
    `store.latest()`, so a consumer can always collapse what it is given."""
    newest: dict[tuple[str, LabelKind], Label] = {}
    for label in labels:
        newest.pop((label.item_digest, label.kind), None)  # re-insert: keeps sequence order
        newest[(label.item_digest, label.kind)] = label
    return list(newest.values())


# ------------------------------------------------------------------ D1 questions + sealing
def seal_digest(questions: Sequence[Question]) -> str:
    """The seal of a file's `heldout2` rows: their parsed content, in id order. Parsed, not raw
    bytes, so a comment or whitespace edit does not break the seal but any change of meaning
    (text, facts, match, relevance, split) does."""
    rows = sorted(
        (q.model_dump(mode="json") for q in questions if q.split is Split.heldout2),
        key=lambda r: str(r["id"]),
    )
    return SEAL_PREFIX + digest(rows)


def _parse_question_file(path: Path) -> tuple[QuestionSet, str | None]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DatasetError(f"{path}: not TOML ({exc})") from None
    if "q" in raw and "questions" not in raw:
        raise DatasetError(
            f"{path}: a v1 question file ([[q]] tables). The v2 form needs schema_version = 2, "
            "[[questions]] tables and `relevance` per answerable question (ADR-0017 D1); the v1 "
            "harness (ragforge-eval run) still reads this file"
        )
    seal = raw.pop("seal", None)
    if seal is not None and not isinstance(seal, str):
        raise DatasetError(f"{path}: `seal` must be a string")
    try:
        qs = QuestionSet.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{path}: {_first_error(exc)}") from None
    return qs, seal


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err["loc"])
    return f"{exc.error_count()} error(s); first at {where}: {err['msg']}"


def verify_seal(path: Path) -> str | None:
    """The file's seal if it holds; None when the file has no `heldout2` rows and no seal.
    Raises SealError on an edited sealed file or on `heldout2` rows without a seal."""
    qs, seal = _parse_question_file(path)
    return _check_seal(path, qs, seal, require=True)


def _check_seal(path: Path, qs: QuestionSet, seal: str | None, *, require: bool) -> str | None:
    has_heldout = any(q.split is Split.heldout2 for q in qs.questions)
    if seal is None:
        if has_heldout and require:
            raise SealError(f"{path}: heldout2 is not sealed (run the seal command first)")
        return None
    if seal != seal_digest(qs.questions):
        raise SealError(f"{path}: heldout2 changed after it was sealed; edits fail the bar")
    return seal


def seal_questions(path: Path) -> str:
    """Seal a question file's `heldout2` rows: put `seal = "<digest>"` first in the file (a TOML
    top-level key must precede every table). Refuses a file that is already sealed: re-sealing an
    edited file is exactly the silent change sealing exists to stop, so unsealing stays a
    deliberate manual edit."""
    text = path.read_text(encoding="utf-8")
    qs, seal = _parse_question_file(path)
    if seal is not None:
        raise SealError(f"{path}: already sealed")
    if not any(q.split is Split.heldout2 for q in qs.questions):
        raise SealError(f"{path}: no heldout2 rows to seal")
    value = seal_digest(qs.questions)
    sealed = f'seal = "{value}"  # heldout2 digest: any edit fails the bar (ADR-0017 D1)\n' + text
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(sealed)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return value


HELDOUT2_FILE = "heldout2.toml"


def load_questions(*paths: Path, require_seal: bool = False) -> QuestionSet:
    """Load one or more v2 question files into one QuestionSet (the demo file, or the private
    `questions.toml` plus its sealed `heldout2.toml`). Every seal present is verified. A file
    named `heldout2.toml` must ALWAYS be sealed: the bar must not depend on every caller
    remembering a flag, and deleting the seal line must not unlock edits. `require_seal` extends
    that to `heldout2` rows in any other file. Ids must be unique across all files."""
    if not paths:
        raise ValueError("load_questions needs at least one path")
    name: str | None = None
    questions: list[Question] = []
    for path in paths:
        qs, seal = _parse_question_file(path)
        _check_seal(path, qs, seal, require=require_seal or path.name == HELDOUT2_FILE)
        name = name or qs.name
        questions.extend(qs.questions)
    dupes = sorted(k for k, n in Counter(q.id for q in questions).items() if n > 1)
    if dupes:
        raise DatasetError(f"duplicate question id(s): {', '.join(dupes)}")
    assert name is not None
    return QuestionSet(name=name, questions=questions)


def _load_json_model[M: (VerifierSet, InterviewSet)](path: Path, model: type[M]) -> M:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: not JSON ({exc.msg})") from None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(f"{path}: {_first_error(exc)}") from None


def load_verifier_set(path: Path) -> VerifierSet:
    """A D3 set (D0 `verifier.json` or a private one written by `build_d3`). Ids are unique and
    every planted item's source is a clean item of the same set (the bootstrap clusters on it)."""
    vs = _load_json_model(path, VerifierSet)
    ids = [i.id for i in vs.items]
    if len(ids) != len(set(ids)):
        raise DatasetError(f"{path}: duplicate item ids")
    clean = {i.id for i in vs.items if i.mutation is None}
    foreign = sorted(i.id for i in vs.items if i.mutation is None and i.source_id != i.id)
    if foreign:
        raise DatasetError(f"{path}: a clean item is its own source: {', '.join(foreign)}")
    orphans = sorted(i.id for i in vs.items if i.mutation is not None and i.source_id not in clean)
    if orphans:
        raise DatasetError(f"{path}: planted items without a clean source: {', '.join(orphans)}")
    return vs


def load_interview_set(path: Path) -> InterviewSet:
    iv = _load_json_model(path, InterviewSet)
    ids = [s.id for s in iv.seeds]
    if len(ids) != len(set(ids)):
        raise DatasetError(f"{path}: duplicate seed ids")
    return iv


def load_datasets(directory: Path, *, require_seal: bool = False) -> Datasets:
    """The bundle every metric and gate function receives. Reads what exists of
    `questions.toml` (+ `heldout2.toml`, which must be sealed), `verifier.json` and
    `interview.json`."""
    question_files = [
        p for p in (directory / "questions.toml", directory / HELDOUT2_FILE) if p.exists()
    ]
    verifier, interview = directory / "verifier.json", directory / "interview.json"
    return Datasets(
        questions=load_questions(*question_files, require_seal=require_seal)
        if question_files
        else None,
        verifier=load_verifier_set(verifier) if verifier.exists() else None,
        interview=load_interview_set(interview) if interview.exists() else None,
    )


# ------------------------------------------------------------------ lexical overlap (D1 step 4)
_WORD = re.compile(r"[a-z0-9][a-z0-9_]*(?:[.\-][a-z0-9_]+)*")
# Function words carry no content: two texts sharing "the" and "is" are not lexically close.
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just me more most my no nor not now of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours use used using get gets
    one two
    """.split()  # noqa: SIM905 - a word list reads better as text
)


def _lemma(word: str) -> str:
    """Light suffix stripping, applied identically on both sides. Jaccard needs "partitions" and
    "partition" to meet, not a linguistically perfect lemma; identifiers with dots or underscores
    are left whole (`max.poll.interval.ms` is one term).

    Three steps in a fixed order, so a plural always meets its singular: (1) drop a plural `s`
    (never `ss`), and nothing else, so `x + "s"` becomes exactly `x`; (2) drop `-ing`/`-ed`;
    (3) normalise the endings that inflection rewrites: a final `e` goes (`cache`/`caches`/
    `cached` all meet, as do `box`/`boxes`), and a final `y` becomes `i` (`policy`/`policies`).
    """
    if "." in word or "_" in word or word.isdigit():
        return word
    if len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix, min_len in (("ing", 6), ("ed", 5)):
        if word.endswith(suffix) and len(word) >= min_len:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("y"):
        word = word[:-1] + "i"
    return word


def content_lemmas(text: str) -> frozenset[str]:
    """Lower-cased content lemmas: stopwords and one-letter tokens dropped."""
    return frozenset(
        _lemma(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def question_page_overlap(question: str, page_text: str) -> float:
    """The highest Jaccard between the question and any one sentence (or line) of the page.

    Per sentence, not per page: a page has hundreds of lemmas, so a whole-page Jaccard is near 0
    for every question and could never flag anything. What the flag must catch is a question
    lifted from one sentence of the notes, which retrieval then finds by keyword alone."""
    q = content_lemmas(question)
    return max(
        (jaccard(q, content_lemmas(s)) for s in _SENTENCE_END.split(page_text) if s.strip()),
        default=0.0,
    )


@dataclass(frozen=True)
class OverlapFlag:
    question_id: str
    page_key: str
    overlap: float
    flagged: bool


def overlap_flags(
    questions: QuestionSet,
    page_texts: Mapping[str, str],
    *,
    threshold: float = OVERLAP_THRESHOLD,
    splits: frozenset[Split] = frozenset({Split.heldout2}),
) -> list[OverlapFlag]:
    """The overlap of each question of `splits` with each of its relevant pages (page_key_str ->
    page text). `flagged` = above `threshold`: reword it. A relevant page without text is skipped
    (the caller fetched what it could; nothing is guessed)."""
    out: list[OverlapFlag] = []
    for q in questions.questions:
        if q.split not in splits:
            continue
        for key in sorted(q.relevance):
            text = page_texts.get(key)
            if text is None:
                continue
            score = question_page_overlap(q.question, text)
            out.append(OverlapFlag(q.id, key, score, score > threshold))
    return out


# ------------------------------------------------------------------ D3 mutation spans and checks
_TOKEN_CHARS = re.compile(r"[A-Za-z0-9_.\-]")
_SOURCE_TOKEN = re.compile(r"[A-Za-z0-9_.]+")


def _widen(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to whole tokens, so a one-letter edit inside a name spans the whole
    name, which is what a reader sees as wrong (mirrors evals/demo/build_demo.py)."""
    while start > 0 and _TOKEN_CHARS.match(text[start - 1]):
        start -= 1
    while end < len(text) and _TOKEN_CHARS.match(text[end]):
        end += 1
    return start, end


def planted_span(clean: str, old: str, new: str) -> tuple[str, Span]:
    """Replace the single occurrence of `old` by `new`; return the mutated answer and the span of
    what changed. The same algorithm as build_demo.py, so private D3 spans mean what D0's mean."""
    if clean.count(old) != 1:
        raise ValueError("the replaced text must occur exactly once in the clean answer")
    start = clean.index(old)
    mutated = clean.replace(old, new)
    if not new:  # a deletion (guard_dropped): the line that now follows the cut
        line_end = mutated.find("\n", start)
        return mutated, Span(start=start, end=line_end if line_end > start else start + 1)
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    q = 0
    while q < min(len(old), len(new)) - p and old[-1 - q] == new[-1 - q]:
        q += 1
    lo, hi = _widen(mutated, start + p, max(start + p, start + len(new) - q))
    while lo < hi and mutated[lo] in ". ":  # an inserted sentence starts at its first word
        lo += 1
    if hi <= lo:
        hi = lo + 1
    return mutated, Span(start=lo, end=hi)


def _source_tokens(texts: Sequence[str]) -> set[str]:
    tokens = {t for s in texts for t in _SOURCE_TOKEN.findall(s)}
    return tokens | {t.strip(".") for t in tokens}


# A sentence is attributable to a chunk when at least half its content lemmas are in it (a proxy
# for the server's attribution, ADR-0015); below that, "the chunk lacks the sentence's content".
LACKS_CONTENT = 0.5


def _containment(sentence: str, text: str) -> float:
    words = content_lemmas(sentence)
    return len(words & content_lemmas(text)) / len(words) if words else 0.0


def check_validity(
    check: str, new_text: str, shown_texts: Sequence[str], *, target_text: str | None = None
) -> bool:
    """ADR-0017 §2 D3's mechanical validity checks: True when the planted text really is wrong
    against the notes. `owner_review` (negation) always passes here and is confirmed by the owner.

    `target_text` is the chunk the check is about: the cited chunk for entity_swap (defaults to
    every shown chunk, which is stricter), the NEW page for citation_moved.
    """
    if check in ("token_absent_from_shown", "number_absent_from_shown"):
        # whole tokens: "min.insync.replica" is NOT in a page that says "min.insync.replicas"
        source = _source_tokens(shown_texts)
        return any(t.strip(".") not in source for t in _SOURCE_TOKEN.findall(new_text))
    if check == "entity_absent_from_cited":
        cited = [target_text] if target_text is not None else list(shown_texts)
        return not any(new_text in s for s in cited)
    if check == "no_label_no_attribution":
        no_label = re.search(r"\[S\d+\]", new_text) is None
        return no_label and all(_containment(new_text, s) < LACKS_CONTENT for s in shown_texts)
    if check == "new_page_lacks_content":
        return target_text is not None and _containment(new_text, target_text) < LACKS_CONTENT
    if check == "line_in_source_block":
        line = new_text.strip()
        return bool(line) and any(line in s for s in shown_texts)
    if check == "owner_review":
        return True
    raise ValueError(f"unknown validity check {check!r}")


# ------------------------------------------------------------------ D3 mutation proposals
@dataclass(frozen=True)
class _Proposal:
    mutation: Mutation
    old: str
    new: str
    check: str
    checked_text: str  # what check_validity looks at
    target_text: str | None = None


@dataclass(frozen=True)
class _Source:
    """One owner-correct answer with its shown notes, ready to mutate."""

    item_id: str
    question: str
    answer: str
    keys: tuple[ChunkKey, ...]
    texts: tuple[str, ...]
    digest: str  # varies the pick among equally good donors, so sources do not all share one


def _rotate[T](options: Sequence[T], src: _Source) -> list[T]:
    """`options` (already sorted) starting at a point fixed by the source digest: deterministic,
    but different sources draw different donor sentences and entities."""
    if not options:
        return []
    k = int(src.digest[:8], 16) % len(options)
    return [*options[k:], *options[:k]]


_IDENT = re.compile(r"(?<![\w.\-])[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)+(?![\w\-]|\.\w)")
_NUMBER = re.compile(r"(?<![\w.\[])\d+(?:\.\d+)?(?![\w.]|\.\d)")
_ENTITY = re.compile(r"(?<=[a-z,;:] )[A-Z][A-Za-z0-9]{2,}\b|\b[A-Z]{2,6}\b")
_LABEL_END = re.compile(r"\[S(\d+)\]\.")
_GUARD = re.compile(r"^\s*(?:if|unless|guard|assert|raise|when|elif)\b|\bexit\s+1\b|\braise\b")
_NEGATIONS: tuple[tuple[str, str], ...] = (
    (" never ", " always "),
    (" always ", " never "),
    (" cannot ", " can "),
    (" does not ", " does "),
    (" is not ", " is "),
    (" only ", " never "),
    (" must ", " must not "),
    (" can ", " cannot "),
    (" is ", " is not "),
)


def _once(text: str, part: str) -> bool:
    return text.count(part) == 1


def _propose_typo(src: _Source) -> _Proposal | None:
    source = _source_tokens(src.texts)
    for ident in sorted(set(_IDENT.findall(src.answer)), key=lambda t: (-len(t), t)):
        if ident not in source or not _once(src.answer, ident):
            continue
        mid = len(ident) // 2
        edits = [ident[:i] + ident[i + 1 :] for i in (mid, mid + 1, len(ident) - 1)]
        for edit in edits:
            if edit != ident and edit.strip(".") not in source:
                return _Proposal(
                    Mutation.identifier_typo, ident, edit, "token_absent_from_shown", edit
                )
    return None


def _propose_number(src: _Source) -> _Proposal | None:
    source = _source_tokens(src.texts)
    for match in _NUMBER.finditer(src.answer):
        num = match.group()
        if num not in source or not _once(src.answer, num):
            continue
        value = float(num)
        for cand in (value * 2, value + 7, value * 10):
            new = str(int(cand)) if "." not in num else f"{cand:g}"
            if new != num and new not in source:
                return _Proposal(Mutation.number_drift, num, new, "number_absent_from_shown", new)
    return None


def _propose_entity(src: _Source, other_pages: Sequence[str]) -> _Proposal | None:
    shown = "\n".join(src.texts)
    others = sorted(
        {e for t in other_pages for e in _ENTITY.findall(t)} - set(_ENTITY.findall(shown))
    )
    for entity in sorted(set(_ENTITY.findall(src.answer))):
        if entity not in shown or not _once(src.answer, entity):
            continue
        for swap in _rotate(others, src):
            if swap != entity and swap not in shown:
                return _Proposal(
                    Mutation.entity_swap, entity, swap, "entity_absent_from_cited", swap
                )
    return None


def _propose_negation(src: _Source) -> _Proposal | None:
    for old, new in _NEGATIONS:
        if _once(src.answer, old):
            return _Proposal(Mutation.negation_flip, old, new, "owner_review", new)
    return None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _propose_fabricated(src: _Source, other_pages: Sequence[str]) -> _Proposal | None:
    ends = list(_LABEL_END.finditer(src.answer))
    if not ends:
        return None
    last = ends[-1]
    old = src.answer[src.answer.rfind(" ", 0, max(0, last.start() - 1)) + 1 : last.end()]
    if not old or not _once(src.answer, old):
        return None
    candidates = [
        s
        for page in other_pages
        for s in _sentences(page)
        if 20 <= len(s) <= 200 and s.endswith(".") and "[" not in s and "\n" not in s
    ]
    for sentence in _rotate(sorted(set(candidates)), src):
        if check_validity("no_label_no_attribution", sentence, src.texts):
            return _Proposal(
                Mutation.fabricated_sentence,
                old,
                f"{old} {sentence}",
                "no_label_no_attribution",
                sentence,
            )
    return None


def _propose_moved(src: _Source) -> _Proposal | None:
    if len(src.texts) < 2:
        return None
    for match in _LABEL_END.finditer(src.answer):
        label = int(match.group(1))
        if not 1 <= label <= len(src.texts):
            continue
        start = src.answer.rfind(". ", 0, match.start())
        sentence = src.answer[start + 2 if start >= 0 else 0 : match.end()]
        if not _once(src.answer, sentence):
            continue
        for other in range(1, len(src.texts) + 1):
            if other == label:
                continue
            target = src.texts[other - 1]
            moved = sentence.replace(f"[S{label}].", f"[S{other}].")
            if check_validity("new_page_lacks_content", sentence, src.texts, target_text=target):
                return _Proposal(
                    Mutation.citation_moved,
                    sentence,
                    moved,
                    "new_page_lacks_content",
                    sentence,
                    target,
                )
    return None


def _propose_guard(src: _Source) -> _Proposal | None:
    in_block = False
    for line in src.answer.split("\n"):
        if line.lstrip().startswith("```"):
            in_block = not in_block
            continue
        guard = in_block and _GUARD.search(line) and _once(src.answer, line + "\n")
        if guard and check_validity("line_in_source_block", line, src.texts):
            return _Proposal(Mutation.guard_dropped, line + "\n", "", "line_in_source_block", line)
    return None


def _proposals(src: _Source, other_pages: Sequence[str]) -> list[_Proposal]:
    found = [
        _propose_typo(src),
        _propose_number(src),
        _propose_entity(src, other_pages),
        _propose_negation(src),
        _propose_fabricated(src, other_pages),
        _propose_moved(src),
        _propose_guard(src),
    ]
    return [p for p in found if p is not None]


# ------------------------------------------------------------------ D3 builder (private runs)
class ChunkSource(Protocol):
    """What the builder needs from the SDK (`RagforgeClient` satisfies it)."""

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse: ...


@dataclass(frozen=True)
class D3Build:
    verifier: VerifierSet
    skipped: Mapping[str, int] = field(default_factory=dict)  # reason -> rows not used


_CHUNK_BATCH = 100


async def _fetch_chunks(source: ChunkSource, ids: Sequence[UUID]) -> dict[UUID, EvalChunk]:
    out: dict[UUID, EvalChunk] = {}
    unique = sorted(set(ids), key=str)
    for i in range(0, len(unique), _CHUNK_BATCH):
        response = await source.eval_chunks(unique[i : i + _CHUNK_BATCH])
        out.update({c.chunk_id: c for c in response.chunks})
    return out


def _skip_reason(row: RunRow, questions: Mapping[str, Question]) -> str | None:
    q = questions.get(row.item_id)
    if q is None:
        return "unknown_question"
    if q.split is not Split.dev:
        return "heldout2"  # D3 is a dev artifact: a held-out answer must never feed it
    if row.error is not None or row.stale:
        return "error_or_stale"
    if not isinstance(row.output, AnswerOutput) or row.output.abstained:
        return "not_answered"
    if not row.shown or any(s.chunk_key is None for s in row.shown):
        return "no_shown_keys"
    return None


def _shown_texts(row: RunRow, chunks: Mapping[UUID, EvalChunk]) -> tuple[str, ...] | None:
    """The shown texts in S-label order, or None when any is missing or its digest moved: the
    text must be EXACTLY what the model saw, or a validity check proves nothing."""
    texts: list[str] = []
    for ref in sorted(row.shown, key=lambda s: s.label):
        chunk = chunks.get(UUID(ref.chunk_id))
        if chunk is None or chunk.display_sha256 != ref.display_sha256:
            return None
        if hashlib.sha256(chunk.display_text.encode()).hexdigest() != ref.display_sha256:
            return None
        texts.append(chunk.display_text)
    return tuple(texts)


def _correct_digests(labels: Sequence[Label]) -> set[str]:
    """Owner-correct answers: the newest `answer` label says correct and not invented. Never the
    server verdict (ADR-0017 D3: selecting on the verdict would bias the clean set). Collapses
    to the newest label itself: given `store.all()`, an old "correct" the owner later overruled
    must not admit the answer as a clean source."""
    return {
        lb.item_digest
        for lb in newest_labels(labels)
        if lb.kind is LabelKind.answer and lb.correct is True and lb.invented is not True
    }


async def build_d3(
    runs: Sequence[RunFile],
    questions: QuestionSet,
    labels: Sequence[Label],
    chunks: ChunkSource,
    *,
    name: str = "private",
) -> D3Build:
    """Build a private D3 set from v2 answer runs (ADR-0017 §2 D3).

    Clean items are rows the owner labelled correct (the NEWEST label per digest and kind of
    `labels`, in sequence order: `LabelReader.latest()` or `.all()` give the same set), whatever
    the server's verdict; the label must be on the digest of the item as shown (question, notes,
    answer, facts; `item_digest_of`), so a label of a different answer never applies. Each clean
    source gets every mutation that applies to it, each with its span and a mechanical validity
    check; a proposal that fails its check is dropped, never kept. `negation_flip` items are
    written `owner_reviewed = False` (the owner reviews every one).

    ONE clean source per question. The bootstrap of ADR-0017 §3 E-V clusters on `source_id` as
    the source QUESTION; a private D3 is built from several runs (arms, A/A, E-G and E-A), so one
    question often has several owner-correct answers, and their mutations are correlated. Keeping
    them all would inflate the clean n and narrow the verifier SLO's intervals. The locked schema
    makes a clean item its own source, so the extra answers are dropped (counted under
    `duplicate_question`), keeping the smallest digest: deterministic, whatever the run order.
    """
    by_id = {q.id: q for q in questions.questions}
    skipped: Counter[str] = Counter()
    candidates: list[RunRow] = []
    for run in runs:
        for row in run.rows:
            reason = _skip_reason(row, by_id)
            if reason:
                skipped[reason] += 1
            else:
                candidates.append(row)
    fetched = await _fetch_chunks(chunks, [UUID(s.chunk_id) for r in candidates for s in r.shown])
    correct = _correct_digests(labels)
    per_question: dict[str, dict[str, _Source]] = defaultdict(dict)
    for row in candidates:
        texts = _shown_texts(row, fetched)
        if texts is None:
            skipped["stale_chunk"] += 1
            continue
        q = by_id[row.item_id]
        assert isinstance(row.output, AnswerOutput)
        d = item_digest_of(q.question, row.output.answer, texts, [f.fact for f in q.facts])
        if d not in correct:
            skipped["not_owner_correct"] += 1
            continue
        if d in per_question[row.item_id]:
            skipped["duplicate_question"] += 1  # the same item again (e.g. an A/A run)
            continue
        keys = tuple(s.chunk_key for s in sorted(row.shown, key=lambda s: s.label) if s.chunk_key)
        source = _Source(row.item_id, q.question, row.output.answer, keys, texts, d)
        per_question[row.item_id][d] = source
    sources: dict[str, _Source] = {}
    for answers in per_question.values():
        keep = min(answers)
        sources[keep] = answers[keep]
        skipped["duplicate_question"] += len(answers) - 1
    return D3Build(VerifierSet(name=name, items=_items(sources)), dict(+skipped))


def _items(sources: Mapping[str, _Source]) -> list[VerifierItem]:
    page_texts: dict[str, list[str]] = defaultdict(list)
    for src in sources.values():
        for key, text in zip(src.keys, src.texts, strict=True):
            page_texts[page_key_str(key.page)].append(text)
    items: list[VerifierItem] = []
    for d, src in sorted(sources.items(), key=lambda kv: (kv[1].item_id, kv[0])):
        source_id = f"c-{src.item_id}-{d[:8]}"
        items.append(
            VerifierItem(
                id=source_id,
                source_id=source_id,
                question=src.question,
                answer=src.answer,
                shown=list(src.keys),
            )
        )
        own = {page_key_str(k.page) for k in src.keys}
        others = [t for page, texts in sorted(page_texts.items()) if page not in own for t in texts]
        for prop in _proposals(src, others):
            mutated, span = planted_span(src.answer, prop.old, prop.new)
            if not check_validity(
                prop.check, prop.checked_text, src.texts, target_text=prop.target_text
            ):
                continue
            items.append(
                VerifierItem(
                    id=f"p-{prop.mutation.value}-{d[:8]}",
                    source_id=source_id,
                    question=src.question,
                    answer=mutated,
                    shown=list(src.keys),
                    mutation=prop.mutation,
                    span=span,
                    validity_check=prop.check,
                    owner_reviewed=prop.mutation is not Mutation.negation_flip,
                )
            )
    return items


# ------------------------------------------------------------------ calibration sample
@dataclass(frozen=True)
class CalibrationCandidate:
    """One answered E-G item the owner could label. `tier` is the deterministic grade (correct /
    incorrect / ungraded), `verdict` the server status: the two strata of ADR-0017 §4."""

    item_id: str
    item_digest: str
    split: Split
    tier: str
    verdict: str

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.tier, self.verdict)


def calibration_candidates(
    runs: Sequence[RunFile],
    questions: QuestionSet,
    digest_for: Callable[[RunRow], str | None],
    *,
    kinds: frozenset[EvalKind] = frozenset({EvalKind.generator}),
) -> list[CalibrationCandidate]:
    """Answered, error-free, current rows of E-G runs as candidates (ADR-0017 §4: 50 E-G items
    from current notes; the judge only scores answered rows). A run of another kind (`kinds`)
    is skipped whole: E-A answers come from another surface and would mix two populations into
    one calibration. A `stale` row was answered from notes that have since changed, so a label
    on it would calibrate the judge on text it will never see again.
    `digest_for(row)` is the item digest as shown (it needs the fetched note text, which the
    caller has); None skips the row. Duplicates (same digest) are kept once."""
    by_id = {q.id: q for q in questions.questions}
    seen: dict[str, CalibrationCandidate] = {}
    for run in runs:
        if run.header.eval_kind not in kinds:
            continue
        for row in run.rows:
            q = by_id.get(row.item_id)
            out = row.output
            if q is None or row.error or row.stale:
                continue
            if not isinstance(out, AnswerOutput) or out.abstained:
                continue
            d = digest_for(row)
            if d is None or d in seen:
                continue
            graded = row.grades.get("correct")
            tier = "ungraded" if graded is None else ("correct" if graded else "incorrect")
            seen[d] = CalibrationCandidate(row.item_id, d, q.split, tier, out.status)
    return list(seen.values())


def _allocate(sizes: Mapping[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """Largest-remainder (Hamilton) allocation of `n` over strata in proportion to their size,
    with every stratum getting at least one item when n allows (a small stratum must still be
    calibrated on), and none more than it holds (what a full stratum cannot take goes to the
    strata furthest below their quota)."""
    total = sum(sizes.values())
    n = min(n, total)
    if n == 0:
        return {s: 0 for s in sizes}
    quota = {s: n * size / total for s, size in sizes.items()}
    least = 1 if n >= len(sizes) else 0
    alloc = {s: min(sizes[s], max(int(quota[s]), least)) for s in sizes}
    while sum(alloc.values()) > n:  # the floors of one overshot: take from the most over-served
        over = max((s for s in alloc if alloc[s] > 1), key=lambda s: (alloc[s] - quota[s], s))
        alloc[over] -= 1
    while sum(alloc.values()) < n:
        room = [s for s in alloc if alloc[s] < sizes[s]]
        under = max(room, key=lambda s: (quota[s] - alloc[s], s))
        alloc[under] += 1
    return alloc


def calibration_sample(
    candidates: Sequence[CalibrationCandidate],
    *,
    n: int = CALIBRATION_N,
    seed: int = 0,
    exclude_digests: frozenset[str] = frozenset(),
) -> list[CalibrationCandidate]:
    """The stratified calibration sample (ADR-0017 §4): never a `heldout2` row, never an item
    already labelled (`exclude_digests`), strata (tier, verdict) in proportion with each one
    represented, reproducible from `seed`. Each sampled item is labelled on ALL its criteria at
    once (one `criteria` label), so n items give n per criterion."""
    pool: dict[tuple[str, str], list[CalibrationCandidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c.item_digest):
        if c.split is not Split.heldout2 and c.item_digest not in exclude_digests:
            pool[c.stratum].append(c)
    alloc = _allocate({s: len(v) for s, v in pool.items()}, n)
    rng = random.Random(seed)  # noqa: S311 - a reproducible sample, not a secret
    picked: list[CalibrationCandidate] = []
    for stratum in sorted(pool):
        picked.extend(rng.sample(pool[stratum], alloc[stratum]))
    return sorted(picked, key=lambda c: c.item_digest)
