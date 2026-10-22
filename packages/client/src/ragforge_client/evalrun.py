"""`make eval-private`: answer-quality eval over the owner's private question set (ADR-0012).

    ragforge-eval run --label <arm> [--split dev|heldout|all] [--notes "..."] [--question-timeout S]
    ragforge-eval score [-v] [--reveal] <run.jsonl> [...]     -> one table row per run
    ragforge-eval compare <run.jsonl> [...]                   -> arms table + success-bar flags
    ragforge-eval label [--reveal] [--relabel] <run.jsonl>    -> human invented-content labels

`run` asks every question in a fresh conversation, one at a time (a local model serves one turn at
a time anyway, so parallel requests would only distort latency), records the raw outcome, and
samples host memory (swap, Ollama RSS) in the background so an arm's cost is measured, not guessed.
`score`/`compare` are separate so expected answers can be corrected after a run without re-asking.

Grading (also documented at the top of questions.toml):
- answerable: correct = answered AND cites an expected page AND >= ceil(2/3 of facts) matched.
  A citation counts whether the model wrote it or the server inferred it (`Citation.inferred`);
  the report splits the two so a "correct" bought by server inference is never mistaken for a
  model that cites on its own. "content" drops the citation requirement.
- unanswerable: correct = abstained.
- invented%: only from human labels (`label`), never from the verifier: the system grading itself
  is not ground truth (CLAUDE.md: never self-score). Without labels the report says "unlabelled".
  One rule needs no human: an answer to a must-abstain question is invented by construction.
- Each row fingerprints the question it asked (split + text/answerable digest), and the run header
  records every question id it set out to ask. Fact and page corrections re-grade silently by
  design; any other edit since the run (text, split, answerable flag, a question removed or added)
  fails the bar.
- Held-out questions exist to catch overfitting, so per-question output for them is hidden unless
  `--reveal` is passed: tuning a prompt while staring at held-out rows silently turns them into
  dev rows. Judging invented% needs a human to read each answer, so the held-out bar verdict needs
  ONE `label --reveal` pass, done after the arm is frozen; the sidecar stamps the first reveal.
- The bar fails on any errored turn and on a question timeout under the p95 bar; a timed-out turn
  counts in latency at its deadline, never as a fast answer.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import AsyncIterator, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO, get_args
from uuid import UUID

import httpx

from ragforge_client.ask import (
    DEFAULT_KEY_DIR,
    DEFAULT_URL,
    KeyFileError,
    clean,
    read_key,
    resolve_key_file,
    unescape_outside_fences,
)
from ragforge_client.client import ApiError, RagforgeClient
from ragforge_contracts.api import Citation
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    ExtraEvent,
    TokenEvent,
    ToolCallEvent,
    VerificationEvent,
)

EVAL_DIR = Path("evals/private")
SPLITS = ("dev", "heldout")
HELDOUT = "heldout"
# Must match prompts/templates/abstain.md: the server's abstain wording.
ABSTAIN_PREFIX = "I couldn't find this in your notes."
# Must match prompts/templates/answer.md rule 2: the model's own refusal. Both are abstains; a
# test in packages/core/tests checks the templates still say exactly these.
MODEL_REFUSAL = "Your notes don't cover this."
ABSTAIN_PHRASES = (ABSTAIN_PREFIX, MODEL_REFUSAL)
# agent/nodes.py `_closest_pages`: the only text the server's abstain may carry after it.
_CLOSEST_HEADER = "The closest pages I found:"
# The model's refusal as the server's allowlist reads it (coverage.refusal_only): the phrase,
# optionally "question"/"topic", with a label and at most a pleasantry after it. Anything else
# after it is an answer (review rounds 2 and 3).
_MODEL_REFUSAL_RE = re.compile(
    r"your notes don't cover (?:this|that)(?: question| topic)?\s*[.!]?"
    r"(?:\s+(?:i hope this helps|sorry|thanks)[.!]?)?",
    re.IGNORECASE,
)
_APOSTROPHES = str.maketrans(dict.fromkeys("\u2018\u2019\u02bc\u0060\u00b4\uff07", "'"))
# Inflections a fact keyword may carry ("weight" matches "weighted"), nothing more.
_SUFFIX = r"(?:s|es|ed|d|ly|ing)?"

# Success bar agreed with the owner before any experiment (docs/NEXT.md, Phase 2b step 1).
BAR_CORRECT_PCT = 80.0
BAR_P95_S = 10.0
BAR_INVENTED_PCT = 2.0

# Generous next to the 10 s bar on purpose: the timeout exists to stop a stalled stream from
# hanging the whole eval, not to grade latency (p95 does that).
DEFAULT_QUESTION_TIMEOUT_S = 300.0
# Arm labels become part of a filename and a markdown cell: no separators, no table syntax.
_LABEL_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


# ------------------------------------------------------------------ question set
@dataclass(frozen=True)
class Fact:
    fact: str
    match: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    question: str
    answerable: bool = True
    pages: tuple[str, ...] = ()
    facts: tuple[Fact, ...] = ()


def load_questions(path: Path) -> list[Question]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for entry in raw.get("q", []):
        q = Question(
            id=entry["id"],
            split=entry["split"],
            question=entry["question"],
            answerable=entry.get("answerable", True),
            pages=tuple(entry.get("pages", ())),
            facts=tuple(
                Fact(fact=f["fact"], match=tuple(m.lower() for m in f["match"]))
                for f in entry.get("facts", ())
            ),
        )
        if q.split not in SPLITS:
            raise ValueError(f"{q.id}: split must be one of {SPLITS}, got {q.split!r}")
        if q.answerable and not (q.pages and q.facts and all(f.match for f in q.facts)):
            raise ValueError(f"{q.id}: an answerable question needs pages and matchable facts")
        questions.append(q)
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


# ------------------------------------------------------------------ running
@dataclass
class Outcome:
    """One question's raw result. New fields default so run files from older versions still load.

    `cited_titles` is every cited page (model or inferred, arrival order). `inferred_titles` is the
    subset cited ONLY by server inference; None means provenance is unknown: the run predates the
    split, or the server never sent `Citation.inferred` (an older build would make every citation
    look model-cited, so a missing field must read as "unknown", never as False).

    `question_split`/`question_sha256` record which question was asked, so `score`/`compare` can
    tell a later edit of its text, split or answerable flag from an intended fact/page correction.
    """

    id: str
    draft: str = ""  # everything streamed before any answer.replace
    final: str = ""  # what the user ends up seeing
    replaced: bool = False
    status: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    cited_titles: list[str] = field(default_factory=list)
    inferred_titles: list[str] | None = None
    tools: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    prompt_version: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    first_token_s: float | None = None
    total_s: float = 0.0
    error: str | None = None
    timed_out: bool = False  # structured, so latency can count it at its deadline (see summarize)
    # ADR-0016 extras ("kind: text"), recorded so their content can be sampled and labelled;
    # never graded (correct% and verified% read the answer text only)
    extras: list[str] = field(default_factory=list)
    question_split: str | None = None  # None: recorded before questions were fingerprinted
    question_sha256: str | None = None

    def model_titles(self) -> list[str] | None:
        """Pages the model cited itself (None when the run did not record provenance)."""
        if self.inferred_titles is None:
            return None
        inferred = set(self.inferred_titles)
        return [t for t in self.cited_titles if t not in inferred]


class ChatSource(Protocol):
    """The one SDK method `ask_one` needs, so tests can drive it without a server."""

    def chat(self, message: str, collection_ids: Sequence[UUID]) -> AsyncIterator[ChatEvent]: ...


def question_digest(q: Question) -> str:
    """Fingerprint of what changes the MEANING of a question: its text and whether it should be
    answered at all. Facts and pages are deliberately left out: correcting those after a run and
    re-scoring is the intended workflow (see the module docstring)."""
    body = json.dumps({"question": q.question, "answerable": q.answerable}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def question_changed(q: Question, o: Outcome) -> bool:
    """True when the run asked a different question than the current file holds. Rows from runs
    that predate fingerprinting cannot be checked and are not flagged (nothing to compare)."""
    if o.question_sha256 is not None and o.question_sha256 != question_digest(q):
        return True
    return o.question_split is not None and o.question_split != q.split


def recorded_question_ids(header: Mapping[str, Any]) -> frozenset[str]:
    """Ids the run set out to ask (header `questions`: id -> question_digest). Empty for runs that
    predate the field, which then cannot be checked for removals (nothing to compare)."""
    asked = header.get("questions")
    return frozenset(asked) if isinstance(asked, dict) else frozenset()


def removed_questions(
    questions: Iterable[Question], outcomes: Iterable[Outcome], recorded: Collection[str] = ()
) -> list[str]:
    """Ids the run asked (per its header, or a fingerprinted row) that the current file lacks.

    Why this is a bar failure: grading drops rows whose id is gone, and completeness is measured
    against the current split, so deleting a question that failed would silently raise correct%
    and still read as a complete run. Legacy rows without a fingerprint are not counted (they
    cannot tell "removed" from "never part of this question set").
    """
    current = {q.id for q in questions}
    asked = set(recorded) | {o.id for o in outcomes if o.question_sha256 is not None}
    return sorted(asked - current)


def _record_citation(out: Outcome, citation: Citation) -> None:
    """A page cited by the model at least once is model-cited, even if also inferred elsewhere:
    the model demonstrably attributed it, so crediting inference would understate the model.

    `model_fields_set` tells "server said inferred=false" from "server never sent the field": the
    current server serialises every field, so absence means a build that predates provenance.
    """
    title, is_new = citation.title, citation.title not in out.cited_titles
    if is_new:
        out.cited_titles.append(title)
    if "inferred" not in citation.model_fields_set:
        out.inferred_titles = None  # unknown for the whole turn: never assume model-cited
    if out.inferred_titles is None:
        return
    if is_new and citation.inferred:
        out.inferred_titles.append(title)
    elif not citation.inferred and title in out.inferred_titles:
        out.inferred_titles.remove(title)


def _record_done(out: Outcome, event: DoneEvent) -> None:
    out.models = sorted({u.model for u in event.usage})
    out.providers = sorted({u.provider for u in event.usage})
    out.prompt_tokens = sum(u.prompt_tokens for u in event.usage)
    out.completion_tokens = sum(u.completion_tokens for u in event.usage)
    out.prompt_version = event.prompt_version


async def _consume(rf: ChatSource, collections: list[UUID], q: Question, out: Outcome) -> None:
    started = time.monotonic()
    async for event in rf.chat(q.question, collections):
        match event:
            case TokenEvent(text=text):
                if out.first_token_s is None:
                    out.first_token_s = time.monotonic() - started
                out.draft += text
            case AnswerReplaceEvent(text=text):
                out.replaced, out.final = True, text
            case CitationEvent(citation=c):
                _record_citation(out, c)
            case ToolCallEvent(tool=tool):
                out.tools.append(tool)
            case VerificationEvent(status=status, unsupported_claims=claims):
                out.status, out.unsupported_claims = str(status), list(claims)
            case ExtraEvent(kind=kind, text=text):
                # stored as the reader sees it (Markdown escapes undone) for labelling
                out.extras.append(clean(f"{kind}: {unescape_outside_fences(text)}"))
            case DoneEvent():
                _record_done(out, event)
            case ErrorEvent(problem=problem):
                out.error = clean(
                    " ".join(filter(None, (str(problem.status), problem.title, problem.detail)))
                )  # server text


async def ask_one(
    rf: ChatSource,
    collections: list[UUID],
    q: Question,
    timeout_s: float | None = DEFAULT_QUESTION_TIMEOUT_S,
) -> Outcome:
    """Ask one question; any per-question failure becomes `Outcome.error`, never an exception.

    Why: the arms this tool measures (e.g. qwen3:8b on a 16 GB Mac) are exactly the ones that get
    OOM-killed or stall mid-stream. A dropped connection or a stuck stream must cost one row, not
    the rest of the run. Cancellation (Ctrl-C) is NOT caught: it must still stop the run.
    """
    out = Outcome(
        id=q.id,
        inferred_titles=[],
        question_split=q.split,
        question_sha256=question_digest(q),
    )
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s if timeout_s and timeout_s > 0 else None):
            await _consume(rf, collections, q, out)
    except ApiError as exc:
        out.error = clean(str(exc))  # problem detail is server text
    except TimeoutError:
        out.error, out.timed_out = f"timeout after {timeout_s:g}s", True
    except httpx.HTTPError as exc:
        # The type names the failure (ConnectError, RemoteProtocolError...); httpx messages carry
        # the URL at most, never request headers, so the key cannot leak through here.
        out.error = f"transport {type(exc).__name__}: {clean(str(exc))}"
    out.total_s = time.monotonic() - started
    if not out.replaced:
        out.final = out.draft
    return out


# ------------------------------------------------------------------ resource sampling
CommandRunner = Callable[[Sequence[str]], str | None]

_SWAP_USED = re.compile(r"used\s*=\s*([\d.]+)\s*([KMGT])", re.IGNORECASE)
_UNIT_MB = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}


def run_command(argv: Sequence[str]) -> str | None:
    """Run a fixed, shell-free command; None on any failure so sampling can never break a run."""
    try:
        done = subprocess.run(  # noqa: S603 - argv is a constant list chosen by this module
            list(argv), capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def parse_swapusage(text: str) -> float | None:
    """`sysctl -n vm.swapusage` -> used MB ("total = 2048.00M  used = 1234.50M  free = ...")."""
    m = _SWAP_USED.search(text)
    return float(m.group(1)) * _UNIT_MB[m.group(2).upper()] if m else None


def parse_meminfo_swap(text: str) -> float | None:
    """Linux /proc/meminfo -> used swap MB (SwapTotal - SwapFree, both in kB)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if name in ("SwapTotal", "SwapFree") and parts:
            with contextlib.suppress(ValueError):
                values[name] = float(parts[0])
    if len(values) != 2:
        return None
    return max(0.0, values["SwapTotal"] - values["SwapFree"]) / 1024


def parse_ollama_rss(ps_output: str) -> float | None:
    """`ps -A -o rss=,comm=` -> summed RSS MB of every ollama process (server + model runners).

    Summed because the server and its runner(s) are separate processes and the model weights live
    in the runner. None when no ollama process exists (not 0: "not running" is not "free").
    """
    total_kb, found = 0.0, False
    for line in ps_output.splitlines():
        rss, _, comm = line.strip().partition(" ")
        if not Path(comm.strip()).name.lower().startswith("ollama"):
            continue
        with contextlib.suppress(ValueError):
            total_kb += float(rss)
            found = True
    return total_kb / 1024 if found else None


@dataclass(frozen=True)
class ResourceSample:
    swap_used_mb: float | None
    ollama_rss_mb: float | None


@dataclass
class ResourcePeaks:
    """Peak values across a run. None = never observed (unsupported OS or tool missing).

    `swap_baseline_mb` is the first swap reading (taken before the first question). This machine
    already sits on many GB of swap before any run, so an absolute peak is the same for every arm;
    peak minus baseline is what the arm itself pushed out, and is what `compare` reports.

    Only the FIRST sampling attempt may set the baseline. If it fails (or reads no swap), the
    baseline stays None and swap Δ shows "-": a mid-run reading as baseline would understate Δ.
    """

    samples: int = 0
    failures: int = 0
    swap_used_mb: float | None = None
    ollama_rss_mb: float | None = None
    swap_baseline_mb: float | None = None

    def add(self, sample: ResourceSample) -> None:
        if self.samples == 0 and self.failures == 0:
            self.swap_baseline_mb = sample.swap_used_mb
        self.samples += 1
        self.swap_used_mb = _max_opt(self.swap_used_mb, sample.swap_used_mb)
        self.ollama_rss_mb = _max_opt(self.ollama_rss_mb, sample.ollama_rss_mb)

    @property
    def swap_delta_mb(self) -> float | None:
        return swap_delta(self.swap_used_mb, self.swap_baseline_mb)


def swap_delta(peak: Any, baseline: Any) -> float | None:
    """Peak minus baseline; None unless both were observed (older footers have no baseline)."""
    if not isinstance(peak, int | float) or not isinstance(baseline, int | float):
        return None
    return max(0.0, float(peak) - float(baseline))


def _max_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    return a if b is None else max(a, b)


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


@dataclass
class ResourceSampler:
    """Reads host memory through injectable seams (command runner, file reader, platform)."""

    platform: str = sys.platform
    runner: CommandRunner = run_command
    read_file: Callable[[Path], str | None] = read_text_or_none

    def sample(self) -> ResourceSample:
        return ResourceSample(swap_used_mb=self._swap(), ollama_rss_mb=self._ollama())

    def _swap(self) -> float | None:
        if self.platform == "darwin":
            out = self.runner(("sysctl", "-n", "vm.swapusage"))
            return parse_swapusage(out) if out else None
        if self.platform.startswith("linux"):
            text = self.read_file(Path("/proc/meminfo"))
            return parse_meminfo_swap(text) if text else None
        return None  # other OSes: report "not observed", never a fake 0

    def _ollama(self) -> float | None:
        if self.platform != "darwin" and not self.platform.startswith("linux"):
            return None
        out = self.runner(("ps", "-A", "-o", "rss=,comm="))
        return parse_ollama_rss(out) if out else None


async def sample_until(
    sampler: ResourceSampler, peaks: ResourcePeaks, interval_s: float, stop: asyncio.Event
) -> None:
    """Sample in a worker thread so a slow `ps` never stalls the SSE stream being timed.

    A failing sample is dropped, not fatal: losing one reading is better than losing the rest of
    the run's peak (or, worse, the run itself).
    """
    while True:
        try:
            peaks.add(await asyncio.to_thread(sampler.sample))
        except Exception:  # any sampler bug must not end sampling
            peaks.failures += 1
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        if stop.is_set():
            return


# ------------------------------------------------------------------ run files
@dataclass
class RunFile:
    path: Path
    header: dict[str, Any]
    outcomes: list[Outcome]
    footer: dict[str, Any] | None  # None: older format, or the run was killed before finishing


_OUTCOME_FIELDS = frozenset(f.name for f in fields(Outcome))


def load_run(path: Path, *, warn: TextIO | None = None) -> RunFile:
    """Parse a run file. A malformed LAST line is skipped with a warning: a SIGKILL (or OOM) can
    truncate the line being written, and that must not make every finished row unscoreable. The
    run then has no footer, so `compare` treats it as incomplete. Damage anywhere else still
    raises, because it means the file was edited or corrupted, not interrupted."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = json.loads(lines[0])["header"]
    outcomes: list[Outcome] = []
    footer: dict[str, Any] | None = None
    for number, line in enumerate(lines[1:], start=2):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines):
                raise
            print(
                f"warning: {path.name}: skipped truncated last line {number}",
                file=sys.stderr if warn is None else warn,
            )
            break
        if "footer" in row:
            footer = row["footer"]
            continue
        # Ignore keys from newer writers so an older checkout can still score a newer run.
        outcomes.append(Outcome(**{k: v for k, v in row.items() if k in _OUTCOME_FIELDS}))
    return RunFile(path, header, outcomes, footer)


def read_run(path: Path) -> tuple[dict[str, Any], list[Outcome]]:
    run_file = load_run(path)
    return run_file.header, run_file.outcomes


def observed_arm(outcomes: Iterable[Outcome]) -> dict[str, Any]:
    """What the client can see about the arm: the server reports provider/model per turn."""
    rows = list(outcomes)
    return {
        "providers": sorted({p for o in rows for p in o.providers}),
        "models": sorted({m for o in rows for m in o.models}),
        "prompt_versions": sorted({o.prompt_version for o in rows if o.prompt_version}),
        "tool_turns": sum(bool(o.tools) for o in rows),
    }


class EvalSetupError(Exception):
    """The run cannot start meaningfully (as opposed to one question failing, which is recorded)."""


def progress_line(i: int, total: int, q: Question, outcome: Outcome) -> str:
    """One visible (non-hidden) question's progress line. Held-out rows never get one: see
    `asking_order` and `run_eval` for why they are reported only as a block."""
    flag = "ERR" if outcome.error else clean(outcome.status or "-")
    return f"[{i:>2}/{total}] {q.id:<34} {flag:<11} {outcome.total_s:5.1f}s"


def asking_order(questions: Sequence[Question], reveal: bool) -> list[Question]:
    """Without --reveal, held-out questions are asked LAST, as one block (stable otherwise).

    Why: a line printed the moment a held-out question finishes leaks its latency through timing,
    and so does the next visible line (its start time is the hidden question's end time). Asking
    the hidden block last, with one line before and one after, exposes only the block's total.
    """
    if reveal:
        return list(questions)
    return sorted(questions, key=lambda q: q.split == HELDOUT)


async def _list_collection_ids(rf: RagforgeClient) -> list[UUID]:
    collections = [c.id for c in (await rf.list_collections(limit=100)).items][:20]
    if not collections:
        raise EvalSetupError("the tenant has no collections: nothing to ask over")
    return collections


async def run_eval(args: argparse.Namespace) -> Path:
    questions = asking_order(
        [
            q
            for q in load_questions(Path(args.questions))
            if args.split == "all" or q.split == args.split
        ],
        args.reveal,
    )
    key = read_key(resolve_key_file(args.key_file, dict(os.environ), DEFAULT_KEY_DIR))
    async with RagforgeClient(args.url, key, timeout=120.0) as rf:
        # Setup BEFORE the run file exists: a server that is down, a bad key or an empty tenant
        # must not leave a 0-row run behind to pollute `compare evals/private/runs/*.jsonl`.
        collections = await _list_collection_ids(rf)
        return await _record_run(args, rf, collections, questions)


async def _record_run(
    args: argparse.Namespace,
    rf: RagforgeClient,
    collections: list[UUID],
    questions: Sequence[Question],
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(args.out_dir) / f"{stamp}-{args.label}.jsonl"
    header = {
        "label": args.label,
        "split": args.split,
        "started": stamp,
        "url": args.url,
        "notes": args.notes,
        # Recorded so the bar can refuse a run whose deadline was itself under the p95 bar.
        "question_timeout_s": effective_timeout(args.question_timeout),
        "pause_s": getattr(args, "pause", 0.0),
        # The asked set, so a question deleted after the run fails the bar (removed_questions).
        "questions": {q.id: question_digest(q) for q in questions},
    }
    total = len(questions)
    n_hidden = sum(q.split == HELDOUT for q in questions) if not args.reveal else 0
    with open_private(dest) as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        peaks, stop = ResourcePeaks(), asyncio.Event()
        sampler_task = (
            asyncio.create_task(sample_until(ResourceSampler(), peaks, args.sample_interval, stop))
            if args.sample_interval > 0
            else None
        )
        outcomes: list[Outcome] = []
        try:
            for i, q in enumerate(questions, start=1):
                if i > 1 and getattr(args, "pause", 0.0) > 0:
                    await asyncio.sleep(args.pause)  # outside ask_one: never counted as latency
                hidden = n_hidden > 0 and q.split == HELDOUT
                if hidden and i == total - n_hidden + 1:
                    print(f"asking {n_hidden} held-out question(s); progress hidden", flush=True)
                outcome = await ask_one(rf, collections, q, args.question_timeout)
                outcomes.append(outcome)
                fh.write(json.dumps(asdict(outcome)) + "\n")
                fh.flush()
                if not hidden:
                    print(progress_line(i, total, q, outcome), flush=True)
            if n_hidden:
                print(f"[{total:>2}/{total}] {n_hidden} held-out question(s) done", flush=True)
        finally:
            # Written even on Ctrl-C/crash so a partial run still records what it cost.
            interrupted = await _stop_sampler(stop, sampler_task)
            footer = {
                "finished": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                "complete": len(outcomes) == len(questions),
                "observed": observed_arm(outcomes),
                "resources": asdict(peaks) | {"swap_delta_mb": peaks.swap_delta_mb},
            }
            fh.write(json.dumps({"footer": footer}) + "\n")
            if interrupted is not None:
                raise interrupted
    return dest


def open_private(dest: Path) -> TextIO:
    """Create the run file owner-only (0600) and never over an existing run.

    A run file holds model answers and verifier claims quoted from the owner's private notes, so
    it gets the same protection as its label sidecar (mkstemp, 0600). O_EXCL turns a same-second
    filename collision into a setup error instead of silently truncating an earlier run.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise EvalSetupError(f"run file already exists: {dest.name}") from None
    return os.fdopen(fd, "w", encoding="utf-8")


async def _stop_sampler(
    stop: asyncio.Event, task: asyncio.Task[None] | None
) -> BaseException | None:
    """Stop sampling without letting a SECOND interrupt lose the footer.

    A Ctrl-C (or cancel) that lands while we wait for the sampler is caught and handed back, so
    the caller writes the footer first and re-raises after. Sampler bugs are swallowed: they must
    never cost the run its footer.
    """
    stop.set()
    if task is None:
        return None
    try:
        with contextlib.suppress(Exception):
            await task
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        task.cancel()
        return exc
    return None


def effective_timeout(value: float | None) -> float | None:
    """`--question-timeout 0` (or negative) means no deadline; record that as None, not 0."""
    return value if value is not None and value > 0 else None


# ------------------------------------------------------------------ human labels (sidecar)
Verdict = Literal["ok", "invented", "unsure"]
VERDICTS: tuple[Verdict, ...] = get_args(Verdict)


@dataclass(frozen=True)
class Label:
    verdict: Verdict
    answer_sha256: str  # hex digest of the labelled final text; stale labels are ignored
    labelled_at: str


def labels_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem + ".labels.json")


def answer_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_sidecar(run_path: Path) -> dict[str, Any] | None:
    path = labels_path(run_path)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))  # JSONDecodeError is a ValueError
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected an object with a 'labels' object")
    return raw


def heldout_revealed_at(run_path: Path) -> str | None:
    """When held-out answers of this run were first shown to a human (None: never)."""
    raw = _read_sidecar(run_path)
    value = None if raw is None else raw.get("heldout_revealed_at")
    return value if isinstance(value, str) else None


def load_labels(run_path: Path) -> dict[str, Label]:
    """Missing sidecar -> no labels. A malformed one raises: silently dropping a human's labels
    would turn a measured invented% back into "unlabelled" without anyone noticing."""
    path = labels_path(run_path)
    raw = _read_sidecar(run_path)
    if raw is None:
        return {}
    entries = raw.get("labels")
    if not isinstance(entries, dict):
        raise ValueError(f"{path.name}: expected an object with a 'labels' object")
    labels: dict[str, Label] = {}
    for qid, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: invalid label for question {qid!r}")
        verdict = entry.get("verdict")
        if verdict not in VERDICTS or not isinstance(entry.get("answer_sha256"), str):
            raise ValueError(f"{path.name}: invalid label for question {qid!r}")
        labels[qid] = Label(verdict, entry["answer_sha256"], str(entry.get("labelled_at", "")))
    return labels


def save_labels(
    run_path: Path, labels: Mapping[str, Label], *, heldout_revealed_at: str | None = None
) -> None:
    """Atomic AND durable replace: these labels are the only ground truth for invented%.

    os.replace alone is atomic but not durable: after a power loss or kernel panic the rename can
    reach disk before the data, leaving an empty sidecar that `load_labels` rejects. So the temp
    file is fsynced before the rename and the directory (best effort) after it.
    """
    path = labels_path(run_path)
    previous = _read_sidecar(run_path) or {}
    old = previous.get("labels") if isinstance(previous.get("labels"), dict) else {}
    # Append-only history (review round 7, P1): every change of an existing verdict is kept with
    # both timestamps and both answer digests, so a relabel (e.g. resolving an "unsure") is
    # auditable, and a re-judgment of the same answer is told apart from a changed answer.
    raw_history = previous.get("history", [])
    if not isinstance(raw_history, list):
        # never silently restart an "append-only" record (review round 8, P2)
        raise EvalSetupError(f"{labels_path(run_path).name}: malformed label history; not saving")
    history = list(raw_history)
    for qid, lbl in sorted(labels.items()):
        before = old.get(qid) if isinstance(old, dict) else None
        if isinstance(before, dict) and (
            before.get("verdict") != lbl.verdict
            or before.get("answer_sha256") != lbl.answer_sha256  # same verdict, changed answer
        ):
            history.append(
                {
                    "id": qid,
                    "from": before.get("verdict"),
                    "to": lbl.verdict,
                    "from_at": before.get("labelled_at"),
                    "to_at": lbl.labelled_at,
                    "from_answer": before.get("answer_sha256"),
                    "to_answer": lbl.answer_sha256,
                }
            )
    if heldout_revealed_at is None and isinstance(previous.get("heldout_revealed_at"), str):
        heldout_revealed_at = previous["heldout_revealed_at"]  # a reveal stamp is never dropped
    body: dict[str, Any] = {
        "version": 1,
        "labels": {qid: asdict(lbl) for qid, lbl in sorted(labels.items())},
    }
    if history:
        body["history"] = history
    if heldout_revealed_at is not None:
        body["heldout_revealed_at"] = heldout_revealed_at
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """Persist the rename itself. Best effort: some filesystems refuse to open a directory."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def answered(o: Outcome) -> bool:
    return o.error is None and not abstained(o.final)


def is_heldout(q: Question | None, o: Outcome) -> bool:
    """Held-out if the question is held-out NOW or was when asked: moving a question to dev after
    a run must not reveal the held-out answer recorded for it."""
    return o.question_split == HELDOUT or (q is not None and q.split == HELDOUT)


RUBRIC = (
    "Label each answer against YOUR NOTES, not general truth: invented = any claim your notes do "
    "not support (even if true); unsure = you cannot tell without checking; ok = every claim is in "
    "your notes. Answers to must-abstain questions are counted as invented without asking."
)


def label_run(
    run_path: Path,
    questions: Sequence[Question],
    *,
    reveal: bool,
    relabel: bool,
    prompt: Callable[[str], str] | None = None,
    out: TextIO | None = None,
) -> dict[str, Label]:
    """Walk answered results and ask for ok/invented/unsure; saves after every answer.

    Held-out protocol: a human must read an answer to judge it, so labelling held-out answers
    (`--reveal`) necessarily exposes them. Do it ONCE, after the arm is frozen, to get the final
    held-out bar verdict. The first reveal is stamped as `heldout_revealed_at` in the sidecar and
    announced on every later reveal, so a tuning session can see the held-out rows were seen.
    """
    # Both resolved per call, so a redirected stdout or a patched input() is honoured.
    out = sys.stdout if out is None else out
    ask = input if prompt is None else prompt
    by_id = {q.id: q for q in questions}
    labels = load_labels(run_path)
    revealed_at = heldout_revealed_at(run_path)
    if reveal and revealed_at:
        print(f"note: held-out answers of this run were first revealed at {revealed_at}", file=out)
    run_file = load_run(run_path)
    print(RUBRIC, file=out)
    hidden = forced = 0
    for o in run_file.outcomes:
        q = by_id.get(o.id)
        if q is None or not answered(o):
            continue
        if is_heldout(q, o) and not reveal:
            hidden += 1
            continue
        if not q.answerable:
            forced += 1  # invented by construction (InventedStats): a label would be ignored
            continue
        current = labels.get(o.id)
        if current and current.answer_sha256 == answer_digest(o.final) and not relabel:
            continue
        if is_heldout(q, o) and revealed_at is None:
            # Stamped before the answer is shown: even a quit right after counts as a reveal.
            revealed_at = datetime.now(UTC).isoformat()
            save_labels(run_path, labels, heldout_revealed_at=revealed_at)
        _show_answer(out, q, o)
        verdict = _ask_verdict(ask)
        if verdict == "quit":
            break
        if verdict == "skip":
            continue
        labels[o.id] = Label(verdict, answer_digest(o.final), datetime.now(UTC).isoformat())
        save_labels(run_path, labels, heldout_revealed_at=revealed_at)
    if forced:
        print(
            f"{forced} answered must-abstain question(s) counted as invented, not shown", file=out
        )
    if hidden:
        print(f"{hidden} held-out answer(s) not shown (pass --reveal to label them)", file=out)
    return labels


def _show_answer(out: TextIO, q: Question, o: Outcome) -> None:
    """Model text is untrusted: strip terminal control characters before printing it."""
    model_cited = o.model_titles()
    inferred = o.inferred_titles or []
    print(f"\n=== {q.id} ===\nQ: {clean(q.question)}\n\n{clean(o.final)}\n", file=out)
    if model_cited is None:
        print(f"cited: {clean(', '.join(o.cited_titles)) or '-'}", file=out)
    else:
        print(f"cited by model: {clean(', '.join(model_cited)) or '-'}", file=out)
        print(f"inferred by server: {clean(', '.join(inferred)) or '-'}", file=out)


_KEYS: dict[str, Verdict | Literal["skip", "quit"]] = {
    "o": "ok",
    "i": "invented",
    "u": "unsure",
    "s": "skip",
    "q": "quit",
}


def _ask_verdict(prompt: Callable[[str], str]) -> Verdict | Literal["skip", "quit"]:
    while True:
        try:
            choice = prompt("[o]k / [i]nvented / [u]nsure / [s]kip / [q]uit: ").strip().lower()
        except EOFError:
            return "quit"
        if choice[:1] in _KEYS:
            return _KEYS[choice[:1]]


# ------------------------------------------------------------------ scoring
def abstained(text: str) -> bool:
    """The WHOLE text is a refusal (review round 2, P2: a prefix test graded "Your notes don't
    cover this. However, Raft ..." as a clean abstain and kept it out of invented%). Allowed:
    the model's refusal alone, or the server's abstain followed only by its closest-pages list.
    Apostrophe look-alikes are folded (U+2019 in "don<U+2019>t")."""
    folded = without_labels(text.translate(_APOSTROPHES)).strip()
    if _MODEL_REFUSAL_RE.fullmatch(folded):
        return True
    if not folded.startswith(ABSTAIN_PREFIX):
        return False
    rest = [ln.strip() for ln in folded[len(ABSTAIN_PREFIX) :].splitlines() if ln.strip()]
    return all(ln == _CLOSEST_HEADER or ln.startswith("- ") for ln in rest)


# A citation label as the answer text carries it: "[S3]", sanitizer-escaped "\[S3\]", "[S1, S2]".
_LABEL_TEXT = re.compile(r"\\?\[\s*S\d+(?:\s*,\s*S\d+)*\s*\\?\]", re.IGNORECASE)


def without_labels(text: str) -> str:
    return _LABEL_TEXT.sub(" ", text)


def fact_matched(fact: Fact, text: str) -> bool:
    # Labels are not content: "3" must not match "[S3]" (review round 3, P2).
    lowered = without_labels(text).lower()
    for m in fact.match:
        # Bare numbers must not match inside other numbers ("8" in "2048"), and words not inside
        # other words ("ack" in "stack", "cas" in "case"). Negation is NOT understood: "is not
        # single-threaded" still hits "single-threaded"; human labels catch that.
        if m.isdigit():
            # nor "0.3" or "3.5" (a decimal is a different number)
            pattern = rf"(?<![\d.]){re.escape(m)}(?![\d]|\.\d)"
        else:
            left = r"(?<!\w)" if m[:1].isalnum() else ""
            right = rf"{_SUFFIX}(?!\w)" if m[-1:].isalnum() else ""
            pattern = f"{left}{re.escape(m)}{right}"
        if re.search(pattern, lowered):
            return True
    return False


CiteSource = Literal["model", "inferred", "unknown", "none"]


@dataclass(frozen=True)
class Grade:
    id: str
    answerable: bool
    answered: bool
    cited_expected: bool  # model OR inferred citation of an expected page
    facts_hit: int
    facts_needed: int
    correct: bool
    content_correct: bool  # ignores the citation requirement
    draft_content_correct: bool  # the model's own draft, before the verifier replaced it
    flagged: bool  # verifier rejected claims or marked the answer unsupported
    cite_source: CiteSource = "none"  # who cited the expected page ("model" wins over inferred)
    correct_model: bool | None = None  # correct counting only model citations; None = unknown


def expected_cite_source(q: Question, o: Outcome) -> CiteSource:
    if not any(t in q.pages for t in o.cited_titles):
        return "none"
    model_cited = o.model_titles()
    if model_cited is None:
        return "unknown"
    return "model" if any(t in q.pages for t in model_cited) else "inferred"


def grade(q: Question, o: Outcome) -> Grade:
    is_answered = answered(o)
    if not q.answerable:
        ok = o.error is None and abstained(o.final)
        return Grade(q.id, False, is_answered, False, 0, 0, ok, ok, ok, False, "none", ok)
    needed = math.ceil(len(q.facts) * 2 / 3)
    hits = sum(fact_matched(f, o.final) for f in q.facts)
    draft_hits = sum(fact_matched(f, o.draft) for f in q.facts)
    source = expected_cite_source(q, o)
    content = is_answered and hits >= needed
    return Grade(
        id=q.id,
        answerable=True,
        answered=is_answered,
        cited_expected=source != "none",
        facts_hit=hits,
        facts_needed=needed,
        correct=content and source != "none",
        content_correct=content,
        draft_content_correct=o.error is None and draft_hits >= needed,
        flagged=is_answered and (bool(o.unsupported_claims) or o.status == "unsupported"),
        cite_source=source,
        correct_model=None if source == "unknown" else content and source == "model",
    )


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (with n=15, p95 is the slowest answer: say so, don't hide it)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else math.nan


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval, in percent. At n = 13 a single question moves the point
    estimate by 8 points: the interval says how little one run proves."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (binomial, p = 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2 * tail)


def paired(base: Sequence[Grade], other: Sequence[Grade]) -> tuple[int, int, int, float] | None:
    """(questions in common, correct only in base, correct only in other, McNemar p) over
    answerable questions both runs asked; None when they share none (different splits)."""
    a = {g.id: g.correct for g in base if g.answerable}
    b = {g.id: g.correct for g in other if g.answerable}
    common = a.keys() & b.keys()
    if not common:
        return None
    only_a = sum(a[i] and not b[i] for i in common)
    only_b = sum(b[i] and not a[i] for i in common)
    return len(common), only_a, only_b, mcnemar_exact(only_a, only_b)


@dataclass(frozen=True)
class InventedStats:
    """invented% over answered results, from human labels plus one rule needing no human.

    The rule: an answer to a must-abstain question is invented BY CONSTRUCTION. Such a question is
    in the set precisely because the notes cannot answer it, so whatever the arm said came from
    somewhere other than the notes. A human reading only the answer cannot see that (a plausible
    general-knowledge answer looks "ok"), so these never ask for a label and a label on one is
    ignored. They count as labelled AND invented; `forced` says how many there were.

    `status` is "labelled" only when every answered result carries a current label; anything less
    is reported as partial, because an unlabelled answer is exactly where invention could hide.
    """

    answered: int
    labelled: int
    invented: int
    unsure: int
    forced: int = 0  # answered must-abstain questions, included in labelled and invented

    @property
    def status(self) -> str:
        if self.answered == 0:
            return "no answers"
        if self.labelled == 0:
            return "unlabelled"
        if self.labelled < self.answered:
            return f"partial {self.labelled}/{self.answered}"
        return "labelled"

    @property
    def pct(self) -> float:
        return _pct(self.invented, self.labelled)

    @property
    def upper_pct(self) -> float:
        """Conservative: "unsure" counts as invented until a human resolves it."""
        return _pct(self.invented + self.unsure, self.labelled)


def invented_stats(
    outcomes: Iterable[Outcome],
    labels: Mapping[str, Label],
    must_abstain: Collection[str] = frozenset(),
) -> InventedStats:
    """Labels apply only when their digest matches the recorded answer text (stale -> ignored).
    Answers to `must_abstain` question ids are invented without a label (see InventedStats)."""
    shown = [o for o in outcomes if answered(o)]
    forced = sum(o.id in must_abstain for o in shown)
    current = [
        labels[o.id]
        for o in shown
        if o.id not in must_abstain
        and o.id in labels
        and labels[o.id].answer_sha256 == answer_digest(o.final)
    ]
    return InventedStats(
        answered=len(shown),
        labelled=len(current) + forced,
        invented=sum(lbl.verdict == "invented" for lbl in current) + forced,
        unsure=sum(lbl.verdict == "unsure" for lbl in current),
        forced=forced,
    )


def is_timeout(o: Outcome) -> bool:
    """Structured flag, or the error text written by runs that predate the flag."""
    return o.timed_out or (o.error or "").startswith("timeout after ")


def latency_samples(outcomes: Iterable[Outcome], question_timeout_s: float | None) -> list[float]:
    """Turn latencies for p50/p95, censoring-aware.

    A completed turn counts at its measured time. A timed-out turn took AT LEAST the deadline, so
    it counts at max(measured, deadline): dropping it would bias p95 toward "fast" exactly for the
    arms that stall. Any other error (a 0.1 s 5xx) is excluded: it is not a fast answer, and the
    bar fails on errors separately (`judge_bar`), so excluding it cannot help an arm pass.
    """
    deadline = question_timeout_s or 0.0
    values: list[float] = []
    for o in outcomes:
        if is_timeout(o):
            values.append(max(o.total_s, deadline))
        elif o.error is None:
            values.append(o.total_s)
    return values


def summarize(
    questions: list[Question],
    outcomes: list[Outcome],
    labels: Mapping[str, Label] | None = None,
    *,
    question_timeout_s: float | None = None,
    recorded_ids: Collection[str] = (),
) -> dict[str, Any]:
    """`recorded_ids` is the run header's asked set (`recorded_question_ids`); see
    `removed_questions` for why a question deleted since the run must be reported."""
    by_id = {q.id: q for q in questions}
    known = [o for o in outcomes if o.id in by_id]
    grades = [grade(by_id[o.id], o) for o in known]
    must_abstain = {q.id for q in questions if not q.answerable}
    ans = [g for g in grades if g.answerable]
    una = [g for g in grades if not g.answerable]
    ans_answered = [g for g in ans if g.answered]
    provenance_known = all(g.correct_model is not None for g in ans)
    inv = invented_stats(known, labels or {}, must_abstain)
    totals = latency_samples(known, question_timeout_s)
    ttfts = [o.first_token_s for o in known if o.error is None and o.first_token_s is not None]
    return {
        "n": len(grades),
        "correct": _pct(sum(g.correct for g in ans), len(ans)),
        "correct_model": (
            _pct(sum(bool(g.correct_model) for g in ans), len(ans)) if provenance_known else None
        ),
        "content": _pct(sum(g.content_correct for g in ans), len(ans)),
        "draft": _pct(sum(g.draft_content_correct for g in ans), len(ans)),
        "answered": _pct(len(ans_answered), len(ans)),
        "cited": _pct(sum(g.cited_expected for g in ans), len(ans)),
        "cited_model": (
            _pct(sum(g.cite_source == "model" for g in ans), len(ans)) if provenance_known else None
        ),
        "cited_inferred": (
            _pct(sum(g.cite_source == "inferred" for g in ans), len(ans))
            if provenance_known
            else None
        ),
        "abstain_ok": _pct(sum(g.correct for g in una), len(una)),
        "abstain_n": (sum(g.correct for g in una), len(una)),
        "flagged": _pct(sum(g.flagged for g in ans_answered), len(ans_answered)),
        # What the reader sees: a correct answer can still be `partial` (claims highlighted).
        "verified": _pct(
            sum(o.status == "verified" for o in known if o.id in {g.id for g in ans_answered}),
            len(ans_answered),
        ),
        "correct_ci": wilson(sum(g.correct for g in ans), len(ans)),
        "invented": inv,
        "errors": sum(o.error is not None for o in known),
        "timeouts": sum(is_timeout(o) for o in known),
        "question_timeout_s": question_timeout_s,
        "tool_turns": sum(bool(o.tools) for o in known),
        "p50_s": percentile(totals, 50),
        "p95_s": percentile(totals, 95),
        "ttft_p50_s": percentile(ttfts, 50),
        "models": sorted({m for o in known for m in o.models}),
        "changed": sorted(o.id for o in known if question_changed(by_id[o.id], o)),
        "removed": removed_questions(questions, outcomes, recorded_ids),
        "grades": grades,
    }


def summarize_run(run_file: RunFile, questions: list[Question]) -> dict[str, Any]:
    """`summarize` with everything the run file itself contributes (labels, deadline, asked set),
    so `score` and `compare` cannot drift apart on what they grade against."""
    return summarize(
        questions,
        run_file.outcomes,
        load_labels(run_file.path),
        question_timeout_s=header_timeout(run_file.header),
        recorded_ids=recorded_question_ids(run_file.header),
    )


def fmt(value: float | None, digits: int = 0) -> str:
    """Absent or undefined values print as a dash, never as a misleading 0 or "nan"."""
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_latency(s: Mapping[str, Any], key: str) -> str:
    """With timed-out turns in the sample the percentile is a lower bound: say so with ">="."""
    text = fmt(s[key], 1)
    return f"≥{text}" if s.get("timeouts") and text != "-" else text


def header_timeout(header: Mapping[str, Any]) -> float | None:
    value = header.get("question_timeout_s")
    return float(value) if isinstance(value, int | float) and value > 0 else None


def fmt_invented(inv: InventedStats) -> str:
    if inv.status in ("unlabelled", "no answers"):
        return inv.status
    # The headline is the number the bar judges: "unsure" counts as invented (upper_pct). The
    # optimistic figure follows in brackets (review round 6, P2: "0 (+3 unsure)" read as 0%).
    text = (
        f"{fmt(inv.upper_pct)} ({fmt(inv.pct)} if {inv.unsure} unsure are ok)"
        if inv.unsure
        else fmt(inv.pct)
    )
    if inv.forced:
        text += f" (incl. {inv.forced} answered must-abstain)"
    return text if inv.status == "labelled" else f"{text} [{inv.status}]"


COLUMNS = (
    ("correct", "correct%"),
    ("content", "content%"),
    ("draft", "draft%"),
    ("cited", "cited%"),
    ("answered", "answered%"),
    ("abstain_ok", "abstain%"),
    ("flagged", "flagged%"),
    ("p50_s", "p50 s"),
    ("p95_s", "p95 s"),
    ("ttft_p50_s", "ttft s"),
)


def _cited_split(s: Mapping[str, Any]) -> str:
    if s["cited_model"] is None:
        return "?"
    return f"{fmt(s['cited_model'])}/{fmt(s['cited_inferred'])}"


def print_details(
    label: str,
    grades: Sequence[Grade],
    hidden_ids: Collection[str],
    reveal: bool,
    out: TextIO,
    changed: Collection[str] = frozenset(),
) -> None:
    print(f"\n{label}:", file=out)
    hidden = 0
    for g in grades:
        if g.id in hidden_ids and not reveal:
            hidden += 1
            continue
        mark = "ok " if g.correct else "-- "
        facts = f"facts {g.facts_hit}/{g.facts_needed}" if g.answerable else "abstain"
        cite = {"model": "cited", "inferred": "inferred", "unknown": "cited"}.get(
            g.cite_source, "uncited"
        )
        ans = "answered" if g.answered else "abstained"
        note = "  (question changed since the run)" if g.id in changed else ""
        print(f"  {mark}{g.id:<34} {ans:<10} {cite:<8} {facts}{note}", file=out)
    if hidden:
        print(f"  ({hidden} held-out row(s) hidden; pass --reveal to show them)", file=out)


def score_runs(
    paths: Sequence[str],
    questions_path: str,
    verbose: bool,
    *,
    reveal: bool = False,
    out: TextIO | None = None,
) -> None:
    out = sys.stdout if out is None else out
    questions = load_questions(Path(questions_path))
    print(
        "| run | n | "
        + " | ".join(h for _, h in COLUMNS)
        + " | model-only% | cited model/inferred% | invented% | tools | err | model |",
        file=out,
    )
    print("|---" * (len(COLUMNS) + 8) + "|", file=out)
    by_id = {q.id: q for q in questions}
    details: list[tuple[str, dict[str, Any], set[str]]] = []
    warnings: list[str] = []
    for path in paths:
        run_file = load_run(Path(path))
        s = summarize_run(run_file, questions)
        label = _md_cell(str(run_file.header.get("label", "?")))
        split = _md_cell(str(run_file.header.get("split", "?")))
        cells = " | ".join(fmt_latency(s, k) if k.endswith("_s") else fmt(s[k]) for k, _ in COLUMNS)
        print(
            f"| {label} ({split}) | {s['n']} | {cells} "
            f"| {fmt(s['correct_model'])} | {_cited_split(s)} | {fmt_invented(s['invented'])} "
            f"| {s['tool_turns']} "
            f"| {s['errors']} | {_md_cell(', '.join(s['models']))} |",
            file=out,
        )
        # Counts only: ids of held-out questions must not leak here.
        if s["changed"]:
            warnings.append(f"warning: {label}: {len(s['changed'])} question(s) changed since run")
        if s["removed"]:
            warnings.append(f"warning: {label}: {len(s['removed'])} question(s) removed since run")
        hidden = {o.id for o in run_file.outcomes if is_heldout(by_id.get(o.id), o)}
        details.append((label, s, hidden))
    for line in warnings:
        print(line, file=out)
    if verbose:
        for label, s, hidden in details:
            print_details(label, s["grades"], hidden, reveal, out, s["changed"])


# ------------------------------------------------------------------ compare arms
@dataclass(frozen=True)
class BarVerdict:
    """`met` is None when the bar cannot be judged yet (invented% not fully labelled)."""

    met: bool | None
    reasons: tuple[str, ...]

    def render(self) -> str:
        if self.met:
            return "**yes**"
        word = "no" if self.met is False else "unknown"
        return f"{word}: {'; '.join(self.reasons)}"


@dataclass(frozen=True)
class Completeness:
    """Did the run ask every question of its split? A bar verdict over a subset is meaningless:
    an arm that answered 3 easy questions before swapping itself to death would look perfect.

    `finished` is the footer's `complete` flag; None when there is no footer (older format, or
    the process was killed so hard that the `finally` never ran).
    """

    asked: int  # distinct questions of the split with a recorded outcome
    expected: int  # questions in the run's split, per the current questions file
    finished: bool | None

    @property
    def missing(self) -> bool:
        return self.finished is False or self.asked < self.expected

    def render(self) -> str:
        if self.missing:
            return f"{self.asked}/{self.expected} incomplete"
        if self.finished is None:
            return f"{self.asked}/{self.expected} no footer"
        return f"{self.asked}/{self.expected}"


def run_completeness(run_file: RunFile, questions: Sequence[Question]) -> Completeness:
    split = run_file.header.get("split", "all")
    wanted = {q.id for q in questions if split == "all" or q.split == split}
    asked = {o.id for o in run_file.outcomes} & wanted
    finished = None if run_file.footer is None else run_file.footer.get("complete") is True
    return Completeness(asked=len(asked), expected=len(wanted), finished=finished)


def judge_bar(s: Mapping[str, Any], completeness: Completeness) -> BarVerdict:
    """Success bar: correct >= 80%, p95 < 10 s, invented <= 2% (unsure counted as invented).

    Errors and deadlines: any errored turn is a miss, because the user got no answer and the
    error's true latency is unknown (a timeout's is only bounded below). A run recorded with a
    question timeout under the p95 bar is a miss too: such a deadline makes "p95 < 10 s" true by
    construction for every turn that completed, so it measures the deadline, not the arm.

    Answered must-abstain questions count as invented without a label (InventedStats), so an arm
    that answers what the notes cannot answer cannot pass on "ok" labels. A question edited since
    the run (text, split or answerable flag) is a miss: the run measured a different question set.
    So is a question removed since the run (dropping its row would re-grade over a smaller,
    friendlier set) or added since (the run is then incomplete for the current split).

    A measured miss beats "unknown": if a run is incomplete, or correct/p95/invented already fail,
    the arm is "no" whatever is still unlabelled, so the owner is not asked to label answers from
    an arm that cannot win. For partial labels, invented/answered is a lower bound on the final
    rate (labelling more can only add invented answers to the numerator; "unsure" is left out
    because a relabel may turn it into "ok"), so exceeding the bar with it is already a miss.
    """
    misses: list[str] = []
    unknowns: list[str] = []
    if completeness.missing:
        misses.append(f"incomplete {completeness.asked}/{completeness.expected}")
    elif completeness.finished is None:
        unknowns.append("no footer")
    misses.extend(_error_misses(s))
    if s.get("changed"):
        misses.append(f"{len(s['changed'])} question(s) changed since the run")
    if s.get("removed"):
        misses.append(f"{len(s['removed'])} question(s) removed since the run")
    if not s["correct"] >= BAR_CORRECT_PCT:  # NaN (no answerable questions) is a miss
        misses.append(f"correct {fmt(s['correct'])}<{BAR_CORRECT_PCT:.0f}")
    if not s["p95_s"] < BAR_P95_S:
        misses.append(f"p95 {fmt_latency(s, 'p95_s')}≥{BAR_P95_S:.0f}s")
    inv: InventedStats = s["invented"]
    if inv.status in ("labelled", "no answers"):
        upper = inv.upper_pct if inv.labelled else 0.0
        if upper > BAR_INVENTED_PCT:
            forced = f" (incl. {inv.forced} answered must-abstain)" if inv.forced else ""
            misses.append(f"invented {fmt(upper, 1)}>{BAR_INVENTED_PCT:.0f}{forced}")
    else:
        lower = _pct(inv.invented, inv.answered)
        if lower > BAR_INVENTED_PCT:
            misses.append(f"invented ≥{fmt(lower, 1)}>{BAR_INVENTED_PCT:.0f} ({inv.status})")
        else:
            unknowns.append(f"invented% {inv.status}")
    if misses:
        return BarVerdict(False, tuple(misses))
    if unknowns:
        return BarVerdict(None, tuple(unknowns))
    return BarVerdict(True, ())


def _error_misses(s: Mapping[str, Any]) -> list[str]:
    misses: list[str] = []
    errors, timeouts = s.get("errors", 0), s.get("timeouts", 0)
    if errors:
        detail = f" ({timeouts} timed out)" if timeouts else ""
        misses.append(f"{errors} errored turn(s){detail}")
    deadline = s.get("question_timeout_s")
    if isinstance(deadline, int | float) and deadline < BAR_P95_S:
        misses.append(f"question-timeout {deadline:g}s<{BAR_P95_S:.0f}s")
    return misses


def _mb_to_gb(value: Any) -> str:
    return fmt(float(value) / 1024, 1) if isinstance(value, int | float) else "-"


def compare_runs(paths: Sequence[str], questions_path: str, out: TextIO | None = None) -> None:
    """Markdown table across arms. Aggregates only, so it is safe for held-out runs."""
    out = sys.stdout if out is None else out
    questions = load_questions(Path(questions_path))
    print(
        "| arm | split | asked | model | notes | correct% (95% CI) | vs first arm "
        "| model-only% | content% | verified% | cited model/inferred% | abstain ok | invented% "
        "| err | p50 s | p95 s | ttft s | peak swap GB | swap Δ GB | peak ollama GB | meets bar |",
        file=out,
    )
    print("|---" * 21 + "|", file=out)
    base: list[Grade] | None = None
    for path in paths:
        run_file = load_run(Path(path))
        s = summarize_run(run_file, questions)
        done = run_completeness(run_file, questions)
        res = (run_file.footer or {}).get("resources", {})
        ok, n_una = s["abstain_n"]
        delta = swap_delta(res.get("swap_used_mb"), res.get("swap_baseline_mb"))
        cells = [
            _md_cell(str(run_file.header.get("label", "?"))),
            _md_cell(str(run_file.header.get("split", "?"))),
            done.render(),
            _md_cell(arm_model(run_file.outcomes)) or "-",
            _md_cell(str(run_file.header.get("notes") or "")) or "-",
            f"{fmt(s['correct'])} ({fmt(s['correct_ci'][0])}-{fmt(s['correct_ci'][1])})",
            _fmt_paired(None if base is None else paired(base, s["grades"])),
            fmt(s["correct_model"]),
            fmt(s["content"]),
            fmt(s["verified"]),
            _cited_split(s),
            f"{ok}/{n_una}",
            fmt_invented(s["invented"]),
            str(s["errors"]),
            fmt_latency(s, "p50_s"),
            fmt_latency(s, "p95_s"),
            fmt(s["ttft_p50_s"], 1),
            _mb_to_gb(res.get("swap_used_mb")),
            _mb_to_gb(delta),
            _mb_to_gb(res.get("ollama_rss_mb")),
            judge_bar(s, done).render(),
        ]
        print("| " + " | ".join(cells) + " |", file=out)
        base = s["grades"] if base is None else base
    print(
        "\nvs first arm: paired on the answerable questions both runs asked, "
        "+k/-m = correct only here / only in the first arm, exact McNemar p. verified% = answered "
        "answerable questions shown as `verified` (the rest show highlighted claims).",
        file=out,
    )
    print(
        f"\nBar: correct ≥ {BAR_CORRECT_PCT:.0f}% (model or inferred citation), "
        f"p95 < {BAR_P95_S:.0f} s, invented ≤ {BAR_INVENTED_PCT:.0f}% of answered (human labels; "
        "unsure counts as invented), every question of the split asked, no errored turn, and a "
        f"question timeout (if any) of at least {BAR_P95_S:.0f} s. A timed-out turn counts in "
        "latency at its deadline (≥ marks a lower bound); other errored turns are excluded from "
        "latency. p95 is nearest-rank: at n < 20 it is the slowest answer. swap Δ = peak minus "
        "the pre-run reading. An answer to a must-abstain question counts as invented without a "
        "label; a question whose text, split or answerable flag changed since the run, or that "
        "was removed or added since, fails the bar.",
        file=out,
    )


def _fmt_paired(result: tuple[int, int, int, float] | None) -> str:
    if result is None:
        return "-"
    n, only_base, only_here, p = result
    return f"n={n} +{only_here}/-{only_base} p={p:.2f}"


def arm_model(outcomes: Iterable[Outcome]) -> str:
    """ "provider: model" as the server reported it, so a hosted arm is never mistaken for local."""
    arm = observed_arm(outcomes)
    models = ", ".join(arm["models"])
    providers = ", ".join(arm["providers"])
    return f"{providers}: {models}" if providers and models else models


def _md_cell(text: str) -> str:
    """Notes are free text typed by the owner: keep them from breaking the table."""
    return clean(text).replace("|", "\\|").replace("\n", " ").strip()


# ------------------------------------------------------------------ CLI
def arm_label(value: str) -> str:
    """argparse type: the label is used in the run filename, so "../x" must not escape --out-dir."""
    if not _LABEL_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("use 1-64 of A-Z a-z 0-9 . _ : - (no slashes or spaces)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge-eval", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.toml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    reveal_help = "show per-question rows for held-out questions (they are hidden by default)"
    r = sub.add_parser("run", help="ask every question and record raw outcomes")
    r.add_argument(
        "--label", required=True, type=arm_label, help="arm name, e.g. A2-qwen4b ([A-Za-z0-9._:-])"
    )
    r.add_argument("--split", choices=(*SPLITS, "all"), default="dev")
    r.add_argument("--url", default=DEFAULT_URL)
    r.add_argument("--key-file")
    r.add_argument("--out-dir", default=str(EVAL_DIR / "runs"))
    r.add_argument("--notes", default="", help="arm settings the client cannot observe")
    r.add_argument(
        "--sample-interval", type=float, default=2.0, help="memory sampling seconds; 0 disables"
    )
    r.add_argument(
        "--question-timeout",
        type=float,
        default=DEFAULT_QUESTION_TIMEOUT_S,
        help="seconds before one question is recorded as a timeout error; 0 disables",
    )
    r.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds to wait between questions, e.g. to stay under a hosted model's free-tier "
        "requests-per-minute quota (gemini-3.5-flash free tier: 5 RPM); not counted in latency",
    )
    r.add_argument("--reveal", action="store_true", help=reveal_help)
    s = sub.add_parser("score", help="grade one or more recorded runs")
    s.add_argument("runs", nargs="+")
    s.add_argument("-v", "--verbose", action="store_true", help="per-question grades")
    s.add_argument("--reveal", action="store_true", help=reveal_help)
    c = sub.add_parser("compare", help="markdown table across arms with success-bar flags")
    c.add_argument("runs", nargs="+")
    lb = sub.add_parser("label", help="label answered results ok / invented / unsure")
    lb.add_argument("run")
    lb.add_argument("--reveal", action="store_true", help="also label held-out answers")
    lb.add_argument("--relabel", action="store_true", help="revisit answers already labelled")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not Path(args.questions).is_file():
        # The default is relative to the repo root (`make eval-private` runs there); elsewhere it
        # would surface as a raw FileNotFoundError traceback (review round 3, P2).
        raise SystemExit(
            f"ragforge-eval: no question set at {clean(args.questions)}; run from the repo root "
            "(make eval-private) or pass --questions"
        )
    if args.cmd == "run":
        try:
            dest = asyncio.run(run_eval(args))
        except KeyboardInterrupt:
            # A started run file already holds every finished row and a complete=false footer.
            print(
                "ragforge-eval: interrupted; the partial run is recorded as incomplete",
                file=sys.stderr,
            )
            raise SystemExit(130) from None
        except (EvalSetupError, KeyFileError, ApiError) as exc:
            # Messages carry paths and HTTP problem titles only, never the key itself. Problem
            # detail is server text, so it is stripped of terminal control characters too.
            raise SystemExit(f"ragforge-eval: {clean(str(exc))}") from None
        except httpx.HTTPError as exc:
            # Setup-time transport failure (API not up). Per-question ones never get here
            # (`ask_one` records them). Only the type is printed: the URL is the user's own flag.
            raise SystemExit(
                f"ragforge-eval: cannot reach {clean(args.url)}: {type(exc).__name__}"
            ) from None
        print(f"\nrecorded {dest}")
        score_runs([str(dest)], args.questions, verbose=True, reveal=args.reveal)
    elif args.cmd == "score":
        score_runs(args.runs, args.questions, args.verbose, reveal=args.reveal)
    elif args.cmd == "compare":
        compare_runs(args.runs, args.questions)
    else:
        questions = load_questions(Path(args.questions))
        try:
            label_run(Path(args.run), questions, reveal=args.reveal, relabel=args.relabel)
        except KeyboardInterrupt:
            # Nothing is lost: label_run saves after every answer. Same exit shape as `run`.
            print(
                f"\nragforge-eval: interrupted; labels saved, resume with: label {args.run}",
                file=sys.stderr,
            )
            raise SystemExit(130) from None


if __name__ == "__main__":
    run()
