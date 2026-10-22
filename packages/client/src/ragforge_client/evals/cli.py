"""`ragforge-eval2`: the v2 eval CLI (ADR-0017). The Phase 2b `ragforge-eval` (evalrun.py) is
untouched and keeps its commands; this one drives the eval layer:

    run <kind> --dataset demo|private --arm A [--resume]   one E-R/E-G/E-V/E-I/E-A run
    summarize <run>                                        (re)write <run>.summary.json
    label <run> [--adjudicate]                             the blind label queue (terminal)
    gate --surface S [--override "..."]                    the regression gate
    baseline set <run> --note "..."                        move a surface's baseline
    snapshot create|status|reembed                         the frozen eval snapshot
    demo                                                   D0 end to end (live stack + model)
    results --check | --write                              docs/RESULTS.md tables (export.py)

Layout: `<root>/<dataset>/` holds the dataset files, `labels.jsonl` and `judge-cache.jsonl`;
`<root>/<dataset>/runs/` holds run files, summaries, baselines, gate verdicts and `target.json`
(the collections a dataset runs against, written by `demo` and `snapshot create`).

Keys: the eval key lives in `~/.config/ragforge/eval/` (a different directory from the owner's
key, so `make ask` never picks it up); the demo tenant's key lives apart in
`~/.config/ragforge/eval/demo/`, so each dataset uses its own key. Key files follow the existing
rules (0600, never printed). Admin commands (`snapshot`, `demo`) use the owner's key directory.

The sibling modules are reached only through the interfaces.py seams, gathered in `Components`
(tests inject fakes there): metrics.summarize, gate.evaluate, judge.make_judge (+ its cache) and
datasets.LabelStore / load_datasets / item_digest_of.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tomllib
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO, cast
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragforge_client.ask import (
    DEFAULT_KEY_DIR,
    DEFAULT_URL,
    KeyFileError,
    clean,
    read_key,
    resolve_key_file,
)
from ragforge_client.client import ApiError, RagforgeClient
from ragforge_client.evals import datasets as datasets_mod
from ragforge_client.evals import export as export_mod
from ragforge_client.evals import gate as gate_mod
from ragforge_client.evals import judge as judge_mod
from ragforge_client.evals import metrics as metrics_mod
from ragforge_client.evals.interfaces import Datasets, Judge, LabelReader, LabelWriter
from ragforge_client.evals.runfile import RunFile, RunFileError, iter_runs, read_run
from ragforge_client.evals.runners import (
    JUDGE_CACHE,
    EvalApi,
    Pacer,
    RetryPolicy,
    RunAborted,
    RunContext,
    Summarizer,
    checked_texts,
    find_resumable,
    load_judge_results,
    run_application,
    run_generator,
    run_interview,
    run_latency,
    run_retrieval,
    run_verifier,
    summarize_run,
    write_private,
)
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    AnswerOutput,
    Criterion,
    EvalKind,
    GateOutcome,
    GateVerdict,
    InterviewSet,
    JudgeResult,
    Label,
    LabelKind,
    Question,
    QuestionSet,
    RetrievalConfig,
    RunRow,
    Split,
    Surface,
    VerifierSet,
    digest,
)
from ragforge_contracts.api import (
    Accepted,
    ApiScope,
    Collection,
    EvalReembedRequest,
    EvalSnapshot,
    EvalSnapshotRequest,
    Job,
    JobEvent,
    TenantProvisioned,
)

DEFAULT_ROOT = Path("evals")
DEFAULT_EVAL_KEY_DIR = DEFAULT_KEY_DIR / "eval"
DATASETS = ("demo", "private")
KINDS = tuple(k.value for k in EvalKind)
QUESTIONS_FILE, VERIFIER_FILE, INTERVIEW_FILE = "questions.toml", "verifier.json", "interview.json"
LABELS_FILE, TARGET_FILE = "labels.jsonl", "target.json"
DEMO_KEY_NAME, DEMO_KEY_DIR = "demo.key", "demo"
DEMO_OWNER_KEY_NAME = "owner.key"  # the demo tenant's owner key, bootstrapped by make eval-demo
# All five; application adds its latency pass. The judged kinds (generator, interview) run
# LAST: the latency pass must not share the 16 GB host with a judge model still loaded
# (Ollama's keep_alive), or the judge's memory pressure would show up as chat latency.
DEMO_KINDS = (
    EvalKind.retrieval,
    EvalKind.application,
    EvalKind.verifier,
    EvalKind.generator,
    EvalKind.interview,
)
EVAL_KEY_SCOPES = (ApiScope.eval, ApiScope.chat, ApiScope.read)
# the worst outcome wins when one surface's checks come from several eval kinds
_SEVERITY = {  # the gate's own order: an incomparable pair outranks a block
    GateOutcome.pass_: 0,
    GateOutcome.warn: 1,
    GateOutcome.block: 2,
    GateOutcome.not_evaluable: 3,
}
SURFACE_KINDS: dict[Surface, tuple[EvalKind, ...]] = {
    Surface.chat: (EvalKind.retrieval, EvalKind.generator, EvalKind.application, EvalKind.latency),
    Surface.verifier: (EvalKind.verifier,),
    Surface.interview: (EvalKind.interview,),
    Surface.judge: (),
}
# ADR-0017 §5: a chat change must also pass the E-V hard floor (false-verified: McNemar on the
# planted items against the verifier baseline AND the absolute SLO). gate.evaluate judges one
# kind, so the chat gate adds the verifier surface's own verdict and ANDs it in (`all_pass`).
HARD_FLOORS: dict[Surface, tuple[tuple[Surface, EvalKind], ...]] = {
    Surface.chat: ((Surface.verifier, EvalKind.verifier),),
}


class CliError(Exception):
    """A clean, user-facing failure (exit 2). Never carries a key or note text."""


# ------------------------------------------------------------------ sibling modules
class LabelStore(LabelReader, LabelWriter, Protocol):
    """datasets.LabelStore: atomic, locked appends plus the newest label per item."""


class GateEvaluate(Protocol):
    def __call__(
        self,
        baseline: RunFile,
        candidate: RunFile,
        aa_runs: Sequence[RunFile],
        datasets: Datasets,
        /,
    ) -> GateVerdict: ...


class OpenJudge(Protocol):
    """Opens a cache-backed judge for one prompt; closing it closes the provider client."""

    def __call__(
        self, table: Mapping[str, Any], prompt: str, cache: Path, /
    ) -> AbstractAsyncContextManager[Judge]: ...


def judge_config(table: Mapping[str, Any], prompt: str) -> judge_mod.JudgeConfig:
    """The `[judge]` table as a JudgeConfig; `key_file` is a path (the judge reads it)."""
    fields = dict(table)
    if fields.get("key_file") is not None:
        fields["key_file"] = Path(str(fields["key_file"])).expanduser()
    try:
        return judge_mod.JudgeConfig(**{**fields, "prompt": prompt})
    except TypeError:
        raise CliError("the judge config has an unknown field (see judge.JudgeConfig)") from None


@asynccontextmanager
async def open_cached_judge(
    table: Mapping[str, Any], prompt: str, cache: Path
) -> AsyncIterator[Judge]:
    """judge.make_judge (probes the provider for the model version the header records), behind
    the judge cache, so a re-run or a resumed judge phase never pays for a score twice."""
    try:
        inner = await judge_mod.make_judge(judge_config(table, prompt))
    except judge_mod.JudgeConfigError as exc:
        raise CliError(f"judge: {exc}") from None
    except judge_mod.JudgeTransportError as exc:
        raise CliError(f"judge unreachable ({exc.provider}, status {exc.status})") from None
    async with inner:
        yield judge_mod.CachingJudge(inner, judge_mod.JudgeCache(cache))


@dataclass(frozen=True)
class Components:
    """The interfaces.py seams this CLI calls; tests replace any of them with a fake."""

    summarize: Summarizer = metrics_mod.summarize
    gate_evaluate: GateEvaluate = gate_mod.evaluate
    open_judge: OpenJudge = open_cached_judge
    label_store: Callable[[Path], LabelStore] = datasets_mod.LabelStore


# ------------------------------------------------------------------ files
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmConfig(_Strict):
    """`--arm-config` TOML: what the server cannot report about itself (the header records it).
    Prompt versions the server reports on `done` override these."""

    chat_model: str | None = None
    verifier_model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    retrieval: RetrievalConfig | None = None


class Target(_Strict):
    """`runs/target.json`: what a dataset runs against (no secrets)."""

    collection_ids: list[UUID] = Field(min_length=1, max_length=20)
    snapshot_collection_id: UUID | None = None
    tenant_id: UUID | None = None


class Baseline(_Strict):
    """`runs/baseline-<surface>-<kind>.json`: moved only by `baseline set` (ADR-0017 §5)."""

    run: str  # the run file's name, in the same runs dir
    run_id: str
    eval_kind: EvalKind
    surface: Surface
    header_digest: str
    note: str
    set_at: datetime


@dataclass(frozen=True)
class Paths:
    root: Path
    dataset: str

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.dataset

    @property
    def runs_dir(self) -> Path:
        return self.dataset_dir / "runs"

    @property
    def target(self) -> Path:
        return self.runs_dir / TARGET_FILE

    def baseline(self, surface: Surface, kind: EvalKind) -> Path:
        return self.runs_dir / f"baseline-{surface.value}-{kind.value}.json"

    def gate(self, surface: Surface) -> Path:
        return self.runs_dir / f"gate-{surface.value}.json"


def load_datasets(directory: Path) -> Datasets:
    """datasets.load_datasets (questions.toml + a sealed heldout2.toml, verifier.json,
    interview.json; each optional: a kind that needs a missing one says so)."""
    try:
        return datasets_mod.load_datasets(directory)
    except (datasets_mod.DatasetError, ValidationError, tomllib.TOMLDecodeError) as exc:
        raise CliError(f"{directory}: {_first_error(exc)}") from None


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return str(exc)


def dataset_for(kind: EvalKind, datasets: Datasets) -> QuestionSet | VerifierSet | InterviewSet:
    found: QuestionSet | VerifierSet | InterviewSet | None
    if kind is EvalKind.verifier:
        found, name = datasets.verifier, VERIFIER_FILE
    elif kind is EvalKind.interview:
        found, name = datasets.interview, INTERVIEW_FILE
    else:
        found, name = datasets.questions, QUESTIONS_FILE
    if found is None:
        raise CliError(f"the {kind.value} eval needs {name}")
    return found


def load_model[M: BaseModel](path: Path, model: type[M], what: str) -> M:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CliError(f"no {what} at {path}") from None
    except ValidationError as exc:
        raise CliError(f"{path}: not a valid {what} ({_first_error(exc)})") from None


def save_model(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(path, value.model_dump_json(indent=2) + "\n")


ARM_FILE = "arm.toml"


def load_arm_config(path: str | None, dataset_dir: Path | None = None) -> ArmConfig:
    """`--arm-config`, else the dataset's own `arm.toml` (what the server cannot report about
    itself: models, retrieval config incl. tau), else empty. A default file keeps every run's
    header complete, so the gate never sees the retrieval config "in one arm only"."""
    if path is None:
        default = dataset_dir / ARM_FILE if dataset_dir is not None else None
        if default is None or not default.exists():
            return ArmConfig()
        path = str(default)
    try:
        return ArmConfig.model_validate(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise CliError(f"no arm config at {path}") from None
    except (ValidationError, tomllib.TOMLDecodeError) as exc:
        raise CliError(f"{path}: not a valid arm config ({_first_error(exc)})") from None


def load_judge_config(path: str | None) -> dict[str, Any] | None:
    """The `[judge]` table handed to judge.make_judge (it owns the keys and their rules)."""
    if path is None:
        return None
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, tomllib.TOMLDecodeError) as exc:
        raise CliError(f"{path}: cannot read the judge config ({type(exc).__name__})") from None
    table = data.get("judge", data)
    if not isinstance(table, dict):
        raise CliError(f"{path}: [judge] must be a table")
    return table


def dataset_matches(run: RunFile, datasets: Datasets) -> bool:
    """A summary over an edited dataset would grade old answers against new facts."""
    try:
        current = dataset_for(run.header.eval_kind, datasets)
    except CliError:
        return False
    return digest(current.model_dump(mode="json")) == run.header.dataset.digest


# ------------------------------------------------------------------ keys and clients
def eval_key_file(
    dataset: str,
    explicit: str | None,
    env: Mapping[str, str],
    key_dir: Path = DEFAULT_EVAL_KEY_DIR,
) -> Path:
    """The eval key FOR A DATASET (its own directory and env var: never the owner's daily key).

    The demo key lives apart, in `<eval dir>/demo/`, so it is never "the only key" a private run
    picks up (RLS would keep that safe, but every private item would come back stale). `demo`
    uses that directory only (`$RAGFORGE_EVAL_KEY_FILE` names the private key); `private`
    refuses the demo key even when passed explicitly."""
    demo_dir = key_dir / DEMO_KEY_DIR
    if dataset == "demo":
        return resolve_key_file(explicit, {}, demo_dir)
    scoped = (
        {"RAGFORGE_KEY_FILE": env["RAGFORGE_EVAL_KEY_FILE"]}
        if env.get("RAGFORGE_EVAL_KEY_FILE")
        else {}
    )
    path = resolve_key_file(explicit, scoped, key_dir)
    resolved = path.expanduser().resolve()
    if resolved.name == DEMO_KEY_NAME or resolved.parent == demo_dir.expanduser().resolve():
        raise CliError("that is the demo tenant's key: a private run needs the private eval key")
    return path


def eval_key(dataset: str, explicit: str | None, env: Mapping[str, str]) -> str:
    return read_key(eval_key_file(dataset, explicit, env))


def owner_key(explicit: str | None, env: Mapping[str, str]) -> str:
    return read_key(resolve_key_file(explicit, dict(env), DEFAULT_KEY_DIR))


class ClientFactory(Protocol):
    def __call__(self, url: str, key: str) -> AbstractAsyncContextManager[Any]: ...


@asynccontextmanager
async def sdk_client(url: str, key: str) -> AsyncIterator[RagforgeClient]:
    async with RagforgeClient(url, key, timeout=120.0) as client:
        yield client


# ------------------------------------------------------------------ run
@dataclass(frozen=True)
class Env:
    """Everything a command touches outside its arguments (tests replace each part)."""

    components: Components = field(default_factory=Components)
    connect: ClientFactory = sdk_client
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    out: TextIO = sys.stdout
    err: TextIO = sys.stderr
    prompt: Callable[[str], str] = input
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def _ctx(
    args: argparse.Namespace,
    env: Env,
    client: EvalApi,
    paths: Paths,
    target: Target,
    *,
    kind: EvalKind,
    resume: bool,
) -> RunContext:
    """`args` carries --arm, --rpm, --judge-rpm and --arm-config (`run` and `demo` both do)."""
    arm = load_arm_config(args.arm_config, paths.dataset_dir)
    store = env.components.label_store
    resume_path = find_resumable(paths.runs_dir, kind, args.arm) if resume else None
    if resume and resume_path is None:
        raise CliError(f"no unfinished {kind.value} run of arm {args.arm} in {paths.runs_dir}")
    return RunContext(
        client=client,
        collection_ids=tuple(target.collection_ids),
        arm=args.arm,
        runs_dir=paths.runs_dir,
        dataset_dir=paths.dataset_dir,
        snapshot_collection_id=target.snapshot_collection_id,
        retrieval=arm.retrieval,
        chat_model=arm.chat_model,
        verifier_model=arm.verifier_model,
        prompt_versions=arm.prompt_versions,
        policy=RetryPolicy(),
        pacer=Pacer(rpm=args.rpm),
        judge_pacer=Pacer(rpm=args.judge_rpm),
        summarize=env.components.summarize,
        labels=store(paths.dataset_dir / LABELS_FILE),
        resume=resume_path,
        log=lambda line: print(line, file=env.err),
        now=env.now,
    )


JUDGE_PROMPTS = {EvalKind.generator: "judge_qa@1", EvalKind.interview: "judge_question@1"}


@asynccontextmanager
async def maybe_judge(
    args: argparse.Namespace, env: Env, kind: EvalKind, paths: Paths
) -> AsyncIterator[Judge | None]:
    """The judge for a kind that has one, when a judge config is given (E-G without a judge
    still records every deterministic grade). `<kind>_prompt` in the table overrides the prompt."""
    table = load_judge_config(getattr(args, "judge_config", None))
    if table is None or kind not in JUDGE_PROMPTS:
        yield None
        return
    prompts = {k: table.pop(f"{k.value}_prompt", v) for k, v in JUDGE_PROMPTS.items()}
    cache = paths.dataset_dir / JUDGE_CACHE
    async with env.components.open_judge(table, str(prompts[kind]), cache) as judge:
        yield judge


def _target(paths: Paths, explicit: Sequence[str] | None) -> Target:
    """`--collection` is for the demo tenant only: a private run always targets the snapshot
    (ADR-0017 principle 3), and a bare collection id carries none."""
    if explicit:
        if paths.dataset != "demo":
            raise CliError("--collection is for --dataset demo: a private run uses the snapshot")
        return Target(collection_ids=[UUID(c) for c in explicit])
    return load_model(paths.target, Target, "run target (run `demo` or `snapshot create`)")


async def run_kind(
    kind: EvalKind, ctx: RunContext, datasets: Datasets, judge: Judge | None, per_seed: int = 2
) -> list[Path]:
    """One kind; `application` is followed by its separate latency pass (never resumed with
    it: a resume continues exactly the run it names)."""
    data = dataset_for(kind, datasets)
    if kind is EvalKind.retrieval:
        return [await run_retrieval(ctx, cast(QuestionSet, data))]
    if kind is EvalKind.generator:
        return [await run_generator(ctx, cast(QuestionSet, data), judge)]
    if kind is EvalKind.verifier:
        return [await run_verifier(ctx, cast(VerifierSet, data))]
    if kind is EvalKind.interview:
        interview = cast(InterviewSet, data)
        return [await run_interview(ctx, interview, judge, questions_per_seed=per_seed)]
    if kind is EvalKind.latency:
        return [await run_latency(ctx, cast(QuestionSet, data))]
    first = await run_application(ctx, cast(QuestionSet, data))
    if ctx.resume is not None:
        return [first]
    return [first, await run_latency(replace(ctx, resume=None), cast(QuestionSet, data))]


async def cmd_run(args: argparse.Namespace, env: Env) -> int:
    paths = Paths(Path(args.root), args.dataset)
    datasets = load_datasets(paths.dataset_dir)
    kind = EvalKind(args.kind)
    dataset_for(kind, datasets)  # fail before any key or network use
    target = _target(paths, args.collection)
    paths.runs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = eval_key(args.dataset, args.key_file, env.environ)
    async with maybe_judge(args, env, kind, paths) as judge, env.connect(args.url, key) as client:
        ctx = _ctx(args, env, client, paths, target, kind=kind, resume=args.resume)
        written = await run_kind(kind, ctx, datasets, judge, args.questions_per_seed)
    for path in written:
        print(path, file=env.out)
    return 0


# ------------------------------------------------------------------ summarize
def cmd_summarize(args: argparse.Namespace, env: Env) -> int:
    run_path = Path(args.run)
    run = _read(run_path)
    if run.footer is None:
        raise CliError(f"{run_path.name} has not finished: resume it first")
    dataset_dir = Path(args.root) / run.header.dataset.name
    datasets = load_datasets(dataset_dir)
    if not dataset_matches(run, datasets):
        raise CliError("the dataset changed since this run (digest differs): re-run it instead")
    out = summarize_run(
        run_path,
        datasets,
        env.components.summarize,
        dataset_dir=dataset_dir,
        labels=env.components.label_store(dataset_dir / LABELS_FILE),
    )
    print(out, file=env.out)
    return 0


def _read(path: Path) -> RunFile:
    try:
        return read_run(path)
    except FileNotFoundError:
        raise CliError(f"no run file at {path}") from None
    except (RunFileError, ValueError) as exc:
        raise CliError(f"{path.name}: not a v2 run file ({type(exc).__name__})") from None


# ------------------------------------------------------------------ label (blind)
@dataclass(frozen=True)
class LabelItem:
    row: RunRow
    question: Question
    answer: str
    notes: tuple[str, ...]
    item_digest: str


SKIP, QUIT = "skip", "quit"


def labelable(run: RunFile, questions: QuestionSet) -> list[tuple[RunRow, Question]]:
    """Answered, clean, current rows; NEVER heldout2 (ADR-0017 D5: no held-out rows in the
    calibration sample). The newest row per item wins (a resumed run may repeat an item)."""
    by_id = {q.id: q for q in questions.questions}
    latest: dict[str, RunRow] = {r.item_id: r for r in run.rows}
    out: list[tuple[RunRow, Question]] = []
    for row in latest.values():
        q = by_id.get(row.item_id)
        output = row.output
        if q is None or q.split is Split.heldout2 or row.error is not None or row.stale:
            continue
        if isinstance(output, AnswerOutput) and not output.abstained and row.shown:
            out.append((row, q))
    return out


async def fetch_label_items(
    client: EvalApi, rows: Sequence[tuple[RunRow, Question]], err: TextIO
) -> list[LabelItem]:
    """The shown notes by id, digest-checked: the owner labels exactly what the model saw, and
    the item digest is the judge's (`item_digest_of`), so calibration can pair the two."""
    items: list[LabelItem] = []
    for row, q in rows:
        assert isinstance(row.output, AnswerOutput)
        response = await client.eval_chunks([UUID(s.chunk_id) for s in row.shown])
        notes = checked_texts(row.shown, response)
        if notes is None:
            print(f"{row.item_id}: skipped (its notes changed since the run)", file=err)
            continue
        answer = row.output.answer
        facts = [f.fact for f in q.facts]
        item_digest = datasets_mod.item_digest_of(q.question, answer, notes, facts)
        items.append(LabelItem(row, q, answer, notes, item_digest))
    return items


def show_item(item: LabelItem, index: int, total: int, out: TextIO) -> None:
    """Question, notes, answer: nothing else (no judge score or reason, no server verdict, no
    arm or model), so the owner's label is independent of what is being calibrated."""
    print(f"\n=== {index}/{total} · {item.row.item_id}", file=out)
    print(f"Question: {clean(item.question.question)}", file=out)
    for n, note in enumerate(item.notes, 1):
        print(f"--- [S{n}]\n{clean(note)}", file=out)
    print(f"--- Answer\n{clean(item.answer)}", file=out)


def ask_score(prompt: Callable[[str], str], criterion: Criterion) -> int | str:
    while True:
        choice = prompt(f"{criterion.value} 0/1/2 (s skip, q quit): ").strip().lower()
        if choice in ("0", "1", "2"):
            return int(choice)
        if choice[:1] == "s":
            return SKIP
        if choice[:1] == "q":
            return QUIT


def ask_yes_no(prompt: Callable[[str], str], question: str) -> bool | str:
    while True:
        choice = prompt(f"{question} y/n (s skip, q quit): ").strip().lower()
        if choice[:1] in ("y", "n"):
            return choice[:1] == "y"
        if choice[:1] == "s":
            return SKIP
        if choice[:1] == "q":
            return QUIT


def label_one(
    item: LabelItem, kind: LabelKind, prompt: Callable[[str], str]
) -> dict[str, Any] | str:
    """The fields of one label, or SKIP / QUIT. All criteria at once (D5: one record per item)."""
    if kind is LabelKind.adjudication:
        # every item in this queue carries the verified badge (the queue's premise, so asking
        # about it reveals nothing); the judge's score and reason stay hidden
        confirmed = ask_yes_no(prompt, "the notes do NOT support this answer (false-verified)?")
        if isinstance(confirmed, str):
            return confirmed
        return {"confirmed": confirmed}
    if kind is LabelKind.criteria:
        scores: dict[Criterion, int] = {}
        for criterion in E_G_CRITERIA:
            choice = ask_score(prompt, criterion)
            if isinstance(choice, str):
                return choice
            scores[criterion] = choice
        return {"scores": scores}
    correct = ask_yes_no(prompt, "correct?")
    if isinstance(correct, str):
        return correct
    invented = ask_yes_no(prompt, "anything invented?")
    if isinstance(invented, str):
        return invented
    return {"correct": correct, "invented": invented}


def calibration_queue(
    run: RunFile,
    questions: QuestionSet,
    items: Sequence[LabelItem],
    labelled: frozenset[str],
    *,
    n: int,
    seed: int,
) -> list[LabelItem]:
    """The criteria queue is the stratified calibration sample (ADR-0017 §4: by tier and
    verdict, never heldout2, never an item already labelled), not every answered row."""
    by_row = {id(i.row): i.item_digest for i in items}
    candidates = datasets_mod.calibration_candidates([run], questions, lambda r: by_row.get(id(r)))
    picked = datasets_mod.calibration_sample(candidates, n=n, seed=seed, exclude_digests=labelled)
    order = {c.item_digest: k for k, c in enumerate(picked)}
    return sorted((i for i in items if i.item_digest in order), key=lambda i: order[i.item_digest])


def label_queue(
    items: Sequence[LabelItem],
    kind: LabelKind,
    store: LabelStore,
    batch_id: str,
    *,
    prompt: Callable[[str], str],
    out: TextIO,
    now: Callable[[], datetime],
) -> int:
    """Label every item not yet labelled (by digest and kind); returns how many were written."""
    done = {lb.item_digest for lb in store.latest() if lb.kind is kind}
    queue = [i for i in items if i.item_digest not in done]
    written = 0
    for index, item in enumerate(queue, 1):
        show_item(item, index, len(queue), out)
        fields = label_one(item, kind, prompt)
        if fields == QUIT:
            break
        if fields == SKIP or isinstance(fields, str):
            continue
        store.append(
            Label(
                item_digest=item.item_digest,
                item_id=item.row.item_id,
                kind=kind,
                batch_id=batch_id,
                labelled_at=now(),
                **fields,
            )
        )
        written += 1
    print(f"\n{written} label(s) written, {len(queue) - written} left", file=out)
    return written


FLAG_BELOW = 2  # a faithfulness score under "yes" flags a verified answer for adjudication


def flagged_verified(
    rows: Sequence[tuple[RunRow, Question]], judge_results: Sequence[JudgeResult]
) -> list[tuple[RunRow, Question]]:
    """ADR-0017 §3 E-G: the verified answers the judge screen flags (faithfulness < 2), which
    the owner adjudicates (confirmed false-verified k/n). An invalid judge reply (score None)
    flags nothing: it is counted by metrics, never imputed."""
    by_key = {r.cache_key: r for r in judge_results}

    def flagged(row: RunRow) -> bool:
        refs = (by_key.get(ref) for ref in row.judge_refs)
        return any(
            r is not None
            and r.criterion is Criterion.faithfulness
            and r.score is not None
            and r.score < FLAG_BELOW
            for r in refs
        )

    return [
        (row, q)
        for row, q in rows
        if isinstance(row.output, AnswerOutput) and row.output.status == "verified" and flagged(row)
    ]


def label_kind(run: RunFile, adjudicate: bool) -> LabelKind:
    if adjudicate:
        if run.header.eval_kind is not EvalKind.generator or run.header.judge is None:
            raise CliError("--adjudicate works on a judged generator run")
        return LabelKind.adjudication
    kinds = {EvalKind.generator: LabelKind.criteria, EvalKind.application: LabelKind.answer}
    kind = kinds.get(run.header.eval_kind)
    if kind is None:
        raise CliError("label works on generator (criteria) and application (answer) runs")
    return kind


async def cmd_label(args: argparse.Namespace, env: Env) -> int:
    run = _read(Path(args.run))
    kind = label_kind(run, args.adjudicate)
    dataset_dir = Path(args.root) / run.header.dataset.name
    questions = load_datasets(dataset_dir).questions
    if questions is None:
        raise CliError(f"{dataset_dir}: no {QUESTIONS_FILE}")
    store = env.components.label_store(dataset_dir / LABELS_FILE)
    batch = args.batch or f"b-{env.now():%Y%m%d}"
    key = eval_key(run.header.dataset.name, args.key_file, env.environ)
    rows = labelable(run, questions)
    if kind is LabelKind.adjudication:
        rows = flagged_verified(rows, load_judge_results(dataset_dir / JUDGE_CACHE))
    async with env.connect(args.url, key) as client:
        items = await fetch_label_items(client, rows, env.err)
    if kind is LabelKind.criteria:
        labelled = frozenset(lb.item_digest for lb in store.latest() if lb.kind is kind)
        items = calibration_queue(run, questions, items, labelled, n=args.sample, seed=args.seed)
    label_queue(items, kind, store, batch, prompt=env.prompt, out=env.out, now=env.now)
    return 0


# ------------------------------------------------------------------ baseline and gate
def cmd_baseline(args: argparse.Namespace, env: Env) -> int:
    run_path = Path(args.run)
    run = _read(run_path)
    if run.footer is None:
        raise CliError("a baseline must be a finished run")
    paths = Paths(Path(args.root), run.header.dataset.name)
    if run_path.resolve().parent != paths.runs_dir.resolve():
        raise CliError(f"a baseline must live in {paths.runs_dir}")
    baseline = Baseline(
        run=run_path.name,
        run_id=run.header.run_id,
        eval_kind=run.header.eval_kind,
        surface=run.header.surface,
        header_digest=run.header_digest,
        note=args.note,
        set_at=env.now(),
    )
    save_model(paths.baseline(run.header.surface, run.header.eval_kind), baseline)
    print(
        f"baseline {run.header.surface.value}/{run.header.eval_kind.value}: {run_path.name}",
        file=env.out,
    )
    return 0


def newest_finished(
    runs_dir: Path, kind: EvalKind, exclude: str, arm: str | None = None
) -> RunFile | None:
    """The newest finished run of `kind` (by `started_at`, never by filename: names sort by arm
    first), other than the baseline. Without `arm`, the finished runs must all be of ONE arm:
    picking silently among arms could gate a stale experiment instead of the run just made."""
    runs = [
        r
        for r in iter_runs(runs_dir)
        if r.header.eval_kind is kind and r.footer is not None and r.header.run_id != exclude
    ]
    if arm is not None:
        runs = [r for r in runs if r.header.arm == arm]
    elif len({r.header.arm for r in runs}) > 1:
        arms = ", ".join(sorted({r.header.arm for r in runs}))
        raise CliError(f"{kind.value}: runs of several arms ({arms}): choose one with --arm")
    return max(runs, key=lambda r: (r.header.started_at, r.path.name), default=None)


def merge_verdicts(
    surface: Surface, verdicts: Sequence[GateVerdict], override: str | None
) -> GateVerdict:
    """One verdict per surface (the locked `gate-<surface>.json`): every kind's checks, the
    worst outcome."""
    outcome = max((v.outcome for v in verdicts), key=_SEVERITY.__getitem__)
    return GateVerdict(
        surface=surface,
        baseline_run=",".join(v.baseline_run for v in verdicts),
        candidate_run=",".join(v.candidate_run for v in verdicts),
        checks=[c for v in verdicts for c in v.checks],
        outcome=outcome,
        override=override,
    )


def store_override(path: Path, reason: str) -> None:
    """Record the override in the candidate's header (shown on the Eval page). `override` is
    outside the header digest, so the run's configuration identity is unchanged."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = read_run(path).header.model_copy(update={"override": reason})
    write_private(path, "\n".join([header.model_dump_json(), *lines[1:]]) + "\n")


@dataclass(frozen=True)
class GateInputs:
    paths: Paths
    datasets: Datasets
    explicit: Sequence[RunFile]
    aa: Sequence[RunFile]
    arm: str | None
    err: TextIO


def gate_kind(
    inputs: GateInputs, evaluate: GateEvaluate, surface: Surface, kind: EvalKind
) -> tuple[GateVerdict, RunFile] | None:
    """One kind's verdict against its baseline (None: no baseline, or no candidate yet)."""
    paths = inputs.paths
    if not paths.baseline(surface, kind).exists():
        return None
    base_ref = load_model(paths.baseline(surface, kind), Baseline, "baseline")
    baseline = _read(paths.runs_dir / base_ref.run)
    candidate = next(
        (c for c in inputs.explicit if c.header.eval_kind is kind), None
    ) or newest_finished(paths.runs_dir, kind, baseline.header.run_id, inputs.arm)
    if candidate is None:
        print(f"{kind.value}: no candidate run", file=inputs.err)
        return None
    if candidate.header.started_at < baseline.header.started_at:
        raise CliError(
            f"{kind.value}: the candidate {candidate.header.run_id} is older than the "
            f"baseline {baseline.header.run_id}: a gate compares a NEW run against the baseline"
        )
    aa = [r for r in inputs.aa if r.header.eval_kind is kind]
    return evaluate(baseline, candidate, aa, inputs.datasets), candidate


def _build(run: RunFile) -> tuple[object, ...]:
    """What makes two runs the same build: the models and every surface's prompt set."""
    h = run.header
    return (h.chat_model, h.verifier_model, tuple(sorted(h.prompt_versions.items())))


def same_build(candidates: Sequence[RunFile], floors: Sequence[RunFile]) -> None:
    """Final review P1-2: a gate must judge ONE build. Every candidate of the surface has the
    same models and prompt sets, and a hard-floor run (E-V on a chat change) ran the SAME
    verifier model and verifier prompt as the chat candidates, never evidence from before the
    change. Refuses (exit 2) otherwise."""
    builds = {_build(r) for r in candidates}
    if len(builds) > 1:
        names = ", ".join(sorted(r.header.run_id for r in candidates))
        raise CliError(f"the candidates are of different builds (models/prompts): {names}")
    for floor in floors:
        for cand in candidates:
            same_model = floor.header.verifier_model == cand.header.verifier_model
            same_prompt = floor.header.prompt_versions.get(
                "verifier"
            ) == cand.header.prompt_versions.get("verifier")
            if not (same_model and same_prompt):
                raise CliError(
                    f"the {floor.header.eval_kind.value} hard-floor run {floor.header.run_id} "
                    f"ran another verifier than {cand.header.run_id}: re-run it on this change"
                )


def hard_floor(
    inputs: GateInputs, evaluate: GateEvaluate, surface: Surface, kind: EvalKind
) -> tuple[GateVerdict, RunFile]:
    """A floor another surface's run must pass: missing evidence refuses the gate (exit 2),
    since a chat change that was never checked for false-verified cannot be called a pass."""
    found = gate_kind(inputs, evaluate, surface, kind)
    if found is None:
        raise CliError(
            f"the {kind.value} hard floor needs a {surface.value} baseline "
            f"(baseline-{surface.value}-{kind.value}.json) and a finished {kind.value} run: "
            f"re-run `run {kind.value}` on this change, then gate again"
        )
    return found


def cmd_gate(args: argparse.Namespace, env: Env) -> int:
    """Exit 0 only when `gate.all_pass` says so over every verdict, the hard floors included:
    an override excuses a block, never a not_evaluable (incomparable runs stay incomparable)."""
    surface = Surface(args.surface)
    paths = Paths(Path(args.root), args.dataset)
    evaluate = env.components.gate_evaluate
    inputs = GateInputs(
        paths=paths,
        datasets=load_datasets(paths.dataset_dir),
        explicit=[_read(Path(p)) for p in args.candidate or []],
        aa=[_read(Path(p)) for p in args.aa or []],
        arm=args.arm,
        err=env.err,
    )
    kinds = [EvalKind(k) for k in args.kind] if args.kind else list(SURFACE_KINDS[surface])
    found = [f for kind in kinds if (f := gate_kind(inputs, evaluate, surface, kind))]
    if not found:
        raise CliError(f"nothing to gate on {surface.value}: set a baseline first (baseline set)")
    # ADR-0017 §5: retrieval changes get the cheap retrieval gate; the E-V hard floor guards a
    # CHAT or VERIFIER change, so a retrieval-only gate does not need it
    retrieval_only = bool(args.kind) and set(kinds) == {EvalKind.retrieval}  # explicit only
    floor_specs = () if retrieval_only else HARD_FLOORS.get(surface, ())
    floors = [hard_floor(inputs, evaluate, s, k) for s, k in floor_specs]
    same_build([run for _, run in found], [run for _, run in floors])
    found += floors
    # the override is THIS invocation's: one recorded on an earlier gate must not pass a new one
    verdicts = [v.model_copy(update={"override": args.override}) for v, _ in found]
    verdict = merge_verdicts(surface, verdicts, args.override)
    save_model(paths.gate(surface), verdict)
    if args.override:
        for _, run in found:
            store_override(run.path, args.override)
    _print_verdict(verdict, env.out)
    return 0 if gate_mod.all_pass(verdicts) else 1


def _print_verdict(verdict: GateVerdict, out: TextIO) -> None:
    for check in verdict.checks:
        extra = f" p={check.p_value:.3g}" if check.p_value is not None else ""
        mde = f" mde={check.mde:.3g}" if check.mde is not None else ""
        print(
            f"{check.outcome.value:>13}  {check.name} ({check.method}, n={check.n}{extra}{mde})",
            file=out,
        )
    suffix = f" (override: {verdict.override})" if verdict.override else ""
    print(f"gate {verdict.surface.value}: {verdict.outcome.value}{suffix}", file=out)


# ------------------------------------------------------------------ snapshot
EVAL_KEY_NAME = "eval.key"


def cmd_seal(args: argparse.Namespace, env: Env) -> int:
    """`seal <file>`: stamp a held-out question file's digest into it; later edits fail runs."""
    digest_ = datasets_mod.seal_questions(Path(args.file))
    print(f"sealed {args.file}: {digest_}", file=env.out)
    return 0


def cmd_d3_build(args: argparse.Namespace, env: Env) -> int:
    """`d3-build`: the private planted-error set from v2 answer runs plus the owner's
    correct-labels (never from verdicts), note text fetched via /v1/eval/chunks (eval key)."""
    paths = Paths(Path(args.root), "private")
    runs = [read_run(Path(r)) for r in args.runs]
    questions = datasets_mod.load_questions(paths.dataset_dir / QUESTIONS_FILE)
    labels = datasets_mod.LabelStore(paths.dataset_dir / LABELS_FILE).latest()
    # the EVAL key: /v1/eval/chunks serves only the eval tenant (final review P1-3)
    key = eval_key("private", args.key_file, env.environ)
    built = asyncio.run(_d3(args.url, key, runs, questions, labels, env))
    out = Path(args.output) if args.output else paths.dataset_dir / VERIFIER_FILE
    if out.exists():
        raise CliError(f"{out} exists: pass --output to write elsewhere (never overwritten)")
    write_private(out, built.verifier.model_dump_json(indent=2) + "\n")
    planted = sum(1 for i in built.verifier.items if i.mutation is not None)
    clean = len(built.verifier.items) - planted
    print(f"{out}: {clean} clean + {planted} planted; skipped {dict(built.skipped)}", file=env.out)
    print("review every negation_flip item before the first verifier run", file=env.out)
    return 0


async def _d3(
    url: str,
    key: str,
    runs: Sequence[RunFile],
    questions: QuestionSet,
    labels: Sequence[Label],
    env: Env,
) -> datasets_mod.D3Build:
    async with env.connect(url, key) as client:
        return await datasets_mod.build_d3(runs, questions, labels, client)


def cmd_tenant(args: argparse.Namespace, env: Env) -> int:
    """`tenant create`: the owner's own key (key-manager: approvals:write) provisions the paired
    eval tenant; its eval key is
    written to a new 0600 file and never printed (hidden-input rule)."""
    key = owner_key(args.key_file, env.environ)
    default = DEFAULT_EVAL_KEY_DIR / EVAL_KEY_NAME
    path = Path(args.eval_key_file or default).expanduser()
    if path.exists():
        raise CliError(f"{path} already exists: the eval tenant was provisioned before")
    tenant_id = asyncio.run(_provision(args.url, key, path, env))
    print(f"eval tenant {tenant_id}: eval key written to {path}", file=env.out)
    return 0


async def _provision(url: str, key: str, path: Path, env: Env) -> str:
    async with env.connect(url, key) as client:
        provisioned = await client.eval_tenant_create()
    save_key(path, provisioned.api_key.secret)
    return str(provisioned.tenant.id)


async def cmd_snapshot(args: argparse.Namespace, env: Env) -> int:
    paths = Paths(Path(args.root), "private")
    key = owner_key(args.key_file, env.environ)
    async with env.connect(args.url, key) as client:
        if args.action == "create":
            snap: EvalSnapshot = await client.eval_snapshot(
                EvalSnapshotRequest(source_collection_id=args.source_collection)
            )
            save_model(
                paths.target,
                Target(
                    collection_ids=[snap.collection_id],
                    snapshot_collection_id=snap.collection_id,
                    tenant_id=snap.eval_tenant_id,
                ),
            )
        else:
            target = load_model(paths.target, Target, "run target (run `snapshot create`)")
            if target.snapshot_collection_id is None:
                raise CliError("the private target names no snapshot")
            if args.action == "reembed":
                snap = await client.eval_reembed(
                    target.snapshot_collection_id, EvalReembedRequest(space_id=args.space)
                )
            else:
                snap = await client.eval_snapshot_status(target.snapshot_collection_id)
    print(
        f"snapshot {snap.collection_id} job {snap.job_id}: "
        f"content {snap.content_digest or 'pending'}, index {snap.index_digest or 'pending'}",
        file=env.out,
    )
    return 0


# ------------------------------------------------------------------ demo
class DemoAdmin(Protocol):
    async def provision_tenant(self, name: str, owner_display_name: str) -> TenantProvisioned: ...


class DemoOwner(Protocol):
    """The demo tenant's OWNER key (key-manager): the same calls the owner makes on real notes."""

    async def create_collection(self, name: str) -> Collection: ...

    async def eval_tenant_create(self) -> TenantProvisioned: ...

    async def eval_snapshot(self, request: EvalSnapshotRequest) -> EvalSnapshot: ...

    async def eval_snapshot_status(self, collection_id: UUID) -> EvalSnapshot: ...

    def job_events(self, job_id: UUID) -> AsyncIterator[JobEvent]: ...

    async def get_job(self, job_id: UUID) -> Job: ...


class Uploader(Protocol):
    async def __call__(self, collection_id: UUID, name: str, content: bytes, /) -> UUID: ...


def http_uploader(
    url: str, key: str, transport: httpx.AsyncBaseTransport | None = None
) -> Uploader:
    """POST /v1/collections/{id}/files (multipart, one `file` part). The SDK has no upload
    method yet (handoff); httpx builds the multipart body, nothing is hand-rolled."""

    async def upload(collection_id: UUID, name: str, content: bytes) -> UUID:
        headers = {"authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(base_url=url, headers=headers, transport=transport) as http:
            response = await http.post(
                f"/v1/collections/{collection_id}/files",
                files={"file": (name, content, "text/markdown")},
            )
        if not response.is_success:
            raise CliError(f"upload of {name} failed: HTTP {response.status_code}")
        return Accepted.model_validate_json(response.content).job_id

    return upload


async def wait_for_job(owner: DemoOwner, job_id: UUID) -> Job:
    async for _ in owner.job_events(job_id):
        pass  # the stream ends when the job finishes
    return await owner.get_job(job_id)


def save_key(path: Path, secret: str) -> None:
    """The eval key, 0600 in its own directory; never printed, never in a URL or a log."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(path, secret + "\n")


@dataclass(frozen=True)
class DemoDeps:
    owner_key: str  # the demo tenant's owner key (bootstrapped by `make eval-demo`)
    owner: Callable[[str], AbstractAsyncContextManager[Any]]
    uploader: Callable[[str], Uploader]
    key_path: Path  # the demo EVAL key (written once, reused on later runs)
    poll_s: float = 1.0


def read_pages(directory: Path) -> list[tuple[str, bytes]]:
    """The D0 pages, by filename (the file page key, principle 6), in a stable order."""
    pages = [(p.name, p.read_bytes()) for p in sorted(directory.glob("*.md"))]
    if not pages:
        raise CliError(f"no pages in {directory}")
    return pages


async def setup_demo(
    deps: DemoDeps, pages: Sequence[tuple[str, bytes]], paths: Paths, env: Env
) -> Target:
    """The demo walks the OWNER's real path end to end (integrator fix: the eval surface serves
    only an eval tenant): upload the D0 pages into a fresh collection of the demo tenant, pair
    the demo tenant with its eval tenant (once; the eval key is written 0600 and reused), then
    snapshot the collection into it and wait for its digests. Runs then use the eval key on
    the snapshot, exactly as the private runs do."""
    stamp = f"{env.now():%Y%m%d%H%M%S}"
    async with deps.owner(deps.owner_key) as owner:
        collection: Collection = await owner.create_collection(f"acme-demo-{stamp}")
        upload = deps.uploader(deps.owner_key)
        for name, content in pages:
            job_id = await upload(collection.id, name, content)
            job = await wait_for_job(owner, job_id)
            if job.status != "done":
                raise CliError(f"ingesting {name} ended {job.status}")
            print(f"ingested {name}", file=env.err)
        if not deps.key_path.exists():
            provisioned: TenantProvisioned = await owner.eval_tenant_create()
            save_key(deps.key_path, provisioned.api_key.secret)
        snap: EvalSnapshot = await owner.eval_snapshot(
            EvalSnapshotRequest(source_collection_id=collection.id)
        )
        copied = await wait_for_job(owner, snap.job_id)
        if copied.status != "done":
            raise CliError(f"the demo snapshot copy ended {copied.status}")
        for _ in range(60):  # the digests are recorded when the copy commits
            snap = await owner.eval_snapshot_status(snap.collection_id)
            if snap.content_digest is not None:
                break
            await asyncio.sleep(deps.poll_s)
        else:
            raise CliError("the demo snapshot never recorded its digests")
    print(f"snapshot {snap.collection_id}: content {snap.content_digest}", file=env.err)
    target = Target(
        collection_ids=[snap.collection_id],
        snapshot_collection_id=snap.collection_id,
        tenant_id=snap.eval_tenant_id,
    )
    save_model(paths.target, target)
    return target


def demo_owner_file(args: argparse.Namespace) -> Path:
    """The demo tenant's owner key file (sync: path checks stay out of the event loop)."""
    default = DEFAULT_EVAL_KEY_DIR / DEMO_KEY_DIR / DEMO_OWNER_KEY_NAME
    owner_file = Path(args.owner_key_file or default).expanduser()
    if not owner_file.exists():
        raise CliError(f"no demo owner key at {owner_file}: run `make eval-demo` (it bootstraps)")
    return owner_file


async def cmd_demo(args: argparse.Namespace, env: Env, owner_file: Path) -> int:
    paths = Paths(Path(args.root), "demo")
    datasets = load_datasets(paths.dataset_dir)
    key_path = owner_file.parent / DEMO_KEY_NAME
    deps = DemoDeps(
        owner_key=read_key(owner_file),
        owner=lambda key: env.connect(args.url, key),
        uploader=lambda key: http_uploader(args.url, key),
        key_path=key_path,
    )
    target = await setup_demo(deps, read_pages(paths.dataset_dir / "pages"), paths, env)
    async with env.connect(args.url, read_key(key_path)) as client:
        for kind in DEMO_KINDS:
            ctx = _ctx(args, env, client, paths, target, kind=kind, resume=False)
            async with maybe_judge(args, env, kind, paths) as judge:
                for path in await run_kind(kind, ctx, datasets, judge, args.questions_per_seed):
                    print(path, file=env.out)
    return 0


# ------------------------------------------------------------------ parser
def arm_label(value: str) -> str:
    """The arm is part of a filename: no separators (the header's own pattern)."""
    if not value or len(value) > 64 or any(not (c.isalnum() or c in "._:-") for c in value):
        raise argparse.ArgumentTypeError("use 1-64 of A-Z a-z 0-9 . _ : - (no slashes or spaces)")
    return value


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(DEFAULT_ROOT), help="the evals directory")
    common.add_argument("--url", default=DEFAULT_URL)
    common.add_argument("--key-file", help="the API key file (never pass the key itself)")

    parser = argparse.ArgumentParser(prog="ragforge-eval2", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="one eval run")
    run.add_argument("kind", choices=KINDS)
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--arm", type=arm_label, required=True)
    run.add_argument("--resume", action="store_true", help="continue the newest unfinished run")
    run.add_argument(
        "--rpm",
        type=positive_float,
        default=None,
        help="SDK requests/minute: about provider RPM / provider calls per item (chat: >= 2)",
    )
    run.add_argument("--arm-config", help="TOML: chat_model, verifier_model, [retrieval]...")
    run.add_argument("--judge-config", help="TOML [judge] table for judge.make_judge")
    run.add_argument("--judge-rpm", type=positive_float, default=None, help="judge calls/minute")
    run.add_argument("--collection", action="append", help="override runs/target.json")
    run.add_argument("--questions-per-seed", type=int, default=2)

    summarize = sub.add_parser("summarize", parents=[common], help="(re)write a run's summary")
    summarize.add_argument("run")

    label = sub.add_parser("label", parents=[common], help="the blind label queue")
    label.add_argument("run")
    label.add_argument("--batch", help="label batch id (calibration recomputes per batch)")
    label.add_argument("--sample", type=int, default=datasets_mod.CALIBRATION_N, help="items")
    label.add_argument("--seed", type=int, default=0, help="the sample's seed (reproducible)")
    label.add_argument(
        "--adjudicate",
        action="store_true",
        help="adjudicate the verified answers the judge flagged (generator runs)",
    )

    gate = sub.add_parser("gate", parents=[common], help="the regression gate")
    gate.add_argument("--surface", choices=[s.value for s in Surface], required=True)
    gate.add_argument("--dataset", choices=DATASETS, default="private")
    gate.add_argument("--candidate", action="append", help="candidate run (default: newest)")
    gate.add_argument("--arm", type=arm_label, help="the candidate arm (needed with several)")
    gate.add_argument("--aa", action="append", help="an A/A run (sets the MDE)")
    gate.add_argument("--override", help="the reason to accept a failing gate (recorded)")
    gate.add_argument(
        "--kind",
        action="append",
        choices=[k.value for k in EvalKind],
        help="gate only these kinds (e.g. retrieval for a retrieval change: no E-V floor)",
    )

    baseline = sub.add_parser("baseline", parents=[common], help="baselines")
    baseline_sub = baseline.add_subparsers(dest="action", required=True)
    baseline_set = baseline_sub.add_parser("set", parents=[common], help="move a baseline")
    baseline_set.add_argument("run")
    baseline_set.add_argument("--note", required=True)

    seal = sub.add_parser("seal", parents=[common], help="seal a held-out question file")
    seal.add_argument("file")
    d3 = sub.add_parser("d3-build", parents=[common], help="build the private planted-error set")
    d3.add_argument("runs", nargs="+", help="v2 generator/application run files")
    d3.add_argument("--output", help="where to write (default: private/verifier.json)")
    tenant = sub.add_parser("tenant", parents=[common], help="the paired eval tenant")
    tenant_sub = tenant.add_subparsers(dest="action", required=True)
    tenant_create = tenant_sub.add_parser(
        "create", parents=[common], help="provision it (your owner key); writes the eval key"
    )
    tenant_create.add_argument("--eval-key-file", help="where to write the eval key (0600)")
    snapshot = sub.add_parser("snapshot", parents=[common], help="the frozen eval snapshot")
    snapshot_sub = snapshot.add_subparsers(dest="action", required=True)
    create = snapshot_sub.add_parser("create", parents=[common], help="copy a live collection")
    create.add_argument("--source-collection", required=True, type=UUID)
    snapshot_sub.add_parser("status", parents=[common], help="the snapshot's digests")
    reembed = snapshot_sub.add_parser("reembed", parents=[common], help="re-embed the snapshot")
    reembed.add_argument("--space", required=True, type=int)

    demo = sub.add_parser("demo", parents=[common], help="D0 end to end (live stack)")
    demo.add_argument(
        "--owner-key-file", help="the demo tenant's owner key (default: written by make eval-demo)"
    )
    demo.add_argument("--arm", type=arm_label, default="demo")
    demo.add_argument("--rpm", type=positive_float, default=None)
    demo.add_argument("--arm-config")
    demo.add_argument("--judge-config")
    demo.add_argument("--judge-rpm", type=positive_float, default=None)
    demo.add_argument("--questions-per-seed", type=int, default=2)

    results = sub.add_parser(
        "results", help="the generated tables of docs/RESULTS.md (from evals/results.toml)"
    )
    results_mode = results.add_mutually_exclusive_group(required=True)
    results_mode.add_argument(
        "--check", action="store_true", help="re-render and exit 1 on any difference"
    )
    results_mode.add_argument(
        "--write", action="store_true", help="re-render the blocks and the header sidecars"
    )
    results.add_argument("--doc", default=str(export_mod.DEFAULT_DOC))
    results.add_argument("--manifest", default=str(export_mod.DEFAULT_MANIFEST))
    results.add_argument("--root", default=str(DEFAULT_ROOT), help="the evals directory")
    results.add_argument(
        "--table", action="append", help="only this table id (repeatable; default: all)"
    )
    return parser


def dispatch(args: argparse.Namespace, env: Env) -> int:
    match args.command:
        case "run":
            return asyncio.run(cmd_run(args, env))
        case "summarize":
            return cmd_summarize(args, env)
        case "label":
            return asyncio.run(cmd_label(args, env))
        case "gate":
            return cmd_gate(args, env)
        case "baseline":
            return cmd_baseline(args, env)
        case "tenant":
            return cmd_tenant(args, env)
        case "seal":
            return cmd_seal(args, env)
        case "d3-build":
            return cmd_d3_build(args, env)
        case "snapshot":
            return asyncio.run(cmd_snapshot(args, env))
        case "demo":
            return asyncio.run(cmd_demo(args, env, demo_owner_file(args)))
        case "results":
            return cmd_results(args, env)
    raise CliError(f"unknown command {args.command}")


def cmd_results(args: argparse.Namespace, env: Env) -> int:
    """Phase 5 R2-3: offline (no server, no key); exit 1 when --check finds a difference."""
    try:
        return export_mod.results(
            mode="write" if args.write else "check",
            doc_path=Path(args.doc),
            manifest_path=Path(args.manifest),
            root=Path(args.root),
            only=args.table,
            out=env.out,
            err=env.err,
        )
    except export_mod.ExportError as exc:
        raise CliError(str(exc)) from None


def main(argv: Sequence[str] | None = None, env: Env | None = None) -> int:
    """Exit codes: 0 ok, 1 the gate failed (no override), 2 a clean error. Errors print the
    failure, never a key, a URL with a key, or note text."""
    args = build_parser().parse_args(argv)
    env = env or Env()
    try:
        return dispatch(args, env)
    except (CliError, RunAborted, KeyFileError, RunFileError) as exc:
        print(f"error: {exc}", file=env.err)
    except ApiError as exc:
        print(f"error: the server said {clean(str(exc))}", file=env.err)
    except httpx.HTTPError as exc:
        print(f"error: cannot reach the server ({type(exc).__name__})", file=env.err)
    return 2


def run() -> None:
    """Console-script entry point (`ragforge-eval2`)."""
    sys.exit(main())
