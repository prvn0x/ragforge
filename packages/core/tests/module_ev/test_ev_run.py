"""`ragforge-eval run` end to end through the real SDK over respx: the run file records arm notes,
observed provider/model, citation provenance and memory peaks; a dropped connection or a stalled
stream costs one row, not the run; a footer survives failure and cancellation; a truncated last
line still loads; and progress output carries no per-question held-out detail."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import respx
from ev_support import GOOD_EOS, citation, write_questions

from ragforge_client import evalrun
from ragforge_client.client import ApiError
from ragforge_client.evalrun import (
    Outcome,
    ResourceSample,
    ask_one,
    build_parser,
    load_run,
    run_eval,
)
from ragforge_contracts.sse import (
    ChatEvent,
    CitationEvent,
    DoneEvent,
    TokenEvent,
    Usage,
    encode_sse,
)

URL = "http://ragforge.test"
KEY = "rf_test_key_never_printed"


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "eval.key"
    path.write_text(KEY)
    path.chmod(0o600)
    return path


class FixedSampler:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> ResourceSample:
        self.calls += 1
        return ResourceSample(swap_used_mb=4096.0, ollama_rss_mb=3072.0)


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> FixedSampler:
    fixed = FixedSampler()
    monkeypatch.setattr(evalrun, "ResourceSampler", lambda: fixed)
    return fixed


def _sse_body() -> bytes:
    events: list[ChatEvent] = [
        TokenEvent(text=GOOD_EOS),
        CitationEvent(citation=citation("Kafka Delivery", inferred=True)),
        DoneEvent(
            conversation_id=uuid4(),
            usage=[
                Usage(provider="ollama", model="qwen3:4b", prompt_tokens=9, completion_tokens=4)
            ],
            prompt_version="v3",
        ),
    ]
    return b"".join(encode_sse(e) for e in events)


def _collections() -> dict[str, Any]:
    created = datetime.now(UTC).isoformat()
    return {
        "items": [{"id": str(uuid4()), "name": "n", "active_space_id": 1, "created_at": created}]
    }


def _args(tmp_path: Path, key_file: Path, *extra: str) -> Any:
    return build_parser().parse_args(
        [
            "--questions",
            str(write_questions(tmp_path)),
            "run",
            "--label",
            "A2-qwen4b",
            "--url",
            URL,
            "--key-file",
            str(key_file),
            "--out-dir",
            str(tmp_path / "runs"),
            "--sample-interval",
            "0.01",
            *extra,
        ]
    )


@respx.mock
async def test_run_records_arm_provenance_and_resources(
    tmp_path: Path, key_file: Path, sampler: FixedSampler, capsys: pytest.CaptureFixture[str]
) -> None:
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    chat = respx.post(f"{URL}/v1/chat").mock(
        return_value=httpx.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )
    )
    dest = await run_eval(_args(tmp_path, key_file, "--split", "all", "--notes", "rounds=0"))
    run = load_run(dest)
    assert run.header["notes"] == "rounds=0"
    assert chat.call_count == 4
    assert chat.calls[0].request.headers["authorization"] == f"Bearer {KEY}"
    eos = run.outcomes[0]
    assert (eos.inferred_titles, eos.providers, eos.models) == (
        ["Kafka Delivery"],
        ["ollama"],
        ["qwen3:4b"],
    )
    assert run.footer is not None
    assert run.footer["complete"] is True
    assert run.footer["observed"]["models"] == ["qwen3:4b"]
    assert run.footer["observed"]["prompt_versions"] == ["v3"]
    assert run.footer["resources"]["swap_used_mb"] == 4096.0
    assert run.footer["resources"]["ollama_rss_mb"] == 3072.0
    assert sampler.calls >= 1
    assert run.footer["resources"]["swap_baseline_mb"] == 4096.0
    assert run.footer["resources"]["swap_delta_mb"] == 0.0
    printed = capsys.readouterr().out
    # Held-out questions run last as one block: one line before, one aggregate line after, and
    # no per-question line (its print time alone would leak that question's latency).
    assert printed.splitlines()[-2:] == [
        "asking 1 held-out question(s); progress hidden",
        "[ 4/4] 1 held-out question(s) done",
    ]
    assert [o.id for o in run.outcomes][-1] == "secret-heldout-raft"
    assert "secret-heldout-raft" not in printed
    assert KEY not in printed + dest.read_text()


@respx.mock
async def test_setup_failure_leaves_no_run_file(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """Review round 4: a 0-row run file from a failed setup polluted `compare runs/*.jsonl`
    with a junk 'no: incomplete 0/N' arm. Setup now happens before the file exists."""
    respx.get(f"{URL}/v1/collections").mock(
        return_value=httpx.Response(503, json={"title": "Unavailable", "status": 503})
    )
    with pytest.raises(ApiError):
        await run_eval(_args(tmp_path, key_file))
    assert not (tmp_path / "runs").exists() or list((tmp_path / "runs").iterdir()) == []
    assert sampler.calls == 0


@respx.mock
async def test_empty_tenant_and_server_down_leave_no_run_file(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    route = respx.get(f"{URL}/v1/collections")
    route.mock(return_value=httpx.Response(200, json={"items": []}))
    with pytest.raises(evalrun.EvalSetupError):
        await run_eval(_args(tmp_path, key_file))
    route.mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await run_eval(_args(tmp_path, key_file))
    assert not (tmp_path / "runs").exists() or list((tmp_path / "runs").iterdir()) == []


@respx.mock
async def test_question_failure_mid_run_still_writes_a_footer(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """Once the run file exists, the footer is still written when the loop dies."""
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    respx.post(f"{URL}/v1/chat").mock(return_value=_ok_response())

    async def boom(*_: Any, **__: Any) -> Outcome:
        raise RuntimeError("bug in ask_one")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(evalrun, "ask_one", boom)
        with pytest.raises(RuntimeError):
            await run_eval(_args(tmp_path, key_file))
    (dest,) = (tmp_path / "runs").iterdir()
    run = load_run(dest)
    assert run.outcomes == []
    assert run.footer is not None
    assert run.footer["complete"] is False


@respx.mock
async def test_sampling_can_be_disabled(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    respx.post(f"{URL}/v1/chat").mock(
        return_value=httpx.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )
    )
    args = _args(tmp_path, key_file)
    args.sample_interval = 0.0
    dest = await run_eval(args)
    footer = json.loads(dest.read_text().splitlines()[-1])["footer"]
    assert sampler.calls == 0
    assert footer["resources"] == {
        "samples": 0,
        "failures": 0,
        "swap_used_mb": None,
        "ollama_rss_mb": None,
        "swap_baseline_mb": None,
        "swap_delta_mb": None,
    }


@respx.mock
async def test_tenant_without_collections_fails_fast(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json={"items": []}))
    chat = respx.post(f"{URL}/v1/chat")
    with pytest.raises(evalrun.EvalSetupError, match="no collections"):
        await run_eval(_args(tmp_path, key_file))
    assert chat.call_count == 0


def test_cli_reports_setup_errors_without_a_traceback(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    missing = tmp_path / "missing.key"
    with pytest.raises(SystemExit, match="ragforge-eval: cannot read key file"):
        evalrun.run(
            ["--questions", str(questions), "run", "--label", "x", "--key-file", str(missing)]
        )


def _ok_response() -> httpx.Response:
    return httpx.Response(200, content=_sse_body(), headers={"content-type": "text/event-stream"})


class StalledStream(httpx.AsyncByteStream):
    """Sends the start of an answer, then never sends anything again (a wedged model runner)."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        yield encode_sse(TokenEvent(text="partial "))
        await asyncio.Event().wait()


@respx.mock
async def test_transport_error_costs_one_row_not_the_run(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    chat = respx.post(f"{URL}/v1/chat").mock(
        side_effect=[httpx.RemoteProtocolError("peer closed"), _ok_response(), _ok_response()]
    )
    dest = await run_eval(_args(tmp_path, key_file))  # dev split: 3 questions
    run = load_run(dest)
    assert chat.call_count == 3
    assert [o.error for o in run.outcomes[1:]] == [None, None]
    assert run.outcomes[0].error == "transport RemoteProtocolError: peer closed"
    assert run.footer is not None
    assert run.footer["complete"] is True  # every question was asked; one failed


@respx.mock
async def test_stalled_stream_times_out_and_the_run_continues(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    stalled = httpx.Response(
        200, stream=StalledStream(), headers={"content-type": "text/event-stream"}
    )
    respx.post(f"{URL}/v1/chat").mock(side_effect=[stalled, _ok_response(), _ok_response()])
    dest = await run_eval(_args(tmp_path, key_file, "--question-timeout", "0.05"))
    first, *rest = load_run(dest).outcomes
    assert first.error == "timeout after 0.05s"
    assert first.draft == "partial "  # what streamed before the stall is kept
    assert [o.error for o in rest] == [None, None]


async def test_ask_one_without_timeout_waits_for_the_stream() -> None:
    """--question-timeout 0 disables the deadline (a slow arm must be measurable, not cut off)."""

    class Slow:
        async def chat(self, message: str, collection_ids: Any) -> AsyncIterator[Any]:
            await asyncio.sleep(0.05)
            yield TokenEvent(text="done")

    question = evalrun.Question(id="q", split="dev", question="?", answerable=False)
    out = await ask_one(Slow(), [], question, timeout_s=0)
    assert (out.error, out.final) == (None, "done")


@respx.mock
async def test_cancelled_run_writes_an_incomplete_footer(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """Ctrl-C mid-question: the run stops (cancellation is not swallowed as a row error) and the
    footer still lands with complete=false, so `compare` can never call it a winner."""
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    stream = StalledStream()
    respx.post(f"{URL}/v1/chat").mock(
        side_effect=[
            _ok_response(),
            httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}),
        ]
    )
    task = asyncio.create_task(run_eval(_args(tmp_path, key_file)))
    await asyncio.wait_for(stream.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (dest,) = (tmp_path / "runs").iterdir()
    run = load_run(dest)
    assert [o.id for o in run.outcomes] == ["dev-eos"]
    assert run.footer is not None
    assert run.footer["complete"] is False


def test_truncated_last_line_still_loads(tmp_path: Path) -> None:
    """SIGKILL mid-write leaves half a JSON line and no footer: finished rows stay scoreable."""
    good = json.dumps(asdict(Outcome(id="dev-eos", final="x")))
    path = tmp_path / "killed.jsonl"
    path.write_text(f'{{"header": {{"label": "a", "split": "dev"}}}}\n{good}\n{{"id": "dev-l')
    warn = io.StringIO()
    run = load_run(path, warn=warn)
    assert [o.id for o in run.outcomes] == ["dev-eos"]
    assert run.footer is None
    assert "skipped truncated last line 3" in warn.getvalue()


def test_corruption_before_the_last_line_still_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"header": {"label": "a", "split": "dev"}}\n{"id": "dev-l\n{"id": "x"}\n')
    with pytest.raises(json.JSONDecodeError):
        load_run(path, warn=io.StringIO())


@pytest.mark.parametrize("label", ["../escape", "a/b", "has space", "pipe|cell", "", "x" * 65])
def test_run_label_cannot_escape_the_out_dir_or_break_a_table(
    label: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--label", label])
    assert "argument --label: use 1-64 of" in capsys.readouterr().err


@respx.mock
async def test_timeout_is_structured_and_recorded_in_the_header(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """The bar needs the deadline as data: `timed_out` per row, `question_timeout_s` per run."""
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    stalled = httpx.Response(
        200, stream=StalledStream(), headers={"content-type": "text/event-stream"}
    )
    respx.post(f"{URL}/v1/chat").mock(side_effect=[stalled, _ok_response(), _ok_response()])
    dest = await run_eval(_args(tmp_path, key_file, "--question-timeout", "0.05"))
    run = load_run(dest)
    assert run.header["question_timeout_s"] == 0.05
    assert [o.timed_out for o in run.outcomes] == [True, False, False]


def test_disabled_timeout_is_recorded_as_none_not_zero() -> None:
    assert evalrun.effective_timeout(0.0) is None
    assert evalrun.effective_timeout(-1.0) is None
    assert evalrun.effective_timeout(300.0) == 300.0


@respx.mock
def test_cli_server_down_at_setup_is_one_line_not_a_traceback(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """Review P2: httpx.ConnectError from list_collections (API not up) escaped as a traceback."""
    respx.get(f"{URL}/v1/collections").mock(side_effect=httpx.ConnectError("refused"))
    argv = [
        "--questions",
        str(write_questions(tmp_path)),
        "run",
        "--label",
        "down",
        "--url",
        URL,
        "--key-file",
        str(key_file),
        "--out-dir",
        str(tmp_path / "runs"),
    ]
    with pytest.raises(SystemExit) as exc:
        evalrun.run(argv)
    message = str(exc.value)
    assert message == f"ragforge-eval: cannot reach {URL}: ConnectError"
    assert KEY not in message


@respx.mock
def test_cli_run_end_to_end_never_prints_heldout_rows(
    tmp_path: Path, key_file: Path, sampler: FixedSampler, capsys: pytest.CaptureFixture[str]
) -> None:
    """run() prints progress AND a post-run `score -v`: neither may show held-out detail."""
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    respx.post(f"{URL}/v1/chat").mock(side_effect=lambda _request: _ok_response())
    argv = [
        "--questions",
        str(write_questions(tmp_path)),
        "run",
        "--label",
        "e2e",
        "--split",
        "all",
        "--url",
        URL,
        "--key-file",
        str(key_file),
        "--out-dir",
        str(tmp_path / "runs"),
        "--sample-interval",
        "0",
    ]
    evalrun.run(argv)
    printed = capsys.readouterr().out
    assert "dev-eos" in printed  # the per-question report did print
    assert "secret-heldout-raft" not in printed
    assert "SECRET heldout question text" not in printed
    assert "1 held-out row(s) hidden" in printed
    assert KEY not in printed
    evalrun.run([*argv[:-4], "--out-dir", str(tmp_path / "runs2"), "--reveal"])
    assert "secret-heldout-raft" in capsys.readouterr().out  # --reveal is honoured end to end


# ------------------------------------------------------------------ review round 3
@respx.mock
async def test_run_file_is_owner_only_and_records_the_asked_question(
    tmp_path: Path, key_file: Path, sampler: FixedSampler
) -> None:
    """Run files quote the owner's private notes: 0600 like the label sidecar, never 0644. Each
    row fingerprints the question it asked, so a later edit is flagged instead of re-graded."""
    respx.get(f"{URL}/v1/collections").mock(return_value=httpx.Response(200, json=_collections()))
    respx.post(f"{URL}/v1/chat").mock(return_value=_ok_response())
    dest = await run_eval(_args(tmp_path, key_file, "--split", "all"))
    assert dest.stat().st_mode & 0o777 == 0o600
    questions = {q.id: q for q in evalrun.load_questions(tmp_path / "questions.toml")}
    for row in load_run(dest).outcomes:
        assert row.question_split == questions[row.id].split
        assert row.question_sha256 == evalrun.question_digest(questions[row.id])


def test_run_file_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    existing = tmp_path / "runs" / "same-second.jsonl"
    existing.parent.mkdir()
    existing.write_text("earlier run\n")
    with pytest.raises(evalrun.EvalSetupError, match="already exists"):
        evalrun.open_private(existing)
    assert existing.read_text() == "earlier run\n"


@respx.mock
def test_cli_setup_api_error_strips_terminal_escapes(tmp_path: Path, key_file: Path) -> None:
    respx.get(f"{URL}/v1/collections").mock(
        return_value=httpx.Response(503, json={"title": "Down\x1b[2J\x1b]0;pwn\x07", "status": 503})
    )
    argv = ["--questions", str(write_questions(tmp_path)), "run", "--label", "x", "--url", URL]
    argv += ["--key-file", str(key_file), "--out-dir", str(tmp_path / "runs")]
    argv += ["--sample-interval", "0"]
    with pytest.raises(SystemExit) as exc:
        evalrun.run(argv)
    message = str(exc.value.code)
    assert message.startswith("ragforge-eval: 503 Down")
    assert "\x1b" not in message
    assert "\x07" not in message


def test_cli_ctrl_c_is_one_line_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def interrupted(_: Any) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(evalrun, "run_eval", interrupted)
    with pytest.raises(SystemExit) as exc:
        evalrun.run(["--questions", str(write_questions(tmp_path)), "run", "--label", "x"])
    assert exc.value.code == 130
    assert capsys.readouterr().err.strip().splitlines() == [
        "ragforge-eval: interrupted; the partial run is recorded as incomplete"
    ]


async def test_second_interrupt_while_stopping_the_sampler_is_deferred() -> None:
    """A Ctrl-C landing while the footer waits for the sampler must not skip the footer: the
    interrupt is handed back for the caller to re-raise AFTER writing it."""
    waiting = asyncio.Event()

    async def wedged_sampler() -> None:
        waiting.set()
        await asyncio.Event().wait()

    sampler_task = asyncio.create_task(wedged_sampler())
    await waiting.wait()
    stopper = asyncio.create_task(evalrun._stop_sampler(asyncio.Event(), sampler_task))
    await asyncio.sleep(0)
    stopper.cancel()
    handed_back = await stopper
    assert isinstance(handed_back, asyncio.CancelledError)
    await asyncio.sleep(0)
    assert sampler_task.cancelled()
