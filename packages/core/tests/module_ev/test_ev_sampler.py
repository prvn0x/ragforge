"""Resource sampling: parses macOS/Linux sources, and degrades to "not observed" (None, never 0)
on any failure so a missing tool can never break or skew an eval run."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ragforge_client import evalrun
from ragforge_client.evalrun import (
    ResourcePeaks,
    ResourceSample,
    ResourceSampler,
    parse_meminfo_swap,
    parse_ollama_rss,
    parse_swapusage,
    run_command,
    sample_until,
)

SWAPUSAGE = "total = 4096.00M  used = 1536.25M  free = 2559.75M  (encrypted)\n"
PS = (
    "  2048 /opt/homebrew/opt/ollama/bin/ollama\n"
    "4194304 /opt/homebrew/opt/ollama/bin/ollama\n"  # the model runner (same binary)
    "999999 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
    "  1024 ollama_llama_server\n"
)
MEMINFO = "MemTotal: 16000000 kB\nSwapTotal:       2097152 kB\nSwapFree:        1048576 kB\n"


def test_parse_swapusage_handles_units_and_garbage() -> None:
    assert parse_swapusage(SWAPUSAGE) == 1536.25
    assert parse_swapusage("total = 8.00G  used = 1.50G  free = 6.50G") == 1536.0
    assert parse_swapusage("vm.swapusage: unknown") is None


def test_parse_ollama_rss_sums_server_and_runners_only() -> None:
    assert parse_ollama_rss(PS) == (2048 + 4194304 + 1024) / 1024
    assert parse_ollama_rss("  100 /usr/bin/python3\n") is None  # not running is not 0 MB
    assert parse_ollama_rss("  abc /bin/ollama\n") is None


def test_parse_meminfo_swap() -> None:
    assert parse_meminfo_swap(MEMINFO) == 1024.0
    assert parse_meminfo_swap("MemTotal: 1 kB\n") is None


def _runner(outputs: dict[str, str | None]) -> evalrun.CommandRunner:
    def run(argv: Sequence[str]) -> str | None:
        return outputs.get(argv[0])

    return run


def test_macos_sampler_reads_sysctl_and_ps() -> None:
    sampler = ResourceSampler(platform="darwin", runner=_runner({"sysctl": SWAPUSAGE, "ps": PS}))
    sample = sampler.sample()
    assert sample.swap_used_mb == 1536.25
    assert sample.ollama_rss_mb is not None
    assert sample.ollama_rss_mb > 4000


def test_linux_sampler_uses_proc_meminfo() -> None:
    sampler = ResourceSampler(
        platform="linux", runner=_runner({"ps": None}), read_file=lambda _: MEMINFO
    )
    assert sampler.sample() == ResourceSample(swap_used_mb=1024.0, ollama_rss_mb=None)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_sampler_degrades_when_every_source_fails(platform: str) -> None:
    sampler = ResourceSampler(platform=platform, runner=_runner({}), read_file=lambda _: None)
    assert sampler.sample() == ResourceSample(None, None)


def test_unsupported_platform_never_shells_out() -> None:
    def explode(argv: Sequence[str]) -> str | None:
        raise AssertionError(f"ran {argv[0]}")

    assert ResourceSampler(platform="win32", runner=explode).sample() == ResourceSample(None, None)


@pytest.mark.parametrize(
    "behaviour",
    [
        FileNotFoundError("sysctl"),
        PermissionError("denied"),
        subprocess.TimeoutExpired("ps", 5),
    ],
)
def test_run_command_swallows_subprocess_failures(
    monkeypatch: pytest.MonkeyPatch, behaviour: BaseException
) -> None:
    def fake_run(*_: Any, **__: Any) -> Any:
        raise behaviour

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_command(("sysctl", "-n", "vm.swapusage")) is None


def test_run_command_treats_nonzero_exit_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0  # a hung tool must not hang the eval
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(argv, 1, stdout="partial", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_command(("ps",)) is None


def test_read_text_or_none_on_missing_file(tmp_path: Path) -> None:
    assert evalrun.read_text_or_none(tmp_path / "nope") is None


def test_peaks_keep_maxima_and_ignore_missing_values() -> None:
    peaks = ResourcePeaks()
    for s in (ResourceSample(10.0, None), ResourceSample(None, 50.0), ResourceSample(5.0, 40.0)):
        peaks.add(s)
    assert (peaks.samples, peaks.swap_used_mb, peaks.ollama_rss_mb) == (3, 10.0, 50.0)


def _stopping_runner(
    readings: Sequence[str | Exception], stop: asyncio.Event
) -> evalrun.CommandRunner:
    """Serves sysctl readings in order and asks the loop to stop after the last one.

    The runner runs in a worker thread, so it signals the event loop thread-safely.
    """
    loop = asyncio.get_running_loop()
    remaining = list(readings)

    def run(argv: Sequence[str]) -> str | None:
        if argv[0] != "sysctl":
            return None
        item = remaining.pop(0)
        if not remaining:
            loop.call_soon_threadsafe(stop.set)
        if isinstance(item, Exception):
            raise item
        return item

    return run


async def test_sample_until_samples_then_stops_promptly() -> None:
    peaks, stop = ResourcePeaks(), asyncio.Event()
    runner = _stopping_runner(["used = 1M", "used = 7M", "used = 3M"], stop)
    sampler = ResourceSampler(platform="darwin", runner=runner)
    await asyncio.wait_for(sample_until(sampler, peaks, 0.01, stop), timeout=2.0)
    assert (peaks.samples, peaks.swap_used_mb) == (3, 7.0)


async def test_sample_until_survives_a_crashing_sample() -> None:
    peaks, stop = ResourcePeaks(), asyncio.Event()
    runner = _stopping_runner([RuntimeError("sampler bug"), "used = 9M"], stop)
    sampler = ResourceSampler(platform="darwin", runner=runner)
    await asyncio.wait_for(sample_until(sampler, peaks, 0.01, stop), timeout=2.0)
    assert (peaks.failures, peaks.samples, peaks.swap_used_mb) == (1, 1, 9.0)


async def test_sample_until_returns_quickly_once_stopped() -> None:
    """A long interval must not delay the end of a run: stop wakes the wait immediately."""
    peaks, stop = ResourcePeaks(), asyncio.Event()
    sampler = ResourceSampler(platform="win32")
    task = asyncio.create_task(sample_until(sampler, peaks, 3600.0, stop))
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert peaks.samples >= 1
