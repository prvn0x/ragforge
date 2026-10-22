"""GET /v1/jobs/{id} and the job progress SSE stream (LISTEN rf_job_progress + poll fallback)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import Fakes, TenantWorld, auth, issue_key
from ragforge_contracts.sse import decode_job_event
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.jobfeed import PgJobFeed, PollingJobFeed
from ragforge_core.apps.api.routes import jobs as jobs_route

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


def _job_events(body: str) -> list[tuple[str, dict[str, object]]]:
    out = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        event_line, data_line = block.splitlines()
        assert event_line == "event: job"
        event = decode_job_event(data_line.removeprefix("data: "))
        out.append((str(event.status), dict(event.progress)))
    return out


async def test_get_job_reads_live_and_history_tables(http: httpx.AsyncClient, world: World) -> None:
    w = world()
    live = (await http.get(f"/v1/jobs/{w.job}", headers=auth(w.key))).json()
    assert (live["id"], live["status"], live["kind"]) == (str(w.job), "queued", "sync_connector")
    done = (await http.get(f"/v1/jobs/{w.history_job}", headers=auth(w.key))).json()
    assert (done["status"], done["kind"]) == ("done", "sync_page")


async def _wait_listening(app: FastAPI) -> PgJobFeed:
    feed = app.state.job_feed
    assert isinstance(feed, PgJobFeed)
    async with asyncio.timeout(10):
        await feed.listening.wait()
    return feed


async def test_notify_wakes_the_stream_without_polling(
    app: FastAPI, http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    """Poll interval 60 s: the stream can only finish in time if NOTIFY wakes it. Foreign and
    malformed payloads are ignored (they carry no data and only trigger a re-read)."""
    app.dependency_overrides[deps.get_job_stream_config] = lambda: deps.JobStreamConfig(
        poll_interval_s=60, keepalive_s=120
    )
    w, other = world(), world()
    feed = await _wait_listening(app)
    request = asyncio.create_task(http.get(f"/v1/jobs/{w.job}/events", headers=auth(w.key)))
    async with asyncio.timeout(10):
        while w.job not in feed._subscribers:  # noqa: ASYNC110 - test waits on private state
            await asyncio.sleep(0.01)

    superuser.execute("NOTIFY rf_job_progress, 'not-a-uuid'")
    superuser.execute("SELECT pg_notify('rf_job_progress', %s)", (str(other.job),))
    superuser.execute("UPDATE rf.jobs SET progress = '{\"pages\": 3}' WHERE id = %s", (w.job,))
    superuser.execute("SELECT pg_notify('rf_job_progress', %s)", (str(w.job),))
    await asyncio.sleep(0.3)  # let the stream emit the progress before the next change
    superuser.execute(
        "UPDATE rf.jobs SET status = 'done', finished_at = now() WHERE id = %s", (w.job,)
    )
    superuser.execute("SELECT pg_notify('rf_job_progress', %s)", (str(w.job),))
    async with asyncio.timeout(10):
        response = await request
    assert response.status_code == 200
    expected = [("queued", {}), ("queued", {"pages": 3}), ("done", {"pages": 3})]
    # The subscription is registered just BEFORE the stream's first read, so that read may
    # already see the progress update (then the initial `queued, {}` state is never observed).
    # Either way every state is emitted once, in order, and only NOTIFY can end it in time.
    assert _job_events(response.text) in (expected, expected[1:])
    assert w.job not in feed._subscribers  # unsubscribed when the stream ended


async def test_poll_fallback_without_any_notify(
    http: httpx.AsyncClient, superuser: Conn, world: World
) -> None:
    w = world()
    request = asyncio.create_task(http.get(f"/v1/jobs/{w.job}/events", headers=auth(w.key)))
    await asyncio.sleep(0.3)  # several 50 ms polls with no change: no duplicate events
    superuser.execute(
        "UPDATE rf.jobs SET status = 'dead', last_error = 'boom', finished_at = now() "
        "WHERE id = %s",
        (w.job,),
    )
    async with asyncio.timeout(10):
        response = await request
    # Keepalive pacing is covered deterministically by test_stream_keepalive_counts_quiet_polls;
    # asserting it here would depend on wall-clock timing.
    assert _job_events(response.text) == [("queued", {}), ("dead", {})]
    assert '"last_error":"boom"' in response.text


def _row(status: str, *, finished: bool = False) -> dict[str, object]:
    return {"id": UUID(int=7), "status": status, "progress": None, "last_error": None,
            "finished_at": datetime.now(UTC) if finished else None}  # fmt: skip


def test_stream_keepalive_counts_quiet_polls() -> None:
    """No feed signal ever fires (PollingJobFeed), so every wait is a poll timeout: with
    keepalive = 3 polls, 7 quiet polls yield exactly 2 keepalives, then the terminal event."""
    rows = [_row("queued")] * 8 + [_row("done", finished=True)]
    loads = iter(rows)

    async def load() -> Any:
        return next(loads)

    async def collect() -> list[bytes]:
        config = deps.JobStreamConfig(poll_interval_s=0.001, keepalive_s=0.0025)
        return [c async for c in jobs_route._stream(UUID(int=7), load, PollingJobFeed(), config)]

    chunks = asyncio.run(collect())
    assert [c for c in chunks if c.startswith(b":")] == [jobs_route.KEEPALIVE] * 2
    assert [decode_job_event(c.decode().split("data: ")[1]).status for c in chunks
            if not c.startswith(b":")] == ["queued", "done"]  # fmt: skip
    assert next(loads, None) is None  # stopped right after the terminal row


async def test_revoked_key_ends_an_open_stream(
    app: FastAPI, http: httpx.AsyncClient, superuser: Conn, world: World, fakes: Fakes
) -> None:
    """Authentication happens once per request: the stream re-checks its key (every poll here,
    reauth_s=0) and ends after revocation although the job never finishes."""
    app.dependency_overrides[deps.get_job_stream_config] = lambda: deps.JobStreamConfig(
        poll_interval_s=0.02, keepalive_s=60, reauth_s=0
    )
    w = world()
    viewer = issue_key(superuser, w.tenant, w.user, ["read"])
    feed = await _wait_listening(app)
    request = asyncio.create_task(http.get(f"/v1/jobs/{w.job}/events", headers=auth(viewer)))
    async with asyncio.timeout(10):  # the stream is open (authenticated, subscribed)
        while w.job not in feed._subscribers:  # noqa: ASYNC110 - test waits on private state
            await asyncio.sleep(0.01)
        # ...and its first read (key still valid -> the "queued" snapshot) has committed. Revoking
        # earlier ends the stream with no event at all: correct, but not what this test checks.
        baseline = fakes.txm.completed
        while fakes.txm.completed <= baseline:  # noqa: ASYNC110 - waits on a test counter
            await asyncio.sleep(0.005)
    superuser.execute(
        "UPDATE rf.api_keys SET revoked_at = now() WHERE prefix = %s", (viewer.split("_")[1],)
    )
    async with asyncio.timeout(10):
        response = await request
    assert response.status_code == 200
    assert _job_events(response.text) == [("queued", {})]


async def test_finished_job_streams_once_and_ends(http: httpx.AsyncClient, world: World) -> None:
    w = world()
    r = await http.get(f"/v1/jobs/{w.history_job}/events", headers=auth(w.key))
    assert _job_events(r.text) == [("done", {})]


def test_feed_dispatch_ignores_garbage_and_cleans_up() -> None:
    async def scenario() -> None:
        feed = PgJobFeed("postgresql://unused")
        job = uuid4()
        async with feed.subscribe(job) as signal:
            feed.dispatch("garbage")
            feed.dispatch(str(uuid4()))
            assert not signal.is_set()
            feed.dispatch(f" {job} ")
            assert signal.is_set()
        assert job not in feed._subscribers

    asyncio.run(scenario())


async def test_unknown_job_is_404(http: httpx.AsyncClient, world: World) -> None:
    w = world()
    for url in (f"/v1/jobs/{uuid4()}", f"/v1/jobs/{UUID(int=5)}/events"):
        assert (await http.get(url, headers=auth(w.key))).status_code == 404
