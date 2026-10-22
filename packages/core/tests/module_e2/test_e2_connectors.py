"""Connectors: the token is sealed, bound to its row, and never appears in any response or log."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
import structlog
from fastapi import FastAPI

from module_e2.e2_support import Fakes, TenantWorld, auth, random_token
from ragforge_core.adapters.crypto.envelope import SecretDecryptionError
from ragforge_core.apps.api.routes.sources import connector_associated_data
from ragforge_core.domain.models import JobPriority, SealedSecret

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


def _sealed(superuser: Conn, connector: UUID) -> SealedSecret:
    row = superuser.execute(
        "SELECT secret_ciphertext, wrapped_dek FROM rf.connectors WHERE id = %s", (connector,)
    ).fetchone()
    assert row is not None
    return SealedSecret(ciphertext=cast(bytes, row[0]), wrapped_dek=cast(bytes, row[1]))


async def test_token_never_echoed_or_logged_and_bound_to_its_row(
    app: FastAPI,
    http: httpx.AsyncClient,
    superuser: Conn,
    world: World,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    w = world()
    token = random_token()
    caplog.set_level(logging.DEBUG)
    bodies: list[str] = []
    with structlog.testing.capture_logs() as logs:
        create = {"kind": "notion", "target_collection_id": str(w.collection),
                  "display_name": "Team wiki", "token": token}  # fmt: skip
        r = await http.post("/v1/connectors", json=create, headers=auth(w.key))
        assert r.status_code == 201, r.text
        connector = UUID(r.json()["id"])
        bodies.append(r.text)
        assert "token" not in r.json()
        for url in ("/v1/connectors", f"/v1/connectors/{connector}", "/v1/audit"):
            r = await http.get(url, headers=auth(w.key))
            assert r.status_code == 200
            bodies.append(r.text)
        # Error paths must not reflect the input either.
        for bad in ({**create, "unexpected": 1}, {**create, "kind": "gdrive"},
                    {**create, "target_collection_id": str(uuid4())}):  # fmt: skip
            r = await http.post("/v1/connectors", json=bad, headers=auth(w.key))
            assert r.status_code in (404, 422), r.text
            bodies.append(r.text)
    captured = capsys.readouterr()
    haystacks = [*bodies, captured.out, captured.err, caplog.text, repr(logs)]
    assert all(token not in h for h in haystacks)
    assert all(token not in str(row) for row in superuser.execute("SELECT * FROM rf.audit_log"))

    # Stored sealed (not plaintext) and bound to (tenant, connector) by the associated data.
    sealed = _sealed(superuser, connector)
    assert token.encode() not in sealed.ciphertext
    box = app.state.container.secrets
    good = connector_associated_data(w.tenant, connector)
    assert box.open(sealed, associated_data=good) == token.encode()
    assert good == f"connector:{w.tenant}:{connector}".encode()
    for wrong in (connector_associated_data(w.tenant, uuid4()),
                  connector_associated_data(uuid4(), connector)):  # fmt: skip
        with pytest.raises(SecretDecryptionError):
            box.open(sealed, associated_data=wrong)


async def test_sync_enqueues_and_delete_destroys_the_secret(
    http: httpx.AsyncClient, fakes: Fakes, superuser: Conn, world: World
) -> None:
    w = world()
    create = {"kind": "notion", "target_collection_id": str(w.collection),
              "display_name": "d", "token": random_token()}  # fmt: skip
    connector = (await http.post("/v1/connectors", json=create, headers=auth(w.key))).json()["id"]

    r = await http.post(f"/v1/connectors/{connector}/sync", headers=auth(w.key))
    assert r.status_code == 202
    assert UUID(r.json()["job_id"])
    sync = fakes.jobs.enqueued[-1]
    # no in-flight-cap slot for the listing job (handoff A: connector_id is for sync_page only)
    assert (sync.kind, sync.priority, sync.connector_id) == (
        "sync_connector",
        JobPriority.interactive,
        None,
    )
    assert sync.payload == {"connector_id": connector}
    assert sync.dedupe_key == f"sync_connector:{connector}"

    r = await http.delete(f"/v1/connectors/{connector}", headers=auth(w.key))
    assert r.status_code == 202
    assert fakes.jobs.enqueued[-1].kind == "delete_connector"
    assert fakes.jobs.enqueued[-1].connector_id is None
    assert superuser.execute(
        "SELECT status, secret_ciphertext, wrapped_dek FROM rf.connectors WHERE id = %s",
        (connector,),
    ).fetchone() == ("deleting", None, None)
    r = await http.post(f"/v1/connectors/{connector}/sync", headers=auth(w.key))
    assert r.status_code == 409
    actions = [
        e["action"] for e in (await http.get("/v1/audit", headers=auth(w.key))).json()["items"]
    ]
    assert actions[:3] == ["connector.delete", "connector.sync", "connector.create"]
