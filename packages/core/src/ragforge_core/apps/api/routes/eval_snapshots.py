"""/v1/eval/tenant and /v1/eval/snapshots: the paired eval tenant, the snapshot copy, its status
and its re-embed (ADR-0017 §1 snapshot row, r3 P0-C, r4). Built by EV-SNAP.

Authorization:
  - POST /v1/eval/tenant and POST /v1/eval/snapshots need the `admin` scope of the SOURCE
    tenant (the owner's tenant). The eval tenant is never named by the caller: it is the
    source's pair, found by `rf.paired_eval_tenant()`, and the database re-checks the pairing
    (`rf.begin_eval_snapshot`, then `rf.copy_collection_to_eval` in the job).
  - GET /v1/eval/snapshots/{id} and POST .../reembed accept the eval tenant's own key (`eval`
    scope; RLS confines it to the eval tenant) or the source's `admin` key, which acts in its
    paired eval tenant only (the pairing comes from the definer, never from the request).
  - The whole surface 404s unless RAGFORGE_EVAL_API is on (`require_eval_api`), before any
    authentication, so a server with the flag off does not even reveal the routes.

The copy itself runs as a job in the source tenant (progress on the existing job feed:
GET /v1/jobs/{job_id}); a re-embed runs as a job in the eval tenant.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Response
from sqlalchemy import RowMapping
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import (
    ApiKeyCreated,
    EvalReembedRequest,
    EvalSnapshot,
    EvalSnapshotRequest,
    Tenant,
    TenantProvisioned,
)
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.deps import AnyKey, EvalApiEnabled, Jobs, KeyManagerKey, TxManager
from ragforge_core.apps.api.sql import (
    audit,
    execute,
    fetch_one,
    insert_api_key,
    set_tenant,
    sqlstate,
)
from ragforge_core.domain.errors import Conflict, Forbidden, NotFound, ValidationFailed
from ragforge_core.domain.models import Scope
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.services.auth.keys import generate_key
from ragforge_core.services.auth.principal import AuthenticatedKey
from ragforge_core.services.auth.scopes import ClientKind, is_key_manager
from ragforge_core.services.eval_snapshot import (
    EVAL_KEY_NAME,
    EVAL_KEY_SCOPES,
    copy_job,
    reembed_job,
    refusal_for,
)
from ragforge_core.services.eval_snapshot.policy import EVAL_OWNER_DISPLAY_NAME

NO_STORE = "no-store"  # the provisioning response carries a key secret


async def require_eval_api(enabled: EvalApiEnabled) -> None:
    """RAGFORGE_EVAL_API, read through the eval surface's one flag dependency (fails closed).
    A router-level dependency, so it runs before authentication: off means a plain 404."""
    if not enabled:
        raise NotFound("Not Found")


router = APIRouter(prefix="/v1/eval", tags=["eval"], dependencies=[Depends(require_eval_api)])


@contextmanager
def _refusals() -> Iterator[None]:
    """The definers' refusals (migration 0012) as problem responses; anything else re-raises.
    Never the driver's message: it can include bound parameters."""
    try:
        yield
    except DBAPIError as exc:
        diag = getattr(exc.orig, "diag", None)
        refusal = refusal_for(
            sqlstate(exc),
            getattr(diag, "message_primary", None),
            constraint=getattr(diag, "constraint_name", None),
        )
        if refusal is None:
            raise
        raise refusal from None


# ----------------------------------------------------------------- the paired eval tenant
@router.post("/tenant", status_code=201)
async def create_eval_tenant(
    key: KeyManagerKey, txm: TxManager, response: Response
) -> TenantProvisioned:
    """Bootstrap (once per owner): the eval tenant paired with the caller's tenant, its user and
    its dedicated eval key (eval + chat + read), in one transaction. 409 if it exists.
    Tenant SELF-administration (like minting a key), so a key-manager key (`approvals:write`, the
    owner's own key, or `admin`) may call it: owners never hold `admin`, the platform-operator
    scope (integrator fix, found in the first live run)."""
    source = key.principal.tenant_id
    new_key = generate_key()
    async with txm.tenant(source) as tx:
        conn = connection(tx)
        with _refusals():
            created = await fetch_one(
                conn,
                "SELECT tenant_id, user_id FROM rf.provision_eval_tenant(:owner)",
                {"owner": EVAL_OWNER_DISPLAY_NAME},
            )
        assert created is not None
        eval_tenant: UUID = created["tenant_id"]
        await set_tenant(conn, eval_tenant)
        key_row = await insert_api_key(
            conn,
            tenant_id=eval_tenant,
            user_id=created["user_id"],
            name=EVAL_KEY_NAME,
            key=new_key,
            scopes=EVAL_KEY_SCOPES,
            client_kind=ClientKind.user,
        )
        tenant = await fetch_one(
            conn,
            "SELECT id, name, kind, created_at FROM rf.tenants WHERE id = :id",
            {"id": eval_tenant},
        )
        await audit(
            conn, key, "api_key.create", str(key_row["id"]),
            {"prefix": new_key.prefix, "scopes": sorted(EVAL_KEY_SCOPES)}, tenant_id=eval_tenant,
        )  # fmt: skip
        await set_tenant(conn, source)
        await audit(conn, key, "eval_tenant.provision", str(eval_tenant), {"kind": "eval"})
    assert tenant is not None
    response.headers["cache-control"] = NO_STORE
    return TenantProvisioned(
        tenant=Tenant.model_validate(dict(tenant)),
        user_id=created["user_id"],
        api_key=ApiKeyCreated.model_validate({**key_row, "secret": new_key.plaintext}),
    )


async def _paired_eval_tenant(conn: AsyncConnection) -> UUID | None:
    row = await fetch_one(conn, "SELECT rf.paired_eval_tenant() AS id")
    return None if row is None else row["id"]


# ----------------------------------------------------------------- snapshots
@router.post("/snapshots", status_code=202)
async def create_snapshot(
    body: EvalSnapshotRequest, key: KeyManagerKey, txm: TxManager, jobs: Jobs
) -> EvalSnapshot:
    """Enqueue the copy (in the SOURCE tenant, so the copy derives its source from the job's
    tenant) and create the pending snapshot in the eval tenant, all or nothing."""
    snapshot_id = uuid4()
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        target = await _paired_eval_tenant(conn)
        if target is None:
            raise Conflict("this tenant has no eval tenant yet: POST /v1/eval/tenant first")
        job_id = await jobs.enqueue(tx, copy_job(snapshot_id, body.source_collection_id, target))
        with _refusals():
            await execute(
                conn,
                "SELECT rf.begin_eval_snapshot(:src, :target, :snap, :job)",
                {"src": body.source_collection_id, "target": target, "snap": snapshot_id,
                 "job": job_id},
            )  # fmt: skip
        await audit(
            conn, key, "eval_snapshot.create", str(snapshot_id),
            {"source_collection_id": str(body.source_collection_id),
             "eval_tenant_id": str(target), "job_id": str(job_id)},
        )  # fmt: skip
    return EvalSnapshot(collection_id=snapshot_id, eval_tenant_id=target, job_id=job_id)


async def _snapshot_tenant(key: AuthenticatedKey, txm: TransactionManager) -> UUID:
    """Where the snapshot rows live for this caller: its own tenant for the eval key, the
    paired eval tenant for the source's key-manager key (the owner's own key, or admin: tenant
    self-administration, the same rule as creating the pair and the snapshot)."""
    if key.principal.has(Scope.eval.value):
        return key.principal.tenant_id
    if not is_key_manager(key.principal):
        raise Forbidden(
            "this API key lacks the `eval` scope (or key-manager rights on the source tenant)"
        )
    async with txm.tenant(key.principal.tenant_id) as tx:
        target = await _paired_eval_tenant(connection(tx))
    if target is None:
        raise NotFound("eval snapshot not found")
    return target


_SNAPSHOT_COLUMNS = "collection_id, tenant_id, job_id, content_digest, index_digest"


def _snapshot(row: RowMapping, **overrides: object) -> EvalSnapshot:
    return EvalSnapshot.model_validate(
        {
            "collection_id": row["collection_id"],
            "eval_tenant_id": row["tenant_id"],
            "job_id": row["job_id"],
            "content_digest": row["content_digest"],
            "index_digest": row["index_digest"],
            **overrides,
        }
    )


@router.get("/snapshots/{collection_id}")
async def get_snapshot(collection_id: UUID, key: AnyKey, txm: TxManager) -> EvalSnapshot:
    """Digests are None until the copy job finished. After an exclusion purged pages they are
    the digests of what remains (the gate then refuses to compare with older runs)."""
    tenant = await _snapshot_tenant(key, txm)
    async with txm.tenant(tenant) as tx:
        row = await fetch_one(
            connection(tx),
            f"SELECT {_SNAPSHOT_COLUMNS} FROM rf.eval_snapshots "  # noqa: S608
            "WHERE collection_id = :id",
            {"id": collection_id},
        )
    if row is None:
        raise NotFound("eval snapshot not found")
    return _snapshot(row)


@router.post("/snapshots/{collection_id}/reembed", status_code=202)
async def reembed_snapshot(
    collection_id: UUID, body: EvalReembedRequest, key: AnyKey, txm: TxManager, jobs: Jobs
) -> EvalSnapshot:
    """Re-embed the snapshot into `space_id` from its stored chunk text (job in the eval
    tenant). `index_digest` is None in this response: the snapshot keeps serving, and GET keeps
    returning, the old space and digest until the job flips both at once."""
    tenant = await _snapshot_tenant(key, txm)
    async with txm.tenant(tenant) as tx:
        conn = connection(tx)
        row = await fetch_one(
            conn,
            f"SELECT {_SNAPSHOT_COLUMNS} FROM rf.eval_snapshots "  # noqa: S608
            "WHERE collection_id = :id",
            {"id": collection_id},
        )
        if row is None:
            raise NotFound("eval snapshot not found")
        if row["content_digest"] is None:
            raise Conflict("the snapshot is still being copied")
        space = await fetch_one(
            conn, "SELECT status FROM rf.embedding_spaces WHERE id = :id", {"id": body.space_id}
        )
        if space is None:
            raise NotFound("embedding space not found")
        if space["status"] == "retired":
            raise ValidationFailed("embedding space is retired")
        job_id = await jobs.enqueue(tx, reembed_job(collection_id, body.space_id))
        await execute(
            conn,
            "UPDATE rf.eval_snapshots SET job_id = :job WHERE collection_id = :id",
            {"job": job_id, "id": collection_id},
        )
        await audit(
            conn, key, "eval_snapshot.reembed", str(collection_id),
            {"space_id": body.space_id, "job_id": str(job_id)}, tenant_id=tenant,
        )  # fmt: skip
    return _snapshot(row, job_id=job_id, index_digest=None)


__all__ = ["require_eval_api", "router"]
