"""Typed async SDK for every /v1 endpoint. Depends only on `ragforge_contracts` and httpx.

Errors: any non-success response raises `ApiError` carrying the server's RFC 9457 Problem.
Streams (`chat`, `resume`, `job_events`) are async iterators; event types this SDK version does not
know are skipped (forward compatibility: a newer server may add events).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Self
from uuid import UUID

import httpx
from pydantic import BaseModel, SecretStr

from ragforge_client.sse import SseMessage, iter_sse
from ragforge_contracts.api import (
    Accepted,
    ApiKey,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiScope,
    ApprovalDecision,
    ApprovalResume,
    AuditEntry,
    ChatRequest,
    Collection,
    CollectionCreate,
    Connector,
    ConnectorCreate,
    ConnectorKind,
    Document,
    EvalChunksResponse,
    EvalReembedRequest,
    EvalResolveRequest,
    EvalResolveResponse,
    EvalSnapshot,
    EvalSnapshotRequest,
    EvalVerifyRequest,
    EvalVerifyResponse,
    Exclusion,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    Job,
    JobEvent,
    Page,
    Problem,
    QueryRequest,
    QueryResponse,
    Readiness,
    TenantCreate,
    TenantKind,
    TenantProvisioned,
)
from ragforge_contracts.sse import ChatEvent, decode_job_event, decode_sse_data


class ApiError(Exception):
    def __init__(self, problem: Problem) -> None:
        super().__init__(f"{problem.status} {problem.title}: {problem.detail or ''}".strip())
        self.problem = problem

    @property
    def status(self) -> int:
        return self.problem.status


class RagforgeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )
        # Streams may be silent for long stretches (model thinking): no read timeout there.
        self._stream_timeout = httpx.Timeout(timeout, read=None)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------- health
    async def readiness(self) -> Readiness:
        """GET /readyz. Returns the body for both 200 and 503 (503 = not ready, not an error)."""
        response = await self._http.get("/readyz")
        if response.status_code in (200, 503):
            return Readiness.model_validate_json(response.content)
        raise self._error(response)

    # ------------------------------------------------------------- tenants & keys
    async def provision_tenant(
        self,
        name: str,
        owner_display_name: str,
        *,
        kind: TenantKind = TenantKind.personal,
        team_visibility_consent: bool = False,
    ) -> TenantProvisioned:
        body = TenantCreate(
            name=name,
            kind=kind,
            owner_display_name=owner_display_name,
            team_visibility_consent=team_visibility_consent,
        )
        return TenantProvisioned.model_validate_json(await self._send("POST", "/v1/tenants", body))

    async def create_api_key(self, name: str, scopes: Sequence[ApiScope]) -> ApiKeyCreated:
        body = ApiKeyCreate(name=name, scopes=list(scopes))
        return ApiKeyCreated.model_validate_json(await self._send("POST", "/v1/api-keys", body))

    async def list_api_keys(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> Page[ApiKey]:
        return Page[ApiKey].model_validate_json(await self._list("/v1/api-keys", cursor, limit))

    async def revoke_api_key(self, key_id: UUID) -> None:
        await self._send("DELETE", f"/v1/api-keys/{key_id}")

    # ------------------------------------------------------------- collections
    async def create_collection(self, name: str) -> Collection:
        body = CollectionCreate(name=name)
        return Collection.model_validate_json(await self._send("POST", "/v1/collections", body))

    async def list_collections(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> Page[Collection]:
        content = await self._list("/v1/collections", cursor, limit)
        return Page[Collection].model_validate_json(content)

    async def get_collection(self, collection_id: UUID) -> Collection:
        content = await self._send("GET", f"/v1/collections/{collection_id}")
        return Collection.model_validate_json(content)

    # ------------------------------------------------------------- connectors
    async def create_connector(
        self,
        *,
        target_collection_id: UUID,
        display_name: str,
        token: str | SecretStr,
        kind: ConnectorKind = ConnectorKind.notion,
    ) -> Connector:
        secret = token if isinstance(token, SecretStr) else SecretStr(token)
        body = ConnectorCreate(
            kind=kind,
            target_collection_id=target_collection_id,
            display_name=display_name,
            token=secret,
        )
        # model_dump masks SecretStr as "**********": put the real value back for the wire only.
        payload = body.model_dump(mode="json") | {"token": secret.get_secret_value()}
        content = await self._send("POST", "/v1/connectors", json=payload)
        return Connector.model_validate_json(content)

    async def list_connectors(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> Page[Connector]:
        content = await self._list("/v1/connectors", cursor, limit)
        return Page[Connector].model_validate_json(content)

    async def get_connector(self, connector_id: UUID) -> Connector:
        content = await self._send("GET", f"/v1/connectors/{connector_id}")
        return Connector.model_validate_json(content)

    async def sync_connector(self, connector_id: UUID) -> Accepted:
        content = await self._send("POST", f"/v1/connectors/{connector_id}/sync")
        return Accepted.model_validate_json(content)

    async def delete_connector(self, connector_id: UUID) -> Accepted:
        content = await self._send("DELETE", f"/v1/connectors/{connector_id}")
        return Accepted.model_validate_json(content)

    # ------------------------------------------------------------- documents & exclusions
    async def list_documents(
        self,
        *,
        collection_id: UUID | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[Document]:
        extra = {"collection_id": str(collection_id)} if collection_id else {}
        content = await self._list("/v1/documents", cursor, limit, extra)
        return Page[Document].model_validate_json(content)

    async def delete_document(self, document_id: UUID, *, forget_exclusion: bool = False) -> None:
        params = {"forget_exclusion": "true"} if forget_exclusion else None
        await self._send("DELETE", f"/v1/documents/{document_id}", params=params)

    async def list_exclusions(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> Page[Exclusion]:
        content = await self._list("/v1/exclusions", cursor, limit)
        return Page[Exclusion].model_validate_json(content)

    async def delete_exclusion(self, exclusion_id: UUID) -> None:
        await self._send("DELETE", f"/v1/exclusions/{exclusion_id}")

    # ------------------------------------------------------------- jobs
    async def get_job(self, job_id: UUID) -> Job:
        return Job.model_validate_json(await self._send("GET", f"/v1/jobs/{job_id}"))

    async def job_events(self, job_id: UUID) -> AsyncIterator[JobEvent]:
        """Yields a JobEvent per change; ends when the job finishes."""
        async for message in self._stream("GET", f"/v1/jobs/{job_id}/events"):
            if message.event == "job":
                yield decode_job_event(message.data)

    # ------------------------------------------------------------- retrieval & chat
    async def query(
        self, query: str, collection_ids: Sequence[UUID], *, k: int = 8, stages: bool = False
    ) -> QueryResponse:
        """`stages=True` (eval scope, ADR-0017) also returns every retrieval stage."""
        body = QueryRequest(query=query, collection_ids=list(collection_ids), k=k, stages=stages)
        # `stages` only when asked: a server that predates it forbids unknown request fields
        payload = body.model_dump(mode="json", exclude={"stages"} if not stages else None)
        return QueryResponse.model_validate_json(
            await self._send("POST", "/v1/query", json=payload)
        )

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        """One interview question at `level`. source=notes: grounded in the notes; pass the chunk
        ids already asked in `exclude_chunk_ids` so nothing repeats; 409 INTERVIEW_NO_NOTES or
        INTERVIEW_EXHAUSTED when there is nothing (left). source=general (only on the user's
        explicit opt-in): model general knowledge, never verified, a topic is required."""
        # `chunk_ids` (eval scope, ADR-0017) only when set: an older server forbids the field
        payload = request.model_dump(
            mode="json", exclude=None if request.chunk_ids else {"chunk_ids"}
        )
        return InterviewQuestion.model_validate_json(
            await self._send("POST", "/v1/interview/question", json=payload)
        )

    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        """Grade an answer at the question's level. source=notes: a verified model answer from
        the notes; source=general: an unverified general-knowledge grade (token required)."""
        return InterviewGrade.model_validate_json(
            await self._send("POST", "/v1/interview/grade", request)
        )

    async def chat(
        self,
        message: str,
        collection_ids: Sequence[UUID],
        *,
        conversation_id: UUID | None = None,
        trace_shown: bool = False,
    ) -> AsyncIterator[ChatEvent]:
        """Render `token` text as plain text until the `verification` event arrives.
        `trace_shown=True` (eval scope, ADR-0017) adds `shown` to the `done` event."""
        body = ChatRequest(
            conversation_id=conversation_id, collection_ids=list(collection_ids), message=message
        )
        url = "/v1/chat?trace=shown" if trace_shown else "/v1/chat"
        async for event in self._chat_stream(url, body):
            yield event

    async def resume(
        self, run_id: UUID, decisions: Mapping[str, ApprovalDecision]
    ) -> AsyncIterator[ChatEvent]:
        """Approve/reject pending tool calls (needs an `approvals:write` key, never MCP)."""
        body = ApprovalResume(decisions=dict(decisions))
        async for event in self._chat_stream(f"/v1/approvals/{run_id}", body):
            yield event

    # ------------------------------------------------------------- eval (ADR-0017, eval scope)
    async def eval_verify(self, request: EvalVerifyRequest) -> EvalVerifyResponse:
        """Run the full verification pipeline on a supplied answer, with per-layer evidence."""
        return EvalVerifyResponse.model_validate_json(
            await self._send("POST", "/v1/eval/verify", request)
        )

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse:
        """The text of chunks the caller can see (the eval tenant); invisible ids are absent."""
        params = {"ids": ",".join(str(i) for i in chunk_ids)}
        return EvalChunksResponse.model_validate_json(
            await self._send("GET", "/v1/eval/chunks", params=params)
        )

    async def eval_resolve(self, request: EvalResolveRequest) -> EvalResolveResponse:
        """Stable page/chunk keys -> current ids (or stale / missing); never auto-remaps."""
        return EvalResolveResponse.model_validate_json(
            await self._send("POST", "/v1/eval/resolve", request)
        )

    async def eval_tenant_create(self) -> TenantProvisioned:
        """Bootstrap once per owner (admin key): the paired eval tenant and its dedicated eval
        key (eval + chat + read). The secret is in `api_key.secret`: store it, never print it."""
        return TenantProvisioned.model_validate_json(
            await self._send("POST", "/v1/eval/tenant", json={})
        )

    async def eval_snapshot(self, request: EvalSnapshotRequest) -> EvalSnapshot:
        """Copy one of the caller's collections into its paired eval tenant (admin key)."""
        return EvalSnapshot.model_validate_json(
            await self._send("POST", "/v1/eval/snapshots", request)
        )

    async def eval_snapshot_status(self, collection_id: UUID) -> EvalSnapshot:
        return EvalSnapshot.model_validate_json(
            await self._send("GET", f"/v1/eval/snapshots/{collection_id}")
        )

    async def eval_reembed(self, collection_id: UUID, request: EvalReembedRequest) -> EvalSnapshot:
        """Re-embed a snapshot into a new embedding space from its stored chunk text."""
        return EvalSnapshot.model_validate_json(
            await self._send("POST", f"/v1/eval/snapshots/{collection_id}/reembed", request)
        )

    # ------------------------------------------------------------- audit
    async def list_audit(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> Page[AuditEntry]:
        content = await self._list("/v1/audit", cursor, limit)
        return Page[AuditEntry].model_validate_json(content)

    # ------------------------------------------------------------- transport
    async def _send(
        self,
        method: str,
        url: str,
        body: BaseModel | None = None,
        *,
        json: Any = None,
        params: Mapping[str, str] | None = None,
    ) -> bytes:
        payload = body.model_dump(mode="json") if body is not None else json
        response = await self._http.request(method, url, json=payload, params=params)
        if response.is_success:
            return response.content
        raise self._error(response)

    async def _list(
        self,
        url: str,
        cursor: str | None,
        limit: int | None,
        extra: Mapping[str, str] | None = None,
    ) -> bytes:
        params = dict(extra or {})
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = str(limit)
        return await self._send("GET", url, params=params)

    async def _chat_stream(self, url: str, body: BaseModel) -> AsyncIterator[ChatEvent]:
        async for message in self._stream("POST", url, body.model_dump(mode="json")):
            event = decode_sse_data(message.data)
            if event is not None:
                yield event

    async def _stream(
        self, method: str, url: str, payload: Any = None
    ) -> AsyncIterator[SseMessage]:
        request = self._http.build_request(method, url, json=payload, timeout=self._stream_timeout)
        response = await self._http.send(request, stream=True)
        try:
            if not response.is_success:
                await response.aread()
                raise self._error(response)
            async for message in iter_sse(response.aiter_lines()):
                yield message
        finally:
            await response.aclose()

    @staticmethod
    def _error(response: httpx.Response) -> ApiError:
        try:
            problem = Problem.model_validate_json(response.content)
        except ValueError:
            problem = Problem(title=response.reason_phrase or "error", status=response.status_code)
        return ApiError(problem)
