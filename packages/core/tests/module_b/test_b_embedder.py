"""Ollama embedder + registry (respx; no real network)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from ragforge_core.adapters.llm.embedder_ollama import EmbeddingFailed, OllamaEmbedder
from ragforge_core.adapters.llm.embedder_registry import (
    EmbeddingSpace,
    EmbeddingSpaceUnavailable,
    EmbeddingSpaceUnsupported,
    OllamaEmbedderRegistry,
)
from ragforge_core.domain.errors import UpstreamRateLimited, ValidationFailed
from ragforge_core.domain.ports import Embedder, EmbedderRegistry

BASE = "http://ollama.test"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=BASE) as c:
        yield c


def embedder(client: httpx.AsyncClient, **kw: object) -> OllamaEmbedder:
    args: dict[str, object] = {
        "model": "nomic-embed-text",
        "dim": 3,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    }
    args.update(kw)
    return OllamaEmbedder(client, **args)  # type: ignore[arg-type]


def echo_vectors(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200, json={"embeddings": [[float(i), 0.5, 1.0] for i, _ in enumerate(body["input"])]}
    )


@respx.mock
async def test_documents_use_doc_prefix_fixed_num_ctx_and_keep_alive(
    client: httpx.AsyncClient,
) -> None:
    route = respx.post(f"{BASE}/api/embed").mock(side_effect=echo_vectors)
    vectors = await embedder(client).embed_documents(["alpha", "beta"])
    assert vectors == [[0.0, 0.5, 1.0], [1.0, 0.5, 1.0]]
    sent = json.loads(route.calls.last.request.content)
    assert sent["input"] == ["search_document: alpha", "search_document: beta"]
    assert sent["model"] == "nomic-embed-text"
    assert sent["options"] == {"num_ctx": 2048}
    assert sent["keep_alive"] == "30m"


@respx.mock
async def test_query_uses_query_prefix(client: httpx.AsyncClient) -> None:
    route = respx.post(f"{BASE}/api/embed").mock(side_effect=echo_vectors)
    assert await embedder(client).embed_query("where?") == [0.0, 0.5, 1.0]
    assert json.loads(route.calls.last.request.content)["input"] == ["search_query: where?"]


@respx.mock
async def test_batches_preserve_order_and_num_ctx_never_changes(client: httpx.AsyncClient) -> None:
    route = respx.post(f"{BASE}/api/embed").mock(side_effect=echo_vectors)
    vectors = await embedder(client, batch_size=2).embed_documents([f"t{i}" for i in range(5)])
    assert len(vectors) == 5
    assert route.call_count == 3
    sizes = [len(json.loads(c.request.content)["input"]) for c in route.calls]
    assert sizes == [2, 2, 1]
    assert {json.loads(c.request.content)["options"]["num_ctx"] for c in route.calls} == {2048}
    assert [v[0] for v in vectors] == [0.0, 1.0, 0.0, 1.0, 0.0]


@respx.mock
async def test_empty_input_makes_no_request(client: httpx.AsyncClient) -> None:
    route = respx.post(f"{BASE}/api/embed")
    assert await embedder(client).embed_documents([]) == []
    assert route.call_count == 0


@respx.mock
async def test_dimension_mismatch_is_rejected(client: httpx.AsyncClient) -> None:
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})
    )
    with pytest.raises(EmbeddingFailed, match="wrong number of dimensions") as caught:
        await embedder(client).embed_query("x")
    message = str(caught.value)
    assert "nomic" not in message, "client-visible: no model name"
    assert "2" not in message, "client-visible: no dims"


@respx.mock
async def test_wrong_count_and_non_finite_are_rejected(client: httpx.AsyncClient) -> None:
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(200, json={"embeddings": []}))
    with pytest.raises(EmbeddingFailed, match="0 embeddings"):
        await embedder(client).embed_query("x")
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, content=b'{"embeddings": [[NaN, 1.0, 1.0]]}')
    )
    with pytest.raises(EmbeddingFailed, match="non-finite"):
        await embedder(client).embed_query("x")


@respx.mock
async def test_rate_limit_maps_to_upstream_rate_limited(client: httpx.AsyncClient) -> None:
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(503, headers={"retry-after": "7"})
    )
    with pytest.raises(UpstreamRateLimited) as info:
        await embedder(client).embed_query("x")
    assert info.value.retry_after_s == 7.0


@respx.mock
async def test_errors_never_echo_input_text(client: httpx.AsyncClient) -> None:
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(404, json={"error": 'model "nomic-embed-text" not found'})
    )
    with pytest.raises(EmbeddingFailed) as info:
        await embedder(client).embed_documents(["SECRET-TENANT-TEXT"])
    assert "not found" in str(info.value)
    assert "SECRET-TENANT-TEXT" not in str(info.value)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        ("the input length 'SECRET-TENANT-TEXT ...' exceeds the context length", "context"),
        ("failed to tokenize 'SECRET-TENANT-TEXT'", "unrecognised error"),
    ],
)
@respx.mock
async def test_error_text_that_quotes_the_input_is_never_copied(
    client: httpx.AsyncClient, error: str, category: str
) -> None:
    """Review finding: Ollama's error field can echo input fragments; only a fixed category
    may reach the exception (and so the job error log)."""
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(500, json={"error": error}))
    with pytest.raises(EmbeddingFailed) as info:
        await embedder(client).embed_documents(["SECRET-TENANT-TEXT"])
    assert category in str(info.value)
    assert "SECRET" not in str(info.value)
    assert "ollama" not in str(info.value).lower()  # no provider name in client-visible text


@respx.mock
async def test_transport_error_is_wrapped(client: httpx.AsyncClient) -> None:
    respx.post(f"{BASE}/api/embed").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(EmbeddingFailed, match="unreachable"):
        await embedder(client).embed_query("x")


def space(id_: int, **kw: object) -> EmbeddingSpace:
    base: dict[str, object] = {
        "id": id_, "provider": "ollama", "model": f"m{id_}", "dim": 3, "doc_prefix": "d: ",
        "query_prefix": "q: ", "distance": "cosine", "status": "active",
    }  # fmt: skip
    base.update(kw)
    return EmbeddingSpace(**base)  # type: ignore[arg-type]


@respx.mock
async def test_registry_maps_space_to_its_model_and_prefixes(client: httpx.AsyncClient) -> None:
    registry = OllamaEmbedderRegistry(
        [space(1), space(2, model="other", query_prefix="Q2: ", status="building")], client
    )
    assert isinstance(registry, EmbedderRegistry)
    second = registry.for_space(2)
    assert isinstance(second, Embedder)
    assert second.model == "other"
    assert second.dim == 3
    route = respx.post(f"{BASE}/api/embed").mock(side_effect=echo_vectors)
    await second.embed_query("hi")
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "other"
    assert sent["input"] == ["Q2: hi"]
    assert registry.for_space(1) is registry.for_space(1)


def test_registry_rejects_unknown_retired_and_foreign_spaces(client: httpx.AsyncClient) -> None:
    registry = OllamaEmbedderRegistry(
        [space(1, status="retired"), space(2, provider="openai")], client
    )
    with pytest.raises(ValidationFailed, match="retired"):
        registry.for_space(1)
    with pytest.raises(EmbeddingSpaceUnsupported):  # fixed message: no provider/id leak
        registry.for_space(2)
    with pytest.raises(EmbeddingSpaceUnavailable):  # retryable 503, not a permanent NotFound
        registry.for_space(99)


@respx.mock
async def test_input_rejected_by_model_is_permanent_not_retryable(
    client: httpx.AsyncClient,
) -> None:
    """A 400 such as "input exceeds the model context" fails identically on every retry, so it
    must be a ValidationFailed (the job runtime treats it as permanent), not EmbeddingFailed."""
    from ragforge_core.adapters.llm.embedder_ollama import EmbeddingInputRejected
    from ragforge_core.domain.errors import ValidationFailed

    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(
            400, json={"error": "the input length exceeds the context length"}
        )
    )
    with pytest.raises(EmbeddingInputRejected) as info:
        await embedder(client).embed_documents(["x"])
    assert isinstance(info.value, ValidationFailed)
    assert "exceeds the context" not in str(info.value)  # provider text never echoed


@respx.mock
async def test_server_errors_stay_retryable(client: httpx.AsyncClient) -> None:
    from ragforge_core.adapters.llm.embedder_ollama import EmbeddingFailed, EmbeddingInputRejected

    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(EmbeddingFailed) as info:
        await embedder(client).embed_documents(["x"])
    assert not isinstance(info.value, EmbeddingInputRejected)


@respx.mock
async def test_too_long_input_is_isolated_and_split_not_failed(client: httpx.AsyncClient) -> None:
    """The chunker's token count is an estimate; the model's limit is authoritative. A rejected
    input is retried alone, then split in halves (averaged + normalised) until it fits."""
    limit = 40

    def model(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        if any(len(t) > limit for t in inputs):
            return httpx.Response(
                400, json={"error": "the input length exceeds the context length"}
            )
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0] for _ in inputs]})

    respx.post(f"{BASE}/api/embed").mock(side_effect=model)
    short, long = "ok", "\n".join(f"line {i} of a long chunk" for i in range(20))
    vectors = await embedder(client, doc_prefix="").embed_documents([short, long])
    assert len(vectors) == 2
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-9 for v in vectors)


@respx.mock
async def test_unsplittable_rejection_still_fails_permanently(client: httpx.AsyncClient) -> None:
    from ragforge_core.adapters.llm.embedder_ollama import EmbeddingInputRejected

    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(
            400, json={"error": "the input length exceeds the context length"}
        )
    )
    with pytest.raises(EmbeddingInputRejected):
        await embedder(client).embed_documents(["x" * 10])
