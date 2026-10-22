# ADR-0004: Ollama runtime: host on macOS (Metal), container elsewhere; model chosen by evals

**Status:** Accepted · 2026-10-16 · Revised in rounds 2, 4 and 5

## Context
- Docker Desktop on macOS doesn't pass the Metal GPU through, so containerized Ollama is CPU-only there.
- On Linux, containers *do* get GPU access (NVIDIA Container Toolkit). A host Ollama, by contrast, would have to be bound to a network interface that containers can reach. Binding it to a bridge IP is fragile (it fails if Ollama starts before Docker, and the host CLI then needs `OLLAMA_HOST`), and `0.0.0.0` exposes an unauthenticated API.
- Ollama unloads idle models (default 5 min) and reloads a model when `num_ctx` changes.

## Decision
- **Runtime:**
  - **macOS + host Ollama reachable:** host via `host.docker.internal` (Metal).
  - **Linux, or macOS without host Ollama:** the `ollama` compose profile. No host port is published, and a `compose.gpu.yml` override adds NVIDIA GPU access when `make doctor` detects it.
- **Models (config):**
  - chat `qwen3:8b` with thinking off if RAM ≥ 24 GB, else `llama3.2:3b`;
  - embeddings `nomic-embed-text` with the space's task prefixes.
  - `make eval-models` reports quality, latency **and verifier precision/recall**. The default changes only with evidence.
- **Request discipline:** a fixed `num_ctx` per model (chat 8192, embed 2048) and `keep_alive: "30m"` on every request. `make doctor` warns if the host's `OLLAMA_MAX_LOADED_MODELS` is below 2 (macOS host mode).
- `LLM_PROVIDER=openai|bedrock` switches adapters.

## Consequences
- Metal speed on Macs, GPU on Linux when present, no network exposure, no reload churn.
