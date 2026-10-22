# syntax=docker/dockerfile:1.7
# One multi-stage Dockerfile; targets: server (api | worker | migrate | db-bootstrap commands).
# mcp and ui targets are added in Phase 3. Base images are pinned by digest.

# Debian 13 (trixie): no CRITICAL and half the HIGH OS CVEs of bookworm (SEC-SCAN, 2026-10-22).
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336

FROM ${UV_IMAGE} AS uv

# ------------------------------------------------------------------ builder
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Dependency layer: only lockfile + package manifests, so code edits don't bust the cache.
COPY pyproject.toml uv.lock ./
COPY packages/contracts/pyproject.toml packages/contracts/
COPY packages/core/pyproject.toml packages/core/
COPY packages/client/pyproject.toml packages/client/
COPY packages/mcp/pyproject.toml packages/mcp/
COPY packages/ui/pyproject.toml packages/ui/
ARG PACKAGE=ragforge-core
ARG EXTRAS="--extra agent --extra ingest"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package "${PACKAGE}" ${EXTRAS}

# Project layer: install workspace packages non-editable (no source needed at runtime).
COPY packages packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package "${PACKAGE}" ${EXTRAS}

# ------------------------------------------------------------------ models
# Bake the reranker (handoff B): at runtime FlashRankReranker never downloads, and FlashRank's
# Ranker writes nothing when the model directory exists, so the read-only root fs is fine.
# Reproducible: the zip is fetched from a PINNED Hugging Face commit and verified by sha256, in
# a stage that depends on nothing in the repo, so code edits never re-download it. To change
# the model, update all three ARGs together AND RAGFORGE_RERANK_MODEL (config.rerank_model).
FROM ${PYTHON_IMAGE} AS models
ARG RERANK_MODEL=ms-marco-TinyBERT-L-2-v2
ARG RERANK_MODEL_COMMIT=858a1ac046a05663a35367eac852d7f76feeefdd
ARG RERANK_MODEL_SHA256=752eddf1c5ece3c5e6115e9ab52eb3b16436bbf9b3a091cbb62b5b8eadc47105
ADD --checksum=sha256:${RERANK_MODEL_SHA256} \
    https://huggingface.co/prithivida/flashrank/resolve/${RERANK_MODEL_COMMIT}/${RERANK_MODEL}.zip \
    /tmp/model.zip
# Extract only the model directory (the zip also carries macOS __MACOSX resource forks).
RUN python -c "import sys, zipfile; m = sys.argv[1]; z = zipfile.ZipFile('/tmp/model.zip'); z.extractall('/opt/models/flashrank', [n for n in z.namelist() if n.startswith(m + '/')])" "${RERANK_MODEL}" \
    && test -f "/opt/models/flashrank/${RERANK_MODEL}/config.json" \
    && chmod -R a+rX,go-w /opt/models/flashrank \
    && rm /tmp/model.zip

# ------------------------------------------------------------------ runtime base
FROM ${PYTHON_IMAGE} AS runtime
# The base image's pip (/usr/local) is never used at runtime (the app runs from /opt/venv, built
# by uv) and carries its own advisories (Trivy: 5 MEDIUM, 1 LOW).
RUN python -m pip uninstall --yes --no-input pip
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
# Mount point for the shared uploads volume: a fresh named volume inherits this ownership.
RUN mkdir -p /data/uploads && chown 10001:10001 /data/uploads
COPY --from=builder /opt/venv /opt/venv
# Root-owned and read-only for uid 10001 (the reranker only reads it).
COPY --from=models /opt/models/flashrank /opt/models/flashrank
# ORT_DISABLE_TELEMETRY: libonnxruntime (the reranker) ships a 1DS telemetry client that
# initialises at import; =1 stops it (verified in this image; SECURITY.md §3.5, flashrank.py).
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1
USER 10001:10001
WORKDIR /tmp

# ------------------------------------------------------------------ server
# No image-level HEALTHCHECK: one image runs several commands (api, worker, migrate); probes are
# defined per service in compose (and per Deployment in Helm).
FROM runtime AS server
EXPOSE 8000
CMD ["ragforge-api"]
