#!/usr/bin/env bash
# Pull the configured models into whichever Ollama runtime .env selected. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
set -a; source .env; set +a

models=("$RAGFORGE_EMBED_MODEL" "$RAGFORGE_CHAT_MODEL")
if [[ "${RAGFORGE_OLLAMA_MODE:-host}" == "host" ]]; then
  for m in "${models[@]}"; do echo "models: pulling $m (host Ollama)"; ollama pull "$m"; done
else
  docker compose --profile ollama up -d --wait ollama
  for m in "${models[@]}"; do echo "models: pulling $m (container Ollama)"; docker compose exec -T ollama ollama pull "$m"; done
fi
