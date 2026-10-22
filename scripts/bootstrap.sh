#!/usr/bin/env bash
# Create .env with fresh secrets (only if missing) and pick an Ollama runtime + models (ADR-0004).
# Idempotent: never overwrites an existing value in .env.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=.env
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

has() { grep -qE "^$1=" "$ENV_FILE"; }
put() { has "$1" || printf '%s=%s\n' "$1" "$2" >>"$ENV_FILE"; }
secret() { openssl rand -base64 33 | tr -d '/+=\n' | cut -c1-32; }

for role in SUPERUSER MIGRATOR API WORKER METRICS BACKUP; do
  put "RAGFORGE_DB_${role}_PASSWORD" "$(secret)"
done
put RAGFORGE_KEK "$(openssl rand -base64 32)"
put RAGFORGE_GRAFANA_ADMIN_PASSWORD "$(secret)"   # Grafana admin (make obs-up); applied on Grafana's first start only
put RAGFORGE_LOG_LEVEL INFO

# ---------------------------------------------------------------- Ollama runtime + model choice
os="$(uname -s)"
if [[ "$os" == "Darwin" ]]; then
  ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
else
  ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
fi
if (( ram_gb >= 24 )); then chat_model="qwen3:8b"; else chat_model="llama3.2:3b"; fi
put RAGFORGE_CHAT_MODEL "$chat_model"
put RAGFORGE_EMBED_MODEL "nomic-embed-text"

if [[ "$os" == "Darwin" ]] && curl -fsS -m 2 http://localhost:11434/api/version >/dev/null 2>&1; then
  put RAGFORGE_OLLAMA_BASE_URL "http://host.docker.internal:11434"
  put RAGFORGE_OLLAMA_MODE host
else
  put RAGFORGE_OLLAMA_BASE_URL "http://ollama:11434"
  put RAGFORGE_OLLAMA_MODE container
fi

# The chat provider is deliberately NOT written here: `make up` asks once in a terminal
# (scripts/setup_provider.py --if-unset), and an absent value means local Ollama.
echo "bootstrap: .env ready (RAM ${ram_gb} GB -> chat model $(grep -E '^RAGFORGE_CHAT_MODEL=' "$ENV_FILE" | cut -d= -f2), ollama $(grep -E '^RAGFORGE_OLLAMA_MODE=' "$ENV_FILE" | cut -d= -f2))"
