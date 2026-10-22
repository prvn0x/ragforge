#!/usr/bin/env bash
# Preflight checks for this machine. Exit non-zero only on blocking problems.
set -uo pipefail

fail=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=1; }

# report_provider OUTPUT RC: one marked line per line of `setup_provider.py --check` output.
# A "Note: ..." line (a shell export overriding .env) is always its own warning: it is advice, not
# the probe's verdict, so it must not inherit the verdict's severity. Every other line carries the
# verdict: rc 0 ok, 2 warning (API unreachable), anything else blocking.
report_provider() {
  local out="$1" rc="$2" line verdicts=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == Note:* ]]; then warn "${line#Note: }"; continue; fi
    verdicts=1
    case $rc in
      0) ok "$line" ;;
      2) warn "$line" ;;
      *) bad "$line" ;;
    esac
  done <<< "$out"
  # A probe that died without saying why must still count: a silent failure is never "ok".
  if (( ! verdicts )); then
    case $rc in
      0) ok "LLM provider check passed" ;;
      2) warn "LLM provider check returned a warning" ;;
      *) bad "LLM provider check failed (exit $rc)" ;;
    esac
  fi
}

# Sourced (by the G2 tests) rather than run: define the helpers above and stop here.
[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0
cd "$(dirname "$0")/.."

echo "RAGForge doctor"
os="$(uname -s)"; arch="$(uname -m)"
ok "OS $os/$arch"

# --- host resources
if [[ "$os" == "Darwin" ]]; then
  ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
else
  ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
fi
if (( ram_gb >= 24 )); then ok "RAM ${ram_gb} GB (8B chat model OK)";
elif (( ram_gb >= 16 )); then ok "RAM ${ram_gb} GB (3B chat model; 8B would swap)";
else warn "RAM ${ram_gb} GB is below 16 GB: expect swapping; prefer a hosted LLM"; fi
free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= 10 )) && ok "Disk free ${free_gb} GB" || bad "Disk free ${free_gb} GB (< 10 GB)"

# --- docker
if docker info >/dev/null 2>&1; then
  vm_gb=$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 ))
  cpus=$(docker info --format '{{.NCPU}}')
  ok "Docker engine $(docker info --format '{{.ServerVersion}}') (${vm_gb} GB, ${cpus} CPUs)"
  (( vm_gb >= 4 )) || bad "Docker VM has ${vm_gb} GB; give it at least 4 GB"
  docker compose version >/dev/null 2>&1 && ok "docker compose $(docker compose version --short)" || bad "docker compose v2 missing"
else
  bad "Docker engine not reachable (start Rancher Desktop / Docker Desktop)"
fi
if command -v rdctl >/dev/null 2>&1 || [[ -x "$HOME/.rd/bin/rdctl" ]]; then
  rd="$(command -v rdctl || echo "$HOME/.rd/bin/rdctl")"
  if "$rd" list-settings 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["kubernetes"]["enabled"] else 1)' 2>/dev/null; then
    warn "Rancher Desktop Kubernetes is enabled (~0.5-1 GB of the VM); not needed (Kubernetes is parked; you may disable it)"
  fi
fi

# --- ports
for port in 8000; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    owner=$(lsof -iTCP:"$port" -sTCP:LISTEN -n -P | awk 'NR==2 {print $1}')
    [[ "$owner" == "com.docke" || "$owner" == "limactl" || "$owner" == "ssh" ]] \
      && ok "Port $port in use by the container runtime (likely RAGForge itself)" \
      || bad "Port $port is in use by $owner"
  else
    ok "Port $port free"
  fi
done

# --- toolchain (dev)
command -v uv >/dev/null && ok "uv $(uv --version | awk '{print $2}')" || bad "uv missing (needed by make tenant, connect-notion, ask, ui, mcp and the tests): brew install uv"
command -v openssl >/dev/null && ok "openssl present" || bad "openssl missing (needed to generate secrets)"

# --- chat model provider (make setup). The probe reads .env itself and never prints the key;
# the key travels only in a request header (scripts/setup_provider.py --check).
if command -v python3 >/dev/null; then
  provider_msg="$(python3 scripts/setup_provider.py --check 2>&1)"; provider_rc=$?
  report_provider "$provider_msg" "$provider_rc"
else
  bad "python3 missing (needed by make setup and make connect-notion)"
fi

# --- ollama (always: embeddings are local even when the chat model is hosted)
if [[ "$os" == "Darwin" ]]; then
  if curl -fsS -m 2 http://localhost:11434/api/version >/dev/null 2>&1; then
    ok "Host Ollama $(curl -fsS -m 2 http://localhost:11434/api/version | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])') (Metal GPU)"
    if docker info >/dev/null 2>&1; then
      if docker run --rm --pull=missing python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 python -c \
          "import urllib.request as u; u.urlopen('http://host.docker.internal:11434/api/version', timeout=3)" \
          >/dev/null 2>&1; then
        ok "Containers reach host Ollama via host.docker.internal"
      else
        warn "Containers cannot reach host Ollama via host.docker.internal (check Docker networking)"
      fi
    fi
    max_loaded="$(launchctl getenv OLLAMA_MAX_LOADED_MODELS 2>/dev/null || true)"
    if [[ -n "$max_loaded" && "$max_loaded" -lt 2 ]]; then
      warn "OLLAMA_MAX_LOADED_MODELS=$max_loaded: chat + embedding models will evict each other"
    fi
  else
    warn "Host Ollama not running: brew services start ollama (otherwise the CPU-only container is used)"
  fi
else
  ok "Linux: containerized Ollama (profile 'ollama'); add the NVIDIA override for GPU"
fi

echo
if (( fail )); then echo "doctor: blocking problems found"; exit 1; fi
echo "doctor: ready"
