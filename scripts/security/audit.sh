#!/usr/bin/env bash
# Dependency + image + config audit (ADR-0018 Security 1-2, R2-8). `make audit`.
#
#   1. pip-audit (via uvx, nothing installed into the project) over `uv export --frozen`: every
#      locked package of every workspace member, extra and group, exactly as uv.lock pins them.
#   2. Trivy over a `docker save` tarball of each built image (api and worker share
#      ragforge/server:dev): OS + Python vulnerabilities, secrets, misconfigurations, and the
#      image config (Dockerfile history) for misconfigurations and secrets.
#   3. Trivy config over docker/ (Dockerfile) and compose.yaml, mounted read-only one by one.
#
# Each step's JSON goes through report.py: HIGH/CRITICAL (any pip-audit advisory, any secret)
# fails unless scripts/security/ignores.toml justifies it. All steps run even if one fails, so
# one run shows everything. Exit: 0 clean, 1 blocking findings, 2 a scan could not run.
#
# Needs network (PyPI/OSV advisories, the Trivy DB). Trivy's DB is cached in the Docker volume
# ragforge-sec-trivy-cache: about 1.5 GB of VM disk (the DB itself, measured
# 2026-10-22); `docker volume rm ragforge-sec-trivy-cache` frees it, the next run re-downloads.
# Override the images with RAGFORGE_AUDIT_IMAGES="img1 img2". Build first: `make build`.
set -euo pipefail
# shellcheck source=lib.sh source-path=SCRIPTDIR
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CACHE_VOLUME="ragforge-sec-trivy-cache"
INPUT_VOLUME="ragforge-sec-input-$$"
tmp="$(mktemp -d)"
# shellcheck disable=SC2317  # invoked by the EXIT trap
cleanup() {
  rm -rf "$tmp"
  docker volume rm -f "$INPUT_VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

worst=0
record() { local rc="$1"; (( rc > worst )) && worst="$rc"; return 0; }

trivy() {
  # Writable: the DB cache volume, and /tmp as an anonymous disk-backed volume that --rm deletes
  # (the DB download unpacks there: hundreds of MB, too much for a RAM tmpfs in a 5 GB VM).
  docker run "${HARDENING[@]}" --mount type=volume,destination=/tmp \
    -v "$CACHE_VOLUME:/root/.cache/trivy" "$@"
}

audit_dependencies() {
  step "pip-audit $PIP_AUDIT_VERSION: uv.lock (all packages, extras and groups)"
  # --no-emit-workspace: our own packages are not on PyPI. The export is fully pinned, so
  # pip-audit needs no resolver (--no-deps --disable-pip): it audits exactly the lock. Environment
  # markers are stripped so every pin is audited, not only those for THIS host (the images are
  # Linux; `watchdog`, for one, is pinned for Linux only). Hashes are dropped with them: advisories
  # are looked up by name and version, so they add nothing here.
  uv export --project "$REPO_ROOT" --frozen --all-packages --all-extras --all-groups \
    --no-emit-workspace --no-hashes --no-annotate --no-header --format requirements.txt --quiet \
    | sed 's/ *;.*$//' > "$tmp/requirements.txt"
  local rc=0
  uvx --quiet --from "pip-audit==$PIP_AUDIT_VERSION" pip-audit \
    --requirement "$tmp/requirements.txt" --no-deps --disable-pip --progress-spinner off \
    --format json --output "$tmp/pip-audit.json" 2>"$tmp/pip-audit.err" || rc=$?
  # pip-audit exits 1 when it found vulnerabilities; anything else means it did not finish.
  if (( rc > 1 )) || [[ ! -s "$tmp/pip-audit.json" ]]; then
    grep -v 'encouraged to fully hash\|pip-compile' "$tmp/pip-audit.err" >&2 || true
    printf 'pip-audit did not complete (exit %s)\n' "$rc" >&2
    record 2; return
  fi
  rc=0; report pip-audit "$tmp/pip-audit.json" --label "uv.lock" || rc=$?
  record "$rc"
}

audit_image() {
  local image="$1" rc=0
  step "Trivy: image $image (vuln, secret, misconfig)"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    printf 'image %s is not built: run make build\n' "$image" >&2
    record 2; return
  fi
  # Stream the tarball into a throwaway volume: Trivy reads it without the Docker socket, and the
  # ~500 MB never lands in the (possibly cloud-synced) working tree.
  docker volume create "$INPUT_VOLUME" >/dev/null
  docker save "$image" | docker run "${HARDENING[@]}" -i -v "$INPUT_VOLUME:/in" \
    --entrypoint sh "$TRIVY_IMAGE" -c 'cat > /in/image.tar' || { record 2; return; }
  trivy -v "$INPUT_VOLUME:/in:ro" "$TRIVY_IMAGE" image --input /in/image.tar --quiet \
    --scanners vuln,secret,misconfig --image-config-scanners misconfig,secret \
    --format json > "$tmp/trivy-image.json" || { record 2; return; }
  docker volume rm -f "$INPUT_VOLUME" >/dev/null
  report trivy "$tmp/trivy-image.json" --label "$image" || rc=$?
  record "$rc"
}

audit_config() {
  local rc=0
  step "Trivy: config (docker/Dockerfile, compose.yaml)"
  # Trivy has no Compose scanner; compose.yaml is mounted so a future Trivy that adds one picks
  # it up, and its hardening is pinned by the project's own compose tests meanwhile.
  trivy -v "$REPO_ROOT/docker:/scan/docker:ro" -v "$REPO_ROOT/compose.yaml:/scan/compose.yaml:ro" \
    "$TRIVY_IMAGE" config /scan --quiet --format json > "$tmp/trivy-config.json" \
    || { record 2; return; }
  report trivy "$tmp/trivy-config.json" --label "docker/ + compose.yaml" || rc=$?
  record "$rc"
}

images() {
  if [[ -n "${RAGFORGE_AUDIT_IMAGES:-}" ]]; then
    local -a requested
    read -r -a requested <<< "$RAGFORGE_AUDIT_IMAGES"
    printf '%s\n' "${requested[@]}"
  else
    # The images compose BUILDS for api and worker (third-party images such as postgres are
    # pinned by digest in compose.yaml and out of scope here).
    docker compose -f "$REPO_ROOT/compose.yaml" config --images api worker 2>/dev/null \
      | grep '^ragforge/' | sort -u
  fi
}

need_docker
audit_dependencies
image_list="$(images)" || true
[[ -n "$image_list" ]] || { printf 'no ragforge/* image found for api/worker\n' >&2; record 2; }
while IFS= read -r image; do
  [[ -n "$image" ]] && audit_image "$image"
done <<< "$image_list"
audit_config

step "audit result"
case $worst in
  0) echo "clean: nothing blocking" ;;
  1) echo "BLOCKED: fix the findings above or justify them in scripts/security/ignores.toml" ;;
  *) echo "INCOMPLETE: a scan could not run (see above); this is not a clean result" ;;
esac
exit "$worst"
