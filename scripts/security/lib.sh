# shellcheck shell=bash disable=SC2034  # the variables are read by the scripts that source this
# Shared by audit.sh and secrets.sh (sourced, not executed).
#
# Scanner images are pinned by version AND digest: a scanner sees every file we hand it, so a
# re-tagged image must not be able to slip in. Bump both together (docs/handoff/SEC-SCAN.md).
#
# Container hardening for every scanner run: the repo (or the one file a scan needs) is mounted
# read-only, never $HOME and never the Docker socket; root fs read-only; all capabilities dropped;
# no-new-privileges. Images reach Trivy as `docker save` tarballs streamed into a throwaway
# volume, so the scanner never talks to the Docker daemon.

TRIVY_IMAGE="aquasec/trivy:0.75.0@sha256:af6acf9a6b85dfe389a1941505c0ce9efef52a4719635e1a962f022a3d855daa"
GITLEAKS_IMAGE="zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f"
PIP_AUDIT_VERSION="2.10.1"

SEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SEC_DIR/../.." && pwd)"

HARDENING=(--rm --read-only --cap-drop ALL --security-opt no-new-privileges:true)

# The gate runs under the project's Python 3.12 (stdlib only) without syncing the environment.
report() { uv run --no-sync --project "$REPO_ROOT" python -I "$SEC_DIR/report.py" "$@"; }

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$1" >&2; exit 2; }

need_docker() {
  docker info >/dev/null 2>&1 || die "Docker is not reachable (is Rancher/Docker Desktop running?)"
}

# secret_masks: mount options that hide the files which exist to hold secrets, so a scanner
# container cannot read them even though the rest of the repo is mounted. An empty file over each
# .env / .env.<name> (except the committed .env.example), an empty tmpfs over evals/private/.
# Only paths that exist are masked: Docker cannot create a mount point inside a read-only bind.
# $1 = the repo's mount point inside the container.
secret_masks() {
  local dest="$1" f
  for f in "$REPO_ROOT"/.env "$REPO_ROOT"/.env.*; do
    [[ -f "$f" && "$(basename "$f")" != ".env.example" ]] || continue
    printf -- '-v\n/dev/null:%s/%s:ro\n' "$dest" "$(basename "$f")"
  done
  if [[ -d "$REPO_ROOT/evals/private" ]]; then
    printf -- '--mount\ntype=tmpfs,destination=%s/evals/private,tmpfs-size=1m\n' "$dest"
  fi
}
