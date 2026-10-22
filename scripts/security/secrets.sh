#!/usr/bin/env bash
# Secret scan of the working tree (ADR-0018 Security 3, R2-8). `make secrets-scan`.
#
# gitleaks (pinned container) over the whole repo with the repo's .gitleaks.toml: gitleaks'
# built-in rules plus RAGForge's own token shapes (rf_ keys, Notion ntn_/secret_, Gemini AIza).
# The tree is not a git repo, so the scope is set by the config's path allowlists, not .gitignore.
#
# Never prints a secret: gitleaks runs with --redact, and report.py prints only rule, file and
# line (it never reads the Secret/Match fields). The files that exist to hold secrets (.env*,
# evals/private/) are masked out of the container entirely (lib.sh secret_masks).
#
# Exit: 0 clean, 1 leak found, 2 the scan itself failed. Usable as a pre-push hook later; once the
# repo is under git, also run `gitleaks git` once over the full history (docs/SECURITY.md).
set -euo pipefail
# shellcheck source=lib.sh source-path=SCRIPTDIR
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

need_docker
masks=()
while IFS= read -r opt; do masks+=("$opt"); done < <(secret_masks /repo)

step "gitleaks: working tree (custom rules: rf_ keys, Notion ntn_/secret_, Gemini AIza)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# --network none: the scan needs nothing from outside. --ignore-gitleaks-allow: inline
# `gitleaks:allow` comments cannot silence a finding; suppressions live in .gitleaks.toml only.
# --exit-code 0: gitleaks' own verdict is replaced by report.py's, so a crash (non-zero here)
# is told apart from "leaks found".
docker run "${HARDENING[@]}" --network none \
  -v "$REPO_ROOT:/repo:ro" ${masks[@]+"${masks[@]}"} -w /repo \
  "$GITLEAKS_IMAGE" dir . \
  --config /repo/.gitleaks.toml --redact --no-banner --no-color --log-level error \
  --ignore-gitleaks-allow --max-target-megabytes 20 --exit-code 0 \
  --report-format json --report-path - > "$tmp/gitleaks.json" \
  || die "gitleaks did not complete (see above)"

report gitleaks "$tmp/gitleaks.json" --label "working tree ($REPO_ROOT)"
