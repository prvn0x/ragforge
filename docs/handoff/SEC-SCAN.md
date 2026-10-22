# Handoff SEC-SCAN: dependency, image and secret scans (ADR-0018 Security 1-3, R2-8)

Implements the scan half of the security pass: `pip-audit` over the lock, Trivy over the built
image and the Dockerfile, gitleaks over the working tree with RAGForge's own token rules, and a
pytest that pins the `.gitignore` / `.dockerignore` coverage. Nothing outside the owned paths
was edited; the locked seams (`observability/catalog.py`, `prompt_context.py`) were not touched
and this module does not need them. Everything ran against the real tree and the real
`ragforge/server:dev` image on 2026-10-22; every test uses synthetic data only. No `.env`,
`evals/private/` or `~/.config/ragforge` content was read (the scanners cannot read them
either, see "Containment").

## What was built

| Path | What it is |
|---|---|
| `scripts/security/audit.sh` | `make audit`: pip-audit → Trivy image (per built `ragforge/*` image of api/worker) → Trivy config. Runs every step even if one fails, then exits with the worst result |
| `scripts/security/secrets.sh` | `make secrets-scan`: gitleaks `dir` over the repo with `.gitleaks.toml` |
| `scripts/security/lib.sh` | Shared: scanner pins (version + digest), container hardening flags, the secret-file masks |
| `scripts/security/report.py` | The gate (stdlib only, `mypy --strict` clean): parses one tool's JSON, applies `ignores.toml`, prints a summary, exit 0/1/2 |
| `scripts/security/ignores.toml` | Justified ignores: 16 entries, all Trivy OS CVEs with no Debian fix, each with a reason and `review_by = 2027-01-31` |
| `.gitleaks.toml` | Built-in rules + 4 custom rules (`ragforge-api-key`, `notion-token-ntn`, `notion-token-secret`, `gemini-api-key`), path allowlist, 3 narrow allowlists |
| `packages/core/tests/test_sec_scan_ignore_files.py` | `.gitignore` / `.dockerignore` coverage, with the real matchers (pathspec's GitIgnoreSpec, docker-py's PatternMatcher) |
| `packages/core/tests/test_sec_scan_gitleaks_rules.py` | The custom rules catch every token shape in 6 contexts, reject near misses and placeholders, spare the repo's fixtures; scope paths; one integration test through the real pinned gitleaks |
| `packages/core/tests/test_sec_scan_report.py` | The gate: thresholds, ignore scoping/expiry, alias matching, secrets never printed, malformed input is exit 2 (not a pass), the shipped `ignores.toml` is valid |

### Gate policy (report.py)
- **pip-audit:** every advisory blocks (PyPI advisories carry no severity).
- **Trivy:** HIGH/CRITICAL block; MEDIUM/LOW are counted only. Any secret found in an image
  blocks, whatever its severity.
- **gitleaks:** every finding blocks. It cannot be ignored in `ignores.toml` (that is an
  error); suppressions live only in `.gitleaks.toml`, next to the rule. `secrets.sh` passes
  `--ignore-gitleaks-allow`, so inline `gitleaks:allow` comments do not work either.
- **Ignores:** `reason` (at least 20 characters), `review_by` (a TOML date) and optional
  `packages` (the scope: the same CVE in another package still blocks). After `review_by` the
  ignore stops hiding anything and the run fails. An ignore that matched nothing is listed on
  one `note` line (stale), without failing.
- **Exit:** 0 clean, 1 blocking, 2 the scan or its input is unusable (bad JSON, wrong schema,
  scanner crash). A crash is never read as "clean".
- **No secret is ever printed:** gitleaks runs with `--redact`, and report.py never reads the
  `Secret`/`Match`/`Code` fields (a test feeds it a canary in all three and checks stdout and
  stderr).

### Containment (how the scanners run)
- Scanner images are pinned by version AND digest in `lib.sh`:
  - `aquasec/trivy:0.75.0@sha256:af6acf9a…55daa` (released 2026-10-01; 0.76.x is under 2 weeks old);
  - `zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0…bbb7f`;
  - `pip-audit==2.10.1` via `uvx`.
- Every scanner container runs with `--rm --read-only --cap-drop ALL --security-opt
  no-new-privileges:true`. Neither `$HOME` nor the Docker socket is mounted.
- **gitleaks:** `--network none`. The repo is mounted read-only, with `/dev/null` mounted over
  `.env` / `.env.<name>` (but not `.env.example`) and an empty tmpfs over `evals/private/`. The
  container cannot read the real secrets. I checked this: inside the container `.env` is 0
  bytes, `evals/private` is empty and the mount refuses writes.
- **Trivy image scan:** `docker save IMAGE` is streamed into a throwaway volume
  (`ragforge-sec-input-<pid>`, removed on exit), so the ~500 MB tarball never touches the
  iCloud-synced tree and Trivy never talks to the daemon. The vulnerability DB is cached in the
  `ragforge-sec-trivy-cache` volume. That is about **1.5 GB of VM disk** (measured): worth
  knowing on the 5 GB Rancher VM. `docker volume rm ragforge-sec-trivy-cache` frees it.
- **Trivy config scan:** only `docker/` and `compose.yaml` are mounted, each read-only.

## Findings (scan of 2026-10-22)

| Scan | Raw findings | Fixed here | Proposed (outside my paths) | Justified-ignored | Below threshold |
|---|---|---|---|---|---|
| gitleaks, working tree | 37 (all `generic-api-key` false positives) + 1 exact-shape canary | 37 + 1 by allowlists in `.gitleaks.toml` | — | — | — |
| pip-audit, uv.lock (125 pins) | 25 advisories in 2 packages | — | 25 (2 version bumps) | 0 | — |
| Trivy image `ragforge/server:dev` | CRITICAL 3, HIGH 61, MEDIUM 116, LOW 109, UNKNOWN 2 | — | 1 CRITICAL (certifi) + base bump + pip removal | 63 (2 CRITICAL, 61 HIGH = 16 CVEs) | 227 |
| Trivy config (Dockerfile) | LOW 1 (DS-0026 no HEALTHCHECK) | — | — | — | 1 (by design: probes per service) |

**gitleaks in detail:**
- **Custom rules:** no real token of any of the four shapes anywhere in `docs/` (handoffs,
  ADRs, workflows), `evals/demo/`, `packages/` or `scripts/`.
- **The allowlists, kept narrow:**
  - 36 hits were SHA-256 `cache_key` digests in `evals/demo/fixtures/*.jsonl`. They are
    allowlisted only for `generic-api-key`, only in that folder, and only when the value is 64
    hex characters.
  - 1 hit was the malformed-key list in `module_e2/test_e2_auth_unit.py`. It is allowlisted for
    that file and that literal only.
  - The g2 fixture `FAKE_KEY` has the exact Gemini shape. It carries the marker `leakcanary`,
    which is a global stopword. A test proves the stopword is load-bearing.

**pip-audit in detail:**
- `certifi 2026.7.22`: PYSEC-2026-4320 / GHSA-6q48-6c7w-9gw2 (CRITICAL in Trivy). Fixed in
  `2026.10.19`. It is in the image (via httpx/requests).
- `pillow 12.3.0`: 24 advisories (CVE-2026-112390…112901). Fixed in `13.0.0`.
  - Pillow comes only through `streamlit` (the UI, not the image).
  - The locked streamlit 1.66.0 requires `pillow<13`, and `streamlit 1.66.1` allows `<14`, so
    both move together.
- **Freshness caveat:** certifi 2026.10.19 is 3 days old, pillow 13.0.0 is 6 days old and
  streamlit 1.66.1 is 1 day old. These are security fixes, so I recommend taking them now. If
  the owner prefers a cool-off, add pip-audit ignores with `review_by` a week or two out. Don't
  leave the gate red.

**Trivy in detail:**
- **OS base:** Debian 12.15 (`python:3.12-slim-bookworm`). All 63 HIGH/CRITICAL OS findings
  have NO fixed version in Debian (status affected / fix_deferred / will_not_fix). The 16 CVEs
  are justified one by one in `ignores.toml`: util-linux mount/nsenter ×6, sqlite ×3, zlib
  minizip, gzip LZH, openssl DTLS, ncurses, systemd-homed, libacl, perl Archive::Tar. None is
  reachable from the app as it runs: uid 10001, read-only root, `cap_drop: [ALL]`, no SQLite, no
  shell-outs.
- **Base comparison (measured, same Trivy DB):**

  | Base | CRITICAL | HIGH | Unique HIGH/CRITICAL CVEs | Fixable |
  |---|---|---|---|---|
  | bookworm (current pin) | 2 | 61 | 16 | 0 |
  | bookworm (latest digest) | 2 | 61 | 16 | 0 |
  | `python:3.12-slim-trixie` | 0 | 44 | 8 | 0 |

  Re-pulling bookworm fixes nothing; trixie removes both CRITICALs and half the CVEs.
- **Python packages in the image:** certifi (above), plus 5 MEDIUM and 1 LOW in the base image's
  own `pip 25.0.1` under `/usr/local`. The app never uses that pip (it runs from `/opt/venv`,
  built by uv), so the fix is to remove it.
- **Secrets in the image and the image config:** 0.

## Integration steps (exact)

### 1. Ignore files (`.gitignore`, `.dockerignore`), then un-xfail the test
Append to `.gitignore`:
```gitignore

# Secrets (ADR-0018 R2-8): env variants, key and certificate files. The template stays committed.
.env.*
!.env.example
*.key
*.pem
```
Append to `.dockerignore` (Docker's `*.key` matches only at the context root, hence `**/`):
```
**/.env
**/.env.*
**/*.key
**/*.pem
```
Then in `packages/core/tests/test_sec_scan_ignore_files.py`, set the third column of every
`_SENSITIVE` row to `frozenset()`. Then delete the now-unused `_BOTH` and `PENDING` and make
`_cases` return `[pytest.param(path, id=case_id) for case_id, path, _ in _SENSITIVE]`.
- **Why:** the 13 pending cases are `xfail(strict=True)`. Once the lines land they XPASS and
  fail on purpose, so the marker cannot be forgotten.
- **Verified:** the exact lines above make all 13 cases pass, with the same matchers. They keep
  `.env.example`, `.gitleaks.toml`, `ignores.toml`, `pyproject.toml`, `uv.lock` and the package
  manifests visible.

Optional, owner's call: `.idea/` is neither git- nor docker-ignored. IDE data-source files can
hold DB passwords. It is scanned (clean today); adding `.idea/` to `.gitignore` is the usual
choice.

### 2. Dependency bumps (uv)
```bash
uv lock --upgrade-package certifi --upgrade-package streamlit --upgrade-package pillow
uv sync --all-packages --all-extras
```
Expected: certifi 2026.10.19, streamlit 1.66.1, pillow 13.0.0. `packages/ui/pyproject.toml`
(`streamlit>=1.66,<2`) needs no change.

### 3. Dockerfile: base image to trixie, drop the unused pip
```dockerfile
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
```
(the multi-arch index digest pulled 2026-10-22). In the `runtime` stage, before `useradd`:
```dockerfile
FROM ${PYTHON_IMAGE} AS runtime
# The base image's pip (/usr/local) is never used at runtime (the app runs from /opt/venv, built
# by uv) and carries its own advisories (Trivy: 5 MEDIUM, 1 LOW).
RUN python -m pip uninstall --yes --no-input pip
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
```
After the bump, delete the "Debian 12 (bookworm), CURRENT base" block from
`scripts/security/ignores.toml`. The stale `note` line in `make audit` lists exactly those 8 IDs.
The "present on BOTH" block already includes trixie's extra package names (`liblastlog2-2`,
`login`).

### 4. Makefile targets
Add `audit secrets-scan` to `.PHONY`, and:
```makefile
audit: ## Dependency + image + config scan: pip-audit, Trivy (needs network; build first: make build)
	@./scripts/security/audit.sh

secrets-scan: ## Secret scan of the working tree (gitleaks + RAGForge token rules); never prints a value
	@./scripts/security/secrets.sh
```
Recommended (so `make check` covers the gate's source like the packages):
```makefile
lint: ## Ruff lint + format check
	@uv run ruff check packages scripts/security && uv run ruff format --check packages scripts/security
```
and append `scripts/security/report.py` to the first `uv run mypy` line of `typecheck`.

### 5. Optional dev-group hygiene
The ignore-file test imports `pathspec` (a mypy dependency) and `docker` (a testcontainers
dependency). Both are transitive dev dependencies today. Declaring `pathspec>=1.1` in the dev
group would make that explicit (needs a re-lock, so it's your call).

## What the integrator must test live
1. **Before the bumps:** `make secrets-scan` gives exit 0 and "0 finding(s)". `make audit`
   gives exit 1, with exactly the certifi + pillow pip-audit advisories and 1 blocking Trivy
   finding (certifi). This proves the gate is not vacuous.
2. Apply steps 1-3, `make build`, then `make audit` gives exit 0. Expected: pip-audit 0; Trivy
   image 0 CRITICAL and about 44 HIGH, all `ignored`; the config scan has 1 LOW.
3. `make up` + `curl localhost:8000/readyz` on the trixie image. Then one chat turn and one
   Notion sync: glibc and OpenSSL move to Debian 13, and the psycopg/flashrank wheels must load.
4. Full `make check`: the 13 xfails are gone, and the
   `test_sec_scan_gitleaks_rules.py::test_real_gitleaks_agrees_with_the_model` integration test
   runs (Docker needed).
5. `docker volume ls | grep ragforge-sec` after `make audit`: only `ragforge-sec-trivy-cache`
   remains (the input volume is removed even when a step fails).

## Known gaps (honest)
- **compose.yaml is not misconfig-scanned:** Trivy 0.75 has no Compose scanner. The file is
  mounted, but no checks apply. The project's own compose tests (no `env_file`, the secret
  scoping) remain the guard.
- **pip-audit matches by name and version** against the PyPI advisory feed. Environment
  markers are stripped, so Linux/Windows-only pins (`watchdog`, `pywin32`) are audited too.
  Packages not on PyPI (the workspace members) are out of scope by construction.
- **The Python model of gitleaks is partial.** In `test_sec_scan_gitleaks_rules.py` it models
  only what the custom rules use (regex, secretGroup, entropy, keywords, global stopwords). The
  path/targetRules allowlists are checked as regexes. The integration test is the ground truth
  against the real binary.
- **`test_shipped_ignore_file_is_valid_and_not_expired` turns red on 2027-02-01** by design.
  That is the review date of every ignore, and the test forces the re-review. Rescan, then
  re-justify or delete.
- **The scans need the network** (advisory feeds, the Trivy DB and checks bundle). The first
  Trivy run downloads the DB (about 1.5 GB unpacked) into the cache volume. The tarball
  volume needs about 500 MB more for the duration of one image scan.
- **Once git exists:** run `gitleaks git` once over the full history (the pre-first-commit
  checklist in SECURITY.md). `secrets.sh` can then also serve as a pre-push hook.
