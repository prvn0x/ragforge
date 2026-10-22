# RAGForge: working agreement for any Claude session

Personal portfolio project (multi-tenant RAG over Notion). The owner wants Staff/Principal-level quality and judges work the way a principal engineer with 25 years at a large company would. **Read `docs/NEXT.md` first** (plan + how parallel phases are run), then `docs/PHASE1.md`.

## Non-negotiables (learned the hard way in earlier sessions)
- **Never self-score.** Self-review overrated the design by about 4 points (9.4 vs 5.5 blind). Quality scores come only from a fresh **blind** reviewer subagent that doesn't read `docs/REVIEW.md`. Fix every P0/P1 and re-review until there are none left. Report scores honestly, including bad ones.
- **Verify by running, not by reasoning.** Every phase ends with `make check` green (lint, `mypy --strict` including module tests, import-linter, full pytest) **and** a real `make up` + `curl localhost:8000/readyz`. Phase 0's first real run found 5 bugs that tests and reviews had missed.
- **Never claim something works or is installed without checking it.** (An earlier session wrongly claimed the extras were installed.)
- **Frozen interfaces** (`domain/**`, `contracts/**`) change only deliberately, with every caller updated in the same change.
- **Never call a model (embedder/LLM) while a DB transaction is open.** SHA-256 values are raw `bytes`. IDs come only from `domain/ids.py`. Every `SECURITY DEFINER` function follows ADR-0013.
- **Git:** none until the project is finished; the owner will do it. Don't run `git init` and don't commit.
- **Secrets:** never ask the owner to paste a token into chat. Use a hidden-input prompt command they run themselves (`! make connect-notion`).
- **Cost:** say the cost and wait for a yes before parallel agents or workflows. The owner's plan limits are real.

## Environment gotchas (this machine: M1 Pro 16 GB, Rancher Desktop moby 5 GB VM, host Ollama)
- testcontainers needs `TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock` (set in `packages/core/tests/conftest.py`).
- **Don't** add `extra_hosts: host-gateway`. Rancher and Docker Desktop already resolve `host.docker.internal` to the Mac, and the override points at the VM instead, which breaks Ollama.
- The migrator has `INHERIT FALSE`: any query it runs needs `SET ROLE rf_system` first.
- Editing an applied migration → `ragforge-migrate` fails on the checksum guard. In dev, run `make nuke && make up`.
- `db/bootstrap.sql` is a psycopg `sql.SQL` template: no literal braces, even in comments.
- Named volumes are seeded with the image directory's owner: create mount points owned by uid 10001 in the Dockerfile.
- Run ruff only on the paths you're changing when other agents may be editing; mypy checks module test dirs one at a time (see `make typecheck`).
- Pydantic smart unions prefer `str` over enums; open enums use `union_mode="left_to_right"`.

## Commands
`make doctor` · `make up` · `make check` · `make test` · `make nuke` (wipes dev data) · `make psql` · `make models`
