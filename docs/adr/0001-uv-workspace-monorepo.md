# ADR-0001: uv workspace monorepo with 5 packages

**Status:** Accepted · 2026-10-16 · Revised in round 3: 7 → 5 packages

## Context
There are four runtime roles (api, worker, mcp, ui) plus an SDK and evals. One engineer maintains it. The API and worker need the same heavy dependencies, so splitting them gains nothing. MCP and UI have very different dependencies (FastMCP vs Streamlit) and must never touch the DB.

## Options
| Option | Pros | Cons |
|---|---|---|
| Single package | Simplest | Fat UI/MCP images; nothing stops the UI from importing the DB layer |
| 7 packages (v0.2) | Very fine-grained | Ceremony with no dependency benefit (api and worker share deps) |
| **5 packages: contracts, core (incl. api + worker apps), client, mcp, ui** | Real boundary where it matters (DB vs no DB); 3 images | Layering *inside* core needs import-linter |

## Decision
- 5 packages and one `uv.lock`.
- `core` exposes console scripts `ragforge-api`, `ragforge-worker` and `ragforge-migrate`, which all ship in the single `server` image.
- import-linter enforces two rules: `{mcp, ui, evals}` cannot import `core`, and inside `core` the layers are `apps → agent → services/rag → domain`, with `adapters → domain` only.

## Revisit when
The API and worker diverge in dependencies or release cadence. Split `apps/worker` into its own package then.
