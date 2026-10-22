# ADR-0008: psycopg 3 as the only Postgres driver (two pools)

**Status:** Accepted · 2026-10-16 · Revised in round 3: "one pool" was overstated

## Context
LangGraph's `AsyncPostgresSaver` and `AsyncPostgresStore` use psycopg 3 and need connections configured with `autocommit=True` and `row_factory=dict_row`. SQLAlchemy needs transactional connections.

## Decision
- **One driver:** psycopg 3 everywhere (SQLAlchemy URL `postgresql+psycopg://`).
- **Two pools** per process:
  1. SQLAlchemy async engine (transactional; `set_config('app.tenant_id', $1, true)` per transaction);
  2. `psycopg_pool.AsyncConnectionPool` for LangGraph (autocommit, dict_row).
- **Plus one dedicated LISTEN connection** per API pod and per worker.
- pgvector type adapters are registered on both pools in `bootstrap.py`.
- **Connection budget:** `max_connections ≥ pods × (sa_pool + lg_pool + 1) + migrate + metrics + headroom`. The API defaults are sa_pool = 10, lg_pool = 5.

## FencedPostgresSaver
It subclasses the LangGraph Postgres saver to run `aput`/`aput_writes` in a transaction with `assert_run_lease`. This couples to saver internals, so the LangGraph version is **pinned**, and a **saver contract test** (round-trip, interrupt/resume, writes ordering) runs on every upgrade. `head_checkpoint_id` keeps resume deterministic regardless of checkpoint-ID ordering across pods (uuid6 depends on each pod's clock).

## Consequences
One set of driver semantics and type adapters. Two pools are sized and monitored explicitly.
