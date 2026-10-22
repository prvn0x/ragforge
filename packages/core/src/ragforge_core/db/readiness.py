"""Schema/code compatibility for /readyz (ADR-0002). Pure function + one query."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_contracts.api import Readiness
from ragforge_core import CODE_REQUIRED_SCHEMA, CODE_VERSION


def evaluate_readiness(
    schema_version: int | None,
    min_code_version: int | None,
    *,
    code_version: int = CODE_VERSION,
    required_schema: int = CODE_REQUIRED_SCHEMA,
) -> Readiness:
    reason: str | None = None
    if schema_version is None or min_code_version is None:
        reason = "schema_meta missing (migrations not applied)"
    elif schema_version < required_schema:
        reason = f"schema {schema_version} is behind required {required_schema}"
    elif code_version < min_code_version:
        reason = f"code {code_version} is older than min_code_version {min_code_version}"
    return Readiness(
        ready=reason is None,
        schema_version=schema_version,
        required_schema=required_schema,
        code_version=code_version,
        min_code_version=min_code_version,
        reason=reason,
    )


async def check_readiness(engine: AsyncEngine, timeout_s: float = 2.0) -> Readiness:
    """Bounded by `timeout_s` (below the 3 s probe timeout) so a hung DB reads as not-ready."""

    async def _read() -> tuple[int, int] | None:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT schema_version, min_code_version FROM rf.schema_meta")
                )
            ).one_or_none()
        return None if row is None else (int(row.schema_version), int(row.min_code_version))

    try:
        result = await asyncio.wait_for(_read(), timeout=timeout_s)
    except Exception as exc:  # DB unreachable or schema missing -> not ready, never raise
        return evaluate_readiness(None, None).model_copy(
            update={"reason": f"database check failed: {type(exc).__name__}"}
        )
    if result is None:
        return evaluate_readiness(None, None)
    return evaluate_readiness(*result)
