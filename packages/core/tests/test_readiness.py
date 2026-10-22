"""Unit tests for the schema/code compatibility rule (ADR-0002)."""

import pytest

from ragforge_core.db.readiness import evaluate_readiness


@pytest.mark.parametrize(
    ("schema", "min_code", "code", "required", "ready"),
    [
        (6, 0, 1, 6, True),  # exact
        (9, 0, 1, 6, True),  # newer expanded schema after a rollback: still ready
        (5, 0, 1, 6, False),  # schema behind the code
        (9, 2, 1, 6, False),  # contract migration requires newer code
        (9, 1, 1, 6, True),  # min_code_version equal to code version
        (None, None, 1, 6, False),  # migrations never applied
    ],
)
def test_readiness_matrix(
    schema: int | None, min_code: int | None, code: int, required: int, ready: bool
) -> None:
    r = evaluate_readiness(schema, min_code, code_version=code, required_schema=required)
    assert r.ready is ready
    assert (r.reason is None) is ready
