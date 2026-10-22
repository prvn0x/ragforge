"""Scope policy: what a key may do and which scopes a new key may receive.

Invariants that live here (the first is also enforced in the database, migration 0006):
  - MCP keys never hold `approvals:write` or `admin` (DESIGN section 10): content injected into an
    agent's context must not be able to approve the agent's own destructive tool calls.
  - No escalation: a key can only mint keys whose scopes are a subset of its own.
  - Privileged scopes on API-minted keys fail CLOSED: a key name must positively declare a human
    client before it may receive `approvals:write` or `admin` (see `validate_new_key_scopes`).
  - `eval` (ADR-0017 §1) lives only on a dedicated eval key: never on an MCP key, and never
    beside anything but `chat` and `read` (`check_eval_scopes`). MCP keys also lose it at
    authentication (`effective_scopes`), like the human-only scopes.

Two levels of administration (docs/handoff/E2.md, section 5):
  - TENANT self-administration (mint/list/revoke keys, read the audit log) needs a "key manager"
    key: one holding `approvals:write` or `admin`. Both are human-only scopes (never on MCP keys),
    so an agent can never mint or revoke keys. Provisioned owner keys qualify.
  - PLATFORM provisioning (POST /v1/tenants) needs `admin`, which only the CLI issues, and only
    with an explicit `--platform-admin` flag.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from ragforge_core.domain.errors import Forbidden, ValidationFailed
from ragforge_core.domain.models import Principal, Scope

ALL_SCOPES: frozenset[str] = frozenset(s.value for s in Scope)
MCP_FORBIDDEN_SCOPES: frozenset[str] = frozenset({Scope.approvals_write.value, Scope.admin.value})
KEY_MANAGER_SCOPES: frozenset[str] = MCP_FORBIDDEN_SCOPES
# ADR-0017 §1: the eval key's scopes are eval + chat + read, nothing else. WHY so narrow: the eval
# scope unlocks tokenless grades and per-stage retrieval traces; a key that could also write,
# approve or administer would mix the harness's identity with a human's.
EVAL_KEY_SCOPES: frozenset[str] = frozenset(s.value for s in (Scope.eval, Scope.chat, Scope.read))
# What an MCP key may never hold. Separate from MCP_FORBIDDEN_SCOPES on purpose: that set is also
# KEY_MANAGER_SCOPES (and mirrored by a DB CHECK), and `eval` must never make a key a key manager.
MCP_EXCLUDED_SCOPES: frozenset[str] = MCP_FORBIDDEN_SCOPES | {Scope.eval.value}
OWNER_KEY_SCOPES: frozenset[str] = frozenset(
    s.value for s in (Scope.read, Scope.write, Scope.chat, Scope.approvals_write)
)

# Until the contract carries an explicit client field (requested in docs/handoff/E2.md), the key
# NAME classifies the client, and both rules fail closed:
#   - "mcp" anywhere in the name (case-insensitive substring) marks an MCP key. A false positive
#     only costs the human-only scopes; a false negative would hand an agent approvals:write.
#   - a privileged scope additionally needs a name that positively declares a human client
#     (`ui` / `user`, alone or followed by `:`, `-` or `_`), so names that say nothing about the
#     client ("cursor", "claude-desktop") can never receive approvals:write.
_MCP_NAME_RE = re.compile(r"mcp", re.IGNORECASE)
_HUMAN_NAME_RE = re.compile(r"^(?:ui|user)(?:[:_-]|$)", re.IGNORECASE)
HUMAN_NAME_HINT = "start the key name with `ui:` or `user:`"


class ClientKind(StrEnum):
    user = "user"
    mcp = "mcp"


def client_kind_for_name(name: str) -> ClientKind:
    return ClientKind.mcp if _MCP_NAME_RE.search(name) else ClientKind.user


def declares_human_client(name: str) -> bool:
    return client_kind_for_name(name) is ClientKind.user and bool(
        _HUMAN_NAME_RE.match(name.strip())
    )


def effective_scopes(scopes: Iterable[str], kind: ClientKind) -> frozenset[str]:
    """Scopes a key actually holds. Unknown scope strings are dropped (fail closed), and MCP keys
    lose the forbidden scopes and `eval` even if a row somehow carried them (defense in depth:
    the CLI's trusted path mints keys without the API's checks)."""
    known = frozenset(scopes) & ALL_SCOPES
    return known - MCP_EXCLUDED_SCOPES if kind is ClientKind.mcp else known


def require_scope(principal: Principal, scope: Scope) -> None:
    if not principal.has(scope.value):
        raise Forbidden(f"this API key lacks the `{scope.value}` scope")


def is_key_manager(principal: Principal) -> bool:
    return bool(principal.scopes & KEY_MANAGER_SCOPES)


def require_key_manager(principal: Principal) -> None:
    if not is_key_manager(principal):
        raise Forbidden(
            "managing API keys and reading the audit log needs the `approvals:write` or "
            "`admin` scope"
        )


def check_mcp_scopes(kind: ClientKind, scopes: Iterable[str]) -> None:
    if kind is ClientKind.mcp and frozenset(scopes) & MCP_FORBIDDEN_SCOPES:
        raise ValidationFailed(
            "MCP keys can never hold approvals:write or admin (a human approves in the UI); "
            "a key name containing `mcp` marks an MCP key"
        )


def check_eval_scopes(kind: ClientKind, scopes: Iterable[str]) -> None:
    """ADR-0017 §1: `eval` is never combined with an MCP key, nor with any scope beyond chat and
    read. Raised as 422 at creation, so a misconfigured harness key fails where it is minted."""
    wanted = frozenset(scopes)
    if Scope.eval.value not in wanted:
        return
    if kind is ClientKind.mcp:
        raise ValidationFailed(
            "the eval scope is never granted to an MCP key (a key name containing `mcp` marks "
            "an MCP key)"
        )
    extra = wanted - EVAL_KEY_SCOPES
    if extra:
        raise ValidationFailed(
            f"an eval key holds only eval, chat and read; remove {sorted(extra)}"
        )


def validate_new_key_scopes(
    caller: Principal, requested: Iterable[str], name: str
) -> tuple[frozenset[str], ClientKind]:
    """Scopes and client kind for a key minted through the API (never the CLI's trusted path)."""
    wanted = frozenset(requested)
    kind = client_kind_for_name(name)
    if not wanted:
        raise ValidationFailed("at least one scope is required")
    unknown = wanted - ALL_SCOPES
    if unknown:
        raise ValidationFailed(f"unknown scopes: {sorted(unknown)}")
    check_mcp_scopes(kind, wanted)
    check_eval_scopes(kind, wanted)
    if wanted & MCP_FORBIDDEN_SCOPES and not declares_human_client(name):
        raise ValidationFailed(
            f"approvals:write and admin are only granted to keys that declare a human client: "
            f"{HUMAN_NAME_HINT}"
        )
    if not wanted <= caller.scopes:
        raise Forbidden("a key cannot grant scopes it does not hold itself")
    return wanted, kind
