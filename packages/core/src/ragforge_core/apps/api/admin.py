"""`ragforge-admin`: bootstrap the first tenant and its admin key (no API key exists yet).

    ragforge-admin create-tenant --name acme --owner "Ada" [--kind team --team-consent]
                                 [--platform-admin]
    ragforge-admin create-key --tenant <uuid> --name eval [--scopes read,chat] [--user <uuid>]

The key gets the owner scopes (read, write, chat, approvals:write): enough to run and administer
its own tenant (mint/revoke keys, read the audit log). `--platform-admin` adds `admin`, which
lets the key provision OTHER tenants through POST /v1/tenants; give it to the operator only.

`create-key` mints an extra key for an EXISTING tenant (for scripts and evals) without needing
an existing key: least privilege by default (`read,chat`), and it can never grant
`approvals:write` or `admin` (those stay with keys a human holds). The key belongs to `--user`, or
to the tenant's first user (its owner) when omitted. Audited as `api_key.create` by the CLI.

Prints the key exactly once on stdout; it is never stored or logged. Runs with the API's database
role (rf_api): provisioning goes through the `provision_tenant` definer function like the API.

Lives in apps (not services/auth) because it needs the database and the composition root, which
the import-linter layering forbids for services; the use case itself is
`services.auth.provisioning.provision_tenant`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.exc import DBAPIError

from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.sql import audit, fetch_one, insert_api_key, sqlstate
from ragforge_core.apps.api.stores import SqlProvisioningStore
from ragforge_core.config import Settings, get_settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.domain.errors import NotFound, RagforgeError, ValidationFailed
from ragforge_core.domain.models import Scope
from ragforge_core.services.auth.keys import GeneratedKey, generate_key
from ragforge_core.services.auth.ports import NewTenant
from ragforge_core.services.auth.provisioning import ProvisionResult, provision_tenant
from ragforge_core.services.auth.scopes import OWNER_KEY_SCOPES, client_kind_for_name

PLATFORM_ADMIN_KEY_SCOPES = OWNER_KEY_SCOPES | {Scope.admin.value}
CLI_KEY_SCOPES = frozenset({Scope.read.value, Scope.write.value, Scope.chat.value})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragforge-admin", description="RAGForge administration")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-tenant", help="create a tenant, owner user and admin key")
    create.add_argument("--name", required=True)
    create.add_argument("--owner", required=True, help="owner display name")
    create.add_argument("--kind", choices=("personal", "team"), default="personal")
    create.add_argument(
        "--team-consent",
        action="store_true",
        help="record that team members may see each other's indexed sources (required for team)",
    )
    create.add_argument(
        "--platform-admin",
        action="store_true",
        help="also grant `admin`: the key may provision other tenants (operator only)",
    )
    create.add_argument("--key-name", default="owner")

    key = commands.add_parser("create-key", help="mint a read/chat key for an existing tenant")
    key.add_argument("--tenant", required=True, type=UUID)
    key.add_argument("--name", required=True, help="key name, e.g. eval")
    key.add_argument("--scopes", default="read,chat", help="subset of read,write,chat")
    key.add_argument("--user", type=UUID, help="owner of the key (default: the tenant's owner)")
    return parser


def _cli_scopes(raw: str) -> frozenset[str]:
    scopes = frozenset(s.strip() for s in raw.split(",") if s.strip())
    if not scopes or not scopes <= CLI_KEY_SCOPES:
        raise ValidationFailed(f"--scopes must be a non-empty subset of {sorted(CLI_KEY_SCOPES)}")
    return scopes


async def _create_key(settings: Settings, args: argparse.Namespace) -> tuple[GeneratedKey, UUID]:
    scopes = _cli_scopes(args.scopes)
    engine = make_engine(settings, "api")
    try:
        async with SqlTransactionManager(engine).tenant(args.tenant) as tx:
            conn = connection(tx)
            user = await fetch_one(  # under RLS: a foreign or unknown tenant has no users
                conn,
                "SELECT id FROM rf.users WHERE (CAST(:user AS uuid) IS NULL OR id = :user) "
                "ORDER BY created_at, id LIMIT 1",
                {"user": args.user},
            )
            if user is None:
                raise NotFound("no such tenant or user")
            new_key = generate_key()
            row = await insert_api_key(
                conn,
                tenant_id=args.tenant,
                user_id=user["id"],
                name=args.name,
                key=new_key,
                scopes=scopes,
                client_kind=client_kind_for_name(args.name),
            )
            await audit(
                conn, "cli:ragforge-admin", "api_key.create", str(row["id"]),
                {"prefix": new_key.prefix, "scopes": sorted(scopes)}, tenant_id=args.tenant,
            )  # fmt: skip
        return new_key, user["id"]
    finally:
        await engine.dispose()


async def _create_tenant(settings: Settings, args: argparse.Namespace) -> ProvisionResult:
    engine = make_engine(settings, "api")
    try:
        return await provision_tenant(
            SqlProvisioningStore(SqlTransactionManager(engine)),
            NewTenant(
                name=args.name,
                kind=args.kind,
                owner_display_name=args.owner,
                team_visibility_consent=args.team_consent,
            ),
            actor="cli:ragforge-admin",
            owner_scopes=PLATFORM_ADMIN_KEY_SCOPES if args.platform_admin else OWNER_KEY_SCOPES,
            key_name=args.key_name,
        )
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = settings or get_settings()
    try:
        if args.command == "create-key":
            new_key, user_id = asyncio.run(_create_key(settings, args))
            print(f"user_id={user_id}")
            print(f"api_key={new_key.plaintext}")
            print("Store this key now: it cannot be shown again.", file=sys.stderr)
            return 0
        result = asyncio.run(_create_tenant(settings, args))
    except RagforgeError as exc:
        print(f"error: {exc.title}: {exc}", file=sys.stderr)
        return 2
    except DBAPIError as exc:
        # Never print the driver message: it includes bound parameters (prefix, key digest).
        print(f"error: database error (SQLSTATE {sqlstate(exc) or 'unknown'})", file=sys.stderr)
        return 3
    print(f"tenant_id={result.tenant_id}")
    print(f"user_id={result.user_id}")
    print(f"scopes={','.join(sorted(result.scopes))}")
    print(f"api_key={result.key.plaintext}")
    print("Store this key now: it cannot be shown again.", file=sys.stderr)
    return 0


def run() -> None:
    sys.exit(main())
