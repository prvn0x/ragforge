# ADR-0005: MCP server is a thin client of the public API

**Status:** Accepted · 2026-10-16 · Supersedes the v0.1 "MCP calls the service layer directly" choice.

## Options
| Option | Pros | Cons |
|---|---|---|
| MCP → service layer directly | No network hop | Second entry point duplicating authn, tenancy and rate limits; second DB pool; MCP image needs the DB and agent deps |
| **MCP → API via `ragforge-client`** | One authorization choke point; tiny image; MCP behaves exactly like any API client; also dogfoods the SDK | ~1 ms extra hop; the API must be up |

## Decision
Use `ragforge-mcp` (FastMCP; stdio and streamable HTTP), depending only on `ragforge-client`. The MCP client's API key is forwarded as a bearer token.

## Consequences
Every capability exposed over MCP must exist in the API first, so the API stays the complete contract.
