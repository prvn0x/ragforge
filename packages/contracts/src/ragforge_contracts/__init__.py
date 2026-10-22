"""Public contract of the RAGForge API: request/response schemas and SSE events.

Every client (SDK, MCP server, UI, evals) depends only on this package. The server
validates against the same models, so a breaking change here is a breaking API change.
"""

API_VERSION = "v1"
