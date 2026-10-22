"""RAGForge server core. Layering is enforced by import-linter (see root pyproject.toml):

apps -> bootstrap -> agent -> services | rag -> domain;  adapters -> domain only.
"""

# Monotonic integers used by /readyz (ADR-0002). Bump CODE_REQUIRED_SCHEMA whenever code
# starts depending on a newer migration; bump CODE_VERSION on every release.
CODE_VERSION = 1
CODE_REQUIRED_SCHEMA = 14
