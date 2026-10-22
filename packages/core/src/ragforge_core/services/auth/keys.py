"""API key format, generation and hashing.

Format: ``rf_<prefix>_<secret>``
  - ``prefix``: 16 lowercase hex chars. Public, unique, indexed: it selects the row without
    revealing anything about the secret. Hex (not urlsafe base64) so it never contains ``_``
    and the key splits unambiguously.
  - ``secret``: ``secrets.token_urlsafe(32)`` (256 bits). May contain ``_`` and ``-``.

Only ``sha256(full key)`` is stored (raw 32 bytes, docs/PHASE1.md). Hashing the FULL key (not just
the secret) binds the digest to its prefix, so a digest copied onto another row never verifies.
A fast unsalted hash is correct here: the secret has 256 bits of entropy, so there is nothing to
brute-force, and a slow KDF would put a CPU-bound step on every request.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass

KEY_SCHEME = "rf"
PREFIX_BYTES = 8  # 16 hex chars
SECRET_BYTES = 32

_KEY_RE = re.compile(r"^rf_(?P<prefix>[0-9a-f]{16})_(?P<secret>[A-Za-z0-9_-]{43})$")


@dataclass(frozen=True, slots=True)
class ParsedKey:
    prefix: str
    sha256: bytes


@dataclass(frozen=True, slots=True)
class GeneratedKey:
    """A freshly minted key. `plaintext` is shown to the caller exactly once, never stored."""

    prefix: str
    plaintext: str
    sha256: bytes

    def __repr__(self) -> str:  # keep the secret out of tracebacks and debug logs
        return f"GeneratedKey(prefix={self.prefix!r}, plaintext='***')"


def hash_key(plaintext: str) -> bytes:
    return hashlib.sha256(plaintext.encode("ascii")).digest()


def generate_key() -> GeneratedKey:
    prefix = secrets.token_hex(PREFIX_BYTES)
    plaintext = f"{KEY_SCHEME}_{prefix}_{secrets.token_urlsafe(SECRET_BYTES)}"
    return GeneratedKey(prefix=prefix, plaintext=plaintext, sha256=hash_key(plaintext))


def parse_key(raw: str) -> ParsedKey | None:
    """Return the lookup prefix and digest, or None if `raw` is not a well-formed key.

    Rejecting malformed input before the database leaks nothing: the format is public.
    """
    match = _KEY_RE.match(raw)
    if match is None:
        return None
    return ParsedKey(prefix=match["prefix"], sha256=hash_key(raw))
