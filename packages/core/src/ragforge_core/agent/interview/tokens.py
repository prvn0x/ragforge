"""`question_token`: a server-signed binding of one /question response to its principal.

WHY: the session is client-held, so /grade receives the question's chunk ids, expected points,
kind and difficulty from the client. Re-loading the chunks through RLS already stops a client
from grading against notes it cannot see; the token additionally lets the server tell "the
question I issued to you" from "a question you composed" (say, expected points rewritten to make
a weak answer look complete). /grade checks it WHEN present; enforcing it is a later, separate
change (clients must ship first).

Format: `v3.<issued_at>.<mac>`, mac = HMAC-SHA256 over a canonical JSON array of the version,
issued_at, tenant, user, the collection ids (sorted, deduplicated), chunk ids, the question text and
expected points (both as sent to the client, i.e. sanitized), kind, difficulty, the RESOLVED level
and the source. v2 added the question and the collections: without them a client could keep a
valid token and swap in an easier question that its answer fits. v3 (Phase 3c) added the level and
the source: without them a Staff question could be graded at the SDE-1 bar, or a notes question
graded as general knowledge (no verification) under a valid token. For source=general the token is
the ONLY binding of the question and its key (there are no notes to re-read), so the contract
requires it there. Older versions are rejected as malformed (400). No payload travels in the
token: the request carries every bound value, so the token stays short (the contract caps it at 2000
chars) and reveals nothing.

Key: derived with HKDF-SHA256 from the server's key-encryption key under a purpose label, so the
KEK itself never signs anything and a token key can never decrypt a connector secret. Rotating
the KEK invalidates outstanding tokens, which costs at most one regenerated question.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ragforge_core.domain.errors import Forbidden, RagforgeError

VERSION: Final = "v3"
_PURPOSE: Final = b"ragforge/interview/question-token/v1"
DEFAULT_TTL_S: Final = 24 * 3600
CLOCK_SKEW_S: Final = 60


class MalformedQuestionToken(RagforgeError):
    status = 400
    title = "Bad request"


class InvalidQuestionToken(Forbidden):
    """Forged, expired, or issued for another question or principal (one error for all, so
    the response does not say which binding failed)."""

    title = "Invalid question token"


@dataclass(frozen=True, slots=True)
class QuestionBinding:
    tenant_id: UUID
    user_id: UUID
    collection_ids: Sequence[UUID]  # order and duplicates do not matter (a set on the wire)
    chunk_ids: Sequence[UUID]
    question: str
    expected_points: Sequence[str]
    kind: str
    difficulty: str
    level: str  # the RESOLVED level (never None: /question resolves it before signing)
    source: str

    def message(self, issued_at: int) -> bytes:
        body = [
            VERSION,
            issued_at,
            str(self.tenant_id),
            str(self.user_id),
            sorted({str(c) for c in self.collection_ids}),
            [str(c) for c in self.chunk_ids],
            self.question,
            list(self.expected_points),
            self.kind,
            self.difficulty,
            self.level,
            self.source,
        ]
        return json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode()


class QuestionTokens:
    def __init__(
        self,
        key: bytes,
        *,
        ttl_s: int = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(key) < 32:
            raise ValueError("question token key must be at least 32 bytes")
        self._key = key
        self._ttl_s = ttl_s
        self._clock = clock

    @classmethod
    def from_secret(
        cls,
        secret: bytes,
        *,
        ttl_s: int = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> QuestionTokens:
        """A purpose-bound subkey of a server secret (HKDF-SHA256, RFC 5869)."""
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_PURPOSE).derive(secret)
        return cls(key, ttl_s=ttl_s, clock=clock)

    def __repr__(self) -> str:  # never the key
        return f"QuestionTokens(ttl_s={self._ttl_s})"

    def issue(self, binding: QuestionBinding) -> str:
        issued_at = int(self._clock())
        return f"{VERSION}.{issued_at}.{self._mac(binding, issued_at)}"

    def verify(self, token: str, binding: QuestionBinding) -> None:
        """Raise unless `token` was issued by this server for exactly `binding`, recently."""
        version, issued_at, mac = _parse(token)
        if version != VERSION:
            raise MalformedQuestionToken("unknown question token version")
        now = self._clock()
        fresh = issued_at - CLOCK_SKEW_S <= now <= issued_at + self._ttl_s
        if not (hmac.compare_digest(mac, self._mac(binding, issued_at)) and fresh):
            raise InvalidQuestionToken("the question token does not match this question")

    def _mac(self, binding: QuestionBinding, issued_at: int) -> str:
        digest = hmac.new(self._key, binding.message(issued_at), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _parse(token: str) -> tuple[str, int, str]:
    parts = token.split(".")
    if len(parts) != 3 or not (parts[1].isascii() and parts[1].isdigit()) or len(parts[1]) > 12:
        raise MalformedQuestionToken("malformed question token")
    try:
        raw = base64.urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4))
    except (binascii.Error, ValueError):
        raise MalformedQuestionToken("malformed question token") from None
    if len(raw) != hashlib.sha256().digest_size:
        raise MalformedQuestionToken("malformed question token")
    return parts[0], int(parts[1]), parts[2]
