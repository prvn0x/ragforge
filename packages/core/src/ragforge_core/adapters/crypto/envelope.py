"""Envelope encryption (AES-256-GCM) implementing the SecretBox port. Phase 0 owned.

ciphertext  = 0x01 | nonce(12) | AESGCM(DEK).encrypt(plaintext, aad)
wrapped_dek = 0x01 | nonce(12) | AESGCM(KEK).encrypt(DEK, b"dek|" + aad)

The associated data binds a secret to its owner (e.g. b"connector:<tenant>:<connector>"), so a
ciphertext copied onto another row fails to decrypt. The version byte allows KEK rotation via
`rewrap` without touching ciphertexts.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ragforge_core.domain.models import SealedSecret

_VERSION = b"\x01"
_NONCE = 12


class SecretDecryptionError(Exception):
    """Wrong key, wrong associated data, or tampered data. Never includes secret material."""


def load_kek(kek_b64: str) -> bytes:
    try:
        kek = base64.b64decode(kek_b64, validate=True)
    except ValueError as exc:
        raise ValueError("RAGFORGE_KEK must be base64") from exc
    if len(kek) != 32:
        raise ValueError("RAGFORGE_KEK must decode to exactly 32 bytes")
    return kek


def _seal(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(_NONCE)
    return _VERSION + nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def _open(key: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < 1 + _NONCE + 16 or blob[:1] != _VERSION:
        raise SecretDecryptionError("unsupported or truncated envelope")
    try:
        return AESGCM(key).decrypt(blob[1 : 1 + _NONCE], blob[1 + _NONCE :], aad)
    except InvalidTag as exc:
        raise SecretDecryptionError("authentication failed") from exc


class EnvelopeSecretBox:
    def __init__(self, kek: bytes) -> None:
        if len(kek) != 32:
            raise ValueError("KEK must be 32 bytes")
        self._kek = kek

    def seal(self, plaintext: bytes, *, associated_data: bytes) -> SealedSecret:
        dek = AESGCM.generate_key(bit_length=256)
        return SealedSecret(
            ciphertext=_seal(dek, plaintext, associated_data),
            wrapped_dek=_seal(self._kek, dek, b"dek|" + associated_data),
        )

    def open(self, sealed: SealedSecret, *, associated_data: bytes) -> bytes:
        dek = _open(self._kek, sealed.wrapped_dek, b"dek|" + associated_data)
        return _open(dek, sealed.ciphertext, associated_data)

    def rewrap(
        self, sealed: SealedSecret, new_kek: bytes, *, associated_data: bytes
    ) -> SealedSecret:
        """KEK rotation: re-encrypt only the DEK; the ciphertext is unchanged."""
        dek = _open(self._kek, sealed.wrapped_dek, b"dek|" + associated_data)
        return SealedSecret(
            ciphertext=sealed.ciphertext,
            wrapped_dek=_seal(new_kek, dek, b"dek|" + associated_data),
        )
