import os

import pytest

from ragforge_core.adapters.crypto.envelope import (
    EnvelopeSecretBox,
    SecretDecryptionError,
    load_kek,
)
from ragforge_core.domain.ports import SecretBox

AAD = b"connector:tenant-a:conn-1"


@pytest.fixture
def box() -> EnvelopeSecretBox:
    return EnvelopeSecretBox(os.urandom(32))


def test_round_trip_and_port(box: EnvelopeSecretBox) -> None:
    assert isinstance(box, SecretBox)
    sealed = box.seal(b"secret_notion_token", associated_data=AAD)
    assert b"secret_notion_token" not in sealed.ciphertext + sealed.wrapped_dek
    assert box.open(sealed, associated_data=AAD) == b"secret_notion_token"


def test_each_seal_uses_fresh_dek_and_nonce(box: EnvelopeSecretBox) -> None:
    a, b = box.seal(b"x", associated_data=AAD), box.seal(b"x", associated_data=AAD)
    assert a.ciphertext != b.ciphertext
    assert a.wrapped_dek != b.wrapped_dek


def test_ciphertext_bound_to_its_owner(box: EnvelopeSecretBox) -> None:
    sealed = box.seal(b"t", associated_data=AAD)
    with pytest.raises(SecretDecryptionError):
        box.open(sealed, associated_data=b"connector:tenant-b:conn-1")


def test_tampering_detected(box: EnvelopeSecretBox) -> None:
    sealed = box.seal(b"t", associated_data=AAD)
    flipped = bytearray(sealed.ciphertext)
    flipped[-1] ^= 1
    with pytest.raises(SecretDecryptionError):
        box.open(type(sealed)(bytes(flipped), sealed.wrapped_dek), associated_data=AAD)


def test_wrong_kek_fails(box: EnvelopeSecretBox) -> None:
    sealed = box.seal(b"t", associated_data=AAD)
    with pytest.raises(SecretDecryptionError):
        EnvelopeSecretBox(os.urandom(32)).open(sealed, associated_data=AAD)


def test_kek_rotation_rewraps_without_reencrypting(box: EnvelopeSecretBox) -> None:
    sealed = box.seal(b"t", associated_data=AAD)
    new_kek = os.urandom(32)
    rotated = box.rewrap(sealed, new_kek, associated_data=AAD)
    assert rotated.ciphertext == sealed.ciphertext
    assert EnvelopeSecretBox(new_kek).open(rotated, associated_data=AAD) == b"t"


def test_kek_validation() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        load_kek("c2hvcnQ=")
    assert len(load_kek("A" * 43 + "=")) == 32
