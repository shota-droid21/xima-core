from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.utils.pairing import PairingStore


def test_create_challenge_invalidates_previous_active_challenge() -> None:
    with TemporaryDirectory() as tmpdir:
        store = PairingStore(Path(tmpdir))

        first_challenge_id, first_otp, _first_nonce, _first_exp = store.create_challenge(
            ttl_seconds=90,
            invalidate_existing=True,
        )
        assert store.count_active_challenges() == 1

        second_challenge_id, second_otp, _second_nonce, _second_exp = store.create_challenge(
            ttl_seconds=90,
            invalidate_existing=True,
        )
        assert first_challenge_id != second_challenge_id
        assert store.count_active_challenges() == 1

        with pytest.raises(PermissionError):
            store.prove(challenge_id=first_challenge_id, otp_raw=first_otp)

        resolved_challenge_id, _challenge = store.prove(otp_raw=second_otp)
        assert resolved_challenge_id == second_challenge_id
        assert store.count_active_challenges() == 0


def test_prove_without_challenge_id_uses_active_challenge() -> None:
    with TemporaryDirectory() as tmpdir:
        store = PairingStore(Path(tmpdir))

        challenge_id, otp, _nonce, _exp = store.create_challenge(
            ttl_seconds=90,
            invalidate_existing=True,
        )

        resolved_challenge_id, _challenge = store.prove(otp_raw=otp)
        assert resolved_challenge_id == challenge_id
