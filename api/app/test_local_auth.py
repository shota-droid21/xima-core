"""local_auth（シークレット生成・検証）の単体テスト。dependency-free。"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from app.local_auth import get_or_create_local_secret, verify_local_key


def test_secret_created_with_0600_and_reused(tmp_path: Path) -> None:
    secret1 = get_or_create_local_secret(tmp_path)
    assert secret1
    path = tmp_path / ".local_secret"
    assert path.exists()

    mode = stat.S_IMODE(os.stat(path).st_mode)
    # 所有者のみ read/write（POSIX）。Windows では権限モデルが異なるため緩和。
    if os.name == "posix":
        assert mode == 0o600, oct(mode)

    # 再取得は同じ値（再起動で再利用）。
    secret2 = get_or_create_local_secret(tmp_path)
    assert secret2 == secret1


def test_secret_is_high_entropy_urlsafe(tmp_path: Path) -> None:
    secret = get_or_create_local_secret(tmp_path)
    # base64 urlsafe（padding 無し）: 32 bytes → 43 文字。
    assert len(secret) >= 40
    assert all(c.isalnum() or c in "-_" for c in secret)


def test_verify_local_key_constant_time_semantics(tmp_path: Path) -> None:
    secret = get_or_create_local_secret(tmp_path)
    assert verify_local_key(secret, secret) is True
    assert verify_local_key("wrong", secret) is False
    assert verify_local_key(None, secret) is False
    assert verify_local_key("", secret) is False
    assert verify_local_key(secret, "") is False
