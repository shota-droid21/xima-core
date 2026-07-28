"""最小ローカル認証（local モード）のシークレット管理と検証。

設計正本: docs/13_local_auth_and_launch.md。

- 起動時に暗号学的乱数のシークレットを生成し、0600 ファイルに保存（再起動で再利用）。
- app↔core の必須ヘッダ `X-Xima-Local-Key` の検証に用いる。
- 信頼アンカーはファイル権限（別ユーザーを防ぐ）＋ loopback バインド＋CORS app 限定。
  同一ユーザーの別プロセスは脅威モデル対象外（docs/13 参照）。
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

# app↔core ハンドシェイクの必須ヘッダ名（local モード）。
LOCAL_KEY_HEADER = "X-Xima-Local-Key"

_SECRET_FILENAME = ".local_secret"
_SECRET_NBYTES = 32


def get_or_create_local_secret(state_dir: Path) -> str:
    """state_dir 配下のシークレットを取得（無ければ 0600 で生成）。"""
    path = state_dir / _SECRET_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except OSError:
        pass

    # 生成: token_urlsafe 相当を os.urandom から。依存を増やさない。
    import base64

    secret = base64.urlsafe_b64encode(os.urandom(_SECRET_NBYTES)).rstrip(b"=").decode("ascii")

    state_dir.mkdir(parents=True, exist_ok=True)
    # 0600 で作成（既存も truncate）。umask の影響を避けるため明示権限で open。
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, secret.encode("ascii"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return secret


def verify_local_key(provided: str | None, expected: str) -> bool:
    """定数時間比較で必須ヘッダ値を検証する。"""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)
