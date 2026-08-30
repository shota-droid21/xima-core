"""エンタイトルメントの発行元（差し替え点）。

トークンの取得元と信頼公開鍵を供給する。今は LocalFileSource（ローカル署名キー）、
将来は ManagerSource（control plane 発行）を追加するだけで済む。検証ロジック
（verifier.py）と app 側状態モデルは不変（docs/design/entitlement.md / Decision 030）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class EntitlementSource(Protocol):
    #: 信頼する発行者（JWT の iss と一致すること）。
    issuer: str

    def load_token(self) -> Optional[str]:
        """署名済みトークン文字列を返す（未所持なら None）。"""
        ...

    def trusted_public_key(self) -> Optional[str]:
        """検証に用いる信頼公開鍵 PEM を返す（未配置なら None）。"""
        ...


def _read_text(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


class LocalFileSource:
    """ローカルの署名済みキーファイルと信頼公開鍵ファイルから供給する。"""

    issuer = "xima-local"

    def __init__(self, *, token_path: Path, public_key_path: Path) -> None:
        self._token_path = token_path
        self._public_key_path = public_key_path

    def load_token(self) -> Optional[str]:
        return _read_text(self._token_path)

    def trusted_public_key(self) -> Optional[str]:
        return _read_text(self._public_key_path)
