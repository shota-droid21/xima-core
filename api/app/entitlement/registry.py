"""発行元の選択（XIMA_ENTITLEMENT_SOURCE）。

既定は local。未知の値でも local にフォールバック（ソフト強制。クラッシュしない）。
パスは env で上書き可（テスト / 配布での差し替え用）。
"""

from __future__ import annotations

import os
from pathlib import Path

from .source import EntitlementSource, LocalFileSource

# 既定の配置。base_dir は core/api なので core/.keys は base_dir.parent/.keys。
_DEFAULT_TOKEN_NAME = "entitlement.jwt"
_DEFAULT_PUBLIC_KEY_NAME = "entitlement_ed25519_public.pem"


def _token_path(config_manager) -> Path:
    override = os.environ.get("XIMA_ENTITLEMENT_TOKEN_PATH")
    if override:
        return Path(override)
    return config_manager.state_dir / _DEFAULT_TOKEN_NAME


def _public_key_path(config_manager) -> Path:
    override = os.environ.get("XIMA_ENTITLEMENT_PUBLIC_KEY_PATH")
    if override:
        return Path(override)
    return config_manager.base_dir.parent / ".keys" / _DEFAULT_PUBLIC_KEY_NAME


def build_source(config_manager) -> EntitlementSource:
    kind = os.environ.get("XIMA_ENTITLEMENT_SOURCE", "local").strip().lower()
    # 現状は local のみ。将来 manager を分岐で追加する（verifier は共通のまま）。
    _ = kind  # 未知値も local にフォールバック
    return LocalFileSource(
        token_path=_token_path(config_manager),
        public_key_path=_public_key_path(config_manager),
    )
