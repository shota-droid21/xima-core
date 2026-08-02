"""Source → Verifier → Status のオーケストレーション。

トークン / 公開鍵が無い場合は none（クラッシュしない）。ソフト強制。
"""

from __future__ import annotations

from .source import EntitlementSource
from .status import EntitlementStatus
from .verifier import EntitlementVerifier


def resolve_entitlement(
    source: EntitlementSource, verifier: EntitlementVerifier
) -> EntitlementStatus:
    token = source.load_token()
    if not token:
        return EntitlementStatus.none()
    public_key = source.trusted_public_key()
    if not public_key:
        # 検証できない（信頼鍵が無い）→ 証明不能なので none 扱い。
        return EntitlementStatus.none()
    return verifier.verify(
        token, public_key=public_key, expected_issuer=source.issuer
    )
