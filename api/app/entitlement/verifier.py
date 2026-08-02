"""エンタイトルメント検証（共通・不変）。

署名 ＋ exp ＋ iss を検証する。信頼公開鍵と発行者は Source（差し替え点）が供給する。
発行元を差し替えても本ロジックは変えない（docs/14 / Decision 030）。
既存 auth_jwt と同じ EdDSA(Ed25519) / PyJWT を再利用（新暗号スタックは足さない）。
"""

from __future__ import annotations

from typing import Any, Optional

import jwt

from .status import EntitlementStatus


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class EntitlementVerifier:
    def __init__(self, *, algorithms: tuple[str, ...] = ("EdDSA",)) -> None:
        self._algorithms = list(algorithms)

    def verify(
        self, token: str, *, public_key: str, expected_issuer: str
    ) -> EntitlementStatus:
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=self._algorithms,
                issuer=expected_issuer,
                options={"require": ["exp", "iat", "iss"]},
            )
        except jwt.ExpiredSignatureError:
            # 署名は有効だが exp 切れ。exp を拾って expired として返す。
            return self._expired(token, public_key=public_key, expected_issuer=expected_issuer)
        except jwt.InvalidTokenError:
            return EntitlementStatus.invalid()

        features = payload.get("features")
        if not isinstance(features, list):
            features = []
        return EntitlementStatus(
            state="valid",
            plan=payload.get("plan"),
            features=[str(f) for f in features],
            exp=_safe_int(payload.get("exp")),
            issuer=payload.get("iss"),
        )

    def _expired(
        self, token: str, *, public_key: str, expected_issuer: str
    ) -> EntitlementStatus:
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=self._algorithms,
                issuer=expected_issuer,
                options={"require": ["exp", "iat", "iss"], "verify_exp": False},
            )
        except jwt.InvalidTokenError:
            # 署名や iss が不正なら invalid（expired ではない）。
            return EntitlementStatus.invalid()
        return EntitlementStatus(state="expired", exp=_safe_int(payload.get("exp")))
