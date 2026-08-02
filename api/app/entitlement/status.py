"""エンタイトルメントの状態モデル。

設計正本: docs/14_entitlement.md。core は判定（この状態）を返すだけで、実際の
ロックは app 側（プレースホルダ）が行う（ソフト強制）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STATE_NONE = "none"  # 未所持（トークン / 公開鍵が無い。エラーではない）
STATE_VALID = "valid"  # 署名・iss・exp すべて有効
STATE_EXPIRED = "expired"  # 署名は有効だが exp 切れ
STATE_INVALID = "invalid"  # 署名不正・改竄・iss 不一致


@dataclass
class EntitlementStatus:
    state: str
    plan: Optional[str] = None
    features: List[str] = field(default_factory=list)
    exp: Optional[int] = None
    issuer: Optional[str] = None

    @classmethod
    def none(cls) -> "EntitlementStatus":
        return cls(state=STATE_NONE)

    @classmethod
    def invalid(cls) -> "EntitlementStatus":
        return cls(state=STATE_INVALID)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"state": self.state}
        if self.state == STATE_VALID:
            data["plan"] = self.plan
            data["features"] = self.features
            data["issuer"] = self.issuer
        if self.exp is not None:
            data["exp"] = self.exp
        return data
