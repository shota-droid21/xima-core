"""GET /entitlement — 現在のエンタイトルメント状態を返す。

local モードでは他のデータ経路と同様に必須ヘッダ（X-Xima-Local-Key）で保護される。
Source は毎リクエスト構築する（env の差し替えを即時反映するため。処理は軽い）。
"""

from __future__ import annotations

from fastapi import APIRouter

from .registry import build_source
from .service import resolve_entitlement
from .verifier import EntitlementVerifier


def create_entitlement_router(config_manager) -> APIRouter:
    router = APIRouter()
    verifier = EntitlementVerifier()

    @router.get("/entitlement")
    def get_entitlement() -> dict:
        source = build_source(config_manager)
        status = resolve_entitlement(source, verifier)
        return status.to_dict()

    return router
