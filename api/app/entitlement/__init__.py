"""エンタイトルメント最小実装（U-3）。設計正本: docs/design/entitlement.md。"""

from __future__ import annotations

from .registry import build_source
from .router import create_entitlement_router
from .service import resolve_entitlement
from .status import (
    STATE_EXPIRED,
    STATE_INVALID,
    STATE_NONE,
    STATE_VALID,
    EntitlementStatus,
)
from .verifier import EntitlementVerifier

__all__ = [
    "EntitlementStatus",
    "EntitlementVerifier",
    "build_source",
    "create_entitlement_router",
    "resolve_entitlement",
    "STATE_NONE",
    "STATE_VALID",
    "STATE_EXPIRED",
    "STATE_INVALID",
]
