"""ジョブ実行 backend の選択（`XIMA_JOB_BACKEND`）。

既定は `local`（in-process・Docker 不要）。未知の値でも `local` にフォールバックする
（クラッシュさせない。`entitlement/registry.py` と同じ方針）。

`celery` は import 時に redis / celery を解決するため、選択されたときだけ import する。
"""

from __future__ import annotations

import os

from .base import JobBackend
from .local import LocalBackend
from ...config import ConfigManager

__all__ = ["JobBackend", "LocalBackend", "build_backend"]

BACKEND_LOCAL = "local"
BACKEND_CELERY = "celery"


def build_backend(
    config_manager: ConfigManager,
    *,
    inspect_timeout_seconds: float = 1.0,
) -> JobBackend:
    kind = os.environ.get("XIMA_JOB_BACKEND", BACKEND_LOCAL).strip().lower()
    if kind == BACKEND_CELERY:
        # 遅延 import: local 既定の環境に celery/redis の解決を強制しない。
        from .celery import CeleryBackend

        return CeleryBackend(inspect_timeout_seconds=inspect_timeout_seconds)
    return LocalBackend(config_manager)
