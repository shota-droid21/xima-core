"""学習/推論スクリプト共通のデバイス選択。

cpu / cuda / mps(metal) の判定は train_epoch と infer_heads で同一だったため、
embed_images の追加にあたって重複を増やさないよう共通化した（挙動は変更なし）。
"""

from __future__ import annotations

from typing import Optional

import torch


def _normalize_device_name(force: Optional[str]) -> Optional[str]:
    if not force:
        return None
    name = force.lower().strip()
    if name == "metal":
        return "mps"
    return name or None


def _mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    checker = getattr(mps, "is_available", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def get_device(force: Optional[str] = None) -> torch.device:
    forced = _normalize_device_name(force)
    if forced:
        if forced == "cpu":
            return torch.device("cpu")
        if forced == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if forced == "mps":
            return torch.device("mps" if _mps_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
