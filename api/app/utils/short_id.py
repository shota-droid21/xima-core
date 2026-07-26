from __future__ import annotations

import re
import secrets

SHORT_ID_RE = re.compile(r"^[a-z0-9]{8}$")
_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_BASE = 36
_LENGTH = 8
_MAX = _BASE**_LENGTH


def is_valid_short_id(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return SHORT_ID_RE.fullmatch(value.strip().lower()) is not None


def require_short_id(value: str, *, label: str) -> str:
    v = (value or "").strip().lower()
    if not v:
        raise ValueError(f"{label} is required")
    if not SHORT_ID_RE.fullmatch(v):
        raise ValueError(f"Invalid {label}: must match [a-z0-9]{{8}}")
    return v


def require_workspace_id(value: str) -> str:
    return require_short_id(value, label="workspace id")


def require_experiment_id(value: str) -> str:
    return require_short_id(value, label="experiment id")


def _to_base36(n: int) -> str:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, _BASE)
        out.append(_ALPHABET[r])
    return "".join(reversed(out))


def generate_short_id() -> str:
    n = secrets.randbelow(_MAX)
    s = _to_base36(n).rjust(_LENGTH, "0")
    return s
