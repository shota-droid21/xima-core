from __future__ import annotations

import re
import secrets
import time
import unicodedata
from typing import Optional


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def is_valid_slug(value: str) -> bool:
    return bool(_SLUG_RE.match((value or "").strip()))


def require_slug(value: str, *, label: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{label} is required")
    if not is_valid_slug(cleaned):
        raise ValueError(
            f"Invalid {label}: must match {_SLUG_RE.pattern} (lowercase a-z0-9, '-', '_' and <=63 chars)"
        )
    return cleaned


def slugify_for_id(display_name: str, *, prefix: str) -> str:
    """Generate a filesystem/URL-safe id slug from display_name.

    - Keeps only lowercase ascii a-z0-9 and separators '-','_'
    - Collapses separators
    - Falls back to '{prefix}-<timestamp>-<rand>' when nothing remains
    """
    raw = unicodedata.normalize("NFKC", (display_name or "").strip()).lower()
    raw = raw.replace(" ", "-")

    # Keep only a-z0-9 and separators.
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw)
    cleaned = re.sub(r"[-_]{2,}", "-", cleaned)
    cleaned = cleaned.strip("-_")

    if cleaned and is_valid_slug(cleaned):
        return cleaned

    ts = time.strftime("%Y%m%d%H%M%S")
    rand = secrets.token_hex(2)
    fallback = f"{prefix}-{ts}-{rand}".lower()
    fallback = fallback[:63]
    if not is_valid_slug(fallback):
        # Very defensive: ensure it always matches.
        fallback = re.sub(r"[^a-z0-9_-]+", "-", fallback).strip("-_")
        fallback = (fallback or f"{prefix}-{rand}")[:63]
    return fallback


def make_unique_slug(
    desired: str, *, exists: callable[[str], bool], max_tries: int = 100
) -> str:
    """Return desired if available, otherwise append -2, -3, ... until available."""
    if not exists(desired):
        return desired

    base = desired
    for i in range(2, max_tries + 1):
        suffix = f"-{i}"
        candidate = (base[: max(1, 63 - len(suffix))] + suffix).rstrip("-_")
        if is_valid_slug(candidate) and not exists(candidate):
            return candidate
    raise ValueError("failed to generate unique id")
