"""Utilities for managing sidebar order (UI state).

Sidebar order is stored at workspaces/.xima/sidebar_order.json as agent-level metadata.
"""

import json
import time
from pathlib import Path


def get_sidebar_order_path(workspaces_root: Path) -> Path:
    """Get the path to sidebar_order.json in .xima directory.

    Args:
        workspaces_root: Root directory of workspaces

    Returns:
        Path to workspaces/.xima/sidebar_order.json
    """
    return (workspaces_root / ".xima" / "sidebar_order.json").resolve()


def load_sidebar_order(workspaces_root: Path) -> dict:
    """Load sidebar order from .xima/sidebar_order.json.

    Args:
        workspaces_root: Root directory of workspaces

    Returns:
        dict with keys:
            - version: int (default 1)
            - workspaces: list[str] (default [])
            - experiments: dict[str, list[str]] (default {})
            - updated_at: str | None (default None)
    """
    p = get_sidebar_order_path(workspaces_root)

    if not p.exists() or not p.is_file():
        return {"version": 1, "workspaces": [], "experiments": {}, "updated_at": None}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "workspaces": [], "experiments": {}, "updated_at": None}

    if not isinstance(data, dict):
        return {"version": 1, "workspaces": [], "experiments": {}, "updated_at": None}

    return {
        "version": int(data.get("version") or 1),
        "workspaces": (
            data.get("workspaces") if isinstance(data.get("workspaces"), list) else []
        ),
        "experiments": (
            data.get("experiments") if isinstance(data.get("experiments"), dict) else {}
        ),
        "updated_at": data.get("updated_at"),
    }


def save_sidebar_order(workspaces_root: Path, *, order: dict) -> dict:
    """Save sidebar order to .xima/sidebar_order.json.

    Creates .xima directory if it doesn't exist.
    Uses atomic write (tmp file + replace) to prevent corruption.

    Args:
        workspaces_root: Root directory of workspaces
        order: dict with optional keys:
            - workspaces: list[str]
            - experiments: dict[str, list[str]]

    Returns:
        Saved payload with version and updated_at added
    """
    p = get_sidebar_order_path(workspaces_root)

    # Ensure .xima directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "workspaces": order.get("workspaces") or [],
        "experiments": order.get("experiments") or {},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Atomic write: write to tmp file, then replace
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

    return payload


def apply_order_with_fallback(
    ids_sorted: list[str], desired: list[str] | None
) -> list[str]:
    """Apply desired order to existing ids.

    - IDs in desired order are placed first (if they exist in ids_sorted)
    - Invalid/duplicate IDs are filtered out
    - Missing IDs are appended in default order

    Args:
        ids_sorted: List of valid IDs in default order
        desired: Desired order (can be None or incomplete)

    Returns:
        Ordered list of IDs

    Example:
        >>> apply_order_with_fallback(['a', 'b', 'c'], ['c', 'a'])
        ['c', 'a', 'b']
        >>> apply_order_with_fallback(['a', 'b', 'c'], ['x', 'c', 'a', 'c'])
        ['c', 'a', 'b']  # x ignored, duplicate c removed
    """
    if not desired:
        return ids_sorted

    existing = set(ids_sorted)
    out: list[str] = []
    seen: set[str] = set()

    # Add IDs from desired order (if they exist)
    for raw in desired:
        if not isinstance(raw, str):
            continue
        v = raw.strip().lower()
        if v and v in existing and v not in seen:
            out.append(v)
            seen.add(v)

    # Append missing IDs in default order
    for v in ids_sorted:
        if v not in seen:
            out.append(v)

    return out
