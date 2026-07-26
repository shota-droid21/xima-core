from __future__ import annotations

from pathlib import Path


def is_subpath(base: Path, target: Path) -> bool:
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False


def safe_resolve(base: Path, relative_or_absolute: str | Path) -> Path:
    base_resolved = base.resolve()
    target = Path(relative_or_absolute)
    resolved = (target if target.is_absolute() else base_resolved / target).resolve()
    if not is_subpath(base_resolved, resolved):
        raise ValueError(f"Path escapes base: {resolved}")
    return resolved


def require_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
