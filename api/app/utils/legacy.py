from __future__ import annotations

from pathlib import Path

from .paths import is_subpath

PATH_KEYS = ["rel_path", "path", "file_path", "dataset_path"]


def normalize_to_source_rel(
    *,
    path_str: str,
    workspace_root: Path,
    source_base: Path,
) -> str:
    candidate = Path(path_str)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (workspace_root / candidate).resolve()
    )

    if is_subpath(source_base, resolved):
        return resolved.relative_to(source_base).as_posix()

    alt = (source_base / candidate).resolve()
    if is_subpath(source_base, alt):
        return alt.relative_to(source_base).as_posix()

    raise ValueError("path is outside allowed roots")
