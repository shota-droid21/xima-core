#!/usr/bin/env python3
"""Delete source images flagged with delete=true in label_input/labels.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import unquote

from job_progress import update_job_progress

PATH_KEYS = ("rel_path", "path", "file_path", "dataset_path")


def _is_subpath(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except Exception:
        return False


def _normalize_split(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value in ("ignore", "delete"):
        return "unassigned"
    return value


def _is_delete_flagged(item: Dict[str, Any]) -> bool:
    raw = item.get("delete")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", ""):
            return False

    labels = item.get("labels")
    if isinstance(labels, dict):
        split = _normalize_split(labels.get("split"))
        if split == "unassigned":
            legacy = str(labels.get("split") or "").strip().lower()
            if legacy == "delete":
                return True

    legacy_split = _normalize_split(item.get("split"))
    if legacy_split == "unassigned":
        raw_split = str(item.get("split") or "").strip().lower()
        if raw_split == "delete":
            return True
    return False


def _source_path_from_item(item: Dict[str, Any]) -> str | None:
    for key in PATH_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_source_path(root: Path, value: str) -> Path | None:
    normalized_value = unquote(value.strip()).replace("\\", "/")
    if not normalized_value:
        return None

    # Accept static URL path format:
    # /static/<workspace>/<source_dir>/<relative_path>
    if normalized_value.startswith("/static/"):
        parts = [p for p in normalized_value.split("/") if p]
        # static, <ws>, <source_dir>, ...
        if len(parts) >= 4:
            normalized_value = "/".join(parts[3:])

    # Accept "source_images/..." style values by stripping root dir name.
    root_name = root.name
    if normalized_value.startswith(f"{root_name}/"):
        normalized_value = normalized_value[len(root_name) + 1 :]

    candidate = Path(normalized_value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved if _is_subpath(root, resolved) else None

    resolved = (root / candidate).resolve()
    return resolved if _is_subpath(root, resolved) else None


def _build_file_id_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem
        if not stem:
            continue
        index.setdefault(stem, []).append(p.resolve())
    return index


def _resolve_by_file_id(
    *,
    root: Path,
    item: Dict[str, Any],
    fallback_raw_path: str | None,
    file_id_index: dict[str, list[Path]],
) -> Path | None:
    raw_file_id = item.get("file_id")
    file_id = str(raw_file_id or "").strip()
    if not file_id and fallback_raw_path:
        file_id = Path(unquote(fallback_raw_path)).stem
    if not file_id:
        return None

    candidates = file_id_index.get(file_id) or []
    if not candidates:
        return None

    # Prefer matching suffix when raw path is available.
    preferred_suffix = ""
    if fallback_raw_path:
        preferred_suffix = Path(unquote(fallback_raw_path)).suffix.lower()
    if preferred_suffix:
        for c in candidates:
            if c.suffix.lower() == preferred_suffix and _is_subpath(root, c):
                return c

    for c in candidates:
        if _is_subpath(root, c):
            return c
    return None


def _iter_items(labels_path: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise SystemExit(f"[ERROR] items is not a list: {labels_path}")
    for item in items:
        if isinstance(item, dict):
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete source images flagged with delete=true."
    )
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        help="Path to label_input/labels.json",
    )
    parser.add_argument(
        "--root",
        "-r",
        required=True,
        help="Source image root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report targets without deleting files",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels).resolve()
    root_dir = Path(args.root).resolve()
    dry_run = bool(args.dry_run)

    if not labels_path.exists() or not labels_path.is_file():
        raise SystemExit(f"[ERROR] labels not found: {labels_path}")
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"[ERROR] source root not found: {root_dir}")

    update_job_progress(phase="start", message="scanning delete-flagged items")

    total_items = 0
    delete_flagged = 0
    targets_by_path: dict[str, Path] = {}
    missing_path_count = 0
    outside_root_count = 0
    fallback_hit_count = 0
    fallback_miss_count = 0
    file_id_index: dict[str, list[Path]] | None = None

    for item in _iter_items(labels_path):
        total_items += 1
        if not _is_delete_flagged(item):
            continue
        delete_flagged += 1

        raw_path = _source_path_from_item(item)
        if not raw_path:
            missing_path_count += 1
            continue

        resolved = _resolve_source_path(root_dir, raw_path)
        if resolved is None:
            outside_root_count += 1
            continue

        if not resolved.exists():
            if file_id_index is None:
                file_id_index = _build_file_id_index(root_dir)
            fallback_resolved = _resolve_by_file_id(
                root=root_dir,
                item=item,
                fallback_raw_path=raw_path,
                file_id_index=file_id_index,
            )
            if fallback_resolved is not None:
                resolved = fallback_resolved
                fallback_hit_count += 1
            else:
                fallback_miss_count += 1

        targets_by_path[str(resolved)] = resolved

        if total_items % 100 == 0:
            update_job_progress(
                phase="scan_items",
                current=total_items,
                total=None,
                message="scanning delete-flagged items",
                extra={
                    "delete_flagged": delete_flagged,
                    "target_candidates": len(targets_by_path),
                },
            )

    targets = list(targets_by_path.values())

    update_job_progress(
        phase="delete_files",
        current=0,
        total=len(targets),
        message="deleting files" if not dry_run else "dry-run (no files deleted)",
        extra={
            "total_items": total_items,
            "delete_flagged": delete_flagged,
            "target_candidates": len(targets),
            "missing_path": missing_path_count,
            "outside_root": outside_root_count,
            "fallback_hit": fallback_hit_count,
            "fallback_miss": fallback_miss_count,
            "dry_run": dry_run,
        },
    )

    deleted_count = 0
    missing_file_count = 0
    error_count = 0
    error_samples: list[str] = []
    missing_samples: list[str] = []

    for idx, target in enumerate(targets, start=1):
        if dry_run:
            deleted_count += 1 if target.exists() else 0
            missing_file_count += 0 if target.exists() else 1
        else:
            try:
                if target.exists():
                    if target.is_file():
                        target.unlink()
                        deleted_count += 1
                    else:
                        error_count += 1
                        if len(error_samples) < 10:
                            error_samples.append(f"not file: {target}")
                else:
                    missing_file_count += 1
                    if len(missing_samples) < 10:
                        missing_samples.append(str(target))
            except Exception as exc:
                error_count += 1
                if len(error_samples) < 10:
                    error_samples.append(f"{target}: {exc}")

        if idx % 20 == 0 or idx == len(targets):
            update_job_progress(
                phase="delete_files",
                current=idx,
                total=len(targets),
                message=(
                    "deleting files"
                    if not dry_run
                    else "dry-run (no files deleted)"
                ),
                extra={
                    "deleted": deleted_count,
                    "missing_files": missing_file_count,
                    "errors": error_count,
                    "dry_run": dry_run,
                },
            )

    summary = {
        "total_items": total_items,
        "delete_flagged_items": delete_flagged,
        "target_candidates": len(targets),
        "missing_path": missing_path_count,
        "outside_root": outside_root_count,
        "fallback_hit": fallback_hit_count,
        "fallback_miss": fallback_miss_count,
        "deleted_files": deleted_count,
        "missing_files": missing_file_count,
        "errors": error_count,
        "dry_run": dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if error_samples:
        for sample in error_samples:
            print(f"[WARN] {sample}")
    if missing_samples:
        for sample in missing_samples:
            print(f"[INFO] missing target: {sample}")

    update_job_progress(
        phase="done",
        current=len(targets),
        total=len(targets),
        message="completed",
        extra=summary,
    )

    if error_count > 0:
        raise SystemExit(f"[ERROR] failed to delete {error_count} files")


if __name__ == "__main__":
    main()
