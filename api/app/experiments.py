from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from .config import ConfigManager
from .eval_scores import (
    ScoresNameError,
    build_rows,
    index_items_by_id,
    load_json_file,
    paginate,
    resolve_heads,
    resolve_scores_path,
    rows_to_csv,
)
from .label_input import load_label_json, normalize_label_thumb_paths
from .utils.atomic_io import write_json_atomic
from .utils.legacy import PATH_KEYS, normalize_to_source_rel
from .utils.meta import ExperimentMeta
from .utils.short_id import (
    generate_short_id,
    is_valid_short_id,
    require_experiment_id,
    require_short_id,
    require_workspace_id,
)
from .utils.sidebar_order import apply_order_with_fallback, load_sidebar_order


def _reserved_experiment_ids_in_trash(trash_root: Path, *, workspace: str) -> set[str]:
    """Collect experiment ids in trash metadata for a given workspace."""

    reserved: set[str] = set()
    if not trash_root.exists() or not trash_root.is_dir():
        return reserved

    for p in trash_root.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        if not meta_path.exists() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict) or meta.get("kind") != "experiment":
            continue

        ws_id = (str(meta.get("workspace") or "")).strip().lower()
        if ws_id != workspace:
            continue

        exp_id = (str(meta.get("experiment") or "")).strip().lower()
        if is_valid_short_id(exp_id):
            reserved.add(exp_id)

    return reserved


class CreateExperimentBody(BaseModel):
    display_name: str
    id: str | None = None


class CreateExperimentFromSourceBody(BaseModel):
    display_name: str
    id: str | None = None


class RenameExperimentBody(BaseModel):
    display_name: str


# Default schema used when creating a new experiment.
DEFAULT_LABEL_SCHEMA = {
    "version": 2,
    "schema_id": "vtuber_v1",
    "heads": [
        {
            "id": "split",
            "label": "データ用途",
            "type": "split",
            "choices": ["train", "val", "unassigned"],
        },
    ],
}


def _ensure_experiment_dirs(root: Path) -> None:
    (root / "label_input" / "history").mkdir(parents=True, exist_ok=True)
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)


def _ensure_labels_json(label_input_dir: Path) -> bool:
    labels_path = label_input_dir / "labels.json"
    if labels_path.exists():
        return False

    legacy_current = label_input_dir / "current.json"
    if legacy_current.exists() and legacy_current.is_file():
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_current.replace(labels_path)
        return True

    template = {
        "items": [],
        "meta": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": "mvp"},
    }
    write_json_atomic(labels_path, template)
    return True


def _ensure_label_schema(label_input_dir: Path) -> bool:
    schema_path = label_input_dir / "label_schema.json"
    if schema_path.exists():
        return False
    write_json_atomic(schema_path, DEFAULT_LABEL_SCHEMA)
    return True


def _count_model_runs(exp_root: Path) -> int:
    models_dir = exp_root / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        return 0
    try:
        return sum(1 for p in models_dir.glob("run_*") if p.is_dir())
    except Exception:
        return 0


def _count_scores(exp_root: Path) -> int:
    eval_dir = exp_root / "eval"
    if not eval_dir.exists() or not eval_dir.is_dir():
        return 0
    try:
        return sum(1 for p in eval_dir.glob("scores_*.json") if p.is_file())
    except Exception:
        return 0


def _count_labels(exp_root: Path) -> int | None:
    label_input_dir = exp_root / "label_input"
    labels_path = label_input_dir / "labels.json"
    legacy_current = label_input_dir / "current.json"
    target = labels_path if labels_path.exists() else legacy_current
    if not target.exists() or not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return len(data.get("items"))
    return None


def _load_label_items(exp_root: Path) -> list[dict[str, Any]]:
    label_input_dir = exp_root / "label_input"
    labels_path = label_input_dir / "labels.json"
    if not labels_path.exists():
        legacy_current = label_input_dir / "current.json"
        if legacy_current.exists() and legacy_current.is_file():
            labels_path = legacy_current

    if not labels_path.exists() or not labels_path.is_file():
        return []

    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def _normalize_split_for_impact(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in ("ignore", "delete"):
        return "unassigned"
    return normalized


def _coerce_delete_flag_for_impact(item: dict[str, Any]) -> bool:
    raw = item.get("delete")
    if isinstance(raw, bool):
        if raw:
            return True
    elif isinstance(raw, (int, float)):
        if bool(raw):
            return True
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True

    labels = item.get("labels")
    if isinstance(labels, dict):
        raw_split = str(labels.get("split") or "").strip().lower()
        if raw_split == "delete":
            return True

    legacy_split = str(item.get("split") or "").strip().lower()
    return legacy_split == "delete"


def _extract_source_rel_for_impact(
    item: dict[str, Any],
    *,
    workspace_root: Path,
    source_root: Path,
) -> str | None:
    source_dir_name = source_root.name
    for key in PATH_KEYS:
        value = item.get(key)
        if not isinstance(value, str):
            continue
        path_str = unquote(value.strip()).replace("\\", "/")
        if not path_str:
            continue
        if path_str.startswith("/static/"):
            parts = [p for p in path_str.split("/") if p]
            if len(parts) >= 4:
                path_str = "/".join(parts[3:])
        if path_str.startswith(f"{source_dir_name}/"):
            path_str = path_str[len(source_dir_name) + 1 :]
        try:
            return normalize_to_source_rel(
                path_str=path_str,
                workspace_root=workspace_root,
                source_base=source_root,
            )
        except ValueError:
            continue
    return None


def _collect_delete_target_paths(
    items: list[dict[str, Any]],
    *,
    workspace_root: Path,
    source_root: Path,
) -> set[str]:
    targets: set[str] = set()
    for item in items:
        if not _coerce_delete_flag_for_impact(item):
            continue
        rel = _extract_source_rel_for_impact(
            item,
            workspace_root=workspace_root,
            source_root=source_root,
        )
        if rel:
            targets.add(rel)
    return targets


def _split_bucket_for_impact(value: str | None) -> str:
    split = _normalize_split_for_impact(value)
    if split == "train":
        return "train"
    if split == "val":
        return "val"
    if split == "unassigned" or split is None:
        return "unassigned"
    return "other"


def _count_impacted_paths_in_experiment(
    *,
    items: list[dict[str, Any]],
    target_paths: set[str],
    workspace_root: Path,
    source_root: Path,
) -> dict[str, int]:
    if not target_paths:
        return {
            "shared": 0,
            "train": 0,
            "val": 0,
            "unassigned": 0,
            "other": 0,
            "flagged": 0,
        }

    split_rank = {"train": 3, "val": 2, "unassigned": 1, "other": 0}
    per_path: dict[str, dict[str, Any]] = {}

    for item in items:
        rel = _extract_source_rel_for_impact(
            item,
            workspace_root=workspace_root,
            source_root=source_root,
        )
        if not rel or rel not in target_paths:
            continue

        labels = item.get("labels")
        split_value: Any = None
        if isinstance(labels, dict):
            split_value = labels.get("split")
        if split_value is None:
            split_value = item.get("split")
        bucket = _split_bucket_for_impact(
            split_value if isinstance(split_value, str) else None
        )
        flagged = _coerce_delete_flag_for_impact(item)

        current = per_path.get(rel)
        if current is None:
            per_path[rel] = {
                "bucket": bucket,
                "rank": split_rank[bucket],
                "flagged": flagged,
            }
            continue

        if split_rank[bucket] > int(current.get("rank") or 0):
            current["bucket"] = bucket
            current["rank"] = split_rank[bucket]
        if flagged:
            current["flagged"] = True

    counts = {
        "shared": len(per_path),
        "train": 0,
        "val": 0,
        "unassigned": 0,
        "other": 0,
        "flagged": 0,
    }

    for state in per_path.values():
        bucket = str(state.get("bucket") or "other")
        if bucket not in ("train", "val", "unassigned", "other"):
            bucket = "other"
        counts[bucket] += 1
        if state.get("flagged") is True:
            counts["flagged"] += 1

    return counts


def _provision_experiment_root(
    cfg: Any, workspace_id: str, display_name: str, desired_id: str | None = None
) -> tuple[Path, str, bool]:
    trash_root = (cfg.workspaces_root / ".trash").resolve()
    reserved_exp_ids = _reserved_experiment_ids_in_trash(
        trash_root, workspace=workspace_id
    )

    next_display_name = (display_name or "").strip()
    if not next_display_name:
        raise HTTPException(status_code=400, detail="display_name is required")

    next_exp_id = (desired_id or "").strip()
    if next_exp_id:
        try:
            next_exp_id = require_experiment_id(next_exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    try:
        ws_root = cfg.workspace_path(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not ws_root.exists() or not ws_root.is_dir():
        raise HTTPException(
            status_code=404,
            detail=(
                "workspace does not exist. "
                f"Create it under {cfg.workspaces_root}/{workspace_id} first."
            ),
        )

    exp_parent = cfg.experiments_path_for(workspace_id)

    if next_exp_id:
        if (exp_parent / next_exp_id).exists():
            raise HTTPException(status_code=409, detail="experiment id already exists")
        if (trash_root / next_exp_id).exists() or next_exp_id in reserved_exp_ids:
            raise HTTPException(
                status_code=409, detail="experiment id conflicts with trashed items"
            )
    else:
        for _ in range(50):
            candidate = generate_short_id()
            if (exp_parent / candidate).exists():
                continue
            if (trash_root / candidate).exists() or candidate in reserved_exp_ids:
                continue
            next_exp_id = candidate
            break
        if not next_exp_id:
            raise HTTPException(
                status_code=500, detail="failed to generate unique experiment id"
            )

    exp_root = exp_parent / next_exp_id
    if exp_root.exists() and not exp_root.is_dir():
        raise HTTPException(status_code=400, detail="experiment path is not a directory")

    created = not exp_root.exists()
    _ensure_experiment_dirs(exp_root)
    _ensure_labels_json(exp_root / "label_input")
    _ensure_label_schema(exp_root / "label_input")

    try:
        ExperimentMeta.create(
            exp_root, experiment_id=next_exp_id, display_name=next_display_name
        )
    except Exception:
        pass

    return exp_root, next_exp_id, created


def create_experiments_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    # ---- Scoped (recommended) API ----

    scoped = APIRouter(prefix="/workspaces/{workspace}/experiments")

    @scoped.get("")
    def list_experiments_scoped(workspace: str, order: str | None = None) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            root = cfg.experiments_path_for(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        items: list[dict] = []
        if root.exists():
            exp_paths: list[Path] = []
            for p in sorted(root.iterdir()):
                if not p.is_dir():
                    continue
                try:
                    require_experiment_id(p.name)
                except ValueError:
                    # Silently ignore non-managed directories.
                    continue
                exp_paths.append(p)

            if order == "sidebar":
                sidebar_order = load_sidebar_order(cfg.workspaces_root)
                wanted_map = sidebar_order.get("experiments")
                wanted = (
                    wanted_map.get(ws_id)
                    if isinstance(wanted_map, dict)
                    and isinstance(wanted_map.get(ws_id), list)
                    else []
                )
                ids_sorted = [require_experiment_id(p.name) for p in exp_paths]
                exp_id_order = apply_order_with_fallback(ids_sorted, wanted)
                exp_by_id = {require_experiment_id(p.name): p for p in exp_paths}
                exp_paths = [exp_by_id[x] for x in exp_id_order if x in exp_by_id]

            for p in exp_paths:
                exp_id = require_experiment_id(p.name)
                try:
                    meta = ExperimentMeta.load(p, experiment_id=exp_id)
                except Exception:
                    meta = ExperimentMeta.load(p, experiment_id=exp_id)
                items.append(
                    {
                        "id": exp_id,
                        "display_name": meta.display_name,
                        "path": str(p.resolve()),
                        "model_runs_count": _count_model_runs(p),
                        "scores_count": _count_scores(p),
                        "labels_count": _count_labels(p),
                        "updated_at": meta.updated_at,
                        "created_at": meta.created_at,
                    }
                )
        return {"workspace": ws_id, "experiments": items}

    @scoped.get("/{experiment}")
    def get_experiment_scoped(workspace: str, experiment: str) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        exp_id = experiment.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            exp_id = require_experiment_id(exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        exp_root = cfg.experiments_path_for(ws_id) / exp_id
        if not exp_root.exists() or not exp_root.is_dir():
            raise HTTPException(status_code=404, detail="experiment not found")

        meta = ExperimentMeta.load(exp_root, experiment_id=exp_id)
        return {
            "workspace": ws_id,
            "experiment": exp_id,
            "display_name": meta.display_name,
            "workspace_uid": meta.workspace_uid,
            "experiment_uid": meta.experiment_uid,
            "path": str(exp_root.resolve()),
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
        }

    @scoped.get("/{experiment}/delete-impact")
    def get_delete_impact_scoped(workspace: str, experiment: str) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        exp_id = experiment.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            exp_id = require_experiment_id(exp_id)
            ws_root = cfg.workspace_path(ws_id).resolve()
            exp_parent = cfg.experiments_path_for(ws_id)
            source_root = cfg.source_path_for(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        exp_root = exp_parent / exp_id
        if not exp_root.exists() or not exp_root.is_dir():
            raise HTTPException(status_code=404, detail="experiment not found")

        target_items = _load_label_items(exp_root)
        target_paths = _collect_delete_target_paths(
            target_items,
            workspace_root=ws_root,
            source_root=source_root,
        )

        experiments: list[dict[str, Any]] = []
        checked_count = 0
        totals = {
            "experiments_affected": 0,
            "shared": 0,
            "train": 0,
            "val": 0,
            "unassigned": 0,
            "other": 0,
            "flagged": 0,
        }

        if exp_parent.exists() and exp_parent.is_dir() and target_paths:
            for p in sorted(exp_parent.iterdir()):
                if not p.is_dir():
                    continue

                other_exp_id = p.name
                try:
                    other_exp_id = require_experiment_id(other_exp_id)
                except ValueError:
                    continue
                if other_exp_id == exp_id:
                    continue

                checked_count += 1
                items = _load_label_items(p)
                counts = _count_impacted_paths_in_experiment(
                    items=items,
                    target_paths=target_paths,
                    workspace_root=ws_root,
                    source_root=source_root,
                )
                if counts["shared"] <= 0:
                    continue

                display_name = other_exp_id
                try:
                    other_meta = ExperimentMeta.load(p, experiment_id=other_exp_id)
                    display_name = other_meta.display_name
                except Exception:
                    pass

                experiments.append(
                    {
                        "id": other_exp_id,
                        "display_name": display_name,
                        "counts": counts,
                    }
                )
                totals["shared"] += counts["shared"]
                totals["train"] += counts["train"]
                totals["val"] += counts["val"]
                totals["unassigned"] += counts["unassigned"]
                totals["other"] += counts["other"]
                totals["flagged"] += counts["flagged"]

        experiments.sort(
            key=lambda it: (
                -int((it.get("counts") or {}).get("train") or 0),
                -int((it.get("counts") or {}).get("val") or 0),
                -int((it.get("counts") or {}).get("shared") or 0),
                str(it.get("id") or ""),
            )
        )
        totals["experiments_affected"] = len(experiments)

        return {
            "workspace": ws_id,
            "experiment": exp_id,
            "delete_targets": len(target_paths),
            "other_experiments_checked": checked_count,
            "totals": totals,
            "experiments": experiments,
        }

    @scoped.post("")
    def create_experiment_scoped(workspace: str, body: CreateExperimentBody) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        display_name = (body.display_name or "").strip()
        exp_root, exp_id, created = _provision_experiment_root(
            cfg, ws_id, display_name, body.id
        )

        return {
            "status": "ok",
            "workspace": ws_id,
            "experiment": exp_id,
            "display_name": display_name,
            "path": str(exp_root),
            "created": created,
        }

    @scoped.post("/{experiment}/reuse")
    def create_experiment_from_source_scoped(
        workspace: str, experiment: str, body: CreateExperimentFromSourceBody
    ) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        src_exp_id = experiment.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            src_exp_id = require_experiment_id(src_exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        src_root = cfg.experiments_path_for(ws_id) / src_exp_id
        if not src_root.exists() or not src_root.is_dir():
            raise HTTPException(status_code=404, detail="source experiment not found")

        display_name = (body.display_name or "").strip()
        target_root, target_exp_id, created = _provision_experiment_root(
            cfg, ws_id, display_name, body.id
        )

        copied = {
            "labels_json": False,
            "label_schema_json": False,
            "cache": False,
        }

        src_label_input = src_root / "label_input"
        dst_label_input = target_root / "label_input"
        src_labels_path = src_label_input / "labels.json"
        if not src_labels_path.exists():
            legacy = src_label_input / "current.json"
            if legacy.exists():
                src_labels_path = legacy
        src_schema_path = src_label_input / "label_schema.json"
        src_cache_dir = src_root / "cache"
        dst_cache_dir = target_root / "cache"

        try:
            if src_labels_path.exists() and src_labels_path.is_file():
                shutil.copy2(src_labels_path, dst_label_input / "labels.json")
                normalized_labels = normalize_label_thumb_paths(
                    load_label_json(dst_label_input / "labels.json"),
                    workspace=ws_id,
                    experiment=target_exp_id,
                )
                write_json_atomic(
                    dst_label_input / "labels.json", normalized_labels
                )
                copied["labels_json"] = True

            if src_schema_path.exists() and src_schema_path.is_file():
                shutil.copy2(src_schema_path, dst_label_input / "label_schema.json")
                copied["label_schema_json"] = True

            if src_cache_dir.exists() and src_cache_dir.is_dir():
                if dst_cache_dir.exists():
                    shutil.rmtree(dst_cache_dir)
                shutil.copytree(src_cache_dir, dst_cache_dir)
                copied["cache"] = True
        except Exception as exc:
            try:
                shutil.rmtree(target_root)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "status": "ok",
            "workspace": ws_id,
            "source_experiment": src_exp_id,
            "experiment": target_exp_id,
            "display_name": display_name,
            "path": str(target_root),
            "created": created,
            "copied": copied,
        }

    @scoped.patch("/{experiment}")
    def rename_experiment_scoped(
        workspace: str, experiment: str, body: RenameExperimentBody
    ) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        exp_id = experiment.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            exp_id = require_experiment_id(exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        display_name = (body.display_name or "").strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="display_name is required")

        exp_root = cfg.experiments_path_for(ws_id) / exp_id
        if not exp_root.exists() or not exp_root.is_dir():
            raise HTTPException(status_code=404, detail="experiment not found")

        meta = ExperimentMeta.load(exp_root, experiment_id=exp_id)
        meta.rename(exp_root, display_name=display_name)
        return {
            "status": "ok",
            "workspace": ws_id,
            "experiment": exp_id,
            "display_name": meta.display_name,
            "path": str(exp_root),
        }

    @scoped.post("/{experiment}/trash")
    def trash_experiment_scoped(workspace: str, experiment: str) -> dict:
        cfg = config_manager.get_config()
        ws_id = workspace.strip()
        exp_id = experiment.strip()
        try:
            ws_id = require_workspace_id(ws_id)
            exp_id = require_experiment_id(exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        exp_root = cfg.experiments_path_for(ws_id) / exp_id
        if not exp_root.exists() or not exp_root.is_dir():
            raise HTTPException(status_code=404, detail="experiment not found")

        display_name = exp_id
        workspace_uid = None
        experiment_uid = None
        try:
            meta = ExperimentMeta.load(exp_root, experiment_id=exp_id)
            display_name = meta.display_name
            workspace_uid = meta.workspace_uid
            experiment_uid = meta.experiment_uid
        except Exception:
            pass

        trash_root = (cfg.workspaces_root / ".trash").resolve()
        trash_root.mkdir(parents=True, exist_ok=True)

        trash_id = ""
        for _ in range(50):
            candidate = uuid.uuid4().hex
            if not (trash_root / candidate).exists():
                trash_id = candidate
                break
        if not trash_id:
            raise HTTPException(status_code=500, detail="failed to generate trash id")

        trash_dir = (trash_root / trash_id).resolve()
        trash_dir.mkdir(parents=True, exist_ok=False)

        moved_to = trash_dir / "item"
        deleted_at = time.time()
        try:
            try:
                exp_root.replace(moved_to)
            except OSError:
                shutil.move(str(exp_root), str(moved_to))

            meta_payload = {
                "version": 1,
                "trash_id": trash_id,
                "kind": "experiment",
                "workspace": ws_id,
                "experiment": exp_id,
                "display_name": display_name,
                "deleted_at": deleted_at,
                "original_rel": f"{ws_id}/{cfg.experiments_dir}/{exp_id}",
            }
            # Add UID fields if present
            if workspace_uid is not None:
                meta_payload["workspace_uid"] = workspace_uid
            if experiment_uid is not None:
                meta_payload["experiment_uid"] = experiment_uid

            (trash_dir / "meta.json").write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            # Best-effort cleanup.
            try:
                if moved_to.exists() and moved_to.is_dir() and not exp_root.exists():
                    try:
                        moved_to.replace(exp_root)
                    except OSError:
                        shutil.move(str(moved_to), str(exp_root))
            except Exception:
                pass
            try:
                shutil.rmtree(trash_dir)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "status": "ok",
            "trash_id": trash_id,
            "kind": "experiment",
            "workspace": ws_id,
            "experiment": exp_id,
            "display_name": display_name,
            "deleted_at": deleted_at,
            "moved_from": str(exp_root),
            "moved_to": str(moved_to),
        }

    def _parse_run_timestamp(run_name: str) -> Optional[str]:
        # run_YYYYmmdd_HHMMSS -> ISO-like string
        if not run_name.startswith("run_"):
            return None
        raw = run_name[len("run_") :]
        try:
            ymd, hms = raw.split("_", 1)
            if len(ymd) != 8 or len(hms) != 6:
                return None
            return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
        except Exception:
            return None

    @scoped.get("/{experiment}/models")
    def list_models_scoped(workspace: str, experiment: str) -> dict:
        cfg = config_manager.get_config()
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
            exp_root = cfg.experiments_path_for(ws_id) / exp_id
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        models_dir = exp_root / "models"
        if not models_dir.exists():
            return {
                "workspace": ws_id,
                "experiment": exp_id,
                "models_dir": str(models_dir),
                "runs": [],
            }

        runs: List[Dict[str, Any]] = []
        for run_dir in sorted(
            models_dir.glob("run_*"), key=lambda p: p.name, reverse=True
        ):
            if not run_dir.is_dir():
                continue

            files: List[Dict[str, Any]] = []
            meta: Optional[Dict[str, Any]] = None

            for meta_name in ("_meta.json", "run_meta.json"):
                meta_path = run_dir / meta_name
                if meta_path.exists() and meta_path.is_file():
                    try:
                        # Keep this lightweight; ignore parse errors.
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = None
                    break

            for p in sorted(run_dir.iterdir(), key=lambda x: x.name):
                if not p.is_file():
                    continue
                try:
                    stat = p.stat()
                except Exception:
                    continue

                try:
                    rel = p.resolve().relative_to(cfg.workspaces_root)
                    url = f"/static/{rel.as_posix()}"
                except Exception:
                    url = None

                files.append(
                    {
                        "name": p.name,
                        "bytes": int(stat.st_size),
                        "mtime": float(stat.st_mtime),
                        "url": url,
                    }
                )

            try:
                run_rel = run_dir.resolve().relative_to(cfg.workspaces_root)
                run_rel_str = run_rel.as_posix()
            except Exception:
                run_rel_str = None

            runs.append(
                {
                    "id": run_dir.name,
                    "created_at": _parse_run_timestamp(run_dir.name),
                    "path": str(run_dir),
                    "rel": run_rel_str,
                    "files": files,
                    "meta": meta,
                }
            )

        return {
            "workspace": ws_id,
            "experiment": exp_id,
            "models_dir": str(models_dir),
            "runs": runs,
        }

    @scoped.get("/{experiment}/eval")
    def list_eval_scoped(workspace: str, experiment: str) -> dict:
        cfg = config_manager.get_config()
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
            exp_root = cfg.experiments_path_for(ws_id) / exp_id
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        eval_dir = exp_root / "eval"
        if not eval_dir.exists():
            return {
                "workspace": ws_id,
                "experiment": exp_id,
                "eval_dir": str(eval_dir),
                "files": [],
            }

        files: List[Dict[str, Any]] = []
        for p in sorted(
            eval_dir.glob("scores_*.json"), key=lambda x: x.name, reverse=True
        ):
            if not p.is_file():
                continue
            try:
                stat = p.stat()
            except Exception:
                continue

            try:
                rel = p.resolve().relative_to(cfg.workspaces_root)
                url = f"/static/{rel.as_posix()}"
            except Exception:
                url = None

            files.append(
                {
                    "name": p.name,
                    "bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                    "url": url,
                }
            )

        return {
            "workspace": ws_id,
            "experiment": exp_id,
            "eval_dir": str(eval_dir),
            "files": files,
        }

    def _load_scores_rows(
        cfg, ws_id: str, exp_id: str, name: str
    ) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """scores_*.json を読み、index.json と突き合わせて表示用の行に整形する。"""
        exp_root = cfg.experiments_path_for(ws_id) / exp_id
        eval_dir = exp_root / "eval"

        try:
            scores_path = resolve_scores_path(eval_dir, name)
        except ScoresNameError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not scores_path.exists() or not scores_path.is_file():
            raise HTTPException(status_code=404, detail="scores file not found")

        try:
            scores_data = load_json_file(scores_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"failed to parse scores file: {exc}"
            )

        meta = scores_data.get("meta") if isinstance(scores_data.get("meta"), dict) else {}
        items = scores_data.get("items") or []
        heads = resolve_heads(meta, items)

        # split と画像 URL を補うために dataset/index.json を突き合わせる（無くても続行）。
        index_data = None
        index_path = exp_root / "dataset" / "index.json"
        if index_path.exists():
            try:
                index_data = load_json_file(index_path)
            except Exception:  # noqa: BLE001
                index_data = None

        dataset_url_base = None
        try:
            rel = (exp_root / "dataset").resolve().relative_to(cfg.workspaces_root)
            dataset_url_base = f"/static/{rel.as_posix()}"
        except Exception:  # noqa: BLE001
            dataset_url_base = None

        rows = build_rows(
            items,
            index_items_by_id(index_data),
            heads,
            dataset_url_base=dataset_url_base,
        )
        return meta, heads, rows

    @scoped.get("/{experiment}/eval/{name}")
    def get_eval_scores_scoped(
        workspace: str,
        experiment: str,
        name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        cfg = config_manager.get_config()
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        meta, heads, rows = _load_scores_rows(cfg, ws_id, exp_id, name)
        # 大量画像でもブラウザを詰まらせないよう上限を設ける。
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))

        return {
            "workspace": ws_id,
            "experiment": exp_id,
            "name": name,
            "meta": meta,
            "heads": heads,
            "total": len(rows),
            "limit": safe_limit,
            "offset": safe_offset,
            "items": paginate(rows, limit=safe_limit, offset=safe_offset),
        }

    @scoped.get("/{experiment}/eval/{name}/export.csv")
    def export_eval_scores_csv_scoped(
        workspace: str, experiment: str, name: str
    ) -> Response:
        cfg = config_manager.get_config()
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        _meta, heads, rows = _load_scores_rows(cfg, ws_id, exp_id, name)
        # 表示中のページではなく全件を書き出す。
        csv_text = rows_to_csv(rows, heads)
        filename = f"{Path(name).stem}.csv"
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    router.include_router(scoped)

    return router
