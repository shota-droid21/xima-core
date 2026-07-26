from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .config import ConfigManager
from .utils.meta import ExperimentMeta, WorkspaceMeta
from .utils.paths import is_subpath
from .utils.short_id import require_experiment_id, require_workspace_id

_TRASH_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_trash_id(value: str) -> str:
    v = (value or "").strip().lower()
    if not _TRASH_ID_RE.match(v):
        raise HTTPException(status_code=400, detail="invalid trash id")
    return v


def _trash_root(cfg) -> Path:
    root = (cfg.workspaces_root / ".trash").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_trash_dir(trash_root: Path, trash_id: str) -> Path:
    d = (trash_root / trash_id).resolve()
    if not is_subpath(trash_root, d):
        raise HTTPException(status_code=400, detail="invalid trash path")
    return d


def _load_meta(trash_dir: Path) -> dict:
    meta_path = trash_dir / "meta.json"
    if not meta_path.exists() or not meta_path.is_file():
        raise HTTPException(status_code=404, detail="trash meta not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read trash meta")
    if not isinstance(meta, dict):
        raise HTTPException(status_code=500, detail="invalid trash meta")
    return meta


def _dir_size_bytes(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += int(entry.stat(follow_symlinks=False).st_size)
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _safe_non_negative_int(value) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _compute_size_fields_for_trash_item(trash_dir: Path, kind: str, cfg) -> dict:
    item_dir = (trash_dir / "item").resolve()
    if not is_subpath(trash_dir, item_dir):
        return {}
    if not item_dir.exists() or not item_dir.is_dir():
        return {}

    result: dict = {"size_bytes_total": _dir_size_bytes(item_dir)}
    if kind == "workspace":
        source_dir = item_dir / cfg.source_dir
        experiments_dir = item_dir / cfg.experiments_dir
        result["size_bytes_source"] = (
            _dir_size_bytes(source_dir)
            if source_dir.exists() and source_dir.is_dir()
            else 0
        )
        result["size_bytes_experiments"] = (
            _dir_size_bytes(experiments_dir)
            if experiments_dir.exists() and experiments_dir.is_dir()
            else 0
        )
    elif kind == "experiment":
        label_input_dir = item_dir / "label_input"
        models_dir = item_dir / "models"
        eval_dir = item_dir / "eval"
        result["size_bytes_label_input"] = (
            _dir_size_bytes(label_input_dir)
            if label_input_dir.exists() and label_input_dir.is_dir()
            else 0
        )
        result["size_bytes_models"] = (
            _dir_size_bytes(models_dir)
            if models_dir.exists() and models_dir.is_dir()
            else 0
        )
        result["size_bytes_eval"] = (
            _dir_size_bytes(eval_dir) if eval_dir.exists() and eval_dir.is_dir() else 0
        )
    return result


def _summarize_meta(meta: dict, fallback_sizes: dict | None = None) -> dict:
    kind = str(meta.get("kind") or "")
    item: dict = {
        "trash_id": str(meta.get("trash_id") or ""),
        "kind": kind,
        "display_name": meta.get("display_name"),
        "deleted_at": meta.get("deleted_at"),
        "original_rel": meta.get("original_rel"),
        "workspace": meta.get("workspace"),
        "experiment": meta.get("experiment"),
        "version": meta.get("version"),
    }
    # Add agent identity fields if present
    if "container_id" in meta:
        item["container_id"] = meta["container_id"]
    if "install_id" in meta:
        item["install_id"] = meta["install_id"]
    if "agent_version" in meta:
        item["agent_version"] = meta["agent_version"]
    # Add workspace/experiment UID fields if present
    if "workspace_item_uid" in meta:
        item["workspace_item_uid"] = meta["workspace_item_uid"]
    if "workspace_uid" in meta:
        item["workspace_uid"] = meta["workspace_uid"]
    if "experiment_uid" in meta:
        item["experiment_uid"] = meta["experiment_uid"]
    if "dependent_experiments_count" in meta:
        item["dependent_experiments_count"] = meta["dependent_experiments_count"]
    if "dependent_experiments" in meta:
        item["dependent_experiments"] = meta["dependent_experiments"]

    if fallback_sizes is None:
        fallback_sizes = {}
    size_fields = (
        "size_bytes_total",
        "size_bytes_source",
        "size_bytes_experiments",
        "size_bytes_label_input",
        "size_bytes_models",
        "size_bytes_eval",
    )
    for key in size_fields:
        preferred = _safe_non_negative_int(meta.get(key))
        if preferred is not None:
            item[key] = preferred
            continue
        fallback = _safe_non_negative_int(fallback_sizes.get(key))
        if fallback is not None:
            item[key] = fallback
    return item


def create_trash_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()
    cfg = config_manager.get_config()

    scoped = APIRouter(prefix="")

    @scoped.get("/trash")
    def list_trash() -> dict:
        trash_root = _trash_root(cfg)
        items: list[dict] = []
        if trash_root.exists() and trash_root.is_dir():
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
                if not isinstance(meta, dict):
                    continue
                # Filter out unexpected entries.
                kind = str(meta.get("kind") or "")
                if kind not in ("workspace", "experiment"):
                    continue
                fallback_sizes = _compute_size_fields_for_trash_item(p, kind, cfg)
                items.append(_summarize_meta(meta, fallback_sizes=fallback_sizes))

        items.sort(key=lambda x: float(x.get("deleted_at") or 0.0), reverse=True)
        return {"items": items, "generated_at": time.time()}

    @scoped.post("/trash/{trash_id}/restore")
    def restore_trash(trash_id: str) -> dict:
        trash_root = _trash_root(cfg)
        tid = _require_trash_id(trash_id)
        trash_dir = _safe_trash_dir(trash_root, tid)
        if not trash_dir.exists() or not trash_dir.is_dir():
            raise HTTPException(status_code=404, detail="trash item not found")

        meta = _load_meta(trash_dir)
        kind = str(meta.get("kind") or "")
        moved_from = str(trash_dir / "item")

        item_dir = (trash_dir / "item").resolve()
        if not is_subpath(trash_dir, item_dir):
            raise HTTPException(status_code=500, detail="invalid item path")
        if not item_dir.exists() or not item_dir.is_dir():
            raise HTTPException(status_code=404, detail="trash item data not found")

        if kind == "workspace":
            ws_id = str(meta.get("workspace") or "")
            try:
                ws_id = require_workspace_id(ws_id)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=str(exc))

            dest = (cfg.workspaces_root / ws_id).resolve()
            if not is_subpath(cfg.workspaces_root, dest):
                raise HTTPException(status_code=400, detail="invalid workspace")
            if dest.exists():
                raise HTTPException(
                    status_code=409,
                    detail="restore target already exists",
                )

            # Move back.
            try:
                item_dir.replace(dest)
            except OSError:
                shutil.move(str(item_dir), str(dest))

            # Best-effort: keep workspace.json consistent with directory name.
            try:
                display_name = str(meta.get("display_name") or ws_id)
                wmeta = WorkspaceMeta.load(dest, workspace_id=ws_id)
                wmeta.id = ws_id
                wmeta.display_name = display_name
                wmeta.updated_at = _now_iso()
                wmeta.save(dest)
            except Exception:
                pass

            # Remove trash container.
            try:
                shutil.rmtree(trash_dir)
            except Exception:
                pass

            return {
                "status": "ok",
                "kind": "workspace",
                "trash_id": tid,
                "workspace": ws_id,
                "display_name": meta.get("display_name"),
                "restored_to": str(dest),
                "restored_from": moved_from,
            }

        if kind == "experiment":
            ws_id = str(meta.get("workspace") or "")
            exp_id = str(meta.get("experiment") or "")
            try:
                ws_id = require_workspace_id(ws_id)
                exp_id = require_experiment_id(exp_id)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=str(exc))

            ws_root = (cfg.workspaces_root / ws_id).resolve()
            if not ws_root.exists() or not ws_root.is_dir():
                raise HTTPException(
                    status_code=409,
                    detail="workspace missing; restore workspace first",
                )

            dest_parent = cfg.experiments_path_for(ws_id).resolve()
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest = (dest_parent / exp_id).resolve()
            if not is_subpath(dest_parent, dest):
                raise HTTPException(status_code=400, detail="invalid experiment")
            if dest.exists():
                raise HTTPException(
                    status_code=409,
                    detail="restore target already exists",
                )

            try:
                item_dir.replace(dest)
            except OSError:
                shutil.move(str(item_dir), str(dest))

            try:
                display_name = str(meta.get("display_name") or exp_id)
                emeta = ExperimentMeta.load(dest, experiment_id=exp_id)
                emeta.id = exp_id
                emeta.display_name = display_name
                emeta.updated_at = _now_iso()
                emeta.save(dest)
            except Exception:
                pass

            try:
                shutil.rmtree(trash_dir)
            except Exception:
                pass

            return {
                "status": "ok",
                "kind": "experiment",
                "trash_id": tid,
                "workspace": ws_id,
                "experiment": exp_id,
                "display_name": meta.get("display_name"),
                "restored_to": str(dest),
                "restored_from": moved_from,
            }

        raise HTTPException(status_code=500, detail="unsupported trash item")

    @scoped.delete("/trash/{trash_id}")
    def delete_trash_item(trash_id: str) -> dict:
        trash_root = _trash_root(cfg)
        tid = _require_trash_id(trash_id)
        trash_dir = _safe_trash_dir(trash_root, tid)
        if not trash_dir.exists() or not trash_dir.is_dir():
            raise HTTPException(status_code=404, detail="trash item not found")

        meta = None
        try:
            meta = _load_meta(trash_dir)
        except Exception:
            meta = None

        cascade_deleted: list[str] = []
        if isinstance(meta, dict):
            kind = str(meta.get("kind") or "")
            if kind == "workspace":
                ws_id = str(meta.get("workspace") or "").strip().lower()
                # Best-effort validation; if invalid, skip cascade search.
                try:
                    ws_id = require_workspace_id(ws_id)
                except ValueError:
                    ws_id = ""

                if ws_id:
                    for p in trash_root.iterdir():
                        if not p.is_dir():
                            continue
                        dep_tid = p.name
                        if dep_tid == tid or not _TRASH_ID_RE.match(dep_tid):
                            continue
                        try:
                            dep_meta = json.loads(
                                (p / "meta.json").read_text(encoding="utf-8"),
                            )
                        except Exception:
                            continue
                        if not isinstance(dep_meta, dict):
                            continue
                        if str(dep_meta.get("kind") or "") != "experiment":
                            continue
                        dep_ws = str(dep_meta.get("workspace") or "").strip().lower()
                        if dep_ws != ws_id:
                            continue
                        try:
                            shutil.rmtree(p)
                            cascade_deleted.append(dep_tid)
                        except Exception as exc:
                            raise HTTPException(
                                status_code=500,
                                detail=f"failed to delete dependent experiment trash item: {dep_tid}: {exc}",
                            )

        try:
            shutil.rmtree(trash_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "status": "ok",
            "trash_id": tid,
            "deleted": True,
            "cascade_deleted": cascade_deleted,
            "meta": _summarize_meta(meta) if isinstance(meta, dict) else None,
        }

    @scoped.delete("/trash")
    def empty_trash() -> dict:
        trash_root = _trash_root(cfg)
        deleted: list[str] = []
        failed: list[str] = []

        if trash_root.exists() and trash_root.is_dir():
            for p in trash_root.iterdir():
                if not p.is_dir():
                    continue
                tid = p.name
                if not _TRASH_ID_RE.match(tid):
                    continue
                try:
                    shutil.rmtree(p)
                    deleted.append(tid)
                except Exception:
                    failed.append(tid)

        return {"status": "ok", "deleted": deleted, "failed": failed}

    router.include_router(scoped)
    return router
