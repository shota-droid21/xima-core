from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .config import ConfigManager
from .label_input import load_label_json
from .native_files import native_file_serving_enabled
from .utils.legacy import PATH_KEYS, normalize_to_source_rel
from .utils.paths import is_subpath
from .utils.short_id import require_experiment_id, require_workspace_id

IMAGE_CONCURRENCY = 4
image_sem = asyncio.Semaphore(IMAGE_CONCURRENCY)

logger = logging.getLogger(__name__)


def _iter_items(label_data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(label_data, list):
        yield from (item for item in label_data if isinstance(item, dict))
    elif isinstance(label_data, dict):
        items = label_data.get("items", [])
        if isinstance(items, list):
            yield from (item for item in items if isinstance(item, dict))


def _build_image_map_scoped(
    label_data: Any, cfg, *, workspace: str
) -> Dict[str, Tuple[Path, str | None]]:
    mapping: Dict[str, Tuple[Path, str | None]] = {}
    base_resolved = cfg.source_path_for(workspace).resolve()
    ws_root = cfg.workspace_path(workspace).resolve()
    for item in _iter_items(label_data):
        # item_key may be 'id' or 'file_id' depending on the lookup we're building for.
        image_id = item.get("id")
        rel_path = next((item.get(key) for key in PATH_KEYS if item.get(key)), None)
        if not image_id or not rel_path:
            continue
        try:
            normalized = normalize_to_source_rel(
                path_str=str(rel_path),
                workspace_root=ws_root,
                source_base=cfg.source_path_for(workspace),
            )
            resolved = (base_resolved / Path(normalized)).resolve()
        except Exception:
            continue
        if not is_subpath(base_resolved, resolved):
            continue
        mapping[str(image_id)] = (resolved, item.get("file_id"))
    return mapping


def _build_fileid_map_scoped(
    label_data: Any, cfg, *, workspace: str
) -> Dict[str, Tuple[Path, str | None]]:
    mapping: Dict[str, Tuple[Path, str | None]] = {}
    base_resolved = cfg.source_path_for(workspace).resolve()
    ws_root = cfg.workspace_path(workspace).resolve()
    for item in _iter_items(label_data):
        file_id = item.get("file_id")
        rel_path = next((item.get(key) for key in PATH_KEYS if item.get(key)), None)
        if not file_id or not rel_path:
            continue
        try:
            normalized = normalize_to_source_rel(
                path_str=str(rel_path),
                workspace_root=ws_root,
                source_base=cfg.source_path_for(workspace),
            )
            resolved = (base_resolved / Path(normalized)).resolve()
        except Exception:
            continue
        if not is_subpath(base_resolved, resolved):
            continue
        mapping[str(file_id)] = (resolved, item.get("file_id"))
    return mapping


def _cache_root_for(label_path: Path) -> Path:
    # label_input/labels.json -> experiments/<exp>/cache/thumbs
    return label_path.parent.parent / "cache" / "thumbs"


def _cache_key(raw: str) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_root: Path, width: int, key: str) -> Path:
    safe_key = _cache_key(key)
    shard = safe_key[:2]
    return cache_root / f"w{width}" / shard / f"{safe_key}.webp"


def _validate_width(w: int | None) -> int | None:
    if w is None:
        return None
    try:
        width = int(w)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="w must be an integer")
    if width <= 0:
        raise HTTPException(status_code=400, detail="w must be positive")
    return width


async def _serve_image(
    src: Path,
    *,
    workspaces_root: Path,
    cache_root: Path,
    width: int | None,
    cache_key: str | None,
):
    if width is None:
        return _x_accel_file_response(
            src,
            workspaces_root=workspaces_root,
            media_type=mimetypes.guess_type(str(src))[0] or "application/octet-stream",
            cache_control="public, max-age=86400",
        )

    if cache_key is None:
        cache_key = src.stem

    cache_path = _cache_path(cache_root, width, cache_key)
    try:
        src_mtime = src.stat().st_mtime
    except OSError:
        raise HTTPException(status_code=404, detail="image file not found")

    if cache_path.exists():
        try:
            if cache_path.stat().st_mtime >= src_mtime:
                return _x_accel_file_response(
                    cache_path,
                    workspaces_root=workspaces_root,
                    media_type="image/webp",
                    cache_control="public, max-age=31536000, immutable",
                )
        except OSError:
            pass

    raise HTTPException(
        status_code=404,
        detail=(
            "thumbnail not found or stale; run make_label_list to pre-generate thumbs"
        ),
    )


def _x_accel_file_response(
    file_path: Path,
    *,
    workspaces_root: Path,
    media_type: str,
    cache_control: str,
) -> Response:
    """Return an empty response that instructs nginx to serve a local file.

    Nginx must provide an internal location mapping:
      location /_internal_ws/ { internal; alias /data/workspaces/; }

    nginx 非在のネイティブ起動（XIMA_SERVE_FILES_NATIVE）では core 自身が
    FileResponse で実配信する（X-Accel は nginx が居ないと無視されるため）。
    """

    if native_file_serving_enabled():
        return FileResponse(
            file_path,
            media_type=media_type,
            headers={"Cache-Control": cache_control},
        )

    # We intentionally return no body; nginx will serve the file.
    headers = {
        "Content-Type": media_type,
        "Cache-Control": cache_control,
    }
    # X-Accel-Redirect path must be an URI, not a filesystem path.
    headers["X-Accel-Redirect"] = _internal_ws_url(
        file_path, workspaces_root=workspaces_root
    )
    return Response(content=b"", status_code=200, headers=headers)


def _internal_ws_url(file_path: Path, *, workspaces_root: Path) -> str:
    fs_path = file_path.resolve()
    ws_root = workspaces_root.resolve()

    try:
        rel = fs_path.relative_to(ws_root)
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="resolved path is outside workspaces_root",
        )

    internal_url = "/_internal_ws/" + rel.as_posix().lstrip("/")
    logger.debug("x-accel: resolved=%s internal=%s", fs_path, internal_url)
    return internal_url


def create_images_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    # ---- Scoped API ----

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/images/by-item/{item_id}"
    )
    async def get_image_by_item_scoped(
        workspace: str, experiment: str, item_id: str, w: int | None = None
    ):
        width = _validate_width(w)
        cfg = config_manager.get_config()
        ws_root = cfg.workspaces_root
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            # One-time migration: current.json -> labels.json
            if not label_path.exists():
                legacy_current = label_path.parent / "current.json"
                if legacy_current.exists() and legacy_current.is_file():
                    legacy_current.replace(label_path)
            label_data = load_label_json(label_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="labels.json not found")

        mapping = _build_image_map_scoped(label_data, cfg, workspace=workspace)
        entry = mapping.get(item_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="item id not found")

        path, file_id = entry
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="image file not found")

        cache_root = _cache_root_for(label_path)
        async with image_sem:
            return await _serve_image(
                path,
                workspaces_root=ws_root,
                cache_root=cache_root,
                width=width,
                cache_key=file_id or item_id,
            )

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/images/by-file/{file_id}"
    )
    async def get_image_by_file_scoped(
        workspace: str, experiment: str, file_id: str, w: int | None = None
    ):
        width = _validate_width(w)
        cfg = config_manager.get_config()
        ws_root = cfg.workspaces_root

        # Thumbnails: do NOT consult labels.json; resolve directly from cache.
        if width is not None:
            try:
                ws_id = require_workspace_id(workspace)
                exp_id = require_experiment_id(experiment)
                cache_root = (
                    cfg.experiments_path_for(ws_id) / exp_id / "cache" / "thumbs"
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            cache_path = _cache_path(cache_root, width, file_id)
            if not cache_path.exists() or not cache_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "thumbnail not found; run make_label_list to pre-generate thumbs"
                    ),
                )

            async with image_sem:
                return _x_accel_file_response(
                    cache_path,
                    workspaces_root=ws_root,
                    media_type="image/webp",
                    cache_control="public, max-age=31536000, immutable",
                )

        # Original image: keep legacy compatibility (resolve via labels.json).
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            # One-time migration: current.json -> labels.json
            if not label_path.exists():
                legacy_current = label_path.parent / "current.json"
                if legacy_current.exists() and legacy_current.is_file():
                    legacy_current.replace(label_path)
            label_data = load_label_json(label_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="labels.json not found")

        mapping = _build_fileid_map_scoped(label_data, cfg, workspace=workspace)
        entry = mapping.get(file_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="file_id not found")

        path, fid = entry
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="image file not found")

        cache_root = _cache_root_for(label_path)
        async with image_sem:
            return await _serve_image(
                path,
                workspaces_root=ws_root,
                cache_root=cache_root,
                width=None,
                cache_key=fid or file_id,
            )

    return router
