"""nginx 非在時（ネイティブ 1 起動）の画像/静的ファイル配信。

通常構成では nginx が `/static/` `/thumbs/` を workspaces から直接配信し、API は
`X-Accel-Redirect` で実配信を nginx に委譲する（images.py）。しかし run-local.sh の
ような nginx を持たないネイティブ起動では配信元が居らず画像が 404 になる。

本モジュールは `XIMA_SERVE_FILES_NATIVE` が真のときだけ有効化し、core 自身が
workspaces 配下の実ファイルを FileResponse で返す。Docker/nginx 構成では
ルータを登録しない（env 未設定）ため既存挙動は完全に不変。

設計正本: docs/13_local_auth_and_launch.md。
"""

from __future__ import annotations

import mimetypes
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .config import ConfigManager
from .utils.paths import safe_resolve

# nginx の `location /static/ { alias workspaces_root; }` 等と同じ写像。
# いずれの prefix も workspaces_root/<rest> を指す。
_WS_PREFIXES = ("/static", "/thumbs")

# webp サムネは content-addressed（sha 名）なので長期キャッシュ可。
_CACHE_CONTROL = "public, max-age=86400"


def native_file_serving_enabled() -> bool:
    """core 自身が workspaces 配下の実ファイルを配信するか（nginx 非在時）。"""
    return os.environ.get("XIMA_SERVE_FILES_NATIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def serve_ws_file(config_manager: ConfigManager, rel_path: str) -> FileResponse:
    """workspaces_root/<rel_path> の実ファイルを返す（パストラバーサル防御込み）。"""
    ws_root = config_manager.get_config().workspaces_root
    try:
        resolved = safe_resolve(ws_root, rel_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="file not found")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return FileResponse(
        resolved,
        media_type=media_type,
        headers={"Cache-Control": _CACHE_CONTROL},
    )


def create_native_files_router(config_manager: ConfigManager) -> APIRouter:
    """`/static/{path}` `/thumbs/{path}` を workspaces から配信するルータ。

    native モードのみ main.py から登録する（"/" catch-all より前に）。
    """
    router = APIRouter()

    @router.get("/static/{path:path}")
    def serve_static(path: str):
        return serve_ws_file(config_manager, path)

    @router.get("/thumbs/{path:path}")
    def serve_thumbs(path: str):
        return serve_ws_file(config_manager, path)

    return router
