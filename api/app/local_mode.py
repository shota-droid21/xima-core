"""local モードの配線（ミドルウェア / /local/session / CORS / 静的配信）。

設計正本: docs/13_local_auth_and_launch.md。main.py を肥大化させないため分離。
open / external モードでは強制を行わず、既存挙動を保つ。
"""

from __future__ import annotations

import os
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth_mode import AUTH_MODE_LOCAL, resolve_agent_auth_mode
from .local_auth import LOCAL_KEY_HEADER, get_or_create_local_secret, verify_local_key

# key 検証を免除するパス（app がまだ key を持てない導線・静的配信・ヘルス）。
#
# app シェルの静的物をここに入れる必要があるのは、**ブラウザが自動で取りに行く
# サブリソースには X-Xima-Local-Key を載せられない**ため。favicon や manifest は
# ページの JS を経由せずブラウザ自身が要求するので、免除しないと 401 になり、
# 「タブに既定アイコンが出るだけ」という原因の分かりにくい形で失敗する。
_EXEMPT_EXACT = frozenset(
    {
        "/health",
        "/_auth",
        "/local/session",
        # ブラウザが自動取得する既知の静的物。app dist に無ければ 404 になるだけで害はない。
        "/favicon.ico",
        "/favicon.svg",
        "/apple-touch-icon.png",
        "/manifest.webmanifest",
        "/robots.txt",
    }
)
_EXEMPT_PREFIX = ("/assets", "/local/")


def _is_exempt(path: str) -> bool:
    if path == "/" or path in _EXEMPT_EXACT:
        return True
    return any(path.startswith(p) for p in _EXEMPT_PREFIX)


def local_cors_origins() -> list[str]:
    """local モードで許可するオリジン（既定は同一オリジンのみ＝空）。

    dev で Vite 別ポートから叩く場合などに XIMA_LOCAL_ALLOWED_ORIGINS で追加する
    （カンマ区切り）。同一オリジン要求はそもそも CORS を要さない。
    """
    raw = os.environ.get("XIMA_LOCAL_ALLOWED_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def setup_local_mode(app, config_manager) -> None:
    """local モードのミドルウェアと /local/session を app に組み込む。"""
    state_dir = config_manager.state_dir

    def _secret() -> str:
        return get_or_create_local_secret(state_dir)

    @app.middleware("http")
    async def _local_key_guard(request: Request, call_next):
        # local 以外は素通し（open/external の既存挙動を保つ）。
        if resolve_agent_auth_mode() != AUTH_MODE_LOCAL:
            return await call_next(request)
        if request.method == "OPTIONS" or _is_exempt(request.url.path):
            return await call_next(request)
        provided = request.headers.get(LOCAL_KEY_HEADER)
        if not verify_local_key(provided, _secret()):
            return JSONResponse({"detail": "local key required"}, status_code=401)
        return await call_next(request)

    @app.get("/local/session")
    def local_session(request: Request):
        """同一オリジンの app にシークレットを手渡す（local モードのみ）。

        主制御は CORS（クロスオリジンはレスポンスを読めない）。本チェックは
        Origin ヘッダによる防御多重化。
        """
        if resolve_agent_auth_mode() != AUTH_MODE_LOCAL:
            return JSONResponse({"detail": "not in local mode"}, status_code=404)
        origin = request.headers.get("origin")
        allowed = local_cors_origins()
        if origin and origin not in allowed:
            # 同一オリジン GET は Origin を送らない。異オリジンかつ未許可は拒否。
            return JSONResponse({"detail": "cross-origin forbidden"}, status_code=403)
        return {"key": _secret(), "header": LOCAL_KEY_HEADER}


def mount_app_static(app) -> bool:
    """XIMA_APP_DIST が指すビルド済み app を同一オリジンで静的配信する。

    設定が無い / ディレクトリが無い場合は何もしない（dev/CI は Vite 併用）。
    ルータ登録の後に呼ぶこと（"/" catch-all が API を隠さないため）。
    """
    dist = os.environ.get("XIMA_APP_DIST")
    if not dist:
        return False
    path = Path(dist)
    if not path.is_dir():
        return False
    # 遅延 import: StaticFiles は starlette 同梱で dependency-free 制約に抵触しない。
    from starlette.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(path), html=True), name="app")
    return True
