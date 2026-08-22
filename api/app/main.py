from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from .auth_mode import AUTH_MODE_EXTERNAL, AUTH_MODE_LOCAL, resolve_agent_auth_mode
from .auth_jwt import require_auth_with_whitelist
from .local_mode import local_cors_origins, mount_app_static, setup_local_mode
from .clustering import create_clustering_router
from .config import ConfigManager
from .demo import create_demo_router
from .entitlement import create_entitlement_router
from .experiments import create_experiments_router
from .images import create_images_router
from .jobs.manager import JobsManager, create_jobs_router
from .label_input import create_label_input_router
from .native_files import create_native_files_router, native_file_serving_enabled
from .trash import create_trash_router
from .validate import create_validate_router
from .utils.identity import ensure_identity
from .utils.version import read_core_version
from .workspace import create_workspace_router

# OpenAPI のタイトル（/docs に表示される）。ワイヤ契約ではないので対外名に合わせる。
# health レスポンスの "agent" フィールドは契約なので下記のとおり据え置く。
#
# version は core/VERSION から読む。ハードコードしていた頃は発行しても値が
# 変わらず、/health が古い版を返し続けていた（#174）。
CORE_VERSION = read_core_version()
app = FastAPI(title="xima-core", version=CORE_VERSION)

# local モードは CORS を app オリジン限定（既定は同一オリジンのみ）。
# open / external は従来どおり無制限（既存挙動を保つ）。
if resolve_agent_auth_mode() == AUTH_MODE_LOCAL:
    _cors_allow_origins = local_cors_origins()
else:
    _cors_allow_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).resolve().parent.parent
config_manager = ConfigManager(base_dir)
jobs_manager = JobsManager(config_manager)

app.include_router(create_workspace_router(config_manager))
app.include_router(create_validate_router(config_manager))
app.include_router(create_experiments_router(config_manager))
app.include_router(create_label_input_router(config_manager))
app.include_router(create_images_router(config_manager))
app.include_router(create_trash_router(config_manager))
app.include_router(create_jobs_router(jobs_manager))
app.include_router(create_demo_router(config_manager, jobs_manager))
app.include_router(create_clustering_router(config_manager))
app.include_router(create_entitlement_router(config_manager))

# nginx 非在のネイティブ起動では core 自身が /static /thumbs を配信する
# （XIMA_SERVE_FILES_NATIVE。Docker/nginx 構成では登録しない＝挙動不変）。
if native_file_serving_enabled():
    app.include_router(create_native_files_router(config_manager))

# local モードのハンドシェイク（ミドルウェア + /local/session）。
setup_local_mode(app, config_manager)


@app.get("/health")
def health() -> dict:
    auth_mode = resolve_agent_auth_mode()

    cfg = config_manager.get_config()
    ws_root = cfg.workspaces_root
    identity = ensure_identity(ws_root, agent_version=CORE_VERSION)

    return {
        "status": "ok",
        "auth_mode": auth_mode,
        "install_id": identity.install_id,
        "container_id": identity.container_id,
        # **実行中の**版を返す。以前は identity.agent_version を返していたが、
        # あれは workspaces を作った時点の記録であり、`ensure_identity` は既存が
        # あればそのまま返すため、更新しても永久に初回の値のままだった（#174）。
        "agent_version": CORE_VERSION,
        # Backward compatibility（ワイヤ契約。app が読むため改称しない・docs/13 参照）
        "version": CORE_VERSION,
        # workspaces を作成した版の記録。実行中の版とは別物なので分けて返す。
        "installed_agent_version": identity.agent_version,
        "agent": "xima-agent",
    }


@app.get("/_auth")
@app.options("/_auth")
def auth_probe(
    request: Request,
    method: str | None = None,  # Query parameter from nginx
    authorization: str | None = Header(default=None),
    x_original_method: str | None = Header(default=None),
) -> dict:
    """
    JWT トークンを検証するエンドポイント。
    nginx の auth_request で利用される。
    Query parameter 'method' が OPTIONS の場合は認証不要で200を返す（CORSプリフライト対応）。
    
    200 を返せば認証成功、401 を返せば認証失敗。
    """
    # Check query parameter 'method' sent by nginx
    if method == "OPTIONS":
        return {"status": "preflight_ok"}
    
    # Check X-Original-Method header sent by nginx
    if x_original_method == "OPTIONS":
        return {"status": "preflight_ok"}
    
    # OPTIONS request for CORS preflight - allow without authentication
    if request.method == "OPTIONS":
        return {"status": "preflight_ok"}
    
    auth_mode = resolve_agent_auth_mode()

    # openモード: 認証を一切行わずに常に200
    if auth_mode != AUTH_MODE_EXTERNAL:
        return {"ok": True, "mode": "open"}

    # externalモード: JWT検証 + claimのagent束縛チェック（Redis whitelist cache付き）
    cfg = config_manager.get_config()
    identity = ensure_identity(cfg.workspaces_root)
    require_auth_with_whitelist(
        authorization,
        expected_install_id=identity.install_id,
        expected_container_runtime_id=identity.container_id,
    )
    return {"ok": True, "mode": AUTH_MODE_EXTERNAL}


# ビルド済み app（dist）の同一オリジン配信。全ルート登録後に mount する
# （"/" catch-all が API/health/_auth を隠さないため）。XIMA_APP_DIST 未設定なら no-op。
mount_app_static(app)
