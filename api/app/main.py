from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from .auth_mode import AUTH_MODE_EXTERNAL, resolve_agent_auth_mode
from .auth_jwt import require_auth_with_whitelist
from .clustering import create_clustering_router
from .config import ConfigManager
from .demo import create_demo_router
from .experiments import create_experiments_router
from .images import create_images_router
from .jobs.manager import JobsManager, create_jobs_router
from .label_input import create_label_input_router
from .pairing_api import create_pairing_router
from .trash import create_trash_router
from .validate import create_validate_router
from .utils.identity import ensure_identity
from .workspace import create_workspace_router

app = FastAPI(title="xima-agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
app.include_router(create_pairing_router(config_manager))
app.include_router(create_demo_router(config_manager, jobs_manager))
app.include_router(create_clustering_router(config_manager))


@app.get("/health")
def health() -> dict:
    auth_mode = resolve_agent_auth_mode()

    cfg = config_manager.get_config()
    ws_root = cfg.workspaces_root
    identity = ensure_identity(ws_root, agent_version=str(app.version))

    return {
        "status": "ok",
        "auth_mode": auth_mode,
        "install_id": identity.install_id,
        "container_id": identity.container_id,
        "agent_version": identity.agent_version,
        # Backward compatibility
        "version": identity.agent_version,
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
