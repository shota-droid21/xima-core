from __future__ import annotations

from fastapi import APIRouter

from .config import ConfigManager
from .utils.short_id import is_valid_short_id


def create_validate_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter(prefix="/validate")

    @router.get("/workspace-id")
    def validate_workspace_id(id: str) -> dict:
        cfg = config_manager.get_config()
        raw = (id or "").strip().lower()
        valid = is_valid_short_id(raw)
        exists = False
        if valid:
            exists = (cfg.workspaces_root / raw).exists()
        return {"id": raw, "valid": valid, "exists": exists}

    @router.get("/experiment-id")
    def validate_experiment_id(workspace: str, id: str) -> dict:
        cfg = config_manager.get_config()
        ws_raw = (workspace or "").strip().lower()
        ws_valid = is_valid_short_id(ws_raw)
        ws_root = cfg.workspaces_root / ws_raw
        workspace_exists = ws_valid and ws_root.exists() and ws_root.is_dir()

        raw = (id or "").strip().lower()
        valid = is_valid_short_id(raw)
        exists = False
        if valid and workspace_exists:
            # Avoid calling cfg.experiments_path_for with invalid workspace id.
            exists = (cfg.experiments_path_for(ws_raw) / raw).exists()

        return {
            "workspace": ws_raw,
            "workspace_valid": ws_valid,
            "workspace_exists": workspace_exists,
            "id": raw,
            "valid": valid,
            "exists": exists,
        }

    @router.post("/workspace-id")
    def validate_workspace_id_post(body: dict) -> dict:
        # convenience for clients that prefer POST
        return validate_workspace_id(id=str(body.get("id", "")))

    @router.post("/experiment-id")
    def validate_experiment_id_post(body: dict) -> dict:
        return validate_experiment_id(
            workspace=str(body.get("workspace", "")), id=str(body.get("id", ""))
        )

    return router
