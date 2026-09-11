"""`GET .../dataset-preview` —— 作り直す前に件数を出す（#325 A-4）。

数え方は `dataset_preview.py`、規則は `pipeline/dataset_split.py` にある。
ここは配線だけで、**判定を 1 行も持たない**。

**`label-input` の下に置かない。** あちらは GET した文書をそのまま PUT へ返す
流れがあり、応答に鍵を足すと `labels.json` へ書き込まれてしまう（#332 で
`schema_violations` を別口にしたのと同じ理由）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .config import ConfigManager
from .cluster_scope import load_schema
from .dataset_preview import head_ids_of, preview
from .label_input import load_label_json


def create_dataset_preview_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    @router.get("/workspaces/{workspace}/experiments/{experiment}/dataset-preview")
    def get_dataset_preview(
        workspace: str,
        experiment: str,
        val_ratio: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> dict:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            data = load_label_json(label_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail="labels.json not found for this experiment"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
        schema = load_schema(cfg, workspace, experiment)
        return {
            "workspace": workspace,
            "experiment": experiment,
            **preview(items, head_ids=head_ids_of(schema), val_ratio=val_ratio),
        }

    return router
