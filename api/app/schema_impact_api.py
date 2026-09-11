"""スキーマ変更の影響照会と、値の付け替え（#330）。

2 本ある。

| | 何をするか |
| --- | --- |
| `POST .../label-schema/impact` | 新しいスキーマを渡すと、**行き先を失う値**を返す。**何も書かない** |
| `POST .../label-schema/remap` | `{head, from, to}` を当てて labels.json を書き換える |

**`PUT /label-schema` は拒否しない。** 止めると、スキーマを直す作業そのものが
2 段階になり、script からは触れなくなる。#330 で実際に困ったのは「変更できたこと」
ではなく **「気づけなかったこと」** である。だから PUT は通したうえで、応答に
`orphans` を載せる（#332 の `schema_violations` と同じ立場）。画面は保存の前に
このエンドポイントで聞き、行き先を失う値があればダイアログを出す。

**順番は「スキーマを保存 → 付け替え」。** 逆にすると、付け替え先が新しいスキーマの
クラス（改名した先など）のとき、**まだ保存されていないスキーマから見て不正な値**を
labels.json へ書くことになる。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException

from .cluster_scope import load_schema
from .config import ConfigManager
from .label_input import (
    load_label_json,
    normalize_label_schema_payload,
    validate_label_schema_payload,
    write_history_snapshot,
)
from .schema_impact import apply_remap, get_head_classes, head_map, orphans
from .utils.atomic_io import write_text_atomic


def _items_of(label_path) -> List[Dict[str, Any]]:
    data = load_label_json(label_path)
    if not isinstance(data, dict):
        raise ValueError("labels.json is not an object")
    return data


def create_schema_impact_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    def _label_path(workspace: str, experiment: str):
        cfg = config_manager.get_config()
        try:
            return cfg, cfg.label_input_path_for(workspace, experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post(
        "/workspaces/{workspace}/experiments/{experiment}/label-schema/impact"
    )
    def post_label_schema_impact(
        workspace: str,
        experiment: str,
        payload: Dict[str, Any] = Body(...),
    ) -> dict:
        """新しいスキーマにしたら、どの値が行き先を失うか。**何も書かない。**"""
        cfg, label_path = _label_path(workspace, experiment)
        try:
            new_schema = normalize_label_schema_payload(payload)
            validate_label_schema_payload(new_schema)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            data = _items_of(label_path)
        except FileNotFoundError:
            # まだラベルが 1 件も無い。失う値も無い。
            return {"workspace": workspace, "experiment": experiment,
                    "heads": [], "items": 0, "entering": 0}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
        return {
            "workspace": workspace,
            "experiment": experiment,
            **orphans(
                items,
                new_schema,
                old_schema=load_schema(cfg, workspace, experiment),
            ),
        }

    @router.post(
        "/workspaces/{workspace}/experiments/{experiment}/label-schema/remap"
    )
    def post_label_schema_remap(
        workspace: str,
        experiment: str,
        payload: Dict[str, Any] = Body(...),
    ) -> dict:
        """`{head, from, to}` を当てて labels.json を書き換える。

        `to` が空なら値を消す。**書く前に履歴へ残す**ので、間違えても戻せる。
        """
        cfg, label_path = _label_path(workspace, experiment)
        remaps = payload.get("remaps")
        if not isinstance(remaps, list) or not remaps:
            raise HTTPException(status_code=400, detail="remaps must be a non-empty list")

        schema_heads = head_map(load_schema(cfg, workspace, experiment))
        for remap in remaps:
            if not isinstance(remap, dict):
                raise HTTPException(status_code=400, detail="each remap must be an object")
            head = str(remap.get("head") or "").strip()
            target = str(remap.get("to") or "").strip()
            if not head:
                raise HTTPException(status_code=400, detail="remap.head is required")
            if not target:
                continue  # 値を消すだけなので、行き先の確認は要らない
            head_def = schema_heads.get(head)
            classes = get_head_classes(head_def) if head_def else []
            if head_def is None or (classes and target not in classes):
                # **行き先が無い付け替えを受け付けない。** 受け付けると #330 の
                # 状態を作り直すだけになる。スキーマを先に保存してから呼ぶこと。
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"remap target is not a class of head '{head}'. "
                        "Save the new label schema first."
                    ),
                )

        try:
            data = _items_of(label_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="labels.json not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
        next_items, changed = apply_remap(items, remaps)
        if not changed:
            return {"status": "unchanged", "changed": {}, "version": None}

        next_data = dict(data)
        next_data["items"] = next_items
        serialized = json.dumps(next_data, ensure_ascii=False, indent=2)
        # 書く**前**の姿を履歴に残す。付け替えは元に戻せない書き換えである。
        version_path = write_history_snapshot(
            label_path, json.dumps(data, ensure_ascii=False, indent=2)
        )
        write_text_atomic(label_path, serialized)
        return {
            "status": "saved",
            "changed": changed,
            "items": sum(changed.values()),
            "version": str(version_path),
        }

    return router
