"""類似画像クラスタリング API。

`cache/embeddings`（#102）を読み、似た画像をクラスタにまとめて返す読み取り専用 API。
UI 側はこの結果を使って「クラスタ単位でまとめてラベル付与」する（付与自体は既存の
`PUT .../label-input` に乗せるため、labels.json 正本フロー・履歴は無改変）。

依存の切り分け:

- クラスタリングの中核ロジックは torch / numpy 非依存の `pipeline.clustering` にある。
- 本モジュールは行列（`embeddings.npy`）の読み込みにのみ numpy を使うが、
  **import は関数内に閉じ込める**。これにより CI（numpy 無し）でも、ローダを
  スタブ化すればルーターを結合テストできる。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from .config import ConfigManager
from .label_input import (
    load_label_json,
    normalize_label_schema_payload,
    thumb_path_from_file_id,
)

# pipeline はスクリプト実行時と同じくトップレベル名で解決する。
_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from clustering import run_clustering  # noqa: E402
from embedding_cache import (  # noqa: E402
    MATRIX_FILENAME,
    embeddings_dir,
    load_index,
)

# 埋め込みに使う既定 CLIP モデル（embed_images の既定と一致させる）。
DEFAULT_CLIP_MODEL = "ViT-B/32"
_SPLIT_HEAD_ID = "split"


def _experiment_cache_root(cfg, workspace: str, experiment: str) -> Path:
    """experiments/<exp>/cache（embed_images と同じ基準）。"""
    label_path = cfg.label_input_path_for(workspace, experiment)
    # .../experiments/<exp>/label_input/labels.json -> .../experiments/<exp>/cache
    return label_path.parent.parent / "cache"


def _load_embeddings(
    cache_dir: Path,
) -> Optional[Tuple[List[Dict[str, Any]], List[List[float]]]]:
    """index.json + embeddings.npy を読み、(items, vectors) を返す。

    キャッシュが無ければ None（＝呼び出し側で 409）。numpy はここでのみ使う。
    """
    index = load_index(cache_dir)
    if not index:
        return None
    items = [it for it in (index.get("items") or []) if isinstance(it, dict)]
    items.sort(key=lambda it: int(it.get("row", 0)))

    import numpy as np  # 実行時のみ（CI では本関数をスタブ化する）

    matrix_path = cache_dir / MATRIX_FILENAME
    matrix = np.load(matrix_path)
    if matrix.ndim != 2:
        raise ValueError("embeddings matrix must be 2D")

    vectors: List[List[float]] = []
    for it in items:
        row = int(it.get("row", -1))
        if row < 0 or row >= matrix.shape[0]:
            raise ValueError("embeddings index/matrix row mismatch")
        vectors.append([float(x) for x in matrix[row].tolist()])
    return items, vectors


def _load_schema(cfg, workspace: str, experiment: str) -> Dict[str, Any]:
    try:
        schema_path = cfg.label_schema_path_for(workspace, experiment)
        return normalize_label_schema_payload(load_label_json(schema_path))
    except (FileNotFoundError, ValueError):
        return {"heads": []}


def _default_head_id(schema: Dict[str, Any]) -> Optional[str]:
    """split 以外で最初に見つかった分類 head の id。"""
    for head in schema.get("heads") or []:
        if not isinstance(head, dict):
            continue
        hid = str(head.get("id") or "").strip()
        if hid and hid != _SPLIT_HEAD_ID:
            return hid
    return None


def _labeled_file_ids(cfg, workspace: str, experiment: str, head_id: str) -> set[str]:
    """指定 head で既にラベル済みの file_id 集合。"""
    try:
        label_path = cfg.label_input_path_for(workspace, experiment)
        data = load_label_json(label_path)
    except (FileNotFoundError, ValueError):
        return set()

    labeled: set[str] = set()
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        fid = item.get("file_id")
        if not isinstance(fid, str):
            continue
        value = (item.get("labels") or {}).get(head_id)
        if value not in (None, "", [], {}):
            labeled.add(fid)
    return labeled


def _member_view(
    item: Dict[str, Any], workspace: str, experiment: str, width: int
) -> Dict[str, Any]:
    fid = str(item.get("file_id") or "")
    view: Dict[str, Any] = {"file_id": fid, "path": item.get("path")}
    try:
        view["thumb_path"] = thumb_path_from_file_id(
            workspace, experiment, fid, width=width
        )
    except ValueError:
        view["thumb_path"] = None
    return view


def create_clustering_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    @router.get("/workspaces/{workspace}/experiments/{experiment}/clusters")
    def get_clusters(
        workspace: str,
        experiment: str,
        k: Optional[int] = None,
        scope: str = "all",
        head: Optional[str] = None,
        width: int = 256,
        seed: int = 0,
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()

        if scope not in ("all", "unlabeled"):
            raise HTTPException(
                status_code=400, detail="scope must be 'all' or 'unlabeled'"
            )
        if k is not None and k < 1:
            raise HTTPException(status_code=400, detail="k must be >= 1")

        try:
            cache_dir = embeddings_dir(
                _experiment_cache_root(cfg, workspace, experiment), DEFAULT_CLIP_MODEL
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        loaded = _load_embeddings(cache_dir)
        if loaded is None:
            # 埋め込み未作成。UI に「先に埋め込みを作る」導線を促す。
            raise HTTPException(
                status_code=409,
                detail=(
                    "embeddings not found; run the embed_images job first "
                    "(cache/embeddings is empty)"
                ),
            )
        items, vectors = loaded

        resolved_head = head
        if scope == "unlabeled":
            if not resolved_head:
                resolved_head = _default_head_id(
                    _load_schema(cfg, workspace, experiment)
                )
            if resolved_head:
                labeled = _labeled_file_ids(
                    cfg, workspace, experiment, resolved_head
                )
                kept = [
                    (it, vec)
                    for it, vec in zip(items, vectors)
                    if str(it.get("file_id") or "") not in labeled
                ]
                items = [it for it, _ in kept]
                vectors = [vec for _, vec in kept]

        result = run_clustering(vectors, k=k, seed=seed)

        clusters_out: List[Dict[str, Any]] = []
        for cluster in result["clusters"]:
            members = [
                _member_view(items[i], workspace, experiment, width)
                for i in cluster["members"]
            ]
            clusters_out.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "size": cluster["size"],
                    "representative": _member_view(
                        items[cluster["representative"]], workspace, experiment, width
                    ),
                    "members": members,
                }
            )

        return {
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "clip_model_name": DEFAULT_CLIP_MODEL,
            "scope": scope,
            "head": resolved_head,
            "k": result["k"],
            "total": len(items),
            "clusters": clusters_out,
        }

    return router
