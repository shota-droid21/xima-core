"""類似画像クラスタリング API。

`cache/embeddings`（#102）を読み、似た画像をクラスタにまとめて返す読み取り専用 API。
UI 側はこの結果を使って「クラスタ単位でまとめてラベル付与」する（付与自体は既存の
`PUT .../label-input` に乗せるため、labels.json 正本フロー・履歴は無改変）。

依存の切り分け:

- クラスタリングの中核ロジックは `pipeline.clustering` にある。純 Python 実装が
  常に在り、numpy が使える実行環境では同手順の高速路に回る（#272）。
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
from .embeddings import collect_models, experiment_cache_root
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

_SPLIT_HEAD_ID = "split"


def _resolve_clip_model(cache_root: Path, requested: Optional[str]) -> str:
    """どのモデルの埋め込みでクラスタを切るかを決める（#261）。

    以前はここが `ViT-B/32` の定数だった。**学習に何を使っていても常に ViT-B/32 で
    切っていた**ため、別モデルで学習した利用者は学習と違う空間のまとまりに対して
    一括ラベル付与をしていた。まとまりがずれたまま付与すると、そのまま学習データの
    質が落ちる。しかも UI にモデルの表示も選択も無く、気づく手段が無かった。

    指定が無いときに既定へ落とさないのは、それが上の不具合そのものだから。
    埋め込みが在るものから選び、1 つも無ければ呼び出し側が 409 にする。
    """
    if requested and requested.strip():
        return requested.strip()

    models = collect_models(cache_root)
    usable = [m for m in models if m.get("usable")]
    if not usable:
        return ""
    # 最も新しく作られたものを既定にする。実際に使うモデルは app が
    # 最新 run の clip_model_name から明示して送るため、ここは API を直接
    # 叩く場合の落としどころ。
    usable.sort(key=lambda m: str(m.get("updated_at") or ""), reverse=True)
    return str(usable[0]["clip_model_name"])


def _load_embeddings(
    cache_dir: Path,
) -> Optional[Tuple[List[Dict[str, Any]], Any]]:
    """index.json + embeddings.npy を読み、(items, vectors) を返す。

    キャッシュが無ければ None（＝呼び出し側で 409）。numpy はここでのみ使う。

    vectors は **numpy の行列のまま返す**。以前は 1 行ずつ Python の list に
    落としていたが、クラスタリング側も numpy で計算するようになったため
    （#272）、変換して戻す意味が無い。テストでは本関数をスタブ化するため、
    list of list を返しても後段はそのまま動く。
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

    rows = [int(it.get("row", -1)) for it in items]
    if any(row < 0 or row >= matrix.shape[0] for row in rows):
        raise ValueError("embeddings index/matrix row mismatch")
    return items, matrix[rows]


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


def _unlabeled_exclusions(
    cfg, workspace: str, experiment: str, head_id: str
) -> set[str]:
    """`scope=unlabeled` で除く file_id の集合。

    除く理由は 2 つある。

    1. その head で **既にラベルが付いている**
    2. **削除マークが付いている**（#293）

    2 は以前は見ていなかった。`delete` は item 直下のフラグで `labels` とは別の
    レイヤーにあり、ここが `labels[head]` しか見ていなかったためである。その結果
    **消すと決めた画像が「残りに付ける対象」として出続けていた**。筋が通らないうえ、
    重複を削除マークで片付けても視界から消えないので、まとまりを 1 つずつ処理する
    使い方が成立しなかった。

    labels.json は 1 度だけ読む。2 つに分けると同じファイルを 2 回読むことになる。
    """
    try:
        label_path = cfg.label_input_path_for(workspace, experiment)
        data = load_label_json(label_path)
    except (FileNotFoundError, ValueError):
        return set()

    excluded: set[str] = set()
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        fid = item.get("file_id")
        if not isinstance(fid, str):
            continue
        if item.get("delete") is True:
            excluded.add(fid)
            continue
        value = (item.get("labels") or {}).get(head_id)
        if value not in (None, "", [], {}):
            excluded.add(fid)
    return excluded


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
        clip_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()

        if scope not in ("all", "unlabeled"):
            raise HTTPException(
                status_code=400, detail="scope must be 'all' or 'unlabeled'"
            )
        if k is not None and k < 1:
            raise HTTPException(status_code=400, detail="k must be >= 1")

        cache_root = experiment_cache_root(cfg, workspace, experiment)
        resolved_model = _resolve_clip_model(cache_root, clip_model)
        if not resolved_model:
            raise HTTPException(
                status_code=409,
                detail=(
                    "embeddings not found; run the embed_images job first "
                    "(cache/embeddings is empty)"
                ),
            )

        try:
            cache_dir = embeddings_dir(cache_root, resolved_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        loaded = _load_embeddings(cache_dir)
        if loaded is None:
            # **どのモデルの埋め込みが要るかまで書く。** 「embed_images を実行しろ」
            # だけでは、既定（ViT-B/32）で作り直して同じ 409 に戻ってくる。
            raise HTTPException(
                status_code=409,
                detail=(
                    f"embeddings not found for clip_model={resolved_model}; "
                    f"run the embed_images job with --clip-model {resolved_model}"
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
                excluded = _unlabeled_exclusions(
                    cfg, workspace, experiment, resolved_head
                )
                keep = [
                    idx
                    for idx, it in enumerate(items)
                    if str(it.get("file_id") or "") not in excluded
                ]
                items = [items[idx] for idx in keep]
                # vectors は numpy 行列でも list でも同じ形で絞れるようにする。
                vectors = (
                    vectors[keep]
                    if hasattr(vectors, "shape")
                    else [vectors[idx] for idx in keep]
                )

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
            "clip_model_name": resolved_model,
            "scope": scope,
            "head": resolved_head,
            "k": result["k"],
            "total": len(items),
            "clusters": clusters_out,
        }

    return router
