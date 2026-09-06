"""埋め込みキャッシュの所在と状態を返す API。

`cache/embeddings/<model_slug>/` に何が置かれているかを、**ジョブを投げる前に**
UI が知るための読み取り専用エンドポイント（#258）。

なぜ要るか:

    候補付与（predict_labels）とクラスタは、**学習に使ったのと同じ CLIP モデル**の
    埋め込みを必要とする。しかしその有無を外から問い合わせる手段が無かったため、
    UI は順序を利用者の記憶に委ねるしかなく、埋め込みが無い状態でも実行ボタンを
    押せてしまっていた（#259 / #261）。

「使えるか」の判定基準:

    `predict_labels._load_embeddings` と `clustering._load_embeddings` は
    **index.json と embeddings.npy が揃っていること**しか要求しない。
    labels.json の一部に埋め込みが無くても、その画像を飛ばして動く。
    よって `usable` はこの 2 ファイルの存在に一致させる。**件数では判定しない。**

件数について:

    `target_count` は labels.json の item 数で、**ファイルの実在は見ない**。
    実在確認は `embed_images.build_targets` が content_hash とデコード可否まで見るため
    1,000 件規模で秒単位かかり、状態を見るだけの GET には重すぎる。
    したがって `count < target_count` は「まだ作っていない」とは限らず、
    欠損画像や読めない画像が除外された結果でもありうる。**進捗の目安として出す。**
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from .config import ConfigManager

# pipeline はスクリプト実行時と同じくトップレベル名で解決する（clustering.py と同じ）。
_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from embedding_cache import INDEX_FILENAME, MATRIX_FILENAME, load_index  # noqa: E402


def experiment_cache_root(cfg, workspace: str, experiment: str) -> Path:
    """experiments/<exp>/cache（embed_images と同じ基準）。

    .../experiments/<exp>/label_input/labels.json -> .../experiments/<exp>/cache
    """
    label_path = cfg.label_input_path_for(workspace, experiment)
    return label_path.parent.parent / "cache"


def _labels_item_count(cfg, workspace: str, experiment: str) -> int:
    """labels.json の item 数。読めなければ 0（存在しない experiment もありうる）。"""
    # load_label_json は label_input 側の実装を使う（例外の型を揃えるため）。
    from .label_input import load_label_json

    try:
        data = load_label_json(cfg.label_input_path_for(workspace, experiment))
    except (FileNotFoundError, ValueError):
        return 0
    items = data.get("items") if isinstance(data, dict) else None
    return len(items) if isinstance(items, list) else 0


def _updated_at(index_path: Path) -> str | None:
    """index.json の mtime を ISO8601(UTC) で返す。"""
    try:
        ts = index_path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def collect_models(cache_root: Path) -> List[Dict[str, Any]]:
    """cache/embeddings/ 配下を舐めて、モデルごとの状態を返す。

    **embeddings.npy は読まない。** 件数と次元は index.json に入っており、
    行列を読むと 1,000 件規模で無駄な I/O になる。

    版が違う / 壊れている index を持つディレクトリは **一覧に出さない**。
    load_index が None を返すものは embed_images が作り直す対象であり、
    「無い」と扱った方が UI の次の一手（作る）と一致する。
    """
    root = cache_root / "embeddings"
    if not root.is_dir():
        return []

    models: List[Dict[str, Any]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        index = load_index(model_dir)
        if index is None:
            continue
        name = str(index.get("clip_model_name") or "").strip()
        if not name:
            # モデル名の分からないキャッシュは使いようがない（どの run に対応するか
            # 判定できない）。出しても UI が選択肢にできないので落とす。
            continue
        models.append(
            {
                "clip_model_name": name,
                "slug": model_dir.name,
                "count": int(index.get("count") or 0),
                "dim": int(index["dim"]) if index.get("dim") is not None else None,
                "usable": (model_dir / MATRIX_FILENAME).is_file(),
                "updated_at": _updated_at(model_dir / INDEX_FILENAME),
            }
        )
    return models


def create_embeddings_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    @router.get("/workspaces/{workspace}/experiments/{experiment}/embeddings")
    def get_embeddings(workspace: str, experiment: str) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        return {
            "target_count": _labels_item_count(cfg, workspace, experiment),
            "models": collect_models(experiment_cache_root(cfg, workspace, experiment)),
        }

    return router
