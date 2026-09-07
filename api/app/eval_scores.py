"""推論結果（eval/scores_*.json）の閲覧・エクスポート用ロジック。

`scores_*.json` は infer_heads が書く**正本**であり、本モジュールは書き換えない。
UI で表示するために「1行 = 1画像」の形へ整形し、CSV へ落とすところまでを担う。

experiments.py を肥大化させないため独立モジュールにしてある。
FastAPI に依存しない純粋な関数群なので、そのまま単体テストできる。
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

# eval ディレクトリ配下の scores ファイルのみを許可する（パストラバーサル防止）。
SCORES_NAME_RE = re.compile(r"^scores_[A-Za-z0-9_.\-]+\.json$")


class ScoresNameError(ValueError):
    """許可されないスコアファイル名。"""


def validate_scores_name(name: str) -> str:
    """`scores_*.json` 以外、およびパス区切りを含む名前を拒否する。"""
    value = (name or "").strip()
    if not value or not SCORES_NAME_RE.fullmatch(value):
        raise ScoresNameError(
            "invalid scores file name (expected: scores_<run>.json)"
        )
    return value


def resolve_scores_path(eval_dir: Path, name: str) -> Path:
    """eval ディレクトリ内に収まることを実パスでも検証して返す。"""
    safe_name = validate_scores_name(name)
    path = (eval_dir / safe_name).resolve()
    eval_root = eval_dir.resolve()
    if path != eval_root and eval_root not in path.parents:
        raise ScoresNameError("scores file must be inside the eval directory")
    return path


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid json object: {path}")
    return data


def index_items_by_id(index_data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """dataset/index.json を id 引きできる形にする（無ければ空）。"""
    out: Dict[str, Dict[str, Any]] = {}
    if not index_data:
        return out
    for item in index_data.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id is None:
            continue
        out[str(item_id)] = item
    return out


def top_prediction(scores: Any) -> Optional[Tuple[str, float]]:
    """1 ヘッド分のクラス確率から最尤クラスを取る。

    同率の場合はクラス名昇順で安定させる（表示・CSV が実行ごとにブレないように）。
    """
    if not isinstance(scores, dict) or not scores:
        return None
    best: Optional[Tuple[str, float]] = None
    for label, value in sorted(scores.items(), key=lambda kv: str(kv[0])):
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if best is None or score > best[1]:
            best = (str(label), score)
    return best


def resolve_heads(meta: Dict[str, Any], items: List[Dict[str, Any]]) -> List[str]:
    """出力対象の head を決める。meta が無い古い結果は items から推定する。"""
    heads = meta.get("heads") if isinstance(meta, dict) else None
    if isinstance(heads, list) and heads:
        return [str(h) for h in heads]

    found: List[str] = []
    for item in items:
        for key, value in item.items():
            if key == "id" or not isinstance(value, dict):
                continue
            if key not in found:
                found.append(key)
    return found


def build_rows(
    items: List[Dict[str, Any]],
    index_by_id: Dict[str, Dict[str, Any]],
    heads: List[str],
    *,
    dataset_url_base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """scores の items を「1行 = 1画像」に整形する。

    dataset/index.json と突き合わせて split と画像 URL を補う。
    index が無い / 一致しない場合でも、推論結果そのものは表示できるようにする。
    """
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", ""))
        index_item = index_by_id.get(item_id) or {}

        image_url = None
        dataset_path = index_item.get("dataset_path")
        if dataset_url_base and isinstance(dataset_path, str) and dataset_path:
            image_url = f"{dataset_url_base.rstrip('/')}/{dataset_path.lstrip('/')}"

        predictions: Dict[str, Any] = {}
        for head in heads:
            best = top_prediction(item.get(head))
            predictions[head] = (
                {"label": best[0], "score": best[1]} if best is not None else None
            )

        rows.append(
            {
                "id": item_id,
                "split": index_item.get("split"),
                "image_url": image_url,
                "predictions": predictions,
            }
        )
    return rows


def paginate(rows: List[Dict[str, Any]], *, limit: int, offset: int) -> List[Dict[str, Any]]:
    start = max(0, int(offset))
    size = max(1, int(limit))
    return rows[start : start + size]


def csv_header(heads: Iterable[str]) -> List[str]:
    header = ["id", "split"]
    for head in heads:
        header.extend([str(head), f"{head}_score"])
    return header


def rows_to_csv(rows: List[Dict[str, Any]], heads: List[str]) -> str:
    """表示中の行ではなく、渡された全行を CSV 化する。

    列は `id, split, <head>, <head>_score ...`。
    クラスごとの全確率は raw JSON 側にあるため CSV には含めない（列数が可変になるため）。
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(csv_header(heads))

    for row in rows:
        record: List[Any] = [row.get("id", ""), row.get("split") or ""]
        predictions = row.get("predictions") or {}
        for head in heads:
            pred = predictions.get(head)
            if isinstance(pred, dict):
                record.append(pred.get("label", ""))
                score = pred.get("score")
                record.append(f"{float(score):.6f}" if score is not None else "")
            else:
                record.extend(["", ""])
        writer.writerow(record)

    return buffer.getvalue()


# 一覧に載せてよい meta の項目（#282）。
#
# **`run_dir` / `index_path` / `schema_path` は載せない。** これらは絶対パスで、
# 実行した機械のディレクトリ構成がそのまま入る。画面にも API にも出す理由が無い。
# run の識別子だけが要るので、`run_dir` の末尾だけを `run` として出す。
_LISTING_META_KEYS = ("clip_model_name", "heads", "head_types", "generated_at")


def listing_meta(path: Path) -> Optional[Dict[str, Any]]:
    """scores ファイルの meta から、一覧に出してよい項目だけを取り出す。

    どの学習結果を、どのモデルの埋め込みで測ったのかは meta にしか無い。
    一覧に無いと、ファイルを 1 つずつ開いて確かめることになる（#282）。

    読めない・壊れているファイルでは None を返す。**一覧そのものは出す。**
    1 つのファイルが壊れているせいで測定結果が 1 件も見えなくなる方が困る。
    """
    try:
        data = load_json_file(path)
    except Exception:
        return None

    meta = data.get("meta")
    if not isinstance(meta, dict):
        return None

    out: Dict[str, Any] = {}
    for key in _LISTING_META_KEYS:
        value = meta.get(key)
        if value not in (None, "", [], {}):
            out[key] = value

    run_dir = meta.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        # 末尾のディレクトリ名だけ。絶対パスは出さない。
        out["run"] = PurePosixPath(run_dir.replace("\\", "/")).name

    return out or None
