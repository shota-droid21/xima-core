"""閾値ごとに「予測が実際どれくらい当たるか」を val で実測する（T2-2 の一括確定用）。

なぜ要るか:

    一括確定は「**この確率以上なら人が見ずに確定してよい**」という操作である。
    ところが適切な閾値は head ごとに全く違う。実データ（3 head・約 670 件ずつ）では、
    0.9 以上に乗るのは character が 34 件、hair_color が **0 件**、eye_color が 1 件だった。
    共通の閾値を 1 つ置くと、head によっては 1 件も確定できないか、逆に当たらない帯まで
    確定してしまう。

    かといって利用者に数字を選ばせても、その数字が何 % 当たるのかは誰も知らない。
    **だから実測する。**val の item は人のラベルと予測の両方を持つので、
    「閾値 t 以上の予測は n 件あり、そのうち a の割合で人のラベルと一致した」が出せる。
    UI はこの表を見せ、実測点の中から閾値を選ばせる。

なぜ val に限るか:

    train の item は head がそれを見て学習しているので、一致率は本番より高く出る。
    実測で character は全ラベル済み 0.968 に対し val だけなら 0.932 だった。
    **一括確定は「見ずに確定する」判断なので、楽観側に振れた数字を見せてはいけない。**
    val が取れない場合は basis="labeled" として、その旨を UI に出せるようにする。

torch / numpy に依存しない（`label_predictions.py` と同じ方針）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# UI に出す閾値の刻み。細かくしても選ぶ側が判断できず、粗すぎると
# hair_color のように上の帯が空の head で選択肢が無くなる。
THRESHOLDS: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def val_source_paths(index_path: Path) -> Set[str]:
    """dataset/index.json から val の `source_path` を集める。

    `source_path` は labels.json の item の `path` と同じ値である
    （`apply_label_mapping.py` がそこから作る）。読めなければ空集合を返し、
    呼び出し側が basis="labeled" に落とせるようにする。
    """
    try:
        with index_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return set()
    return {
        str(it.get("source_path"))
        for it in items
        if isinstance(it, dict)
        and str(it.get("split") or "").strip().lower() == "val"
        and it.get("source_path")
    }


def item_source_path(item: Mapping[str, Any]) -> str:
    """labels.json の item を dataset/index.json と突き合わせるためのキー。"""
    return str(item.get("path") or item.get("rel_path") or "")


def is_agreement(predicted_value: Any, label_value: Any, *, head_type: str) -> bool:
    """予測が人のラベルと一致したか。

    multi_label は**集合として完全一致**したときだけ一致と見なす。部分一致を
    一致に数えると、一括確定で「1 つ足りない / 1 つ余分な」ラベルが付いた状態を
    「当たった」と表示してしまう。確定するのは集合そのものなので、評価も集合で行う。
    """
    if head_type == "multi_label":
        pred = {str(v) for v in (predicted_value or []) if str(v).strip()}
        gold_raw = label_value if isinstance(label_value, (list, tuple)) else [label_value]
        gold = {str(v) for v in (gold_raw or []) if str(v).strip()}
        return pred == gold
    return str(predicted_value or "").strip() == str(label_value or "").strip()


def agreement_points(
    samples: Iterable[Tuple[float, bool]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> List[Dict[str, Any]]:
    """(confidence, 一致したか) の列から、閾値ごとの件数と一致率を出す。

    件数 0 の閾値も **落とさずに残す。**「その帯には 1 件も無い」こと自体が
    利用者の判断材料になる（hair_color の 0.9 以上が 0 件だったように、
    選んでも何も起きない閾値を選ばせないため）。
    """
    collected = [(float(c), bool(ok)) for c, ok in samples]
    points: List[Dict[str, Any]] = []
    for t in thresholds:
        subset = [ok for c, ok in collected if c >= t]
        n = len(subset)
        points.append(
            {
                "threshold": round(float(t), 4),
                "n": n,
                "agreement": round(sum(1 for ok in subset if ok) / n, 6) if n else None,
            }
        )
    return points


def collect_samples(
    items: Sequence[Mapping[str, Any]],
    *,
    head: str,
    head_type: str,
    confidence_by_file_id: Mapping[str, float],
    predicted_by_file_id: Mapping[str, Any],
    limit_to_paths: Optional[Set[str]] = None,
) -> List[Tuple[float, bool]]:
    """人のラベルと予測の両方を持つ item から (confidence, 一致したか) を集める。

    `limit_to_paths` を渡すとその集合（＝val）だけに絞る。
    """
    from label_predictions import has_label, is_delete_flagged

    samples: List[Tuple[float, bool]] = []
    for item in items:
        if not isinstance(item, dict) or is_delete_flagged(item):
            continue
        if limit_to_paths is not None and item_source_path(item) not in limit_to_paths:
            continue
        if not has_label(item, head):
            continue
        file_id = str(item.get("file_id") or "")
        if file_id not in confidence_by_file_id:
            continue
        gold = (item.get("labels") or {}).get(head)
        samples.append(
            (
                confidence_by_file_id[file_id],
                is_agreement(predicted_by_file_id[file_id], gold, head_type=head_type),
            )
        )
    return samples


def build_reliability(
    heads: Mapping[str, Dict[str, Any]],
    *,
    run_name: str,
    basis: str,
    now: str,
) -> Dict[str, Any]:
    """labels.json の meta に入れる形にまとめる。

    meta へ置くのは、これが item ごとの値ではなく **run 全体の性質**だからである。
    UI（一括確定ダイアログ）は labels.json を読むだけでこの表を得られる。
    """
    return {"run": run_name, "at": now, "basis": basis, "heads": dict(heads)}
