"""**値はあるのに学習に使えないラベル**を数える（#333）。

データセット生成と学習は、スキーマに無いクラスの値を**黙って外す**。落ち方は
head の型で違い、**片方は結果が静かに歪む**。

| head の型 | 何が起きるか |
| --- | --- |
| `multi_class` | `build_targets` の `class_to_idx.get(..., -1)` が -1 になり、**その item はこの head の学習・検証から丸ごと外れる**（`valid=0`） |
| `multi_label` | `_normalize_multi_label_value` が**その値だけ**落とす。item は学習に残るので、**「その属性は付いていない」と教えることになる** |

`multi_label` の方が静かで害が大きい。件数が減るのではなく、**正解が間違ったまま
学習される**。

**どこにも件数が出ていなかった。** 実運用では、ラベルの定義から消したクラスの値を
持つ 19 件（train 18 / val 1）が学習から外れていたが、`apply_label` の
`processed / skipped` にも `run_meta.json` にも現れなかった。val の 1 件は評価からも
消えている。

`skipped` は「dataset の外にある（`split` が train/val でない）」という**正常な状態**
（#316）であって、これとは別の数である。

**この 1 つを `apply_label_mapping` と `train_epoch` の両方から使う。** 別々に数えると
片方だけ直したときに数が食い違い、どちらが正しいか分からなくなる。

**ただし見えている範囲が違う。**

| 呼ぶ側 | 読むもの | 拾えるもの |
| --- | --- | --- |
| `apply_label_mapping` | `labels.json` | **両方**（`multi_label` が落ちるのはここ） |
| `train_epoch` | `dataset/index.json` | **`multi_class` だけ** |

`multi_label` の落ちた値は `apply_label_mapping` が `index.json` を書く時点で消えて
いるので、学習側からはもう見えない。**だから生成側で数える必要がある。**
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from label_schema import (
    _normalize_scalar_label_value,
    canonical_head_type,
    raw_label_values,
)


def _labels_of(item: Any) -> Dict[str, Any]:
    """`ItemRec`（学習側）と生の dict（生成側）の両方から labels を取る。"""
    labels = getattr(item, "labels", None)
    if labels is None and isinstance(item, dict):
        labels = item.get("labels")
    return labels if isinstance(labels, dict) else {}


def unusable_values(
    value: Any, *, head_type: str, classes: Sequence[str]
) -> List[str]:
    """その値のうち、`classes` に無いもの。

    `classes` が空のときは**判定できない**ので空を返す。スキーマにクラスが無い
    head は、学習側が items から集めて作る（`collect_classes_from_items`）ため、
    そこでは「無いクラス」が原理的に存在しない。
    """
    allowed = set(classes)
    if not allowed:
        return []

    kind = canonical_head_type(head_type)
    if kind == "multi_label":
        candidates: Iterable[str] = raw_label_values(value)
    else:
        scalar = _normalize_scalar_label_value(value)
        candidates = [scalar] if scalar else []

    return [v for v in candidates if v not in allowed]


def count_unusable(
    items: Sequence[Any],
    *,
    class_head: str,
    head_type: str,
    classes: Sequence[str],
) -> Tuple[Dict[str, int], int]:
    """その head で使えない値を数える。

    返すのは `(値ごとの件数, そういう item の数)`。

    `multi_label` は 1 つの item が複数の値で落ちうるので、**値の合計と item の数は
    一致しない**。両方返すのはそのためである。
    """
    values: Dict[str, int] = {}
    n_items = 0
    for item in items:
        bad = unusable_values(
            _labels_of(item).get(class_head), head_type=head_type, classes=classes
        )
        if not bad:
            continue
        n_items += 1
        for value in bad:
            values[value] = values.get(value, 0) + 1
    return values, n_items


def summarize(values: Dict[str, int], n_items: int) -> Optional[Dict[str, Any]]:
    """`run_meta` / summary に載せる形。0 件なら `None`（鍵ごと出さない）。"""
    if n_items <= 0:
        return None
    return {"items": n_items, "values": dict(sorted(values.items()))}


def describe(values: Dict[str, int], n_items: int, *, where: str) -> Optional[str]:
    """人が読む 1 行。`run_meta` の `problems` に混ぜて画面へ出す。

    画面には既に `problems` を出す口があるので（`TrainingProblems.tsx`）、
    **新しい表示を作らずに済む**。
    """
    if n_items <= 0:
        return None
    listed = ", ".join(
        f"{value}（{count} 件）" for value, count in sorted(values.items())
    )
    return (
        f"ラベルの定義に無い値が {n_items} 件あり、{where}に使えていません: {listed}。"
        "ラベルの定義にその値を足すか、画像側の値を直してください。"
    )
