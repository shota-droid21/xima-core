"""確定値がどこから来たかを読む（#412 / 再現は #383）。

一括確定は**人が 1 枚も見ていない値**を `labels` に書く。それをそのまま評価の
正解として数えると、**同じモデルの予測を、そのモデルが書いた値と突き合わせる**
ことになり、一致率も `val_acc` も本当より高く出る。

`labels` そのものには来歴が無い（値だけ）。読めるのは `predicted[head]` に
一括確定が残した記録で、app 側の `isBulkConfirmed`（`app/src/lib/predictions.ts`）と
同じ読み方をここへ持ってくる。

## 「未確認」の線をどこで引くか

**`predicted[head].confirmed` があり、かつ `labels[head]` がその値のままなら未確認**
とする。人が後から値を変えれば記録と食い違うので、**変えた時点で未確認ではなくなる。**

`item.committed` は**使わない。**あれは item 単位で、head 単位ではない。1 つの head を
見て commit した人が、別の head も見たとは限らない。使えば「見ていないものを見たこと
にする」側へ倒れる。head ごとの確認状態は #383 / #345 の論点であり、ここでは決めない。

**人が見て「そのままでよい」と判断した値も、未確認として外れる。**区別する情報が
保存されていないためで、**推測で補完しない**（#412 Non-Goals）。標本が減る側に倒れるので、
一致率が本当より高く出ることはない。

## 効かなくなる場面（#383。この issue では直していない）

**候補を作り直すと、この記録は消える。**`record_prediction` は `predicted[head]` を
丸ごと書き換えるので、`confirmed` も一緒に落ちる。

    一括確定 -> 学習 -> もう一度候補を付ける

という普通の流れを 1 周すると、未確認だった値は**人が付けた値と見分けが付かなくなる。**
つまりここの判定が効くのは、**次に候補を付けるまで**である。

半端に引き継ぐことはしない。`confirmed` は「どの値を確定したか」と対であり、
新しい予測が別の値を出したときに記録だけ残すと、**未確認のものを確認済みと読む**方へ
倒れる。来歴をどう保存するかは #383 / #345 の論点なので、そこで決める。
"""

from __future__ import annotations

from typing import Any, Mapping


def _same_value(a: Any, b: Any) -> bool:
    """multi_label は順番を無視して比べる。それ以外は文字列として比べる。"""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return sorted(str(v) for v in a) == sorted(str(v) for v in b)
    return str(a if a is not None else "") == str(b if b is not None else "")


def is_unreviewed_bulk_value(item: Mapping[str, Any], head: str) -> bool:
    """その head の確定値が、一括確定が書いたまま変わっていないか。

    記録が無い（人が付けた / 一括確定を通っていない）なら `False`。
    """
    predicted = item.get("predicted")
    if not isinstance(predicted, dict):
        return False
    entry = predicted.get(head)
    if not isinstance(entry, dict) or not entry.get("confirmed"):
        return False

    labels = item.get("labels")
    current = labels.get(head) if isinstance(labels, dict) else None
    return _same_value(entry.get("value"), current)


def has_unreviewed_bulk_value(item: Mapping[str, Any], *, skip: Any = ()) -> bool:
    """どれか 1 つでも未確認の確定値を持つ head があるか。

    `skip` に挙げた head（`split` など、分類の対象でないもの）は見ない。
    """
    predicted = item.get("predicted")
    if not isinstance(predicted, dict):
        return False
    skipped = set(skip or ())
    return any(
        head not in skipped and is_unreviewed_bulk_value(item, head)
        for head in predicted
    )
