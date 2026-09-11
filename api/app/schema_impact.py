"""スキーマを変えたとき、**行き先を失う値**を数えて付け替える（#330）。

experiment を進めている途中でスキーマを足す・変える・消すのは普通の作業である。
問題は**消したときに何も起きないこと**だった。`PUT /label-schema` はスキーマ自体の
形しか見ず、labels.json との突き合わせをしない。消した瞬間は静かで、**次に保存
しようとした人が理由の分からない 400 を見る**（#332 で保存は止まらなくなったが、
値が宙に浮いていること自体は変わらない）。

行き先を失う経路は 3 つある。

| 変更 | 何が起きるか |
| --- | --- |
| **クラスを消す / 改名する** | その値を持つ item が、その head の学習から外れる |
| **head を消す** | その head の値が全部行き先を失う |
| **型を変える** | `multi_label` → `multi_class` は、値が 2 つ以上ある item を表せない |

**数えるだけでなく、付け替えられるようにする。** 数が出ても直す手段が無ければ、
利用者は labels.json を手で編集するしかない。

**規則は借りる。** `multi_label` の値の切り出しは `pipeline/label_schema.py` の
`raw_label_values` を使う（`pipeline_modules.py`）。ここで別に書くと、**数えた値と
実際に落ちる値がずれる**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .pipeline_modules import load_pipeline_module

_label_schema = load_pipeline_module("label_schema")
_dataset_split = load_pipeline_module("dataset_split")

raw_label_values = _label_schema.raw_label_values
canonical_head_type = _label_schema.canonical_head_type
get_head_classes = _label_schema.get_head_classes

SPLIT_HEAD_ID = "split"

#: 行き先を失う理由。画面の文言はこの値で出し分ける。
REASON_HEAD_REMOVED = "head_removed"
REASON_CLASS_REMOVED = "class_removed"
REASON_TYPE_CHANGED = "type_changed"


def head_map(schema: Any) -> Dict[str, Dict[str, Any]]:
    """head id -> head。`split` は含めない（値の正否は #325 の規則が決める）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for head in (schema or {}).get("heads") or []:
        if not isinstance(head, dict):
            continue
        hid = str(head.get("id") or "").strip()
        if not hid or hid == SPLIT_HEAD_ID:
            continue
        if canonical_head_type(head.get("type")) == "split":
            continue
        out[hid] = head
    return out


def _labels(item: Any) -> Dict[str, Any]:
    labels = item.get("labels") if isinstance(item, dict) else None
    return labels if isinstance(labels, dict) else {}


def _values_for(value: Any, head: Optional[Dict[str, Any]]) -> Tuple[List[str], bool]:
    """その値を、**新しい head の型で読んだときの値の並び**にする。

    返すのは `(値の並び, 型に収まらないか)`。`multi_label` → `multi_class` で値が
    2 つ以上あるときが「収まらない」で、**どれを残すかを人が決めるしかない**。
    """
    values = raw_label_values(value)
    if head is None:
        return values, False
    if canonical_head_type(head.get("type")) == "multi_label":
        return values, False
    return values, len(values) > 1


def orphans(
    items: List[Any],
    new_schema: Any,
    *,
    old_schema: Any = None,
) -> Dict[str, Any]:
    """新しいスキーマにしたとき、行き先を失う値。

    返すのは head ごとの一覧で、値ごとに件数と**いま学習に入っている数**を持つ。
    「学習に入っている数」は、いま直すべきかの判断に効く（入っていなければ
    次の作り直しまで実害が無い）。
    """
    new_heads = head_map(new_schema)
    old_heads = head_map(old_schema) if old_schema is not None else {}
    head_ids = list(old_heads.keys()) or list(new_heads.keys())

    # head -> value -> {items, entering}
    found: Dict[str, Dict[str, Dict[str, int]]] = {}
    reasons: Dict[str, str] = {}
    overflow: Dict[str, int] = {}
    affected_items = 0
    affected_entering = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        labels = _labels(item)
        entering = _enters_dataset(item, labels, head_ids)
        touched = False

        for hid, value in labels.items():
            if hid == SPLIT_HEAD_ID or not value:
                continue
            if old_heads and hid not in old_heads and hid not in new_heads:
                # もともとスキーマに無い head。#330 の話ではない（#332 が扱う）。
                continue
            new_head = new_heads.get(hid)
            values, does_not_fit = _values_for(value, new_head)
            if not values:
                continue

            if new_head is None:
                reasons[hid] = REASON_HEAD_REMOVED
                lost = values
            else:
                classes = set(get_head_classes(new_head))
                lost = [v for v in values if classes and v not in classes]
                if does_not_fit:
                    # 値そのものは残せても、**型に収まらない**ので全部が対象。
                    lost = values
                    overflow[hid] = overflow.get(hid, 0) + 1
                    reasons[hid] = REASON_TYPE_CHANGED
                elif lost:
                    reasons.setdefault(hid, REASON_CLASS_REMOVED)
            if not lost:
                continue

            touched = True
            per_head = found.setdefault(hid, {})
            for v in lost:
                bucket = per_head.setdefault(v, {"items": 0, "entering": 0})
                bucket["items"] += 1
                if entering:
                    bucket["entering"] += 1

        if touched:
            affected_items += 1
            if entering:
                affected_entering += 1

    heads_out = []
    for hid in sorted(found):
        new_head = new_heads.get(hid)
        heads_out.append(
            {
                "head": hid,
                "reason": reasons.get(hid, REASON_CLASS_REMOVED),
                # 付け替え先の候補。head ごと消えるなら空（＝消すしかない）。
                "targets": get_head_classes(new_head) if new_head else [],
                "type": canonical_head_type((new_head or {}).get("type"))
                if new_head
                else None,
                "values": [
                    {"value": value, **counts}
                    for value, counts in sorted(found[hid].items())
                ],
                # `multi_label` → `multi_class` で値が 2 つ以上ある item の数。
                "too_many_values": overflow.get(hid, 0),
            }
        )

    return {
        "heads": heads_out,
        "items": affected_items,
        "entering": affected_entering,
    }


def _enters_dataset(item: Dict[str, Any], labels: Dict[str, Any], head_ids: List[str]) -> bool:
    split, _ = _dataset_split.resolve_split(
        labels,
        deleted=item.get("delete") is True,
        key=str(item.get("file_id") or item.get("path") or ""),
        head_ids=head_ids,
    )
    return split is not None


def apply_remap(
    items: List[Any],
    remaps: List[Dict[str, Any]],
) -> Tuple[List[Any], Dict[str, int]]:
    """`{head, from, to}` の一覧を当てる。`to` が空なら**その値を消す**。

    返すのは `(新しい items, head ごとに書き換えた item 数)`。**item は複製する**
    ので、呼び出し側は失敗したら元をそのまま書き戻せる。

    **2 回当てても結果は変わらない。** `from` で選ぶので、1 回目のあとは合致する
    item がもう無い。通信がやり直されても二重に当たらない。
    """
    table: Dict[str, Dict[str, str]] = {}
    for remap in remaps:
        if not isinstance(remap, dict):
            continue
        head = str(remap.get("head") or "").strip()
        source = str(remap.get("from") or "").strip()
        if not head or not source or head == SPLIT_HEAD_ID:
            continue
        table.setdefault(head, {})[source] = str(remap.get("to") or "").strip()

    changed: Dict[str, int] = {}
    out: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        labels = _labels(item)
        next_labels = dict(labels)
        item_changed = False

        for head, mapping in table.items():
            if head not in labels:
                continue
            value = labels[head]
            if isinstance(value, list):
                before = raw_label_values(value)
                after: List[str] = []
                for v in before:
                    replaced = mapping.get(v, v)
                    if replaced and replaced not in after:
                        after.append(replaced)
                if after == before:
                    continue
                if after:
                    next_labels[head] = after
                else:
                    next_labels.pop(head, None)
            else:
                before_scalar = str(value).strip() if value is not None else ""
                if before_scalar not in mapping:
                    continue
                replaced = mapping[before_scalar]
                if replaced:
                    next_labels[head] = replaced
                else:
                    next_labels.pop(head, None)
            item_changed = True
            changed[head] = changed.get(head, 0) + 1

        if item_changed:
            next_item = dict(item)
            next_item["labels"] = next_labels
            out.append(next_item)
        else:
            out.append(item)

    return out, changed
