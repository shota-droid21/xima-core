"""走査結果と既存 `labels.json` の突き合わせ（#409）。

`make_label_list` の中で**唯一、付けたラベルを失いうる**ところなので、
本体から切り出して単体で確かめられるようにした。

突き合わせの鍵は `file_id`（[file_id.py](file_id.py)）である。**鍵が衝突すると
2 枚の画像が 1 件に畳まれ、片方に付けたラベルがもう片方へ移る。**それが #380 で
再現した不具合で、原因は鍵がファイル名の語幹だったことにある。ここでは鍵が
一意であることを前提にせず、**畳まれたら分かる**ように数を返す。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def merge_items(
    new_items: List[Dict[str, Any]],
    existing_by_file_id: Dict[str, Dict[str, Any]],
    *,
    remove_labels: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """今回の走査結果に既存のラベルを引き継ぐ。

    返すのは `(items, collapsed)`。`collapsed` は **`file_id` が重なって
    畳まれた件数**で、0 でなければ鍵の作り方が壊れている。

    `id` は `path` の順に振り直す。`label` / `split`（単数形。旧形式の名残）は
    落とす。`remove_labels` に挙げた head は `labels` から取り除く。
    """
    new_by_file_id: Dict[str, Dict[str, Any]] = {}
    for item in new_items:
        new_by_file_id[item["file_id"]] = item
    collapsed = len(new_items) - len(new_by_file_id)

    merged_by_file_id: Dict[str, Dict[str, Any]] = {}
    for fid, new_item in new_by_file_id.items():
        existing = existing_by_file_id.get(fid)
        if existing:
            merged = dict(existing)
            merged["path"] = new_item["path"]
            if not merged.get("created_at") and new_item.get("created_at"):
                merged["created_at"] = new_item.get("created_at")
        else:
            merged = new_item
        merged_by_file_id[fid] = merged

    drop_heads = set(remove_labels or ())

    merged_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(
        sorted(merged_by_file_id.values(), key=lambda x: x.get("path", ""))
    ):
        item["id"] = idx
        item.pop("label", None)
        item.pop("split", None)

        if drop_heads and isinstance(item.get("labels"), dict):
            for lk in drop_heads:
                item["labels"].pop(lk, None)

        merged_items.append(item)

    return merged_items, collapsed
