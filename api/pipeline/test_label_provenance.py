"""確定値が一括確定のまま人が見ていないものか、を読む（#412 / 再現は #383）。

一括確定は**人が 1 枚も見ていない値**を `labels` に書く。それを評価の正解として
数えると、同じモデルの予測をそのモデルが書いた値と突き合わせることになる。

この検査が守るのは 3 つ。

1. 一括確定が書いたままの値は**未確認**と読む
2. 人が後から値を変えたら**未確認ではなくなる**
3. `committed` は見ない。item 単位の情報で head 単位の確認を代弁させない
"""

from __future__ import annotations

from typing import Any, Dict

from label_provenance import has_unreviewed_bulk_value, is_unreviewed_bulk_value


def _bulk(value: Any, current: Any = None, *, head: str = "shape") -> Dict[str, Any]:
    """一括確定が `value` を書いた item。`current` を渡すと人が変えた後。"""
    return {
        "file_id": "a/photo.png",
        "labels": {head: current if current is not None else value},
        "predicted": {
            head: {
                "value": value,
                "confidence": 0.9,
                "confirmed": {"at": "2026-09-23T00:00:00", "threshold": 0.8},
            }
        },
    }


def test_value_written_by_bulk_confirm_is_unreviewed() -> None:
    assert is_unreviewed_bulk_value(_bulk("circle"), "shape")


def test_value_the_person_changed_is_not_unreviewed() -> None:
    assert not is_unreviewed_bulk_value(_bulk("circle", "square"), "shape")


def test_candidate_without_confirmation_is_not_unreviewed() -> None:
    """候補があるだけで一括確定を通っていないものは、人が付けた値である。"""
    item = {"labels": {"shape": "circle"}, "predicted": {"shape": {"value": "circle"}}}
    assert not is_unreviewed_bulk_value(item, "shape")


def test_hand_written_value_is_not_unreviewed() -> None:
    assert not is_unreviewed_bulk_value({"labels": {"shape": "circle"}}, "shape")


def test_committed_does_not_make_it_reviewed() -> None:
    """`committed` は item 単位。head を見た証拠にはならない（#383 / #345）。"""
    item = _bulk("circle")
    item["committed"] = True
    assert is_unreviewed_bulk_value(item, "shape")


def test_multi_label_ignores_order() -> None:
    item = _bulk(["a", "b"], ["b", "a"])
    assert is_unreviewed_bulk_value(item, "shape")


def test_multi_label_detects_a_real_change() -> None:
    item = _bulk(["a", "b"], ["a"])
    assert not is_unreviewed_bulk_value(item, "shape")


def test_any_head_is_enough_for_the_item() -> None:
    item = _bulk("circle", head="color")
    item["labels"]["shape"] = "square"  # 人が付けた head もある
    assert has_unreviewed_bulk_value(item)


def test_skipped_heads_are_not_counted() -> None:
    """`split` はデータ管理用で、分類の正解ではない。"""
    item = _bulk("val", head="split")
    assert has_unreviewed_bulk_value(item)
    assert not has_unreviewed_bulk_value(item, skip=("split",))
