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
                "confirmed": {
                    "at": "2026-09-23T00:00:00",
                    "threshold": 0.8,
                    "value": value,
                },
            }
        },
    }


def _legacy_bulk(value: Any, current: Any = None) -> Dict[str, Any]:
    """#418 より前に書かれた記録。`confirmed` が値を持たない。"""
    item = _bulk(value, current)
    item["predicted"]["shape"]["confirmed"].pop("value")
    return item


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


def test_the_record_survives_a_new_prediction() -> None:
    """候補を付け直しても「人が見ていない」は消えない（#418）。

    引き継がないと「一括確定 -> 学習 -> もう一度候補を付ける」を 1 周しただけで、
    未確認の値が評価の正解側へ戻る。
    """
    from label_predictions import Prediction, record_prediction

    item = _bulk("circle")
    assert is_unreviewed_bulk_value(item, "shape")

    record_prediction(
        item,
        head="shape",
        head_type="multi_class",
        prediction=Prediction(
            value="circle", score=0.9, margin=0.4, top=[("circle", 0.9)], confidence=0.9
        ),
        run_name="run_y",
    )

    assert is_unreviewed_bulk_value(item, "shape"), "候補を付け直したら記録が消えた"


def test_a_new_prediction_with_another_value_does_not_defeat_it() -> None:
    """新しい予測が別の値を出しても「人が直した」とは読まない（#418）。

    比べる相手は `confirmed` が持つ値であって、付け直された `value` ではない。
    """
    from label_predictions import Prediction, record_prediction

    item = _bulk("circle")
    record_prediction(
        item,
        head="shape",
        head_type="multi_class",
        prediction=Prediction(
            value="square", score=0.7, margin=0.2, top=[("square", 0.7)], confidence=0.7
        ),
        run_name="run_y",
    )

    assert item["predicted"]["shape"]["value"] == "square"
    assert item["labels"]["shape"] == "circle"
    assert is_unreviewed_bulk_value(item, "shape"), "新しい予測につられて判定が狂った"


def test_the_person_can_still_take_it_back() -> None:
    """引き継いでも、人が値を直せば未確認ではなくなる（#418）。"""
    from label_predictions import Prediction, record_prediction

    item = _bulk("circle")
    item["labels"]["shape"] = "square"  # 人が直した
    record_prediction(
        item,
        head="shape",
        head_type="multi_class",
        prediction=Prediction(
            value="circle", score=0.9, margin=0.4, top=[("circle", 0.9)], confidence=0.9
        ),
        run_name="run_y",
    )

    assert not is_unreviewed_bulk_value(item, "shape")


def test_records_written_before_this_change_are_still_read() -> None:
    """`confirmed` が値を持たない古い記録は、従来どおり `value` と比べる（#418）。"""
    assert is_unreviewed_bulk_value(_legacy_bulk("circle"), "shape")
    assert not is_unreviewed_bulk_value(_legacy_bulk("circle", "square"), "shape")


def test_a_prediction_without_a_confirmation_stays_clean() -> None:
    """一括確定を通っていない item に、記録が生えたりしない（#418）。"""
    from label_predictions import Prediction, record_prediction

    item: Dict[str, Any] = {"labels": {"shape": "circle"}}
    record_prediction(
        item,
        head="shape",
        head_type="multi_class",
        prediction=Prediction(
            value="circle", score=0.9, margin=0.4, top=[("circle", 0.9)], confidence=0.9
        ),
        run_name="run_y",
    )

    assert "confirmed" not in item["predicted"]["shape"]
    assert not is_unreviewed_bulk_value(item, "shape")


def test_skipped_heads_are_not_counted() -> None:
    """`split` はデータ管理用で、分類の正解ではない。"""
    item = _bulk("val", head="split")
    assert has_unreviewed_bulk_value(item)
    assert not has_unreviewed_bulk_value(item, skip=("split",))
