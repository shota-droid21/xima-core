"""値はあるのに学習に使えないラベルを数える（#333）。

**この検査が守るもの。**

1. `multi_class` は item ごと外れる。`multi_label` は値だけ落ちる。**どちらも数える**
2. `apply_label_mapping` と `train_epoch` が**同じ数**を出す（数え方が 1 つである）
3. スキーマにクラスが無い head では数えない（学習側が items から集めるため）
4. 0 件のときは鍵ごと出さない
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from label_schema import normalize_label_for_head, raw_label_values
from unusable_labels import (
    count_unusable,
    describe,
    summarize,
    unusable_values,
)

MULTI_CLASS = {"id": "shape", "type": "multi_class", "classes": ["circle", "square"]}
MULTI_LABEL = {"id": "tone", "type": "multi_label", "classes": ["warm", "cool"]}


@dataclass
class ItemRec:
    """`train_epoch` が使う形。`labels` は属性で持つ。"""

    labels: Dict[str, Any] = field(default_factory=dict)


def raw(labels: Dict[str, Any]) -> Dict[str, Any]:
    """`apply_label_mapping` が使う形。`labels` は dict の鍵。"""
    return {"path": "x.jpg", "labels": labels}


# --- 1. 型ごとの落ち方 ---------------------------------------------------------


def test_multi_class_value_outside_classes_is_counted():
    assert unusable_values("gone", head_type="multi_class", classes=["circle"]) == [
        "gone"
    ]


def test_multi_class_value_inside_classes_is_not_counted():
    assert unusable_values("circle", head_type="multi_class", classes=["circle"]) == []


def test_multi_label_counts_only_the_values_that_fall_out():
    """**item は学習に残る。落ちるのはその値だけ。** だから値ごとに数える。"""
    got = unusable_values(
        ["warm", "gone", "cool"], head_type="multi_label", classes=["warm", "cool"]
    )
    assert got == ["gone"]


def test_multi_label_comma_string_is_parsed_the_same_way_as_the_normalizer():
    """区切りの解釈が正規化とずれると、数が合わなくなる。"""
    value = "warm, gone"
    assert raw_label_values(value) == ["warm", "gone"]
    assert normalize_label_for_head(value, MULTI_LABEL) == ["warm"]
    assert unusable_values(value, head_type="multi_label", classes=["warm", "cool"]) == [
        "gone"
    ]


def test_empty_value_is_not_counted():
    for value in (None, "", "   ", []):
        assert unusable_values(value, head_type="multi_class", classes=["circle"]) == []


# --- 2. 数え方は 1 つ（両方の呼び出し側が同じ形を渡せる）----------------------


def test_counts_items_and_values_separately():
    """`multi_label` は 1 item が複数落ちうるので、値の合計と item 数は一致しない。"""
    items = [
        ItemRec({"tone": ["gone", "alsogone"]}),
        ItemRec({"tone": ["gone"]}),
        ItemRec({"tone": ["warm"]}),
    ]
    values, n_items = count_unusable(
        items, class_head="tone", head_type="multi_label", classes=["warm", "cool"]
    )
    assert values == {"gone": 2, "alsogone": 1}
    assert n_items == 2, "item は 2 件（値は 3 つ）"


def test_attribute_items_and_dict_items_give_the_same_answer():
    """`train_epoch`（属性）と `apply_label_mapping`（dict）で数が変わらない。"""
    labels = {"shape": "gone"}
    kwargs = dict(class_head="shape", head_type="multi_class", classes=["circle"])
    assert count_unusable([ItemRec(labels)], **kwargs) == count_unusable(
        [raw(labels)], **kwargs
    )


def test_items_without_labels_are_ignored():
    kwargs = dict(class_head="shape", head_type="multi_class", classes=["circle"])
    assert count_unusable([ItemRec(), raw({}), {}, None], **kwargs) == ({}, 0)


# --- 3. クラスが無い head では数えない ----------------------------------------


def test_no_classes_means_nothing_is_unusable():
    """スキーマにクラスが無い head は、学習側が items から集めて作る。

    そこでは「定義に無いクラス」が原理的に存在しないので、数えると嘘になる。
    """
    assert unusable_values("anything", head_type="multi_class", classes=[]) == []
    assert count_unusable(
        [ItemRec({"shape": "anything"})],
        class_head="shape",
        head_type="multi_class",
        classes=[],
    ) == ({}, 0)


# --- 4. 出し方 -----------------------------------------------------------------


def test_summarize_returns_none_when_clean():
    assert summarize({}, 0) is None


def test_summarize_sorts_values():
    assert summarize({"b": 1, "a": 2}, 3) == {"items": 3, "values": {"a": 2, "b": 1}}


def test_describe_returns_none_when_clean():
    assert describe({}, 0, where="train") is None


def test_describe_names_the_values_and_the_count():
    message = describe({"gone": 19}, 19, where="train の学習・評価")
    assert message is not None
    assert "19" in message and "gone" in message
    # `skipped` と混同させない。あちらは正常な状態である（#316）。
    assert "skipped" not in message


# --- 5. 実運用で踏んだ形 -------------------------------------------------------


def test_the_shape_that_actually_happened():
    """定義から消したクラスの値が train 18 / val 1 残っていた（#333）。"""
    train = [ItemRec({"shape": "gone"}) for _ in range(18)]
    val = [ItemRec({"shape": "gone"})]
    kwargs = dict(class_head="shape", head_type="multi_class", classes=["circle"])

    assert summarize(*count_unusable(train, **kwargs)) == {
        "items": 18,
        "values": {"gone": 18},
    }
    assert summarize(*count_unusable(val, **kwargs)) == {
        "items": 1,
        "values": {"gone": 1},
    }
