"""ラベルのある item がどちら側へ入るか（#325 / Decision 045）。

**この検査が守るもの。**

1. **既定は「入る」。** `split` を書かなくても、ラベルがあれば学習に入る
2. 人が書いた `train` / `val` / `exclude` は動かさない
3. 旧値の読み替え —— `unassigned` は自動、`ignore` は除外、`delete` は自動
4. **自動割りは決定的。** 何度回しても、件数が増えても、同じ item は同じ側
5. ラベルが無いものは入らない（学習するものが無い）
"""

from __future__ import annotations

from dataset_split import (
    DEFAULT_VAL_RATIO,
    SOURCE_AUTO,
    SOURCE_PINNED,
    auto_side,
    has_any_label,
    normalize_split,
    resolve_split,
)

HEADS = ["shape", "color"]


def decide(labels, *, key="f1", deleted=False, val_ratio=DEFAULT_VAL_RATIO):
    return resolve_split(
        labels, deleted=deleted, key=key, head_ids=HEADS, val_ratio=val_ratio
    )


# --- 1. 既定は「入る」------------------------------------------------------------


def test_labeled_item_without_split_enters_the_dataset():
    """#316 の 766 件がここで救われる。**書かなくても入る。**"""
    split, source = decide({"shape": "circle"})
    assert split in ("train", "val")
    assert source == SOURCE_AUTO


def test_unlabeled_item_does_not_enter():
    """学習するものが無い item を入れても意味が無い。"""
    assert decide({}) == (None, "unlabeled")
    assert decide({"split": "unassigned"}) == (None, "unlabeled")


def test_empty_values_are_not_labels():
    for value in (None, "", [], {}):
        assert decide({"shape": value}) == (None, "unlabeled")


def test_delete_mark_wins_over_everything():
    assert decide({"shape": "circle"}, deleted=True) == (None, "deleted")
    assert decide({"shape": "circle", "split": "train"}, deleted=True) == (
        None,
        "deleted",
    )


# --- 2. 人が書いたものは動かさない -----------------------------------------------


def test_pinned_values_are_kept():
    assert decide({"shape": "circle", "split": "train"}) == ("train", SOURCE_PINNED)
    assert decide({"shape": "circle", "split": "val"}) == ("val", SOURCE_PINNED)


def test_pinned_is_independent_of_the_ratio():
    """比率を変えても、人が決めた item は動かない。**過去の学習と比べられる。**"""
    for ratio in (0.0, 0.2, 0.9, 1.0):
        assert decide({"shape": "a", "split": "val"}, val_ratio=ratio)[0] == "val"


def test_exclude_keeps_the_item_out():
    assert decide({"shape": "circle", "split": "exclude"}) == (None, "excluded")


def test_pinned_does_not_require_a_label():
    """`split=train` と書いてあれば、項目が空でも人の意思として通す。"""
    assert decide({"split": "train"}) == ("train", SOURCE_PINNED)


def test_unknown_split_value_does_not_silently_enter():
    """スキーマ外の値は自動へ寄せない。**黙って学習に入れない**（#330 と同じ立場）。"""
    assert decide({"shape": "circle", "split": "nope"}) == (None, "invalid_split")


# --- 3. 旧値の読み替え -----------------------------------------------------------


def test_legacy_values():
    assert normalize_split("unassigned") is None, "自動（＝入る）"
    assert normalize_split("ignore") == "exclude", "語が「外す」を意味する"
    assert normalize_split("delete") is None, "意味は削除マークが担う"
    assert normalize_split("  TRAIN ") == "train"
    assert normalize_split("") is None
    assert normalize_split(None) is None


def test_legacy_unassigned_enters_when_labeled():
    """**766 件の移行そのもの。** 値はあるが項目が付いているので入る。"""
    split, source = decide({"shape": "circle", "split": "unassigned"})
    assert split in ("train", "val")
    assert source == SOURCE_AUTO


def test_legacy_ignore_stays_out():
    assert decide({"shape": "circle", "split": "ignore"}) == (None, "excluded")


# --- 4. 自動割りは決定的 ---------------------------------------------------------


def test_same_key_always_lands_on_the_same_side():
    first = [auto_side(f"f{i}", 0.2) for i in range(200)]
    second = [auto_side(f"f{i}", 0.2) for i in range(200)]
    assert first == second


def test_adding_items_does_not_move_existing_ones():
    """**後から増えても既存 item は移動しない。** 比率で上から切ると移動する。"""
    before = {f"f{i}": auto_side(f"f{i}", 0.2) for i in range(100)}
    after = {f"f{i}": auto_side(f"f{i}", 0.2) for i in range(1000)}
    assert all(after[k] == v for k, v in before.items())


def test_ratio_is_roughly_honoured():
    sides = [auto_side(f"file-{i}", 0.2) for i in range(4000)]
    share = sides.count("val") / len(sides)
    assert 0.17 < share < 0.23, share


def test_extreme_ratios():
    assert auto_side("anything", 0.0) == "train"
    assert auto_side("anything", 1.0) == "val"


def test_the_key_matters_not_the_order():
    """key が同じなら、渡す順番や件数に関係なく同じ側。"""
    assert auto_side("abc", 0.5) == auto_side("abc", 0.5)


# --- 5. ラベルがあるかの判定 -----------------------------------------------------


def test_split_is_not_counted_as_a_label():
    assert has_any_label({"split": "train"}, HEADS) is False
    assert has_any_label({"shape": "circle"}, HEADS) is True


def test_without_schema_any_key_but_split_counts():
    """スキーマが読めないとき、`split` 以外に値があればラベル済みとみなす。

    ここで全部を「ラベル無し」に倒すと、**スキーマが壊れている間は 1 枚も
    学習に入らない**。気づきにくい形で学習が空になるほうが害が大きい。
    """
    assert has_any_label({"anything": "x"}, []) is True
    assert has_any_label({"split": "train"}, []) is False
    assert has_any_label({}, []) is False


def test_non_dict_labels_are_not_labels():
    assert has_any_label(None, HEADS) is False
    assert has_any_label("circle", HEADS) is False
