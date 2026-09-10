"""変更のある item だけ厳格に検証する（#332）。

**この検査が守るもの。**

1. 触っていない item の違反で、無関係な保存が止まらない
2. 変更のある item の違反は今までどおり拒否される
3. 残った違反が件数で分かる
4. **エラーの文言から head を拾えている**。`label_input_partial` は文言を正規表現で
   読むので、`label_input` 側の文言が変わると黙って `(不明)` に落ちる。ここで固定する
"""

from __future__ import annotations

import copy

import pytest

from .label_input import normalize_label_input_payload_with_schema
from .label_input_partial import (
    UNKNOWN_HEAD,
    count_schema_violations,
    normalize_tolerating_unchanged,
    unchanged_file_ids,
)

SCHEMA = {
    "heads": [
        {"id": "shape", "type": "multi_class", "classes": ["circle", "square"]},
        {"id": "tone", "type": "multi_label", "classes": ["warm", "cool"]},
    ]
}


def item(file_id: str, **labels):
    return {"file_id": file_id, "path": f"{file_id}.jpg", "labels": dict(labels)}


def normalize(payload, schema):
    return normalize_label_input_payload_with_schema(payload, schema)


def run(payload, current):
    return normalize_tolerating_unchanged(
        payload, SCHEMA, current, normalize=normalize
    )


# --- 1. 触っていない item の違反は保存を止めない -------------------------------


def test_unchanged_violation_does_not_block_other_changes():
    """実運用で踏んだ形。19 件の違反で 131 件の確定が止まった（#332）。"""
    broken = item("a", shape="gone")  # スキーマから消したクラス
    current = [broken, item("b", shape="circle")]

    payload = {"items": [copy.deepcopy(broken), item("b", shape="square")]}
    out, violations = run(payload, current)

    assert out["items"][1]["labels"]["shape"] == "square", "変更は保存される"
    assert out["items"][0]["labels"]["shape"] == "gone", "違反はそのまま残る"
    assert violations == {"shape": 1}, "黙って残さない"


def test_unchanged_violation_is_kept_verbatim():
    """正規化できない値なので、**手を加えずそのまま**残す。"""
    broken = item("a", shape="gone", tone=["warm"])
    out, _ = run({"items": [copy.deepcopy(broken)]}, [broken])
    assert out["items"][0] == broken


# --- 2. 変更のある item は今までどおり拒否 -------------------------------------


def test_changed_item_with_violation_is_rejected():
    current = [item("a", shape="circle")]
    payload = {"items": [item("a", shape="gone")]}
    with pytest.raises(ValueError, match="must be one of"):
        run(payload, current)


def test_new_item_with_violation_is_rejected():
    """current に無い item は「変更あり」。**新しく壊れた値は入れない。**"""
    with pytest.raises(ValueError, match="must be one of"):
        run({"items": [item("new", shape="gone")]}, [])


def test_item_without_file_id_is_rejected():
    """file_id が無いと同一性が決められない。**判断がつかないなら厳格に倒す。**"""
    with pytest.raises(ValueError):
        run({"items": [{"path": "x.jpg", "labels": {"shape": "gone"}}]}, [])


def test_rejected_message_keeps_the_real_item_index():
    """1 件だけ渡して判定するので、番号が 0 に潰れないことを確かめる（#331）。"""
    current = [item("a", shape="circle"), item("b", shape="circle")]
    payload = {"items": [item("a", shape="circle"), item("b", shape="gone")]}
    with pytest.raises(ValueError) as got:
        run(payload, current)
    assert "items[1]" in str(got.value)


def test_current_unreadable_means_nothing_is_tolerated():
    """現在の中身が取れないときは何も緩めない（`_current_items` は空を返す）。"""
    with pytest.raises(ValueError):
        run({"items": [item("a", shape="gone")]}, [])


# --- 3. 変更の判定 -------------------------------------------------------------


def test_unchanged_ignores_keys_outside_labels_and_delete():
    """表示用の値が変わっただけでは「変更」にしない。"""
    current = [item("a", shape="circle")]
    moved = item("a", shape="circle")
    moved["thumb_path"] = "/static/changed.webp"
    assert unchanged_file_ids([moved], current) == {"a"}


def test_delete_flag_change_counts_as_changed():
    current = [item("a", shape="circle")]
    flagged = item("a", shape="circle")
    flagged["delete"] = True
    assert unchanged_file_ids([flagged], current) == set()


def test_multi_label_reorder_counts_as_changed():
    """**並び順が違えば「変更あり」**とする。

    比較は正規化の**前**に行う。正規化してから比べられれば順序差を吸収できるが、
    正規化できない値（＝この仕組みが助けたい値）では正規化そのものが落ちるので、
    比べる前に落ちてしまう。したがって生の値で比べるしかない。

    **緩い側ではなく厳しい側へ倒れる**ので安全である。GET で取った文書をそのまま
    PUT へ返す通常の流れでは、順序は変わらない。
    """
    current = [item("a", tone=["warm", "cool"])]
    reordered = [item("a", tone=["cool", "warm"])]
    assert unchanged_file_ids(reordered, current) == set()


# --- 4. 通る payload は今までと完全に同じ -------------------------------------


def test_clean_payload_matches_strict_normalization():
    payload = {"items": [item("a", shape="circle"), item("b", tone=["warm"])]}
    out, violations = run(copy.deepcopy(payload), [])
    assert violations == {}
    assert out == normalize(copy.deepcopy(payload), SCHEMA)


# --- 5. 文言から head を拾えている（正規表現の固定） --------------------------


@pytest.mark.parametrize(
    "labels, expected_head",
    [
        ({"shape": "gone"}, "shape"),  # multi_class: must be one of
        ({"tone": ["gone"]}, "tone"),  # multi_label: contains unknown labels
        ({"tone": {"warm": True}}, "tone"),  # multi_label: must be a string array
        ({"tone": [{"warm": True}]}, "tone"),  # multi_label: 要素が文字列でない
        ({"shape": 1}, "shape"),  # must be a string
        ({"nosuch": "x"}, "nosuch"),  # unknown head ids
    ],
)
def test_head_is_extracted_from_every_message_shape(labels, expected_head):
    """`label_input` の文言が変わると、黙って `(不明)` に落ちる。ここで気づく。"""
    heads, count = count_schema_violations(
        [item("a", **labels)], SCHEMA, normalize=normalize
    )
    assert count == 1
    assert list(heads) == [expected_head], f"文言から head を拾えていない: {heads}"
    assert UNKNOWN_HEAD not in heads


def test_multi_label_accepts_a_plain_string():
    """`multi_label` に文字列を渡しても壊れない（1 要素の配列に直る）。

    違反として数えないことを固定する。#316 でこの寛容さを確認しており、
    ここが変わると `schema_violations` の件数が跳ねる。
    """
    heads, count = count_schema_violations(
        [item("a", tone="warm")], SCHEMA, normalize=normalize
    )
    assert (heads, count) == ({}, 0)


def test_count_schema_violations_counts_items_and_heads():
    items = [
        item("a", shape="gone"),
        item("b", shape="gone"),
        item("c", shape="circle"),
    ]
    heads, count = count_schema_violations(items, SCHEMA, normalize=normalize)
    assert heads == {"shape": 2}
    assert count == 2


def test_split_is_not_a_schema_head_but_is_still_validated():
    """`split` はスキーマに無いが検証される。head として拾えることを確かめる。"""
    heads, count = count_schema_violations(
        [item("a", split="nope")], SCHEMA, normalize=normalize
    )
    assert count == 1
    assert list(heads) == ["split"]
