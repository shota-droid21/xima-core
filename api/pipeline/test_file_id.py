"""`file_id` は item 自身の相対パスだけで決まること（#409）。

守るのは 2 つ。

1. **同名の別画像が別の鍵になる。** ここが #380 の原因だった
2. **鍵が周囲に左右されない。** 他の画像が在るか、走査の順番がどうだったかで
   変わると、次に一覧を作り直したときに既存の item と突き合わせられず、
   **付けたラベルを失う**
"""

from __future__ import annotations

from file_id import file_id_for_path


def test_same_name_in_another_folder_gets_a_different_id() -> None:
    assert file_id_for_path("a/photo.png") != file_id_for_path("b/photo.png")


def test_same_stem_with_another_extension_gets_a_different_id() -> None:
    # 語幹だけで作っていた頃は、これも同じ鍵になっていた。
    assert file_id_for_path("a/photo.png") != file_id_for_path("a/photo.jpg")


def test_depends_only_on_its_own_path() -> None:
    """同じパスなら、いつ・どこで呼んでも同じ鍵になる。"""
    assert file_id_for_path("circle/circle_001.png") == "circle/circle_001.png"


def test_separator_and_leading_dot_are_normalized() -> None:
    """OS の区切りや `./` の有無で鍵が変わらない。"""
    expected = "a/photo.png"
    assert file_id_for_path("a\\photo.png") == expected
    assert file_id_for_path("./a/photo.png") == expected
    assert file_id_for_path("a//photo.png") == expected
    assert file_id_for_path("  a/photo.png  ") == expected


def test_parent_reference_is_not_collapsed() -> None:
    """`..` は畳まない。畳むと別の場所を指す 2 つが同じ鍵になりうる。"""
    assert file_id_for_path("../x/y.png") == "../x/y.png"
    assert file_id_for_path("a/../x/y.png") != file_id_for_path("x/y.png")


def test_missing_path_has_no_id() -> None:
    assert file_id_for_path(None) == ""
    assert file_id_for_path("") == ""
    assert file_id_for_path("   ") == ""
