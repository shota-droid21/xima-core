"""白黒化で無効にする head の決め方（#251 A-1）。

以前はここに開発者のスキーマ由来の head 名が直書きされており、その名前を
使っていない利用者では色ラベルが白黒画像に残ったままデータセットへ入っていた。
名前で当てにいくのをやめたので、宣言された head だけが対象になることを確かめる。
"""

import json
from pathlib import Path

from augment_gray_from_dataset import (
    GRAY_INVALIDATE_FLAG,
    infer_schema_path,
    resolve_gray_invalidated_heads,
)


def _write_schema(tmp_path: Path, heads: list) -> Path:
    p = tmp_path / "label_schema.json"
    p.write_text(json.dumps({"version": 2, "heads": heads}), encoding="utf-8")
    return p


def test_explicit_wins_and_dedupes() -> None:
    got = resolve_gray_invalidated_heads(" color , tone , color ", None)
    assert got == ["color", "tone"]


def test_explicit_empty_string_means_nothing() -> None:
    assert resolve_gray_invalidated_heads("", None) == []


def test_reads_flag_from_schema(tmp_path: Path) -> None:
    schema = _write_schema(
        tmp_path,
        [
            {"id": "shape", "type": "multi_class"},
            {"id": "tone", "type": "multi_class", GRAY_INVALIDATE_FLAG: True},
        ],
    )
    assert resolve_gray_invalidated_heads(None, schema) == ["tone"]


def test_flag_must_be_true_not_truthy(tmp_path: Path) -> None:
    schema = _write_schema(
        tmp_path, [{"id": "tone", GRAY_INVALIDATE_FLAG: "yes"}]
    )
    assert resolve_gray_invalidated_heads(None, schema) == []


def test_no_flag_means_nothing_is_invalidated(tmp_path: Path) -> None:
    """head 名から推測しない。宣言が無ければ 0 件。"""
    schema = _write_schema(
        tmp_path,
        [{"id": "palette"}, {"id": "tint"}, {"id": "color"}],
    )
    assert resolve_gray_invalidated_heads(None, schema) == []


def test_missing_or_broken_schema_is_not_fatal(tmp_path: Path) -> None:
    assert resolve_gray_invalidated_heads(None, tmp_path / "nope.json") == []
    broken = tmp_path / "label_schema.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert resolve_gray_invalidated_heads(None, broken) == []


def test_explicit_takes_priority_over_schema(tmp_path: Path) -> None:
    schema = _write_schema(tmp_path, [{"id": "tone", GRAY_INVALIDATE_FLAG: True}])
    assert resolve_gray_invalidated_heads("color", schema) == ["color"]


def test_schema_path_is_inferred_next_to_dataset_root() -> None:
    got = infer_schema_path(Path("/w/experiments/e1/dataset"))
    assert got == Path("/w/experiments/e1/label_input/label_schema.json")
