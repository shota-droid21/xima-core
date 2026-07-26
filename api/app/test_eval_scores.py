"""eval_scores（推論結果の整形・CSV 化・名前検証）の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.eval_scores import (
    ScoresNameError,
    build_rows,
    index_items_by_id,
    paginate,
    resolve_heads,
    resolve_scores_path,
    rows_to_csv,
    top_prediction,
    validate_scores_name,
)


def test_validate_scores_name_accepts_expected_pattern() -> None:
    assert validate_scores_name("scores_run_20260101_000000.json") == (
        "scores_run_20260101_000000.json"
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "labels.json",
        "scores_.json.txt",
        "../scores_a.json",
        "scores_a/../../etc/passwd",
        "scores_a.json/../secret",
        "/etc/passwd",
    ],
)
def test_validate_scores_name_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ScoresNameError):
        validate_scores_name(name)


def test_resolve_scores_path_keeps_file_inside_eval_dir(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    resolved = resolve_scores_path(eval_dir, "scores_run_a.json")
    assert resolved.parent == eval_dir.resolve()


def test_resolve_scores_path_rejects_traversal(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    with pytest.raises(ScoresNameError):
        resolve_scores_path(eval_dir, "../../secret.json")


def test_top_prediction_picks_highest_score() -> None:
    assert top_prediction({"a": 0.1, "b": 0.7, "c": 0.2}) == ("b", 0.7)


def test_top_prediction_is_stable_on_ties() -> None:
    # 同率なら常にクラス名昇順で先頭（表示/CSV がブレない）
    assert top_prediction({"b": 0.5, "a": 0.5})[0] == "a"


def test_top_prediction_handles_empty_or_invalid() -> None:
    assert top_prediction({}) is None
    assert top_prediction(None) is None
    assert top_prediction({"a": "not-a-number"}) is None


def test_resolve_heads_prefers_meta() -> None:
    heads = resolve_heads({"heads": ["character", "tags"]}, [])
    assert heads == ["character", "tags"]


def test_resolve_heads_falls_back_to_items() -> None:
    # meta の無い古い結果でも head を推定できる
    items = [{"id": "1", "character": {"a": 1.0}}, {"id": "2", "tags": {"x": 1.0}}]
    assert resolve_heads({}, items) == ["character", "tags"]


def test_build_rows_joins_index_for_split_and_image() -> None:
    items = [{"id": "img1", "character": {"cat": 0.9, "dog": 0.1}}]
    index_by_id = index_items_by_id(
        {"items": [{"id": "img1", "split": "val", "dataset_path": "val/img1.png"}]}
    )

    rows = build_rows(
        items,
        index_by_id,
        ["character"],
        dataset_url_base="/static/ws/experiments/exp/dataset",
    )

    assert rows[0]["id"] == "img1"
    assert rows[0]["split"] == "val"
    assert rows[0]["image_url"] == "/static/ws/experiments/exp/dataset/val/img1.png"
    assert rows[0]["predictions"]["character"] == {"label": "cat", "score": 0.9}


def test_build_rows_works_without_index() -> None:
    # index.json が無くても推論結果自体は表示できる
    rows = build_rows([{"id": "x", "tags": {"a": 0.4}}], {}, ["tags"])

    assert rows[0]["split"] is None
    assert rows[0]["image_url"] is None
    assert rows[0]["predictions"]["tags"]["label"] == "a"


def test_build_rows_marks_missing_head_as_none() -> None:
    rows = build_rows([{"id": "x"}], {}, ["character"])
    assert rows[0]["predictions"]["character"] is None


def test_paginate_slices_rows() -> None:
    rows = [{"id": str(i)} for i in range(10)]
    assert [r["id"] for r in paginate(rows, limit=3, offset=0)] == ["0", "1", "2"]
    assert [r["id"] for r in paginate(rows, limit=3, offset=8)] == ["8", "9"]
    # 範囲外は空
    assert paginate(rows, limit=3, offset=99) == []


def test_rows_to_csv_has_header_and_values() -> None:
    rows = build_rows(
        [{"id": "a", "character": {"cat": 0.75, "dog": 0.25}}],
        index_items_by_id({"items": [{"id": "a", "split": "train"}]}),
        ["character"],
    )

    csv_text = rows_to_csv(rows, ["character"])
    lines = csv_text.strip().splitlines()

    assert lines[0] == "id,split,character,character_score"
    assert lines[1] == "a,train,cat,0.750000"


def test_rows_to_csv_leaves_missing_predictions_blank() -> None:
    rows = build_rows([{"id": "a"}], {}, ["character"])
    csv_text = rows_to_csv(rows, ["character"])

    assert csv_text.strip().splitlines()[1] == "a,,,"
