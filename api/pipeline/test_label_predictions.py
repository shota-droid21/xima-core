"""label_predictions のテスト（torch 不要）。

守りたいのは「**`labels` に絶対に書かない**」「候補だと分かる」の 2 点なので、そこに厚く書く。
"""

from __future__ import annotations

import json
from pathlib import Path

from label_predictions import (
    MULTI_LABEL_DECISION_POINT,
    PREDICTED_KEY,
    TOP_K,
    Prediction,
    PredictionReport,
    has_label,
    history_snapshot_path,
    is_delete_flagged,
    plan_predictions,
    prediction_from_scores,
    record_prediction,
    take_snapshot,
    write_json_atomic,
)


def _item(file_id: str, **kw):
    it = {"file_id": file_id, "path": f"{file_id}.jpg", "labels": {}}
    it.update(kw)
    return it


# --------------------------------------------------------------- has_label


def test_has_label_ignores_predictions():
    # 候補があっても「ラベルあり」にはならない。ここが崩れると確定と候補が混ざる。
    item = _item("a")
    record_prediction(
        item, head="character", head_type="multi_class",
        prediction=Prediction("alice", 0.9, 0.0, []), run_name="run_x",
    )
    assert has_label(item, "character") is False


def test_has_label_empty_string_is_unlabeled():
    assert has_label(_item("a", labels={"character": ""}), "character") is False


def test_has_label_empty_list_is_unlabeled():
    assert has_label(_item("a", labels={"tags": []}), "tags") is False


def test_has_label_is_per_head():
    item = _item("a", labels={"character": "alice"})
    assert has_label(item, "character") is True
    assert has_label(item, "tags") is False


def test_has_label_survives_broken_labels():
    assert has_label({"file_id": "a", "labels": "nope"}, "character") is False


# -------------------------------------------------------------- delete flag


def test_is_delete_flagged_reads_flag():
    assert is_delete_flagged(_item("a", delete=True)) is True


def test_is_delete_flagged_reads_legacy_split():
    assert is_delete_flagged(_item("a", labels={"split": "delete"})) is True


def test_is_delete_flagged_normal_item():
    assert is_delete_flagged(_item("a", labels={"split": "train"})) is False


# -------------------------------------------------- prediction_from_scores


def test_multi_class_picks_argmax_without_threshold():
    # 低スコアでも候補は出す。足切りは一括確定側の仕事。
    p = prediction_from_scores({"alice": 0.31, "bob": 0.30}, head_type="multi_class")
    assert p is not None and p.value == "alice" and p.score == 0.31


def test_margin_is_gap_between_top_two():
    # score の絶対値は当てにならない（クラス数が多いと softmax が平坦になる）。
    # 順位差のほうがクラス数に依存せず比較しやすい。
    p = prediction_from_scores({"a": 0.31, "b": 0.30, "c": 0.29}, head_type="multi_class")
    assert p is not None and abs(p.margin - 0.01) < 1e-9


def test_margin_is_score_when_single_class():
    p = prediction_from_scores({"only": 1.0}, head_type="multi_class")
    assert p is not None and p.margin == 1.0


def test_record_stores_margin():
    item = _item("a")
    record_prediction(
        item, head="h", head_type="multi_class",
        prediction=Prediction("x", 0.31, 0.01, []), run_name="r",
    )
    assert item[PREDICTED_KEY]["h"]["margin"] == 0.01


def test_multi_label_uses_sigmoid_decision_point():
    p = prediction_from_scores({"a": 0.9, "b": 0.6, "c": 0.1}, head_type="multi_label")
    assert p is not None and sorted(p.value) == ["a", "b"]


def test_multi_label_can_select_nothing():
    p = prediction_from_scores({"a": 0.2, "b": 0.1}, head_type="multi_label")
    assert p is not None and p.value == []


def test_decision_point_is_sigmoid_midpoint():
    assert MULTI_LABEL_DECISION_POINT == 0.5


def test_empty_scores_yields_none():
    assert prediction_from_scores({}, head_type="multi_class") is None


def test_top_is_capped():
    p = prediction_from_scores(
        {f"c{i}": 0.9 - i * 0.05 for i in range(20)}, head_type="multi_class"
    )
    assert p is not None and len(p.top) == TOP_K


# ---------------------------------------------------------- record_prediction


def test_record_never_touches_labels():
    item = _item("a", labels={"split": "unassigned"})
    record_prediction(
        item, head="character", head_type="multi_class",
        prediction=Prediction("alice", 0.83, 0.0, [{"class": "alice", "score": 0.83}]),
        run_name="run_x", now="2026-08-31T00:00:00",
    )
    # ここが本モジュールの最重要契約
    assert item["labels"] == {"split": "unassigned"}
    assert "character" not in item["labels"]


def test_record_does_not_set_committed():
    item = _item("a")
    record_prediction(
        item, head="h", head_type="multi_class",
        prediction=Prediction("x", 0.9, 0.0, []), run_name="run_x",
    )
    assert "committed" not in item


def test_record_stores_provenance():
    item = _item("a")
    record_prediction(
        item, head="character", head_type="multi_class",
        prediction=Prediction("alice", 0.83, 0.0, [{"class": "alice", "score": 0.83}]),
        run_name="run_20260830", now="2026-08-31T00:00:00",
    )
    rec = item[PREDICTED_KEY]["character"]
    assert rec["value"] == "alice"
    assert rec["score"] == 0.83
    assert rec["run"] == "run_20260830"
    assert rec["at"] == "2026-08-31T00:00:00"
    assert rec["head_type"] == "multi_class"


def test_record_keeps_other_heads():
    item = _item("a")
    record_prediction(item, head="h1", head_type="multi_class",
                      prediction=Prediction("x", 0.9, 0.0, []), run_name="r")
    record_prediction(item, head="h2", head_type="multi_class",
                      prediction=Prediction("y", 0.8, 0.0, []), run_name="r")
    assert set(item[PREDICTED_KEY]) == {"h1", "h2"}


def test_record_overwrites_same_head_on_rerun():
    # 学習し直したら候補は更新されてよい。labels ではないため作業は失われない。
    item = _item("a")
    record_prediction(item, head="h", head_type="multi_class",
                      prediction=Prediction("x", 0.5, 0.0, []), run_name="run_old")
    record_prediction(item, head="h", head_type="multi_class",
                      prediction=Prediction("y", 0.9, 0.0, []), run_name="run_new")
    assert item[PREDICTED_KEY]["h"]["value"] == "y"
    assert item[PREDICTED_KEY]["h"]["run"] == "run_new"


# ---------------------------------------------------------- plan_predictions


def test_plan_includes_already_labeled_items():
    # class 追加・再ラベルの場面で既ラベルの候補が要る。labels に書かないので安全。
    items = [_item("a", labels={"character": "alice"})]
    report = PredictionReport()
    planned = plan_predictions(
        items, scores_by_file_id={"a": {"bob": 0.9}},
        head_type="multi_class", report=report,
    )
    assert len(planned) == 1


def test_plan_skips_delete_flagged():
    report = PredictionReport()
    planned = plan_predictions(
        [_item("a", delete=True)], scores_by_file_id={"a": {"x": 0.9}},
        head_type="multi_class", report=report,
    )
    assert planned == [] and report.skipped_deleted == 1


def test_plan_skips_items_without_embedding():
    report = PredictionReport()
    planned = plan_predictions(
        [_item("a")], scores_by_file_id={}, head_type="multi_class", report=report,
    )
    assert planned == [] and report.skipped_no_embedding == 1


def test_plan_does_not_mutate_items():
    items = [_item("a")]
    plan_predictions(
        items, scores_by_file_id={"a": {"alice": 0.8}},
        head_type="multi_class", report=PredictionReport(),
    )
    assert items[0]["labels"] == {}
    assert PREDICTED_KEY not in items[0]


def test_plan_skips_item_without_file_id():
    planned = plan_predictions(
        [{"labels": {}}], scores_by_file_id={"": {"a": 0.9}},
        head_type="multi_class", report=PredictionReport(),
    )
    assert planned == []


# --------------------------------------------------------------- file I/O


def test_write_json_atomic_replaces_inode(tmp_path: Path):
    # truncate 書きだと inode が変わらない。並行読み手が壊れた JSON を見る（#197 と同型）。
    p = tmp_path / "out.json"
    write_json_atomic(p, {"a": 1})
    first = p.stat().st_ino
    write_json_atomic(p, {"a": 2})
    assert p.stat().st_ino != first
    assert json.loads(p.read_text())["a"] == 2


def test_write_json_atomic_leaves_no_temp_file(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json_atomic(p, {"a": 1})
    assert [q.name for q in tmp_path.iterdir()] == ["out.json"]


def test_write_json_atomic_keeps_original_on_failure(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json_atomic(p, {"a": 1})
    try:
        write_json_atomic(p, {"bad": object()})
    except TypeError:
        pass
    assert json.loads(p.read_text())["a"] == 1
    assert [q.name for q in tmp_path.iterdir()] == ["out.json"]


def test_snapshot_matches_label_input_naming(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text('{"items": []}')
    p = take_snapshot(labels, now="20260831_120000")
    assert p.parent == tmp_path / "history"
    assert p.name == "labels_20260831_120000.json"
    assert json.loads(p.read_text()) == {"items": []}


def test_snapshot_avoids_collision(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}")
    (tmp_path / "history").mkdir()
    (tmp_path / "history" / "labels_20260831_120000.json").write_text("{}")
    p = history_snapshot_path(labels, now="20260831_120000")
    assert p.name == "labels_20260831_120000-1.json"
