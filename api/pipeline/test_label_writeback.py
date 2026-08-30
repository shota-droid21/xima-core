"""label_writeback の判断ロジックのテスト（torch 不要）。

守りたいのは「人が付けたラベルを壊さない」「予測だと分かる」の 2 点なので、
そこに厚く書く。
"""

from __future__ import annotations

from pathlib import Path

from label_writeback import (
    DEFAULT_MIN_SCORE,
    PREDICTED_KEY,
    Prediction,
    WritebackReport,
    apply_prediction,
    has_label,
    history_snapshot_path,
    is_delete_flagged,
    plan_writeback,
    value_from_scores,
)


def _item(file_id: str, **kw):
    it = {"file_id": file_id, "path": f"{file_id}.jpg", "labels": {}}
    it.update(kw)
    return it


# ---------------------------------------------------------------- has_label


def test_has_label_treats_empty_string_as_unlabeled():
    assert has_label(_item("a", labels={"character": ""}), "character") is False


def test_has_label_treats_empty_list_as_unlabeled():
    assert has_label(_item("a", labels={"tags": []}), "tags") is False


def test_has_label_is_per_head():
    item = _item("a", labels={"character": "alice"})
    assert has_label(item, "character") is True
    assert has_label(item, "tags") is False


def test_has_label_survives_broken_labels():
    assert has_label({"file_id": "a", "labels": "not-a-dict"}, "character") is False


# ------------------------------------------------------------ delete flag


def test_is_delete_flagged_reads_flag():
    assert is_delete_flagged(_item("a", delete=True)) is True


def test_is_delete_flagged_reads_legacy_split():
    assert is_delete_flagged(_item("a", labels={"split": "delete"})) is True


def test_is_delete_flagged_normal_item():
    assert is_delete_flagged(_item("a", labels={"split": "train"})) is False


# -------------------------------------------------------- value_from_scores


def test_multi_class_picks_argmax():
    p = value_from_scores(
        {"alice": 0.7, "bob": 0.2, "carol": 0.1},
        head_type="multi_class",
        min_score=0.5,
    )
    assert p is not None and p.value == "alice" and p.score == 0.7


def test_multi_class_below_threshold_is_skipped():
    p = value_from_scores(
        {"alice": 0.4, "bob": 0.35}, head_type="multi_class", min_score=0.5
    )
    assert p is None


def test_multi_label_selects_every_class_over_threshold():
    p = value_from_scores(
        {"a": 0.9, "b": 0.6, "c": 0.1}, head_type="multi_label", min_score=0.5
    )
    assert p is not None and sorted(p.value) == ["a", "b"]


def test_multi_label_with_nothing_over_threshold_is_skipped():
    p = value_from_scores(
        {"a": 0.2, "b": 0.1}, head_type="multi_label", min_score=0.5
    )
    assert p is None


def test_empty_scores_is_skipped():
    assert value_from_scores({}, head_type="multi_class", min_score=0.0) is None


def test_top_is_capped_at_three():
    p = value_from_scores(
        {f"c{i}": 0.9 - i * 0.1 for i in range(6)},
        head_type="multi_class",
        min_score=0.0,
    )
    assert p is not None and len(p.top) == 3


# ------------------------------------------------------------ plan_writeback


def test_plan_never_touches_a_human_label():
    items = [_item("a", labels={"character": "alice"})]
    report = WritebackReport()
    planned = plan_writeback(
        items,
        head="character",
        head_type="multi_class",
        scores_by_file_id={"a": {"bob": 0.99}},
        min_score=0.5,
        report=report,
    )
    assert planned == []
    assert report.skipped_existing == 1
    # 計画段階では item を触らない
    assert items[0]["labels"]["character"] == "alice"


def test_plan_skips_delete_flagged():
    report = WritebackReport()
    planned = plan_writeback(
        [_item("a", delete=True)],
        head="character",
        head_type="multi_class",
        scores_by_file_id={"a": {"alice": 0.99}},
        min_score=0.5,
        report=report,
    )
    assert planned == [] and report.skipped_deleted == 1


def test_plan_skips_items_without_embedding():
    report = WritebackReport()
    planned = plan_writeback(
        [_item("a")],
        head="character",
        head_type="multi_class",
        scores_by_file_id={},
        min_score=0.5,
        report=report,
    )
    assert planned == [] and report.skipped_no_embedding == 1


def test_plan_selects_unlabeled_item():
    items = [_item("a")]
    report = WritebackReport()
    planned = plan_writeback(
        items,
        head="character",
        head_type="multi_class",
        scores_by_file_id={"a": {"alice": 0.8, "bob": 0.2}},
        min_score=0.5,
        report=report,
    )
    assert len(planned) == 1
    assert planned[0][1].value == "alice"


def test_plan_does_not_mutate_items():
    items = [_item("a")]
    plan_writeback(
        items,
        head="character",
        head_type="multi_class",
        scores_by_file_id={"a": {"alice": 0.8}},
        min_score=0.5,
        report=WritebackReport(),
    )
    assert items[0]["labels"] == {}
    assert PREDICTED_KEY not in items[0]


def test_plan_skips_item_without_file_id():
    report = WritebackReport()
    planned = plan_writeback(
        [{"labels": {}}],
        head="character",
        head_type="multi_class",
        scores_by_file_id={"": {"alice": 0.9}},
        min_score=0.5,
        report=report,
    )
    assert planned == []


# --------------------------------------------------------- apply_prediction


def test_apply_writes_value_and_provenance():
    item = _item("a")
    apply_prediction(
        item,
        head="character",
        prediction=Prediction("a", "character", "alice", 0.83, [{"class": "alice", "score": 0.83}]),
        run_name="run_20260830_120000",
        now="2026-08-30T12:00:00",
    )
    assert item["labels"]["character"] == "alice"
    rec = item[PREDICTED_KEY]["character"]
    assert rec["value"] == "alice"
    assert rec["score"] == 0.83
    assert rec["run"] == "run_20260830_120000"
    assert rec["at"] == "2026-08-30T12:00:00"


def test_apply_does_not_set_committed():
    item = _item("a")
    apply_prediction(
        item,
        head="character",
        prediction=Prediction("a", "character", "alice", 0.9, []),
        run_name="run_x",
    )
    assert "committed" not in item


def test_apply_does_not_touch_split():
    item = _item("a", labels={"split": "unassigned"})
    apply_prediction(
        item,
        head="character",
        prediction=Prediction("a", "character", "alice", 0.9, []),
        run_name="run_x",
    )
    assert item["labels"]["split"] == "unassigned"


def test_apply_keeps_other_heads_provenance():
    item = _item("a")
    apply_prediction(
        item, head="h1",
        prediction=Prediction("a", "h1", "x", 0.9, []), run_name="run_x",
    )
    apply_prediction(
        item, head="h2",
        prediction=Prediction("a", "h2", "y", 0.8, []), run_name="run_x",
    )
    assert set(item[PREDICTED_KEY]) == {"h1", "h2"}


# ------------------------------------------------------ history_snapshot_path


def test_history_snapshot_matches_label_input_naming(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}")
    p = history_snapshot_path(labels, now="20260830_120000")
    assert p.parent == tmp_path / "history"
    assert p.name == "labels_20260830_120000.json"


def test_history_snapshot_avoids_collision(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}")
    (tmp_path / "history").mkdir()
    (tmp_path / "history" / "labels_20260830_120000.json").write_text("{}")
    p = history_snapshot_path(labels, now="20260830_120000")
    assert p.name == "labels_20260830_120000-1.json"


def test_default_min_score_is_not_zero():
    # 0 にすると確信の無い予測を全部書き戻してしまう。既定で守る。
    assert DEFAULT_MIN_SCORE > 0


# ------------------------------------------------------- commit_writeback

import json  # noqa: E402

from label_writeback import commit_writeback, write_json_atomic  # noqa: E402


def _labels_file(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_commit_takes_snapshot_then_writes(tmp_path: Path):
    labels = _labels_file(tmp_path, {"items": [{"file_id": "a"}]})
    updated = {"items": [{"file_id": "a", "labels": {"character": "alice"}}]}

    snapshot = commit_writeback(labels, updated, written=1, now="20260830_120000")

    assert snapshot is not None and snapshot.exists()
    # スナップショットは **書き換え前** の中身であること
    assert json.loads(snapshot.read_text())["items"][0] == {"file_id": "a"}
    assert json.loads(labels.read_text())["items"][0]["labels"]["character"] == "alice"


def test_commit_does_nothing_when_nothing_written(tmp_path: Path):
    labels = _labels_file(tmp_path, {"items": []})
    before = labels.read_text()

    assert commit_writeback(labels, {"items": ["changed"]}, written=0) is None

    assert labels.read_text() == before
    assert not (tmp_path / "history").exists()


def test_commit_snapshot_is_restorable_by_label_history(tmp_path: Path):
    # LabelHistory は history/<stem>_<id>.json を列挙する。命名が揃っていないと
    # 一括書き換えを戻せない。
    labels = _labels_file(tmp_path, {"items": []})
    snapshot = commit_writeback(labels, {"items": [1]}, written=1, now="20260830_120000")
    assert snapshot is not None
    assert snapshot.parent.name == "history"
    assert snapshot.name.startswith("labels_")


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


def test_write_json_atomic_keeps_original_on_serialize_failure(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json_atomic(p, {"a": 1})
    try:
        write_json_atomic(p, {"bad": object()})
    except TypeError:
        pass
    assert json.loads(p.read_text())["a"] == 1
    assert [q.name for q in tmp_path.iterdir()] == ["out.json"]
