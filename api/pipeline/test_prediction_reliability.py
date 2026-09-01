"""prediction_reliability の単体テスト（torch 不要）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from label_predictions import multi_label_confidence, prediction_from_scores
from prediction_reliability import (
    agreement_points,
    build_reliability,
    collect_samples,
    is_agreement,
    val_source_paths,
)


def test_agreement_points_counts_and_rate():
    samples = [(0.95, True), (0.85, True), (0.75, False), (0.55, True)]
    points = {p["threshold"]: p for p in agreement_points(samples, (0.9, 0.7, 0.5))}
    assert points[0.9]["n"] == 1 and points[0.9]["agreement"] == 1.0
    assert points[0.7]["n"] == 3
    assert points[0.7]["agreement"] == round(2 / 3, 6)
    assert points[0.5]["n"] == 4


def test_agreement_points_keeps_empty_bands():
    """件数 0 の帯も残す。「選んでも何も起きない閾値」を利用者に見せるため。"""
    points = agreement_points([(0.55, True)], (0.9, 0.5))
    by_t = {p["threshold"]: p for p in points}
    assert by_t[0.9]["n"] == 0
    assert by_t[0.9]["agreement"] is None


def test_multi_label_agreement_requires_exact_set():
    assert is_agreement(["a", "b"], ["b", "a"], head_type="multi_label")
    # 部分一致は一致ではない。確定するのは集合そのもの。
    assert not is_agreement(["a"], ["a", "b"], head_type="multi_label")
    assert not is_agreement(["a", "b"], ["a"], head_type="multi_label")


def test_multi_class_agreement_ignores_surrounding_space():
    assert is_agreement("cat", " cat ", head_type="multi_class")
    assert not is_agreement("cat", "dog", head_type="multi_class")


def test_multi_label_confidence_uses_weakest_decision():
    """1 位が確実でも、0.5 付近のクラスがあれば集合としては危うい。"""
    scores = {"a": 0.99, "b": 0.52, "c": 0.01}
    # b の判定は max(0.52, 0.48) = 0.52 でいちばん弱い
    assert multi_label_confidence(scores) == 0.52
    pred = prediction_from_scores(scores, head_type="multi_label")
    assert pred is not None
    assert pred.score == 0.99          # 1 位の確率はあくまで 0.99
    assert pred.confidence == 0.52     # 一括確定が見るのはこちら


def test_multi_class_confidence_is_top_score():
    pred = prediction_from_scores({"a": 0.8, "b": 0.2}, head_type="multi_class")
    assert pred is not None
    assert pred.confidence == pred.score == 0.8


def test_collect_samples_limits_to_val_and_skips_unlabeled():
    items = [
        {"file_id": "f1", "path": "a.jpg", "labels": {"h": "cat"}},
        {"file_id": "f2", "path": "b.jpg", "labels": {"h": "dog"}},   # train なので除外
        {"file_id": "f3", "path": "c.jpg", "labels": {}},             # 未ラベルなので除外
        {"file_id": "f4", "path": "d.jpg", "labels": {"h": "cat"}, "delete": True},
    ]
    samples = collect_samples(
        items,
        head="h",
        head_type="multi_class",
        confidence_by_file_id={"f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6},
        predicted_by_file_id={"f1": "cat", "f2": "cat", "f3": "cat", "f4": "cat"},
        limit_to_paths={"a.jpg", "c.jpg", "d.jpg"},
    )
    assert samples == [(0.9, True)]


def test_val_source_paths(tmp_path: Path):
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "items": [
                    {"source_path": "a.jpg", "split": "val"},
                    {"source_path": "b.jpg", "split": "train"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert val_source_paths(index) == {"a.jpg"}


def test_val_source_paths_missing_file_is_empty(tmp_path: Path):
    """読めないときは空。呼び出し側が basis='labeled' に落とせる。"""
    assert val_source_paths(tmp_path / "nope.json") == set()


def test_build_reliability_shape():
    out = build_reliability(
        {"h": {"head_type": "multi_class", "n": 1, "points": []}},
        run_name="run_x",
        basis="val",
        now="2026-09-01T00:00:00",
    )
    assert out["run"] == "run_x" and out["basis"] == "val"
    assert "h" in out["heads"]
