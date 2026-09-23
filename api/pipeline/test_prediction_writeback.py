"""推論の書き戻しが、あいだに保存された人の編集を消さないこと（#411 / 再現は #382）。

直す前は `labels.json` を**最初に丸ごと読み**、数分かかる推論を挟んで**丸ごと
書き戻して**いた。読んだ時点からの変化を見ていないので、推論が走っているあいだに
画面から保存すると、その保存は候補と一緒に上書きされていた。

この検査が守るのは 4 つ。

1. あいだに保存された **`labels`（確定値）が残る**
2. それでも**候補は失われない**
3. あいだに消された item の候補は**戻さない**
4. あいだに保存が無いときの結果が**変わらない**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from label_predictions import Prediction, record_prediction, write_json_atomic
from prediction_writeback import (
    PendingPrediction,
    apply_pending,
    read_json_with_digest,
)


def _labels(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"meta": {}, "items": items}


def _item(fid: str, **labels: Any) -> Dict[str, Any]:
    return {"id": 0, "file_id": fid, "path": fid, "labels": dict(labels)}


def _pending(fid: str, value: str) -> PendingPrediction:
    return PendingPrediction(
        file_id=fid,
        head="shape",
        head_type="multi_class",
        prediction=Prediction(
            value=value, score=0.9, margin=0.4, top=[(value, 0.9)], confidence=0.9
        ),
        run_name="run_x",
    )


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_a_save_during_inference_survives(tmp_path: Path) -> None:
    """#382 の再現。`circle` → 保存で `square` → 書き戻しで `circle` に戻ってはならない。"""
    p = tmp_path / "labels.json"
    _write(p, _labels([_item("a/photo.png", shape="circle")]))

    # 推論のためにジョブが読む
    _stale, digest_at_load = read_json_with_digest(p)

    # ── 推論の最中に、人が画面から保存した ──
    _write(p, _labels([_item("a/photo.png", shape="square")]))

    # 書き戻し: 読み直してから載せる
    fresh, digest_now = read_json_with_digest(p)
    assert digest_now != digest_at_load, "保存を検知できていない"

    apply_pending(fresh["items"], [_pending("a/photo.png", "circle")], record=record_prediction)
    write_json_atomic(p, fresh)

    saved = json.loads(p.read_text(encoding="utf-8"))["items"][0]
    assert saved["labels"]["shape"] == "square", "人が保存した確定値が上書きされた"
    assert saved["predicted"]["shape"]["value"] == "circle", "候補まで失われた"


def test_nothing_changes_when_no_save_happened(tmp_path: Path) -> None:
    """あいだに保存が無ければ、指紋は一致し、結果はこれまでと同じ。"""
    p = tmp_path / "labels.json"
    _write(p, _labels([_item("a/photo.png", shape="circle")]))

    fresh, digest_at_load = read_json_with_digest(p)
    _again, digest_now = read_json_with_digest(p)
    assert digest_now == digest_at_load

    apply_pending(fresh["items"], [_pending("a/photo.png", "square")], record=record_prediction)
    write_json_atomic(p, fresh)

    saved = json.loads(p.read_text(encoding="utf-8"))["items"][0]
    assert saved["labels"]["shape"] == "circle"
    assert saved["predicted"]["shape"]["value"] == "square"


def test_item_removed_during_inference_is_not_brought_back(tmp_path: Path) -> None:
    """推論のあいだに消された item の候補は書かない。"""
    items = [_item("a/photo.png"), _item("b/photo.png")]
    fresh = _labels([items[0]])  # b は消えた

    report = apply_pending(
        fresh["items"],
        [_pending("a/photo.png", "circle"), _pending("b/photo.png", "square")],
        record=record_prediction,
    )

    assert report.applied == 1
    assert report.skipped_missing == 1
    assert report.missing_file_ids == ["b/photo.png"]
    assert len(fresh["items"]) == 1, "消された item が戻ってきた"


def test_candidates_never_touch_the_confirmed_value(tmp_path: Path) -> None:
    """候補を載せても `labels` / `committed` / `split` は動かない。"""
    item = _item("a/photo.png", shape="circle", split="train")
    item["committed"] = {"shape": True}

    apply_pending([item], [_pending("a/photo.png", "square")], record=record_prediction)

    assert item["labels"] == {"shape": "circle", "split": "train"}
    assert item["committed"] == {"shape": True}


def test_digest_comes_from_the_same_bytes_that_were_parsed(tmp_path: Path) -> None:
    """指紋と中身を別々に読まない。ずれると「一致するのに違う」が起こる。"""
    p = tmp_path / "labels.json"
    _write(p, _labels([_item("a/photo.png", shape="circle")]))

    data, digest = read_json_with_digest(p)

    import hashlib

    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    assert data["items"][0]["labels"]["shape"] == "circle"
