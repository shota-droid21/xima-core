"""一括確定のまま人が見ていない値を val に置かないこと（#412 / 再現は #383）。

一括確定はモデル自身の予測を `labels` に書く。それを val の正解に使うと、
**モデルを自分の答えで採点する**ことになり、`val_acc` が本当より高く出る。
`train_epoch` も `feature_cache` も `committed` を見ていないので、
`labels.json` → `index.json` と流れたものはそのまま正解側に入っていた。

train には残す。疑似ラベルとしては使えるので、捨てる理由が無い。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_PIPELINE = Path(__file__).resolve().parent


def _png(path: Path) -> None:
    """最小の PNG。中身は読まれない（コピーされるだけ）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c6360000002000100ffff0300000600"
            "05570c9b0000000049454e44ae426082"
        )
    )


def _item(i: int, value: str, *, bulk: bool, split: str | None = None) -> Dict[str, Any]:
    labels: Dict[str, Any] = {"shape": value}
    if split:
        labels["split"] = split
    item: Dict[str, Any] = {
        "id": i,
        "file_id": f"img_{i:03d}.png",
        "path": f"img_{i:03d}.png",
        "labels": labels,
    }
    if bulk:
        item["predicted"] = {
            "shape": {
                "value": value,
                "confirmed": {"at": "2026-09-23T00:00:00", "threshold": 0.8},
            }
        }
    return item


def _run(tmp_path: Path, items: List[Dict[str, Any]], *, val_ratio: float = 0.2):
    root = tmp_path / "source_images"
    for it in items:
        _png(root / it["path"])

    labels_path = tmp_path / "label_input" / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps({"meta": {}, "items": items}, ensure_ascii=False), encoding="utf-8"
    )

    schema_path = tmp_path / "label_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "heads": [
                    {
                        "id": "shape",
                        "type": "multi_class",
                        "choices": ["circle", "square"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "dataset"
    proc = subprocess.run(
        [
            sys.executable,
            str(_PIPELINE / "apply_label_mapping.py"),
            "--labels", str(labels_path),
            "--root", str(root),
            "--dataset-root", str(dataset_root),
            "--schema", str(schema_path),
            "--val-ratio", str(val_ratio),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    index = json.loads((dataset_root / "index.json").read_text(encoding="utf-8"))
    return index, proc.stdout


def test_unreviewed_values_never_land_in_val(tmp_path: Path) -> None:
    # 人が付けた 20 件と、一括確定のままの 20 件。
    items = [_item(i, "circle", bulk=False) for i in range(20)]
    items += [_item(100 + i, "square", bulk=True) for i in range(20)]

    index, _out = _run(tmp_path, items)

    val = [it for it in index["items"] if it["split"] == "val"]
    unreviewed_paths = {it["path"] for it in items if "predicted" in it}

    assert val, "val が空になってしまった（この標本では人のラベルが足りている想定）"
    assert not (unreviewed_paths & {it["source_path"] for it in val}), (
        "一括確定のままの値が val に入っている"
    )


def test_they_stay_in_train(tmp_path: Path) -> None:
    """捨てない。疑似ラベルとしては使える。"""
    items = [_item(i, "circle", bulk=False) for i in range(20)]
    items += [_item(100 + i, "square", bulk=True) for i in range(20)]

    index, _out = _run(tmp_path, items)

    train_paths = {it["source_path"] for it in index["items"] if it["split"] == "train"}
    for it in items:
        if "predicted" in it:
            assert it["path"] in train_paths, "未確認の item が dataset から消えた"


def test_a_pinned_val_is_moved_and_said_out_loud(tmp_path: Path) -> None:
    """人が val を指定していても移す。**黙っては動かさない。**"""
    items = [_item(0, "square", bulk=True, split="val")]
    items += [_item(i, "circle", bulk=False) for i in range(1, 10)]

    index, out = _run(tmp_path, items)

    moved = next(it for it in index["items"] if it["source_path"] == "img_000.png")
    assert moved["split"] == "train"
    assert moved["split_source"] == "moved_unreviewed"
    assert "img_000.png" in out and "train へ移します" in out


def test_val_emptied_by_the_rule_is_reported(tmp_path: Path) -> None:
    """全部が未確認なら val は 0 件。**高い val_acc を出すより、測れないと言う。**"""
    items = [_item(i, "square", bulk=True) for i in range(20)]

    index, out = _run(tmp_path, items)

    assert [it for it in index["items"] if it["split"] == "val"] == []
    assert index["meta"]["moved_out_of_val_unreviewed"] > 0
    assert "val が 0 件です" in out


def test_counts_are_recorded_in_the_index(tmp_path: Path) -> None:
    items = [_item(i, "circle", bulk=False) for i in range(20)]
    items += [_item(100 + i, "square", bulk=True) for i in range(20)]

    index, _out = _run(tmp_path, items)

    assert index["meta"]["moved_out_of_val_unreviewed"] >= 1
