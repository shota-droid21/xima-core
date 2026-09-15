"""削除対象の解決（#410）。

**この検査が守るのは 1 つ。** 記録されたパスに画像が無いとき、**別の画像を代わりに
消さない**こと。原本の削除は取り返しがつかない。

直す前は `file_id`（＝ファイル名の語幹）で別の場所を探し、フォルダを見ずに最初に
見つかったものを消していた。`a/photo.png` を対象にしたつもりで `b/photo.png` が
消えることがあった。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import purge_deleted_images


def _write_labels(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")


def _run(monkeypatch, capsys, *, labels: Path, root: Path, dry_run: bool = False) -> dict:
    argv = ["purge_deleted_images.py", "--labels", str(labels), "--root", str(root)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    purge_deleted_images.main()
    out = capsys.readouterr().out
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"summary が出ていない:\n{out}")


def test_does_not_delete_a_same_named_image_in_another_folder(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """**#380 で再現した場面。** 対象が無く、同名の別画像が 1 つだけある。

    候補が 1 つしか無くても消さない。**同じ語幹であることは、同じ画像である根拠に
    ならない**（「候補が 1 つのときだけ消す」では直らない理由）。
    """
    root = tmp_path / "source_images"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    victim = root / "b" / "photo.png"
    victim.write_bytes(b"b-image")
    # a/photo.png は既に無い

    labels = tmp_path / "label_input" / "labels.json"
    _write_labels(labels, [{"file_id": "photo", "path": "a/photo.png", "delete": True}])

    summary = _run(monkeypatch, capsys, labels=labels, root=root)

    assert victim.exists(), "別フォルダの同名画像が消された"
    assert victim.read_bytes() == b"b-image"
    assert summary["deleted_files"] == 0
    assert summary["not_found"] == 1
    assert summary["target_candidates"] == 0


def test_deletes_when_the_recorded_path_exists(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """正しく見つかるときの動きは変えていない。"""
    root = tmp_path / "source_images"
    (root / "a").mkdir(parents=True)
    target = root / "a" / "photo.png"
    target.write_bytes(b"a-image")
    keep = root / "a" / "keep.png"
    keep.write_bytes(b"keep")

    labels = tmp_path / "label_input" / "labels.json"
    _write_labels(
        labels,
        [
            {"file_id": "photo", "path": "a/photo.png", "delete": True},
            {"file_id": "keep", "path": "a/keep.png", "delete": False},
        ],
    )

    summary = _run(monkeypatch, capsys, labels=labels, root=root)

    assert not target.exists()
    assert keep.exists()
    assert summary["deleted_files"] == 1
    assert summary["not_found"] == 0


def test_not_found_is_reported(tmp_path: Path, monkeypatch, capsys) -> None:
    """**消していないことが分かる。** 黙って飛ばすと「消えたはず」が残る。"""
    root = tmp_path / "source_images"
    root.mkdir(parents=True)

    labels = tmp_path / "label_input" / "labels.json"
    _write_labels(
        labels,
        [
            {"file_id": "x", "path": "a/x.png", "delete": True},
            {"file_id": "y", "path": "a/y.png", "delete": True},
        ],
    )

    summary = _run(monkeypatch, capsys, labels=labels, root=root)
    out = capsys.readouterr().out + ""

    assert summary["not_found"] == 2
    assert summary["deleted_files"] == 0
    assert summary["delete_flagged_items"] == 2


def test_dry_run_deletes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "source_images"
    (root / "a").mkdir(parents=True)
    target = root / "a" / "photo.png"
    target.write_bytes(b"a-image")

    labels = tmp_path / "label_input" / "labels.json"
    _write_labels(labels, [{"file_id": "photo", "path": "a/photo.png", "delete": True}])

    summary = _run(monkeypatch, capsys, labels=labels, root=root, dry_run=True)

    assert target.exists()
    assert summary["dry_run"] is True


def test_paths_outside_root_are_not_deleted(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """root の外は消さない（既存の守り。合わせて押さえる）。"""
    root = tmp_path / "source_images"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    labels = tmp_path / "label_input" / "labels.json"
    _write_labels(labels, [{"file_id": "outside", "path": "../outside.png", "delete": True}])

    summary = _run(monkeypatch, capsys, labels=labels, root=root)

    assert outside.exists()
    assert summary["outside_root"] == 1
    assert summary["deleted_files"] == 0
