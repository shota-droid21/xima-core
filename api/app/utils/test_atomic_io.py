"""atomic_io の契約テスト（#151）。

守りたいのは速度ではなく「**書き込みが失敗しても元のファイルが壊れない**」こと。
そこだけを直接検証する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .atomic_io import write_json_atomic, write_text_atomic


def test_write_text_atomic_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "labels.json"
    target.write_text("old", encoding="utf-8")

    write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


def test_write_text_atomic_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "labels.json"

    write_text_atomic(target, "x")

    assert target.read_text(encoding="utf-8") == "x"


def test_original_survives_when_write_is_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    """書き込み途中で中断されても、元の内容がそのまま残ること。

    これが本 helper の存在理由。`Path.write_text` は truncate してから書くため、
    ここで元ファイルが失われる（切り詰められた不正な JSON が残る）。
    """
    target = tmp_path / "labels.json"
    original = json.dumps({"items": [{"id": "0"}]}, indent=2)
    target.write_text(original, encoding="utf-8")

    import os as os_module

    def boom(*args, **kwargs):
        raise KeyboardInterrupt("interrupted mid-write")

    # 一時ファイルへの書き込みは成功し、差し替えの直前で落ちる状況を作る。
    monkeypatch.setattr(os_module, "replace", boom)

    with pytest.raises(KeyboardInterrupt):
        write_text_atomic(target, "TRUNCATED")

    assert target.read_text(encoding="utf-8") == original


def test_no_temp_file_is_left_behind_on_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "labels.json"
    target.write_text("keep", encoding="utf-8")

    import os as os_module

    monkeypatch.setattr(
        os_module, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
    )

    with pytest.raises(OSError):
        write_text_atomic(target, "x")

    # 元ファイルだけが残っていること（.tmp が散らからない）。
    assert [p.name for p in tmp_path.iterdir()] == ["labels.json"]


def test_write_json_atomic_does_not_touch_file_when_not_serializable(
    tmp_path: Path,
) -> None:
    """JSON にできない値でも既存ファイルを壊さないこと。

    シリアライズを書き込みより先に行っているため、ここで失敗しても
    ファイルには一切触れない。
    """
    target = tmp_path / "labels.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})

    assert target.read_text(encoding="utf-8") == "original"


def test_write_json_atomic_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "label_schema.json"
    payload = {"heads": [{"id": "h1", "type": "multi_class"}], "名前": "日本語"}

    write_json_atomic(target, payload)

    assert json.loads(target.read_text(encoding="utf-8")) == payload
    # ensure_ascii=False が既定。日本語がエスケープされないこと。
    assert "日本語" in target.read_text(encoding="utf-8")
