"""atomic_io の契約テスト（#151）。

守りたいのは速度ではなく「**書き込みが失敗しても元のファイルが壊れない**」こと。
そこだけを直接検証する。
"""

from __future__ import annotations

import json
import time
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


def test_concurrent_reader_never_sees_a_torn_file(tmp_path: Path) -> None:
    """書き込み中に読んでも、壊れた JSON は決して観測されない（#197）。

    ジョブ状態ファイルは「書き手が進捗のたびに上書きし、読み手がポーリングする」形で
    使われる。`Path.write_text()` は truncate してから書くため、この使い方では
    読み手が切り詰められた JSON を掴む。実測で同時読み出しの約 1.4% が壊れ、
    xima-core の CI が 3 週間 red になっていた。

    ここで固定したいのは helper の中身ではなく **「並行して読んでも壊れない」** という
    性質そのものである。write_text に戻せばこのテストが落ちる。
    """
    import threading

    target = tmp_path / "job.json"
    payload = {
        "id": "a" * 32,
        "status": "running",
        "progress": {"percent": 50, "message": "extracting backup archive"},
        # 1 回の write が一瞬で終わらない程度の大きさにする。小さすぎると
        # 壊れた実装でも窓が閉じてしまい、回帰を検出できない。
        "pad": [f"item-{i}" for i in range(400)],
    }
    write_json_atomic(target, payload)

    stop = threading.Event()
    torn: list[str] = []
    reads = [0]

    def writer() -> None:
        while not stop.is_set():
            write_json_atomic(target, payload)

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(target.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                torn.append(f"{type(exc).__name__}: {exc}")
            reads[0] += 1

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join()

    assert reads[0] > 500, f"読み出し回数が少なすぎて検出力が無い: {reads[0]}"
    assert torn == [], f"{len(torn)} 件の壊れた読み出し（{reads[0]} 回中）: {torn[:3]}"
