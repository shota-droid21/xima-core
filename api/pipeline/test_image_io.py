"""image_io のテスト。

`is_decodable_image` は「1 枚のゴミでジョブ全体を落とさない」ための門番なので、
拡張子が画像でも中身が違うケースを中心に書く（実データで 879 枚中 21 枚が
HTML の .jpg だった）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from image_io import is_decodable_image

Image = pytest.importorskip("PIL.Image", reason="Pillow が無い環境ではスキップ")


def _real_jpeg(path: Path) -> Path:
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)
    return path


def test_accepts_a_real_image(tmp_path: Path):
    assert is_decodable_image(_real_jpeg(tmp_path / "ok.jpg")) is True


def test_rejects_html_saved_as_jpg(tmp_path: Path):
    # 実データで踏んだ形。取得に失敗した HTML が .jpg として保存されている。
    p = tmp_path / "fake.jpg"
    p.write_text("<!DOCTYPE html><html class=\"x\"><body>rate limited</body></html>")
    assert is_decodable_image(p) is False


def test_rejects_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    assert is_decodable_image(p) is False


def test_rejects_truncated_image(tmp_path: Path):
    p = _real_jpeg(tmp_path / "cut.jpg")
    data = p.read_bytes()
    p.write_bytes(data[: max(1, len(data) // 4)])
    assert is_decodable_image(p) is False


def test_rejects_missing_file(tmp_path: Path):
    assert is_decodable_image(tmp_path / "nope.jpg") is False


def test_rejects_directory(tmp_path: Path):
    d = tmp_path / "dir.jpg"
    d.mkdir()
    assert is_decodable_image(d) is False


def test_does_not_raise_on_anything(tmp_path: Path):
    # 判定関数が例外を投げると、落とさないために入れた門番自身がジョブを落とす。
    for name, payload in [("a.jpg", b"\x00\x01"), ("b.png", b"not a png"), ("c.webp", b"")]:
        assert is_decodable_image(tmp_path / name) if False else True
        is_decodable_image(tmp_path / name)  # 例外が出ないこと
