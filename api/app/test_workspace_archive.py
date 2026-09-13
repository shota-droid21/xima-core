"""バックアップ書庫の扱い（#385 / #391）。

**外部コマンドを使わないこと**を前提に確かめる。`tar` / `zstd` が PATH に
無くても、作成・読み出し・展開が通らなければならない。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import zstandard

from app.workspace_archive import (
    ARCHIVE_EXT,
    COMPRESSION_LEVEL,
    create_workspace_backup_archive,
    extract_archive_into_dir,
    read_workspace_meta_from_archive,
)


def _make_workspace(root: Path, workspace_id: str = "ws51aa04") -> None:
    (root / "source_images" / "circle").mkdir(parents=True)
    (root / "experiments" / "exp1").mkdir(parents=True)
    (root / "workspace.json").write_text(
        json.dumps({"workspace_id": workspace_id, "display_name": "test"}),
        encoding="utf-8",
    )
    (root / "source_images" / "circle" / "a.png").write_bytes(b"image-a")
    (root / "source_images" / "circle" / "b.png").write_bytes(b"image-b")
    (root / "experiments" / "exp1" / "experiment.json").write_text("{}")


def _members() -> list[str]:
    return ["workspace.json", "source_images", "experiments"]


def _entries(archive: Path) -> list[str]:
    with archive.open("rb") as fh:
        with zstandard.ZstdDecompressor().stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                return [m.name for m in tar]


def _write_archive(archive: Path, src: Path, names: list[str]) -> None:
    compressor = zstandard.ZstdCompressor(level=1)
    with archive.open("wb") as fh, compressor.stream_writer(fh) as writer:
        with tarfile.open(fileobj=writer, mode="w|") as tar:
            for name in names:
                tar.add(src / name, arcname=name)


def test_roundtrip_without_external_commands(tmp_path: Path, monkeypatch) -> None:
    """`tar` / `zstd` が PATH に無くても往復できる（#391）。

    PATH を空にするので、外部コマンドを呼んでいれば必ず落ちる。
    """
    monkeypatch.setenv("PATH", "")

    root = tmp_path / "ws"
    _make_workspace(root)
    archive = tmp_path / f"backup{ARCHIVE_EXT}"

    create_workspace_backup_archive(
        workspace_root=root, archive_path=archive, members=_members()
    )
    assert archive.stat().st_size > 0

    meta = read_workspace_meta_from_archive(archive)
    assert meta["workspace_id"] == "ws51aa04"
    assert meta["workspace_json"]["display_name"] == "test"

    staging = tmp_path / "staging"
    staging.mkdir()
    extract_archive_into_dir(archive, staging)

    assert (staging / "workspace.json").exists()
    assert (staging / "source_images" / "circle" / "a.png").read_bytes() == b"image-a"
    assert (staging / "source_images" / "circle" / "b.png").read_bytes() == b"image-b"
    assert (staging / "experiments" / "exp1" / "experiment.json").exists()


@pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="requires tar and zstd to build the legacy archive",
)
def test_archive_made_by_the_old_external_command_is_still_readable(
    tmp_path: Path,
) -> None:
    """**以前の版が作った書庫が読める（#391）。**

    `tar --use-compress-program "zstd -T0 -19"` は v0.4.1 までの作り方である。
    形式を変えていないので、そのまま読めなければならない。
    """
    root = tmp_path / "ws"
    _make_workspace(root, workspace_id="ws51aa05")
    archive = tmp_path / f"legacy{ARCHIVE_EXT}"

    result = subprocess.run(
        [
            "tar",
            "--use-compress-program",
            "zstd -T0 -19",
            "-C",
            str(root),
            "-cf",
            str(archive),
            *_members(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    meta = read_workspace_meta_from_archive(archive)
    assert meta["workspace_id"] == "ws51aa05"

    staging = tmp_path / "staging"
    staging.mkdir()
    extract_archive_into_dir(archive, staging)
    assert (staging / "source_images" / "circle" / "a.png").read_bytes() == b"image-a"


def test_apple_double_entries_are_not_restored(tmp_path: Path) -> None:
    """macOS の古い書庫に混ざる `._*` を展開しない（#385）。

    作成が `tarfile` になったので新しい書庫には入らない。**手元にある古い書庫の
    ために残している**ので、展開側を確かめる。
    """
    src = tmp_path / "src"
    (src / "source_images").mkdir(parents=True)
    (src / "workspace.json").write_text('{"workspace_id": "ws51aa06"}')
    (src / "source_images" / "a.png").write_bytes(b"real")
    (src / "source_images" / "._a.png").write_bytes(b"xattr blob")
    (src / "._workspace.json").write_bytes(b"xattr blob")

    archive = tmp_path / f"apple{ARCHIVE_EXT}"
    _write_archive(archive, src, ["workspace.json", "._workspace.json", "source_images"])

    staging = tmp_path / "staging"
    staging.mkdir()
    extract_archive_into_dir(archive, staging)

    assert (staging / "source_images" / "a.png").read_bytes() == b"real"
    assert not (staging / "._workspace.json").exists()
    assert not (staging / "source_images" / "._a.png").exists()
    assert [p.name for p in (staging / "source_images").iterdir()] == ["a.png"]


def test_apple_double_workspace_json_is_not_read_as_meta(tmp_path: Path) -> None:
    """`._workspace.json` を `workspace.json` と取り違えない（#385 / #391）。

    順に読む形にしたので、**AppleDouble が先に来る並び**で確かめる。
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "workspace.json").write_text('{"workspace_id": "ws51aa07"}')
    (src / "._workspace.json").write_bytes(b"not json at all")

    archive = tmp_path / f"apple-meta{ARCHIVE_EXT}"
    _write_archive(archive, src, ["._workspace.json", "workspace.json"])

    assert read_workspace_meta_from_archive(archive)["workspace_id"] == "ws51aa07"


def test_missing_workspace_json_is_reported(tmp_path: Path) -> None:
    archive = tmp_path / f"empty{ARCHIVE_EXT}"
    compressor = zstandard.ZstdCompressor(level=1)
    with archive.open("wb") as fh, compressor.stream_writer(fh) as writer:
        with tarfile.open(fileobj=writer, mode="w|"):
            pass

    with pytest.raises(ValueError, match="workspace.json not found"):
        read_workspace_meta_from_archive(archive)


def test_workspace_json_without_id_is_reported(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "workspace.json").write_text('{"display_name": "no id"}')
    archive = tmp_path / f"noid{ARCHIVE_EXT}"
    _write_archive(archive, src, ["workspace.json"])

    with pytest.raises(ValueError, match="workspace_id"):
        read_workspace_meta_from_archive(archive)


def test_broken_archive_is_reported(tmp_path: Path) -> None:
    """壊れた入力で例外の種類が変わらない（呼び出し側は ValueError を見ている）。"""
    archive = tmp_path / f"broken{ARCHIVE_EXT}"
    archive.write_bytes(b"this is not a zstd stream")

    with pytest.raises(ValueError):
        read_workspace_meta_from_archive(archive)

    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError):
        extract_archive_into_dir(archive, staging)


def test_archive_has_no_apple_double_entries(tmp_path: Path) -> None:
    """作成側が `._*` を入れない（#385）。`tarfile` で書くので元から入らない。"""
    root = tmp_path / "ws"
    _make_workspace(root, workspace_id="ws51aa08")
    archive = tmp_path / f"clean{ARCHIVE_EXT}"
    create_workspace_backup_archive(
        workspace_root=root, archive_path=archive, members=_members()
    )

    names = _entries(archive)
    assert [n for n in names if Path(n).name.startswith("._")] == []
    assert "source_images/circle/a.png" in names


def test_missing_members_are_skipped(tmp_path: Path) -> None:
    """無い member は飛ばす。`workspace.py` は存在確認をしてから渡すが、念のため。"""
    root = tmp_path / "ws"
    _make_workspace(root, workspace_id="ws51aa09")
    archive = tmp_path / f"partial{ARCHIVE_EXT}"
    create_workspace_backup_archive(
        workspace_root=root,
        archive_path=archive,
        members=["workspace.json", "does_not_exist"],
    )
    assert _entries(archive) == ["workspace.json"]


def test_compression_level_is_the_documented_default() -> None:
    """既定を下げた理由は本文に書いてある（#391）。値が動いたら気づけるようにする。"""
    assert COMPRESSION_LEVEL == 3
