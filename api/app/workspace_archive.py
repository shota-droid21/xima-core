"""workspace のバックアップ書庫（`tar` + zstd）を作る・読む・展開する。

**外部コマンドを使わない（#391）。** 以前は `tar` と `zstd` を `subprocess` で
呼んでいたため、入っていない環境では動かなかった。`zstd` は macOS にも Windows にも
標準では無く、GPU 用のイメージにも入っていなかった（#385）。利用者に「まず `zstd` を
入れてください」と言わせないために、圧縮は `zstandard`、書庫は標準ライブラリの
`tarfile` で扱う。

**形式は変えない。** 拡張子も中身も `.tar.zst` のままなので、以前の版が
`tar --use-compress-program "zstd -T0 -19"` で作った書庫はそのまま読める。

`workspace.py` は 1,700 行近くあり、そこへ圧縮の詳細を置くと読めなくなる。
バックアップと復元が何に依存しているかは、この 1 か所にまとめる。
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import zstandard

from .utils.paths import is_subpath
from .utils.short_id import require_workspace_id

#: バックアップ書庫の拡張子。`workspace.py` もこれを使う。
ARCHIVE_EXT = ".tar.zst"

#: 圧縮の強さ。
#:
#: **高くしても効かない（#391 で実測）。** workspace の中身は JPEG・PNG・
#: サムネイルで、すでに圧縮済みである。実際の workspace（19 MB）で測ると、
#: 無圧縮 20.34 MB に対して level 19 が 18.42 MB、level 3 が 19.47 MB だった。
#: **level 19 は level 3 の 40 倍の時間をかけて 5% 多く縮めているだけ**なので、
#: 待ち時間の短い方を既定にする。
COMPRESSION_LEVEL = 3

#: 書庫を読むときに一度に取り出す上限。**壊れた書庫で無限に読まないための蓋**で、
#: 正常な workspace がこれを超えることは想定していない。
_MAX_MEMBERS = 2_000_000


def _is_apple_double(relative: Path) -> bool:
    """AppleDouble（`._<名前>`）か。

    **macOS の `tar`（bsdtar）で作られた書庫のために要る。** あれは拡張属性を
    `._<名前>` という別エントリとして書庫へ入れ、自分で読むときは属性へ戻して
    隠す。こちらは `tarfile` で読むのでその作法が無く、**そのままファイルとして
    実体化する**（#385 で、デモ 36 枚が復元後に 72 枚になることを確認した）。

    作成が `tarfile` になった今、新しい書庫には入らない。**手元にある古い書庫の
    ために残す。**
    """
    return relative.name.startswith("._")


def normalize_archive_member_name(name: str) -> Path:
    cleaned = (name or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("archive contains empty entry name")
    if cleaned.startswith("/"):
        raise ValueError(f"archive entry is absolute path: {name}")
    parts = Path(cleaned).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"archive entry has invalid path: {name}")
    return Path(*parts)


def _open_archive_stream(archive_path: Path):
    """`.tar.zst` を読み出し用に開く。

    `stream_reader` は seek できないので `tarfile` は `r|`（順に読む）で開く。
    戻り値はどちらも閉じる必要があるため、呼び出し側で `with` を重ねる。
    """
    fh = archive_path.open("rb")
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
    except Exception:
        fh.close()
        raise
    return fh, reader


def read_workspace_meta_from_archive(archive_path: Path) -> dict[str, Any]:
    """書庫の `workspace.json` だけを読む。

    以前は `tar -xOf <書庫> workspace.json` に任せていた。順に読む形になったので、
    **見つけた時点で打ち切る**（`workspace.json` は先頭付近に入る）。
    """
    payload: Any = None
    try:
        fh, reader = _open_archive_stream(archive_path)
        with fh, reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for index, member in enumerate(tar):
                if index >= _MAX_MEMBERS:
                    raise ValueError("archive has too many entries")
                if not member.isfile():
                    continue
                try:
                    relative = normalize_archive_member_name(member.name)
                except ValueError:
                    continue
                if _is_apple_double(relative):
                    continue
                if relative != Path("workspace.json"):
                    continue
                src = tar.extractfile(member)
                if src is None:
                    raise ValueError("failed to read workspace.json")
                with src:
                    raw = src.read()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise ValueError(f"workspace.json is not UTF-8: {exc}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"workspace.json is not valid JSON: {exc.msg}")
                break
    except tarfile.TarError as exc:
        raise ValueError(f"invalid backup archive: {exc}") from exc
    except zstandard.ZstdError as exc:
        raise ValueError(f"invalid backup archive: {exc}") from exc

    if payload is None:
        raise ValueError("invalid backup archive: workspace.json not found")

    if not isinstance(payload, dict):
        raise ValueError("workspace.json must be a JSON object")

    raw_workspace_id = str(
        payload.get("workspace_id") or payload.get("id") or ""
    ).strip()
    if not raw_workspace_id:
        raise ValueError("workspace.json does not contain workspace_id")

    try:
        workspace_id = require_workspace_id(raw_workspace_id)
    except ValueError as exc:
        raise ValueError(str(exc))

    return {"workspace_id": workspace_id, "workspace_json": payload}


def extract_archive_into_dir(archive_path: Path, staging_dir: Path) -> None:
    try:
        fh, reader = _open_archive_stream(archive_path)
        with fh, reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for index, member in enumerate(tar):
                if index >= _MAX_MEMBERS:
                    raise ValueError("archive has too many entries")
                relative = normalize_archive_member_name(member.name)
                # macOS で作られた古い書庫に混ざる拡張属性の実体。展開しない（#385）。
                if _is_apple_double(relative):
                    continue
                destination = (staging_dir / relative).resolve()
                if not is_subpath(staging_dir, destination):
                    raise ValueError(f"archive path escapes staging dir: {member.name}")

                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(
                        f"unsupported archive entry type for restore: {member.name}"
                    )

                if not member.isfile():
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    raise ValueError(f"failed to extract file: {member.name}")
                with src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except tarfile.TarError as exc:
        raise ValueError(f"invalid tar archive: {exc}") from exc
    except zstandard.ZstdError as exc:
        raise ValueError(f"failed to decompress backup archive: {exc}") from exc


def create_workspace_backup_archive(
    *,
    workspace_root: Path,
    archive_path: Path,
    members: list[str],
) -> None:
    """`members` を `workspace_root` からの相対で書庫へ入れる。

    `threads=-1` は zstd に使えるコアを任せる指定で、以前の `-T0` と同じ意図。
    """
    compressor = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
    with archive_path.open("wb") as fh:
        with compressor.stream_writer(fh) as writer:
            with tarfile.open(fileobj=writer, mode="w|") as tar:
                for name in members:
                    source = workspace_root / name
                    if not source.exists():
                        continue
                    tar.add(source, arcname=name)
