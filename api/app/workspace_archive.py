"""workspace のバックアップ書庫（`tar` + `zstd`）を作る・読む・展開する。

**外部コマンドを知っているのはこの層だけにする。**

`workspace.py` は 1,700 行を超えており、そこへ「どのコマンドが要るか」を足すと
さらに読めなくなる。バックアップと復元が `tar` / `zstd` に依存していることは
1 か所にまとまっている方が、入っていない環境への手当ても 1 か所で済む。

**押す前に分かるようにする（#385）。** `tar` はあるが `zstd` が無い環境では、
`FileNotFoundError` は起きない（`tar` は起動できてしまう）。`tar` が子プロセスの
`zstd` を見つけられず **status 127 で落ちる**ため、ジョブを走らせた後にしか
失敗が分からなかった。`missing_archive_tools()` を呼び出し側の入口で使い、
ジョブを作る前に止める。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from .utils.paths import is_subpath
from .utils.short_id import require_workspace_id

#: バックアップ書庫の拡張子。`workspace.py` もこれを使う。
ARCHIVE_EXT = ".tar.zst"

#: 書庫の作成・読み出し・展開に要る外部コマンド。
REQUIRED_TOOLS = ("tar", "zstd")


def _tar_env() -> dict[str, str]:
    """`tar` を動かす環境変数（#385）。

    **macOS の `tar`（bsdtar）は、拡張属性を `._<名前>` という別エントリとして
    書庫へ入れる。** bsdtar 自身は読むときにそれを属性へ戻して隠すため、
    `tar -tf` では見えない。しかし復元は Python の `tarfile` で読んでおり、
    そちらにその作法は無いので**そのままファイルとして materialize される**。

    実測（2026-09-13・デモ 36 枚）: 書庫のエントリは bsdtar から見て 122、
    `tarfile` から見ると 244 で、差の 122 が `._*` だった。復元すると
    `source_images` の画像が 36 枚から 72 枚に増える。

    `COPYFILE_DISABLE=1` は macOS の `tar` がこの動作を止めるための環境変数で、
    Linux の GNU tar は見ないので、そのまま渡してよい。
    """
    env = dict(os.environ)
    env["COPYFILE_DISABLE"] = "1"
    return env


def _is_apple_double(relative: Path) -> bool:
    """AppleDouble（`._<名前>`）か。

    **既に作られた書庫のために要る。** 作成側を直しても、手元にある古い書庫には
    `._*` が入ったままなので、展開側でも落とす。
    """
    return relative.name.startswith("._")


class ArchiveToolsMissing(RuntimeError):
    """`tar` / `zstd` が見つからない。

    呼び出し側はこれを捕まえて、**ジョブを作らずに**利用者へ返す。
    """

    def __init__(self, tools: list[str]) -> None:
        self.tools = tools
        joined = ", ".join(tools)
        super().__init__(
            f"{joined} is not installed on this agent; "
            "workspace backup and restore need it"
        )


def missing_archive_tools() -> list[str]:
    """見つからない外部コマンドを返す。全て揃っていれば空。

    `shutil.which` は PATH を見るだけなので安く、入口で毎回呼んでよい。
    """
    return [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]


def require_archive_tools() -> None:
    """揃っていなければ `ArchiveToolsMissing` を投げる。"""
    missing = missing_archive_tools()
    if missing:
        raise ArchiveToolsMissing(missing)


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


def read_workspace_meta_from_archive(archive_path: Path) -> dict[str, Any]:
    require_archive_tools()
    cmd = [
        "tar",
        "--use-compress-program",
        "zstd -dc",
        "-xOf",
        str(archive_path),
        "workspace.json",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=_tar_env()
        )
    except FileNotFoundError:
        raise ValueError("tar or zstd is not installed on the agent")

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "workspace.json not found"
        raise ValueError(f"invalid backup archive: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"workspace.json is not valid JSON: {exc.msg}")

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
    require_archive_tools()
    try:
        zstd_proc = subprocess.Popen(
            ["zstd", "-dc", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ValueError("zstd is not installed on the agent")

    try:
        if zstd_proc.stdout is None:
            raise ValueError("failed to read backup archive stream")
        with tarfile.open(fileobj=zstd_proc.stdout, mode="r|") as tar:
            for member in tar:
                relative = normalize_archive_member_name(member.name)
                # macOS で作った書庫に混ざる拡張属性の実体。展開しない（#385）。
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
        zstd_proc.kill()
        raise ValueError(f"invalid tar archive: {exc}") from exc
    except Exception:
        zstd_proc.kill()
        raise
    finally:
        if zstd_proc.stdout is not None:
            zstd_proc.stdout.close()

    stderr = b""
    if zstd_proc.stderr is not None:
        stderr = zstd_proc.stderr.read() or b""
        zstd_proc.stderr.close()
    rc = zstd_proc.wait()
    if rc != 0:
        detail = stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(detail or "failed to decompress backup archive")


def create_workspace_backup_archive(
    *,
    workspace_root: Path,
    archive_path: Path,
    members: list[str],
) -> None:
    require_archive_tools()
    cmd = [
        "tar",
        "--use-compress-program",
        "zstd -T0 -19",
        "-C",
        str(workspace_root),
        "-cf",
        str(archive_path),
        *members,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=_tar_env()
        )
    except FileNotFoundError:
        raise ValueError("tar or zstd is not installed on the agent")

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "backup command failed"
        raise ValueError(detail)
