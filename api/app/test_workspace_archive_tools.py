"""`tar` / `zstd` が無い環境での入口の振る舞い（#385）。

**走らせてから 127 で落とさない**ことを確かめる。実際に `zstd` を消すことは
できないので、`missing_archive_tools` を差し替えて「無い」状態を作る。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import workspace as workspace_module
from app.config import ConfigManager
from app.workspace import create_workspace_router
from app.workspace_archive import ArchiveToolsMissing, missing_archive_tools


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_workspace_router(config_manager))
    return TestClient(app)


def _create_workspace(client: TestClient, workspace: str) -> None:
    created = client.post(
        "/workspaces",
        json={"workspace": workspace, "display_name": "backup tools test"},
    )
    assert created.status_code == 200, created.text


def test_missing_archive_tools_is_empty_when_present(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.workspace_archive.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    assert missing_archive_tools() == []


def test_missing_archive_tools_lists_only_what_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.workspace_archive.shutil.which",
        lambda name: None if name == "zstd" else f"/usr/bin/{name}",
    )
    assert missing_archive_tools() == ["zstd"]


def test_archive_tools_missing_message_names_the_command() -> None:
    message = str(ArchiveToolsMissing(["zstd"]))
    assert "zstd" in message
    assert "backup" in message


def test_backup_job_is_not_created_when_zstd_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa02"
    _create_workspace(client, workspace)

    monkeypatch.setattr(workspace_module, "missing_archive_tools", lambda: ["zstd"])

    res = client.post(f"/workspaces/{workspace}/backup-jobs")
    assert res.status_code == 503, res.text
    assert "zstd" in res.json()["detail"]

    # **ジョブが作られていない。** 走ってから落ちる形に戻っていないことを確かめる。
    listed = client.get(f"/workspaces/{workspace}/backup-jobs")
    assert listed.status_code == 200, listed.text
    assert listed.json().get("jobs") == []


def test_sync_backup_is_refused_when_zstd_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa03"
    _create_workspace(client, workspace)

    monkeypatch.setattr(workspace_module, "missing_archive_tools", lambda: ["zstd"])

    res = client.get(f"/workspaces/{workspace}/backup")
    assert res.status_code == 503, res.text
    assert "zstd" in res.json()["detail"]


def test_restore_job_upload_is_refused_before_reading_body(
    tmp_path: Path, monkeypatch
) -> None:
    client = _make_client(tmp_path)
    monkeypatch.setattr(workspace_module, "missing_archive_tools", lambda: ["zstd"])

    res = client.post("/workspaces/restore-jobs", content=b"not-a-real-archive")
    assert res.status_code == 503, res.text
    assert "zstd" in res.json()["detail"]


def test_sync_restore_is_refused_when_zstd_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    client = _make_client(tmp_path)
    monkeypatch.setattr(workspace_module, "missing_archive_tools", lambda: ["zstd"])

    res = client.post("/workspaces/restore", content=b"not-a-real-archive")
    assert res.status_code == 503, res.text
    assert "zstd" in res.json()["detail"]


def test_apple_double_entries_are_not_restored(tmp_path: Path) -> None:
    """macOS の書庫に混ざる `._*` を展開しない（#385）。

    既に作られた書庫のための手当てなので、**作成側ではなく展開側**を確かめる。
    """
    import subprocess
    import tarfile

    from app.workspace_archive import extract_archive_into_dir, missing_archive_tools

    if missing_archive_tools():
        import pytest

        pytest.skip("requires tar and zstd")

    src = tmp_path / "src"
    (src / "source_images").mkdir(parents=True)
    (src / "workspace.json").write_text('{"workspace_id": "ws51aa04"}')
    (src / "source_images" / "a.png").write_bytes(b"real")
    # macOS の tar が作るのと同じ形の AppleDouble を手で入れる。
    (src / "source_images" / "._a.png").write_bytes(b"xattr blob")
    (src / "._workspace.json").write_bytes(b"xattr blob")

    tar_path = tmp_path / "plain.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in ("workspace.json", "._workspace.json", "source_images"):
            tar.add(src / name, arcname=name)

    archive = tmp_path / f"backup{'.tar.zst'}"
    subprocess.run(
        ["zstd", "-q", "-o", str(archive), str(tar_path)], check=True
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    extract_archive_into_dir(archive, staging)

    assert (staging / "workspace.json").exists()
    assert (staging / "source_images" / "a.png").read_bytes() == b"real"
    assert not (staging / "._workspace.json").exists()
    assert not (staging / "source_images" / "._a.png").exists()
    assert [p.name for p in (staging / "source_images").iterdir()] == ["a.png"]


def test_created_archive_has_no_apple_double_entries(tmp_path: Path) -> None:
    """作成側でも `._*` を書庫へ入れない（#385）。

    macOS 以外では元々入らないので、**入っていないこと**だけを見る。
    """
    import subprocess
    import tarfile

    from app.workspace_archive import (
        create_workspace_backup_archive,
        missing_archive_tools,
    )

    if missing_archive_tools():
        import pytest

        pytest.skip("requires tar and zstd")

    root = tmp_path / "ws"
    (root / "source_images").mkdir(parents=True)
    (root / "workspace.json").write_text('{"workspace_id": "ws51aa05"}')
    (root / "source_images" / "a.png").write_bytes(b"real")

    archive = tmp_path / "backup.tar.zst"
    create_workspace_backup_archive(
        workspace_root=root,
        archive_path=archive,
        members=["workspace.json", "source_images"],
    )

    proc = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            names = [m.name for m in tar]
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()

    apple = [n for n in names if Path(n).name.startswith("._")]
    assert apple == [], apple
    assert "source_images/a.png" in names
