from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.workspace import create_workspace_router


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_workspace_router(config_manager))
    return TestClient(app)


def _has_backup_toolchain() -> bool:
    return shutil.which("tar") is not None and shutil.which("zstd") is not None


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_backup_and_restore_roundtrip(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa01"

    created = client.post(
        "/workspaces",
        json={"display_name": "Backup Target", "id": workspace},
    )
    assert created.status_code == 200
    ws_root = tmp_path / workspace

    source_file = ws_root / "source_images" / "a.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("hello", encoding="utf-8")

    exp_file = ws_root / "experiments" / "expa0001" / "label_input" / "labels.json"
    exp_file.parent.mkdir(parents=True, exist_ok=True)
    exp_file.write_text('{"items":[]}', encoding="utf-8")

    backup_res = client.get(f"/workspaces/{workspace}/backup")
    assert backup_res.status_code == 200
    archive_bytes = backup_res.content
    assert archive_bytes

    shutil.rmtree(ws_root)
    assert not ws_root.exists()

    restore_res = client.post(
        "/workspaces/restore",
        data=archive_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert restore_res.status_code == 200
    restore_payload = restore_res.json()
    assert restore_payload["status"] == "restored"
    assert restore_payload["workspace"] == workspace

    assert (ws_root / "workspace.json").exists()
    assert source_file.read_text(encoding="utf-8") == "hello"
    assert exp_file.read_text(encoding="utf-8") == '{"items":[]}'


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_restore_blocks_when_workspace_id_already_exists(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa02"

    created = client.post(
        "/workspaces",
        json={"display_name": "Conflict Target", "id": workspace},
    )
    assert created.status_code == 200

    backup_res = client.get(f"/workspaces/{workspace}/backup")
    assert backup_res.status_code == 200

    restore_res = client.post(
        "/workspaces/restore",
        data=backup_res.content,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert restore_res.status_code == 409
    assert "already exists" in str(restore_res.json().get("detail", ""))


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_backup_job_creates_downloadable_artifact(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa03"

    created = client.post(
        "/workspaces",
        json={"display_name": "Job Backup Target", "id": workspace},
    )
    assert created.status_code == 200

    ws_root = tmp_path / workspace
    src = ws_root / "source_images" / "a.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("hello", encoding="utf-8")

    created_job = client.post(f"/workspaces/{workspace}/backup-jobs")
    assert created_job.status_code == 200
    payload = created_job.json()
    assert payload["status"] == "queued"
    job_id = payload["job_id"]

    deadline = time.time() + 20
    latest = None
    while time.time() < deadline:
        job_res = client.get(f"/workspaces/{workspace}/backup-jobs/{job_id}")
        assert job_res.status_code == 200
        latest = job_res.json()
        if latest["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    assert latest is not None
    assert latest["status"] == "done", latest
    assert latest.get("artifact_path")
    assert Path(latest["artifact_path"]).exists()

    list_res = client.get(f"/workspaces/{workspace}/backup-jobs")
    assert list_res.status_code == 200
    listed = list_res.json()["jobs"]
    assert listed
    assert listed[0]["id"] == job_id

    dl_res = client.get(f"/workspaces/{workspace}/backup-jobs/{job_id}/download")
    assert dl_res.status_code == 200
    assert dl_res.content


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_restore_job_roundtrip(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa04"

    created = client.post(
        "/workspaces",
        json={"display_name": "Restore Job Target", "id": workspace},
    )
    assert created.status_code == 200
    ws_root = tmp_path / workspace

    source_file = ws_root / "source_images" / "a.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("hello", encoding="utf-8")

    backup_res = client.get(f"/workspaces/{workspace}/backup")
    assert backup_res.status_code == 200
    archive_bytes = backup_res.content
    assert archive_bytes

    shutil.rmtree(ws_root)
    assert not ws_root.exists()

    restore_job_res = client.post(
        "/workspaces/restore-jobs",
        data=archive_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert restore_job_res.status_code == 200
    restore_job = restore_job_res.json()
    assert restore_job["status"] == "queued"
    assert restore_job["workspace"] == workspace
    job_id = restore_job["job_id"]

    deadline = time.time() + 20
    latest = None
    while time.time() < deadline:
        job_res = client.get(f"/workspaces/restore-jobs/{job_id}")
        assert job_res.status_code == 200
        latest = job_res.json()
        if latest["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    assert latest is not None
    assert latest["status"] == "done", latest
    assert (ws_root / "workspace.json").exists()
    assert source_file.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_restore_job_blocks_when_workspace_already_exists(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa05"

    created = client.post(
        "/workspaces",
        json={"display_name": "Restore Job Conflict", "id": workspace},
    )
    assert created.status_code == 200

    backup_res = client.get(f"/workspaces/{workspace}/backup")
    assert backup_res.status_code == 200

    restore_job_res = client.post(
        "/workspaces/restore-jobs",
        data=backup_res.content,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert restore_job_res.status_code == 409
    assert "already exists" in str(restore_job_res.json().get("detail", ""))


def test_workspace_tmp_cleanup_and_job_persistence_on_startup(tmp_path: Path) -> None:
    stale_artifact = (
        tmp_path / ".tmp" / "backups" / "artifacts" / "ws51aa06" / "stale.tar.zst"
    )
    stale_artifact.parent.mkdir(parents=True, exist_ok=True)
    stale_artifact.write_bytes(b"stale")

    backup_job_id = "a" * 32
    backup_job_file = (
        tmp_path / ".xima" / "jobs" / "workspace_backup" / f"{backup_job_id}.json"
    )
    backup_job_file.parent.mkdir(parents=True, exist_ok=True)
    backup_job_file.write_text(
        '{"id":"' + backup_job_id + '","type":"workspace_backup","workspace":"ws51aa06","status":"running","progress":{"percent":10,"message":"running"}}',
        encoding="utf-8",
    )

    restore_job_id = "b" * 32
    restore_job_file = (
        tmp_path / ".xima" / "jobs" / "workspace_restore" / f"{restore_job_id}.json"
    )
    restore_job_file.parent.mkdir(parents=True, exist_ok=True)
    restore_job_file.write_text(
        '{"id":"' + restore_job_id + '","type":"workspace_restore","workspace":"ws51aa07","status":"queued","progress":{"percent":0,"message":"queued"}}',
        encoding="utf-8",
    )

    client = _make_client(tmp_path)

    assert not stale_artifact.exists()
    assert (tmp_path / ".tmp").exists()
    assert backup_job_file.exists()
    assert restore_job_file.exists()

    backup_job_res = client.get(f"/workspaces/ws51aa06/backup-jobs/{backup_job_id}")
    assert backup_job_res.status_code == 200
    backup_payload = backup_job_res.json()
    assert backup_payload["status"] == "error"
    assert "interrupted by restart" in str(backup_payload.get("error", ""))

    restore_job_res = client.get(f"/workspaces/restore-jobs/{restore_job_id}")
    assert restore_job_res.status_code == 200
    restore_payload = restore_job_res.json()
    assert restore_payload["status"] == "error"
    assert "interrupted by restart" in str(restore_payload.get("error", ""))


@pytest.mark.skipif(
    not _has_backup_toolchain(),
    reason="workspace backup/restore tests require tar and zstd",
)
def test_workspace_backup_download_returns_410_when_artifact_missing(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "ws51aa08"

    created = client.post(
        "/workspaces",
        json={"display_name": "Artifact Expire Target", "id": workspace},
    )
    assert created.status_code == 200

    created_job = client.post(f"/workspaces/{workspace}/backup-jobs")
    assert created_job.status_code == 200
    job_id = created_job.json()["job_id"]

    deadline = time.time() + 20
    latest = None
    while time.time() < deadline:
        job_res = client.get(f"/workspaces/{workspace}/backup-jobs/{job_id}")
        assert job_res.status_code == 200
        latest = job_res.json()
        if latest["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    assert latest is not None
    assert latest["status"] == "done", latest
    artifact_path = Path(str(latest.get("artifact_path") or ""))
    assert artifact_path.exists()
    artifact_path.unlink()

    dl_res = client.get(f"/workspaces/{workspace}/backup-jobs/{job_id}/download")
    assert dl_res.status_code == 410
    assert "expired" in str(dl_res.json().get("detail", "")).lower()
