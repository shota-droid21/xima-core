"""embed_images ジョブの登録と既定引数の回帰テスト。

ジョブ種別はレジストリ 3 箇所（schemas / runner / manager）に揃っている必要があり、
どれか 1 つでも漏れると実行時に初めて失敗する。ここで配線を固定する。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.jobs.manager import JobsManager, create_jobs_router
from app.jobs.runner import build_command
from app.jobs.schemas import JOB_TYPES


class _DummyAsyncResult:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    manager = JobsManager(config_manager)
    app = FastAPI()
    app.include_router(create_jobs_router(manager))
    return TestClient(app)


def test_embed_images_is_a_known_job_type() -> None:
    assert "embed_images" in JOB_TYPES


def test_build_command_resolves_embed_images_script(tmp_path: Path) -> None:
    cfg = ConfigManager(tmp_path).get_config()
    cmd = build_command(
        "embed_images",
        {"labels": "/w/labels.json", "root": "/w/source_images", "clip_model": "ViT-B/32"},
        cfg,
    )

    assert cmd[-1] != ""
    assert any(part.endswith("embed_images.py") for part in cmd)
    # snake_case の引数はハイフン付きフラグに変換される
    assert "--clip-model" in cmd
    assert "--labels" in cmd


def test_create_embed_images_job_fills_default_args(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        calls.append({"command": command})
        return task_id or "task-1"

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    workspace = "ws51aa20"
    experiment = "ex51aa20"
    (tmp_path / workspace).mkdir(parents=True, exist_ok=True)

    client = _make_client(tmp_path)
    res = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs",
        json={"type": "embed_images", "args": {}},
    )

    assert res.status_code == 200
    assert len(calls) == 1

    command = calls[0]["command"]
    # 手動でパスを渡さなくても labels / root / cache-root が補完される
    assert "--labels" in command
    assert "--root" in command
    assert "--cache-root" in command

    cache_root = command[command.index("--cache-root") + 1]
    assert cache_root.endswith(f"{experiment}/cache")
