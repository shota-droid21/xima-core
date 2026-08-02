"""デモ作成エンドポイントの結合テスト。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.demo import create_demo_router
from app.jobs.manager import JobsManager


def _make_client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue",
        lambda self, **kwargs: "task-demo",
    )
    # ここでは worker を動かさないため、取り込み待ちはスタブ化する
    # （待ち処理そのものは _wait_for_import の単体テストで検証する）。
    monkeypatch.setattr("app.demo._wait_for_import", lambda *a, **k: True)
    config_manager = ConfigManager(tmp_path)
    jobs_manager = JobsManager(config_manager)
    app = FastAPI()
    app.include_router(create_demo_router(config_manager, jobs_manager))
    return TestClient(app)


def test_create_demo_builds_workspace_experiment_and_images(
    monkeypatch, tmp_path: Path
) -> None:
    client = _make_client(tmp_path, monkeypatch)

    res = client.post("/demo", params={"images_per_class": 2})

    assert res.status_code == 200
    body = res.json()
    assert body["images"] == 6  # 3 クラス × 2 枚
    assert body["classes"] == ["circle", "square", "triangle"]

    ws_dir = tmp_path / body["workspace"]
    exp_dir = ws_dir / "experiments" / body["experiment"]
    assert ws_dir.is_dir()
    assert exp_dir.is_dir()

    # 通常作成と同じく meta が置かれる
    assert (ws_dir / "workspace.json").exists()
    assert (exp_dir / "experiment.json").exists()

    # 画像がクラス別に生成される
    pngs = sorted((ws_dir / "source_images").rglob("*.png"))
    assert len(pngs) == 6

    # 追加設定なしで学習まで通せるよう schema を同梱する
    schema = json.loads(
        (exp_dir / "label_input" / "label_schema.json").read_text(encoding="utf-8")
    )
    head_ids = [h["id"] for h in schema["heads"]]
    assert "split" in head_ids
    assert "shape" in head_ids


def test_create_demo_enqueues_make_label_list(monkeypatch, tmp_path: Path) -> None:
    client = _make_client(tmp_path, monkeypatch)

    body = client.post("/demo", params={"images_per_class": 1}).json()

    # 取り込みは通常 workflow と同じジョブ経由で行う
    assert body["job_id"]


def test_create_demo_waits_for_import_to_finish(monkeypatch, tmp_path: Path) -> None:
    """取り込み完了前に返すと UI が「0 件」をキャッシュしてしまうため、完了を待つ。"""
    from app.demo import _wait_for_import

    class _Job:
        def __init__(self, statuses):
            self._statuses = list(statuses)
            self.status = self._statuses[0]

        def advance(self):
            if len(self._statuses) > 1:
                self._statuses.pop(0)
            self.status = self._statuses[0]

    class _Mgr:
        def __init__(self, job):
            self.job = job
            self.calls = 0

        def get_job(self, job_id):
            self.calls += 1
            job = self.job
            job.advance()
            return job

    mgr = _Mgr(_Job(["queued", "running", "done"]))
    monkeypatch.setattr("app.demo.time.sleep", lambda _s: None)

    assert _wait_for_import(mgr, "job-1") is True
    assert mgr.calls >= 2


def test_wait_for_import_gives_up_on_failure(monkeypatch, tmp_path: Path) -> None:
    from app.demo import _wait_for_import

    class _Mgr:
        def get_job(self, job_id):
            class J:
                status = "error"

            return J()

    monkeypatch.setattr("app.demo.time.sleep", lambda _s: None)
    # 失敗ジョブを延々と待たない
    assert _wait_for_import(_Mgr(), "job-1") is False
    assert _wait_for_import(_Mgr(), None) is False


def test_create_demo_uses_short_ids(monkeypatch, tmp_path: Path) -> None:
    import re

    client = _make_client(tmp_path, monkeypatch)
    body = client.post("/demo", params={"images_per_class": 1}).json()

    assert re.fullmatch(r"[a-z0-9]{8}", body["workspace"])
    assert re.fullmatch(r"[a-z0-9]{8}", body["experiment"])


def test_create_demo_rejects_out_of_range_count(monkeypatch, tmp_path: Path) -> None:
    client = _make_client(tmp_path, monkeypatch)

    assert client.post("/demo", params={"images_per_class": 0}).status_code == 400
    assert client.post("/demo", params={"images_per_class": 999}).status_code == 400


def test_repeated_demo_creates_separate_workspaces(monkeypatch, tmp_path: Path) -> None:
    client = _make_client(tmp_path, monkeypatch)

    a = client.post("/demo", params={"images_per_class": 1}).json()
    b = client.post("/demo", params={"images_per_class": 1}).json()

    assert a["workspace"] != b["workspace"]
