from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.jobs.manager import JobsManager, create_jobs_router
from app.jobs.schemas import JobRecord


class _DummyAsyncResult:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    manager = JobsManager(config_manager)
    app = FastAPI()
    app.include_router(create_jobs_router(manager))
    return TestClient(app)


def test_jobs_can_be_queued_multiple_times(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        calls.append(
            {"job_id": job_id, "command": command, "workspace": workspace, "task_id": task_id}
        )
        return task_id or f"task-{len(calls)}"

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    workspace = "ws51aa10"
    experiment = "ex51aa10"

    ws_root = tmp_path / workspace
    ws_root.mkdir(parents=True, exist_ok=True)

    client = _make_client(tmp_path)

    res1 = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs",
        json={"type": "make_label_list", "args": {}},
    )
    assert res1.status_code == 200
    assert res1.json().get("status") == "queued"

    res2 = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs",
        json={"type": "make_label_list", "args": {}},
    )
    assert res2.status_code == 200
    assert res2.json().get("status") == "queued"

    list_res = client.get(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs?limit=10&offset=0"
    )
    assert list_res.status_code == 200
    jobs = list_res.json().get("jobs", [])
    assert len(jobs) == 2
    assert all(str(job.get("status")) == "queued" for job in jobs)

    assert len(calls) == 2


def test_create_job_returns_503_when_enqueue_fails(monkeypatch, tmp_path: Path) -> None:
    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    workspace = "ws51aa11"
    experiment = "ex51aa11"

    ws_root = tmp_path / workspace
    ws_root.mkdir(parents=True, exist_ok=True)

    client = _make_client(tmp_path)

    res = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs",
        json={"type": "make_label_list", "args": {}},
    )
    assert res.status_code == 503

    jobs_dir = tmp_path / "state" / "jobs"
    json_files = sorted(jobs_dir.glob("*.json"))
    assert len(json_files) == 1

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload.get("status") == "error"
    assert payload.get("exit_code") == -1
    assert "failed to enqueue job" in str(payload.get("error") or "")
    log_path = tmp_path / "state" / "jobs" / f"{payload.get('id')}.log"
    assert log_path.exists()
    assert "enqueue_failed" in log_path.read_text(encoding="utf-8")


def test_create_job_preserves_running_status_when_worker_starts_immediately(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        job_file = tmp_path / "state" / "jobs" / f"{job_id}.json"
        payload = json.loads(job_file.read_text(encoding="utf-8"))
        payload["status"] = "running"
        payload["started_at"] = 1_772_339_000.0
        payload["progress"] = {
            "phase": "train",
            "message": "running",
            "updated_at": 1_772_339_000.0,
        }
        job_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return task_id or "fast-task-id"

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    workspace = "ws51aa23"
    experiment = "ex51aa23"
    (tmp_path / workspace).mkdir(parents=True, exist_ok=True)

    client = _make_client(tmp_path)
    res = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/jobs",
        json={"type": "make_label_list", "args": {}},
    )
    assert res.status_code == 200

    job_id = str(res.json().get("job_id"))
    payload = json.loads(
        (tmp_path / "state" / "jobs" / f"{job_id}.json").read_text(encoding="utf-8")
    )
    assert payload.get("status") == "running"
    assert payload.get("started_at") == 1_772_339_000.0
    assert payload.get("worker_task_id")


def test_delete_jobs_skips_queued_jobs(monkeypatch, tmp_path: Path) -> None:
    seq = 0

    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        nonlocal seq
        seq += 1
        return task_id or f"task-{seq}"

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    workspace = "ws51aa12"
    experiment = "ex51aa12"

    ws_root = tmp_path / workspace
    ws_root.mkdir(parents=True, exist_ok=True)

    client = _make_client(tmp_path)

    created_ids: list[str] = []
    for _ in range(2):
        created = client.post(
            f"/workspaces/{workspace}/experiments/{experiment}/jobs",
            json={"type": "make_label_list", "args": {}},
        )
        assert created.status_code == 200
        created_ids.append(str(created.json().get("job_id")))

    delete_res = client.request("DELETE", "/jobs", json={"job_ids": created_ids})
    assert delete_res.status_code == 200
    body = delete_res.json()
    assert sorted(body.get("skipped_running", [])) == sorted(created_ids)
    assert body.get("deleted", []) == []


def test_list_jobs_marks_stale_queued_as_error_when_missing_from_broker(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.queue_stale_grace_seconds = 0
    manager.queue_lost_confirm_seconds = 0

    now = 1_772_337_360.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    # queue-lost の確定には「worker も当該 task を知らない」ことの確認が必要
    # （worker 生存が不明な場合は race を避けて error 化しない現行契約）。
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: set(),
    )

    job = JobRecord(
        id="deadbeefdeadbeefdeadbeefdeadbeef",
        type="make_label_list",
        args={},
        status="queued",
        created_at=now - 60,
        command=["python", "-u", "make_label_list.py"],
        started_at=None,
        ended_at=None,
        exit_code=None,
        progress={"phase": "queued", "message": "queued", "updated_at": now - 60},
        workspace="ws51aa13",
        experiment="ex51aa13",
        worker_task_id="11111111-2222-3333-4444-555555555555",
    )
    manager._save_job(job)

    got = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got) == 1
    assert got[0].status == "error"
    assert got[0].exit_code == -1
    log_path = tmp_path / "state" / "jobs" / f"{job.id}.log"
    assert log_path.exists()
    assert "queue_lost" in log_path.read_text(encoding="utf-8")


def test_list_jobs_keeps_queued_when_broker_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.queue_stale_grace_seconds = 0

    now = 1_772_337_460.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: None,
    )

    job = JobRecord(
        id="cafebabecafebabecafebabecafebabe",
        type="make_label_list",
        args={},
        status="queued",
        created_at=now - 60,
        command=["python", "-u", "make_label_list.py"],
        started_at=None,
        ended_at=None,
        exit_code=None,
        progress={"phase": "queued", "message": "queued", "updated_at": now - 60},
        workspace="ws51aa14",
        experiment="ex51aa14",
        worker_task_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    manager._save_job(job)

    got = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got) == 1
    assert got[0].status == "queued"


def test_list_jobs_marks_stale_running_as_error_when_worker_lost(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.running_stale_grace_seconds = 0
    manager.running_lost_confirm_seconds = 0

    now = 1_772_338_060.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: set(),
    )

    job = JobRecord(
        id="11112222333344445555666677778888",
        type="train_epoch",
        args={},
        status="running",
        created_at=now - 300,
        command=["python", "-u", "train_epoch.py"],
        started_at=now - 250,
        ended_at=None,
        exit_code=None,
        progress={"phase": "train", "message": "running", "updated_at": now - 240},
        workspace="ws51aa15",
        experiment="ex51aa15",
        worker_task_id="aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
    )
    manager._save_job(job)

    got = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got) == 1
    assert got[0].status == "error"
    assert got[0].exit_code == -1
    assert "worker restart" in str(got[0].error or "").lower()
    log_path = tmp_path / "state" / "jobs" / f"{job.id}.log"
    assert log_path.exists()
    assert "worker_lost" in log_path.read_text(encoding="utf-8")


def test_list_jobs_keeps_running_when_worker_state_unknown(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.running_stale_grace_seconds = 0

    now = 1_772_338_160.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: None,
    )

    job = JobRecord(
        id="9999aaaabbbbccccddddeeeeffff0000",
        type="train_epoch",
        args={},
        status="running",
        created_at=now - 300,
        command=["python", "-u", "train_epoch.py"],
        started_at=now - 250,
        ended_at=None,
        exit_code=None,
        progress={"phase": "train", "message": "running", "updated_at": now - 240},
        workspace="ws51aa16",
        experiment="ex51aa16",
        worker_task_id="cccccccc-1111-2222-3333-dddddddddddd",
    )
    manager._save_job(job)

    got = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got) == 1
    assert got[0].status == "running"


def test_celery_backend_active_task_ids_returns_none_when_state_apis_fail(
    monkeypatch, tmp_path: Path
) -> None:
    from app.jobs.backends.celery import CeleryBackend

    class _FlakyInspect:
        def ping(self):
            return {"worker@local": {"ok": "pong"}}

        def active(self):
            raise RuntimeError("active failed")

        def reserved(self):
            raise RuntimeError("reserved failed")

        def scheduled(self):
            raise RuntimeError("scheduled failed")

    monkeypatch.setattr(
        "app.jobs.backends.celery.celery_app.control.inspect",
        lambda timeout=1.0: _FlakyInspect(),
    )

    backend = CeleryBackend()
    assert backend.active_task_ids() is None

    # manager は backend の None（状態不明）をそのまま伝播する（fail/open）。
    manager = JobsManager(ConfigManager(tmp_path), backend=backend)
    assert manager._worker_known_task_ids() is None


def test_delete_jobs_any_reconciles_stale_running_before_skip(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.running_stale_grace_seconds = 0
    manager.running_lost_confirm_seconds = 0

    now = 1_772_338_560.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: set(),
    )

    job = JobRecord(
        id="12121212121212121212121212121212",
        type="train_epoch",
        args={},
        status="running",
        created_at=now - 120,
        command=["python", "-u", "train_epoch.py"],
        started_at=now - 110,
        ended_at=None,
        exit_code=None,
        progress={"phase": "train", "message": "running", "updated_at": now - 100},
        workspace="ws51aa17",
        experiment="ex51aa17",
        worker_task_id="12345678-1234-1234-1234-123456789abc",
    )
    manager._save_job(job)

    out = manager.delete_jobs_any(job_ids=[job.id])
    assert out["skipped_running"] == []
    assert out["deleted"] == [job.id]


def test_reconcile_uses_ttl_cache(monkeypatch, tmp_path: Path) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.reconcile_ttl_seconds = 7.0
    manager.queue_stale_grace_seconds = 0

    now = {"value": 1_772_339_000.0}
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now["value"])

    # broker への問い合わせは「stale 候補が存在するとき」だけ発生する。
    # TTL キャッシュの効き方を観測するため、常に stale 候補となる queued job を 1 件置く。
    stale_job = JobRecord(
        id="cafebabecafebabecafebabecafebabe",
        type="make_label_list",
        args={},
        status="queued",
        created_at=now["value"] - 60,
        command=["python", "-u", "make_label_list.py"],
        started_at=None,
        ended_at=None,
        exit_code=None,
        progress={"phase": "queued", "message": "queued", "updated_at": now["value"] - 60},
        workspace="ws51aa14",
        experiment="ex51aa14",
        worker_task_id="99999999-8888-7777-6666-555555555555",
    )
    manager._save_job(stale_job)

    calls = {"broker": 0, "worker": 0}

    def _broker(self):
        calls["broker"] += 1
        return set()

    def _worker(self):
        calls["worker"] += 1
        return set()

    monkeypatch.setattr("app.jobs.manager.JobsManager._broker_queued_task_ids", _broker)
    monkeypatch.setattr("app.jobs.manager.JobsManager._worker_known_task_ids", _worker)

    manager._reconcile_stale_jobs()
    now["value"] += 1.0
    manager._reconcile_stale_jobs()
    now["value"] += 8.0
    manager._reconcile_stale_jobs()

    assert calls["broker"] == 2
    assert calls["worker"] == 2


def test_list_jobs_keeps_queued_when_task_is_reserved_on_worker(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.queue_stale_grace_seconds = 0
    manager.queue_lost_confirm_seconds = 0

    now = 1_772_339_100.0
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now)
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: {"11111111-2222-3333-4444-555555555555"},
    )

    job = JobRecord(
        id="abcdabcdabcdabcdabcdabcdabcdabcd",
        type="make_label_list",
        args={},
        status="queued",
        created_at=now - 60,
        command=["python", "-u", "make_label_list.py"],
        started_at=None,
        ended_at=None,
        exit_code=None,
        progress={"phase": "queued", "message": "queued", "updated_at": now - 60},
        workspace="ws51aa18",
        experiment="ex51aa18",
        worker_task_id="11111111-2222-3333-4444-555555555555",
    )
    manager._save_job(job)

    got = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got) == 1
    assert got[0].status == "queued"


def test_running_lost_requires_confirm_window(
    monkeypatch, tmp_path: Path
) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    manager.running_stale_grace_seconds = 0
    manager.running_lost_confirm_seconds = 10

    now = {"value": 1_772_339_200.0}
    monkeypatch.setattr("app.jobs.manager.time.time", lambda: now["value"])
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._broker_queued_task_ids",
        lambda self: set(),
    )
    monkeypatch.setattr(
        "app.jobs.manager.JobsManager._worker_known_task_ids",
        lambda self: set(),
    )

    job = JobRecord(
        id="eeeeffffeeeeffffeeeeffffeeeeffff",
        type="train_epoch",
        args={},
        status="running",
        created_at=now["value"] - 300,
        command=["python", "-u", "train_epoch.py"],
        started_at=now["value"] - 250,
        ended_at=None,
        exit_code=None,
        progress={
            "phase": "train",
            "message": "running",
            "updated_at": now["value"] - 240,
        },
        workspace="ws51aa19",
        experiment="ex51aa19",
        worker_task_id="aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
    )
    manager._save_job(job)

    got1 = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got1) == 1
    assert got1[0].status == "running"

    now["value"] += 11
    got2 = manager.list_jobs_filtered(limit=10, offset=0)
    assert len(got2) == 1
    assert got2[0].status == "error"


def test_cancel_queued_job_marks_canceled_and_logs(monkeypatch, tmp_path: Path) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    monkeypatch.setattr("app.jobs.manager.JobsManager._reconcile_stale_jobs", lambda self: None)

    revoked: list[dict] = []

    def fake_revoke(self, task_id: str, *, terminate: bool = False):
        revoked.append({"task_id": task_id, "terminate": terminate})

    monkeypatch.setattr("app.jobs.backends.local.LocalBackend.revoke", fake_revoke)

    job = JobRecord(
        id="abababababababababababababababab",
        type="make_label_list",
        args={},
        status="queued",
        created_at=1_772_339_300.0,
        command=["python", "-u", "make_label_list.py"],
        workspace="ws51aa20",
        experiment="ex51aa20",
        worker_task_id="11111111-2222-3333-4444-555555555555",
    )
    manager._save_job(job)

    out = manager.cancel_job(job.id)
    assert out.status == "canceled"
    assert out.exit_code == -1
    assert revoked == [
        {
            "task_id": "11111111-2222-3333-4444-555555555555",
            "terminate": False,
        }
    ]
    log_path = tmp_path / "state" / "jobs" / f"{job.id}.log"
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "[cancel]" in text
    assert "[canceled]" in text


def test_cancel_queued_job_with_started_at_terminates_worker(monkeypatch, tmp_path: Path) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    monkeypatch.setattr("app.jobs.manager.JobsManager._reconcile_stale_jobs", lambda self: None)

    revoked: list[dict] = []

    def fake_revoke(self, task_id: str, *, terminate: bool = False):
        revoked.append({"task_id": task_id, "terminate": terminate})

    monkeypatch.setattr("app.jobs.backends.local.LocalBackend.revoke", fake_revoke)

    job = JobRecord(
        id="10101010101010101010101010101010",
        type="train_epoch",
        args={},
        status="queued",
        created_at=1_772_339_350.0,
        started_at=1_772_339_351.0,
        command=["python", "-u", "train_epoch.py"],
        workspace="ws51aa20",
        experiment="ex51aa20",
        worker_task_id="22222222-3333-4444-5555-666666666666",
    )
    manager._save_job(job)

    out = manager.cancel_job(job.id)
    assert out.status == "canceled"
    assert revoked == [
        {
            "task_id": "22222222-3333-4444-5555-666666666666",
            "terminate": True,
        }
    ]


def test_cancel_done_job_rejected(monkeypatch, tmp_path: Path) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    monkeypatch.setattr("app.jobs.manager.JobsManager._reconcile_stale_jobs", lambda self: None)

    job = JobRecord(
        id="cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
        type="make_label_list",
        args={},
        status="done",
        created_at=1_772_339_400.0,
        command=["python", "-u", "make_label_list.py"],
        workspace="ws51aa21",
        experiment="ex51aa21",
    )
    manager._save_job(job)

    try:
        manager.cancel_job(job.id)
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 409


def test_rerun_done_job_creates_new_queued_job(monkeypatch, tmp_path: Path) -> None:
    manager = JobsManager(ConfigManager(tmp_path))
    monkeypatch.setattr("app.jobs.manager.JobsManager._reconcile_stale_jobs", lambda self: None)

    # create workspace root required by create_job
    ws = "ws51aa22"
    ex = "ex51aa22"
    (tmp_path / ws).mkdir(parents=True, exist_ok=True)

    seq = {"n": 0}

    def fake_enqueue(
        self, *, job_id: str, command: list[str], workspace: str | None, task_id: str = ""
    ) -> str:
        seq["n"] += 1
        return task_id or f"task-rerun-{seq['n']}"

    monkeypatch.setattr(
        "app.jobs.backends.local.LocalBackend.enqueue", fake_enqueue
    )

    source = JobRecord(
        id="efefefefefefefefefefefefefefefef",
        type="make_label_list",
        args={},
        status="done",
        created_at=1_772_339_500.0,
        command=["python", "-u", "make_label_list.py"],
        workspace=ws,
        experiment=ex,
        ended_at=1_772_339_550.0,
        exit_code=0,
    )
    manager._save_job(source)

    out = manager.rerun_job(source.id)
    assert out.id != source.id
    assert out.status == "queued"
    assert out.type == source.type
    assert out.workspace == ws
    assert out.experiment == ex

    src_log = (tmp_path / "state" / "jobs" / f"{source.id}.log").read_text(
        encoding="utf-8"
    )
    assert "[rerun]" in src_log
