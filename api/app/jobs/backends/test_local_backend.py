"""LocalBackend（in-process ジョブ backend）の単体テスト。

正本は `state/jobs/<job_id>.json`。ここでは backend が「投入 → 実行 → 終了状態の書き込み」
と「取り消し」を満たすかだけを見る（Decision 036）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.config import ConfigManager
from app.jobs.backends import build_backend
from app.jobs.backends.local import LocalBackend


def _seed_job(tmp_path: Path, job_id: str, task_id: str) -> Path:
    jobs_dir = tmp_path / "state" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = jobs_dir / f"{job_id}.json"
    path.write_text(
        json.dumps(
            {
                "id": job_id,
                "type": "make_label_list",
                "status": "queued",
                "created_at": time.time(),
                "worker_task_id": task_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _wait_for_status(path: Path, expected: set[str], timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    payload: dict = {}
    while time.time() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = {}
        if str(payload.get("status") or "") in expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"status={payload.get('status')!r} did not reach {expected}")


def test_build_backend_defaults_to_local(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XIMA_JOB_BACKEND", raising=False)
    backend = build_backend(ConfigManager(tmp_path))
    assert backend.name == "local"


def test_build_backend_falls_back_to_local_for_unknown_value(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XIMA_JOB_BACKEND", "no-such-backend")
    backend = build_backend(ConfigManager(tmp_path))
    assert backend.name == "local"


def test_enqueue_runs_job_and_writes_done(tmp_path: Path) -> None:
    config_manager = ConfigManager(tmp_path)
    backend = LocalBackend(config_manager)
    job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    task_id = "11111111-2222-3333-4444-555555555555"
    path = _seed_job(tmp_path, job_id, task_id)

    returned = backend.enqueue(
        job_id=job_id,
        command=[sys.executable, "-c", "print('hello from job')"],
        workspace=None,
        task_id=task_id,
    )
    assert returned == task_id

    payload = _wait_for_status(path, {"done", "error"})
    assert payload["status"] == "done"
    assert payload["exit_code"] == 0

    log_text = (tmp_path / "state" / "jobs" / f"{job_id}.log").read_text(encoding="utf-8")
    assert "hello from job" in log_text


def test_nonzero_exit_marks_error(tmp_path: Path) -> None:
    config_manager = ConfigManager(tmp_path)
    backend = LocalBackend(config_manager)
    job_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    task_id = "22222222-3333-4444-5555-666666666666"
    path = _seed_job(tmp_path, job_id, task_id)

    backend.enqueue(
        job_id=job_id,
        command=[sys.executable, "-c", "raise SystemExit(3)"],
        workspace=None,
        task_id=task_id,
    )

    payload = _wait_for_status(path, {"done", "error"})
    assert payload["status"] == "error"
    assert payload["exit_code"] == 3


def test_revoke_queued_job_is_never_executed(tmp_path: Path) -> None:
    """並列度 1 なので、先行ジョブの実行中に後続を revoke すれば起動しない。"""
    config_manager = ConfigManager(tmp_path)
    backend = LocalBackend(config_manager)

    blocker_id = "cccccccccccccccccccccccccccccccc"
    blocker_task = "33333333-4444-5555-6666-777777777777"
    blocker_path = _seed_job(tmp_path, blocker_id, blocker_task)

    victim_id = "dddddddddddddddddddddddddddddddd"
    victim_task = "44444444-5555-6666-7777-888888888888"
    victim_path = _seed_job(tmp_path, victim_id, victim_task)
    marker = tmp_path / "victim-ran.txt"

    backend.enqueue(
        job_id=blocker_id,
        command=[sys.executable, "-c", "import time; time.sleep(3)"],
        workspace=None,
        task_id=blocker_task,
    )
    backend.enqueue(
        job_id=victim_id,
        command=[
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').write('ran')",
        ],
        workspace=None,
        task_id=victim_task,
    )

    # 先行ジョブが走り出す（= 後続はキュー待ち）まで待つ。
    _wait_for_status(blocker_path, {"running", "done", "error"})
    assert victim_task in (backend.queued_task_ids() or set())

    backend.revoke(victim_task, terminate=False)
    assert victim_task not in (backend.queued_task_ids() or set())

    _wait_for_status(blocker_path, {"done", "error"}, timeout=20.0)
    time.sleep(0.5)

    assert not marker.exists()
    assert json.loads(victim_path.read_text(encoding="utf-8"))["status"] == "queued"


def test_revoke_running_job_terminates_process(tmp_path: Path) -> None:
    config_manager = ConfigManager(tmp_path)
    backend = LocalBackend(config_manager)
    job_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    task_id = "55555555-6666-7777-8888-999999999999"
    path = _seed_job(tmp_path, job_id, task_id)

    backend.enqueue(
        job_id=job_id,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        workspace=None,
        task_id=task_id,
    )

    _wait_for_status(path, {"running"})
    assert task_id in (backend.active_task_ids() or set())

    backend.revoke(task_id, terminate=True)

    payload = _wait_for_status(path, {"done", "error"}, timeout=20.0)
    assert payload["status"] == "error"
    assert payload["exit_code"] != 0


def test_task_id_sets_are_authoritative(tmp_path: Path) -> None:
    """in-process なので状態不明（None）を返さない。"""
    backend = LocalBackend(ConfigManager(tmp_path))
    assert backend.queued_task_ids() == set()
    assert backend.active_task_ids() == set()
