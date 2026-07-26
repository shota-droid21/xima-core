from __future__ import annotations

import json
import os
import time
import tempfile
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Any, List

from .celery_app import celery_app
from ..config import ConfigManager

_BASE_DIR = Path(__file__).resolve().parents[2]
_CONFIG_MANAGER = ConfigManager(_BASE_DIR)


def _job_paths(job_id: str) -> dict[str, Path]:
    jobs_dir = _CONFIG_MANAGER.jobs_dir
    return {
        "json": jobs_dir / f"{job_id}.json",
        "log": jobs_dir / f"{job_id}.log",
    }


def _load_job_payload(job_id: str) -> dict[str, Any]:
    paths = _job_paths(job_id)
    if not paths["json"].exists():
        return {"id": job_id}
    for _ in range(3):
        try:
            data = json.loads(paths["json"].read_text())
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            time.sleep(0.02)
    return {"id": job_id}


def _read_job_payload_or_none(job_id: str) -> dict[str, Any] | None:
    paths = _job_paths(job_id)
    if not paths["json"].exists():
        return None
    try:
        data = json.loads(paths["json"].read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_job_payload(job_id: str, payload: dict[str, Any]) -> None:
    paths = _job_paths(job_id)
    existing = _read_job_payload_or_none(job_id)
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(payload)
        payload = merged

    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=paths["json"].parent,
            prefix=f".{paths['json'].name}.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write(json.dumps(payload, indent=2))
            tf.flush()
            os.fsync(tf.fileno())
            tmp_path = Path(tf.name)
        os.replace(tmp_path, paths["json"])
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:  # noqa: BLE001
                pass


@celery_app.task(name="app.jobs.run_job")
def run_job_task(job_id: str, command: List[str], workspace: str | None = None) -> None:
    paths = _job_paths(job_id)
    cfg = _CONFIG_MANAGER.get_config()

    job = _load_job_payload(job_id)
    if str(job.get("status") or "").strip().lower() == "canceled":
        with open(paths["log"], "a", encoding="utf-8") as log_file:
            log_file.write("Job was canceled before execution; skip run.\n")
        return

    now = time.time()
    job["status"] = "running"
    job["started_at"] = now
    _save_job_payload(job_id, job)

    exit_code = -1
    try:
        with open(paths["log"], "a", encoding="utf-8") as log_file:
            env = os.environ.copy()
            env["AGENT_JOB_ID"] = job_id
            env["AGENT_JOB_FILE"] = str(paths["json"].resolve())

            try:
                cwd = str(cfg.workspace_path(workspace or ""))
            except Exception:
                cwd = str(cfg.workspaces_root)

            process = Popen(
                command,
                cwd=cwd,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
            process.wait()
            exit_code = int(process.returncode)
    except Exception as exc:  # noqa: BLE001
        with open(paths["log"], "a", encoding="utf-8") as log_file:
            log_file.write(f"Job failed to start: {exc}\n")
        exit_code = -1
    finally:
        ended_at = time.time()
        latest = _load_job_payload(job_id)
        progress = latest.get("progress") if isinstance(latest, dict) else None
        latest_status = str(latest.get("status") or "").strip().lower()

        if latest_status == "canceled":
            latest["ended_at"] = latest.get("ended_at") or ended_at
            if latest.get("exit_code") is None:
                latest["exit_code"] = -1
            if progress is not None:
                latest["progress"] = progress
        else:
            latest["ended_at"] = ended_at
            latest["exit_code"] = exit_code
            latest["status"] = "done" if exit_code == 0 else "error"
            if progress is not None:
                latest["progress"] = progress
        _save_job_payload(job_id, latest)


def enqueue_job_task(
    *,
    job_id: str,
    command: List[str],
    workspace: str | None,
    task_id: str | None = None,
) -> Any:
    if isinstance(task_id, str) and task_id.strip():
        return run_job_task.apply_async(
            kwargs={
                "job_id": job_id,
                "command": command,
                "workspace": workspace,
            },
            task_id=task_id.strip(),
        )
    return run_job_task.delay(job_id=job_id, command=command, workspace=workspace)
