"""ジョブ実体（subprocess）の実行。backend 非依存。

`tasks.py`（celery タスク）と `backends/local.py`（in-process ワーカー）の双方から
呼ばれる共通部。ジョブの正本は `state/jobs/<job_id>.json` / `.log`（Decision 004）で
あり、本モジュールはその読み書きと Popen の実行だけを担う。

設計正本: `docs/decisions.md` Decision 036。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Any, Callable, Dict, List, Optional

from ..config import ConfigManager


def job_paths(config_manager: ConfigManager, job_id: str) -> Dict[str, Path]:
    jobs_dir = config_manager.jobs_dir
    return {
        "json": jobs_dir / f"{job_id}.json",
        "log": jobs_dir / f"{job_id}.log",
    }


def read_job_payload_or_none(
    config_manager: ConfigManager, job_id: str
) -> Optional[Dict[str, Any]]:
    paths = job_paths(config_manager, job_id)
    if not paths["json"].exists():
        return None
    try:
        data = json.loads(paths["json"].read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_job_payload(config_manager: ConfigManager, job_id: str) -> Dict[str, Any]:
    paths = job_paths(config_manager, job_id)
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


def save_job_payload(
    config_manager: ConfigManager, job_id: str, payload: Dict[str, Any]
) -> None:
    paths = job_paths(config_manager, job_id)
    existing = read_job_payload_or_none(config_manager, job_id)
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


def run_job(
    config_manager: ConfigManager,
    job_id: str,
    command: List[str],
    workspace: Optional[str] = None,
    *,
    on_start: Optional[Callable[[Popen], None]] = None,
) -> None:
    """ジョブを subprocess として実行し、状態とログを正本へ書き込む。

    `on_start` は Popen 生成直後に呼ばれる。in-process backend が実行中プロセスを
    握って revoke(terminate) できるようにするためのフック（celery 側は使わない）。
    """
    paths = job_paths(config_manager, job_id)
    cfg = config_manager.get_config()

    job = load_job_payload(config_manager, job_id)
    if str(job.get("status") or "").strip().lower() == "canceled":
        with open(paths["log"], "a", encoding="utf-8") as log_file:
            log_file.write("Job was canceled before execution; skip run.\n")
        return

    now = time.time()
    job["status"] = "running"
    job["started_at"] = now
    save_job_payload(config_manager, job_id, job)

    exit_code = -1
    try:
        with open(paths["log"], "a", encoding="utf-8") as log_file:
            env = os.environ.copy()
            env["AGENT_JOB_ID"] = job_id
            env["AGENT_JOB_FILE"] = str(paths["json"].resolve())

            try:
                cwd = str(cfg.workspace_path(workspace or ""))
            except Exception:  # noqa: BLE001
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
            if on_start is not None:
                try:
                    on_start(process)
                except Exception:  # noqa: BLE001
                    pass
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
        latest = load_job_payload(config_manager, job_id)
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
        save_job_payload(config_manager, job_id, latest)
