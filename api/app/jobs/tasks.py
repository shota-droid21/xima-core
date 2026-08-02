"""celery タスク定義。

ジョブ実体の実行は `execute.run_job`（backend 非依存）に委譲する。本モジュールは
celery への登録と投入だけを担う。単一マシンの既定は `backends/local.py` であり、
本モジュールは docker compose 構成（`XIMA_JOB_BACKEND=celery`）でのみ使われる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .celery_app import celery_app
from .execute import run_job
from ..config import ConfigManager

_BASE_DIR = Path(__file__).resolve().parents[2]
_CONFIG_MANAGER = ConfigManager(_BASE_DIR)


@celery_app.task(name="app.jobs.run_job")
def run_job_task(job_id: str, command: List[str], workspace: str | None = None) -> None:
    run_job(_CONFIG_MANAGER, job_id, command, workspace)


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
