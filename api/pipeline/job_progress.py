#!/usr/bin/env python3
"""agent/pipeline/job_progress.py

Shared, best-effort job progress reporting for pipeline scripts.

- Scripts are executed via JobsManager which injects:
  - AGENT_JOB_ID
  - AGENT_JOB_FILE (absolute path to agent/state/jobs/<id>.json)

This module intentionally never raises on failures.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def job_file_from_env() -> Optional[Path]:
    p = os.environ.get("AGENT_JOB_FILE")
    if not p:
        return None
    try:
        return Path(p)
    except Exception:
        return None


def update_job_progress(
    *,
    phase: str,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Update state/jobs/<id>.json with a progress object.

    This is best-effort and safe to call frequently.
    """

    job_file = job_file_from_env()
    if job_file is None:
        return

    try:
        if job_file.exists():
            data = json.loads(job_file.read_text(encoding="utf-8"))
        else:
            data = {}
        if not isinstance(data, dict):
            return

        progress = data.get("progress")
        if not isinstance(progress, dict):
            progress = {}

        progress.update(
            {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "updated_at": time.time(),
            }
        )
        if extra:
            progress.update(extra)

        data["progress"] = progress

        tmp = job_file.with_suffix(job_file.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(job_file)
    except Exception:
        return
