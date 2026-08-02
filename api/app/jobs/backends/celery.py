"""celery + redis(broker) ジョブ backend。

docker compose 構成（GPU worker / 複数マシン）向け。`XIMA_JOB_BACKEND=celery` で選択する。
`manager.py` にあった broker 直参照（redis lrange / celery inspect / revoke）を、挙動を
変えずにここへ移設したもの（Decision 036）。

broker 不達時は `None` を返し、`manager` 側の stale 判定を抑止する（fail/open）。
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Set

from redis import Redis

from ..celery_app import celery_app
from ..tasks import enqueue_job_task

_UUID_RE = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")


class CeleryBackend:
    name = "celery"

    def __init__(self, *, inspect_timeout_seconds: float = 1.0) -> None:
        self.inspect_timeout_seconds = float(inspect_timeout_seconds)

    # ---- JobBackend ----

    def enqueue(
        self,
        *,
        job_id: str,
        command: List[str],
        workspace: Optional[str],
        task_id: str,
    ) -> str:
        task = enqueue_job_task(
            job_id=job_id,
            command=command,
            workspace=workspace,
            task_id=task_id,
        )
        returned = str(getattr(task, "id", "") or "").strip()
        return returned or str(task_id or "").strip()

    def queued_task_ids(self) -> Optional[Set[str]]:
        broker_url = os.environ.get("XIMA_CELERY_BROKER_URL", "redis://redis:6379/0")
        queue_name = os.environ.get("XIMA_CELERY_QUEUE", "xima_jobs")

        try:
            client = Redis.from_url(broker_url, decode_responses=False)
            raw_messages = client.lrange(queue_name, 0, -1)
        except Exception:  # noqa: BLE001
            return None

        task_ids: Set[str] = set()
        for raw in raw_messages:
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                text = str(raw)

            try:
                payload = json.loads(text)
                headers = payload.get("headers")
                tid = headers.get("id") if isinstance(headers, dict) else None
                if isinstance(tid, str) and tid.strip():
                    task_ids.add(tid.strip().lower())
                    continue
            except Exception:  # noqa: BLE001
                pass

            for m in _UUID_RE.findall(text.lower()):
                task_ids.add(m)

        return task_ids

    def active_task_ids(self) -> Optional[Set[str]]:
        try:
            inspect = celery_app.control.inspect(timeout=self.inspect_timeout_seconds)
        except Exception:  # noqa: BLE001
            return None
        if inspect is None:
            return None

        try:
            ping_map = inspect.ping() or {}
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(ping_map, dict) or len(ping_map) == 0:
            # No responding worker -> unknown state, skip running-stale judgment.
            return None
        ping_workers = {str(k).strip() for k in ping_map.keys() if str(k).strip()}
        if not ping_workers:
            return None

        known: Set[str] = set()
        saw_any_state = False

        def _collect_from_requests(data: dict | None) -> None:
            if not isinstance(data, dict):
                return
            for _worker, entries in data.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    tid = e.get("id")
                    if isinstance(tid, str) and tid.strip():
                        known.add(tid.strip().lower())

        def _collect_from_scheduled(data: dict | None) -> None:
            if not isinstance(data, dict):
                return
            for _worker, entries in data.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    req = e.get("request")
                    if not isinstance(req, dict):
                        continue
                    tid = req.get("id")
                    if isinstance(tid, str) and tid.strip():
                        known.add(tid.strip().lower())

        try:
            active_data = inspect.active()
            if isinstance(active_data, dict) and len(active_data) > 0:
                saw_any_state = True
                _collect_from_requests(active_data)
        except Exception:  # noqa: BLE001
            pass
        try:
            reserved_data = inspect.reserved()
            if isinstance(reserved_data, dict) and len(reserved_data) > 0:
                saw_any_state = True
                _collect_from_requests(reserved_data)
        except Exception:  # noqa: BLE001
            pass
        try:
            scheduled_data = inspect.scheduled()
            if isinstance(scheduled_data, dict) and len(scheduled_data) > 0:
                saw_any_state = True
                _collect_from_scheduled(scheduled_data)
        except Exception:  # noqa: BLE001
            pass

        # If worker replied to ping but task-state APIs all failed, keep state unknown.
        if not saw_any_state:
            return None

        return known

    def revoke(self, task_id: str, *, terminate: bool) -> None:
        celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
