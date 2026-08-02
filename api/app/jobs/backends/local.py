"""in-process ジョブ backend（既定）。

uvicorn プロセス内のワーカースレッドがキューを取り出し、`execute.run_job` で
subprocess として実行する。celery worker が行っていた処理と同一であり、ジョブの実体は
元から subprocess である（`runner.build_command`）。

broker を持たないため **Docker / redis が不要**になる（Decision 036）。その代償として
**プロセス終了で実行中ジョブも停止する**（起動時の扱いは `manager._mark_interrupted_jobs`
が従来どおり queued / running を error にする）。

分散実行（GPU worker / 複数マシン）が要る場合は `XIMA_JOB_BACKEND=celery` を使う。
"""

from __future__ import annotations

import os
import threading
from collections import deque
from subprocess import Popen
from typing import Deque, Dict, List, Optional, Set, Tuple

from ..execute import run_job
from ...config import ConfigManager

# (task_id, job_id, command, workspace)
_QueueEntry = Tuple[str, str, List[str], Optional[str]]


class LocalBackend:
    """スレッド 1 本（既定）でキューを直列処理する backend。"""

    name = "local"

    def __init__(self, config_manager: ConfigManager, *, concurrency: int | None = None) -> None:
        self._config_manager = config_manager
        if concurrency is None:
            try:
                concurrency = int(os.environ.get("XIMA_LOCAL_JOB_CONCURRENCY", "1"))
            except Exception:  # noqa: BLE001
                concurrency = 1
        self._concurrency = max(1, concurrency)

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: Deque[_QueueEntry] = deque()
        self._running: Dict[str, Optional[Popen]] = {}
        self._revoked: Set[str] = set()
        self._workers: List[threading.Thread] = []
        self._started = False

    # ---- JobBackend ----

    def enqueue(
        self,
        *,
        job_id: str,
        command: List[str],
        workspace: Optional[str],
        task_id: str,
    ) -> str:
        tid = str(task_id or "").strip().lower()
        if not tid:
            raise ValueError("task_id is required for LocalBackend")
        self._ensure_workers()
        with self._wake:
            self._revoked.discard(tid)
            self._pending.append((tid, job_id, list(command), workspace))
            self._wake.notify()
        return tid

    def queued_task_ids(self) -> Optional[Set[str]]:
        with self._lock:
            return {entry[0] for entry in self._pending}

    def active_task_ids(self) -> Optional[Set[str]]:
        with self._lock:
            return set(self._running.keys())

    def revoke(self, task_id: str, *, terminate: bool) -> None:
        tid = str(task_id or "").strip().lower()
        if not tid:
            return
        with self._lock:
            self._revoked.add(tid)
            remaining = deque(e for e in self._pending if e[0] != tid)
            self._pending = remaining
            process = self._running.get(tid)

        if terminate and process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:  # noqa: BLE001
                pass

    # ---- worker ----

    def _ensure_workers(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for i in range(self._concurrency):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"xima-job-worker-{i}",
                    daemon=True,
                )
                self._workers.append(t)
                t.start()

    def _take_next(self) -> Optional[_QueueEntry]:
        """pending から 1 件取り出し、同一ロック下で running に登録する。

        pending からも running からも消えた「中間状態」を作らないことが重要。
        中間状態があると `manager._reconcile_stale_jobs` が実行直前のジョブを
        queue_lost と誤判定しうる。
        """
        with self._wake:
            while not self._pending:
                self._wake.wait(timeout=1.0)
                if not self._pending:
                    continue
            entry = self._pending.popleft()
            self._running[entry[0]] = None
            return entry

    def _worker_loop(self) -> None:
        while True:
            entry = self._take_next()
            if entry is None:
                continue
            task_id, job_id, command, workspace = entry
            try:
                with self._lock:
                    revoked = task_id in self._revoked
                if revoked:
                    continue

                def _on_start(process: Popen, _tid: str = task_id) -> None:
                    with self._lock:
                        self._running[_tid] = process

                run_job(
                    self._config_manager,
                    job_id,
                    command,
                    workspace,
                    on_start=_on_start,
                )
            except Exception:  # noqa: BLE001
                # run_job は内部で状態を error に落とすため、ここでは worker を
                # 落とさないことだけを保証する。
                pass
            finally:
                with self._lock:
                    self._running.pop(task_id, None)
                    self._revoked.discard(task_id)
