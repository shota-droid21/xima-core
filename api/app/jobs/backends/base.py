"""ジョブ実行 backend のインターフェース。

`manager.py` は broker の実体を知らない。現行の broker 依存点と同じ 4 操作だけを
公開する（Decision 036）。

`queued_task_ids` / `active_task_ids` が `None` を返すのは「状態を確定できない」の意で、
`manager` はその場合に stale 判定を行わない（fail/open）。in-process backend は状態が
常に確定するため `None` を返さない。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Set, runtime_checkable


@runtime_checkable
class JobBackend(Protocol):
    #: ログ・診断用の識別子（`local` / `celery`）
    name: str

    def enqueue(
        self,
        *,
        job_id: str,
        command: List[str],
        workspace: Optional[str],
        task_id: str,
    ) -> str:
        """ジョブを投入し、実際に採番された task id を返す。失敗時は例外を送出する。"""
        ...

    def queued_task_ids(self) -> Optional[Set[str]]:
        """実行待ちの task id 集合。確定できない場合は None。"""
        ...

    def active_task_ids(self) -> Optional[Set[str]]:
        """ワーカーが把握している（実行中 / 予約済み）task id 集合。確定できない場合は None。"""
        ...

    def revoke(self, task_id: str, *, terminate: bool) -> None:
        """投入済みジョブを取り消す。`terminate` が真なら実行中プロセスも停止する。"""
        ...
