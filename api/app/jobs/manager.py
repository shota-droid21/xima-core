from __future__ import annotations

import json
import os
import re
import threading
import time
import tempfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Query

from .backends import JobBackend, build_backend
from .runner import build_command
from .schemas import JOB_TYPES, JobCreateRequest, JobDeleteRequest, JobRecord
from ..config import ConfigManager
from ..label_input import ensure_label_file
from ..utils.short_id import require_experiment_id, require_workspace_id


class JobsManager:
    def __init__(
        self, config_manager: ConfigManager, *, backend: JobBackend | None = None
    ) -> None:
        self.config_manager = config_manager
        self.jobs_dir = config_manager.jobs_dir
        self.lock = threading.Lock()
        self.queue_stale_grace_seconds = int(
            os.environ.get("XIMA_QUEUE_STALE_GRACE_SECONDS", "30")
        )
        self.running_stale_grace_seconds = int(
            os.environ.get("XIMA_RUNNING_STALE_GRACE_SECONDS", "45")
        )
        self.queue_lost_confirm_seconds = float(
            os.environ.get("XIMA_QUEUE_LOST_CONFIRM_SECONDS", "5")
        )
        self.running_lost_confirm_seconds = float(
            os.environ.get("XIMA_RUNNING_LOST_CONFIRM_SECONDS", "10")
        )
        self.worker_inspect_timeout_seconds = float(
            os.environ.get("XIMA_CELERY_INSPECT_TIMEOUT_SECONDS", "1.0")
        )
        try:
            ttl = float(os.environ.get("XIMA_RECONCILE_TTL_SECONDS", "7"))
        except Exception:  # noqa: BLE001
            ttl = 7.0
        self.reconcile_ttl_seconds = max(0.0, ttl)
        self._reconcile_cache_lock = threading.Lock()
        self._reconcile_cache_until = 0.0
        self._missing_task_first_seen: dict[str, float] = {}
        self.backend: JobBackend = backend or build_backend(
            config_manager,
            inspect_timeout_seconds=self.worker_inspect_timeout_seconds,
        )
        # プロセス終了で in-process ジョブは失われるため、起動時に queued / running を
        # error へ落とす（LocalBackend では実態と一致する。Decision 036）。
        self._mark_interrupted_jobs()

    def _job_paths(self, job_id: str) -> Dict[str, Path]:
        return {
            "json": self.jobs_dir / f"{job_id}.json",
            "log": self.jobs_dir / f"{job_id}.log",
        }

    def _save_job(self, job: JobRecord) -> None:
        paths = self._job_paths(job.id)
        data = job.to_dict()
        # Preserve progress that may be written by the running script.
        try:
            if paths["json"].exists():
                existing = json.loads(paths["json"].read_text())
                if isinstance(existing, dict) and existing.get("progress") is not None:
                    if data.get("progress") is None:
                        data["progress"] = existing.get("progress")
        except Exception:  # noqa: BLE001
            pass

        self._write_json_atomic(paths["json"], data)

    def _load_job(self, job_id: str) -> JobRecord:
        paths = self._job_paths(job_id)
        if not paths["json"].exists():
            raise FileNotFoundError(job_id)
        data = json.loads(paths["json"].read_text())
        return JobRecord(**data)

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tf.write(json.dumps(payload, indent=2))
                tf.flush()
                os.fsync(tf.fileno())
                tmp_path = Path(tf.name)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:  # noqa: BLE001
                    pass

    def _append_job_log(self, job_id: str, message: str, *, ts: float | None = None) -> None:
        paths = self._job_paths(job_id)
        when = float(ts if ts is not None else time.time())
        prefix = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
        try:
            with open(paths["log"], "a", encoding="utf-8") as fp:
                fp.write(f"[{prefix}] {message}\n")
        except Exception:  # noqa: BLE001
            return

    def _is_active_status(self, status: str | None) -> bool:
        return str(status or "").strip().lower() in ("queued", "running")

    def _is_terminal_status(self, status: str | None) -> bool:
        return not self._is_active_status(status)

    def _broker_queued_task_ids(self) -> set[str] | None:
        """実行待ちの task id 集合。backend が状態を確定できない場合は None。"""
        try:
            return self.backend.queued_task_ids()
        except Exception:  # noqa: BLE001
            return None

    def _worker_known_task_ids(self) -> set[str] | None:
        """ワーカーが把握している task id 集合。確定できない場合は None。"""
        try:
            return self.backend.active_task_ids()
        except Exception:  # noqa: BLE001
            return None

    def _clear_missing_task(self, task_id: str) -> None:
        if not task_id:
            return
        with self._reconcile_cache_lock:
            self._missing_task_first_seen.pop(task_id, None)

    def _confirm_missing_task(
        self, task_id: str, *, now: float, confirm_seconds: float
    ) -> bool:
        if not task_id:
            return False
        if confirm_seconds <= 0:
            self._clear_missing_task(task_id)
            return True
        with self._reconcile_cache_lock:
            first = self._missing_task_first_seen.get(task_id)
            if first is None:
                self._missing_task_first_seen[task_id] = now
                return False
            if now - first < float(confirm_seconds):
                return False
            self._missing_task_first_seen.pop(task_id, None)
            return True

    def _reconcile_stale_jobs(self) -> None:
        now = time.time()
        if self.reconcile_ttl_seconds > 0:
            with self._reconcile_cache_lock:
                if now < self._reconcile_cache_until:
                    return
                self._reconcile_cache_until = now + self.reconcile_ttl_seconds

        has_stale_candidate = False
        for p in self.jobs_dir.glob("*.json"):
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue

            status = str(payload.get("status") or "").strip().lower()
            task_id = str(payload.get("worker_task_id") or "").strip().lower()
            if not task_id:
                continue

            created_raw = payload.get("created_at")
            try:
                created_at = float(created_raw or 0)
            except Exception:  # noqa: BLE001
                created_at = 0.0
            if created_at <= 0:
                self._clear_missing_task(task_id)
                continue

            if status == "queued":
                if payload.get("started_at") is None and now - created_at >= float(
                    self.queue_stale_grace_seconds
                ):
                    has_stale_candidate = True
                    break
                self._clear_missing_task(task_id)
                continue

            if status == "running":
                progress = payload.get("progress")
                updated_at = None
                if isinstance(progress, dict):
                    raw_updated = progress.get("updated_at")
                    try:
                        updated_at = float(raw_updated)
                    except Exception:  # noqa: BLE001
                        updated_at = None
                started_raw = payload.get("started_at")
                try:
                    started_at = float(started_raw) if started_raw is not None else None
                except Exception:  # noqa: BLE001
                    started_at = None
                last_seen = max(
                    [t for t in (updated_at, started_at, created_at) if isinstance(t, float)],
                    default=created_at,
                )
                if now - last_seen >= float(self.running_stale_grace_seconds):
                    has_stale_candidate = True
                    break
                self._clear_missing_task(task_id)
                continue

            self._clear_missing_task(task_id)

        if not has_stale_candidate:
            return

        queued_task_ids = self._broker_queued_task_ids()
        if queued_task_ids is None:
            return
        worker_known_task_ids = self._worker_known_task_ids()
        now = time.time()
        for p in self.jobs_dir.glob("*.json"):
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue

            status = str(payload.get("status") or "").strip().lower()
            task_id = str(payload.get("worker_task_id") or "").strip().lower()
            if not task_id:
                continue

            created_raw = payload.get("created_at")
            try:
                created_at = float(created_raw or 0)
            except Exception:  # noqa: BLE001
                created_at = 0.0

            if created_at <= 0:
                self._clear_missing_task(task_id)
                continue

            if status == "queued":
                if payload.get("started_at") is not None:
                    self._clear_missing_task(task_id)
                    continue
                if now - created_at < float(self.queue_stale_grace_seconds):
                    self._clear_missing_task(task_id)
                    continue
                if task_id in queued_task_ids:
                    self._clear_missing_task(task_id)
                    continue
                # Reserved/active on worker should not be treated as queue-lost.
                if worker_known_task_ids is None:
                    continue
                if task_id in worker_known_task_ids:
                    self._clear_missing_task(task_id)
                    continue
                if not self._confirm_missing_task(
                    task_id,
                    now=now,
                    confirm_seconds=self.queue_lost_confirm_seconds,
                ):
                    continue

                payload["status"] = "error"
                payload["ended_at"] = now
                payload["exit_code"] = -1
                payload["error"] = (
                    "queued task was lost before execution "
                    "(queue entry not found in job backend)"
                )
                progress = payload.get("progress")
                if not isinstance(progress, dict):
                    progress = {}
                progress.update(
                    {
                        "phase": "queue_lost",
                        "message": "failed",
                        "updated_at": now,
                    }
                )
                payload["progress"] = progress
                self._append_job_log(
                    str(payload.get("id") or p.stem),
                    f"[queue_lost] queue entry was not found in backend="
                    f"{getattr(self.backend, 'name', '?')} after grace period "
                    "(e.g. broker restart or message loss)",
                    ts=now,
                )
                try:
                    self._write_json_atomic(p, payload)
                except Exception:  # noqa: BLE001
                    continue
                self._clear_missing_task(task_id)
                continue

            if status == "running":
                # Without worker liveness info, running-state judgment is unsafe.
                if worker_known_task_ids is None:
                    continue

                progress = payload.get("progress")
                updated_at = None
                if isinstance(progress, dict):
                    raw_updated = progress.get("updated_at")
                    try:
                        updated_at = float(raw_updated)
                    except Exception:  # noqa: BLE001
                        updated_at = None
                started_raw = payload.get("started_at")
                try:
                    started_at = float(started_raw) if started_raw is not None else None
                except Exception:  # noqa: BLE001
                    started_at = None

                last_seen = max(
                    [t for t in (updated_at, started_at, created_at) if isinstance(t, float)],
                    default=created_at,
                )
                if now - last_seen < float(self.running_stale_grace_seconds):
                    self._clear_missing_task(task_id)
                    continue
                if task_id in worker_known_task_ids:
                    self._clear_missing_task(task_id)
                    continue
                if task_id in queued_task_ids:
                    self._clear_missing_task(task_id)
                    continue
                if not self._confirm_missing_task(
                    task_id,
                    now=now,
                    confirm_seconds=self.running_lost_confirm_seconds,
                ):
                    continue

                payload["status"] = "error"
                payload["ended_at"] = now
                payload["exit_code"] = -1
                payload["error"] = (
                    "running task disappeared from worker "
                    "(likely worker restart/crash during execution)"
                )
                if not isinstance(progress, dict):
                    progress = {}
                progress.update(
                    {
                        "phase": "worker_lost",
                        "message": "failed",
                        "updated_at": now,
                    }
                )
                payload["progress"] = progress
                self._append_job_log(
                    str(payload.get("id") or p.stem),
                    "[worker_lost] task disappeared from worker active/reserved/scheduled "
                    "and broker queue after grace period",
                    ts=now,
                )
                try:
                    self._write_json_atomic(p, payload)
                except Exception:  # noqa: BLE001
                    continue
                self._clear_missing_task(task_id)
                continue

            self._clear_missing_task(task_id)

    def _mark_interrupted_jobs(self) -> None:
        now = time.time()
        for p in self.jobs_dir.glob("*.json"):
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "").strip().lower()
            if status not in ("queued", "running"):
                continue

            payload["status"] = "error"
            payload["ended_at"] = now
            payload["exit_code"] = -1
            progress = payload.get("progress")
            if not isinstance(progress, dict):
                progress = {}
            progress.update(
                {
                    "phase": "interrupted",
                    "message": "failed",
                    "updated_at": now,
                }
            )
            payload["progress"] = progress
            try:
                self._write_json_atomic(p, payload)
            except Exception:  # noqa: BLE001
                continue

    def create_job(
        self,
        job_type: str,
        args: Dict,
        *,
        workspace: str,
        experiment: str,
    ) -> JobRecord:
        if job_type not in JOB_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported job type")

        with self.lock:
            job_id = uuid4().hex
            cfg = self.config_manager.get_config()

            try:
                ws = require_workspace_id(workspace)
                exp = require_experiment_id(experiment)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            try:
                ws_root = cfg.workspace_path(ws)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if not ws_root.exists() or not ws_root.is_dir():
                raise HTTPException(
                    status_code=404,
                    detail=f"workspace does not exist. Create it under {cfg.workspaces_root}/{ws} first.",
                )

            # Default args: allow "02_train-like" workflow via HTTP without manual paths.
            if job_type == "apply_label":
                args = dict(args or {})
                labels_path = ensure_label_file(cfg, ws, exp)
                schema_path = cfg.label_schema_path_for(ws, exp)
                args.setdefault("labels", str(labels_path))
                args.setdefault("root", str(cfg.source_path_for(ws)))
                args.setdefault(
                    "dataset_root",
                    str(cfg.experiments_path_for(ws) / exp / "dataset"),
                )
                raw = args.get("raw_args") or args.get("_raw_args")
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--schema(\s|=)", raw)
                ):
                    args.setdefault("schema", str(schema_path))
            elif job_type == "augment_gray":
                args = dict(args or {})
                args.setdefault(
                    "dataset_root",
                    str(cfg.experiments_path_for(ws) / exp / "dataset"),
                )
            elif job_type == "make_label_list":
                args = dict(args or {})
                # In agent workspaces, paths in labels.json are stored relative to source_dir.
                # So both input-dir and html-dir should be the source root.
                args.setdefault("input_dir", str(cfg.source_path_for(ws)))
                args.setdefault("html_dir", str(cfg.source_path_for(ws)))
                args.setdefault("output", str(cfg.label_input_path_for(ws, exp)))
                args.setdefault("thumb_width", 256)
            elif job_type == "purge_deleted_images":
                args = dict(args or {})
                labels_path = ensure_label_file(cfg, ws, exp)
                args.setdefault("labels", str(labels_path))
                args.setdefault("root", str(cfg.source_path_for(ws)))
            elif job_type == "embed_images":
                args = dict(args or {})
                labels_path = ensure_label_file(cfg, ws, exp)
                args.setdefault("labels", str(labels_path))
                args.setdefault("root", str(cfg.source_path_for(ws)))
                args.setdefault(
                    "cache_root",
                    str(cfg.experiments_path_for(ws) / exp / "cache"),
                )
            elif job_type in ("train", "train_epoch"):
                args = dict(args or {})
                dataset_index = (
                    cfg.experiments_path_for(ws) / exp / "dataset" / "index.json"
                )
                schema_path = cfg.label_schema_path_for(ws, exp)
                raw = args.get("raw_args") or args.get("_raw_args")
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--index(\s|=)", raw)
                ):
                    args.setdefault("index", str(dataset_index))
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--schema(\s|=)", raw)
                ):
                    args.setdefault("schema", str(schema_path))
            elif job_type == "predict_labels":
                # 予測の候補記録（T2-2）。infer_heads と違い入力は dataset/index.json
                # ではなく labels.json であり、出力も JSON ではなく labels.json の
                # item["predicted"] への記録になる。labels（確定値）は変えない。
                args = dict(args or {})
                exp_dir = cfg.experiments_path_for(ws) / exp
                models_dir = exp_dir / "models"
                runs = (
                    [p for p in models_dir.glob("run_*") if p.is_dir()]
                    if models_dir.is_dir()
                    else []
                )
                if not runs:
                    raise HTTPException(
                        status_code=404,
                        detail=f"no model runs found under {models_dir}. Run train_epoch first.",
                    )
                runs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                args.setdefault("labels", str(ensure_label_file(cfg, ws, exp)))
                args.setdefault("run_dir", str(runs[0]))
                args.setdefault("cache_root", str(exp_dir / "cache"))
                args.setdefault("schema", str(cfg.label_schema_path_for(ws, exp)))
            elif job_type in ("infer_scores", "infer_heads"):
                args = dict(args or {})
                exp_dir = cfg.experiments_path_for(ws) / exp
                models_dir = exp_dir / "models"
                schema_path = cfg.label_schema_path_for(ws, exp)
                # pick latest run_* directory
                runs: List[Path] = []
                if models_dir.exists() and models_dir.is_dir():
                    runs = [p for p in models_dir.glob("run_*") if p.is_dir()]
                if not runs:
                    raise HTTPException(
                        status_code=404,
                        detail=f"no model runs found under {models_dir}. Run train_epoch first.",
                    )
                runs.sort(
                    key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True
                )
                latest_run = runs[0]

                dataset_index = exp_dir / "dataset" / "index.json"
                default_output = exp_dir / "eval" / f"scores_{latest_run.name}.json"

                raw = args.get("raw_args") or args.get("_raw_args")
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--run-dir(\s|=)", raw)
                ):
                    args.setdefault("run_dir", str(latest_run))
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--index(\s|=)", raw)
                ):
                    args.setdefault("index", str(dataset_index))
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--output(\s|=)", raw)
                ):
                    args.setdefault("output", str(default_output))
                if not (
                    isinstance(raw, str) and re.search(r"(^|\s)--schema(\s|=)", raw)
                ):
                    args.setdefault("schema", str(schema_path))
            try:
                command = build_command(job_type, args, cfg)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            now = time.time()
            # Pre-assign the backend task id before enqueue to avoid a race where
            # worker-side "running" status gets overwritten by a late queued save.
            worker_task_id = str(uuid4())
            job = JobRecord(
                id=job_id,
                type=job_type,
                workspace=ws,
                experiment=exp,
                args=args,
                status="queued",
                progress={
                    "phase": "queued",
                    "message": "queued",
                    "updated_at": now,
                },
                created_at=now,
                started_at=None,
                ended_at=None,
                exit_code=None,
                command=command,
                worker_task_id=worker_task_id,
            )
            self._save_job(job)

        try:
            task_id = str(
                self.backend.enqueue(
                    job_id=job.id,
                    command=job.command,
                    workspace=job.workspace,
                    task_id=str(job.worker_task_id or ""),
                )
                or ""
            ).strip()
            if task_id and task_id != str(job.worker_task_id or "").strip():
                # In normal flow this should not happen because task_id is pre-assigned.
                # If the backend returned a different id, patch only task id while
                # preserving whatever status/progress may already be written by worker.
                latest = self._load_job(job.id)
                latest.worker_task_id = task_id
                self._save_job(latest)
                job = latest
        except Exception as exc:  # noqa: BLE001
            failed_at = time.time()
            job.status = "error"
            job.ended_at = failed_at
            job.exit_code = -1
            job.error = f"failed to enqueue job: {exc}"
            job.progress = {
                "phase": "enqueue",
                "message": f"failed to enqueue job: {exc}",
                "updated_at": failed_at,
            }
            self._save_job(job)
            self._append_job_log(
                job.id,
                f"[enqueue_failed] failed to enqueue via backend="
                f"{getattr(self.backend, 'name', '?')}: {exc}",
                ts=failed_at,
            )
            raise HTTPException(status_code=503, detail="failed to enqueue job")

        return job

    def cancel_job(self, job_id: str) -> JobRecord:
        self._reconcile_stale_jobs()
        try:
            job = self._load_job(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

        status = str(job.status or "").strip().lower()
        if status not in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"job is not cancelable in status={job.status}",
            )

        now = time.time()
        payload = job.to_dict()
        task_id = str(payload.get("worker_task_id") or "").strip()

        should_terminate = status == "running" or payload.get("started_at") is not None
        if task_id:
            try:
                self.backend.revoke(task_id, terminate=should_terminate)
                self._append_job_log(
                    job.id,
                    f"[cancel] revoke sent (task_id={task_id}, terminate={should_terminate})",
                    ts=now,
                )
            except Exception as exc:  # noqa: BLE001
                self._append_job_log(
                    job.id,
                    f"[cancel] failed to send revoke (task_id={task_id}): {exc}",
                    ts=now,
                )

        payload["status"] = "canceled"
        payload["ended_at"] = now
        payload["exit_code"] = -1
        payload["error"] = "canceled by user"
        progress = payload.get("progress")
        if not isinstance(progress, dict):
            progress = {}
        progress.update(
            {
                "phase": "canceled",
                "message": "canceled by user",
                "updated_at": now,
            }
        )
        payload["progress"] = progress
        self._write_json_atomic(self._job_paths(job.id)["json"], payload)
        self._append_job_log(job.id, "[canceled] canceled by user request", ts=now)

        return JobRecord(**payload)

    def rerun_job(self, job_id: str) -> JobRecord:
        self._reconcile_stale_jobs()
        try:
            source = self._load_job(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

        if not self._is_terminal_status(source.status):
            raise HTTPException(
                status_code=409,
                detail=f"job is not rerunnable in status={source.status}",
            )
        if not source.workspace or not source.experiment:
            raise HTTPException(
                status_code=400,
                detail="job cannot be rerun because workspace/experiment is missing",
            )

        args = dict(source.args or {})
        rerun = self.create_job(
            source.type,
            args,
            workspace=str(source.workspace),
            experiment=str(source.experiment),
        )
        self._append_job_log(
            source.id,
            f"[rerun] requeued as job_id={rerun.id}",
            ts=time.time(),
        )
        return rerun

    def get_job(self, job_id: str) -> JobRecord:
        self._reconcile_stale_jobs()
        try:
            return self._load_job(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

    def list_jobs(
        self,
        *,
        workspace: str,
        experiment: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[JobRecord]:
        return self.list_jobs_filtered(
            workspace=workspace,
            experiment=experiment,
            limit=limit,
            offset=offset,
        )

    def list_jobs_filtered(
        self,
        *,
        workspace: Optional[str] = None,
        experiment: Optional[str] = None,
        q: Optional[str] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[JobRecord]:
        self._reconcile_stale_jobs()

        def norm(v: Optional[str]) -> str:
            return (v or "").strip().lower()

        qn = norm(q)
        wn = norm(workspace)
        en = norm(experiment)
        sn = norm(status)
        tn = norm(job_type)

        jobs: List[JobRecord] = []

        for p in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                job = JobRecord(**data)
            except Exception:  # noqa: BLE001
                continue

            if wn and norm(job.workspace) != wn:
                continue
            if en and norm(job.experiment) != en:
                continue
            if sn and norm(job.status) != sn:
                continue
            if tn and norm(job.type) != tn:
                continue

            if qn:
                hay = " ".join(
                    [
                        str(job.id or ""),
                        str(job.type or ""),
                        str(job.status or ""),
                        str(job.workspace or ""),
                        str(job.experiment or ""),
                        str((job.progress or {}).get("message") or ""),
                    ]
                ).lower()
                if qn not in hay:
                    continue

            jobs.append(job)

        jobs.sort(key=lambda j: float(j.created_at or 0), reverse=True)
        if offset < 0:
            offset = 0
        if limit <= 0:
            return []
        return jobs[offset : offset + limit]

    def delete_jobs_any(self, *, job_ids: List[str]) -> Dict[str, List[str]]:
        self._reconcile_stale_jobs()

        deleted: List[str] = []
        not_found: List[str] = []
        skipped_running: List[str] = []

        for job_id in job_ids:
            try:
                job = self._load_job(job_id)
            except FileNotFoundError:
                not_found.append(job_id)
                continue
            except Exception:  # noqa: BLE001
                not_found.append(job_id)
                continue

            if self._is_active_status(job.status):
                skipped_running.append(job_id)
                continue

            paths = self._job_paths(job_id)
            try:
                if paths["json"].exists():
                    paths["json"].unlink()
                if paths["log"].exists():
                    paths["log"].unlink()
                deleted.append(job_id)
            except Exception:  # noqa: BLE001
                not_found.append(job_id)

        return {
            "deleted": deleted,
            "skipped_running": skipped_running,
            "not_found": not_found,
        }

    def delete_jobs(
        self,
        *,
        workspace: str,
        experiment: str,
        job_ids: List[str],
    ) -> Dict[str, List[str]]:
        self._reconcile_stale_jobs()

        deleted: List[str] = []
        not_found: List[str] = []
        skipped_running: List[str] = []

        for job_id in job_ids:
            try:
                job = self._load_job(job_id)
            except FileNotFoundError:
                not_found.append(job_id)
                continue
            except Exception:  # noqa: BLE001
                not_found.append(job_id)
                continue

            if job.workspace and job.workspace != workspace:
                not_found.append(job_id)
                continue
            if job.experiment and job.experiment != experiment:
                not_found.append(job_id)
                continue
            if self._is_active_status(job.status):
                skipped_running.append(job_id)
                continue

            paths = self._job_paths(job_id)
            try:
                if paths["json"].exists():
                    paths["json"].unlink()
                if paths["log"].exists():
                    paths["log"].unlink()
                deleted.append(job_id)
            except Exception:  # noqa: BLE001
                # If deletion fails, report as not_found to avoid leaking details.
                not_found.append(job_id)

        return {
            "deleted": deleted,
            "skipped_running": skipped_running,
            "not_found": not_found,
        }

    def tail_logs(self, job_id: str, tail: int) -> str:
        paths = self._job_paths(job_id)
        if not paths["log"].exists():
            raise HTTPException(status_code=404, detail="log not found")
        if tail <= 0:
            return paths["log"].read_text()
        lines: Deque[str] = deque(maxlen=tail)
        with open(paths["log"], "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                lines.append(line)
        return "".join(lines)


def create_jobs_router(manager: JobsManager) -> APIRouter:
    router = APIRouter()

    # ---- Global API ----
    global_router = APIRouter(prefix="/jobs")

    @global_router.get("")
    def list_jobs_global(
        q: Optional[str] = Query(None, max_length=200),
        workspace: Optional[str] = Query(None),
        experiment: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        job_type: Optional[str] = Query(None, alias="type"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0, le=100000),
    ) -> dict:
        ws_id: Optional[str] = None
        exp_id: Optional[str] = None
        try:
            if workspace is not None and str(workspace).strip() != "":
                ws_id = require_workspace_id(workspace)
            if experiment is not None and str(experiment).strip() != "":
                exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        jobs = manager.list_jobs_filtered(
            workspace=ws_id,
            experiment=exp_id,
            q=q,
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )
        return {"jobs": [j.to_dict() for j in jobs]}

    @global_router.get("/{job_id}")
    def get_job_global(job_id: str) -> dict:
        job = manager.get_job(job_id)
        return job.to_dict()

    @global_router.get("/{job_id}/logs")
    def get_logs_global(
        job_id: str,
        tail: int = Query(5000, ge=1, le=20000),
    ) -> str:
        _ = manager.get_job(job_id)
        return manager.tail_logs(job_id, tail)

    @global_router.post("/{job_id}/cancel")
    def cancel_job_global(job_id: str) -> dict:
        job = manager.cancel_job(job_id)
        return {"job_id": job.id, "status": job.status}

    @global_router.post("/{job_id}/rerun")
    def rerun_job_global(job_id: str) -> dict:
        job = manager.rerun_job(job_id)
        return {"job_id": job.id, "status": job.status}

    @global_router.delete("")
    def delete_jobs_global(req: JobDeleteRequest = Body(...)) -> dict:
        return manager.delete_jobs_any(job_ids=req.job_ids)

    router.include_router(global_router)

    # ---- Scoped API ----

    scoped = APIRouter(prefix="/workspaces/{workspace}/experiments/{experiment}/jobs")

    @scoped.get("")
    def list_jobs_scoped(
        workspace: str,
        experiment: str,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0, le=100000),
    ) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        jobs = manager.list_jobs(
            workspace=ws_id,
            experiment=exp_id,
            limit=limit,
            offset=offset,
        )
        return {"jobs": [j.to_dict() for j in jobs]}

    @scoped.post("")
    def create_job_scoped(
        workspace: str, experiment: str, req: JobCreateRequest
    ) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job = manager.create_job(req.type, req.args, workspace=ws_id, experiment=exp_id)
        return {"job_id": job.id, "status": job.status}

    @scoped.delete("")
    def delete_jobs_scoped(
        workspace: str,
        experiment: str,
        req: JobDeleteRequest = Body(...),
    ) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        result = manager.delete_jobs(
            workspace=ws_id,
            experiment=exp_id,
            job_ids=req.job_ids,
        )
        return result

    @scoped.get("/{job_id}")
    def get_job_scoped(workspace: str, experiment: str, job_id: str) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job = manager.get_job(job_id)
        if job.workspace and job.workspace != ws_id:
            raise HTTPException(status_code=404, detail="job not found")
        if job.experiment and job.experiment != exp_id:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @scoped.get("/{job_id}/logs")
    def get_logs_scoped(
        workspace: str,
        experiment: str,
        job_id: str,
        tail: int = Query(5000, ge=1, le=20000),
    ) -> str:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job = manager.get_job(job_id)
        if job.workspace and job.workspace != ws_id:
            raise HTTPException(status_code=404, detail="job not found")
        if job.experiment and job.experiment != exp_id:
            raise HTTPException(status_code=404, detail="job not found")
        return manager.tail_logs(job_id, tail)

    @scoped.post("/{job_id}/cancel")
    def cancel_job_scoped(workspace: str, experiment: str, job_id: str) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        job = manager.get_job(job_id)
        if job.workspace and job.workspace != ws_id:
            raise HTTPException(status_code=404, detail="job not found")
        if job.experiment and job.experiment != exp_id:
            raise HTTPException(status_code=404, detail="job not found")
        out = manager.cancel_job(job_id)
        return {"job_id": out.id, "status": out.status}

    @scoped.post("/{job_id}/rerun")
    def rerun_job_scoped(workspace: str, experiment: str, job_id: str) -> dict:
        try:
            ws_id = require_workspace_id(workspace)
            exp_id = require_experiment_id(experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        job = manager.get_job(job_id)
        if job.workspace and job.workspace != ws_id:
            raise HTTPException(status_code=404, detail="job not found")
        if job.experiment and job.experiment != exp_id:
            raise HTTPException(status_code=404, detail="job not found")
        out = manager.rerun_job(job_id)
        return {"job_id": out.id, "status": out.status}

    router.include_router(scoped)

    return router
