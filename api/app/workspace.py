from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .config import ConfigManager
from .utils.atomic_io import write_json_atomic
from .utils.identity import ensure_identity
from .utils.meta import ExperimentMeta, WorkspaceMeta
from .utils.paths import is_subpath, require_dir
from .utils.short_id import (
    generate_short_id,
    is_valid_short_id,
    require_experiment_id,
    require_short_id,
    require_workspace_id,
)
from .utils.sidebar_order import (
    apply_order_with_fallback,
    load_sidebar_order,
    save_sidebar_order,
)


_BACKUP_ID_RE = re.compile(r"^\d{8}_\d{6}(?:-\d+)?$")
_WORKSPACE_BACKUP_EXT = ".tar.zst"
_WORKSPACE_BACKUP_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


class CreateWorkspaceBody(BaseModel):
    display_name: str
    id: str | None = None


class RenameWorkspaceBody(BaseModel):
    display_name: str


class SidebarOrderBody(BaseModel):
    # Order of workspace ids in sidebar.
    workspaces: list[str] | None = None
    # Order of experiment ids per workspace id in sidebar.
    experiments: dict[str, list[str]] | None = None


def _cleanup_dir(path_str: str) -> None:
    try:
        shutil.rmtree(path_str, ignore_errors=True)
    except Exception:
        pass


def _mark_interrupted_workspace_jobs(job_dir: Path) -> None:
    if not job_dir.exists() or not job_dir.is_dir():
        return
    now = time.time()
    for p in job_dir.glob("*.json"):
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").lower()
        if status not in ("queued", "running"):
            continue
        payload["status"] = "error"
        payload["ended_at"] = now
        payload["progress"] = {"percent": 100, "message": "failed"}
        if not str(payload.get("error") or "").strip():
            payload["error"] = (
                "interrupted by restart; temporary backup/restore files were cleaned up"
            )
        try:
            write_json_atomic(p, payload)
        except Exception:
            continue


def _ensure_trash_root(ws_root) -> None:
    trash_root = (ws_root / ".trash").resolve()
    trash_root.mkdir(parents=True, exist_ok=True)


def _generate_unique_trash_id(trash_root) -> str:
    for _ in range(50):
        candidate = uuid.uuid4().hex
        if not (trash_root / candidate).exists():
            return candidate
    raise RuntimeError("failed to generate unique trash id")


def _trash_move(src, dest) -> None:
    # Prefer atomic rename when possible; fall back to shutil.move.
    try:
        src.replace(dest)
    except OSError:
        shutil.move(str(src), str(dest))


def _write_json(path, data: dict) -> None:
    write_json_atomic(path, data)


def _dir_size_bytes(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _count_files(root: Path) -> int:
    count = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            count += 1
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return count


def _reserved_workspace_ids_in_trash(trash_root: Path) -> set[str]:
    """Collect workspace ids that are currently present in trash metadata.

    Rationale:
    - Prevent creating a new workspace with the same id as a trashed workspace.
      That avoids restore conflicts and makes IDs stable over time.
    """

    reserved: set[str] = set()
    if not trash_root.exists() or not trash_root.is_dir():
        return reserved

    for p in trash_root.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        if not meta_path.exists() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict) or meta.get("kind") != "workspace":
            continue
        ws_id = (str(meta.get("workspace") or "")).strip().lower()
        if is_valid_short_id(ws_id):
            reserved.add(ws_id)

    return reserved


def _normalize_archive_member_name(name: str) -> Path:
    cleaned = (name or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("archive contains empty entry name")
    if cleaned.startswith("/"):
        raise ValueError(f"archive entry is absolute path: {name}")
    parts = Path(cleaned).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"archive entry has invalid path: {name}")
    return Path(*parts)


def _read_workspace_meta_from_archive(archive_path: Path) -> dict[str, Any]:
    cmd = [
        "tar",
        "--use-compress-program",
        "zstd -dc",
        "-xOf",
        str(archive_path),
        "workspace.json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise ValueError("tar or zstd is not installed on the agent")

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "workspace.json not found"
        raise ValueError(f"invalid backup archive: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"workspace.json is not valid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        raise ValueError("workspace.json must be a JSON object")

    raw_workspace_id = str(
        payload.get("workspace_id") or payload.get("id") or ""
    ).strip()
    if not raw_workspace_id:
        raise ValueError("workspace.json does not contain workspace_id")

    try:
        workspace_id = require_workspace_id(raw_workspace_id)
    except ValueError as exc:
        raise ValueError(str(exc))

    return {"workspace_id": workspace_id, "workspace_json": payload}


def _extract_archive_into_dir(archive_path: Path, staging_dir: Path) -> None:
    try:
        zstd_proc = subprocess.Popen(
            ["zstd", "-dc", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ValueError("zstd is not installed on the agent")

    try:
        if zstd_proc.stdout is None:
            raise ValueError("failed to read backup archive stream")
        with tarfile.open(fileobj=zstd_proc.stdout, mode="r|") as tar:
            for member in tar:
                relative = _normalize_archive_member_name(member.name)
                destination = (staging_dir / relative).resolve()
                if not is_subpath(staging_dir, destination):
                    raise ValueError(f"archive path escapes staging dir: {member.name}")

                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(
                        f"unsupported archive entry type for restore: {member.name}"
                    )

                if not member.isfile():
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    raise ValueError(f"failed to extract file: {member.name}")
                with src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except tarfile.TarError as exc:
        zstd_proc.kill()
        raise ValueError(f"invalid tar archive: {exc}") from exc
    except Exception:
        zstd_proc.kill()
        raise
    finally:
        if zstd_proc.stdout is not None:
            zstd_proc.stdout.close()

    stderr = b""
    if zstd_proc.stderr is not None:
        stderr = zstd_proc.stderr.read() or b""
        zstd_proc.stderr.close()
    rc = zstd_proc.wait()
    if rc != 0:
        detail = stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(detail or "failed to decompress backup archive")


def _create_workspace_backup_archive(
    *,
    workspace_root: Path,
    archive_path: Path,
    members: list[str],
) -> None:
    cmd = [
        "tar",
        "--use-compress-program",
        "zstd -T0 -19",
        "-C",
        str(workspace_root),
        "-cf",
        str(archive_path),
        *members,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise ValueError("tar or zstd is not installed on the agent")

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "backup command failed"
        raise ValueError(detail)


def create_workspace_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()
    cfg = config_manager.get_config()
    ws_root = cfg.workspaces_root
    ws_root.mkdir(parents=True, exist_ok=True)
    _ensure_trash_root(ws_root)

    xima_root = (ws_root / ".xima").resolve()
    if not is_subpath(ws_root, xima_root):
        raise RuntimeError("xima root escapes workspaces root")
    jobs_root = (xima_root / "jobs").resolve()
    if not is_subpath(xima_root, jobs_root):
        raise RuntimeError("jobs root escapes .xima")
    backup_jobs_dir = (jobs_root / "workspace_backup").resolve()
    if not is_subpath(jobs_root, backup_jobs_dir):
        raise RuntimeError("backup jobs dir escapes jobs root")
    restore_jobs_dir = (jobs_root / "workspace_restore").resolve()
    if not is_subpath(jobs_root, restore_jobs_dir):
        raise RuntimeError("restore jobs dir escapes jobs root")

    tmp_root = (ws_root / ".tmp").resolve()
    if not is_subpath(ws_root, tmp_root):
        raise RuntimeError("tmp root escapes workspaces root")
    backup_root = (tmp_root / "backups").resolve()
    restore_tmp_root = (tmp_root / "restore").resolve()
    backup_artifacts_root = (backup_root / "artifacts").resolve()
    backup_work_root = backup_root
    restore_uploads_dir = (restore_tmp_root / "uploads").resolve()

    # Keep jobs metadata, but always cleanup temporary runtime files at startup.
    _cleanup_dir(str(tmp_root))

    legacy_backup_root = (ws_root / ".workspace_backups").resolve()
    if is_subpath(ws_root, legacy_backup_root):
        _cleanup_dir(str(legacy_backup_root))

    xima_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    backup_jobs_dir.mkdir(parents=True, exist_ok=True)
    restore_jobs_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    restore_tmp_root.mkdir(parents=True, exist_ok=True)
    backup_artifacts_root.mkdir(parents=True, exist_ok=True)
    restore_uploads_dir.mkdir(parents=True, exist_ok=True)
    _mark_interrupted_workspace_jobs(backup_jobs_dir)
    _mark_interrupted_workspace_jobs(restore_jobs_dir)

    backup_lock = threading.Lock()
    running_backup_by_workspace: dict[str, str] = {}
    restore_lock = threading.Lock()
    running_restore_by_workspace: dict[str, str] = {}

    def _purge_expired_backups(*, max_age_seconds: float) -> None:
        now = time.time()
        cutoff = now - max_age_seconds
        protected_ids = set(running_backup_by_workspace.values())

        for p in backup_artifacts_root.rglob(f"*{_WORKSPACE_BACKUP_EXT}"):
            if not p.is_file():
                continue
            job_id = p.stem.lower()
            if job_id in protected_ids:
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
            except OSError:
                continue

        for p in sorted(backup_artifacts_root.rglob("*"), reverse=True):
            if not p.is_dir():
                continue
            try:
                next(p.iterdir())
            except StopIteration:
                try:
                    p.rmdir()
                except OSError:
                    pass
            except OSError:
                pass

    def _workspace_backup_job_path(job_id: str) -> Path:
        jid = (job_id or "").strip().lower()
        if not _WORKSPACE_BACKUP_JOB_ID_RE.match(jid):
            raise HTTPException(status_code=400, detail="invalid backup job id")
        p = (backup_jobs_dir / f"{jid}.json").resolve()
        if not is_subpath(backup_jobs_dir, p):
            raise HTTPException(status_code=400, detail="invalid backup job id")
        return p

    def _save_workspace_backup_job(job: dict[str, Any]) -> None:
        # ジョブ実行中は進捗のたびにこのファイルを上書きする一方、状態エンドポイントは
        # ポーリングされる前提で設計されている。非アトミックに書くと、読み手が
        # 切り詰められた JSON を掴んで 500 を返す（#197）。
        job_id = str(job.get("id") or "").strip().lower()
        if not _WORKSPACE_BACKUP_JOB_ID_RE.match(job_id):
            raise ValueError("invalid backup job id")
        path = _workspace_backup_job_path(job_id)
        write_json_atomic(path, job)

    def _load_workspace_backup_job(job_id: str) -> dict[str, Any]:
        path = _workspace_backup_job_path(job_id)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="backup job not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="backup job file is invalid")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail="backup job file is invalid")
        return payload

    def _list_workspace_backup_jobs(
        workspace: str, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        ws_id = require_workspace_id(workspace)
        jobs: list[dict[str, Any]] = []
        for p in sorted(backup_jobs_dir.glob("*.json")):
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("type") or "") != "workspace_backup":
                continue
            if str(payload.get("workspace") or "") != ws_id:
                continue
            jobs.append(payload)

        jobs.sort(key=lambda x: float(x.get("created_at") or 0.0), reverse=True)
        if offset < 0:
            offset = 0
        if limit <= 0:
            return []
        return jobs[offset : offset + limit]

    def _run_workspace_backup_job(job_id: str, workspace: str) -> None:
        ws_id = workspace
        tmp_dir: Path | None = None
        try:
            job = _load_workspace_backup_job(job_id)
            job["status"] = "running"
            job["started_at"] = time.time()
            job["progress"] = {"percent": 10, "message": "preparing files"}
            _save_workspace_backup_job(job)

            cfg_local = config_manager.get_config()
            root = (cfg_local.workspaces_root / ws_id).resolve()
            if not is_subpath(cfg_local.workspaces_root, root):
                raise ValueError("invalid workspace")
            if not root.exists() or not root.is_dir():
                raise ValueError("workspace not found")

            workspace_meta_path = WorkspaceMeta.path_for(root)
            if not workspace_meta_path.exists():
                meta = WorkspaceMeta.load(root, workspace_id=ws_id)
                meta.save(root)

            members: list[str] = ["workspace.json"]
            if (root / cfg_local.source_dir).exists():
                members.append(cfg_local.source_dir)
            if (root / cfg_local.experiments_dir).exists():
                members.append(cfg_local.experiments_dir)

            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            filename = f"workspace_{ws_id}_{ts}{_WORKSPACE_BACKUP_EXT}"
            ws_artifacts_dir = (backup_artifacts_root / ws_id).resolve()
            if not is_subpath(backup_artifacts_root, ws_artifacts_dir):
                raise ValueError("invalid artifacts dir")
            ws_artifacts_dir.mkdir(parents=True, exist_ok=True)

            tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"backup-{job_id}-",
                    dir=str(backup_work_root),
                )
            ).resolve()
            if not is_subpath(backup_work_root, tmp_dir):
                raise ValueError("invalid tmp dir")
            archive_tmp_path = tmp_dir / filename

            job["progress"] = {"percent": 50, "message": "creating archive"}
            _save_workspace_backup_job(job)

            _create_workspace_backup_archive(
                workspace_root=root,
                archive_path=archive_tmp_path,
                members=members,
            )

            artifact_path = (ws_artifacts_dir / f"{job_id}{_WORKSPACE_BACKUP_EXT}").resolve()
            if not is_subpath(backup_artifacts_root, artifact_path):
                raise ValueError("invalid artifact path")
            shutil.move(str(archive_tmp_path), str(artifact_path))

            size_bytes = 0
            try:
                size_bytes = int(artifact_path.stat().st_size)
            except OSError:
                size_bytes = 0

            job["status"] = "done"
            job["ended_at"] = time.time()
            job["progress"] = {"percent": 100, "message": "completed"}
            job["filename"] = filename
            job["size_bytes"] = size_bytes
            job["artifact_path"] = str(artifact_path)
            job["error"] = None
            _save_workspace_backup_job(job)
        except Exception as exc:
            try:
                job = _load_workspace_backup_job(job_id)
                job["status"] = "error"
                job["ended_at"] = time.time()
                job["progress"] = {"percent": 100, "message": "failed"}
                job["error"] = str(exc)
                _save_workspace_backup_job(job)
            except Exception:
                pass
        finally:
            if tmp_dir is not None:
                _cleanup_dir(str(tmp_dir))
            with backup_lock:
                if running_backup_by_workspace.get(ws_id) == job_id:
                    running_backup_by_workspace.pop(ws_id, None)

    def _workspace_restore_job_path(job_id: str) -> Path:
        jid = (job_id or "").strip().lower()
        if not _WORKSPACE_BACKUP_JOB_ID_RE.match(jid):
            raise HTTPException(status_code=400, detail="invalid restore job id")
        p = (restore_jobs_dir / f"{jid}.json").resolve()
        if not is_subpath(restore_jobs_dir, p):
            raise HTTPException(status_code=400, detail="invalid restore job id")
        return p

    def _workspace_restore_upload_path(job_id: str) -> Path:
        jid = (job_id or "").strip().lower()
        if not _WORKSPACE_BACKUP_JOB_ID_RE.match(jid):
            raise HTTPException(status_code=400, detail="invalid restore job id")
        p = (restore_uploads_dir / f"{jid}{_WORKSPACE_BACKUP_EXT}").resolve()
        if not is_subpath(restore_uploads_dir, p):
            raise HTTPException(status_code=400, detail="invalid restore job id")
        return p

    def _save_workspace_restore_job(job: dict[str, Any]) -> None:
        # backup 側と同じ理由でアトミックに書く（#197）。復元はユーザ資産の回復経路であり、
        # 進行中に理由のない 500 を返すと、利用者は失敗したと解釈しうる。
        job_id = str(job.get("id") or "").strip().lower()
        path = _workspace_restore_job_path(job_id)
        write_json_atomic(path, job)

    def _load_workspace_restore_job(job_id: str) -> dict[str, Any]:
        path = _workspace_restore_job_path(job_id)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="restore job not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="restore job file is invalid")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail="restore job file is invalid")
        return payload

    def _run_workspace_restore_job(job_id: str, workspace: str) -> None:
        ws_id = workspace
        staging_dir: Path | None = None
        try:
            job = _load_workspace_restore_job(job_id)
            job["status"] = "running"
            job["started_at"] = time.time()
            job["progress"] = {"percent": 10, "message": "extracting backup archive"}
            _save_workspace_restore_job(job)

            cfg_local = config_manager.get_config()
            upload_path = _workspace_restore_upload_path(job_id)
            if not upload_path.exists() or not upload_path.is_file():
                raise ValueError("restore upload is missing")

            target_root = (cfg_local.workspaces_root / ws_id).resolve()
            if not is_subpath(cfg_local.workspaces_root, target_root):
                raise ValueError("invalid workspace id")
            if target_root.exists():
                raise ValueError("workspace id already exists")

            staging_dir = Path(
                tempfile.mkdtemp(prefix=f"restore-{job_id}-", dir=str(restore_tmp_root))
            ).resolve()
            if not is_subpath(restore_tmp_root, staging_dir):
                raise ValueError("invalid restore staging path")

            _extract_archive_into_dir(upload_path, staging_dir)

            restored_meta_path = WorkspaceMeta.path_for(staging_dir)
            if not restored_meta_path.exists() or not restored_meta_path.is_file():
                raise ValueError("workspace.json not found at archive root")
            try:
                restored_meta = json.loads(restored_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raise ValueError("workspace.json is invalid")

            restored_id = str(
                restored_meta.get("workspace_id") or restored_meta.get("id") or ""
            ).strip()
            restored_id = require_workspace_id(restored_id)
            if restored_id != ws_id:
                raise ValueError("workspace id mismatch in backup archive")

            job["progress"] = {"percent": 75, "message": "applying workspace restore"}
            _save_workspace_restore_job(job)

            try:
                target_root.mkdir(parents=False, exist_ok=False)
                for child in staging_dir.iterdir():
                    shutil.move(str(child), str(target_root / child.name))
            except Exception as exc:
                _cleanup_dir(str(target_root))
                raise ValueError(f"failed to finalize workspace restore: {exc}")

            (target_root / cfg_local.source_dir).mkdir(parents=True, exist_ok=True)
            (target_root / cfg_local.experiments_dir).mkdir(parents=True, exist_ok=True)

            job["status"] = "done"
            job["ended_at"] = time.time()
            job["progress"] = {"percent": 100, "message": "completed"}
            job["error"] = None
            _save_workspace_restore_job(job)
        except Exception as exc:
            try:
                job = _load_workspace_restore_job(job_id)
                job["status"] = "error"
                job["ended_at"] = time.time()
                job["progress"] = {"percent": 100, "message": "failed"}
                job["error"] = str(exc)
                _save_workspace_restore_job(job)
            except Exception:
                pass
        finally:
            if staging_dir is not None:
                _cleanup_dir(str(staging_dir))
            upload_path = restore_uploads_dir / f"{job_id}{_WORKSPACE_BACKUP_EXT}"
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            with restore_lock:
                if running_restore_by_workspace.get(ws_id) == job_id:
                    running_restore_by_workspace.pop(ws_id, None)

    # ---- Scoped (recommended) API ----

    scoped = APIRouter(prefix="/workspaces")

    @scoped.get("")
    def list_workspaces(order: str | None = None) -> dict:
        items: list[dict] = []
        if ws_root.exists():
            ws_paths: list[Path] = []
            for p in sorted(ws_root.iterdir()):
                if not p.is_dir():
                    continue
                try:
                    require_workspace_id(p.name)
                except ValueError:
                    # Silently ignore non-managed directories.
                    continue
                ws_paths.append(p)

            if order == "sidebar":
                sidebar_order = load_sidebar_order(ws_root)
                ids_sorted = [require_workspace_id(p.name) for p in ws_paths]
                wanted = (
                    sidebar_order.get("workspaces")
                    if isinstance(sidebar_order.get("workspaces"), list)
                    else []
                )
                ws_id_order = apply_order_with_fallback(ids_sorted, wanted)
                ws_by_id = {require_workspace_id(p.name): p for p in ws_paths}
                ws_paths = [ws_by_id[x] for x in ws_id_order if x in ws_by_id]

            for p in ws_paths:
                ws_id = require_workspace_id(p.name)
                try:
                    meta = WorkspaceMeta.load(p, workspace_id=ws_id)
                except Exception:
                    meta = WorkspaceMeta.load(p, workspace_id=ws_id)

                exp_root = cfg.experiments_path_for(ws_id)
                exp_count = 0
                if exp_root.exists() and exp_root.is_dir():
                    for exp_path in exp_root.iterdir():
                        if not exp_path.is_dir():
                            continue
                        try:
                            require_experiment_id(exp_path.name)
                        except ValueError:
                            continue
                        exp_count += 1

                source_root = p / cfg.source_dir
                source_images_count = 0
                if source_root.exists() and source_root.is_dir():
                    source_images_count = _count_files(source_root)

                size_bytes = _dir_size_bytes(p)
                items.append(
                    {
                        "id": ws_id,
                        "display_name": meta.display_name,
                        "workspace_root": str(p.resolve()),
                        "experiments_count": exp_count,
                        "source_images_count": source_images_count,
                        "size_bytes": size_bytes,
                        "created_at": meta.created_at,
                        "updated_at": meta.updated_at,
                    }
                )
        return {"workspaces": items}

    @scoped.get("/sidebar-order")
    def get_sidebar_order() -> dict:
        return load_sidebar_order(ws_root)

    @scoped.put("/sidebar-order")
    def put_sidebar_order(body: SidebarOrderBody) -> dict:
        # Last write wins. We still validate ids and drop invalid values.
        existing = load_sidebar_order(ws_root)
        next_order = {
            "workspaces": existing.get("workspaces") or [],
            "experiments": existing.get("experiments") or {},
        }

        if body.workspaces is not None:
            cleaned: list[str] = []
            seen: set[str] = set()
            for raw in body.workspaces:
                if not isinstance(raw, str):
                    continue
                v = raw.strip().lower()
                if not v or v in seen:
                    continue
                try:
                    v = require_workspace_id(v)
                except ValueError:
                    continue
                cleaned.append(v)
                seen.add(v)
            next_order["workspaces"] = cleaned

        if body.experiments is not None:
            exp_map: dict[str, list[str]] = {}
            for ws_raw, exp_list in body.experiments.items():
                if not isinstance(ws_raw, str) or not isinstance(exp_list, list):
                    continue
                ws_id = ws_raw.strip().lower()
                try:
                    ws_id = require_workspace_id(ws_id)
                except ValueError:
                    continue

                cleaned_exps: list[str] = []
                seen_exps: set[str] = set()
                for exp_raw in exp_list:
                    if not isinstance(exp_raw, str):
                        continue
                    exp_id = exp_raw.strip().lower()
                    if not exp_id or exp_id in seen_exps:
                        continue
                    try:
                        exp_id = require_experiment_id(exp_id)
                    except ValueError:
                        continue
                    cleaned_exps.append(exp_id)
                    seen_exps.add(exp_id)

                exp_map[ws_id] = cleaned_exps

            next_order["experiments"] = exp_map

        return save_sidebar_order(ws_root, order=next_order)

    @scoped.get("/sidebar-tree")
    def get_sidebar_tree() -> dict:
        """Return a compact tree for sidebar navigation.

        Goal:
          - Avoid N+1 calls from the UI by aggregating workspace/experiment info.
          - Keep this lightweight: only ids/names and simple counts/flags.
        """

        cfg = config_manager.get_config()
        sidebar_order = load_sidebar_order(ws_root)

        # Load workspaces identity (agent level)
        agent_identity = None
        try:
            agent_identity = ensure_identity(ws_root)
        except Exception:
            pass

        workspaces: list[dict] = []
        if ws_root.exists():
            ws_paths: list[Path] = []
            for ws_path in sorted(ws_root.iterdir()):
                if not ws_path.is_dir():
                    continue
                try:
                    require_workspace_id(ws_path.name)
                except ValueError:
                    # Skip unexpected directories.
                    continue
                ws_paths.append(ws_path)

            ids_sorted = [require_workspace_id(p.name) for p in ws_paths]
            wanted_ws = (
                sidebar_order.get("workspaces")
                if isinstance(sidebar_order.get("workspaces"), list)
                else []
            )
            ws_id_order = apply_order_with_fallback(ids_sorted, wanted_ws)
            ws_by_id = {require_workspace_id(p.name): p for p in ws_paths}
            ws_paths = [ws_by_id[x] for x in ws_id_order if x in ws_by_id]

            wanted_exps_map = (
                sidebar_order.get("experiments")
                if isinstance(sidebar_order.get("experiments"), dict)
                else {}
            )

            for ws_path in ws_paths:
                ws_id = require_workspace_id(ws_path.name)

                try:
                    ws_meta = WorkspaceMeta.load(ws_path, workspace_id=ws_id)
                    ws_display_name = ws_meta.display_name
                    ws_item_uid = ws_meta.workspace_item_uid
                    ws_uid = ws_meta.workspace_uid
                except Exception:
                    ws_display_name = ws_id
                    ws_item_uid = None
                    ws_uid = None

                exp_root = cfg.experiments_path_for(ws_id)
                experiments: list[dict] = []
                if exp_root.exists():
                    exp_paths: list[Path] = []
                    for exp_path in sorted(exp_root.iterdir()):
                        if not exp_path.is_dir():
                            continue
                        try:
                            require_experiment_id(exp_path.name)
                        except ValueError:
                            continue
                        exp_paths.append(exp_path)

                    exp_ids_sorted = [require_experiment_id(p.name) for p in exp_paths]
                    wanted_exps = (
                        wanted_exps_map.get(ws_id)
                        if isinstance(wanted_exps_map.get(ws_id), list)
                        else []
                    )
                    exp_id_order = apply_order_with_fallback(
                        exp_ids_sorted, wanted_exps
                    )
                    exp_by_id = {require_experiment_id(p.name): p for p in exp_paths}
                    exp_paths = [exp_by_id[x] for x in exp_id_order if x in exp_by_id]

                    for exp_path in exp_paths:
                        exp_id = require_experiment_id(exp_path.name)

                        try:
                            exp_meta = ExperimentMeta.load(
                                exp_path, experiment_id=exp_id
                            )
                            exp_display_name = exp_meta.display_name
                            exp_workspace_uid = exp_meta.workspace_uid
                            exp_uid = exp_meta.experiment_uid
                        except Exception:
                            exp_display_name = exp_id
                            exp_workspace_uid = None
                            exp_uid = None

                        models_dir = exp_path / "models"
                        model_runs_count = 0
                        if models_dir.exists() and models_dir.is_dir():
                            try:
                                model_runs_count = sum(
                                    1
                                    for p in models_dir.glob("run_*")
                                    if p.exists() and p.is_dir()
                                )
                            except Exception:
                                model_runs_count = 0

                        eval_dir = exp_path / "eval"
                        eval_scores_count = 0
                        if eval_dir.exists() and eval_dir.is_dir():
                            try:
                                eval_scores_count = sum(
                                    1
                                    for p in eval_dir.glob("scores_*.json")
                                    if p.exists() and p.is_file()
                                )
                            except Exception:
                                eval_scores_count = 0

                        label_input_dir = exp_path / "label_input"
                        labels_path = label_input_dir / "labels.json"
                        legacy_current = label_input_dir / "current.json"
                        has_labels = labels_path.exists() or legacy_current.exists()

                        schema_path = label_input_dir / "label_schema.json"
                        has_label_schema = schema_path.exists()

                        history_dir = label_input_dir / "history"
                        label_backups_count = 0
                        label_backups: list[dict] = []
                        label_backups_truncated = False
                        label_schema_backups_count = 0
                        label_schema_backups: list[dict] = []
                        label_schema_backups_truncated = False
                        if history_dir.exists() and history_dir.is_dir():
                            try:
                                # labels history:
                                # - labels_<backup_id>.json
                                # - current_<backup_id>.json (legacy)
                                label_candidates = []
                                schema_candidates = []
                                for p in history_dir.glob("*.json"):
                                    if not p.is_file():
                                        continue
                                    name = p.name
                                    if not name.endswith(".json"):
                                        continue

                                    prefix = None
                                    if (
                                        name.startswith("labels_")
                                        or name.startswith("current_")
                                    ):
                                        prefix = name.split("_", 1)[0] + "_"
                                    elif name.startswith("label_schema_"):
                                        prefix = "label_schema_"
                                    if prefix is None:
                                        continue

                                    backup_id = name[len(prefix) : -len(".json")]
                                    if not _BACKUP_ID_RE.match(backup_id):
                                        continue

                                    try:
                                        st = p.stat()
                                    except OSError:
                                        continue

                                    candidate = {
                                        "id": backup_id,
                                        "filename": name,
                                        "size_bytes": int(st.st_size),
                                        "modified_at": float(st.st_mtime),
                                    }
                                    if prefix == "label_schema_":
                                        schema_candidates.append(candidate)
                                    else:
                                        label_candidates.append(candidate)

                                label_candidates.sort(
                                    key=lambda x: x.get("modified_at", 0.0),
                                    reverse=True,
                                )
                                schema_candidates.sort(
                                    key=lambda x: x.get("modified_at", 0.0),
                                    reverse=True,
                                )

                                limit = 20
                                label_backups_count = len(label_candidates)
                                label_backups = label_candidates[:limit]
                                label_backups_truncated = (
                                    len(label_candidates) > limit
                                )
                                label_schema_backups_count = len(schema_candidates)
                                label_schema_backups = schema_candidates[:limit]
                                label_schema_backups_truncated = (
                                    len(schema_candidates) > limit
                                )
                            except Exception:
                                label_backups_count = 0
                                label_backups = []
                                label_backups_truncated = False
                                label_schema_backups_count = 0
                                label_schema_backups = []
                                label_schema_backups_truncated = False

                        experiments.append(
                            {
                                "id": exp_id,
                                "display_name": exp_display_name,
                                "workspace_uid": exp_workspace_uid,
                                "experiment_uid": exp_uid,
                                "counts": {
                                    "model_runs": model_runs_count,
                                    "eval_scores": eval_scores_count,
                                    "label_backups": label_backups_count,
                                    "label_schema_backups": label_schema_backups_count,
                                },
                                "flags": {
                                    "has_labels": has_labels,
                                    "has_label_schema": has_label_schema,
                                },
                                "label_backups": label_backups,
                                "label_backups_truncated": label_backups_truncated,
                                "label_schema_backups": label_schema_backups,
                                "label_schema_backups_truncated": label_schema_backups_truncated,
                            }
                        )

                workspaces.append(
                    {
                        "id": ws_id,
                        "display_name": ws_display_name,
                        "workspace_item_uid": ws_item_uid,
                        "workspace_uid": ws_uid,
                        "workspace_root": str(ws_path.resolve()),
                        "experiments": experiments,
                    }
                )

        response = {"generated_at": time.time(), "workspaces": workspaces}

        # Add agent identity if available
        if agent_identity:
            response["agent"] = {
                "container_id": agent_identity.container_id,
                "install_id": agent_identity.install_id,
                "agent_version": agent_identity.agent_version,
            }

        return response

    @scoped.post("")
    def create_workspace_scoped(body: CreateWorkspaceBody) -> dict:
        display_name = (body.display_name or "").strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="display_name is required")

        trash_root = (ws_root / ".trash").resolve()
        reserved_ws_ids = _reserved_workspace_ids_in_trash(trash_root)

        desired_id = (body.id or "").strip()
        if desired_id:
            try:
                ws_id = require_workspace_id(desired_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if (ws_root / ws_id).exists():
                raise HTTPException(
                    status_code=409, detail="workspace id already exists"
                )
            if (trash_root / ws_id).exists() or ws_id in reserved_ws_ids:
                raise HTTPException(
                    status_code=409,
                    detail="workspace id conflicts with trashed items",
                )
        else:
            ws_id = ""
            for _ in range(50):
                candidate = generate_short_id()
                if (ws_root / candidate).exists():
                    continue
                # Avoid collisions with trash entries (both directory names and original ids).
                if (trash_root / candidate).exists() or candidate in reserved_ws_ids:
                    continue
                # Avoid creating a workspace id equal to the trash folder name itself.
                if candidate == ".trash":
                    continue
                ws_id = candidate
                break
            if not ws_id:
                raise HTTPException(
                    status_code=500, detail="failed to generate unique workspace id"
                )

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        require_dir(root)
        created = not root.exists()
        root.mkdir(parents=True, exist_ok=True)

        cfg = config_manager.get_config()
        (root / cfg.source_dir).mkdir(parents=True, exist_ok=True)
        (root / cfg.experiments_dir).mkdir(parents=True, exist_ok=True)

        try:
            WorkspaceMeta.create(root, workspace_id=ws_id, display_name=display_name)
        except Exception:
            # Best-effort; workspace can still exist.
            pass

        return {
            "status": "ok",
            "workspace": ws_id,
            "display_name": display_name,
            "workspace_root": str(root),
            "created": created,
        }

    @scoped.get("/{workspace}")
    def get_workspace_scoped(workspace: str) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        cfg = config_manager.get_config()

        # Load agent identity
        agent_identity = None
        try:
            agent_identity = ensure_identity(ws_root)
        except Exception:
            pass

        display_name = ws_id
        workspace_item_uid = None
        workspace_uid = None
        try:
            meta = WorkspaceMeta.load(root, workspace_id=ws_id)
            display_name = meta.display_name
            workspace_item_uid = meta.workspace_item_uid
            workspace_uid = meta.workspace_uid
        except Exception:
            pass

        response = {
            "workspace": ws_id,
            "display_name": display_name,
            "workspace_item_uid": workspace_item_uid,
            "workspace_uid": workspace_uid,
            "workspace_root": str(root),
            "exists": root.exists() and root.is_dir(),
            "source_dir": cfg.source_dir,
            "experiments_dir": cfg.experiments_dir,
        }

        # Add agent identity if available
        if agent_identity:
            response["agent"] = {
                "container_id": agent_identity.container_id,
                "install_id": agent_identity.install_id,
                "agent_version": agent_identity.agent_version,
            }

        return response

    @scoped.patch("/{workspace}")
    def rename_workspace_scoped(workspace: str, body: RenameWorkspaceBody) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        display_name = (body.display_name or "").strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="display_name is required")

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=404, detail="workspace not found")

        meta = WorkspaceMeta.load(root, workspace_id=ws_id)
        meta.rename(root, display_name=display_name)

        return {
            "status": "ok",
            "workspace": ws_id,
            "display_name": meta.display_name,
            "workspace_root": str(root),
        }

    @scoped.post("/{workspace}/backup-jobs")
    def create_workspace_backup_job_scoped(workspace: str) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=404, detail="workspace not found")

        with backup_lock:
            _purge_expired_backups(max_age_seconds=24 * 60 * 60)
            running_id = running_backup_by_workspace.get(ws_id)
            if running_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"backup job already running: {running_id}",
                )

            latest_jobs = _list_workspace_backup_jobs(ws_id, limit=10_000, offset=0)
            for job in latest_jobs:
                if str(job.get("status") or "") in ("queued", "running"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"backup job already running: {job.get('id')}",
                    )

            job_id = uuid.uuid4().hex
            running_backup_by_workspace[ws_id] = job_id

            job = {
                "id": job_id,
                "type": "workspace_backup",
                "workspace": ws_id,
                "status": "queued",
                "progress": {"percent": 0, "message": "queued"},
                "created_at": time.time(),
                "started_at": None,
                "ended_at": None,
                "filename": None,
                "size_bytes": None,
                "artifact_path": None,
                "error": None,
            }
            _save_workspace_backup_job(job)

        thread = threading.Thread(
            target=_run_workspace_backup_job,
            args=(job_id, ws_id),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "queued", "workspace": ws_id}

    @scoped.get("/{workspace}/backup-jobs")
    def list_workspace_backup_jobs_scoped(
        workspace: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        jobs = _list_workspace_backup_jobs(ws_id, limit=limit, offset=offset)
        return {"workspace": ws_id, "jobs": jobs}

    @scoped.get("/{workspace}/backup-jobs/{job_id}")
    def get_workspace_backup_job_scoped(workspace: str, job_id: str) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job = _load_workspace_backup_job(job_id)
        if str(job.get("workspace") or "") != ws_id:
            raise HTTPException(status_code=404, detail="backup job not found")
        return job

    @scoped.get("/{workspace}/backup-jobs/{job_id}/download")
    def download_workspace_backup_job_scoped(workspace: str, job_id: str):
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job = _load_workspace_backup_job(job_id)
        if str(job.get("workspace") or "") != ws_id:
            raise HTTPException(status_code=404, detail="backup job not found")
        if str(job.get("status") or "") != "done":
            raise HTTPException(status_code=409, detail="backup job is not completed")

        raw_artifact_path = str(job.get("artifact_path") or "").strip()
        if not raw_artifact_path:
            raise HTTPException(
                status_code=410,
                detail="backup artifact expired; run backup again",
            )
        artifact_path = Path(raw_artifact_path).resolve()
        if not is_subpath(backup_artifacts_root, artifact_path):
            raise HTTPException(status_code=500, detail="invalid backup artifact path")
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(
                status_code=410,
                detail="backup artifact expired; run backup again",
            )

        filename = str(job.get("filename") or "").strip() or artifact_path.name
        return FileResponse(
            path=str(artifact_path),
            media_type="application/octet-stream",
            filename=filename,
        )

    @scoped.post("/restore-jobs")
    async def create_workspace_restore_job_scoped(request: Request) -> dict:
        job_id = uuid.uuid4().hex
        upload_path = _workspace_restore_upload_path(job_id)
        total_bytes = 0
        try:
            with upload_path.open("wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    f.write(chunk)
        except Exception as exc:
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(status_code=500, detail=f"failed to receive upload: {exc}")

        if total_bytes <= 0:
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(status_code=400, detail="backup payload is empty")

        try:
            archive_meta = _read_workspace_meta_from_archive(upload_path)
        except ValueError as exc:
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(status_code=400, detail=str(exc))

        ws_id = str(archive_meta.get("workspace_id") or "")
        target_root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, target_root):
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(status_code=400, detail="invalid workspace id")
        if target_root.exists():
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(status_code=409, detail="workspace id already exists")

        trash_root = (ws_root / ".trash").resolve()
        reserved_ws_ids = _reserved_workspace_ids_in_trash(trash_root)
        if (trash_root / ws_id).exists() or ws_id in reserved_ws_ids:
            if upload_path.exists():
                try:
                    upload_path.unlink()
                except OSError:
                    pass
            raise HTTPException(
                status_code=409,
                detail="workspace id conflicts with trashed items",
            )

        with restore_lock:
            running_id = running_restore_by_workspace.get(ws_id)
            if running_id:
                if upload_path.exists():
                    try:
                        upload_path.unlink()
                    except OSError:
                        pass
                raise HTTPException(
                    status_code=409,
                    detail=f"restore job already running: {running_id}",
                )
            running_restore_by_workspace[ws_id] = job_id

            job = {
                "id": job_id,
                "type": "workspace_restore",
                "workspace": ws_id,
                "status": "queued",
                "progress": {"percent": 0, "message": "queued"},
                "created_at": time.time(),
                "started_at": None,
                "ended_at": None,
                "error": None,
                "bytes": total_bytes,
            }
            _save_workspace_restore_job(job)

        thread = threading.Thread(
            target=_run_workspace_restore_job,
            args=(job_id, ws_id),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "queued", "workspace": ws_id}

    @scoped.get("/restore-jobs/{job_id}")
    def get_workspace_restore_job_scoped(job_id: str) -> dict:
        return _load_workspace_restore_job(job_id)

    @scoped.get("/{workspace}/backup")
    def backup_workspace_scoped(workspace: str):
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=404, detail="workspace not found")

        workspace_meta_path = WorkspaceMeta.path_for(root)
        if not workspace_meta_path.exists():
            # Ensure workspace.json is always included at restore decision time.
            try:
                meta = WorkspaceMeta.load(root, workspace_id=ws_id)
                meta.save(root)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"failed to prepare workspace.json: {exc}"
                )

        members: list[str] = ["workspace.json"]
        if (root / cfg.source_dir).exists():
            members.append(cfg.source_dir)
        if (root / cfg.experiments_dir).exists():
            members.append(cfg.experiments_dir)

        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        download_name = f"workspace_{ws_id}_{ts}{_WORKSPACE_BACKUP_EXT}"
        tmp_dir = Path(
            tempfile.mkdtemp(prefix=f"xima-ws-backup-{ws_id}-", dir=str(backup_work_root))
        )
        archive_path = tmp_dir / download_name

        try:
            _create_workspace_backup_archive(
                workspace_root=root, archive_path=archive_path, members=members
            )
        except ValueError as exc:
            _cleanup_dir(str(tmp_dir))
            raise HTTPException(status_code=500, detail=str(exc))

        return FileResponse(
            path=str(archive_path),
            media_type="application/octet-stream",
            filename=download_name,
            background=BackgroundTask(_cleanup_dir, str(tmp_dir)),
        )

    @scoped.post("/restore")
    async def restore_workspace_scoped(request: Request) -> dict:
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix="xima-ws-restore-upload-",
                dir=str(restore_tmp_root),
            )
        )
        archive_path = tmp_dir / f"upload{_WORKSPACE_BACKUP_EXT}"
        staging_dir: Path | None = None
        total_bytes = 0
        try:
            with archive_path.open("wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    f.write(chunk)

            if total_bytes <= 0:
                raise HTTPException(status_code=400, detail="backup payload is empty")

            try:
                archive_meta = _read_workspace_meta_from_archive(archive_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            ws_id = archive_meta["workspace_id"]
            target_root = (ws_root / ws_id).resolve()
            if not is_subpath(ws_root, target_root):
                raise HTTPException(status_code=400, detail="invalid workspace id")
            if target_root.exists():
                raise HTTPException(status_code=409, detail="workspace id already exists")

            trash_root = (ws_root / ".trash").resolve()
            reserved_ws_ids = _reserved_workspace_ids_in_trash(trash_root)
            if (trash_root / ws_id).exists() or ws_id in reserved_ws_ids:
                raise HTTPException(
                    status_code=409,
                    detail="workspace id conflicts with trashed items",
                )

            staging_dir = Path(
                tempfile.mkdtemp(prefix=f".restore-{ws_id}-", dir=str(restore_tmp_root))
            ).resolve()
            if not is_subpath(restore_tmp_root, staging_dir):
                raise HTTPException(status_code=500, detail="invalid staging path")

            try:
                _extract_archive_into_dir(archive_path, staging_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            restored_meta_path = WorkspaceMeta.path_for(staging_dir)
            if not restored_meta_path.exists() or not restored_meta_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail="workspace.json not found at archive root",
                )

            try:
                restored_meta = json.loads(restored_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="workspace.json is invalid")
            restored_id = str(
                restored_meta.get("workspace_id") or restored_meta.get("id") or ""
            ).strip()
            try:
                restored_id = require_workspace_id(restored_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if restored_id != ws_id:
                raise HTTPException(
                    status_code=400,
                    detail="workspace id mismatch in backup archive",
                )

            try:
                target_root.mkdir(parents=False, exist_ok=False)
                for child in staging_dir.iterdir():
                    shutil.move(str(child), str(target_root / child.name))
            except Exception as exc:
                _cleanup_dir(str(target_root))
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to finalize workspace restore: {exc}",
                )

            # Keep expected workspace layout even if the backup is minimal.
            (target_root / cfg.source_dir).mkdir(parents=True, exist_ok=True)
            (target_root / cfg.experiments_dir).mkdir(parents=True, exist_ok=True)

            return {
                "status": "restored",
                "workspace": ws_id,
                "workspace_root": str(target_root),
                "bytes": total_bytes,
            }
        finally:
            if staging_dir is not None:
                _cleanup_dir(str(staging_dir))
            _cleanup_dir(str(tmp_dir))

    @scoped.post("/{workspace}/trash")
    def trash_workspace_scoped(workspace: str) -> dict:
        ws_id = workspace.strip()
        try:
            ws_id = require_workspace_id(ws_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        root = (ws_root / ws_id).resolve()
        if not is_subpath(ws_root, root):
            raise HTTPException(status_code=400, detail="invalid workspace")
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=404, detail="workspace not found")

        # Collect dependent experiments that will be moved together with workspace.
        dependent_experiments: list[str] = []
        exp_root = cfg.experiments_path_for(ws_id)
        if exp_root.exists() and exp_root.is_dir():
            for exp_path in exp_root.iterdir():
                if not exp_path.is_dir():
                    continue
                try:
                    dependent_experiments.append(
                        require_experiment_id(exp_path.name),
                    )
                except ValueError:
                    continue
        dependent_experiments.sort()
        dependent_experiments_count = len(dependent_experiments)

        # Load agent identity
        agent_identity = None
        try:
            agent_identity = ensure_identity(ws_root)
        except Exception:
            pass

        display_name = ws_id
        workspace_item_uid = None
        workspace_uid = None
        try:
            meta = WorkspaceMeta.load(root, workspace_id=ws_id)
            display_name = meta.display_name
            workspace_item_uid = meta.workspace_item_uid
            workspace_uid = meta.workspace_uid
        except Exception:
            pass

        trash_root = (ws_root / ".trash").resolve()
        trash_root.mkdir(parents=True, exist_ok=True)

        try:
            trash_id = _generate_unique_trash_id(trash_root)
        except RuntimeError:
            raise HTTPException(status_code=500, detail="failed to generate trash id")

        trash_dir = (trash_root / trash_id).resolve()
        if not is_subpath(trash_root, trash_dir):
            raise HTTPException(status_code=500, detail="invalid trash path")
        trash_dir.mkdir(parents=True, exist_ok=False)

        moved_to = trash_dir / "item"
        deleted_at = time.time()
        try:
            _trash_move(root, moved_to)
            meta_payload = {
                "version": 1,
                "trash_id": trash_id,
                "kind": "workspace",
                "workspace": ws_id,
                "display_name": display_name,
                "deleted_at": deleted_at,
                "original_rel": ws_id,
                "dependent_experiments_count": dependent_experiments_count,
                "dependent_experiments": dependent_experiments,
            }
            # Add agent identity if available
            if agent_identity:
                meta_payload["container_id"] = agent_identity.container_id
                meta_payload["install_id"] = agent_identity.install_id
                meta_payload["agent_version"] = agent_identity.agent_version
            # Add workspace UID fields if present
            if workspace_item_uid is not None:
                meta_payload["workspace_item_uid"] = workspace_item_uid
            if workspace_uid is not None:
                meta_payload["workspace_uid"] = workspace_uid
            _write_json(trash_dir / "meta.json", meta_payload)
        except Exception as exc:
            # Best-effort cleanup.
            try:
                if moved_to.exists() and moved_to.is_dir() and not root.exists():
                    _trash_move(moved_to, root)
            except Exception:
                pass
            try:
                shutil.rmtree(trash_dir)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "status": "ok",
            "trash_id": trash_id,
            "kind": "workspace",
            "workspace": ws_id,
            "display_name": display_name,
            "deleted_at": deleted_at,
            "dependent_experiments_count": dependent_experiments_count,
            "dependent_experiments": dependent_experiments,
            "moved_from": str(root),
            "moved_to": str(moved_to),
        }

    router.include_router(scoped)

    return router
