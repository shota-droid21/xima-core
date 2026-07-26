from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class WorkspaceMeta:
    id: str
    display_name: str
    created_at: str
    updated_at: str
    workspace_item_uid: Optional[str] = None
    workspace_uid: Optional[str] = None

    @staticmethod
    def path_for(workspace_root: Path) -> Path:
        return workspace_root / "workspace.json"

    @staticmethod
    def load(workspace_root: Path, *, workspace_id: str) -> "WorkspaceMeta":
        p = WorkspaceMeta.path_for(workspace_root)
        if not p.exists():
            now = _now_iso()
            return WorkspaceMeta(
                id=workspace_id,
                display_name=workspace_id,
                created_at=now,
                updated_at=now,
            )
        data = json.loads(p.read_text(encoding="utf-8"))
        now = _now_iso()
        return WorkspaceMeta(
            id=str(data.get("workspace_id") or data.get("id") or workspace_id),
            display_name=str(data.get("display_name") or workspace_id),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
            workspace_item_uid=data.get("workspace_item_uid"),
            workspace_uid=data.get("workspace_uid"),
        )

    def save(self, workspace_root: Path) -> None:
        p = WorkspaceMeta.path_for(workspace_root)
        payload = {
            "workspace_id": self.id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        # Add UID fields if present (schema v1 extension)
        if self.workspace_item_uid is not None:
            payload["workspace_item_uid"] = self.workspace_item_uid
        if self.workspace_uid is not None:
            payload["workspace_uid"] = self.workspace_uid
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def create(
        workspace_root: Path, *, workspace_id: str, display_name: str
    ) -> "WorkspaceMeta":
        now = _now_iso()
        meta = WorkspaceMeta(
            id=workspace_id,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        meta.save(workspace_root)
        return meta

    def rename(self, workspace_root: Path, *, display_name: str) -> "WorkspaceMeta":
        self.display_name = display_name
        self.updated_at = _now_iso()
        self.save(workspace_root)
        return self


@dataclass
class ExperimentMeta:
    id: str
    display_name: str
    created_at: str
    updated_at: str
    workspace_uid: Optional[str] = None
    experiment_uid: Optional[str] = None

    @staticmethod
    def path_for(experiment_root: Path) -> Path:
        return experiment_root / "experiment.json"

    @staticmethod
    def load(experiment_root: Path, *, experiment_id: str) -> "ExperimentMeta":
        p = ExperimentMeta.path_for(experiment_root)
        if not p.exists():
            now = _now_iso()
            return ExperimentMeta(
                id=experiment_id,
                display_name=experiment_id,
                created_at=now,
                updated_at=now,
            )
        data = json.loads(p.read_text(encoding="utf-8"))
        now = _now_iso()
        return ExperimentMeta(
            id=str(data.get("experiment_id") or data.get("id") or experiment_id),
            display_name=str(data.get("display_name") or experiment_id),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
            workspace_uid=data.get("workspace_uid"),
            experiment_uid=data.get("experiment_uid"),
        )

    def save(self, experiment_root: Path) -> None:
        p = ExperimentMeta.path_for(experiment_root)
        payload = {
            "experiment_id": self.id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        # Add UID fields if present (schema v1 extension)
        if self.workspace_uid is not None:
            payload["workspace_uid"] = self.workspace_uid
        if self.experiment_uid is not None:
            payload["experiment_uid"] = self.experiment_uid
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def create(
        experiment_root: Path, *, experiment_id: str, display_name: str
    ) -> "ExperimentMeta":
        now = _now_iso()
        meta = ExperimentMeta(
            id=experiment_id,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        meta.save(experiment_root)
        return meta

    def rename(self, experiment_root: Path, *, display_name: str) -> "ExperimentMeta":
        self.display_name = display_name
        self.updated_at = _now_iso()
        self.save(experiment_root)
        return self
