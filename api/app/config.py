from __future__ import annotations

import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .utils.paths import is_subpath, safe_resolve
from .utils.short_id import require_experiment_id, require_workspace_id


@dataclass
class Config:
    workspaces_root: Path
    source_dir: str = "source_images"
    experiments_dir: str = "experiments"
    agent_port: int = 27800

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["workspaces_root"] = str(self.workspaces_root)
        return data

    def _validate_segment(self, name: str, *, label: str) -> str:
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValueError(f"{label} is required")
        if cleaned in (".", ".."):
            raise ValueError(f"Invalid {label}: {cleaned}")
        if "/" in cleaned or "\\" in cleaned:
            raise ValueError(f"Invalid {label}: must be a single path segment")
        return cleaned

    def workspace_path(self, workspace: str) -> Path:
        ws = require_workspace_id(workspace)
        return safe_resolve(self.workspaces_root, ws)

    def resolve_in_workspace(self, workspace: str, relative_path: str | Path) -> Path:
        return safe_resolve(self.workspace_path(workspace), relative_path)

    def source_path_for(self, workspace: str) -> Path:
        return self.resolve_in_workspace(workspace, self.source_dir)

    def experiments_path_for(self, workspace: str) -> Path:
        return self.resolve_in_workspace(workspace, self.experiments_dir)

    def label_input_path_for(self, workspace: str, experiment: str) -> Path:
        exp = require_experiment_id(experiment)
        ws_root = self.workspace_path(workspace)
        path = (
            self.experiments_path_for(workspace) / exp / "label_input" / "labels.json"
        ).resolve()
        if not is_subpath(ws_root, path):
            raise ValueError("Experiment label path escapes workspace")
        return path

    def label_schema_path_for(self, workspace: str, experiment: str) -> Path:
        exp = require_experiment_id(experiment)
        ws_root = self.workspace_path(workspace)
        path = (
            self.experiments_path_for(workspace)
            / exp
            / "label_input"
            / "label_schema.json"
        ).resolve()
        if not is_subpath(ws_root, path):
            raise ValueError("Experiment schema path escapes workspace")
        return path


class ConfigManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.state_dir = base_dir / "state"
        self.jobs_dir = self.state_dir / "jobs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._config = self._default_config()

    def _default_config(self) -> Config:
        env_ws = os.environ.get("XIMA_WORKSPACES_ROOT")
        env_port = os.environ.get("XIMA_AGENT_PORT")
        ws_root = (
            Path(env_ws).resolve()
            if env_ws
            else (self.base_dir / "workspaces").resolve()
        )
        return Config(
            workspaces_root=ws_root,
            # Keep these fixed to avoid UI/agent mismatch.
            source_dir="source_images",
            experiments_dir="experiments",
            agent_port=int(env_port) if env_port else 27800,
        )

    def get_config(self) -> Config:
        with self._lock:
            return self._config

    def save_config(self, cfg: Config) -> Config:
        with self._lock:
            self._config = cfg
            return cfg


def workspaces_root(agent_root: Path) -> Path:
    env_ws = os.environ.get("XIMA_WORKSPACES_ROOT")
    if env_ws:
        return Path(env_ws).resolve()
    return (agent_root / "workspaces").resolve()
