from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from ..config import Config
from ..utils.paths import is_subpath

SCRIPT_FILENAMES = {
    "apply_label": "apply_label_mapping.py",
    "augment_gray": "augment_gray_from_dataset.py",
    "make_label_list": "make_label_list.py",
    "purge_deleted_images": "purge_deleted_images.py",
    "embed_images": "embed_images.py",
    "train": "train_epoch.py",
    "infer_scores": "infer_heads.py",
    "train_epoch": "train_epoch.py",
    "infer_heads": "infer_heads.py",
    "predict_labels": "predict_labels.py",
}


def _agent_root() -> Path:
    # agent/app/jobs/runner.py -> agent/
    return Path(__file__).resolve().parents[2]


def _resolve_script_path(job_type: str, cfg: Config) -> Path:
    script_name = SCRIPT_FILENAMES[job_type]
    agent_root = _agent_root()

    # New canonical location: agent/pipeline/<script>.py
    candidates: List[Path] = [agent_root / "pipeline" / script_name]

    # Transitional fallback: allow resolving from repo root 02_train for scripts not yet migrated.
    repo_root = agent_root.parent
    candidates.append(repo_root / "02_train" / script_name)

    for candidate in candidates:
        resolved = candidate.resolve()
        # Scripts are allowed under agent/ or repo root; do not require workspace subpath.
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"script not found for job_type={job_type}; expected one of {[str(c) for c in candidates]}"
    )


def build_command(job_type: str, args: Dict[str, Any], cfg: Config) -> List[str]:
    script_path = _resolve_script_path(job_type, cfg)
    cmd: List[str] = [sys.executable, "-u", str(script_path)]

    # Allow passing a raw argv string (primarily for train_epoch options input).
    raw_args = None
    if isinstance(args, dict):
        raw_args = args.get("raw_args") or args.get("_raw_args")

    import shlex

    if isinstance(raw_args, str) and raw_args.strip():
        cmd.extend(shlex.split(raw_args))

    for key, value in args.items():
        if key in ("raw_args", "_raw_args"):
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd
