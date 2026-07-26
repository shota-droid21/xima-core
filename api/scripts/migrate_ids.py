from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running from anywhere: ensure "api" directory is on sys.path
API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from app.utils.meta import ExperimentMeta, WorkspaceMeta  # noqa: E402
from app.utils.short_id import generate_short_id, is_valid_short_id  # noqa: E402


@dataclass
class RenamePlan:
    kind: str  # workspace | experiment
    src: Path
    dst: Path
    old_id: str
    new_id: str
    workspace_old_id: str | None = None
    workspace_new_id: str | None = None


def _default_workspaces_root() -> Path:
    env_ws = os.environ.get("XIMA_WORKSPACES_ROOT")
    if env_ws:
        return Path(env_ws).resolve()
    # repo default: core/workspaces
    agent_root = API_DIR.parent
    return (agent_root / "workspaces").resolve()


def _iter_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def _pick_unique_short_id(used: set[str]) -> str:
    for _ in range(50):
        candidate = generate_short_id()
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError("failed to generate unique short id")


def build_plan(
    workspaces_root: Path,
) -> Tuple[List[RenamePlan], Dict[str, Dict[str, str]]]:
    plans: List[RenamePlan] = []

    trash_root = (workspaces_root / ".trash").resolve()
    used_ws_ids = {p.name for p in _iter_dirs(workspaces_root)}
    used_trash_ids = {p.name for p in _iter_dirs(trash_root)}
    used_all = set(used_ws_ids) | set(used_trash_ids)

    mapping: Dict[str, Dict[str, str]] = {"workspaces": {}, "experiments": {}}

    for ws_path in _iter_dirs(workspaces_root):
        if ws_path.name.startswith("."):
            continue

        ws_old = ws_path.name
        if is_valid_short_id(ws_old):
            ws_new = ws_old
        else:
            ws_new = _pick_unique_short_id(used_all)
            plans.append(
                RenamePlan(
                    kind="workspace",
                    src=ws_path,
                    dst=workspaces_root / ws_new,
                    old_id=ws_old,
                    new_id=ws_new,
                )
            )
            mapping["workspaces"][ws_old] = ws_new

        exp_root = ws_path / "experiments"
        for exp_path in _iter_dirs(exp_root):
            exp_old = exp_path.name
            if is_valid_short_id(exp_old):
                continue
            exp_new = _pick_unique_short_id(used_all)
            plans.append(
                RenamePlan(
                    kind="experiment",
                    src=exp_path,
                    # Destination must use the *new* workspace id, because the
                    # workspace may be renamed in the first phase.
                    dst=workspaces_root / ws_new / "experiments" / exp_new,
                    old_id=exp_old,
                    new_id=exp_new,
                    workspace_old_id=ws_old,
                    workspace_new_id=ws_new,
                )
            )
            mapping["experiments"][f"{ws_old}/{exp_old}"] = f"{ws_new}/{exp_new}"

    return plans, mapping


def apply_plan(
    plans: List[RenamePlan], *, workspaces_root: Path, dry_run: bool
) -> None:
    # Workspaces first (so experiment roots remain stable under new ws id)
    for p in plans:
        if p.kind != "workspace":
            continue
        if dry_run:
            continue
        if p.dst.exists():
            raise RuntimeError(f"destination already exists: {p.dst}")
        p.src.replace(p.dst)

        # Update workspace.json
        meta = WorkspaceMeta.load(p.dst, workspace_id=p.new_id)
        meta.id = p.new_id
        # Preserve existing display_name unless it was basically the old id.
        if meta.display_name in (p.old_id, meta.id, p.new_id):
            meta.display_name = p.old_id
        meta.save(p.dst)

    # Experiments
    for p in plans:
        if p.kind != "experiment":
            continue
        if dry_run:
            continue
        src = p.src
        if (
            p.workspace_old_id
            and p.workspace_new_id
            and p.workspace_old_id != p.workspace_new_id
        ):
            moved_src = workspaces_root / p.workspace_new_id / "experiments" / p.old_id
            if moved_src.exists():
                src = moved_src

        if p.dst.exists():
            raise RuntimeError(f"destination already exists: {p.dst}")
        if not src.exists():
            raise RuntimeError(f"source does not exist: {src}")

        src.replace(p.dst)

        meta = ExperimentMeta.load(p.dst, experiment_id=p.new_id)
        meta.id = p.new_id
        if meta.display_name in (p.old_id, meta.id, p.new_id):
            meta.display_name = p.old_id
        meta.save(p.dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate workspace/experiment directory names to short ids (^[a-z0-9]{8}$)."
    )
    parser.add_argument(
        "--workspaces-root",
        type=str,
        default=str(_default_workspaces_root()),
        help="Workspaces root directory (default: env XIMA_WORKSPACES_ROOT or core/workspaces).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--write-mapping",
        action="store_true",
        help="Write mapping JSON next to workspaces root as migrate_ids_mapping.json.",
    )

    args = parser.parse_args()
    workspaces_root = Path(args.workspaces_root).resolve()

    plans, mapping = build_plan(workspaces_root)

    print(f"workspaces_root: {workspaces_root}")
    print(f"planned renames: {len(plans)}")
    for p in plans:
        print(f"- {p.kind}: {p.src} -> {p.dst} ({p.old_id} -> {p.new_id})")

    if args.write_mapping:
        mapping_path = workspaces_root / "migrate_ids_mapping.json"
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote mapping: {mapping_path}")

    if not args.apply:
        print("dry-run complete. Re-run with --apply to perform moves.")
        return

    apply_plan(plans, workspaces_root=workspaces_root, dry_run=False)
    print("apply complete.")


if __name__ == "__main__":
    main()
