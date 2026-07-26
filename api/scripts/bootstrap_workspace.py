from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running from anywhere: ensure "agent" directory is on sys.path
AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from app.config import ConfigManager  # noqa: E402

SCRIPT_FILES: list[str] = []


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap agent workspace by copying existing assets"
    )
    parser.add_argument(
        "--experiment",
        default="default",
        help="Experiment name to populate under experiments/",
    )
    parser.add_argument(
        "--workspace",
        default="ws_test",
        help="Workspace name under agent/workspaces/",
    )
    parser.add_argument(
        "--legacy-root",
        default=str(AGENT_DIR.parent),
        help="Legacy root containing 01_processing/ and 02_train/ (default: repo root)",
    )
    args = parser.parse_args()

    manager = ConfigManager(AGENT_DIR)
    cfg = manager.get_config()

    # Target workspace under agent/workspaces/<name>
    workspace_root = (AGENT_DIR / "workspaces" / args.workspace).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / cfg.source_dir).mkdir(parents=True, exist_ok=True)
    (workspace_root / cfg.experiments_dir).mkdir(parents=True, exist_ok=True)

    # Copy source images
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    legacy_downloads = legacy_root / "01_processing" / "downloads"
    target_source = (workspace_root / cfg.source_dir).resolve()
    copy_dir(legacy_downloads, target_source)

    print(f"Copied downloads -> {target_source}")
    print("Copied scripts -> (skipped; scripts are fixed under agent/pipeline)")
    print("Done. This script only copies; it never deletes or moves legacy data.")
    print("Next: POST /workspace/select and POST /experiments/select in the API.")


if __name__ == "__main__":
    main()
