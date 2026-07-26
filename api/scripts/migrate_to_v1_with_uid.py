#!/usr/bin/env python3
"""Migration script: Add UIDs to existing workspaces and experiments.

This script:
1. Creates workspaces/.xima/identity.json (agent identity)
2. Adds workspace_uid to workspace.json
3. Adds experiment_uid to experiment.json
4. Updates workspace.json with workspace_uid reference in experiments

Usage:
    python scripts/migrate_to_v1_with_uid.py [workspaces_root]

Example:
    python scripts/migrate_to_v1_with_uid.py ./workspaces
    python scripts/migrate_to_v1_with_uid.py /path/to/core/workspaces

If workspaces_root is not provided, it will use ../workspaces relative to this script.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.identity import ensure_identity
from app.utils.meta import ExperimentMeta, WorkspaceMeta
from app.utils.short_id import is_valid_short_id
from app.utils.uid import generate_uid


def migrate_workspace(
    workspace_path: Path, workspace_id: str, workspace_uid: str, dry_run: bool = False
) -> dict:
    """Migrate a single workspace to include UIDs.

    Args:
        workspace_path: Path to workspace directory
        workspace_id: Workspace short ID
        workspace_uid: Workspace UID to assign
        dry_run: If True, don't modify files

    Returns:
        dict with migration results
    """
    result = {
        "workspace_id": workspace_id,
        "workspace_uid_added": False,
        "experiments_migrated": 0,
        "errors": [],
    }

    # Add workspace_uid to workspace.json
    try:
        meta = WorkspaceMeta.load(workspace_path, workspace_id=workspace_id)
        if meta.workspace_uid is None:
            if not dry_run:
                meta.workspace_uid = workspace_uid
                meta.save(workspace_path)
            result["workspace_uid_added"] = True
            print(
                f"  ✓ Added workspace_uid to workspace.json: {workspace_uid if not dry_run else '(would add)'}"
            )
        else:
            print(f"  → workspace_uid already exists: {meta.workspace_uid}")
    except Exception as e:
        error_msg = f"Failed to update workspace.json: {e}"
        result["errors"].append(error_msg)
        print(f"  ✗ {error_msg}")
        return result

    # Migrate experiments
    experiments_dir = workspace_path / "experiments"
    if not experiments_dir.exists():
        print("  → No experiments directory")
        return result

    for exp_path in experiments_dir.iterdir():
        if not exp_path.is_dir():
            continue

        exp_id = exp_path.name
        if not is_valid_short_id(exp_id):
            continue

        try:
            exp_meta = ExperimentMeta.load(exp_path, experiment_id=exp_id)

            needs_update = False

            # Add workspace_uid reference
            if exp_meta.workspace_uid is None:
                if not dry_run:
                    exp_meta.workspace_uid = workspace_uid
                needs_update = True

            # Add experiment_uid
            if exp_meta.experiment_uid is None:
                if not dry_run:
                    exp_meta.experiment_uid = generate_uid("experiment")
                needs_update = True

            if needs_update:
                if not dry_run:
                    exp_meta.save(exp_path)
                result["experiments_migrated"] += 1
                print(f"    ✓ Migrated experiment: {exp_id}")
            else:
                print(f"    → Experiment already migrated: {exp_id}")

        except Exception as e:
            error_msg = f"Failed to migrate experiment {exp_id}: {e}"
            result["errors"].append(error_msg)
            print(f"    ✗ {error_msg}")

    return result


def main():
    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--dry-run"]

    # Determine workspaces root
    if len(args) > 0:
        workspaces_root = Path(args[0])
    else:
        # Default: ../workspaces relative to api directory (not scripts directory)
        workspaces_root = Path(__file__).parent.parent.parent / "workspaces"

    workspaces_root = workspaces_root.resolve()

    if not workspaces_root.exists():
        print(f"Error: Workspaces root not found: {workspaces_root}")
        print("Usage: python scripts/migrate_to_v1_with_uid.py [workspaces_root]")
        sys.exit(1)

    print(f"Migrating workspaces in: {workspaces_root}")
    print()

    if dry_run:
        print("*** DRY RUN MODE - No changes will be made ***")
        print()

    total_results = {
        "workspaces_processed": 0,
        "agent_identity_created": False,
        "workspace_uids_added": 0,
        "experiments_migrated": 0,
        "errors": [],
    }

    # Step 1: Ensure workspaces/.xima/identity.json exists (agent identity)
    print("Checking agent identity...")
    try:
        if not dry_run:
            agent_identity = ensure_identity(workspaces_root)
            print(
                f"✓ Agent identity: {agent_identity.container_id} / {agent_identity.install_id}"
            )
            total_results["agent_identity_created"] = True
        else:
            print("  (dry run: would ensure agent identity)")
    except Exception as e:
        error_msg = f"Failed to ensure agent identity: {e}"
        total_results["errors"].append(error_msg)
        print(f"✗ {error_msg}")
        sys.exit(1)

    print()

    # Step 2: Process each workspace
    for workspace_path in sorted(workspaces_root.iterdir()):
        if not workspace_path.is_dir():
            continue

        workspace_id = workspace_path.name

        # Skip special directories
        if workspace_id.startswith("."):
            continue

        # Validate workspace ID format
        if not is_valid_short_id(workspace_id):
            print(f"Skipping invalid workspace ID: {workspace_id}")
            continue

        print(f"Processing workspace: {workspace_id}")

        # Generate workspace_uid
        workspace_uid = generate_uid("workspace")

        result = migrate_workspace(
            workspace_path, workspace_id, workspace_uid, dry_run=dry_run
        )

        total_results["workspaces_processed"] += 1
        if result["workspace_uid_added"]:
            total_results["workspace_uids_added"] += 1
        total_results["experiments_migrated"] += result["experiments_migrated"]
        total_results["errors"].extend(result["errors"])

        print()

    # Summary
    print("=" * 60)
    print("Migration Summary:")
    print(
        f"  Agent identity created: {'Yes' if total_results['agent_identity_created'] else 'No'}"
    )
    print(f"  Workspaces processed: {total_results['workspaces_processed']}")
    print(f"  Workspace UIDs added: {total_results['workspace_uids_added']}")
    print(f"  Experiments migrated: {total_results['experiments_migrated']}")
    print(f"  Errors: {len(total_results['errors'])}")

    if total_results["errors"]:
        print()
        print("Errors encountered:")
        for error in total_results["errors"]:
            print(f"  - {error}")
        sys.exit(1)

    if dry_run:
        print()
        print("*** Dry run completed - no changes were made ***")
        print("Run without --dry-run to apply changes.")
    else:
        print()
        print("✓ Migration completed successfully!")


if __name__ == "__main__":
    main()
