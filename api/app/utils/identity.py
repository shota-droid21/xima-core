"""Identity management for workspaces container (agent).

This module manages workspaces container identity via identity.json files
stored at workspaces/.xima/identity.json.

The identity.json file binds a workspaces directory (agent) to unique IDs:
- container_id: origin ID (immutable, survives copies)
- install_id: installation instance ID (can be regenerated via rebind)
- agent_version: version of the agent that created/updated this identity
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.utils.uid import generate_uid, is_valid_uid


# Default agent version
DEFAULT_AGENT_VERSION = "0.1.0"


@dataclass
class WorkspacesIdentity:
    """Represents the identity of a workspaces container (agent).

    Attributes:
        container_id: Origin ID (wsc_*), immutable across copies
        install_id: Installation instance ID (wsi_*), regenerated on rebind
        agent_version: Version string (e.g., "0.1.0")
        created_at: ISO 8601 timestamp when the identity was created
        updated_at: ISO 8601 timestamp when the identity was last updated
    """

    container_id: str
    install_id: str
    agent_version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "container_id": self.container_id,
            "install_id": self.install_id,
            "agent_version": self.agent_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspacesIdentity":
        """Create from dictionary loaded from JSON."""
        return cls(
            container_id=data["container_id"],
            install_id=data["install_id"],
            agent_version=data.get("agent_version", DEFAULT_AGENT_VERSION),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
        )


def get_identity_path(workspaces_root: Path) -> Path:
    """Get the path to identity.json for workspaces container.

    Args:
        workspaces_root: Path to workspaces root directory

    Returns:
        Path to workspaces/.xima/identity.json
    """
    return workspaces_root / ".xima" / "identity.json"


def _find_legacy_identity(workspaces_root: Path) -> Optional[str]:
    """Find legacy identity.json from workspace subdirectories.

    This handles migration from the old incorrect placement at
    <workspace>/.xima/identity.json to the correct placement at
    workspaces/.xima/identity.json.

    Args:
        workspaces_root: Path to workspaces root directory

    Returns:
        Legacy uid (wsc_*) if found, None otherwise
    """
    if not workspaces_root.exists():
        return None

    for ws_dir in workspaces_root.iterdir():
        if not ws_dir.is_dir() or ws_dir.name.startswith("."):
            continue

        legacy_identity_path = ws_dir / ".xima" / "identity.json"
        if not legacy_identity_path.exists():
            continue

        try:
            with legacy_identity_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Old format: {"uid": "wsc_...", "created_at": "..."}
            if "uid" in data and is_valid_uid(data["uid"], "workspace_container"):
                return data["uid"]
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return None


def load_identity(workspaces_root: Path) -> Optional[WorkspacesIdentity]:
    """Load workspaces identity from identity.json.

    Args:
        workspaces_root: Path to workspaces root directory

    Returns:
        WorkspacesIdentity if file exists and is valid, None otherwise
    """
    identity_path = get_identity_path(workspaces_root)

    if not identity_path.exists():
        return None

    try:
        with identity_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        required = ["container_id", "install_id", "created_at"]
        if not all(key in data for key in required):
            return None

        # Validate UID formats
        if not is_valid_uid(data["container_id"], "workspace_container"):
            return None
        if not is_valid_uid(data["install_id"], "workspace_item"):
            return None

        return WorkspacesIdentity.from_dict(data)

    except (json.JSONDecodeError, OSError, KeyError):
        return None


def create_identity(
    workspaces_root: Path,
    container_id: Optional[str] = None,
    agent_version: str = DEFAULT_AGENT_VERSION,
) -> WorkspacesIdentity:
    """Create and save a new workspaces identity.

    Args:
        workspaces_root: Path to workspaces root directory
        container_id: Optional container_id to reuse (for migration)
        agent_version: Agent version string

    Returns:
        Newly created WorkspacesIdentity

    Raises:
        OSError: If unable to create .xima directory or write file
    """
    # Generate IDs
    if container_id is None:
        container_id = generate_uid("workspace_container")
    install_id = generate_uid("workspace_item")

    now = datetime.now(timezone.utc).isoformat()

    identity = WorkspacesIdentity(
        container_id=container_id,
        install_id=install_id,
        agent_version=agent_version,
        created_at=now,
        updated_at=now,
    )

    # Ensure .xima directory exists
    xima_dir = workspaces_root / ".xima"
    xima_dir.mkdir(parents=True, exist_ok=True)

    # Write identity.json
    identity_path = get_identity_path(workspaces_root)
    with identity_path.open("w", encoding="utf-8") as f:
        json.dump(identity.to_dict(), f, indent=2, ensure_ascii=False)

    return identity


def ensure_identity(
    workspaces_root: Path, agent_version: str = DEFAULT_AGENT_VERSION
) -> WorkspacesIdentity:
    """Ensure workspaces has an identity, creating one if necessary.

    This handles migration from legacy incorrect placement:
    - If workspaces/.xima/identity.json exists, use it
    - Otherwise, check for legacy <workspace>/.xima/identity.json
    - If found, migrate container_id from legacy, generate new install_id
    - If not found, generate both IDs fresh

    Args:
        workspaces_root: Path to workspaces root directory
        agent_version: Agent version string

    Returns:
        WorkspacesIdentity (existing or newly created)
    """
    identity = load_identity(workspaces_root)

    if identity is not None:
        return identity

    # Check for legacy identity (migration path)
    legacy_container_id = _find_legacy_identity(workspaces_root)

    # Create new identity (reusing legacy container_id if found)
    identity = create_identity(
        workspaces_root, container_id=legacy_container_id, agent_version=agent_version
    )

    return identity


def rebind_identity(
    workspaces_root: Path,
    force: bool = False,
    agent_version: str = DEFAULT_AGENT_VERSION,
) -> WorkspacesIdentity:
    """Rebind workspaces to a new install_id.

    This regenerates install_id while keeping container_id unchanged.
    Use this when cloning/copying workspaces to create an independent instance.

    Args:
        workspaces_root: Path to workspaces root directory
        force: If False, raises error if identity.json doesn't exist.
               If True, creates new identity regardless.
        agent_version: Agent version string

    Returns:
        Updated WorkspacesIdentity

    Raises:
        ValueError: If identity.json doesn't exist and force=False
        OSError: If unable to write identity.json
    """
    identity_path = get_identity_path(workspaces_root)

    if not force and not identity_path.exists():
        raise ValueError(
            f"Cannot rebind: identity.json not found at {identity_path}. "
            "Use force=True to create new identity."
        )

    # Load existing identity to preserve container_id
    existing = load_identity(workspaces_root)

    if existing is not None:
        container_id = existing.container_id
    else:
        # Check legacy for container_id
        container_id = _find_legacy_identity(workspaces_root)
        if container_id is None:
            # No existing identity, generate fresh
            container_id = generate_uid("workspace_container")

    # Generate new install_id
    install_id = generate_uid("workspace_item")
    now = datetime.now(timezone.utc).isoformat()

    # Preserve created_at if exists
    created_at = existing.created_at if existing else now

    identity = WorkspacesIdentity(
        container_id=container_id,
        install_id=install_id,
        agent_version=agent_version,
        created_at=created_at,
        updated_at=now,
    )

    # Ensure .xima directory exists
    xima_dir = workspaces_root / ".xima"
    xima_dir.mkdir(parents=True, exist_ok=True)

    # Write identity.json
    with identity_path.open("w", encoding="utf-8") as f:
        json.dump(identity.to_dict(), f, indent=2, ensure_ascii=False)

    return identity
