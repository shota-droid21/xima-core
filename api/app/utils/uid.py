"""UID (Unique Identifier) generation utilities.

This module provides functions to generate and validate UIDs for
workspace containers, workspace items, workspaces, and experiments.

UIDs use UUID v4 with a 16-character hex prefix for uniqueness while
maintaining readability. They are distinct from short IDs (8 chars)
used in URLs.
"""

import uuid
from typing import Literal

UidKind = Literal[
    "workspace_container",
    "workspace_item",
    "workspace",
    "experiment",
]


def generate_uid(kind: UidKind) -> str:
    """Generate a new UID with appropriate prefix.

    Args:
        kind: The type of entity the UID is for

    Returns:
        UID string in format: {prefix}_{16_hex_chars}

    Examples:
        >>> generate_uid("workspace_container")
        'wsc_a1b2c3d4e5f67890'
        >>> generate_uid("experiment")
        'exp_1234567890abcdef'
    """
    prefix_map = {
        "workspace_container": "wsc",
        "workspace_item": "wsi",
        "workspace": "ws",
        "experiment": "exp",
    }

    prefix = prefix_map[kind]
    uid_hex = uuid.uuid4().hex[:16]
    return f"{prefix}_{uid_hex}"


def is_valid_uid(uid: str, kind: UidKind | None = None) -> bool:
    """Validate UID format.

    Args:
        uid: The UID string to validate
        kind: Optional specific kind to check prefix against

    Returns:
        True if UID has valid format, False otherwise

    Examples:
        >>> is_valid_uid("wsc_a1b2c3d4e5f67890")
        True
        >>> is_valid_uid("wsc_a1b2c3d4e5f67890", "workspace_container")
        True
        >>> is_valid_uid("wsc_a1b2c3d4e5f67890", "workspace")
        False
        >>> is_valid_uid("invalid")
        False
    """
    if not isinstance(uid, str):
        return False

    parts = uid.split("_", 1)
    if len(parts) != 2:
        return False

    prefix, hex_part = parts

    # Check valid prefix
    valid_prefixes = {"wsc", "wsi", "ws", "exp"}
    if prefix not in valid_prefixes:
        return False

    # Check hex part is 16 characters
    if len(hex_part) != 16:
        return False

    # Check hex part contains only hex chars
    try:
        int(hex_part, 16)
    except ValueError:
        return False

    # If specific kind is requested, check prefix matches
    if kind is not None:
        prefix_map = {
            "workspace_container": "wsc",
            "workspace_item": "wsi",
            "workspace": "ws",
            "experiment": "exp",
        }
        expected_prefix = prefix_map[kind]
        if prefix != expected_prefix:
            return False

    return True


def get_uid_kind(uid: str) -> UidKind | None:
    """Get the kind of entity from a UID.

    Args:
        uid: The UID string

    Returns:
        The kind of entity, or None if invalid UID

    Examples:
        >>> get_uid_kind("wsc_a1b2c3d4e5f67890")
        'workspace_container'
        >>> get_uid_kind("exp_1234567890abcdef")
        'experiment'
        >>> get_uid_kind("invalid")
        None
    """
    if not is_valid_uid(uid):
        return None

    prefix = uid.split("_", 1)[0]

    kind_map = {
        "wsc": "workspace_container",
        "wsi": "workspace_item",
        "ws": "workspace",
        "exp": "experiment",
    }

    return kind_map.get(prefix)
