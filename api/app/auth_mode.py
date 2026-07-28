from __future__ import annotations

import os

AUTH_MODE_OPEN = "open"
AUTH_MODE_EXTERNAL = "external"
AUTH_MODE_LOCAL = "local"


def resolve_agent_auth_mode() -> str:
    raw = (os.environ.get("XIMA_AGENT_AUTH_MODE") or AUTH_MODE_OPEN).strip().lower()
    if raw == "saas":
        return AUTH_MODE_EXTERNAL
    return raw
