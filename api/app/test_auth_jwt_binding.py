from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth_jwt import enforce_agent_binding


def test_enforce_agent_binding_accepts_matching_install_id() -> None:
    payload = {"agent_install_id": "wsi_abc123"}
    enforce_agent_binding(payload, expected_install_id="wsi_abc123")


def test_enforce_agent_binding_rejects_missing_install_id() -> None:
    payload: dict = {}
    with pytest.raises(HTTPException) as exc:
        enforce_agent_binding(payload, expected_install_id="wsi_abc123")
    assert exc.value.status_code == 401
    assert "missing agent_install_id" in str(exc.value.detail)


def test_enforce_agent_binding_rejects_mismatched_install_id() -> None:
    payload = {"agent_install_id": "wsi_other"}
    with pytest.raises(HTTPException) as exc:
        enforce_agent_binding(payload, expected_install_id="wsi_abc123")
    assert exc.value.status_code == 401
    assert "not bound to this agent" in str(exc.value.detail)


def test_enforce_agent_binding_accepts_matching_runtime_container() -> None:
    payload = {
        "agent_install_id": "wsi_abc123",
        "agent_container_runtime_id": "wsc_runtime",
    }
    enforce_agent_binding(
        payload,
        expected_install_id="wsi_abc123",
        expected_container_runtime_id="wsc_runtime",
    )


def test_enforce_agent_binding_rejects_mismatched_runtime_container() -> None:
    payload = {
        "agent_install_id": "wsi_abc123",
        "agent_container_runtime_id": "wsc_other",
    }
    with pytest.raises(HTTPException) as exc:
        enforce_agent_binding(
            payload,
            expected_install_id="wsi_abc123",
            expected_container_runtime_id="wsc_runtime",
        )
    assert exc.value.status_code == 401
    assert "not bound to this agent container" in str(exc.value.detail)
