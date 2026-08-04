"""local モードの配線（ミドルウェア / /local/session）の統合テスト。

TestClient で local key ガードと同一オリジン手渡しを検証。dependency-free。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.local_auth import LOCAL_KEY_HEADER
from app.local_mode import setup_local_mode


def _make_client(tmp_path: Path) -> tuple[TestClient, ConfigManager]:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/guarded")
    def guarded() -> dict:
        return {"ok": True}

    setup_local_mode(app, config_manager)
    return TestClient(app), config_manager


def _fetch_key(client: TestClient) -> str:
    res = client.get("/local/session")
    assert res.status_code == 200, res.text
    return res.json()["key"]


def test_local_mode_blocks_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    client, _ = _make_client(tmp_path)

    res = client.get("/guarded")
    assert res.status_code == 401

    key = _fetch_key(client)
    ok = client.get("/guarded", headers={LOCAL_KEY_HEADER: key})
    assert ok.status_code == 200
    assert ok.json() == {"ok": True}


def test_local_mode_health_is_exempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    client, _ = _make_client(tmp_path)
    # key 無しでも health は通る（app が key を取りに来る前段）。
    assert client.get("/health").status_code == 200


def test_local_mode_browser_static_is_exempt(tmp_path: Path, monkeypatch) -> None:
    """ブラウザが自動取得する静的物は key 無しで 401 にならないこと。

    これらは JS を経由せずブラウザ自身が要求するため X-Xima-Local-Key を載せられない。
    免除が漏れると「タブに既定アイコンが出るだけ」という気づきにくい失敗になる
    （実際に /favicon.svg の免除漏れで発生した）。

    app dist を mount していないので 404 になるが、**401 でなければ免除は効いている**。
    """
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    client, _ = _make_client(tmp_path)
    for path in ("/favicon.svg", "/favicon.ico", "/apple-touch-icon.png"):
        assert client.get(path).status_code != 401, path


def test_local_session_rejects_foreign_origin(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    client, _ = _make_client(tmp_path)
    res = client.get("/local/session", headers={"origin": "http://evil.test"})
    assert res.status_code == 403


def test_local_session_allows_configured_origin(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    monkeypatch.setenv("XIMA_LOCAL_ALLOWED_ORIGINS", "http://localhost:3500")
    client, _ = _make_client(tmp_path)
    res = client.get("/local/session", headers={"origin": "http://localhost:3500"})
    assert res.status_code == 200


def test_open_mode_does_not_enforce(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "open")
    client, _ = _make_client(tmp_path)
    # open は素通し（既存挙動）。/local/session は local 専用なので 404。
    assert client.get("/guarded").status_code == 200
    assert client.get("/local/session").status_code == 404


def test_wrong_key_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "local")
    client, _ = _make_client(tmp_path)
    res = client.get("/guarded", headers={LOCAL_KEY_HEADER: "not-the-secret"})
    assert res.status_code == 401
