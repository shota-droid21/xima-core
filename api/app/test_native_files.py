"""ネイティブ画像/静的配信（nginx 非在時）の統合テスト。

XIMA_SERVE_FILES_NATIVE 有効時に /static /thumbs が workspaces から実配信され、
X-Accel フォールバックが FileResponse になることを検証。dependency-free。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import FileResponse

from app.config import ConfigManager
from app.native_files import (
    create_native_files_router,
    native_file_serving_enabled,
    serve_ws_file,
)


def _make_client(tmp_path: Path, monkeypatch) -> tuple[TestClient, ConfigManager]:
    ws_root = tmp_path / "workspaces"
    ws_root.mkdir()
    monkeypatch.setenv("XIMA_WORKSPACES_ROOT", str(ws_root))
    config_manager = ConfigManager(tmp_path / "agent")
    app = FastAPI()
    app.include_router(create_native_files_router(config_manager))
    return TestClient(app), config_manager


def test_flag_parsing(monkeypatch) -> None:
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("XIMA_SERVE_FILES_NATIVE", truthy)
        assert native_file_serving_enabled() is True
    for falsy in ("", "0", "false", "no"):
        monkeypatch.setenv("XIMA_SERVE_FILES_NATIVE", falsy)
        assert native_file_serving_enabled() is False


def test_static_serves_file_from_workspaces(tmp_path: Path, monkeypatch) -> None:
    client, cfg = _make_client(tmp_path, monkeypatch)
    ws_root = cfg.get_config().workspaces_root
    target = ws_root / "ws1" / "experiments" / "exp1" / "cache" / "thumbs" / "a.webp"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"webp-bytes")

    res = client.get("/static/ws1/experiments/exp1/cache/thumbs/a.webp")
    assert res.status_code == 200, res.text
    assert res.content == b"webp-bytes"
    assert res.headers["content-type"] == "image/webp"


def test_thumbs_prefix_also_serves(tmp_path: Path, monkeypatch) -> None:
    client, cfg = _make_client(tmp_path, monkeypatch)
    ws_root = cfg.get_config().workspaces_root
    target = ws_root / "ws1" / "img.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"png")

    res = client.get("/thumbs/ws1/img.png")
    assert res.status_code == 200
    assert res.content == b"png"


def test_missing_file_is_404(tmp_path: Path, monkeypatch) -> None:
    client, _ = _make_client(tmp_path, monkeypatch)
    res = client.get("/static/ws1/nope.webp")
    assert res.status_code == 404


def test_path_traversal_rejected(tmp_path: Path, monkeypatch) -> None:
    client, _ = _make_client(tmp_path, monkeypatch)
    # workspaces_root を抜ける相対は 404（safe_resolve が弾く）。
    res = client.get("/static/../../etc/passwd")
    assert res.status_code == 404


def test_x_accel_fallback_returns_fileresponse(tmp_path: Path, monkeypatch) -> None:
    """native 時は _x_accel_file_response が X-Accel ではなく FileResponse を返す。"""
    from app.images import _x_accel_file_response

    ws_root = tmp_path / "workspaces"
    f = ws_root / "ws1" / "orig.jpg"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"jpg")

    monkeypatch.setenv("XIMA_SERVE_FILES_NATIVE", "1")
    resp = _x_accel_file_response(
        f, workspaces_root=ws_root, media_type="image/jpeg", cache_control="no-cache"
    )
    assert isinstance(resp, FileResponse)
    assert "X-Accel-Redirect" not in resp.headers

    # 未設定（Docker/nginx 構成）では従来どおり X-Accel ヘッダを返す。
    monkeypatch.setenv("XIMA_SERVE_FILES_NATIVE", "0")
    resp2 = _x_accel_file_response(
        f, workspaces_root=ws_root, media_type="image/jpeg", cache_control="no-cache"
    )
    assert resp2.headers.get("X-Accel-Redirect") == "/_internal_ws/ws1/orig.jpg"


def test_serve_ws_file_helper(tmp_path: Path, monkeypatch) -> None:
    ws_root = tmp_path / "workspaces"
    ws_root.mkdir()
    monkeypatch.setenv("XIMA_WORKSPACES_ROOT", str(ws_root))
    cfg = ConfigManager(tmp_path / "agent")
    f = ws_root / "x.webp"
    f.write_bytes(b"w")
    resp = serve_ws_file(cfg, "x.webp")
    assert isinstance(resp, FileResponse)
