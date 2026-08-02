"""エンタイトルメント検証（U-3 S1）の統合/単体テスト。

Ed25519 鍵ペアを生成してトークンを発行し、valid / expired / invalid / none と
GET /entitlement を検証。dependency-free（PyJWT[crypto] は base 依存）。
"""

from __future__ import annotations

import time
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.entitlement import (
    STATE_EXPIRED,
    STATE_INVALID,
    STATE_NONE,
    STATE_VALID,
    EntitlementVerifier,
    create_entitlement_router,
)


def _keypair() -> tuple[str, str]:
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pub_pem = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return priv_pem, pub_pem


def _mint(
    priv_pem: str,
    *,
    issuer: str = "xima-local",
    plan: str = "pro",
    features=("pre_label",),
    ttl: int = 3600,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "sub": "local",
            "plan": plan,
            "features": list(features),
            "iat": now,
            "exp": now + ttl,
        },
        priv_pem,
        algorithm="EdDSA",
    )


# ---- verifier 単体 ----


def test_verifier_valid() -> None:
    priv, pub = _keypair()
    token = _mint(priv)
    st = EntitlementVerifier().verify(token, public_key=pub, expected_issuer="xima-local")
    assert st.state == STATE_VALID
    assert st.plan == "pro"
    assert st.features == ["pre_label"]
    assert st.issuer == "xima-local"
    assert st.exp is not None


def test_verifier_expired() -> None:
    priv, pub = _keypair()
    token = _mint(priv, ttl=-10)  # 既に期限切れ
    st = EntitlementVerifier().verify(token, public_key=pub, expected_issuer="xima-local")
    assert st.state == STATE_EXPIRED
    assert st.exp is not None
    assert st.plan is None  # expired は plan/features を出さない


def test_verifier_wrong_key_is_invalid() -> None:
    priv, _ = _keypair()
    _, other_pub = _keypair()
    token = _mint(priv)
    st = EntitlementVerifier().verify(
        token, public_key=other_pub, expected_issuer="xima-local"
    )
    assert st.state == STATE_INVALID


def test_verifier_wrong_issuer_is_invalid() -> None:
    priv, pub = _keypair()
    token = _mint(priv, issuer="someone-else")
    st = EntitlementVerifier().verify(token, public_key=pub, expected_issuer="xima-local")
    assert st.state == STATE_INVALID


def test_verifier_tampered_is_invalid() -> None:
    priv, pub = _keypair()
    token = _mint(priv)
    tampered = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
    st = EntitlementVerifier().verify(
        tampered, public_key=pub, expected_issuer="xima-local"
    )
    assert st.state == STATE_INVALID


# ---- GET /entitlement 統合 ----


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("XIMA_AGENT_AUTH_MODE", "open")  # local key ガードを外す
    config_manager = ConfigManager(tmp_path / "agent")
    app = FastAPI()
    app.include_router(create_entitlement_router(config_manager))
    return TestClient(app)


def _place(tmp_path: Path, monkeypatch, *, token: str, pub: str) -> None:
    token_path = tmp_path / "entitlement.jwt"
    pub_path = tmp_path / "pub.pem"
    token_path.write_text(token, encoding="utf-8")
    pub_path.write_text(pub, encoding="utf-8")
    monkeypatch.setenv("XIMA_ENTITLEMENT_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("XIMA_ENTITLEMENT_PUBLIC_KEY_PATH", str(pub_path))


def test_endpoint_none_when_absent(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    res = client.get("/entitlement")
    assert res.status_code == 200
    assert res.json() == {"state": STATE_NONE}


def test_endpoint_valid(tmp_path: Path, monkeypatch) -> None:
    priv, pub = _keypair()
    _place(tmp_path, monkeypatch, token=_mint(priv), pub=pub)
    client = _client(tmp_path, monkeypatch)
    body = client.get("/entitlement").json()
    assert body["state"] == STATE_VALID
    assert body["plan"] == "pro"
    assert body["features"] == ["pre_label"]
    assert body["issuer"] == "xima-local"


def test_endpoint_expired(tmp_path: Path, monkeypatch) -> None:
    priv, pub = _keypair()
    _place(tmp_path, monkeypatch, token=_mint(priv, ttl=-10), pub=pub)
    client = _client(tmp_path, monkeypatch)
    body = client.get("/entitlement").json()
    assert body["state"] == STATE_EXPIRED


def test_endpoint_none_when_no_public_key(tmp_path: Path, monkeypatch) -> None:
    """トークンはあるが信頼公開鍵が無い → 検証不能なので none（クラッシュしない）。"""
    priv, _ = _keypair()
    token_path = tmp_path / "entitlement.jwt"
    token_path.write_text(_mint(priv), encoding="utf-8")
    monkeypatch.setenv("XIMA_ENTITLEMENT_TOKEN_PATH", str(token_path))
    monkeypatch.setenv(
        "XIMA_ENTITLEMENT_PUBLIC_KEY_PATH", str(tmp_path / "missing.pem")
    )
    client = _client(tmp_path, monkeypatch)
    assert client.get("/entitlement").json() == {"state": STATE_NONE}
