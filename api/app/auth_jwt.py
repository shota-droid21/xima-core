from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict

import jwt
from fastapi import Header, HTTPException
from redis import Redis

JWT_PUBLIC_KEY_PATH = os.environ.get("XIMA_JWT_PUBLIC_KEY_PATH", "")
JWT_ALG = os.environ.get("XIMA_JWT_ALG", "EdDSA")
JWT_AUDIENCE = os.environ.get("XIMA_AGENT_JWT_AUDIENCE", "xima-agent")
JWT_ISSUER = os.environ.get("XIMA_AGENT_JWT_ISSUER", "xima-saas")


def _read_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        return default


AUTH_WHITELIST_ENABLED = (
    os.environ.get("XIMA_AGENT_AUTH_WHITELIST_ENABLED", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
AUTH_WHITELIST_TTL_SECONDS = max(
    0,
    _read_int_env("XIMA_AGENT_AUTH_WHITELIST_TTL_SECONDS", 300),
)
AUTH_WHITELIST_REDIS_URL = os.environ.get(
    "XIMA_AGENT_AUTH_WHITELIST_REDIS_URL",
    os.environ.get("XIMA_CELERY_BROKER_URL", "redis://redis:6379/0"),
)
AUTH_WHITELIST_KEY_PREFIX = os.environ.get(
    "XIMA_AGENT_AUTH_WHITELIST_KEY_PREFIX", "xima:auth:whitelist:v1"
)

# 公開鍵をロード（起動時に1度だけ読み込み）
_public_key: str | None = None
_redis_client: Redis | None = None
_redis_client_lock = threading.Lock()


def _load_public_key() -> str:
    """公開鍵を読み込む。"""
    global _public_key
    if _public_key is None:
        if not JWT_PUBLIC_KEY_PATH:
            raise RuntimeError("XIMA_JWT_PUBLIC_KEY_PATH is not set")
        if not os.path.exists(JWT_PUBLIC_KEY_PATH):
            raise RuntimeError(f"Public key file not found: {JWT_PUBLIC_KEY_PATH}")
        with open(JWT_PUBLIC_KEY_PATH, "r") as f:
            _public_key = f.read()
    return _public_key


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return token


def _decode_jwt_token(token: str) -> Dict[str, Any]:
    try:
        public_key = _load_public_key()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load public key: {str(e)}")

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[JWT_ALG],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"authentication failed: {str(e)}")

    # exp が存在し、未来であることを確認（念のため二重チェック）
    exp = payload.get("exp")
    if exp is None or int(exp) < int(time.time()):
        raise HTTPException(status_code=401, detail="token expired/invalid")

    return payload


def _auth_whitelist_enabled() -> bool:
    return (
        AUTH_WHITELIST_ENABLED
        and AUTH_WHITELIST_TTL_SECONDS > 0
        and bool((AUTH_WHITELIST_REDIS_URL or "").strip())
    )


def _auth_whitelist_client() -> Redis | None:
    if not _auth_whitelist_enabled():
        return None

    global _redis_client
    if _redis_client is not None:
        return _redis_client

    with _redis_client_lock:
        if _redis_client is None:
            _redis_client = Redis.from_url(
                AUTH_WHITELIST_REDIS_URL,
                decode_responses=True,
            )
    return _redis_client


def _auth_whitelist_key(
    token: str,
    *,
    expected_install_id: str,
    expected_container_runtime_id: str | None,
) -> str:
    raw = "|".join(
        [
            token,
            expected_install_id.strip(),
            (expected_container_runtime_id or "").strip(),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{AUTH_WHITELIST_KEY_PREFIX}:{digest}"


def _auth_whitelist_get(
    token: str,
    *,
    expected_install_id: str,
    expected_container_runtime_id: str | None,
) -> Dict[str, Any] | None:
    client = _auth_whitelist_client()
    if client is None:
        return None

    key = _auth_whitelist_key(
        token,
        expected_install_id=expected_install_id,
        expected_container_runtime_id=expected_container_runtime_id,
    )

    try:
        raw = client.get(key)
    except Exception:  # noqa: BLE001
        return None

    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        try:
            client.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return None

    try:
        exp = int(payload.get("exp", 0))
    except Exception:  # noqa: BLE001
        exp = 0
    if exp <= int(time.time()):
        try:
            client.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return None

    claim_install_id = payload.get("agent_install_id")
    if not isinstance(claim_install_id, str) or claim_install_id.strip() != expected_install_id:
        try:
            client.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return None

    claim_runtime_id = payload.get("agent_container_runtime_id")
    if claim_runtime_id is not None and expected_container_runtime_id:
        if not isinstance(claim_runtime_id, str) or claim_runtime_id.strip() != expected_container_runtime_id:
            try:
                client.delete(key)
            except Exception:  # noqa: BLE001
                pass
            return None

    return payload


def _auth_whitelist_put(
    token: str,
    payload: Dict[str, Any],
    *,
    expected_install_id: str,
    expected_container_runtime_id: str | None,
) -> None:
    client = _auth_whitelist_client()
    if client is None:
        return

    exp = payload.get("exp")
    if exp is None:
        return

    try:
        exp_unix = int(exp)
    except Exception:  # noqa: BLE001
        return

    ttl_remaining = exp_unix - int(time.time())
    if ttl_remaining <= 0:
        return

    ttl_seconds = min(AUTH_WHITELIST_TTL_SECONDS, ttl_remaining)
    if ttl_seconds <= 0:
        return

    cache_payload = {
        "exp": exp_unix,
        "agent_install_id": expected_install_id,
        "agent_container_runtime_id": (
            expected_container_runtime_id.strip()
            if isinstance(expected_container_runtime_id, str)
            and expected_container_runtime_id.strip()
            else None
        ),
    }

    key = _auth_whitelist_key(
        token,
        expected_install_id=expected_install_id,
        expected_container_runtime_id=expected_container_runtime_id,
    )

    try:
        client.setex(key, ttl_seconds, json.dumps(cache_payload, separators=(",", ":")))
    except Exception:  # noqa: BLE001
        return


def require_auth(authorization: str | None = Header(default=None)) -> Dict[str, Any]:
    """
    JWT トークンを検証し、ペイロードを返す（公開鍵方式: EdDSA）。

    対応アルゴリズム: EdDSA (Ed25519)

    Args:
        authorization: HTTP Authorization ヘッダーの値 (例: "Bearer <token>")

    Returns:
        デコードされた JWT ペイロード

    Raises:
        HTTPException: 認証失敗時 (401) または設定不備 (500)
    """
    token = _extract_bearer_token(authorization)
    return _decode_jwt_token(token)


def require_auth_with_whitelist(
    authorization: str | None,
    *,
    expected_install_id: str,
    expected_container_runtime_id: str | None = None,
) -> Dict[str, Any]:
    token = _extract_bearer_token(authorization)

    cached = _auth_whitelist_get(
        token,
        expected_install_id=expected_install_id,
        expected_container_runtime_id=expected_container_runtime_id,
    )
    if cached is not None:
        return cached

    payload = _decode_jwt_token(token)
    enforce_agent_binding(
        payload,
        expected_install_id=expected_install_id,
        expected_container_runtime_id=expected_container_runtime_id,
    )

    _auth_whitelist_put(
        token,
        payload,
        expected_install_id=expected_install_id,
        expected_container_runtime_id=expected_container_runtime_id,
    )

    return payload


def enforce_agent_binding(
    payload: Dict[str, Any],
    *,
    expected_install_id: str,
    expected_container_runtime_id: str | None = None,
) -> None:
    """
    Ensure JWT claims are bound to this agent instance.

    Required:
    - payload.agent_install_id must exist and match expected_install_id

    Optional but enforced when present:
    - payload.agent_container_runtime_id must match expected_container_runtime_id
    """
    if not expected_install_id:
        raise HTTPException(status_code=500, detail="agent identity is not initialized")

    claim_install_id = payload.get("agent_install_id")
    if not isinstance(claim_install_id, str) or not claim_install_id.strip():
        raise HTTPException(status_code=401, detail="token missing agent_install_id")
    if claim_install_id.strip() != expected_install_id:
        raise HTTPException(status_code=401, detail="token not bound to this agent")

    claim_runtime_id = payload.get("agent_container_runtime_id")
    if claim_runtime_id is None:
        return

    if not isinstance(claim_runtime_id, str) or not claim_runtime_id.strip():
        raise HTTPException(
            status_code=401,
            detail="token invalid agent_container_runtime_id",
        )
    if (
        expected_container_runtime_id
        and claim_runtime_id.strip() != expected_container_runtime_id
    ):
        raise HTTPException(
            status_code=401,
            detail="token not bound to this agent container",
        )
