from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.auth_jwt as auth_jwt


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.setex_calls: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value
        self.setex_calls.append((key, ttl, value))

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


class _FailingRedis:
    def get(self, key: str) -> str | None:
        raise RuntimeError("redis unavailable")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise RuntimeError("redis unavailable")

    def delete(self, key: str) -> int:
        raise RuntimeError("redis unavailable")


def _setup_common(monkeypatch, now: int, *, redis_client) -> dict[str, int]:
    decode_calls = {"count": 0}
    payload = {
        "exp": now + 900,
        "iat": now,
        "iss": "xima-saas",
        "aud": "xima-agent",
        "agent_install_id": "wsi_test",
        "agent_container_runtime_id": "wsc_test",
    }

    monkeypatch.setattr(auth_jwt.time, "time", lambda: now)
    monkeypatch.setattr(auth_jwt, "_load_public_key", lambda: "public-key")
    monkeypatch.setattr(auth_jwt, "AUTH_WHITELIST_ENABLED", True)
    monkeypatch.setattr(auth_jwt, "AUTH_WHITELIST_TTL_SECONDS", 300)
    monkeypatch.setattr(auth_jwt, "_auth_whitelist_client", lambda: redis_client)

    def _fake_decode(token, key, algorithms, audience, issuer, options):  # noqa: ANN001
        decode_calls["count"] += 1
        return dict(payload)

    monkeypatch.setattr(auth_jwt.jwt, "decode", _fake_decode)
    return decode_calls


def test_require_auth_with_whitelist_uses_cache_after_first_verify(monkeypatch) -> None:
    now = 1_700_000_000
    fake_redis = _FakeRedis()
    decode_calls = _setup_common(monkeypatch, now, redis_client=fake_redis)

    authorization = "Bearer token-a"
    auth_jwt.require_auth_with_whitelist(
        authorization,
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )
    auth_jwt.require_auth_with_whitelist(
        authorization,
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )

    assert decode_calls["count"] == 1
    assert len(fake_redis.setex_calls) == 1


def test_require_auth_with_whitelist_limits_ttl_by_remaining_lifetime(monkeypatch) -> None:
    now = 1_700_000_000
    fake_redis = _FakeRedis()
    decode_calls = {"count": 0}
    payload = {
        "exp": now + 120,
        "iat": now,
        "iss": "xima-saas",
        "aud": "xima-agent",
        "agent_install_id": "wsi_test",
        "agent_container_runtime_id": "wsc_test",
    }

    monkeypatch.setattr(auth_jwt.time, "time", lambda: now)
    monkeypatch.setattr(auth_jwt, "_load_public_key", lambda: "public-key")
    monkeypatch.setattr(auth_jwt, "AUTH_WHITELIST_ENABLED", True)
    monkeypatch.setattr(auth_jwt, "AUTH_WHITELIST_TTL_SECONDS", 300)
    monkeypatch.setattr(auth_jwt, "_auth_whitelist_client", lambda: fake_redis)

    def _fake_decode(token, key, algorithms, audience, issuer, options):  # noqa: ANN001
        decode_calls["count"] += 1
        return dict(payload)

    monkeypatch.setattr(auth_jwt.jwt, "decode", _fake_decode)

    auth_jwt.require_auth_with_whitelist(
        "Bearer token-b",
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )

    assert decode_calls["count"] == 1
    assert len(fake_redis.setex_calls) == 1
    _key, ttl, _value = fake_redis.setex_calls[0]
    assert ttl == 120


def test_require_auth_with_whitelist_fallbacks_when_redis_unavailable(monkeypatch) -> None:
    now = 1_700_000_000
    failing_redis = _FailingRedis()
    decode_calls = _setup_common(monkeypatch, now, redis_client=failing_redis)

    auth_jwt.require_auth_with_whitelist(
        "Bearer token-c",
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )
    auth_jwt.require_auth_with_whitelist(
        "Bearer token-c",
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )

    # キャッシュを使えないので毎回JWT検証が走る
    assert decode_calls["count"] == 2


def test_require_auth_with_whitelist_does_not_cross_install_id_cache(monkeypatch) -> None:
    now = 1_700_000_000
    fake_redis = _FakeRedis()
    decode_calls = _setup_common(monkeypatch, now, redis_client=fake_redis)

    auth_jwt.require_auth_with_whitelist(
        "Bearer token-d",
        expected_install_id="wsi_test",
        expected_container_runtime_id="wsc_test",
    )

    with pytest.raises(HTTPException) as exc:
        auth_jwt.require_auth_with_whitelist(
            "Bearer token-d",
            expected_install_id="wsi_other",
            expected_container_runtime_id="wsc_test",
        )
    assert exc.value.status_code == 401
    assert "not bound to this agent" in str(exc.value.detail)
    assert decode_calls["count"] == 2
