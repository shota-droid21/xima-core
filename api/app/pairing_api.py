from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .auth_mode import AUTH_MODE_EXTERNAL, resolve_agent_auth_mode
from .config import ConfigManager
from .utils.agent_pair_keys import ensure_keypair, sign_payload
from .utils.identity import ensure_identity
from .utils.pairing import PairingStore


class PairProveBody(BaseModel):
    challenge_id: str | None = None
    otp: str
    client_name: str | None = None


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def create_pairing_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter(prefix="/auth/pair", tags=["auth"])

    @router.post("/prove")
    def pair_prove(
        body: PairProveBody,
    ) -> dict:
        auth_mode = resolve_agent_auth_mode()

        # openモードは完全無認証を維持（pairingは公開しない）
        if auth_mode != AUTH_MODE_EXTERNAL:
            raise HTTPException(status_code=404, detail="not found")

        # externalモード: pairingはJWTとは独立したフロー。
        # OTPでの物理操作確認→署名付きproof発行を行うため、このエンドポイント自体は無認証で呼べる必要がある。

        cfg = config_manager.get_config()
        ws_root = cfg.workspaces_root

        store = PairingStore(ws_root)
        try:
            resolved_challenge_id, ch = store.prove(
                challenge_id=body.challenge_id,
                otp_raw=body.otp,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="pairing not started")
        except KeyError:
            raise HTTPException(status_code=404, detail="challenge not found")
        except TimeoutError:
            raise HTTPException(status_code=410, detail="challenge expired")
        except PermissionError:
            raise HTTPException(status_code=409, detail="challenge already used")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        identity = ensure_identity(ws_root)
        public_pem, private_pem, fp = ensure_keypair(ws_root)

        iat = int(time.time())
        exp = iat + 90

        payload: dict[str, Any] = {
            "challenge_id": resolved_challenge_id,
            "nonce": ch.nonce,
            "iat": iat,
            "exp": exp,
            "agent_install_id": identity.install_id,
            "agent_container_id": identity.container_id,
            "agent_public_key": public_pem,
            "agent_public_key_fingerprint": fp,
            "client_name": (body.client_name or "").strip() or None,
        }

        # Remove nulls to keep canonical minimal.
        payload = {k: v for k, v in payload.items() if v is not None}

        sig = sign_payload(private_pem, _canonical_json_bytes(payload))

        # Step C+ TODO: SaaS backend should verify signature, bind agent to user, and persist.
        return {
            "proof_type": "xima-agent-pair-proof",
            "proof_version": 1,
            "payload": payload,
            "signature": sig,
        }

    return router
