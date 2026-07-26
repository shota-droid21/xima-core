from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.utils.identity import ensure_identity
from app.utils.pairing import PairingStore


def _format_otp_display(otp: str) -> str:
    v = otp.strip().replace("-", "")
    if len(v) == 6:
        return f"{v[:3]}-{v[3:]}"
    return otp


def main() -> int:
    parser = argparse.ArgumentParser(description="Start xima-agent pairing (verification code)")
    parser.add_argument(
        "--workspaces",
        type=str,
        default=os.environ.get("XIMA_WORKSPACES_ROOT", ""),
        help="Workspaces root path (default: XIMA_WORKSPACES_ROOT)",
    )
    args = parser.parse_args()

    if not args.workspaces:
        raise SystemExit("--workspaces is required (or set XIMA_WORKSPACES_ROOT)")

    ws_root = Path(args.workspaces).expanduser().resolve()
    identity = ensure_identity(ws_root)

    store = PairingStore(ws_root)
    active_before = store.count_active_challenges()
    _challenge_id, otp, _nonce, expires_at = store.create_challenge(
        ttl_seconds=90,
        invalidate_existing=True,
    )
    otp_display = _format_otp_display(otp)

    print("=== XIMA PAIRING ===")
    print(f"Verification code: {otp_display}")
    print("Copy this code and enter it in the app.")
    print("====================")
    print("")
    print("Pairing started")
    if active_before > 0:
        print(
            f"info: invalidated {active_before} pending challenge(s) and issued a new verification code"
        )
    print(f"expires_at: {expires_at}")
    print(f"install_id: {identity.install_id}")
    print(f"container_id: {identity.container_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
