from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fcntl


_PAIRING_VERSION = 1
_OTP_RE = re.compile(r"^\d{6}$")

# Keep recently-used/expired challenges for a while so we can return
# meaningful errors (409 used / 410 expired) even after GC runs.
_GC_RETENTION_SECONDS = 3600


def pairing_path(workspaces_root: Path) -> Path:
    return (workspaces_root / ".xima" / "pairing.json").resolve()


def pairing_lock_path(workspaces_root: Path) -> Path:
    return (workspaces_root / ".xima" / "pairing.json.lock").resolve()


def normalize_otp(raw: str) -> str:
    v = (raw or "").strip().replace("-", "")
    if not _OTP_RE.match(v):
        raise ValueError("otp must be 6 digits")
    return v


def otp_hash(otp: str) -> str:
    h = hashlib.sha256(otp.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(dt: datetime) -> str:
    # ISO8601(Z) in seconds precision
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass
class PairChallenge:
    otp_hash: str
    nonce: str
    expires_at: str
    used: bool
    created_at: str

    @classmethod
    def from_dict(cls, data: dict) -> "PairChallenge":
        return cls(
            otp_hash=str(data.get("otp_hash") or ""),
            nonce=str(data.get("nonce") or ""),
            expires_at=str(data.get("expires_at") or ""),
            used=bool(data.get("used")),
            created_at=str(data.get("created_at") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "otp_hash": self.otp_hash,
            "nonce": self.nonce,
            "expires_at": self.expires_at,
            "used": self.used,
            "created_at": self.created_at,
        }


def _default_state() -> dict:
    return {"version": _PAIRING_VERSION, "active_challenge_id": None, "challenges": {}}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file on the same filesystem then replace.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp_name = f.name

    Path(tmp_name).replace(path)


class PairingStore:
    def __init__(self, workspaces_root: Path):
        self.workspaces_root = workspaces_root
        self.path = pairing_path(workspaces_root)
        self.lock_path = pairing_lock_path(workspaces_root)

    def _gc(self, state: dict, *, now: datetime) -> dict:
        challenges = state.get("challenges")
        if not isinstance(challenges, dict):
            return _default_state()

        cutoff = now - timedelta(seconds=_GC_RETENTION_SECONDS)
        out: dict[str, Any] = {}
        active_candidates: list[tuple[datetime, str]] = []
        for cid, raw in challenges.items():
            if not isinstance(cid, str) or not isinstance(raw, dict):
                continue
            ch = PairChallenge.from_dict(raw)
            try:
                exp = parse_ts(ch.expires_at)
            except Exception:
                continue
            try:
                created = parse_ts(ch.created_at)
            except Exception:
                created = now

            # Retain recent entries; drop old used/expired for file hygiene.
            if (ch.used or exp <= now) and created < cutoff:
                continue
            out[cid] = ch.to_dict()
            if not ch.used and exp > now:
                active_candidates.append((created, cid))

        active_challenge_id = state.get("active_challenge_id")
        if not isinstance(active_challenge_id, str) or active_challenge_id not in out:
            active_challenge_id = None

        if active_challenge_id is not None:
            active_raw = out.get(active_challenge_id)
            if isinstance(active_raw, dict):
                active_ch = PairChallenge.from_dict(active_raw)
                try:
                    active_exp = parse_ts(active_ch.expires_at)
                except Exception:
                    active_exp = now
                if active_ch.used or active_exp <= now:
                    active_challenge_id = None
            else:
                active_challenge_id = None

        if active_challenge_id is None and active_candidates:
            active_candidates.sort(key=lambda item: item[0], reverse=True)
            active_challenge_id = active_candidates[0][1]

        return {
            "version": _PAIRING_VERSION,
            "active_challenge_id": active_challenge_id,
            "challenges": out,
        }

    def locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = self.lock_path.open("a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f

    def load(self) -> dict:
        if not self.path.exists() or not self.path.is_file():
            return _default_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return _default_state()
        if not isinstance(data, dict):
            return _default_state()
        state = {
            "version": int(data.get("version") or _PAIRING_VERSION),
            "active_challenge_id": data.get("active_challenge_id")
            if isinstance(data.get("active_challenge_id"), str)
            else None,
            "challenges": data.get("challenges") if isinstance(data.get("challenges"), dict) else {},
        }
        return self._gc(state, now=now_utc())

    def save(self, state: dict) -> None:
        cleaned = self._gc(state, now=now_utc())
        _atomic_write_json(self.path, cleaned)

    def _latest_active_challenge_id(self, *, challenges: dict[str, Any], now: datetime) -> str | None:
        active: list[tuple[datetime, str]] = []
        for cid, raw in challenges.items():
            if not isinstance(cid, str) or not isinstance(raw, dict):
                continue
            ch = PairChallenge.from_dict(raw)
            try:
                exp = parse_ts(ch.expires_at)
                created = parse_ts(ch.created_at)
            except Exception:
                continue
            if ch.used or exp <= now:
                continue
            active.append((created, cid))
        if not active:
            return None
        active.sort(key=lambda item: item[0], reverse=True)
        return active[0][1]

    def count_active_challenges(self) -> int:
        now = now_utc()
        with self.locked() as lockf:
            _ = lockf
            state = self.load()
            challenges = state.get("challenges")
            if not isinstance(challenges, dict):
                return 0
            count = 0
            for raw in challenges.values():
                if not isinstance(raw, dict):
                    continue
                ch = PairChallenge.from_dict(raw)
                try:
                    exp = parse_ts(ch.expires_at)
                except Exception:
                    continue
                if not ch.used and exp > now:
                    count += 1
            return count

    def create_challenge(
        self,
        *,
        ttl_seconds: int = 90,
        invalidate_existing: bool = True,
    ) -> tuple[str, str, str, str]:
        challenge_id = str(uuid.uuid4())
        otp = str(secrets.randbelow(1_000_000)).zfill(6)
        nonce = b64url(secrets.token_bytes(32))
        created = now_utc()
        expires = created + timedelta(seconds=ttl_seconds)

        with self.locked() as lockf:
            _ = lockf  # keep fd open for lock lifetime
            state = self.load()
            challenges = state.setdefault("challenges", {})
            if invalidate_existing:
                for cid, raw in list(challenges.items()):
                    if not isinstance(raw, dict):
                        continue
                    ch = PairChallenge.from_dict(raw)
                    try:
                        exp = parse_ts(ch.expires_at)
                    except Exception:
                        continue
                    if ch.used or exp <= created:
                        continue
                    ch.used = True
                    challenges[cid] = ch.to_dict()
            challenges[challenge_id] = PairChallenge(
                otp_hash=otp_hash(otp),
                nonce=nonce,
                expires_at=fmt_ts(expires),
                used=False,
                created_at=fmt_ts(created),
            ).to_dict()
            state["active_challenge_id"] = challenge_id
            self.save(state)

        return challenge_id, otp, nonce, fmt_ts(expires)

    def prove(self, *, otp_raw: str, challenge_id: str | None = None) -> tuple[str, PairChallenge]:
        otp_norm = normalize_otp(otp_raw)
        hashed = otp_hash(otp_norm)
        now = now_utc()

        with self.locked() as lockf:
            _ = lockf
            if not self.path.exists():
                raise FileNotFoundError("pairing.json not found")

            state = self.load()
            challenges = state.get("challenges")
            if not isinstance(challenges, dict):
                raise KeyError("challenge not found")
            selected_id = (challenge_id or "").strip() or state.get("active_challenge_id")
            if not isinstance(selected_id, str) or selected_id not in challenges:
                selected_id = self._latest_active_challenge_id(challenges=challenges, now=now)
            if not selected_id or selected_id not in challenges:
                raise KeyError("challenge not found")

            ch = PairChallenge.from_dict(challenges[selected_id])

            exp = parse_ts(ch.expires_at)
            if exp <= now:
                raise TimeoutError("challenge expired")
            if ch.used:
                raise PermissionError("challenge already used")
            if ch.otp_hash != hashed:
                raise ValueError("invalid otp")

            # Mark used immediately and persist to prevent replay.
            ch.used = True
            challenges[selected_id] = ch.to_dict()
            if state.get("active_challenge_id") == selected_id:
                state["active_challenge_id"] = None
            self.save(state)

            return selected_id, ch
