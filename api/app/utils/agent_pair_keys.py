from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import fcntl


def keys_dir(workspaces_root: Path) -> Path:
    return (workspaces_root / ".xima" / "keys").resolve()


def _keys_lock_path(workspaces_root: Path) -> Path:
    return keys_dir(workspaces_root) / ".agent_ed25519.lock"


def _locked_keys(workspaces_root: Path):
    kdir = keys_dir(workspaces_root)
    kdir.mkdir(parents=True, exist_ok=True)
    lock_path = _keys_lock_path(workspaces_root)
    f = lock_path.open("a+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    return f


def _atomic_write_bytes(path: Path, data: bytes, *, chmod: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        tmp_name = f.name
    tmp_path = Path(tmp_name)
    if chmod is not None:
        try:
            tmp_path.chmod(chmod)
        except Exception:
            pass
    tmp_path.replace(path)


def private_key_path(workspaces_root: Path) -> Path:
    return keys_dir(workspaces_root) / "agent_ed25519_private.pem"


def public_key_path(workspaces_root: Path) -> Path:
    return keys_dir(workspaces_root) / "agent_ed25519_public.pem"


def ensure_keypair(workspaces_root: Path) -> tuple[str, str, str]:
    """Ensure agent pairing keypair exists.

    Returns:
        (public_pem, private_pem, fingerprint)
    """
    priv_path = private_key_path(workspaces_root)
    pub_path = public_key_path(workspaces_root)

    with _locked_keys(workspaces_root) as lockf:
        _ = lockf

        if priv_path.exists() and pub_path.exists():
            public_pem = pub_path.read_text(encoding="utf-8")
            private_pem = priv_path.read_text(encoding="utf-8")
            fp = fingerprint_public_pem(public_pem)
            return public_pem, private_pem, fp

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_pem_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        _atomic_write_bytes(priv_path, private_pem_bytes, chmod=0o600)
        _atomic_write_bytes(pub_path, public_pem_bytes)

        public_pem = public_pem_bytes.decode("utf-8")
        private_pem = private_pem_bytes.decode("utf-8")
        fp = fingerprint_public_pem(public_pem)
        return public_pem, private_pem, fp


def fingerprint_public_pem(public_pem: str) -> str:
    pub = serialization.load_pem_public_key(public_pem.encode("utf-8"))
    raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    h = hashlib.sha256(raw).hexdigest()
    return f"sha256:{h}"


def sign_payload(private_pem: str, payload_bytes: bytes) -> str:
    """Sign canonical JSON bytes, returning b64url(signature)."""
    priv = serialization.load_pem_private_key(private_pem.encode("utf-8"), password=None)
    sig = priv.sign(payload_bytes)
    import base64

    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
