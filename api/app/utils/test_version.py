"""`read_core_version` の検証。

版が読めないときに古い数字へフォールバックしないこと（＝嘘の版を返さないこと）が
本 Issue の主眼なので、そこを重点的に見る（#174）。
"""

from __future__ import annotations

from pathlib import Path

from app.utils import version as version_mod


def _with_version_file(monkeypatch, tmp_path: Path, content: str | None) -> None:
    target = tmp_path / "VERSION"
    if content is not None:
        target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(version_mod, "_VERSION_FILE", target)


def test_reads_version_file(monkeypatch, tmp_path: Path) -> None:
    _with_version_file(monkeypatch, tmp_path, "0.1.2\n")
    assert version_mod.read_core_version() == "0.1.2"


def test_strips_surrounding_whitespace(monkeypatch, tmp_path: Path) -> None:
    _with_version_file(monkeypatch, tmp_path, "  0.2.0  \n\n")
    assert version_mod.read_core_version() == "0.2.0"


def test_missing_file_returns_unknown(monkeypatch, tmp_path: Path) -> None:
    _with_version_file(monkeypatch, tmp_path, None)
    assert version_mod.read_core_version() == version_mod.UNKNOWN_VERSION


def test_empty_file_returns_unknown(monkeypatch, tmp_path: Path) -> None:
    _with_version_file(monkeypatch, tmp_path, "\n  \n")
    assert version_mod.read_core_version() == version_mod.UNKNOWN_VERSION


def test_repository_version_file_matches_app_package_json() -> None:
    """core/VERSION と app/package.json のずれを検出する。

    出所を 1 つに固定しても、リリース時に片方だけ上げれば同じ問題が再発する。
    """
    import json

    core_root = Path(version_mod.__file__).resolve().parents[3]
    package_json = core_root.parent / "app" / "package.json"
    if not package_json.exists():
        # 公開 core（subtree split）には app/ が無い。そちらでは検査対象外。
        return

    core_version = (core_root / "VERSION").read_text(encoding="utf-8").strip()
    app_version = json.loads(package_json.read_text(encoding="utf-8"))["version"]
    assert core_version == app_version, (
        f"core/VERSION ({core_version}) と app/package.json ({app_version}) が"
        "一致していません。リリース時は両方を上げてください。"
    )
