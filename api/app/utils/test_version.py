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


def _repository_paths() -> tuple[Path, Path, Path] | None:
    """monorepo の `core/VERSION` / `app/package.json` / `app/package-lock.json`。

    公開 core（subtree split）には `app/` が無いので、その場合は None を返す。
    """
    core_root = Path(version_mod.__file__).resolve().parents[3]
    app_root = core_root.parent / "app"
    package_json = app_root / "package.json"
    if not package_json.exists():
        return None
    return core_root / "VERSION", package_json, app_root / "package-lock.json"


def test_repository_version_file_matches_app_package_json() -> None:
    """core/VERSION と app/package.json のずれを検出する。

    出所を 1 つに固定しても、リリース時に片方だけ上げれば同じ問題が再発する。
    """
    import json

    paths = _repository_paths()
    if paths is None:
        return
    version_file, package_json, _ = paths

    core_version = version_file.read_text(encoding="utf-8").strip()
    app_version = json.loads(package_json.read_text(encoding="utf-8"))["version"]
    assert core_version == app_version, (
        f"core/VERSION ({core_version}) と app/package.json ({app_version}) が"
        "一致していません。リリース時は両方を上げてください。"
    )


def test_repository_version_file_matches_app_package_lock() -> None:
    """app/package-lock.json の置き去りを検出する（#169 / known-gaps G-002）。

    lock の version は依存解決に使われないため、ずれても動く。だからこそ人の目では
    見つからず、v0.3.5 の時点で 5 版ぶん放置されていた。`version` は lock の中に
    2 箇所あり、片方だけ直しても気づけないので両方を見る。

    上げる作業は monorepo の `scripts/bump-version.sh` に寄せてある（公開 core には
    無い。この検査も公開 core では対象外になる）。
    """
    import json

    paths = _repository_paths()
    if paths is None:
        return
    version_file, _, package_lock = paths

    core_version = version_file.read_text(encoding="utf-8").strip()
    lock = json.loads(package_lock.read_text(encoding="utf-8"))
    found = {
        'package-lock.json["version"]': lock["version"],
        'package-lock.json["packages"][""]["version"]': lock["packages"][""]["version"],
    }
    stale = {name: value for name, value in found.items() if value != core_version}
    assert not stale, (
        f"core/VERSION ({core_version}) と一致していません: {stale}。"
        "`./scripts/bump-version.sh <X.Y.Z>` で 3 ファイルまとめて上げてください。"
    )
