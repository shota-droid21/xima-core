"""pytest 共通設定。

FastAPI ルート統合テストは `ConfigManager(tmp_path)` を組み立てた上で、
`tmp_path` 直下に workspace ディレクトリを作成する前提で書かれている。

一方 `ConfigManager` は既定で `base_dir / "workspaces"` を workspaces_root と
するため、`XIMA_WORKSPACES_ROOT` を明示しないとテストの想定パスと一致せず、
ルートが 404 を返す（本番は docker-compose 側で env を注入している）。

各テストの tmp_path を workspaces_root として明示することで、テストの前提と
実装の解決規則を一致させる。env は monkeypatch 経由なのでテストごとに自動復元される。
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _workspaces_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XIMA_WORKSPACES_ROOT", str(tmp_path))
