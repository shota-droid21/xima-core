"""実行中の core のバージョンを返す。

以前は `main.py` に `FastAPI(..., version="0.1.0")` とハードコードされており、
v0.1.1 / v0.1.2 を発行しても `/health` は `0.1.0` を返し続けていた。利用者は
自分が何で動いているのかを知る手段が無かった（#174）。

出所は `core/VERSION` の 1 つに固定する。公開 core（subtree split）ではこの
ファイルがリポジトリ直下に来るため、monorepo と公開側で同じ経路になる。
`app/package.json` と一致していることは CI で検査する。
"""

from __future__ import annotations

from pathlib import Path

# api/app/utils/version.py -> utils -> app -> api -> core
_VERSION_FILE = Path(__file__).resolve().parents[3] / "VERSION"

# 読めなかったときに古い数字を返すと「嘘の版」になる。分からないことを
# そのまま示す方が、利用者にとっても報告を受ける側にとっても安全。
UNKNOWN_VERSION = "unknown"


def read_core_version() -> str:
    try:
        value = _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return UNKNOWN_VERSION
    return value or UNKNOWN_VERSION
