"""`pipeline/` の規則モジュールを、app 側から名指しで 1 つだけ読む。

app と pipeline は別の process として動く（pipeline は job が subprocess で起動する）。
そのため import 経路が繋がっていないが、**規則そのものは 1 つであるべき**ものがある。

- `dataset_split` —— ラベルのある item がどちら側へ入るか（#325）
- `label_schema` —— `multi_label` の値の切り出し方（#333）

写し直すと**片方だけ直したときに数が食い違い、どちらが本当か分からなくなる**。
それは #333 で実際に起きた。

**`sys.path` には足さない。** pipeline を丸ごと import 可能にすると、`label_schema`
のような一般的な名前が app 側の import と衝突しうる。ファイルを名指しして、
`xima_pipeline_<name>` という衝突しない名前で読む。

**flat import を持つモジュールは読めない。** pipeline のモジュールは
`from label_schema import ...` のように書かれているものが多く、それらはここでは
落ちる。**落ちたら「規則の置き場を見直す合図」**であって、数え方を写し直す合図では
ない。
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"


@lru_cache(maxsize=None)
def load_pipeline_module(name: str) -> ModuleType:
    """`pipeline/<name>.py` を読み込む。読み込み結果は使い回す。"""
    path = _PIPELINE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"xima_pipeline_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 構成の破損
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
