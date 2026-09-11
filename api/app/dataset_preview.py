"""データセットを作り直したら何件が学習に入るかを、**実行する前に**数える（#325 A-4）。

`split` の既定が反転した（Decision 045）。書かなければ学習に入るので、**古い
experiment では「入る件数」が突然増える**。本番相当のデータでは 766 件がこれに当たる。

増えること自体は目的だが、`unassigned` には**「意図して外した」も混ざっていた**。
それは自動では拾えない（ラベリング画面の experiment で多くて 5 件）。
**この Issue で唯一失いうるものがそれで、件数を先に出すことが唯一の手当てである。**
数が見えていれば、その数件を `exclude` へ付け直せる。

**規則は `pipeline/dataset_split.py` から借りる。** ここで数え方を書き直すと、
`apply_label_mapping` が実際に作る中身と食い違い、**先に出した数字が嘘になる**。
process が分かれているだけで、規則は 1 つである（#333 と同じ立場）。

`sys.path` は触らない。pipeline を丸ごと import 可能にすると、`label_schema` の
ような名前が app 側の import と衝突しうる。**ファイルを名指しで 1 つだけ**読む。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"


def _load_dataset_split():
    """`pipeline/dataset_split.py` を、sys.path を汚さずに読み込む。

    このモジュールが flat import（`from label_schema import ...`）を持つように
    なったらここで落ちる。**そのときは規則の置き場を見直す合図**で、黙って
    数え方を写し直してはいけない。
    """
    path = _PIPELINE_DIR / "dataset_split.py"
    spec = importlib.util.spec_from_file_location("xima_dataset_split", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 構成の破損
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_split = _load_dataset_split()

#: 反転する**前**に学習へ入っていた `split`。差分を出すためだけに要る。
_OLD_RULE_SPLITS = ("train", "val")


def _labels(item: Any) -> Dict[str, Any]:
    labels = item.get("labels") if isinstance(item, dict) else None
    return labels if isinstance(labels, dict) else {}


def _entered_before(labels: Dict[str, Any], deleted: bool) -> bool:
    """反転する前の規則で学習に入っていたか（`split in (train, val)`）。"""
    if deleted:
        return False
    value = labels.get("split")
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _OLD_RULE_SPLITS


def _deleted(item: Any) -> bool:
    return isinstance(item, dict) and item.get("delete") is True


def _key(item: Dict[str, Any]) -> str:
    return str(item.get("file_id") or item.get("path") or item.get("id") or "")


def head_ids_of(schema: Any) -> List[str]:
    """スキーマの head id。`split` は学習する head ではないので除く。"""
    out: List[str] = []
    for head in (schema or {}).get("heads") or []:
        if not isinstance(head, dict):
            continue
        hid = str(head.get("id") or "").strip()
        if hid and hid != "split" and str(head.get("type") or "") != "split":
            out.append(hid)
    return out


def preview(
    items: List[Any],
    *,
    head_ids: List[str],
    val_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """作り直したらどうなるかの件数。

    `enters` が入る側、`out` が入らない側で、**合計は必ず `total` になる**。
    `newly_entering` が A-4 で出す数 —— 反転前は入らず、いまは入るもの。
    """
    ratio = dataset_split.DEFAULT_VAL_RATIO if val_ratio is None else val_ratio
    ratio = min(max(float(ratio), 0.0), 1.0)

    enters = {"train": 0, "val": 0}
    source = {"auto": 0, "pinned": 0}
    out: Dict[str, int] = {}
    newly_entering = 0
    legacy_unassigned = 0
    labeled = 0
    unlabeled = 0
    deleted_count = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        labels = _labels(item)
        deleted = _deleted(item)
        before = _entered_before(labels, deleted)

        # **「ラベルが付いているか」は `split` と無関係である。** ここを split の
        # 分布で代用していたのが元の間違いで、既定が反転すると完全に壊れる
        # （書かない item が増えるほど「ラベルなし」が増えてしまう）。
        if deleted:
            deleted_count += 1
        elif dataset_split.has_any_label(labels, head_ids):
            labeled += 1
        else:
            unlabeled += 1

        raw_split = labels.get("split")
        if isinstance(raw_split, str) and raw_split.strip().lower() == "unassigned":
            legacy_unassigned += 1

        split, reason = dataset_split.resolve_split(
            labels,
            deleted=deleted,
            key=_key(item),
            head_ids=head_ids,
            val_ratio=ratio,
        )
        if split is None:
            out[reason] = out.get(reason, 0) + 1
            continue
        enters[split] += 1
        source[reason] = source.get(reason, 0) + 1
        if not before:
            newly_entering += 1

    return {
        "total": len(items),
        "val_ratio": ratio,
        # 削除マークを別に数えるので、**3 つ足すと total になる**。
        "labeled": labeled,
        "unlabeled": unlabeled,
        "deleted": deleted_count,
        "enters": {**enters, "total": enters["train"] + enters["val"]},
        "split_source": source,
        "out": out,
        # A-4 で出す数。**反転で減ることはない**（新しい規則は古い規則の上位集合で、
        # 旧規則で入っていた `train` / `val` はそのまま pinned として入る）。
        # だから「減る数」は出さない —— 常に 0 の欄は読む側を迷わせる。
        "newly_entering": newly_entering,
        # 移行のあいだだけ意味のある数。0 になったら移行が終わっている。
        "legacy_unassigned": legacy_unassigned,
    }
