"""ラベルのある item がデータセットのどちら側へ入るかを決める（#325 / Decision 045）。

**既定は「入る」。** ラベルがあれば、`split` を書かなくても学習・検証に入る。
これまでは逆で、`split` が `train` / `val` のときだけ入っていた。その既定のせいで、
クラスタから付けた **1,042 件のうち 766 件（73.5%）が学習に入っていなかった**（#316）。

| `split` の値 | 扱い | 誰が書くか |
| --- | --- | --- |
| **キー無し / `None`** | **自動**。ラベルがあれば入る。train / val はここで決める | 誰も書かない |
| `train` / `val` | そこに固定する（`pinned`） | 人 |
| `exclude` | 学習から外す | 人 |

`unassigned` は**廃止した**。読むときは「キー無し」と同じに扱う（Decision 045-3）。
旧値 `ignore` は語が「外す」を意味するので `exclude` として読む —— 自動へ寄せると
**外したつもりのものが黙って学習に入る**。

**val はハッシュで決める。** 比率で上から切ると、データが増えたときに既存 item の
所属が変わり、前の学習と比べられなくなる。キーのハッシュで決めれば、**何度回しても、
何件増えても、同じ item は同じ側に居る**。

**ここを 1 つにしておく理由。** 同じ判定が `apply_label_mapping`（データセットを作る）と
移行前の件数照会（「N 件が新たに学習に入ります」）の両方に要る。別々に書くと、
**片方だけ直したときに数字が食い違い、どちらが本当か分からなくなる**（#333 と同じ轍）。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Optional, Tuple

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_EXCLUDE = "exclude"

#: 人が書ける値。`label_input._SPLIT_CHOICES` と対。
SPLIT_CHOICES = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_EXCLUDE)

#: 旧い値の読み替え。`None` は「キー無しと同じ（自動）」。
#: NOTE: labels.json から旧値が消えたら落とす。
LEGACY_SPLIT_ALIASES: Dict[str, Optional[str]] = {
    "unassigned": None,       # 「未決定」と「意図して外した」が混ざっていた。自動へ寄せる
    "ignore": SPLIT_EXCLUDE,  # 語が「外す」を意味する。自動へ寄せない
    "delete": None,           # 意味は削除マークが担う
}

#: `split_source`。どちらの経路で決まったか。
SOURCE_AUTO = "auto"
SOURCE_PINNED = "pinned"

#: 比率の分解能。`0.2` なら 2000。
_RESOLUTION = 10_000

DEFAULT_VAL_RATIO = 0.2

#: 「値が無い」の判定。`cluster_scope._EMPTY_VALUES` と同じ。
_EMPTY_VALUES = (None, "", [], {})


def normalize_split(value: Any) -> Optional[str]:
    """`split` の値を `SPLIT_CHOICES` か `None`（自動）へ落とす。

    知らない値は `None` にしない。**そのまま返して、呼び出し側が気づけるようにする。**
    黙って自動へ寄せると、スキーマ外の値が学習に入ってしまう。
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip().lower()
    if not text:
        return None
    if text in LEGACY_SPLIT_ALIASES:
        return LEGACY_SPLIT_ALIASES[text]
    return text


def has_any_label(labels: Any, head_ids: Iterable[str]) -> bool:
    """学習に使える値が 1 つでもあるか。

    `split` は学習する head ではないので数えない（Decision 011）。`head_ids` が
    空のとき（スキーマが無い）は、**`split` 以外のキーに値があるか**で見る。
    """
    if not isinstance(labels, dict):
        return False
    ids = [h for h in head_ids if h and h != "split"]
    if not ids:
        ids = [k for k in labels.keys() if k != "split"]
    return any(labels.get(hid) not in _EMPTY_VALUES for hid in ids)


def auto_side(key: str, val_ratio: float) -> str:
    """自動割りで train / val のどちらへ入れるか。

    `key` は `file_id`（無ければ元パス）。**同じ key なら常に同じ側**に入る。
    """
    if val_ratio <= 0:
        return SPLIT_TRAIN
    if val_ratio >= 1:
        return SPLIT_VAL
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _RESOLUTION
    return SPLIT_VAL if bucket < round(val_ratio * _RESOLUTION) else SPLIT_TRAIN


def resolve_split(
    labels: Any,
    *,
    deleted: bool,
    key: str,
    head_ids: Iterable[str],
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Tuple[Optional[str], str]:
    """この item がどちら側へ入るか。

    返すのは `(split, split_source)`。**入らないときは `(None, 理由)`** で、
    理由は `deleted` / `excluded` / `unlabeled` / `invalid_split` のいずれか。
    """
    if deleted:
        return None, "deleted"

    split = normalize_split(labels.get("split") if isinstance(labels, dict) else None)

    if split in (SPLIT_TRAIN, SPLIT_VAL):
        return split, SOURCE_PINNED
    if split == SPLIT_EXCLUDE:
        return None, "excluded"
    if split is not None:
        # スキーマ外の値。黙って学習へ入れない（#330 と同じ立場）。
        return None, "invalid_split"

    if not has_any_label(labels, head_ids):
        return None, "unlabeled"
    return auto_side(key, val_ratio), SOURCE_AUTO
