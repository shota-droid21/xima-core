"""変更のある item だけを厳格に検証して保存する（#332）。

`PUT /label-input` は**ドキュメント全体**をスキーマで検証する。そのため、触って
いない item に 1 つでもスキーマ外の値があると、**その experiment の保存が全部
400 になる**。クラスタの確定もラベリングの保存も、何を直そうとしていても止まる。

実運用で踏んだ。スキーマから消したクラスの値を持つ item が **19 / 3,489 件**残って
いたため、**無関係な 131 件の確定が通らなくなった**。

スキーマ外の値が入る経路は PUT だけではない。

- スキーマからクラスを消す・改名する（#330）
- pipeline の書き込み（`make_label_list` / `predict_labels` は PUT を通らない）
- 別マシンからの rsync（スキーマと labels.json の組が食い違う）

「壊れた値を入れない」は正しい。しかし**壊れた値がすでにあるとき**に、それと無関係な
保存まで止めるのは過剰である。そこで規則を分ける。

| item | 扱い |
| --- | --- |
| **変更がある** | 今までどおり厳格。**新しく壊れた値は入れない** |
| **変更が無い** | **そのまま残す。** 保存は止めない |

**黙って残さない。** 残った違反は件数で返し、呼び出し側が出せるようにする。

**`label_input.py` には置かない。** あちらは 900 行を超えており、保存の制御を足すと
さらに膨らむ（#306 も同じ方針）。正規化そのものは引数で受け取るので、このモジュールは
`label_input` を import しない（循環参照を作らない）。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

#: 変更の有無を見るキー。**検証が触る範囲だけ**を見る。
#: `thumb_path` のような表示用の値が変わっただけで「変更あり」にすると、
#: 保存のたびに全 item が厳格判定になり、この仕組みが意味を失う。
_COMPARED_KEYS = ("labels", "delete")

#: `items[12].labels.character must be one of ...` から head を取る。
#: 文言は `label_input.py` の `_normalize_*_label_value` が作る。
#: **形が変わったら拾えなくなる**ので、`test_label_input_partial.py` で固定している。
_HEAD_IN_MESSAGE = re.compile(r"items\[\d+\]\.labels\.([A-Za-z0-9_][A-Za-z0-9_-]*)")

#: `items[12].labels contains unknown head ids: a,b`
_UNKNOWN_HEADS_IN_MESSAGE = re.compile(
    r"items\[\d+\]\.labels contains unknown head ids: (.+)$"
)

#: head を特定できなかった違反の置き場。件数だけは失わない。
UNKNOWN_HEAD = "(不明)"

#: 1 payload に対して item ごとの検証を試みる上限。
#: 実データは 3,489 件で、これを超える規模では**そもそも全体検証が重い**。
#: 上限に達したら厳格に倒す（安全側）。
_MAX_ITEMS = 100_000


def _file_id(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    fid = item.get("file_id")
    return fid if isinstance(fid, str) and fid else None


def _comparable(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """変更の有無を見るための、比較できる形。"""
    return tuple(_freeze(item.get(key)) for key in _COMPARED_KEYS)


def _freeze(value: Any) -> Any:
    """dict / list を比較できる形へ落とす。順序の違いは変更とみなさない。"""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return ("__list__", tuple(_freeze(v) for v in value))
    return value


def unchanged_file_ids(
    new_items: List[Any], current_items: List[Any]
) -> set[str]:
    """保存しようとしている item のうち、**中身が変わっていない** file_id。

    比較するのは `labels` と `delete` だけ（`_COMPARED_KEYS`）。検証が見るのが
    そこだけだからである。
    """
    current: Dict[str, Tuple[Any, ...]] = {}
    for item in current_items:
        fid = _file_id(item)
        if fid is not None and isinstance(item, dict):
            current[fid] = _comparable(item)

    unchanged: set[str] = set()
    for item in new_items:
        fid = _file_id(item)
        if fid is None or not isinstance(item, dict):
            continue
        before = current.get(fid)
        if before is not None and before == _comparable(item):
            unchanged.add(fid)
    return unchanged


def _heads_in(message: str) -> List[str]:
    """エラーの文言から、違反した head の id を取る。"""
    unknown = _UNKNOWN_HEADS_IN_MESSAGE.search(message)
    if unknown:
        return [h.strip() for h in unknown.group(1).split(",") if h.strip()]
    found = _HEAD_IN_MESSAGE.search(message)
    if found:
        return [found.group(1)]
    return [UNKNOWN_HEAD]


Normalize = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def normalize_tolerating_unchanged(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    current_items: List[Any],
    *,
    normalize: Normalize,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """変更のある item だけ厳格に検証して正規化する。

    返すのは `(正規化した payload, head ごとの違反件数)`。

    `normalize` は `label_input.normalize_label_input_payload_with_schema` を
    想定する。**引数で受け取る**のは、このモジュールが `label_input` を import
    しないため（あちらがこちらを import する）。

    **まず全体を厳格に通す。** 通ればそこで終わりで、これまでと完全に同じ結果に
    なる。違反があるときだけ item ごとに調べる。
    """
    try:
        return normalize(payload, schema), {}
    except ValueError as first_error:
        strict_error = first_error

    items = payload.get("items")
    if not isinstance(items, list) or len(items) > _MAX_ITEMS:
        # item 単位に落とせない（payload の形が違う）か、規模が大きすぎる。
        # **判断がつかないときは厳格に倒す。**
        raise strict_error

    unchanged = unchanged_file_ids(items, current_items)
    violations: Dict[str, int] = {}
    out_items: List[Any] = []

    for index, item in enumerate(items):
        try:
            normalized = normalize({"items": [item]}, schema)["items"][0]
        except ValueError as exc:
            fid = _file_id(item)
            if fid is None or fid not in unchanged:
                # **変更のある item は今までどおり拒否する。**
                # 文言の item 番号は 1 件だけ渡したせいで 0 になっているので、
                # 本当の位置へ戻す（#331 で出すようにした理由がここ）。
                raise ValueError(
                    str(exc).replace("items[0]", f"items[{index}]", 1)
                ) from exc
            for head in _heads_in(str(exc)):
                violations[head] = violations.get(head, 0) + 1
            # 変更が無いので、**手を加えずそのまま残す**。正規化しようとすると
            # 落ちる値なので、いま在る形が唯一保てる形である。
            out_items.append(item)
            continue
        out_items.append(normalized)

    out = dict(payload)
    out["items"] = out_items
    return out, violations


def count_schema_violations(
    items: List[Any],
    schema: Dict[str, Any],
    *,
    normalize: Normalize,
) -> Tuple[Dict[str, int], int]:
    """いま在る items のうち、スキーマ外の値を持つものを数える（#332）。

    返すのは `(head ごとの件数, 違反を持つ item の数)`。

    保存とは独立に呼べる。**「保存しようとしたら分かる」では遅い**ので、
    ダッシュボードなどが先に出せるようにしておく。
    """
    heads: Dict[str, int] = {}
    bad_items = 0
    for item in items:
        try:
            normalize({"items": [item]}, schema)
        except ValueError as exc:
            bad_items += 1
            for head in _heads_in(str(exc)):
                heads[head] = heads.get(head, 0) + 1
    return heads, bad_items
