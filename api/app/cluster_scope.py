"""クラスタ API の `scope` の解決と絞り込み（#293 / #316）。

`scope` は「どの画像をクラスタリングの対象にするか」を決める。3 つある。

| scope | 対象 |
| --- | --- |
| `all` | 全部。labels.json を読まない |
| `unlabeled` | 指定 head にまだ値が無いもの（削除マークは除く） |
| `labeled_unassigned` | **いずれかの項目に値があり、学習に入っていないもの** |

`labeled_unassigned` が #316 で足りないと分かったもの。ラベルを付けても `split` が
`train` / `val` でなければデータセットに入らない（`apply_label_mapping.py:183`）が、
`split` を付ける機会が付与の流れの中に無かった。クラスタのまとめ付与は項目だけを書く
（#293 で意図的にそうした）ため、本番相当の experiment では**項目に値がある 1,042 件の
うち 766 件（73.5%）が学習に入っていなかった**。この scope はその 766 件を拾う。

**ルータから分けている理由。** `clustering.py` は埋め込みの読み込みと応答の整形が本体で、
そこに labels.json の解釈が混ざると、どちらを直しているのか分からなくなる。scope が
増えるのは今回が初めてではない（#293 で削除マークの条件が入った）。

**labels.json は 1 度しか読まない。** scope ごとに読み分けると同じファイルを 2 回読む
ことになる。トップレベルの `split` / `label`（`make_label_list` が作る旧形式）は見ない。
学習に入るかを決めている `apply_label_mapping` が `item["labels"]` しか見ておらず、
**そちらが正**だからである（旧形式は PUT のときに `labels` へ正規化される）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set

from .label_input import (
    load_label_json,
    normalize_label_schema_payload,
)

SCOPE_ALL = "all"
SCOPE_UNLABELED = "unlabeled"
SCOPE_LABELED_UNASSIGNED = "labeled_unassigned"

#: `scope` に指定できる値。API の 400 メッセージもここから作る。
SCOPES = (SCOPE_ALL, SCOPE_UNLABELED, SCOPE_LABELED_UNASSIGNED)

SPLIT_HEAD_ID = "split"

# 学習・検証データに入る `split`。`apply_label_mapping.py:183` と同じ境界。
# ここから外れるのは `unassigned` と、値が無いもの（`None`）。
_IN_DATASET_SPLITS = ("train", "val")

# 「値が無い」の判定。`_unlabeled_exclusions`（#293）から引き継いでいる。
_EMPTY_VALUES = (None, "", [], {})


def _has_value(value: Any) -> bool:
    return value not in _EMPTY_VALUES


def _load_items(cfg, workspace: str, experiment: str) -> list[Dict[str, Any]]:
    """labels.json の items。読めなければ空（＝絞り込みの材料が無い）。"""
    try:
        label_path = cfg.label_input_path_for(workspace, experiment)
        data = load_label_json(label_path)
    except (FileNotFoundError, ValueError):
        return []
    return [it for it in (data.get("items") or []) if isinstance(it, dict)]


def _file_id(item: Dict[str, Any]) -> Optional[str]:
    fid = item.get("file_id")
    return fid if isinstance(fid, str) else None


def _labels(item: Dict[str, Any]) -> Dict[str, Any]:
    labels = item.get("labels")
    return labels if isinstance(labels, dict) else {}


def load_schema(cfg, workspace: str, experiment: str) -> Dict[str, Any]:
    try:
        schema_path = cfg.label_schema_path_for(workspace, experiment)
        return normalize_label_schema_payload(load_label_json(schema_path))
    except (FileNotFoundError, ValueError):
        return {"heads": []}


def default_head_id(schema: Dict[str, Any]) -> Optional[str]:
    """split 以外で最初に見つかった分類 head の id。"""
    for head in schema.get("heads") or []:
        if not isinstance(head, dict):
            continue
        hid = str(head.get("id") or "").strip()
        if hid and hid != SPLIT_HEAD_ID:
            return hid
    return None


def resolve_head(
    cfg, workspace: str, experiment: str, scope: str, requested: Optional[str]
) -> Optional[str]:
    """その scope が使う head id。

    自動解決するのは `unlabeled` だけ。**`labeled_unassigned` は head を見ない**
    （「いずれかの項目に値がある」が条件のため）ので、要求された値をそのまま返す。
    `all` も同じ。応答の `head` は「何で絞ったか」を表すので、絞っていないなら
    core が勝手に埋めない。
    """
    if requested:
        return requested
    if scope == SCOPE_UNLABELED:
        return default_head_id(load_schema(cfg, workspace, experiment))
    return requested


def _unlabeled_exclusions(items: list[Dict[str, Any]], head_id: str) -> Set[str]:
    """`scope=unlabeled` で除く file_id の集合。

    除く理由は 2 つある。

    1. その head で **既にラベルが付いている**
    2. **削除マークが付いている**（#293）

    2 は以前は見ていなかった。`delete` は item 直下のフラグで `labels` とは別の
    レイヤーにあり、ここが `labels[head]` しか見ていなかったためである。その結果
    **消すと決めた画像が「残りに付ける対象」として出続けていた**。筋が通らないうえ、
    重複を削除マークで片付けても視界から消えないので、まとまりを 1 つずつ処理する
    使い方が成立しなかった。
    """
    excluded: Set[str] = set()
    for item in items:
        fid = _file_id(item)
        if fid is None:
            continue
        if item.get("delete") is True:
            excluded.add(fid)
            continue
        if _has_value(_labels(item).get(head_id)):
            excluded.add(fid)
    return excluded


def _labeled_unassigned_ids(items: list[Dict[str, Any]]) -> Set[str]:
    """`scope=labeled_unassigned` で残す file_id の集合。

    残すのは次を**すべて**満たすもの。

    1. `split` 以外のいずれかの head に値がある（＝ラベル済み）
    2. `split` が `train` でも `val` でもない（＝学習に入っていない）
    3. 削除マークが付いていない

    3 を条件に入れるのは #293 と同じ理由である。**消すと決めた画像は「学習へ入れ
    られる候補」ではない。** 削除マークが付いていれば `apply_label_mapping` も
    `split` に関係なく捨てるので、ここに出しても救えない。

    1 で `split` を除くのは、`split` だけが付いた item を「ラベル済み」と呼べない
    ため。`split=unassigned` は**値が入っているが項目は空**という状態で、これを
    含めると `unlabeled` とほぼ同義の集合になる。
    """
    keep: Set[str] = set()
    for item in items:
        fid = _file_id(item)
        if fid is None:
            continue
        if item.get("delete") is True:
            continue

        labels = _labels(item)
        split = labels.get(SPLIT_HEAD_ID)
        if isinstance(split, str) and split.strip().lower() in _IN_DATASET_SPLITS:
            continue

        has_label = any(
            _has_value(value)
            for hid, value in labels.items()
            if hid != SPLIT_HEAD_ID
        )
        if has_label:
            keep.add(fid)
    return keep


def keep_predicate(
    cfg,
    workspace: str,
    experiment: str,
    *,
    scope: str,
    head_id: Optional[str],
) -> Optional[Callable[[str], bool]]:
    """file_id を受け取り「残すか」を返す関数。絞り込まないときは None。

    None を返すのは 2 つの場合。

    - `scope=all`。labels.json を読む必要が無い
    - `scope=unlabeled` で head が決まらない。**何で絞るかが無いまま全部落とすより、
      絞らずに返す方が安全**である（従来の挙動）
    """
    if scope == SCOPE_UNLABELED:
        if not head_id:
            return None
        excluded = _unlabeled_exclusions(
            _load_items(cfg, workspace, experiment), head_id
        )
        return lambda fid: fid not in excluded

    if scope == SCOPE_LABELED_UNASSIGNED:
        keep = _labeled_unassigned_ids(_load_items(cfg, workspace, experiment))
        return lambda fid: fid in keep

    return None
