"""`scope` の絞り込みの単体テスト（#316）。

HTTP を通さず `cluster_scope` を直接呼ぶ。ここで見るのは **誰が残るか** の定義で
あって、ルーターの配線ではない（そちらは `test_clustering_api.py`）。

`labeled_unassigned` は「ラベルは付いているのに学習に入らない」item を拾う scope で、
本番相当の experiment で 766 件（項目に値がある 1,042 件の 73.5%）がこれに当たった。
境界を間違えると救えないか、救う必要の無いものまで並ぶ。
"""

from __future__ import annotations

import json
from pathlib import Path

from app import cluster_scope
from app.config import ConfigManager

WS = "wsdemo01"
EXP = "expdemo1"


def _cfg(tmp_path: Path, items: list[dict], heads: list[dict] | None = None):
    config_manager = ConfigManager(tmp_path)
    cfg = config_manager.get_config()

    label_path = cfg.label_input_path_for(WS, EXP)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        json.dumps({"meta": {}, "items": items}, ensure_ascii=False), encoding="utf-8"
    )

    schema_path = cfg.label_schema_path_for(WS, EXP)
    schema_path.write_text(
        json.dumps(
            {
                "version": 2,
                "schema_id": "t",
                "heads": heads
                or [
                    {"id": "split", "type": "split", "choices": ["train", "val"]},
                    {"id": "shape", "type": "multi_class", "classes": ["a", "b"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return cfg


def _item(fid: str, labels: dict | None = None, *, delete: bool = False) -> dict:
    item: dict = {"file_id": fid, "path": f"a/{fid}.png"}
    if labels is not None:
        item["labels"] = labels
    if delete:
        item["delete"] = True
    return item


def _kept(cfg, items_ids: list[str], scope: str, head_id: str | None = None):
    keep = cluster_scope.keep_predicate(
        cfg, WS, EXP, scope=scope, head_id=head_id
    )
    if keep is None:
        return list(items_ids)
    return [fid for fid in items_ids if keep(fid)]


def test_labeled_unassigned_keeps_only_items_excluded_on_purpose(
    tmp_path: Path,
) -> None:
    """**人が `exclude` と書いたものだけ**を拾う（#325 で反転）。

    以前は「`split` が無い」を拾っていた。既定が反転して、書かなければ学習に
    入るようになったので、**書き忘れはもうここに出ない**。
    """
    cfg = _cfg(
        tmp_path,
        [
            _item("keep_excluded", {"shape": "a", "split": "exclude"}),
            _item("skip_no_split", {"shape": "a"}),
            _item("skip_train", {"shape": "a", "split": "train"}),
            _item("skip_val", {"shape": "a", "split": "val"}),
            _item("skip_unlabeled", {}),
        ],
    )
    kept = _kept(
        cfg,
        ["keep_excluded", "skip_no_split", "skip_train", "skip_val", "skip_unlabeled"],
        cluster_scope.SCOPE_LABELED_UNASSIGNED,
    )
    assert kept == ["keep_excluded"]


def test_labeled_unassigned_does_not_keep_legacy_unassigned(tmp_path: Path) -> None:
    """`split=unassigned` は廃止され、**キー無しと同じ（自動）**になった（#325）。

    ここに出すと「外したもの」に書き忘れが混ざる。旧値 `ignore` は逆で、語が
    「外す」を意味するので拾う。
    """
    cfg = _cfg(
        tmp_path,
        [
            _item("skip_unassigned", {"shape": "a", "split": "unassigned"}),
            _item("skip_none", {"shape": "a"}),
            _item("keep_legacy_ignore", {"shape": "a", "split": "ignore"}),
        ],
    )
    kept = _kept(
        cfg,
        ["skip_unassigned", "skip_none", "keep_legacy_ignore"],
        cluster_scope.SCOPE_LABELED_UNASSIGNED,
    )
    assert kept == ["keep_legacy_ignore"]


def test_labeled_unassigned_ignores_split_only_items(tmp_path: Path) -> None:
    """`split` だけが入っている item は「ラベル済み」ではない。

    含めると `unlabeled` とほぼ同じ集合になり、新しく作る意味が無くなる。
    """
    cfg = _cfg(
        tmp_path,
        [
            _item("split_only", {"split": "exclude"}),
            _item("empty_labels", {}),
            _item("no_labels_key"),
        ],
    )
    kept = _kept(
        cfg,
        ["split_only", "empty_labels", "no_labels_key"],
        cluster_scope.SCOPE_LABELED_UNASSIGNED,
    )
    assert kept == []


def test_labeled_unassigned_excludes_delete_marked(tmp_path: Path) -> None:
    """削除マークは除く（#293 と同じ理由）。

    消すと決めた画像は「学習へ入れられる候補」ではない。`apply_label_mapping` も
    `split` に関係なく捨てるので、ここに出しても救えない。
    """
    cfg = _cfg(
        tmp_path,
        [
            _item("marked", {"shape": "a", "split": "exclude"}, delete=True),
            _item("kept", {"shape": "a", "split": "exclude"}),
        ],
    )
    kept = _kept(
        cfg, ["marked", "kept"], cluster_scope.SCOPE_LABELED_UNASSIGNED
    )
    assert kept == ["kept"]


def test_labeled_unassigned_accepts_value_in_any_head(tmp_path: Path) -> None:
    """**いずれかの**項目に値があればよい。head は絞り込みに使わない。

    head を 1 つに限ると、別の項目だけ付けた item が漏れる。
    """
    cfg = _cfg(
        tmp_path,
        [
            _item("other_head", {"color": "red", "split": "exclude"}),
            _item("multi_label_head", {"color": ["red", "blue"], "split": "exclude"}),
        ],
        heads=[
            {"id": "split", "type": "split", "choices": ["train", "val"]},
            {"id": "shape", "type": "multi_class", "classes": ["a", "b"]},
            {"id": "color", "type": "multi_label", "classes": ["red", "blue"]},
        ],
    )
    kept = _kept(
        cfg,
        ["other_head", "multi_label_head"],
        cluster_scope.SCOPE_LABELED_UNASSIGNED,
        head_id="shape",
    )
    assert kept == ["other_head", "multi_label_head"]


def test_labeled_unassigned_ignores_empty_values(tmp_path: Path) -> None:
    """空文字・空配列は「値がある」ではない（`unlabeled` と同じ判定）。"""
    cfg = _cfg(
        tmp_path,
        [
            _item("empty_string", {"shape": "", "split": "exclude"}),
            _item("empty_list", {"color": [], "split": "exclude"}),
            _item("has_value", {"shape": "a", "split": "exclude"}),
        ],
        heads=[
            {"id": "split", "type": "split", "choices": ["train", "val"]},
            {"id": "shape", "type": "multi_class", "classes": ["a", "b"]},
            {"id": "color", "type": "multi_label", "classes": ["red", "blue"]},
        ],
    )
    kept = _kept(
        cfg,
        ["empty_string", "empty_list", "has_value"],
        cluster_scope.SCOPE_LABELED_UNASSIGNED,
    )
    assert kept == ["has_value"]


def test_labeled_unassigned_without_labels_json_keeps_nothing(tmp_path: Path) -> None:
    """labels.json が無ければ、ラベル済みの item も無い。

    `unlabeled` は「読めなければ絞らない」（安全側）だが、こちらは逆である。
    絞らずに全件返すと **1 枚もラベルが付いていない experiment で全部並ぶ**。
    """
    config_manager = ConfigManager(tmp_path)
    cfg = config_manager.get_config()
    kept = _kept(cfg, ["a", "b"], cluster_scope.SCOPE_LABELED_UNASSIGNED)
    assert kept == []


def test_all_scope_does_not_filter(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, [_item("x", {"shape": "a", "split": "train"})])
    assert (
        cluster_scope.keep_predicate(
            cfg, WS, EXP, scope=cluster_scope.SCOPE_ALL, head_id=None
        )
        is None
    )


def test_unlabeled_scope_is_unchanged(tmp_path: Path) -> None:
    """既存の `unlabeled` の挙動を変えていないことを押さえる。"""
    cfg = _cfg(
        tmp_path,
        [
            _item("labeled", {"shape": "a"}),
            _item("marked", {}, delete=True),
            _item("plain", {}),
        ],
    )
    kept = _kept(
        cfg,
        ["labeled", "marked", "plain"],
        cluster_scope.SCOPE_UNLABELED,
        head_id="shape",
    )
    assert kept == ["plain"]


def test_resolve_head_only_auto_resolves_for_unlabeled(tmp_path: Path) -> None:
    """`labeled_unassigned` は head を見ないので、core が勝手に埋めない。

    応答の `head` は「何で絞ったか」を表す。絞っていないのに値が入っていると、
    その head で絞ったように読めてしまう。
    """
    cfg = _cfg(tmp_path, [])

    assert (
        cluster_scope.resolve_head(cfg, WS, EXP, cluster_scope.SCOPE_UNLABELED, None)
        == "shape"
    )
    assert (
        cluster_scope.resolve_head(
            cfg, WS, EXP, cluster_scope.SCOPE_LABELED_UNASSIGNED, None
        )
        is None
    )
    assert cluster_scope.resolve_head(cfg, WS, EXP, cluster_scope.SCOPE_ALL, None) is None
    # 明示された head はどの scope でもそのまま返す。
    assert (
        cluster_scope.resolve_head(
            cfg, WS, EXP, cluster_scope.SCOPE_LABELED_UNASSIGNED, "color"
        )
        == "color"
    )
