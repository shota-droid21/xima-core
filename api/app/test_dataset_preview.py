"""作り直す前に出す件数（#325 A-4）。

**この検査が守るもの。**

1. **合計が合う。** 入る側と入らない側を足すと `total` になる（どこにも消えない）
2. `newly_entering` が「反転で新たに入る数」になっている
3. **規則が pipeline と同じ 1 つである**（写し直していない）
4. ルータまで配線されている
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from .config import ConfigManager
from .dataset_preview import head_ids_of, preview
from .dataset_preview_api import create_dataset_preview_router

WS = "wsprev01"
EXP = "expprev1"

SCHEMA = {
    "version": 2,
    "heads": [
        {"id": "split", "type": "split", "choices": ["train", "val", "exclude"]},
        {"id": "shape", "type": "multi_class", "classes": ["circle", "square"]},
    ],
}


def item(file_id: str, labels: dict, delete: bool = False) -> dict:
    out = {"file_id": file_id, "path": f"{file_id}.jpg", "labels": labels}
    if delete:
        out["delete"] = True
    return out


HEADS = ["shape"]


# --- 1. 合計が合う -------------------------------------------------------------


def test_every_item_lands_somewhere():
    items = [
        item("a", {"shape": "circle"}),
        item("b", {"shape": "circle", "split": "train"}),
        item("c", {"shape": "circle", "split": "exclude"}),
        item("d", {}),
        item("e", {"shape": "circle"}, delete=True),
        item("f", {"shape": "circle", "split": "nope"}),
    ]
    got = preview(items, head_ids=HEADS)
    assert got["enters"]["total"] + sum(got["out"].values()) == got["total"] == 6
    assert got["out"] == {
        "excluded": 1,
        "unlabeled": 1,
        "deleted": 1,
        "invalid_split": 1,
    }


def test_labeled_unlabeled_deleted_add_up_to_total():
    """**「ラベルが付いているか」は `split` と無関係。**

    ここを `split` の分布で代用していたのが元の間違いで、既定が反転すると
    「書いていない＝ラベルなし」になって完全に壊れる。
    """
    items = [
        item("labeled_no_split", {"shape": "circle"}),
        item("labeled_pinned", {"shape": "circle", "split": "train"}),
        item("labeled_excluded", {"shape": "circle", "split": "exclude"}),
        item("split_only", {"split": "train"}),
        item("nothing", {}),
        item("marked", {"shape": "circle"}, delete=True),
    ]
    got = preview(items, head_ids=HEADS)
    assert got["labeled"] == 3, "split だけの item はラベル済みではない"
    assert got["unlabeled"] == 2
    assert got["deleted"] == 1
    assert got["labeled"] + got["unlabeled"] + got["deleted"] == got["total"]


def test_split_source_totals_match_entering():
    items = [
        item("a", {"shape": "circle"}),
        item("b", {"shape": "circle", "split": "val"}),
    ]
    got = preview(items, head_ids=HEADS)
    assert sum(got["split_source"].values()) == got["enters"]["total"] == 2


def test_zero_items_is_all_zero():
    got = preview([], head_ids=HEADS)
    assert got["total"] == 0
    assert got["enters"]["total"] == 0
    assert got["out"] == {}
    assert got["newly_entering"] == 0


# --- 2. 差分の数 ---------------------------------------------------------------


def test_newly_entering_counts_only_what_the_old_rule_kept_out():
    """**A-4 で出す数。** 反転前は `split in (train, val)` だけが入っていた。"""
    items = [
        item("already_in", {"shape": "circle", "split": "train"}),
        item("was_out", {"shape": "circle"}),
        item("was_out_unassigned", {"shape": "circle", "split": "unassigned"}),
        item("still_out", {"shape": "circle", "split": "exclude"}),
    ]
    got = preview(items, head_ids=HEADS)
    assert got["newly_entering"] == 2
    assert got["enters"]["total"] == 3


def test_the_reversal_never_removes_anything():
    """**反転で減ることはない。** 新しい規則は古い規則の上位集合である。

    旧規則で入っていたのは `split in (train, val)` かつ削除マーク無しで、それは
    そのまま `pinned` として入る。ここが崩れたら「N 件増えます」だけを出す
    A-4 の表示が嘘になるので、網羅的に確かめる。
    """
    values = [None, "train", "val", "exclude", "unassigned", "ignore", "nope"]
    for value in values:
        for deleted in (False, True):
            for labels in ({"shape": "circle"}, {}):
                full = dict(labels)
                if value is not None:
                    full["split"] = value
                got = preview([item("a", full, delete=deleted)], head_ids=HEADS)
                entered_before = (not deleted) and value in ("train", "val")
                if entered_before:
                    assert got["enters"]["total"] == 1, (value, deleted, labels)


def test_legacy_unassigned_is_counted_until_it_is_gone():
    items = [
        item("a", {"shape": "circle", "split": "unassigned"}),
        item("b", {"shape": "circle"}),
    ]
    assert preview(items, head_ids=HEADS)["legacy_unassigned"] == 1
    assert preview([item("b", {"shape": "circle"})], head_ids=HEADS)[
        "legacy_unassigned"
    ] == 0


# --- 3. 規則は pipeline の 1 つ -------------------------------------------------


def test_rule_comes_from_the_pipeline_module():
    """写し直していないこと。**ここが落ちたら置き場を見直す合図**である。"""
    from .dataset_preview import dataset_split

    assert dataset_split.SPLIT_CHOICES == ("train", "val", "exclude")
    assert dataset_split.normalize_split("unassigned") is None


def test_val_ratio_is_honoured_and_clamped():
    items = [item(f"f{i}", {"shape": "circle"}) for i in range(400)]
    assert preview(items, head_ids=HEADS, val_ratio=0.0)["enters"]["val"] == 0
    assert preview(items, head_ids=HEADS, val_ratio=1.0)["enters"]["train"] == 0
    share = preview(items, head_ids=HEADS, val_ratio=0.25)["enters"]["val"] / 400
    assert 0.19 < share < 0.31, share


def test_pinned_items_ignore_the_ratio():
    items = [item(f"f{i}", {"shape": "circle", "split": "train"}) for i in range(50)]
    assert preview(items, head_ids=HEADS, val_ratio=1.0)["enters"]["train"] == 50


def test_head_ids_skip_the_split_head():
    assert head_ids_of(SCHEMA) == ["shape"]
    assert head_ids_of({"heads": []}) == []
    assert head_ids_of(None) == []


# --- 4. 配線 -------------------------------------------------------------------


def _client(tmp_path: Path, items: list[dict]) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    cfg = config_manager.get_config()
    label_dir = cfg.label_input_path_for(WS, EXP).parent
    label_dir.mkdir(parents=True, exist_ok=True)
    cfg.label_input_path_for(WS, EXP).write_text(
        json.dumps({"meta": {}, "items": items}, ensure_ascii=False), encoding="utf-8"
    )
    cfg.label_schema_path_for(WS, EXP).write_text(
        json.dumps(SCHEMA, ensure_ascii=False), encoding="utf-8"
    )
    app = FastAPI()
    app.include_router(create_dataset_preview_router(config_manager))
    return TestClient(app)


def test_endpoint_returns_the_counts(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        [
            item("a", {"shape": "circle"}),
            item("b", {"shape": "circle", "split": "exclude"}),
        ],
    )
    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/dataset-preview")
    assert res.status_code == 200
    body = res.json()
    assert body["workspace"] == WS
    assert body["enters"]["total"] == 1
    assert body["out"]["excluded"] == 1
    assert body["newly_entering"] == 1


def test_endpoint_accepts_val_ratio(tmp_path: Path) -> None:
    client = _client(tmp_path, [item(f"f{i}", {"shape": "circle"}) for i in range(20)])
    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/dataset-preview",
        params={"val_ratio": 1.0},
    )
    assert res.status_code == 200
    assert res.json()["enters"]["train"] == 0


def test_endpoint_rejects_a_ratio_outside_the_range(tmp_path: Path) -> None:
    client = _client(tmp_path, [])
    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/dataset-preview",
        params={"val_ratio": 2},
    )
    assert res.status_code == 422


def test_endpoint_404s_without_labels_json(tmp_path: Path) -> None:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_dataset_preview_router(config_manager))
    res = TestClient(app).get(f"/workspaces/{WS}/experiments/{EXP}/dataset-preview")
    assert res.status_code == 404
