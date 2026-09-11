"""スキーマ変更で行き先を失う値と、その付け替え（#330）。

**この検査が守るもの。**

1. 行き先を失う 3 つの経路（クラス削除 / head 削除 / 型変更）を拾う
2. **いま学習に入っている数**が分かる（直す優先度がここで決まる）
3. 付け替えは `multi_label` でも壊れない。空の `to` は値を消す
4. **行き先の無い付け替えを受け付けない**（#330 の状態を作り直さない）
5. 書く前に履歴へ残す
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from .config import ConfigManager
from .schema_impact import (
    REASON_CLASS_REMOVED,
    REASON_HEAD_REMOVED,
    REASON_TYPE_CHANGED,
    apply_remap,
    orphans,
)
from .schema_impact_api import create_schema_impact_router

WS = "wsimpact"
EXP = "expimpct"


def schema(*heads) -> dict:
    return {"version": 2, "heads": list(heads)}


MULTI_CLASS = {"id": "shape", "type": "multi_class", "classes": ["circle", "square"]}
MULTI_LABEL = {"id": "tone", "type": "multi_label", "classes": ["warm", "cool"]}
SPLIT = {"id": "split", "type": "split", "choices": ["train", "val", "exclude"]}


def item(file_id: str, labels: dict, delete: bool = False) -> dict:
    out = {"file_id": file_id, "path": f"{file_id}.jpg", "labels": labels}
    if delete:
        out["delete"] = True
    return out


def with_classes(head: dict, classes: list[str]) -> dict:
    return {**head, "classes": classes}


# --- 1. 3 つの経路 -------------------------------------------------------------


def test_removing_a_class_is_found():
    old = schema(with_classes(MULTI_CLASS, ["circle", "square", "gone"]))
    new = schema(MULTI_CLASS)
    got = orphans([item("a", {"shape": "gone"})], new, old_schema=old)
    assert got["items"] == 1
    head = got["heads"][0]
    assert head["head"] == "shape"
    assert head["reason"] == REASON_CLASS_REMOVED
    assert head["values"] == [{"value": "gone", "items": 1, "entering": 1}]
    # 付け替え先の候補は、**新しいスキーマに残っているクラス**である。
    assert head["targets"] == ["circle", "square"]


def test_removing_a_head_is_found():
    old = schema(MULTI_CLASS, MULTI_LABEL)
    new = schema(MULTI_CLASS)
    got = orphans([item("a", {"tone": ["warm", "cool"]})], new, old_schema=old)
    head = got["heads"][0]
    assert head["reason"] == REASON_HEAD_REMOVED
    assert [v["value"] for v in head["values"]] == ["cool", "warm"]
    # head ごと消えるので、付け替え先は無い（消すしかない）。
    assert head["targets"] == []


def test_narrowing_multi_label_to_multi_class_is_found():
    """**値が 2 つ以上ある item は表せない。** どれを残すかは人が決めるしかない。"""
    old = schema(MULTI_LABEL)
    new = schema({"id": "tone", "type": "multi_class", "classes": ["warm", "cool"]})
    got = orphans(
        [item("a", {"tone": ["warm", "cool"]}), item("b", {"tone": ["warm"]})],
        new,
        old_schema=old,
    )
    head = got["heads"][0]
    assert head["reason"] == REASON_TYPE_CHANGED
    assert head["too_many_values"] == 1
    assert got["items"] == 1, "1 つだけの item は収まるので対象外"


def test_nothing_lost_means_nothing_reported():
    old = schema(MULTI_CLASS, MULTI_LABEL)
    got = orphans(
        [item("a", {"shape": "circle", "tone": ["warm"]})], old, old_schema=old
    )
    assert got == {"heads": [], "items": 0, "entering": 0}


def test_adding_a_class_loses_nothing():
    old = schema(MULTI_CLASS)
    new = schema(with_classes(MULTI_CLASS, ["circle", "square", "triangle"]))
    assert orphans([item("a", {"shape": "circle"})], new, old_schema=old)["items"] == 0


def test_split_is_never_reported():
    """`split` の値の正否は #325 の規則が決める。ここでは見ない。"""
    old = schema(SPLIT, MULTI_CLASS)
    new = schema(MULTI_CLASS)
    got = orphans([item("a", {"shape": "circle", "split": "train"})], new, old_schema=old)
    assert got["items"] == 0


def test_head_without_classes_loses_nothing():
    """クラスを持たない head は、学習側が items から集める。**無い値が存在しない。**"""
    old = schema({"id": "free", "type": "multi_class"})
    new = schema({"id": "free", "type": "multi_class"})
    assert orphans([item("a", {"free": "anything"})], new, old_schema=old)["items"] == 0


# --- 2. いま学習に入っているか -------------------------------------------------


def test_entering_counts_only_items_in_the_dataset():
    """外してある item は、いま直さなくても実害が無い。**優先度がこれで決まる。**"""
    old = schema(SPLIT, with_classes(MULTI_CLASS, ["circle", "gone"]))
    new = schema(SPLIT, MULTI_CLASS)
    items = [
        item("in", {"shape": "gone"}),
        item("excluded", {"shape": "gone", "split": "exclude"}),
        item("marked", {"shape": "gone"}, delete=True),
    ]
    got = orphans(items, new, old_schema=old)
    assert got["items"] == 3
    assert got["entering"] == 1
    assert got["heads"][0]["values"][0] == {"value": "gone", "items": 3, "entering": 1}


# --- 3. 付け替え ---------------------------------------------------------------


def test_remap_replaces_a_scalar_value():
    items = [item("a", {"shape": "gone"}), item("b", {"shape": "circle"})]
    out, changed = apply_remap(items, [{"head": "shape", "from": "gone", "to": "circle"}])
    assert changed == {"shape": 1}
    assert [i["labels"]["shape"] for i in out] == ["circle", "circle"]


def test_remap_with_empty_target_removes_the_key():
    items = [item("a", {"shape": "gone", "tone": ["warm"]})]
    out, changed = apply_remap(items, [{"head": "shape", "from": "gone", "to": ""}])
    assert changed == {"shape": 1}
    assert out[0]["labels"] == {"tone": ["warm"]}, "key ごと消す"


def test_remap_on_multi_label_keeps_the_other_values():
    items = [item("a", {"tone": ["warm", "old", "cool"]})]
    out, _ = apply_remap(items, [{"head": "tone", "from": "old", "to": ""}])
    assert out[0]["labels"]["tone"] == ["warm", "cool"]


def test_remap_on_multi_label_does_not_duplicate():
    """付け替え先が既に入っているとき、**同じ値を 2 つにしない。**"""
    items = [item("a", {"tone": ["warm", "old"]})]
    out, _ = apply_remap(items, [{"head": "tone", "from": "old", "to": "warm"}])
    assert out[0]["labels"]["tone"] == ["warm"]


def test_remap_removes_the_key_when_nothing_is_left():
    items = [item("a", {"tone": ["old"]})]
    out, _ = apply_remap(items, [{"head": "tone", "from": "old", "to": ""}])
    assert "tone" not in out[0]["labels"]


def test_remap_does_not_touch_items_without_that_value():
    items = [item("a", {"shape": "circle"})]
    out, changed = apply_remap(items, [{"head": "shape", "from": "gone", "to": "square"}])
    assert changed == {}
    assert out[0] is items[0], "手を加えていない item はそのまま返す"


def test_remap_is_idempotent():
    """**同じ付け替えを 2 回当てても結果が変わらない。**

    `from` で選ぶので、1 回目のあとはもう合致する item が無い。通信がやり直された
    ときに二重に当たらない、という保証でもある。
    """
    remaps = [{"head": "shape", "from": "gone", "to": "circle"}]
    once, first = apply_remap([item("a", {"shape": "gone"})], remaps)
    twice, second = apply_remap(once, remaps)
    assert first == {"shape": 1}
    assert second == {}
    assert twice[0]["labels"] == once[0]["labels"]


def test_remap_never_touches_split():
    items = [item("a", {"split": "train"})]
    out, changed = apply_remap(items, [{"head": "split", "from": "train", "to": "val"}])
    assert changed == {}
    assert out[0]["labels"]["split"] == "train"


def test_remap_copies_items_so_the_caller_can_roll_back():
    items = [item("a", {"shape": "gone"})]
    out, _ = apply_remap(items, [{"head": "shape", "from": "gone", "to": "circle"}])
    assert items[0]["labels"]["shape"] == "gone", "元の items を書き換えない"
    assert out[0]["labels"]["shape"] == "circle"


# --- 4 / 5. 配線・拒否・履歴 ---------------------------------------------------


def _client(tmp_path: Path, items: list[dict], heads: list[dict]):
    config_manager = ConfigManager(tmp_path)
    cfg = config_manager.get_config()
    label_path = cfg.label_input_path_for(WS, EXP)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        json.dumps({"meta": {}, "items": items}, ensure_ascii=False), encoding="utf-8"
    )
    cfg.label_schema_path_for(WS, EXP).write_text(
        json.dumps(schema(*heads), ensure_ascii=False), encoding="utf-8"
    )
    app = FastAPI()
    app.include_router(create_schema_impact_router(config_manager))
    return TestClient(app), label_path


def test_impact_endpoint_writes_nothing(tmp_path: Path) -> None:
    client, label_path = _client(
        tmp_path,
        [item("a", {"shape": "gone"})],
        [with_classes(MULTI_CLASS, ["circle", "gone"])],
    )
    before = label_path.read_text(encoding="utf-8")
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/impact",
        json=schema(MULTI_CLASS),
    )
    assert res.status_code == 200
    assert res.json()["items"] == 1
    assert label_path.read_text(encoding="utf-8") == before, "照会は何も書かない"


def test_remap_endpoint_writes_and_keeps_history(tmp_path: Path) -> None:
    client, label_path = _client(
        tmp_path, [item("a", {"shape": "gone"})], [MULTI_CLASS]
    )
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/remap",
        json={"remaps": [{"head": "shape", "from": "gone", "to": "circle"}]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "saved"
    assert body["changed"] == {"shape": 1}

    saved = json.loads(label_path.read_text(encoding="utf-8"))
    assert saved["items"][0]["labels"]["shape"] == "circle"

    history = sorted((label_path.parent / "history").glob("labels_*.json"))
    assert len(history) == 1, "書く前の姿が残っている"
    kept = json.loads(history[0].read_text(encoding="utf-8"))
    assert kept["items"][0]["labels"]["shape"] == "gone"


def test_remap_rejects_a_target_that_does_not_exist(tmp_path: Path) -> None:
    """**行き先の無い付け替えを受け付けない。** 受け付けると #330 を作り直すだけ。"""
    client, _ = _client(tmp_path, [item("a", {"shape": "gone"})], [MULTI_CLASS])
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/remap",
        json={"remaps": [{"head": "shape", "from": "gone", "to": "nosuch"}]},
    )
    assert res.status_code == 400
    assert "Save the new label schema first" in res.json()["detail"]


def test_remap_allows_clearing_even_for_a_removed_head(tmp_path: Path) -> None:
    """head ごと消えたあとでも、**値を消す**ことはできる必要がある。"""
    client, label_path = _client(
        tmp_path, [item("a", {"tone": ["warm"]})], [MULTI_CLASS]
    )
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/remap",
        json={"remaps": [{"head": "tone", "from": "warm", "to": ""}]},
    )
    assert res.status_code == 200
    saved = json.loads(label_path.read_text(encoding="utf-8"))
    assert "tone" not in saved["items"][0]["labels"]


def test_remap_reports_unchanged_without_writing(tmp_path: Path) -> None:
    client, label_path = _client(
        tmp_path, [item("a", {"shape": "circle"})], [MULTI_CLASS]
    )
    before = label_path.read_text(encoding="utf-8")
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/remap",
        json={"remaps": [{"head": "shape", "from": "gone", "to": "square"}]},
    )
    assert res.json()["status"] == "unchanged"
    assert label_path.read_text(encoding="utf-8") == before
    assert not (label_path.parent / "history").exists(), "何も変わらないなら履歴も残さない"


def test_remap_requires_a_non_empty_list(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, [], [MULTI_CLASS])
    res = client.post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/remap", json={"remaps": []}
    )
    assert res.status_code == 400


def test_impact_on_an_experiment_without_labels(tmp_path: Path) -> None:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_schema_impact_router(config_manager))
    res = TestClient(app).post(
        f"/workspaces/{WS}/experiments/{EXP}/label-schema/impact",
        json=schema(MULTI_CLASS),
    )
    assert res.status_code == 200
    assert res.json()["items"] == 0
