"""`GET /workspaces/sidebar-tree` の応答を固定する。

なぜ書くか（#225）:
    このエンドポイントは 258 行あり、**テストが 1 本も無かった**。
    それでいて app は全ページの読み込みで毎回これを呼ぶ（`app/src/lib/api.ts`）ため、
    壊れると画面全体が沈黙する。

    #225 で `app/sidebar_tree.py` へ切り出す予定があり、その分割は
    「振る舞いを変えない」ことが前提になる。core には app でやったような
    「変更前後のビルドを入れ替えて画面を突き合わせる」検証が使えないので、
    ここで応答の形を先に固定しておくことが唯一の安全網になる。

固定しているもの:
    - 応答の構造（キーの集合）
    - 集計の対象と除外の規則（何を数えて何を数えないか）
    - 並び順（sidebar_order と、それが無いときの既定）
    - 壊れた入力での落ち方（例外を投げず、その項目を飛ばす）

固定していないもの:
    絶対パス・mtime・`generated_at` といった環境依存の値。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router
from app.workspace import create_workspace_router

WS = "wstree01"
EXP_A = "expaaaa1"
EXP_B = "expbbbb2"


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_workspace_router(config_manager))
    app.include_router(create_experiments_router(config_manager))
    return TestClient(app)


def _make_workspace(tmp_path: Path, ws: str) -> Path:
    root = tmp_path / ws
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_experiment(client: TestClient, ws: str, exp: str) -> None:
    res = client.post(
        f"/workspaces/{ws}/experiments",
        json={"display_name": exp, "id": exp},
    )
    assert res.status_code == 200, res.text


def _tree(client: TestClient) -> dict:
    res = client.get("/workspaces/sidebar-tree")
    assert res.status_code == 200, res.text
    return res.json()


def test_empty_root_returns_no_workspaces(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    body = _tree(client)

    assert body["workspaces"] == []
    assert isinstance(body["generated_at"], float)


def test_workspace_entry_shape(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    root = _make_workspace(tmp_path, WS)

    body = _tree(client)
    assert len(body["workspaces"]) == 1
    ws_entry = body["workspaces"][0]

    assert set(ws_entry) == {
        "id",
        "display_name",
        "workspace_item_uid",
        "workspace_uid",
        "workspace_root",
        "experiments",
    }
    assert ws_entry["id"] == WS
    # meta が無いときは id をそのまま表示名にする。
    assert ws_entry["display_name"] == WS
    assert ws_entry["workspace_root"] == str(root.resolve())
    assert ws_entry["experiments"] == []


def test_experiment_entry_shape(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)

    exp = _tree(client)["workspaces"][0]["experiments"][0]

    assert set(exp) == {
        "id",
        "display_name",
        "workspace_uid",
        "experiment_uid",
        "counts",
        "flags",
        "label_backups",
        "label_backups_truncated",
        "label_schema_backups",
        "label_schema_backups_truncated",
    }
    assert set(exp["counts"]) == {
        "model_runs",
        "eval_scores",
        "label_backups",
        "label_schema_backups",
    }
    assert set(exp["flags"]) == {"has_labels", "has_label_schema"}

    # experiment を作った直後は labels.json と label_schema.json が用意される。
    assert exp["flags"] == {"has_labels": True, "has_label_schema": True}
    assert exp["counts"] == {
        "model_runs": 0,
        "eval_scores": 0,
        "label_backups": 0,
        "label_schema_backups": 0,
    }


def test_ids_that_are_not_short_ids_are_skipped(tmp_path: Path) -> None:
    """`[a-z0-9]{8}` に合わないディレクトリは無視する。

    `.trash` `.xima` `.tmp` といった運用用のディレクトリが
    workspace として並んでしまわないための規則。
    """
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    for name in ("TooLongWorkspaceName", "short", ".trash", "has-dash"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "notadir.txt").write_text("x", encoding="utf-8")

    body = _tree(client)
    assert [w["id"] for w in body["workspaces"]] == [WS]


def test_uppercase_dir_name_is_reported_lowercased(tmp_path: Path) -> None:
    """大文字のディレクトリ名は **id としては小文字化されて出る**。

    `require_workspace_id` が `.lower()` を通すため。結果として
    `id` と `workspace_root` の末尾が食い違う。ここでは現状をそのまま固定する
    （直すかどうかは #225 の分割とは別の判断）。
    """
    client = _make_client(tmp_path)
    (tmp_path / "ABCDEFGH").mkdir(parents=True, exist_ok=True)

    ws_entry = _tree(client)["workspaces"][0]
    assert ws_entry["id"] == "abcdefgh"
    assert Path(ws_entry["workspace_root"]).name == "ABCDEFGH"


def test_experiment_ids_that_are_not_short_ids_are_skipped(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    (tmp_path / WS / "experiments" / "NotAnExperiment").mkdir(parents=True)

    exps = _tree(client)["workspaces"][0]["experiments"]
    assert [e["id"] for e in exps] == [EXP_A]


def test_counts_only_run_dirs_and_scores_files(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    exp_root = tmp_path / WS / "experiments" / EXP_A

    (exp_root / "models" / "run_20260101_010101").mkdir(parents=True)
    (exp_root / "models" / "run_20260102_020202").mkdir(parents=True)
    # run_ で始まらないディレクトリと、ディレクトリでない run_ は数えない。
    (exp_root / "models" / "notarun").mkdir(parents=True)
    (exp_root / "models" / "run_notadir").write_text("x", encoding="utf-8")

    (exp_root / "eval").mkdir(parents=True, exist_ok=True)
    (exp_root / "eval" / "scores_a.json").write_text("{}", encoding="utf-8")
    (exp_root / "eval" / "scores_b.json").write_text("{}", encoding="utf-8")
    # scores_ で始まらないものは数えない。
    (exp_root / "eval" / "other.json").write_text("{}", encoding="utf-8")

    counts = _tree(client)["workspaces"][0]["experiments"][0]["counts"]
    assert counts["model_runs"] == 2
    assert counts["eval_scores"] == 2


def test_label_backups_are_classified_and_filtered(tmp_path: Path) -> None:
    """history の中から backup として認めるものを固定する。

    - `labels_<id>.json` と、旧データの `current_<id>.json` は labels 側
    - `label_schema_<id>.json` は schema 側
    - `<id>` は `YYYYmmdd_HHMMSS`（末尾に `-<連番>` が付くことがある）
    """
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    history = tmp_path / WS / "experiments" / EXP_A / "label_input" / "history"
    history.mkdir(parents=True, exist_ok=True)

    accepted = [
        "labels_20260101_010101.json",
        "current_20260101_010102.json",
        "labels_20260101_010103-2.json",
        "label_schema_20260101_010104.json",
    ]
    rejected = [
        "labels_bogus.json",
        "labels_2026-01-01.json",
        "labels_.json",
        "snapshot_20260101_010101.json",
        "20260101_010101.json",
        "readme.txt",
    ]
    for name in accepted + rejected:
        (history / name).write_text("{}", encoding="utf-8")

    exp = _tree(client)["workspaces"][0]["experiments"][0]
    assert exp["counts"]["label_backups"] == 3
    assert exp["counts"]["label_schema_backups"] == 1
    assert {b["filename"] for b in exp["label_backups"]} == {
        "labels_20260101_010101.json",
        "current_20260101_010102.json",
        "labels_20260101_010103-2.json",
    }
    assert [b["id"] for b in exp["label_schema_backups"]] == ["20260101_010104"]
    assert set(exp["label_backups"][0]) == {
        "id",
        "filename",
        "size_bytes",
        "modified_at",
    }


def test_label_backups_are_newest_first(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    history = tmp_path / WS / "experiments" / EXP_A / "label_input" / "history"
    history.mkdir(parents=True, exist_ok=True)

    # 並びは**ファイル名ではなく mtime** で決まる。名前と逆の時刻を与えて確かめる。
    names = [
        "labels_20260101_010101.json",
        "labels_20260101_010102.json",
        "labels_20260101_010103.json",
    ]
    for i, name in enumerate(names):
        p = history / name
        p.write_text("{}", encoding="utf-8")
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
    # 一番古い名前に一番新しい mtime を与える。
    os.utime(history / names[0], (2_000_000, 2_000_000))

    backups = _tree(client)["workspaces"][0]["experiments"][0]["label_backups"]
    assert [b["filename"] for b in backups] == [
        "labels_20260101_010101.json",
        "labels_20260101_010103.json",
        "labels_20260101_010102.json",
    ]


def test_label_backups_are_truncated_at_20(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    history = tmp_path / WS / "experiments" / EXP_A / "label_input" / "history"
    history.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        (history / f"labels_20260101_{i:06d}.json").write_text("{}", encoding="utf-8")

    exp = _tree(client)["workspaces"][0]["experiments"][0]
    # count は全件、配列は 20 件まで。
    assert exp["counts"]["label_backups"] == 25
    assert len(exp["label_backups"]) == 20
    assert exp["label_backups_truncated"] is True


def test_has_labels_accepts_legacy_current_json(tmp_path: Path) -> None:
    """旧データは `current.json` に labels を持っている。"""
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    label_input = tmp_path / WS / "experiments" / EXP_A / "label_input"
    label_input.mkdir(parents=True, exist_ok=True)
    (label_input / "current.json").write_text("{}", encoding="utf-8")

    flags = _tree(client)["workspaces"][0]["experiments"][0]["flags"]
    assert flags == {"has_labels": True, "has_label_schema": False}


def test_sidebar_order_controls_workspace_and_experiment_order(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    ws_b = "wstree02"
    _make_workspace(tmp_path, WS)
    _make_workspace(tmp_path, ws_b)
    _make_experiment(client, WS, EXP_A)
    _make_experiment(client, WS, EXP_B)

    # 既定は名前順。
    body = _tree(client)
    assert [w["id"] for w in body["workspaces"]] == [WS, ws_b]
    assert [e["id"] for e in body["workspaces"][0]["experiments"]] == [EXP_A, EXP_B]

    saved = client.put(
        "/workspaces/sidebar-order",
        json={"workspaces": [ws_b, WS], "experiments": {WS: [EXP_B, EXP_A]}},
    )
    assert saved.status_code == 200, saved.text

    body = _tree(client)
    assert [w["id"] for w in body["workspaces"]] == [ws_b, WS]
    reordered = next(w for w in body["workspaces"] if w["id"] == WS)
    assert [e["id"] for e in reordered["experiments"]] == [EXP_B, EXP_A]


def test_unknown_ids_in_sidebar_order_are_ignored(tmp_path: Path) -> None:
    """並び順に載っていない / 実在しない id があっても落ちない。

    並び順は別ファイルなので、workspace の削除や追加と簡単にずれる。
    """
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)

    saved = client.put(
        "/workspaces/sidebar-order",
        json={"workspaces": ["deadbee1", WS], "experiments": {"deadbee1": ["nope0001"]}},
    )
    assert saved.status_code == 200, saved.text

    body = _tree(client)
    assert [w["id"] for w in body["workspaces"]] == [WS]


def test_broken_experiment_meta_falls_back_to_id(tmp_path: Path) -> None:
    """`experiment.json` が壊れていても、その experiment を落とさない。"""
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)
    meta_path = tmp_path / WS / "experiments" / EXP_A / "experiment.json"
    meta_path.write_text("{ this is not json", encoding="utf-8")

    exp = _tree(client)["workspaces"][0]["experiments"][0]
    assert exp["id"] == EXP_A
    assert exp["display_name"] == EXP_A


def test_display_name_comes_from_meta(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)
    _make_experiment(client, WS, EXP_A)

    renamed = client.patch(
        f"/workspaces/{WS}/experiments/{EXP_A}",
        json={"display_name": "実験その 1"},
    )
    assert renamed.status_code == 200, renamed.text

    exp = _tree(client)["workspaces"][0]["experiments"][0]
    assert exp["display_name"] == "実験その 1"
    assert exp["id"] == EXP_A


def test_agent_identity_is_included(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _make_workspace(tmp_path, WS)

    agent = _tree(client)["agent"]
    assert set(agent) == {"container_id", "install_id", "agent_version"}
    assert agent["container_id"]
    assert agent["install_id"]
