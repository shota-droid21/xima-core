"""workspace / experiment を「ゴミ箱へ入れる」側の応答を固定する。

なぜ書くか（#225）:
    ゴミ箱の**中身を扱う側**（一覧 / 復元 / 完全削除）は `app/trash.py` にあるが、
    **入れる側**は `app/workspace.py` と `app/experiments.py` に分かれて置かれており、
    どちらもテストが無かった。

    削除は取り返しがつかない操作で、`meta.json` に何を書き残すかが復元の可否を決める。
    #225 で入れる側を `app/trash.py` へ寄せる予定があるため、その前に
    「今どこへ何を書いているか」を固定する。

固定しているもの:
    - 応答の構造と、ディレクトリが実際に移動していること
    - `meta.json` に残す情報（**復元に必要なのは `original_rel`**）
    - workspace を捨てるときに、道連れになる experiment を数え上げていること
    - 入力が不正なときの status code
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router
from app.workspace import create_workspace_router

WS = "wstrash1"
EXP_A = "expaaaa1"
EXP_B = "expbbbb2"


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_workspace_router(config_manager))
    app.include_router(create_experiments_router(config_manager))
    return TestClient(app)


def _setup(client: TestClient, tmp_path: Path, *experiments: str) -> Path:
    root = tmp_path / WS
    root.mkdir(parents=True, exist_ok=True)
    for exp in experiments:
        created = client.post(
            f"/workspaces/{WS}/experiments",
            json={"display_name": exp, "id": exp},
        )
        assert created.status_code == 200, created.text
    return root


def _trash_meta(tmp_path: Path, trash_id: str) -> dict:
    path = tmp_path / ".trash" / trash_id / "meta.json"
    assert path.exists(), f"meta.json not written for {trash_id}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---- experiment ------------------------------------------------------------


def test_trash_experiment_moves_the_directory(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A)
    exp_root = tmp_path / WS / "experiments" / EXP_A
    marker = exp_root / "label_input" / "labels.json"
    assert marker.exists()

    res = client.post(f"/workspaces/{WS}/experiments/{EXP_A}/trash")
    assert res.status_code == 200, res.text
    body = res.json()

    assert set(body) == {
        "status",
        "trash_id",
        "kind",
        "workspace",
        "experiment",
        "display_name",
        "deleted_at",
        "moved_from",
        "moved_to",
    }
    assert body["status"] == "ok"
    assert body["kind"] == "experiment"
    assert body["workspace"] == WS
    assert body["experiment"] == EXP_A

    # 元の場所から消え、ゴミ箱の item/ 配下に中身ごと移っている。
    assert not exp_root.exists()
    moved = tmp_path / ".trash" / body["trash_id"] / "item"
    assert moved.is_dir()
    assert (moved / "label_input" / "labels.json").exists()


def test_trash_experiment_meta_records_the_original_location(tmp_path: Path) -> None:
    """`original_rel` が無いと復元先が決まらない。"""
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A)

    res = client.post(f"/workspaces/{WS}/experiments/{EXP_A}/trash")
    meta = _trash_meta(tmp_path, res.json()["trash_id"])

    assert meta["version"] == 1
    assert meta["kind"] == "experiment"
    assert meta["workspace"] == WS
    assert meta["experiment"] == EXP_A
    assert meta["original_rel"] == f"{WS}/experiments/{EXP_A}"
    assert meta["trash_id"] == res.json()["trash_id"]
    assert meta["deleted_at"] == res.json()["deleted_at"]


def test_trash_experiment_keeps_the_display_name(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A)
    renamed = client.patch(
        f"/workspaces/{WS}/experiments/{EXP_A}",
        json={"display_name": "けもの実験"},
    )
    assert renamed.status_code == 200, renamed.text

    res = client.post(f"/workspaces/{WS}/experiments/{EXP_A}/trash")
    assert res.json()["display_name"] == "けもの実験"
    assert _trash_meta(tmp_path, res.json()["trash_id"])["display_name"] == "けもの実験"


def test_trash_experiment_each_call_gets_its_own_directory(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A, EXP_B)

    first = client.post(f"/workspaces/{WS}/experiments/{EXP_A}/trash").json()
    second = client.post(f"/workspaces/{WS}/experiments/{EXP_B}/trash").json()

    assert first["trash_id"] != second["trash_id"]
    assert (tmp_path / ".trash" / first["trash_id"] / "item").is_dir()
    assert (tmp_path / ".trash" / second["trash_id"] / "item").is_dir()


def test_trash_experiment_rejects_bad_input(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A)

    assert client.post(f"/workspaces/{WS}/experiments/nope/trash").status_code == 400
    assert (
        client.post(f"/workspaces/{WS}/experiments/expzzzz9/trash").status_code == 404
    )
    assert client.post(f"/workspaces/nope/experiments/{EXP_A}/trash").status_code == 400


# ---- workspace -------------------------------------------------------------


def test_trash_workspace_moves_the_directory(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    root = _setup(client, tmp_path, EXP_A)

    res = client.post(f"/workspaces/{WS}/trash")
    assert res.status_code == 200, res.text
    body = res.json()

    assert set(body) == {
        "status",
        "trash_id",
        "kind",
        "workspace",
        "display_name",
        "deleted_at",
        "dependent_experiments_count",
        "dependent_experiments",
        "moved_from",
        "moved_to",
    }
    assert body["kind"] == "workspace"
    assert not root.exists()
    assert (tmp_path / ".trash" / body["trash_id"] / "item" / "experiments").is_dir()


def test_trash_workspace_lists_the_experiments_it_takes_with_it(
    tmp_path: Path,
) -> None:
    """workspace を捨てると experiment も道連れになる。

    利用者が「何件が一緒に消えるか」を確認できるよう、応答と meta.json の
    両方に残している。**数だけでなく id も残す**のは、復元後に照合するため。
    """
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_B, EXP_A)
    # short id でないディレクトリは道連れの数に入れない。
    (tmp_path / WS / "experiments" / "NotAnExperiment").mkdir(parents=True)

    body = client.post(f"/workspaces/{WS}/trash").json()
    assert body["dependent_experiments_count"] == 2
    assert body["dependent_experiments"] == [EXP_A, EXP_B]  # ソート済み

    meta = _trash_meta(tmp_path, body["trash_id"])
    assert meta["dependent_experiments"] == [EXP_A, EXP_B]
    assert meta["dependent_experiments_count"] == 2


def test_trash_workspace_meta_records_the_original_location(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path)

    res = client.post(f"/workspaces/{WS}/trash")
    meta = _trash_meta(tmp_path, res.json()["trash_id"])

    assert meta["version"] == 1
    assert meta["kind"] == "workspace"
    assert meta["workspace"] == WS
    assert meta["original_rel"] == WS
    # workspace 側だけ agent identity も残している（experiment 側は残さない）。
    assert {"container_id", "install_id", "agent_version"} <= set(meta)


def test_trash_workspace_rejects_bad_input(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path)

    assert client.post("/workspaces/nope/trash").status_code == 400
    assert client.post("/workspaces/wsabsent/trash").status_code == 404


def test_trashed_workspace_disappears_from_the_listings(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path, EXP_A)
    assert [w["id"] for w in client.get("/workspaces").json()["workspaces"]] == [WS]

    client.post(f"/workspaces/{WS}/trash")

    assert client.get("/workspaces").json()["workspaces"] == []
    # `.trash` 自体が workspace として並ばないことも確かめる。
    assert client.get("/workspaces/sidebar-tree").json()["workspaces"] == []
