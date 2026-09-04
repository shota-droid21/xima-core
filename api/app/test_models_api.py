"""`GET /workspaces/{ws}/experiments/{exp}/models` の応答を固定する。

なぜ書くか（#225）:
    app の実験ダッシュボードが学習結果の一覧に使っているが、テストが無かった。
    #225 で `app/experiments.py` を解体する際に、ここが静かに変わると
    「学習は回ったのに画面に出ない」という形で表面化する。

固定しているもの:
    - run の判定（`run_*` のディレクトリだけ）と並び（**名前の降順 = 新しい順**）
    - `created_at` を run 名から作る規則と、作れないときの `None`
    - meta の読み先の優先順（`_meta.json` → `run_meta.json`）と、壊れていたときの `None`
    - 配信 URL（`/static/<workspaces_root からの相対パス>`）
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router

WS = "wsmodel1"
EXP = "expaaaa1"


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_experiments_router(config_manager))
    return TestClient(app)


def _setup(client: TestClient, tmp_path: Path) -> Path:
    (tmp_path / WS).mkdir(parents=True, exist_ok=True)
    created = client.post(
        f"/workspaces/{WS}/experiments",
        json={"display_name": EXP, "id": EXP},
    )
    assert created.status_code == 200, created.text
    return tmp_path / WS / "experiments" / EXP


def _models(client: TestClient) -> dict:
    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/models")
    assert res.status_code == 200, res.text
    return res.json()


def test_missing_models_dir_returns_empty_runs(tmp_path: Path) -> None:
    """まだ 1 度も学習していない experiment でも 404 にしない。"""
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    assert not (exp_root / "models").exists()

    body = _models(client)
    assert set(body) == {"workspace", "experiment", "models_dir", "runs"}
    assert body["workspace"] == WS
    assert body["experiment"] == EXP
    assert body["runs"] == []


def test_only_run_directories_are_listed(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    models = exp_root / "models"
    (models / "run_20260101_010101").mkdir(parents=True)
    (models / "notarun").mkdir(parents=True)
    # `run_` で始まってもファイルなら run ではない。
    (models / "run_notadir").write_text("x", encoding="utf-8")

    assert [r["id"] for r in _models(client)["runs"]] == ["run_20260101_010101"]


def test_runs_are_newest_first(tmp_path: Path) -> None:
    """並びは run 名の降順。run 名が時刻そのものなので新しい順になる。"""
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    for name in (
        "run_20260101_010101",
        "run_20260103_030303",
        "run_20260102_020202",
    ):
        (exp_root / "models" / name).mkdir(parents=True)

    assert [r["id"] for r in _models(client)["runs"]] == [
        "run_20260103_030303",
        "run_20260102_020202",
        "run_20260101_010101",
    ]


def test_created_at_is_parsed_from_the_run_name(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    (exp_root / "models" / "run_20260102_030405").mkdir(parents=True)

    assert _models(client)["runs"][0]["created_at"] == "2026-01-02T03:04:05"


def test_created_at_is_none_when_the_name_is_not_a_timestamp(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    for name in ("run_manual", "run_2026_01", "run_20260102_0304"):
        (exp_root / "models" / name).mkdir(parents=True)

    runs = {r["id"]: r["created_at"] for r in _models(client)["runs"]}
    assert runs == {
        "run_manual": None,
        "run_2026_01": None,
        "run_20260102_0304": None,
    }


def test_run_entry_shape_and_files(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    run = exp_root / "models" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "head.pt").write_bytes(b"x" * 7)
    (run / "run_meta.json").write_text(
        json.dumps({"heads": ["character"]}), encoding="utf-8"
    )
    # ディレクトリはファイル一覧に入れない。
    (run / "nested").mkdir()

    entry = _models(client)["runs"][0]
    assert set(entry) == {"id", "created_at", "path", "rel", "files", "meta"}
    assert entry["rel"] == f"{WS}/experiments/{EXP}/models/run_20260101_010101"

    # ファイルは名前の昇順。
    assert [f["name"] for f in entry["files"]] == ["head.pt", "run_meta.json"]
    assert set(entry["files"][0]) == {"name", "bytes", "mtime", "url"}
    assert entry["files"][0]["bytes"] == 7
    assert (
        entry["files"][0]["url"]
        == f"/static/{WS}/experiments/{EXP}/models/run_20260101_010101/head.pt"
    )


def test_meta_prefers_underscore_meta_json(tmp_path: Path) -> None:
    """`_meta.json` があればそちらを読み、`run_meta.json` は見ない。"""
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    run = exp_root / "models" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "_meta.json").write_text(json.dumps({"from": "legacy"}), encoding="utf-8")
    (run / "run_meta.json").write_text(
        json.dumps({"from": "current"}), encoding="utf-8"
    )

    assert _models(client)["runs"][0]["meta"] == {"from": "legacy"}


def test_meta_falls_back_to_run_meta_json(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    run = exp_root / "models" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(
        json.dumps({"from": "current"}), encoding="utf-8"
    )

    assert _models(client)["runs"][0]["meta"] == {"from": "current"}


def test_broken_meta_becomes_none_without_dropping_the_run(tmp_path: Path) -> None:
    """meta が壊れていても run 自体は一覧に残す。

    ここで run ごと消すと、利用者からは「学習が消えた」ように見える。
    """
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    run = exp_root / "models" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text("{ not json", encoding="utf-8")

    entry = _models(client)["runs"][0]
    assert entry["id"] == "run_20260101_010101"
    assert entry["meta"] is None


def test_missing_meta_is_none(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    exp_root = _setup(client, tmp_path)
    (exp_root / "models" / "run_20260101_010101").mkdir(parents=True)

    assert _models(client)["runs"][0]["meta"] is None


def test_rejects_bad_ids(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _setup(client, tmp_path)

    assert client.get(f"/workspaces/{WS}/experiments/nope/models").status_code == 400
    assert client.get(f"/workspaces/nope/experiments/{EXP}/models").status_code == 400


def test_unknown_experiment_returns_empty_runs(tmp_path: Path) -> None:
    """存在しない experiment でも 404 にはならない。

    `models` ディレクトリの有無だけを見ているため。**現状をそのまま固定する**。
    """
    client = _make_client(tmp_path)
    _setup(client, tmp_path)

    res = client.get(f"/workspaces/{WS}/experiments/expzzzz9/models")
    assert res.status_code == 200
    assert res.json()["runs"] == []
