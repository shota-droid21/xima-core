"""推論結果の閲覧 / CSV エクスポート API の結合テスト。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router

WORKSPACE = "ws51aa30"
EXPERIMENT = "ex51aa30"


def _make_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(create_experiments_router(ConfigManager(tmp_path)))
    return TestClient(app)


def _seed_experiment(tmp_path: Path, *, with_index: bool = True) -> Path:
    exp_root = tmp_path / WORKSPACE / "experiments" / EXPERIMENT
    (exp_root / "eval").mkdir(parents=True, exist_ok=True)
    (exp_root / "dataset").mkdir(parents=True, exist_ok=True)

    scores = {
        "items": [
            {"id": "img1", "character": {"cat": 0.9, "dog": 0.1}},
            {"id": "img2", "character": {"cat": 0.2, "dog": 0.8}},
        ],
        "meta": {"heads": ["character"], "clip_model_name": "ViT-B/32"},
    }
    (exp_root / "eval" / "scores_run_a.json").write_text(
        json.dumps(scores), encoding="utf-8"
    )

    if with_index:
        index = {
            "items": [
                {"id": "img1", "split": "train", "dataset_path": "train/img1.png"},
                {"id": "img2", "split": "val", "dataset_path": "val/img2.png"},
            ]
        }
        (exp_root / "dataset" / "index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
    return exp_root


def test_get_scores_returns_rows_joined_with_index(tmp_path: Path) -> None:
    _seed_experiment(tmp_path)
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}/eval/scores_run_a.json"
    )

    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert body["heads"] == ["character"]

    first = body["items"][0]
    assert first["id"] == "img1"
    assert first["split"] == "train"
    assert first["image_url"].endswith(
        f"{EXPERIMENT}/dataset/train/img1.png"
    )
    assert first["predictions"]["character"] == {"label": "cat", "score": 0.9}


def test_get_scores_paginates(tmp_path: Path) -> None:
    _seed_experiment(tmp_path)
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}/eval/scores_run_a.json",
        params={"limit": 1, "offset": 1},
    )

    body = res.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == "img2"


def test_get_scores_works_without_dataset_index(tmp_path: Path) -> None:
    _seed_experiment(tmp_path, with_index=False)
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}/eval/scores_run_a.json"
    )

    assert res.status_code == 200
    item = res.json()["items"][0]
    assert item["split"] is None
    assert item["image_url"] is None
    # index が無くても推論結果は読める
    assert item["predictions"]["character"]["label"] == "cat"


def test_get_scores_missing_file_returns_404(tmp_path: Path) -> None:
    _seed_experiment(tmp_path)
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}/eval/scores_missing.json"
    )
    assert res.status_code == 404


def test_get_scores_rejects_non_scores_name(tmp_path: Path) -> None:
    exp_root = _seed_experiment(tmp_path)
    # eval 配下に別ファイルがあっても scores_*.json 以外は読ませない
    (exp_root / "eval" / "secret.json").write_text("{}", encoding="utf-8")
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}/eval/secret.json"
    )
    assert res.status_code == 400


def test_export_csv_contains_all_rows(tmp_path: Path) -> None:
    _seed_experiment(tmp_path)
    client = _make_client(tmp_path)

    res = client.get(
        f"/workspaces/{WORKSPACE}/experiments/{EXPERIMENT}"
        f"/eval/scores_run_a.json/export.csv"
    )

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]
    assert "scores_run_a.csv" in res.headers["content-disposition"]

    lines = res.text.strip().splitlines()
    assert lines[0] == "id,split,character,character_score"
    # ページングに関係なく全件が出る
    assert len(lines) == 3
    assert lines[1].startswith("img1,train,cat,")
    assert lines[2].startswith("img2,val,dog,")
