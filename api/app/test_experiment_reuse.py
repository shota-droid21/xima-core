from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_experiments_router(config_manager))
    return TestClient(app)


def test_reuse_experiment_copies_only_labels_schema_and_cache(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "wstest01"
    source_exp = "src00001"

    # Provision workspace and source experiment.
    ws_root = tmp_path / workspace
    ws_root.mkdir(parents=True, exist_ok=True)
    created = client.post(
        f"/workspaces/{workspace}/experiments",
        json={"display_name": "Source Experiment", "id": source_exp},
    )
    assert created.status_code == 200

    src_root = ws_root / "experiments" / source_exp
    src_labels = src_root / "label_input" / "labels.json"
    src_schema = src_root / "label_input" / "label_schema.json"
    src_cache_file = src_root / "cache" / "thumbs" / "w200" / "ab" / "x.webp"
    src_dataset_file = src_root / "dataset" / "index.json"
    src_model_file = src_root / "models" / "run_20260101_000000" / "model.pt"
    src_score_file = src_root / "eval" / "scores_20260101_000000.json"

    src_labels.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "1",
                        "file_id": "a",
                        "thumb_path": "/static/ws_test/experiments/exp1/cache/thumbs/w256/86/86f7e437faa5a7fce15d1ddcb9eaeaea377667b8.webp",
                    }
                ],
                "meta": {"version": "x"},
            }
        )
    )
    src_schema.write_text(
        json.dumps(
            {
                "version": 2,
                "schema_id": "schema_a",
                "heads": [{"id": "head1", "type": "multi_class", "classes": ["a"]}],
            }
        )
    )
    src_cache_file.parent.mkdir(parents=True, exist_ok=True)
    src_cache_file.write_bytes(b"webp")
    src_dataset_file.write_text(json.dumps({"items": ["should_not_copy"]}))
    src_model_file.parent.mkdir(parents=True, exist_ok=True)
    src_model_file.write_bytes(b"model")
    src_score_file.parent.mkdir(parents=True, exist_ok=True)
    src_score_file.write_text(json.dumps({"acc": 0.99}))

    reused = client.post(
        f"/workspaces/{workspace}/experiments/{source_exp}/reuse",
        json={"display_name": "Reused Experiment"},
    )
    assert reused.status_code == 200
    reused_data = reused.json()
    assert reused_data["status"] == "ok"
    assert reused_data["source_experiment"] == source_exp
    assert reused_data["copied"] == {
        "labels_json": True,
        "label_schema_json": True,
        "cache": True,
    }

    target_exp = reused_data["experiment"]
    target_root = ws_root / "experiments" / target_exp
    target_labels = target_root / "label_input" / "labels.json"
    target_schema = target_root / "label_input" / "label_schema.json"
    target_cache_file = target_root / "cache" / "thumbs" / "w200" / "ab" / "x.webp"
    target_dataset_file = target_root / "dataset" / "index.json"
    target_model_file = target_root / "models" / "run_20260101_000000" / "model.pt"
    target_score_file = target_root / "eval" / "scores_20260101_000000.json"

    assert target_labels.exists()
    target_labels_data = json.loads(target_labels.read_text())
    assert target_labels_data["items"][0]["thumb_path"] == (
        f"/static/{workspace}/experiments/{target_exp}/cache/thumbs/w256/"
        "86/86f7e437faa5a7fce15d1ddcb9eaeaea377667b8.webp"
    )
    assert target_schema.exists()
    assert target_schema.read_text() == src_schema.read_text()
    assert target_cache_file.exists()
    assert target_cache_file.read_bytes() == b"webp"

    assert not target_dataset_file.exists()
    assert not target_model_file.exists()
    assert not target_score_file.exists()
