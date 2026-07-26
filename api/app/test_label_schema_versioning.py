from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.label_input import create_label_input_router


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_label_input_router(config_manager))
    return TestClient(app)


def _schema(schema_id: str, head_id: str) -> dict:
    return {
        "version": 2,
        "schema_id": schema_id,
        "heads": [
            {
                "id": head_id,
                "type": "multi_class",
                "classes": ["a", "b"],
            }
        ],
    }


def _backup_id_from_version_path(path_str: str) -> str:
    name = Path(path_str).name
    prefix = "label_schema_"
    suffix = ".json"
    assert name.startswith(prefix), name
    assert name.endswith(suffix), name
    return name[len(prefix) : -len(suffix)]


def test_label_schema_save_and_restore_are_versioned(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "wstest01"
    experiment = "exp00001"

    schema_v1 = _schema("schema_v1", "character")
    schema_v2 = _schema("schema_v2", "tag")

    put_v1 = client.put(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema",
        json=schema_v1,
    )
    assert put_v1.status_code == 200
    put_v1_data = put_v1.json()
    assert put_v1_data["status"] == "saved"
    assert "version" in put_v1_data
    assert Path(put_v1_data["version"]).exists()

    put_v2 = client.put(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema",
        json=schema_v2,
    )
    assert put_v2.status_code == 200
    put_v2_data = put_v2.json()
    assert put_v2_data["status"] == "saved"
    assert "version" in put_v2_data
    assert Path(put_v2_data["version"]).exists()

    history_before_restore = client.get(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema/history"
    )
    assert history_before_restore.status_code == 200
    backups = history_before_restore.json()["backups"]
    assert len(backups) == 2

    backup_id_for_v1 = None
    for backup in backups:
        res = client.get(
            f"/workspaces/{workspace}/experiments/{experiment}/label-schema/history/{backup['id']}"
        )
        assert res.status_code == 200
        if res.json() == schema_v1:
            backup_id_for_v1 = backup["id"]
            break
    assert backup_id_for_v1 is not None

    restore = client.post(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema/history/{backup_id_for_v1}/restore"
    )
    assert restore.status_code == 200
    restore_data = restore.json()
    assert restore_data["status"] == "restored"
    assert restore_data["version"]
    assert Path(restore_data["version"]).exists()

    current = client.get(f"/workspaces/{workspace}/experiments/{experiment}/label-schema")
    assert current.status_code == 200
    assert current.json() == schema_v1

    history_after_restore = client.get(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema/history"
    )
    assert history_after_restore.status_code == 200
    backups_after = history_after_restore.json()["backups"]
    assert len(backups_after) == 3

    restored_version_id = _backup_id_from_version_path(restore_data["version"])
    restored_version = client.get(
        f"/workspaces/{workspace}/experiments/{experiment}/label-schema/history/{restored_version_id}"
    )
    assert restored_version.status_code == 200
    assert restored_version.json() == schema_v1
