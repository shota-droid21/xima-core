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


def _write_labels(
    root: Path,
    workspace: str,
    experiment: str,
    *,
    items: list[dict],
) -> None:
    labels_path = (
        root / workspace / "experiments" / experiment / "label_input" / "labels.json"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps({"items": items, "meta": {"version": "mvp"}}),
        encoding="utf-8",
    )


def test_delete_impact_counts_shared_train_and_val(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "wstest01"
    exp_a = "expa0001"
    exp_b = "expb0001"
    exp_c = "expc0001"

    # experiments router requires workspace root to exist.
    (tmp_path / workspace).mkdir(parents=True, exist_ok=True)

    for exp_id in (exp_a, exp_b, exp_c):
        created = client.post(
            f"/workspaces/{workspace}/experiments",
            json={"display_name": exp_id, "id": exp_id},
        )
        assert created.status_code == 200

    _write_labels(
        tmp_path,
        workspace,
        exp_a,
        items=[
            {"id": "1", "path": "a.jpg", "labels": {"split": "train"}, "delete": True},
            {"id": "2", "path": "dir/b.jpg", "labels": {"split": "val"}, "delete": True},
            {"id": "3", "path": "c.jpg", "labels": {"split": "train"}},
        ],
    )
    _write_labels(
        tmp_path,
        workspace,
        exp_b,
        items=[
            {"id": "10", "path": "a.jpg", "labels": {"split": "train"}},
            {"id": "11", "path": "dir/b.jpg", "labels": {"split": "val"}},
            {"id": "12", "path": "c.jpg", "labels": {"split": "train"}},
        ],
    )
    _write_labels(
        tmp_path,
        workspace,
        exp_c,
        items=[
            {"id": "20", "path": "a.jpg", "labels": {"split": "unassigned"}},
            {"id": "21", "path": "dir/b.jpg", "labels": {"split": "train"}, "delete": True},
        ],
    )

    res = client.get(f"/workspaces/{workspace}/experiments/{exp_a}/delete-impact")
    assert res.status_code == 200
    payload = res.json()

    assert payload["delete_targets"] == 2
    assert payload["other_experiments_checked"] == 2
    assert payload["totals"] == {
        "experiments_affected": 2,
        "shared": 4,
        "train": 2,
        "val": 1,
        "unassigned": 1,
        "other": 0,
        "flagged": 1,
    }

    by_id = {row["id"]: row for row in payload["experiments"]}
    assert by_id[exp_b]["counts"] == {
        "shared": 2,
        "train": 1,
        "val": 1,
        "unassigned": 0,
        "other": 0,
        "flagged": 0,
    }
    assert by_id[exp_c]["counts"] == {
        "shared": 2,
        "train": 1,
        "val": 0,
        "unassigned": 1,
        "other": 0,
        "flagged": 1,
    }


def test_delete_impact_returns_zero_when_no_delete_targets(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    workspace = "wstest01"
    exp_a = "expa0002"
    exp_b = "expb0002"

    (tmp_path / workspace).mkdir(parents=True, exist_ok=True)

    for exp_id in (exp_a, exp_b):
        created = client.post(
            f"/workspaces/{workspace}/experiments",
            json={"display_name": exp_id, "id": exp_id},
        )
        assert created.status_code == 200

    _write_labels(
        tmp_path,
        workspace,
        exp_a,
        items=[{"id": "1", "path": "a.jpg", "labels": {"split": "train"}}],
    )
    _write_labels(
        tmp_path,
        workspace,
        exp_b,
        items=[{"id": "2", "path": "a.jpg", "labels": {"split": "train"}}],
    )

    res = client.get(f"/workspaces/{workspace}/experiments/{exp_a}/delete-impact")
    assert res.status_code == 200
    payload = res.json()

    assert payload["delete_targets"] == 0
    assert payload["other_experiments_checked"] == 0
    assert payload["totals"] == {
        "experiments_affected": 0,
        "shared": 0,
        "train": 0,
        "val": 0,
        "unassigned": 0,
        "other": 0,
        "flagged": 0,
    }
    assert payload["experiments"] == []
