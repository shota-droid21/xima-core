"""クラスタリング API の結合テスト。

行列読み込み（numpy）はスタブ化し、CI（numpy 無し）でもルーターの配線・
スコープ絞り込み・レスポンス整形を検証できるようにする。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.clustering import create_clustering_router
from app.config import ConfigManager

WS = "wsdemo01"
EXP = "expdemo1"


def _make_client(tmp_path: Path) -> tuple[TestClient, ConfigManager]:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_clustering_router(config_manager))
    return TestClient(app), config_manager


def _blob_items():
    """3 つに分かれた点群 + 対応する index items。"""
    items = [{"file_id": f"img_{i}", "path": f"a/img_{i}.png", "row": i} for i in range(9)]
    vectors = [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
        [-10.0, 5.0], [-10.1, 5.0], [-10.0, 5.1],
    ]
    return items, vectors


def _write_labels(config_manager, items, labeled_head_by_fid):
    """labels.json / label_schema.json を用意する。"""
    cfg = config_manager.get_config()
    label_path = cfg.label_input_path_for(WS, EXP)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    docs = {
        "meta": {},
        "items": [
            {
                "id": i,
                "file_id": it["file_id"],
                "path": it["path"],
                "labels": labeled_head_by_fid.get(it["file_id"], {}),
            }
            for i, it in enumerate(items)
        ],
    }
    label_path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    schema_path = cfg.label_schema_path_for(WS, EXP)
    schema = {
        "version": 2,
        "schema_id": "t",
        "heads": [
            {"id": "split", "type": "split", "choices": ["train", "val"]},
            {"id": "shape", "type": "multi_class", "classes": ["a", "b", "c"]},
        ],
    }
    schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")


def test_returns_409_when_embeddings_missing(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path)
    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/clusters")
    assert res.status_code == 409
    assert "embed" in res.json()["detail"].lower()


def test_clusters_group_similar_vectors(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path)
    items, vectors = _blob_items()
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters", params={"k": 3}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["k"] == 3
    assert body["total"] == 9
    assert len(body["clusters"]) == 3
    # 各クラスタは file_id / path / thumb_path を持つ
    first = body["clusters"][0]
    assert first["representative"]["file_id"].startswith("img_")
    assert first["representative"]["thumb_path"].startswith("/static/")
    # 全メンバが重複なく現れる
    seen = [m["file_id"] for c in body["clusters"] for m in c["members"]]
    assert sorted(seen) == sorted(it["file_id"] for it in items)


def test_scope_unlabeled_excludes_labeled_items(tmp_path, monkeypatch):
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    # img_0 だけ shape を付与済み → unlabeled から除外されるはず
    _write_labels(cfg, items, {"img_0": {"shape": "a"}})
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters",
        params={"scope": "unlabeled"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["scope"] == "unlabeled"
    assert body["head"] == "shape"  # schema から自動解決
    assert body["total"] == 8
    seen = [m["file_id"] for c in body["clusters"] for m in c["members"]]
    assert "img_0" not in seen


def test_rejects_bad_scope(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path)
    items, vectors = _blob_items()
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))
    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters", params={"scope": "bogus"}
    )
    assert res.status_code == 400
