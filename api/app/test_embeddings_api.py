"""埋め込み状態 API の結合テスト（#258）。

**numpy を使わない。** `embeddings.npy` は存在だけを見て中身を読まないので、
CI（numpy 無し）でもそのまま通る。ダミーのバイト列で十分。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.embeddings import create_embeddings_router

WS = "wsdemo01"
EXP = "expdemo1"


def _make_client(tmp_path: Path) -> tuple[TestClient, ConfigManager]:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_embeddings_router(config_manager))
    return TestClient(app), config_manager


def _cache_root(config_manager: ConfigManager) -> Path:
    cfg = config_manager.get_config()
    return cfg.label_input_path_for(WS, EXP).parent.parent / "cache"


def _write_labels(config_manager: ConfigManager, count: int) -> None:
    cfg = config_manager.get_config()
    label_path = cfg.label_input_path_for(WS, EXP)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    docs = {
        "meta": {},
        "items": [
            {"id": i, "file_id": f"img_{i}", "path": f"a/img_{i}.png", "labels": {}}
            for i in range(count)
        ],
    }
    label_path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")


def _write_cache(
    config_manager: ConfigManager,
    slug: str,
    *,
    model_name: str | None = "ViT-B/32",
    count: int = 3,
    dim: int | None = 512,
    version: str = "1",
    with_matrix: bool = True,
    broken_index: bool = False,
) -> Path:
    d = _cache_root(config_manager) / "embeddings" / slug
    d.mkdir(parents=True, exist_ok=True)
    if broken_index:
        (d / "index.json").write_text("{ not json", encoding="utf-8")
    else:
        payload: dict = {
            "version": version,
            "clip_model_name": model_name,
            "dim": dim,
            "count": count,
            "items": [{"file_id": f"img_{i}", "row": i} for i in range(count)],
        }
        (d / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    if with_matrix:
        # 中身は読まれない。存在だけが `usable` の条件。
        (d / "embeddings.npy").write_bytes(b"\x00")
    return d


def test_returns_empty_when_no_cache(tmp_path):
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 5)

    body = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()

    assert body == {"target_count": 5, "models": []}


def test_lists_two_models_with_counts_from_index(tmp_path):
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 10)
    _write_cache(cm, "vit_b_32", model_name="ViT-B/32", count=10, dim=512)
    _write_cache(cm, "vit_l_14_336px", model_name="ViT-L/14@336px", count=4, dim=768)

    body = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()

    assert body["target_count"] == 10
    by_name = {m["clip_model_name"]: m for m in body["models"]}
    assert set(by_name) == {"ViT-B/32", "ViT-L/14@336px"}
    assert by_name["ViT-B/32"]["count"] == 10
    assert by_name["ViT-B/32"]["dim"] == 512
    assert by_name["ViT-B/32"]["slug"] == "vit_b_32"
    assert by_name["ViT-L/14@336px"]["count"] == 4
    assert by_name["ViT-L/14@336px"]["dim"] == 768
    assert all(m["updated_at"] for m in body["models"])


def test_usable_is_false_without_matrix(tmp_path):
    """index.json だけでは predict_labels も clusters も動かない。"""
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "vit_b_32", with_matrix=False)

    models = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()["models"]

    assert len(models) == 1
    assert models[0]["usable"] is False


def test_usable_is_true_with_matrix(tmp_path):
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "vit_b_32", with_matrix=True)

    models = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()["models"]

    assert models[0]["usable"] is True


def test_broken_index_is_ignored_without_failing(tmp_path):
    """壊れた index を持つディレクトリがあっても、他のモデルは返る。"""
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "broken", broken_index=True)
    _write_cache(cm, "vit_b_32", model_name="ViT-B/32")

    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings")

    assert res.status_code == 200
    assert [m["clip_model_name"] for m in res.json()["models"]] == ["ViT-B/32"]


def test_stale_cache_version_is_ignored(tmp_path):
    """版が違うキャッシュは embed_images が作り直す。「無い」と扱う。"""
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "vit_b_32", version="0")

    assert client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()["models"] == []


def test_cache_without_model_name_is_ignored(tmp_path):
    """モデル名が分からないキャッシュは、どの run に対応するか判定できない。"""
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "unknown", model_name="")

    assert client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()["models"] == []


def test_target_count_is_zero_when_labels_missing(tmp_path):
    """labels.json が無い experiment でも 200 を返す（UI の初回表示で叩かれる）。"""
    client, cm = _make_client(tmp_path)
    _write_cache(cm, "vit_b_32")

    body = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings").json()

    assert body["target_count"] == 0
    assert len(body["models"]) == 1


def test_files_in_embeddings_dir_are_skipped(tmp_path):
    """ディレクトリ以外（.DS_Store など）が混ざっても落ちない。"""
    client, cm = _make_client(tmp_path)
    _write_labels(cm, 3)
    _write_cache(cm, "vit_b_32")
    (_cache_root(cm) / "embeddings" / ".DS_Store").write_bytes(b"\x00")

    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/embeddings")

    assert res.status_code == 200
    assert len(res.json()["models"]) == 1
