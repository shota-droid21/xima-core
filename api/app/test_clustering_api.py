"""クラスタリング API の結合テスト。

行列読み込み（numpy）はスタブ化し、CI（numpy 無し）でもルーターの配線・
スコープ絞り込み・レスポンス整形を検証できるようにする。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.clustering import create_clustering_router
from app.config import ConfigManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from embedding_cache import embeddings_dir, model_slug  # noqa: E402

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


def _write_labels(config_manager, items, labeled_head_by_fid, delete_fids=()):
    """labels.json / label_schema.json を用意する。

    `delete_fids` は削除マークを付ける file_id（#293）。
    """
    cfg = config_manager.get_config()
    label_path = cfg.label_input_path_for(WS, EXP)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    marked = set(delete_fids)
    docs = {
        "meta": {},
        "items": [
            {
                "id": i,
                "file_id": it["file_id"],
                "path": it["path"],
                "labels": labeled_head_by_fid.get(it["file_id"], {}),
                **({"delete": True} if it["file_id"] in marked else {}),
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


def _write_cache(config_manager, model_name: str, count: int = 9) -> Path:
    """埋め込みキャッシュの体裁だけ作る（#261 のモデル解決に要る）。

    行列の中身は `_load_embeddings` をスタブ化するので読まれない。
    ここで作るのは「そのモデルの埋め込みが在る」ことを示す index.json と
    embeddings.npy の 2 ファイル。

    **ディレクトリ名は `embeddings_dir` と同じ規則で決める。** 手で書くと
    解決先とずれ、スタブに隠れて気づけない。
    """
    cfg = config_manager.get_config()
    cache_root = cfg.label_input_path_for(WS, EXP).parent.parent / "cache"
    d = embeddings_dir(cache_root, model_name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_text(
        json.dumps(
            {
                "version": "1",
                "clip_model_name": model_name,
                "dim": 2,
                "count": count,
                "items": [{"file_id": f"img_{i}", "row": i} for i in range(count)],
            }
        ),
        encoding="utf-8",
    )
    (d / "embeddings.npy").write_bytes(b"\x00")
    return d

def test_returns_409_when_embeddings_missing(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path)
    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/clusters")
    assert res.status_code == 409
    assert "embed" in res.json()["detail"].lower()


def test_clusters_group_similar_vectors(tmp_path, monkeypatch):
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_cache(cfg, "ViT-B/32")
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
    _write_cache(cfg, "ViT-B/32")
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
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_cache(cfg, "ViT-B/32")
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))
    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters", params={"scope": "bogus"}
    )
    assert res.status_code == 400


def test_uses_requested_clip_model(tmp_path, monkeypatch):
    """clip_model を指定すると、そのモデルのキャッシュを読みに行く。"""
    seen: list[str] = []
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_cache(cfg, "ViT-B/32")
    _write_cache(cfg, "ViT-L/14@336px")

    def _stub(cache_dir):
        seen.append(cache_dir.name)
        return items, vectors

    monkeypatch.setattr("app.clustering._load_embeddings", _stub)

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters",
        params={"clip_model": "ViT-L/14@336px"},
    )

    assert res.status_code == 200
    assert res.json()["clip_model_name"] == "ViT-L/14@336px"
    assert seen == [model_slug("ViT-L/14@336px")]


def test_defaults_to_newest_embeddings_not_vit_b_32(tmp_path, monkeypatch):
    """指定が無いとき **ViT-B/32 に落とさない**。最も新しい埋め込みを使う。

    これが #261 の本体。以前は定数で ViT-B/32 に固定しており、
    学習に別モデルを使っていてもクラスタだけ別の空間で切られていた。
    """
    import os
    import time

    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_cache(cfg, "ViT-B/32")
    vit_l_dir = _write_cache(cfg, "ViT-L/14@336px")

    # ViT-L の index.json を新しくする（mtime が updated_at の元）。
    newer = time.time() + 60
    os.utime(vit_l_dir / "index.json", (newer, newer))

    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    res = client.get(f"/workspaces/{WS}/experiments/{EXP}/clusters")

    assert res.status_code == 200
    assert res.json()["clip_model_name"] == "ViT-L/14@336px"


def test_409_names_the_required_model(tmp_path, monkeypatch):
    """どのモデルの埋め込みが要るかを 409 に含める。"""
    client, cfg = _make_client(tmp_path)
    _write_cache(cfg, "ViT-B/32")

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters",
        params={"clip_model": "ViT-L/14@336px"},
    )

    assert res.status_code == 409
    assert "ViT-L/14@336px" in res.json()["detail"]


def test_scope_unlabeled_excludes_delete_marked_items(tmp_path, monkeypatch):
    """削除マークを付けたものは「残りに付ける対象」ではない（#293）。

    以前は `labels[head]` しか見ておらず、**消すと決めた画像が出続けていた**。
    重複を削除マークで片付けても視界から消えないため、まとまりを 1 つずつ
    処理する使い方が成立しなかった。
    """
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    # img_1 はラベル無し・削除マークあり → 除外されるはず
    _write_labels(cfg, items, {}, delete_fids=["img_1"])
    _write_cache(cfg, "ViT-B/32")
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters",
        params={"scope": "unlabeled"},
    )
    assert res.status_code == 200
    body = res.json()
    seen = [m["file_id"] for c in body["clusters"] for m in c["members"]]
    assert "img_1" not in seen
    assert body["total"] == len(items) - 1


def test_scope_all_keeps_delete_marked_items(tmp_path, monkeypatch):
    """`scope=all` は絞り込まない。削除マークでも消さない。"""
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_labels(cfg, items, {}, delete_fids=["img_1"])
    _write_cache(cfg, "ViT-B/32")
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    body = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters", params={"scope": "all"}
    ).json()
    seen = [m["file_id"] for c in body["clusters"] for m in c["members"]]
    assert "img_1" in seen
    assert body["total"] == len(items)


def test_scope_labeled_unassigned_filters_end_to_end(tmp_path, monkeypatch):
    """`labeled_unassigned` がルーターまで通っていることを見る（#316）。

    誰が残るかの境界は `test_cluster_scope.py` が持つ。ここで見るのは配線
    （クエリが届き、items と vectors が同じ形で絞られ、応答に scope が返る）。
    """
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_labels(
        cfg,
        items,
        {
            # 項目に値があり、人が学習から外した → 残る（#325 で意味が変わった）
            "img_0": {"shape": "a", "split": "exclude"},
            "img_1": {"shape": "b", "split": "ignore"},
            # 書かなければ学習に入る → 除く
            "img_2": {"shape": "a"},
            "img_3": {"shape": "a", "split": "train"},
            # split だけ → ラベル済みではない
            "img_4": {"split": "exclude"},
        },
    )
    _write_cache(cfg, "ViT-B/32")
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    res = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters",
        params={"scope": "labeled_unassigned"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["scope"] == "labeled_unassigned"
    # head は絞り込みに使わないので、core が勝手に埋めない。
    assert body["head"] is None
    assert body["total"] == 2
    seen = sorted(m["file_id"] for c in body["clusters"] for m in c["members"])
    assert seen == ["img_0", "img_1"]


def test_bad_scope_message_lists_every_scope(tmp_path, monkeypatch):
    """400 のメッセージから、指定できる値が全部わかること。

    値を増やしたときにメッセージだけ古いままになると、`scope` が 2 つしか無いと
    読める。
    """
    client, cfg = _make_client(tmp_path)
    items, vectors = _blob_items()
    _write_cache(cfg, "ViT-B/32")
    monkeypatch.setattr("app.clustering._load_embeddings", lambda _d: (items, vectors))

    detail = client.get(
        f"/workspaces/{WS}/experiments/{EXP}/clusters", params={"scope": "bogus"}
    ).json()["detail"]
    for scope in ("all", "unlabeled", "labeled_unassigned"):
        assert scope in detail
