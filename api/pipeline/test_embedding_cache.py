"""embedding_cache（埋め込みキャッシュのレイアウトと差分計画）の単体テスト。

対象は torch / numpy 非依存のため、CI（軽量依存）でもそのまま実行できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipeline.embedding_cache import (  # noqa: E402
    EMBEDDING_CACHE_VERSION,
    build_index_entries,
    build_index_payload,
    build_output_plan,
    content_hash,
    embeddings_dir,
    load_index,
    model_slug,
    plan_embeddings,
    save_index,
)


def _target(file_id: str, digest: str, path: str | None = None) -> dict:
    return {
        "file_id": file_id,
        "path": path or f"{file_id}.png",
        "content_hash": digest,
    }


def test_model_slug_is_filesystem_safe() -> None:
    assert model_slug("ViT-B/32") == "ViT-B_32"
    assert model_slug("RN50x4") == "RN50x4"
    # 空/未知はディレクトリ名として破綻しない値になる
    assert model_slug("") == "unknown"


def test_embeddings_dir_separates_models(tmp_path: Path) -> None:
    a = embeddings_dir(tmp_path, "ViT-B/32")
    b = embeddings_dir(tmp_path, "ViT-L/14")
    assert a != b
    assert a.parent == tmp_path / "embeddings"


def test_content_hash_detects_change(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    first = content_hash(p)
    assert first == content_hash(p)

    p.write_bytes(b"hello!")
    assert content_hash(p) != first


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    payload = build_index_payload(
        model_name="ViT-B/32",
        dim=4,
        entries=[{"file_id": "a", "path": "a.png", "content_hash": "h1", "row": 0}],
    )
    save_index(tmp_path, payload)

    loaded = load_index(tmp_path)
    assert loaded is not None
    assert loaded["version"] == EMBEDDING_CACHE_VERSION
    assert loaded["clip_model_name"] == "ViT-B/32"
    assert loaded["dim"] == 4
    assert loaded["count"] == 1
    assert loaded["items"][0]["file_id"] == "a"


def test_load_index_missing_or_broken_is_none(tmp_path: Path) -> None:
    assert load_index(tmp_path) is None

    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    assert load_index(tmp_path) is None


def test_load_index_rejects_other_version(tmp_path: Path) -> None:
    save_index(tmp_path, {"version": "999", "items": []})
    # 版が違うキャッシュは作り直す（None 扱い）
    assert load_index(tmp_path) is None


def test_plan_without_existing_cache_embeds_everything() -> None:
    targets = [_target("a", "h1"), _target("b", "h2")]
    plan = plan_embeddings(targets, None)

    assert [t["file_id"] for t in plan.to_embed] == ["a", "b"]
    assert plan.reuse == []
    assert plan.dropped == []
    assert plan.total == 2


def test_plan_reuses_unchanged_and_reembeds_changed() -> None:
    existing = build_index_payload(
        model_name="ViT-B/32",
        dim=4,
        entries=[
            {"file_id": "a", "path": "a.png", "content_hash": "h1", "row": 0},
            {"file_id": "b", "path": "b.png", "content_hash": "h2", "row": 1},
        ],
    )
    targets = [
        _target("a", "h1"),  # 変更なし -> reuse
        _target("b", "CHANGED"),  # 内容が変わった -> 再計算
        _target("c", "h3"),  # 新規 -> 計算
    ]

    plan = plan_embeddings(targets, existing)

    assert [t["file_id"] for t in plan.reuse] == ["a"]
    # 既存行を流用できるよう、元の row を保持している
    assert plan.reuse[0]["source_row"] == 0
    assert sorted(t["file_id"] for t in plan.to_embed) == ["b", "c"]
    assert plan.dropped == []


def test_plan_reports_dropped_files() -> None:
    existing = build_index_payload(
        model_name="ViT-B/32",
        dim=4,
        entries=[
            {"file_id": "a", "path": "a.png", "content_hash": "h1", "row": 0},
            {"file_id": "gone", "path": "gone.png", "content_hash": "h9", "row": 1},
        ],
    )
    plan = plan_embeddings([_target("a", "h1")], existing)

    assert plan.dropped == ["gone"]
    assert [t["file_id"] for t in plan.reuse] == ["a"]


def test_plan_ignores_entries_without_usable_row() -> None:
    # row が壊れている既存エントリは信用せず再計算する
    existing = {
        "version": EMBEDDING_CACHE_VERSION,
        "items": [{"file_id": "a", "content_hash": "h1", "row": None}],
    }
    plan = plan_embeddings([_target("a", "h1")], existing)

    assert [t["file_id"] for t in plan.to_embed] == ["a"]
    assert plan.reuse == []


def test_build_output_plan_assigns_sequential_rows_and_sources() -> None:
    existing = build_index_payload(
        model_name="ViT-B/32",
        dim=4,
        entries=[
            # 既存では row=5 に入っている（行番号は詰め直される）
            {"file_id": "keep", "path": "keep.png", "content_hash": "h1", "row": 5},
        ],
    )
    targets = [_target("keep", "h1"), _target("new1", "h2"), _target("new2", "h3")]

    output = build_output_plan(plan_embeddings(targets, existing))

    # 出力順は「流用 -> 新規」、row は 0 から連番に振り直される
    assert [t["file_id"] for t in output] == ["keep", "new1", "new2"]
    assert [t["row"] for t in output] == [0, 1, 2]

    # 流用分は既存行列の元の行を指す
    assert output[0]["source"] == "reuse"
    assert output[0]["source_row"] == 5

    # 新規分は「新規計算した行列」の 0 始まりの行を指す
    assert [t["source"] for t in output[1:]] == ["new", "new"]
    assert [t["source_row"] for t in output[1:]] == [0, 1]


def test_build_output_plan_without_reuse_is_all_new() -> None:
    output = build_output_plan(plan_embeddings([_target("a", "h1")], None))

    assert output[0]["source"] == "new"
    assert output[0]["source_row"] == 0
    assert output[0]["row"] == 0


def test_build_index_entries_drops_internal_keys() -> None:
    output = build_output_plan(plan_embeddings([_target("a", "h1")], None))
    entries = build_index_entries(output)

    assert entries == [
        {"file_id": "a", "path": "a.png", "content_hash": "h1", "row": 0}
    ]
    # source / source_row といった内部用キーは index.json に載せない
    assert "source" not in entries[0]
    assert "source_row" not in entries[0]
