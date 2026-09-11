from __future__ import annotations

import json

import pytest

from app.label_input import (
    normalize_label_input_payload_with_schema,
    normalize_label_thumb_paths,
    thumb_path_from_file_id,
)


def _schema() -> dict:
    return {
        "version": 2,
        "heads": [
            {"id": "character", "type": "multi_class", "classes": ["alice", "bob"]},
            {
                "id": "tags",
                "type": "multi_label",
                "classes": ["cute", "cool", "mature"],
            },
        ],
    }


def test_normalize_label_input_payload_with_schema_normalizes_values() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "character": "alice",
                "labels": {
                    "split": " Train ",
                    "character": " alice ",
                    "tags": ["cool", "cute", "cool"],
                },
            }
        ],
        "meta": {"version": "mvp"},
    }

    normalized = normalize_label_input_payload_with_schema(payload, _schema())
    labels = normalized["items"][0]["labels"]

    assert labels["split"] == "train"
    assert labels["character"] == "alice"
    # multi_label is normalized to schema.classes order.
    assert labels["tags"] == ["cute", "cool"]


def test_normalize_label_input_payload_with_schema_maps_legacy_ignore_to_exclude() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "labels": {"split": "ignore", "character": "alice"},
            }
        ],
        "meta": {},
    }

    normalized = normalize_label_input_payload_with_schema(payload, _schema())
    item = normalized["items"][0]
    labels = item["labels"]

    # `ignore` は語が「外す」を意味するので `exclude` として読む（#325）。
    assert labels["split"] == "exclude"
    assert item.get("delete") is None


def test_normalize_label_input_payload_with_schema_maps_legacy_delete_to_flag() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "labels": {"split": "delete", "character": "alice"},
            }
        ],
        "meta": {},
    }

    normalized = normalize_label_input_payload_with_schema(payload, _schema())
    item = normalized["items"][0]
    labels = item["labels"]

    # `delete` の意味は削除マークが担うので、`split` はキーごと落ちる（#325）。
    assert "split" not in labels
    assert item["delete"] is True


def test_normalize_label_input_payload_with_schema_accepts_delete_flag_string_true() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "delete": "true",
                "labels": {"split": "train", "character": "alice"},
            }
        ],
        "meta": {},
    }

    normalized = normalize_label_input_payload_with_schema(payload, _schema())
    item = normalized["items"][0]

    assert item["delete"] is True


def test_normalize_label_input_payload_with_schema_rejects_unknown_head() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "labels": {"split": "train", "legacy_head": "x"},
            }
        ],
        "meta": {},
    }

    with pytest.raises(ValueError, match="unknown head ids"):
        normalize_label_input_payload_with_schema(payload, _schema())


def test_normalize_label_input_payload_with_schema_rejects_invalid_multiclass_type() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "labels": {"split": "train", "character": ["alice"]},
            }
        ],
        "meta": {},
    }

    with pytest.raises(ValueError, match="character must be a string"):
        normalize_label_input_payload_with_schema(payload, _schema())


def test_normalize_label_input_payload_with_schema_rejects_unknown_multilabel_class() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "labels": {"split": "train", "tags": ["cute", "unknown"]},
            }
        ],
        "meta": {},
    }

    with pytest.raises(ValueError, match="contains unknown labels"):
        normalize_label_input_payload_with_schema(payload, _schema())


def test_normalize_label_input_payload_with_schema_accepts_legacy_top_level_labels() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "split": "val",
                "character": "bob",
                "labels": {},
            }
        ],
        "meta": {},
    }

    normalized = normalize_label_input_payload_with_schema(payload, _schema())
    labels = normalized["items"][0]["labels"]
    assert labels["split"] == "val"
    assert labels["character"] == "bob"


def test_thumb_path_from_file_id_uses_workspace_experiment_and_sha1() -> None:
    path = thumb_path_from_file_id("uybwcicl", "a8h131ej", "a")
    assert path == (
        "/static/uybwcicl/experiments/a8h131ej/cache/thumbs/w256/"
        "86/86f7e437faa5a7fce15d1ddcb9eaeaea377667b8.webp"
    )


def test_normalize_label_thumb_paths_rewrites_legacy_thumb_path() -> None:
    payload = {
        "items": [
            {
                "id": "1",
                "file_id": "a",
                "thumb_path": "/static/ws_test/experiments/exp1/cache/thumbs/w256/xx/legacy.webp",
            },
            {"id": "2", "file_id": "b"},
        ],
        "meta": {},
    }

    normalized = normalize_label_thumb_paths(
        payload, workspace="uybwcicl", experiment="a8h131ej"
    )
    assert normalized["items"][0]["thumb_path"] == (
        "/static/uybwcicl/experiments/a8h131ej/cache/thumbs/w256/"
        "86/86f7e437faa5a7fce15d1ddcb9eaeaea377667b8.webp"
    )
    assert normalized["items"][1]["thumb_path"] == (
        "/static/uybwcicl/experiments/a8h131ej/cache/thumbs/w256/"
        "e9/e9d71f5ee7c92d6dc9e92ffdad17b8bd49418f98.webp"
    )


def test_label_revision_changes_on_any_write(tmp_path):
    """版は **PUT を通らない書き換えでも変わる**（#294）。

    make_label_list / apply_label / purge_deleted_images はファイルを直接書く。
    PUT の回数を数える方式だと、それらを取りこぼす。
    """
    from app.label_input import label_revision

    path = tmp_path / "labels.json"
    assert label_revision(path) == ""  # まだ無い

    path.write_text(json.dumps({"items": [], "meta": {}}), encoding="utf-8")
    first = label_revision(path)
    assert first != ""

    # 直接書き換える（ジョブがやること）
    path.write_text(
        json.dumps({"items": [{"id": 0}], "meta": {}}), encoding="utf-8"
    )
    assert label_revision(path) != first


def test_label_revision_is_stable_without_writes(tmp_path):
    from app.label_input import label_revision

    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"items": [], "meta": {}}), encoding="utf-8")
    assert label_revision(path) == label_revision(path)
