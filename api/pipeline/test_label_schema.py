from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipeline.label_schema import (
    normalize_label_for_head,
    normalize_schema,
    validate_schema,
)


def test_normalize_schema_converts_single_class_to_multi_class() -> None:
    schema = {
        "version": 2,
        "heads": [
            {"id": "character", "type": "single_class", "choices": ["a", "b"]},
            {"id": "split", "type": "split", "choices": ["train", "val"]},
        ],
    }

    normalized = normalize_schema(schema)
    heads = {h["id"]: h for h in normalized["heads"]}

    assert heads["character"]["type"] == "multi_class"
    assert heads["character"]["classes"] == ["a", "b"]


def test_normalize_label_for_head_multilabel_filters_unknown_by_schema_classes() -> None:
    head = {"id": "tags", "type": "multi_label", "classes": ["cat", "dog", "bird"]}

    out = normalize_label_for_head(["dog", "unknown", "cat", "dog"], head)

    assert out == ["cat", "dog"]


def test_normalize_label_for_head_multilabel_requires_schema_classes() -> None:
    head = {"id": "tags", "type": "multi_label"}

    out = normalize_label_for_head(["cat", "dog"], head)

    assert out == []


def test_validate_schema_rejects_head_id_with_space() -> None:
    schema = {
        "version": 2,
        "heads": [{"id": "bad id", "type": "multi_class", "classes": ["a", "b"]}],
    }

    with pytest.raises(ValueError, match="id must match"):
        validate_schema(schema)


def test_validate_schema_rejects_head_id_with_fullwidth() -> None:
    schema = {
        "version": 2,
        "heads": [{"id": "キャラ", "type": "multi_class", "classes": ["a", "b"]}],
    }

    with pytest.raises(ValueError, match="id must match"):
        validate_schema(schema)


def test_validate_schema_rejects_duplicate_head_id() -> None:
    schema = {
        "version": 2,
        "heads": [
            {"id": "character", "type": "multi_class", "classes": ["a", "b"]},
            {"id": "character", "type": "multi_class", "classes": ["x", "y"]},
        ],
    }

    with pytest.raises(ValueError, match="duplicate head id"):
        validate_schema(schema)


def test_validate_schema_requires_classes_for_multilabel() -> None:
    schema = {
        "version": 2,
        "heads": [{"id": "tags", "type": "multi_label"}],
    }

    with pytest.raises(ValueError, match="multi_label requires non-empty classes"):
        validate_schema(schema)
