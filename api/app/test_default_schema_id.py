"""既定スキーマの識別子（#251 B-1）。

以前はここに開発者の非公開データ由来の値が入っており、experiment を作った
利用者全員の label_schema.json に書き込まれていた。中立な値であること、
そして core の 2 箇所が食い違わないことを固定する。
"""

from __future__ import annotations

from app.experiments import DEFAULT_LABEL_SCHEMA
from app.label_input import _schema_template

# 開発者の非公開データ由来だった語。CI の禁止語検査（#251）と同じ対象。
FORBIDDEN = ("vtuber", "poc_ws", "hair_color", "eye_color", "concept_color")


def test_default_schema_id_is_neutral() -> None:
    schema_id = str(DEFAULT_LABEL_SCHEMA["schema_id"]).lower()
    for word in FORBIDDEN:
        assert word not in schema_id


def test_template_matches_experiment_default() -> None:
    """core の 2 箇所は同じ識別子を使う。片方だけ変えると利用者の目に食い違いが出る。"""
    assert _schema_template()["schema_id"] == DEFAULT_LABEL_SCHEMA["schema_id"]


def test_default_schema_has_only_the_reserved_split_head() -> None:
    heads = DEFAULT_LABEL_SCHEMA["heads"]
    assert [h["id"] for h in heads] == ["split"]
