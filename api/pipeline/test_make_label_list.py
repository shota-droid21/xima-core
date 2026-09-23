"""同名の別画像が 1 件に畳まれないこと（#409 / 再現は #380）。

直す前は `file_id` がファイル名の語幹だけで作られていた。`a/photo.png` と
`b/photo.png` がどちらも `photo` になり、突き合わせの dict で**後から来た方が
前を上書き**していた。一覧は 2 枚に対して 1 件になり、**先に付けたラベルが
別の画像へ移っていた。**

この検査が守るのは 4 つ。

1. 同名の別画像が**別の item として残る**
2. 既存のラベルが、**移行を走らせずに**新しい鍵の上へ引き継がれる
3. ラベルが**別の画像へ移らない**
4. サムネイルの置き場が**別の画像と重ならない**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from label_list_merge import merge_items
from make_label_list import cache_path, collect_images, load_existing


def _put(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a real image, but make_label_list only looks at the suffix")
    return p


def _write_labels(out_path: Path, items: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"meta": {}, "items": items}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_same_named_images_stay_separate(tmp_path: Path) -> None:
    _put(tmp_path, "a/photo.png")
    _put(tmp_path, "b/photo.png")

    items = collect_images(tmp_path, tmp_path)

    assert len(items) == 2
    assert {it["file_id"] for it in items} == {"a/photo.png", "b/photo.png"}


def test_label_does_not_move_to_another_image(tmp_path: Path) -> None:
    """#380 の再現。`a` に付けたラベルが `b` へ移ってはならない。"""
    _put(tmp_path, "a/photo.png")
    _put(tmp_path, "b/photo.png")

    out_path = tmp_path / "experiments" / "exp1" / "labels.json"
    # 旧版が書いた形。鍵は語幹だけで、item は 1 件しか無い。
    _write_labels(
        out_path,
        [
            {
                "id": 0,
                "file_id": "photo",
                "path": "a/photo.png",
                "labels": {"shape": "circle"},
            }
        ],
    )

    _meta, existing = load_existing(out_path)
    merged, collapsed = merge_items(collect_images(tmp_path, tmp_path), existing)

    assert collapsed == 0, "file_id が重なって item が失われた"
    assert len(merged) == 2

    by_path = {it["path"]: it for it in merged}
    assert by_path["a/photo.png"].get("labels") == {"shape": "circle"}
    assert not by_path["b/photo.png"].get("labels"), "別画像にラベルが移った"


def test_existing_labels_survive_the_key_change(tmp_path: Path) -> None:
    """旧 `file_id` のままの `labels.json` を読んでもラベルを失わない。

    移行スクリプトは要らない。`path` から鍵を引き直すことで突き合わせが続く。
    """
    _put(tmp_path, "circle/circle_001.png")

    out_path = tmp_path / "experiments" / "exp1" / "labels.json"
    _write_labels(
        out_path,
        [
            {
                "id": 0,
                "file_id": "circle_001",  # 旧版の鍵
                "path": "circle/circle_001.png",
                "labels": {"shape": "circle"},
            }
        ],
    )

    _meta, existing = load_existing(out_path)
    assert set(existing) == {"circle/circle_001.png"}, "鍵が引き直されていない"

    merged, _ = merge_items(collect_images(tmp_path, tmp_path), existing)

    assert len(merged) == 1
    assert merged[0]["labels"] == {"shape": "circle"}
    assert merged[0]["file_id"] == "circle/circle_001.png"


def test_thumbnail_locations_do_not_collide(tmp_path: Path) -> None:
    _put(tmp_path, "a/photo.png")
    _put(tmp_path, "b/photo.png")

    cache_root = tmp_path / "cache" / "thumbs"
    paths = {
        cache_path(cache_root, 256, it["file_id"])
        for it in collect_images(tmp_path, tmp_path)
    }

    assert len(paths) == 2, "同名の別画像が同じサムネイルを指している"


def test_item_without_path_keeps_a_key_of_its_own(tmp_path: Path) -> None:
    """`path` を持たない item があっても、他の item と混ざらない。"""
    out_path = tmp_path / "experiments" / "exp1" / "labels.json"
    _write_labels(
        out_path,
        [
            {"id": 0, "file_id": "orphan", "labels": {"shape": "circle"}},
            {"id": 1, "file_id": "other", "labels": {"shape": "square"}},
        ],
    )

    _meta, existing = load_existing(out_path)

    assert len(existing) == 2
