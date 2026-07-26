"""デモ用サンプル画像生成の単体テスト（torch 非依存）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipeline.demo_images import (  # noqa: E402
    DEMO_CLASSES,
    build_demo_label_schema,
    generate_demo_images,
)


def test_generate_creates_one_dir_per_class(tmp_path: Path) -> None:
    counts = generate_demo_images(tmp_path, images_per_class=3)

    assert sorted(counts) == sorted(DEMO_CLASSES)
    for kind in DEMO_CLASSES:
        files = sorted((tmp_path / kind).glob("*.png"))
        assert len(files) == 3
        assert files[0].name == f"{kind}_000.png"


def test_generated_images_are_readable_and_sized(tmp_path: Path) -> None:
    from PIL import Image

    generate_demo_images(tmp_path, images_per_class=1)
    path = next((tmp_path / DEMO_CLASSES[0]).glob("*.png"))

    with Image.open(path) as im:
        assert im.size == (160, 160)
        assert im.mode == "RGB"


def test_generation_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_demo_images(a, images_per_class=2)
    generate_demo_images(b, images_per_class=2)

    for kind in DEMO_CLASSES:
        for name in ("000", "001"):
            fa = (a / kind / f"{kind}_{name}.png").read_bytes()
            fb = (b / kind / f"{kind}_{name}.png").read_bytes()
            assert fa == fb, f"{kind}_{name} should be reproducible"


def test_classes_are_visually_distinct(tmp_path: Path) -> None:
    # 学習が成立するよう、クラス間で画像が異なることを担保する
    generate_demo_images(tmp_path, images_per_class=1)
    blobs = {
        kind: next((tmp_path / kind).glob("*.png")).read_bytes()
        for kind in DEMO_CLASSES
    }
    assert len(set(blobs.values())) == len(DEMO_CLASSES)


def test_rejects_invalid_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_demo_images(tmp_path, images_per_class=0)


def test_demo_schema_covers_generated_classes() -> None:
    schema = build_demo_label_schema()
    heads = {h["id"]: h for h in schema["heads"]}

    assert heads["split"]["type"] == "split"
    # 生成したクラスがそのまま選べること（追加設定なしで学習まで通せる）
    assert heads["shape"]["type"] == "multi_class"
    assert heads["shape"]["classes"] == list(DEMO_CLASSES)
