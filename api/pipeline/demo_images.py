"""デモ用サンプル画像の生成。

初見のユーザが「自分の画像を用意する」前に、xima が何をできるのかを
数分で体感できるようにするためのサンプルデータを手続き的に生成する。

**画像をリポジトリに同梱せず、実行時に生成する**方針を採っている。

- ライセンス上安全（第三者素材を一切含まない）
- リポジトリにバイナリを増やさない
- オフラインで完結し、常に同じデータが再現できる（seed 固定）

生成されるのは 3 クラス（circle / square / triangle）の幾何図形で、
CLIP + Linear head で高い精度が出るため「学習が回って精度が上がる」ことを
短時間で確認できる。実データ（イラスト等）の代替ではない点には注意。

torch に依存しない（PIL のみ）ため、そのまま単体テストできる。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

# デモの分類クラス。ディレクトリ名としてもそのまま使う。
DEMO_CLASSES: List[str] = ["circle", "square", "triangle"]

DEFAULT_IMAGES_PER_CLASS = 12
IMAGE_SIZE = 160
DEMO_SEED = 20260725


def _draw_shape(draw: ImageDraw.ImageDraw, kind: str, rng: random.Random) -> None:
    """クラスごとに、位置と色を少しゆらした図形を描く。

    まったく同一の画像だと学習が自明になりすぎるため、
    ゆらぎを入れて「それらしい」データセットにする。
    """
    jx = rng.randint(-12, 12)
    jy = rng.randint(-12, 12)

    if kind == "circle":
        color = (rng.randint(180, 240), rng.randint(30, 80), rng.randint(30, 80))
        draw.ellipse([30 + jx, 30 + jy, 130 + jx, 130 + jy], fill=color)
    elif kind == "square":
        color = (rng.randint(30, 80), rng.randint(150, 220), rng.randint(60, 110))
        draw.rectangle([35 + jx, 35 + jy, 125 + jx, 125 + jy], fill=color)
    elif kind == "triangle":
        color = (rng.randint(40, 90), rng.randint(70, 120), rng.randint(190, 240))
        draw.polygon(
            [(80 + jx, 28 + jy), (128 + jx, 128 + jy), (32 + jx, 128 + jy)],
            fill=color,
        )
    else:  # pragma: no cover - DEMO_CLASSES 以外は呼ばれない
        raise ValueError(f"unknown demo class: {kind}")


def generate_demo_images(
    source_root: Path,
    *,
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    seed: int = DEMO_SEED,
) -> Dict[str, int]:
    """`source_root/<class>/<class>_NNN.png` を生成し、クラス別の枚数を返す。

    source_root は workspace の source_images を想定している。
    既存ファイルは上書きするが、無関係なファイルは触らない。
    """
    if images_per_class < 1:
        raise ValueError("images_per_class must be >= 1")

    rng = random.Random(seed)
    counts: Dict[str, int] = {}

    for kind in DEMO_CLASSES:
        class_dir = source_root / kind
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (250, 250, 250))
            _draw_shape(ImageDraw.Draw(image), kind, rng)
            image.save(class_dir / f"{kind}_{i:03d}.png")
        counts[kind] = images_per_class

    return counts


def build_demo_label_schema() -> dict:
    """デモ用の label schema。

    `split` に加えて、生成した 3 クラスをそのまま選べる `shape` head を持たせ、
    ラベリング → 学習 → 精度確認まで追加設定なしで通せるようにする。
    """
    return {
        "version": 2,
        "schema_id": "demo_shapes_v1",
        "heads": [
            {
                "id": "split",
                "label": "データ用途",
                "type": "split",
                "choices": ["train", "val", "unassigned"],
            },
            {
                "id": "shape",
                "label": "図形",
                "type": "multi_class",
                "classes": list(DEMO_CLASSES),
            },
        ],
    }
