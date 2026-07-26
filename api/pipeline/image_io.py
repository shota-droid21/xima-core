"""画像読み込みの共通処理。

train_epoch / infer_heads で同一実装が重複していたため、
embed_images の追加にあたって共通化した（挙動は変更なし）。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image_rgb(path: Path) -> Image.Image:
    """RGB 画像として読み込む。アルファがある場合は白背景に合成する。"""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    else:
        im = im.convert("RGB")
    return im
