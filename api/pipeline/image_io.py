"""画像読み込みの共通処理。

train_epoch / infer_heads で同一実装が重複していたため、
embed_images の追加にあたって共通化した。
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


def is_decodable_image(path: Path) -> bool:
    """その画像が実際に読めるか。**中身を全部デコードせずに判定する。**

    なぜ要るか:
        `source_images/` はユーザーが自分で集めたディレクトリであり、拡張子が `.jpg` でも
        中身が画像とは限らない。実データで、取得に失敗した HTML（レート制限やエラーページ）が
        836 KB の `.jpg` として保存されている例が **879 枚中 21 枚**あった。

        1 枚でも読めないと `Image.open` が `UnidentifiedImageError` を投げ、
        DataLoader 経由で**ジョブ全体が落ちる**。1,800 枚を回している途中で 1 枚のゴミに
        当たって全部やり直しになるのは、ローカルで自分のデータを扱う道具として成立しない。

        1 枚消しても次の 1 枚で同じことが起きるため、「消して回避」は解にならない。

    `Image.open` はヘッダを読んで形式を判定するところまでしか行わないため、
    ファイル全体のデコードは走らない。`verify()` も構造チェックのみ。
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 — 読めない理由は問わない。読めないことだけが判断材料
        return False
