"""feature_cache の単体テスト（torch が要る環境でのみ動かす）。

この refactor で唯一こわいのは **行のずれ**である。ずれても学習は動いてしまい、
「なぜか精度が少し悪い」としか見えない。だから順序と対応を固定する。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from feature_cache import (  # noqa: E402
    build_targets,
    encode_features,
    feature_loader,
    feature_memory_bytes,
)


@dataclass
class FakeItem:
    dataset_path: str
    labels: Dict[str, Any] = field(default_factory=dict)


class MarkerModel(torch.nn.Module):
    """入力の左上ピクセルをそのまま特徴に写す。順序の検証用。"""

    def encode_image(self, xb: torch.Tensor) -> torch.Tensor:
        marker = xb[:, 0, 0, 0].reshape(-1, 1)
        return torch.cat([marker, torch.ones_like(marker)], dim=1)


def _make_images(root: Path, n: int) -> List[FakeItem]:
    """i 番目の画像の左上を i にする。符号化後もその順序が保たれるはず。"""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        img.putpixel((0, 0), (i, i, i))
        img.save(root / f"{i}.png")
        items.append(FakeItem(dataset_path=f"{i}.png"))
    return items


def _preprocess(img):
    import numpy as np

    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)


def _encode(items, root, **kw):
    return encode_features(
        items,
        dataset_root=root,
        clip_preprocess=_preprocess,
        clip_model=MarkerModel(),
        device=torch.device("cpu"),
        num_workers=0,
        pin_memory=False,
        **kw,
    )


def test_encode_features_preserves_item_order(tmp_path: Path):
    items = _make_images(tmp_path, 7)
    # 端数の出るバッチサイズで、バッチ跨ぎのずれを拾う
    feats = _encode(items, tmp_path, batch_size=3)
    assert feats.shape[0] == 7
    # 正規化後も 2 成分の比は marker:1 のままなので、単調増加で順序を確認できる
    ratios = (feats[:, 0] / feats[:, 1]).tolist()
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(0.0)


def test_encode_features_empty_items(tmp_path: Path):
    assert _encode([], tmp_path, batch_size=2).numel() == 0


def test_encode_features_reports_progress(tmp_path: Path):
    items = _make_images(tmp_path, 5)
    seen: List[tuple] = []
    _encode(items, tmp_path, batch_size=2, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (5, 5)


def test_build_targets_multi_class_marks_missing_as_invalid():
    items = [
        FakeItem("0.png", {"h": "cat"}),
        FakeItem("1.png", {}),                # ラベル無し
        FakeItem("2.png", {"h": "unknown"}),  # クラスに無い
        FakeItem("3.png", {"h": "dog"}),
    ]
    y, valid = build_targets(
        items, class_head="h", head_type="multi_class",
        head_schema={"type": "multi_class"}, class_to_idx={"cat": 0, "dog": 1},
    )
    assert y.tolist() == [0, -1, -1, 1]
    assert valid.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_build_targets_multi_label_sets_each_class():
    items = [
        FakeItem("0.png", {"h": ["a", "c"]}),
        FakeItem("1.png", {}),
        FakeItem("2.png", {"h": ["b"]}),
    ]
    # multi_label は schema の classes に照らして正規化される（label_schema の仕様）。
    # main() 側も classes の無い multi_label head は先に弾いている。
    schema = {"type": "multi_label", "classes": ["a", "b", "c"]}
    y, valid = build_targets(
        items, class_head="h", head_type="multi_label",
        head_schema=schema, class_to_idx={"a": 0, "b": 1, "c": 2},
    )
    assert y.tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert valid.tolist() == [1.0, 0.0, 1.0]


def test_build_targets_multi_label_drops_unknown_classes():
    """schema に無いクラスは落ちる。head の出力に対応する列が無いため。"""
    schema = {"type": "multi_label", "classes": ["a", "b"]}
    y, valid = build_targets(
        [FakeItem("0.png", {"h": ["a", "zzz"]})],
        class_head="h", head_type="multi_label",
        head_schema=schema, class_to_idx={"a": 0, "b": 1},
    )
    assert y.tolist() == [[1.0, 0.0]]
    assert valid.tolist() == [1.0]


def test_features_and_targets_share_row_order(tmp_path: Path):
    """**この refactor の要**。i 行目の特徴と i 行目の正解が同じ item を指すこと。"""
    items = _make_images(tmp_path, 4)
    for i, it in enumerate(items):
        it.labels = {"h": "even" if i % 2 == 0 else "odd"}

    feats = _encode(items, tmp_path, batch_size=3)
    y, valid = build_targets(
        items, class_head="h", head_type="multi_class",
        head_schema={"type": "multi_class"}, class_to_idx={"even": 0, "odd": 1},
    )
    assert y.tolist() == [0, 1, 0, 1]
    assert valid.tolist() == [1.0] * 4
    # marker は item の index なので、特徴からも item を復元できる
    ratios = (feats[:, 0] / feats[:, 1]).tolist()
    for i in range(4):
        assert ratios[i] == pytest.approx(float(i), abs=1e-4)


def test_feature_loader_yields_triples():
    feats = torch.randn(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    loader = feature_loader(feats, y, torch.ones(6), batch_size=4, shuffle=False)
    batches = list(loader)
    assert [tuple(b[0].shape) for b in batches] == [(4, 3), (2, 3)]
    assert torch.equal(torch.cat([b[1] for b in batches]), y)


def test_feature_memory_bytes():
    assert feature_memory_bytes(100, 20, 768) == 120 * 768 * 4
