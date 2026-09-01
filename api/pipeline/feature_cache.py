"""CLIP の特徴を **run 内で 1 回だけ** 計算して使い回す（学習の前処理）。

なぜ要るか:

    train_epoch は毎エポック `clip_model.encode_image` を呼んでいた。しかし

      - backbone は `requires_grad_(False)` で凍結され、`eval()` かつ `no_grad` 下
      - CLIP の preprocess は Resize → CenterCrop → ToTensor → Normalize で
        **ランダム要素が無い**

    ため、**毎エポック同じ入力から同じベクトルを再計算していただけ**である
    （実測で preprocess も encode_image もビット単位で同一だった）。しかも head ごとに
    ループしているので、3 head × 30 エポックで 90 回同じ計算を繰り返していた。

    実データ（1,022 枚 / ViT-L/14@336px / MPS）で **67 秒/epoch → 0.02 秒/epoch**。
    30 エポックの学習が 37 分から 1.5 分になる。これは速度だけの話ではない:
    「エポックを増やす」が現実的でなかったために head が未収束のまま出荷されていた。

なぜ既存の cache/embeddings/ を使わないか:

    あれは labels.json の `file_id`（= **元画像**）で引く。一方 dataset には
    gray 増幅で作られた別画像が含まれ、それらは元画像と `source_path` が同じである。
    素朴に source_path で引くと **増幅が黙って消える**（同じ行を重複して学習する）。
    ここは run 内で dataset の画像そのものを符号化する。

なぜメモリに置くか（disk キャッシュにしないか）:

    `dataset/` は apply_label のたびに作り直されるため、disk に置くと無効化の判断を
    抱えることになる。特徴量は N x D の float32 で、実データでは 939 x 768 x 4B = 2.9MB。
    10 万枚でも 307MB に収まる。まずは無効化の要らないメモリで持つ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from image_io import is_decodable_image, load_image_rgb
from label_schema import normalize_label_for_head

# これを超えたら警告を出す。**代替経路は用意しない。**
# 使われない分岐は腐り、いざ通ったときに壊れている（本リポジトリで繰り返した失敗）。
# 超えた場合に何が起きるかを伝えて、利用者が判断できるようにする方を採る。
FEATURE_MEMORY_WARN_BYTES = 2 * 1024 * 1024 * 1024


class ImageDataset(torch.utils.data.Dataset):
    """dataset_path を前処理済みテンソルにするだけ。**ラベルは持たない。**

    特徴の計算は head に依存しないので、ここでラベルを扱うと head ごとに
    符号化をやり直す構造に戻ってしまう。
    """

    def __init__(self, items: Sequence[Any], *, dataset_root: Path, clip_preprocess) -> None:
        self.items = list(items)
        self.dataset_root = dataset_root
        self.clip_preprocess = clip_preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = (self.dataset_root / self.items[i].dataset_path).resolve()
        return self.clip_preprocess(load_image_rgb(path))


def usable_items(items: Sequence[Any], *, dataset_root: Path) -> Tuple[List[Any], List[str]]:
    """読める画像だけを残す。返り値は (残した items, 落とした dataset_path)。

    **items を先に絞ってから符号化する**のが要点である。符号化の途中で落とすと、
    特徴の行と `build_targets` の行がずれ、**間違ったラベルで学習しても気づけない**。

    1 枚のゴミで学習ジョブ全体が落ちるのを避ける方針は `embed_images` と揃える
    （実データで 879 枚中 21 枚が、拡張子だけ .jpg の HTML だった）。
    ヘッダしか読まないので、CLIP の符号化に比べれば無視できる時間で済む。
    """
    kept: List[Any] = []
    dropped: List[str] = []
    for item in items:
        if is_decodable_image((dataset_root / item.dataset_path).resolve()):
            kept.append(item)
        else:
            dropped.append(str(item.dataset_path))
    return kept, dropped


@torch.no_grad()
def encode_features(
    items: Sequence[Any],
    *,
    dataset_root: Path,
    clip_preprocess,
    clip_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """items を 1 周して、L2 正規化済みの特徴を CPU 上の [N, D] で返す。

    正規化までここで行うのは、学習側とも `predict_labels` 側とも**同じ形**に
    揃えるためである（head は単位ベクトルの上の Linear として学習されている）。
    """
    if not items:
        return torch.empty(0)

    loader = torch.utils.data.DataLoader(
        ImageDataset(items, dataset_root=dataset_root, clip_preprocess=clip_preprocess),
        batch_size=batch_size,
        shuffle=False,          # 順序は items と 1 対 1 に保つ
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    clip_model.eval()
    chunks: List[torch.Tensor] = []
    done = 0
    for xb in loader:
        feats = clip_model.encode_image(xb.to(device)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        chunks.append(feats.detach().cpu())
        done += int(xb.shape[0])
        if on_progress is not None:
            on_progress(done, len(items))
    return torch.cat(chunks, dim=0)


def build_targets(
    items: Sequence[Any],
    *,
    class_head: str,
    head_type: str,
    head_schema: Optional[Dict[str, Any]],
    class_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """head 1 つ分の正解と有効マスクを作る。**画像には触らない。**

    返り値の並びは `items` と同じなので、`encode_features` の出力と行で対応する。
    有効マスクが 0 の行は、その head のラベルが無い（= 損失に入れない）item。
    """
    schema = head_schema or {"type": head_type}
    n = len(items)
    valid = torch.zeros(n, dtype=torch.float32)

    if head_type == "multi_label":
        y = torch.zeros((n, len(class_to_idx)), dtype=torch.float32)
        for i, item in enumerate(items):
            norm = normalize_label_for_head((item.labels or {}).get(class_head), schema)
            if norm is None:
                continue
            for label in (norm if isinstance(norm, list) else [str(norm)]):
                idx = class_to_idx.get(str(label))
                if idx is not None:
                    y[i, idx] = 1.0
            valid[i] = 1.0
        return y, valid

    y = torch.full((n,), -1, dtype=torch.long)
    for i, item in enumerate(items):
        norm = normalize_label_for_head((item.labels or {}).get(class_head), schema)
        idx = class_to_idx.get(str(norm), -1) if norm is not None else -1
        y[i] = idx
        valid[i] = 1.0 if idx >= 0 else 0.0
    return y, valid


def feature_loader(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """(特徴, 正解, 有効マスク) を出すローダ。既存の学習ループの形をそのまま保つ。

    画像を読まないので `num_workers` は使わない（プロセス間でテンソルを渡す方が遅い）。
    """
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets, valid),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def feature_memory_bytes(*counts_and_dim: int) -> int:
    """[N, D] float32 の合計バイト数。警告の判定に使う。"""
    *counts, dim = counts_and_dim
    return sum(counts) * dim * 4
