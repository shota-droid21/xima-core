"""numpy 高速路が純 Python 実装と同じ結果を返すことの確認（#272）。

`clustering_fast` はアルゴリズムを変えず距離計算だけをベクトル化したもの。
**同じ入力・同じ seed で同じ割り当てが出る**ことがこの分離の前提なので、
ここで両経路を直接突き合わせる。numpy が無い環境ではスキップする。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

pytest.importorskip("numpy", reason="高速路は numpy が在るときだけ使われる")

from pipeline.clustering import run_clustering  # noqa: E402
from pipeline.clustering_fast import as_matrix  # noqa: E402


def _blobs(n_clusters: int, per: int, dim: int, spread: float, seed: int):
    """seed 固定で作る、重なりのある点群。"""
    rng = random.Random(seed)
    return [
        [rng.gauss(c * spread, 1.0) for _ in range(dim)]
        for c in range(n_clusters)
        for _ in range(per)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "n_clusters,per,dim,spread",
    [(4, 25, 8, 5.0), (3, 17, 16, 4.0), (5, 9, 5, 6.0), (2, 40, 3, 3.0)],
)
def test_fast_matches_pure(seed, n_clusters, per, dim, spread):
    vectors = _blobs(n_clusters, per, dim, spread, seed)
    pure = run_clustering(vectors, seed=seed, use_numpy=False)
    fast = run_clustering(vectors, seed=seed)
    assert fast == pure


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1.0, 2.0]],
        # 全点が同一（k-means++ の total <= 0 の枝を通る）
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        # 1 点だけ離れていて空クラスタ救済が起きうる
        [[0.0], [0.0], [0.0], [0.0], [100.0]],
    ],
)
def test_fast_matches_pure_on_edge_cases(vectors):
    assert run_clustering(vectors, seed=0) == run_clustering(
        vectors, seed=0, use_numpy=False
    )


def test_accepts_numpy_matrix_directly():
    """API 側は `embeddings.npy` の行列をそのまま渡す（list に落とさない）。"""
    import numpy as np

    vectors = _blobs(3, 10, 4, 5.0, 0)
    matrix = np.asarray(vectors, dtype=np.float32)
    assert run_clustering(matrix, seed=0) == run_clustering(vectors, seed=0)


def test_as_matrix_rejects_non_2d():
    import numpy as np

    with pytest.raises(ValueError):
        as_matrix(np.zeros((3,)))
