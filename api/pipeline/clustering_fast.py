"""`clustering.py` の k-means を numpy で計算し直す高速路。

**アルゴリズムは変えない。** `clustering.py` の純 Python 実装と制御フローを 1 対 1 で
写し、内側の距離計算だけをベクトル化する。乱数も同じく `random.Random(seed)` から
同じ順序で引くため、seed 固定での再現性と結果は純 Python 経路と揃う。

なぜ分けるか:

- `clustering.py` は「numpy が無くても中核ロジックが動き、単体テストできる」ことを
  前提に書かれている（#104）。その前提は壊さず、numpy が在るときだけこちらに来る。
- 実データ（3,489 枚 / 768 次元）で純 Python の k-means が 24.7 秒かかり、app の
  タイムアウトに達していた（#272）。距離計算が n × k × dim で効くため、枚数が
  増えるほど確実に悪化する。

浮動小数点について: 総和の順序が numpy（pairwise）と Python（逐次）で違うため、
最下位ビットまで一致する保証は無い。距離は代数変形せず差の二乗和をそのまま取り、
純 Python の式から離れないようにしている（`|x|^2 - 2x·c + |c|^2` に展開すると速いが、
桁落ちする形になるため採らない）。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np

# 距離行列を一度に作らず、行をこの単位で区切って計算する。
# n × k × dim を丸ごと確保すると実データ規模で数百 MB になる。
_CHUNK_ROWS = 512


def as_matrix(vectors: Any) -> np.ndarray:
    """入力を 2 次元の float64 行列にする。

    既に ndarray ならコピーしない。純 Python 経路が Python の float（倍精度）で
    計算するため、float32 のキャッシュを読んだ場合もここで倍精度に揃える。
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must be 2D")
    return matrix


def _sq_dists_to_center(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """全行から 1 つの中心までの距離^2。"""
    diff = matrix - center
    return np.einsum("ij,ij->i", diff, diff)


def _sq_dists_to_centers(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """全行 × 全中心の距離^2（n, k）。行を区切って確保量を抑える。"""
    n = matrix.shape[0]
    out = np.empty((n, centers.shape[0]), dtype=np.float64)
    for start in range(0, n, _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, n)
        diff = matrix[start:end, None, :] - centers[None, :, :]
        out[start:end] = np.einsum("ijk,ijk->ij", diff, diff)
    return out


def _kmeans_pp_init(
    matrix: np.ndarray, k: int, rng: random.Random
) -> np.ndarray:
    """k-means++ 初期化。乱数の引き方を純 Python 実装と揃える。"""
    n = matrix.shape[0]
    first = rng.randrange(n)
    centers: List[np.ndarray] = [matrix[first].copy()]
    dist_sq = _sq_dists_to_center(matrix, centers[0])

    while len(centers) < k:
        total = float(dist_sq.sum())
        if total <= 0.0:
            # すべて既存中心と一致（重複データ）。残りは任意点で埋める。
            centers.append(matrix[rng.randrange(n)].copy())
            continue
        threshold = rng.random() * total
        # 純 Python 側は先頭から足して threshold に達した最初の index を採る。
        acc = np.cumsum(dist_sq)
        chosen = int(np.searchsorted(acc, threshold, side="left"))
        if chosen >= n:
            chosen = n - 1
        centers.append(matrix[chosen].copy())
        np.minimum(dist_sq, _sq_dists_to_center(matrix, centers[-1]), out=dist_sq)

    return np.array(centers, dtype=np.float64)


def _farthest_point(
    matrix: np.ndarray, centers: np.ndarray, labels: np.ndarray
) -> int:
    """自クラスタ中心から最も遠い点の index（空クラスタ救済用）。"""
    diff = matrix - centers[labels]
    return int(np.einsum("ij,ij->i", diff, diff).argmax())


def kmeans(
    matrix: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    max_iters: int,
) -> List[int]:
    """各行のクラスタ番号を返す。純 Python 実装と同じ手順を踏む。"""
    n = matrix.shape[0]
    if n == 0:
        return []
    if k <= 1:
        return [0] * n
    if k >= n:
        return list(range(n))

    rng = random.Random(seed)
    centers = _kmeans_pp_init(matrix, k, rng)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iters):
        # 同点は小さいクラスタ番号を採る（argmin の既定＝純 Python 側の `<` と一致）。
        assigned = _sq_dists_to_centers(matrix, centers).argmin(axis=1)
        changed = bool(np.any(assigned != labels))
        labels = assigned.astype(np.int64, copy=False)

        # members は中心更新の前に 1 度だけ作る。空クラスタ救済で labels を
        # 書き換えても members は更新しない（純 Python 実装と同じ）。
        members = [np.flatnonzero(labels == c) for c in range(k)]

        for c in range(k):
            if members[c].size:
                centers[c] = matrix[members[c]].mean(axis=0)
            else:
                far = _farthest_point(matrix, centers, labels)
                centers[c] = matrix[far]
                labels[far] = c
                changed = True

        if not changed:
            break

    return [int(x) for x in labels]


def assemble_clusters(labels: List[int], matrix: np.ndarray) -> List[Dict[str, object]]:
    """割り当て結果をクラスタ配列に組み立てる（`clustering.assemble_clusters` と同形）。"""
    if not labels:
        return []

    label_arr = np.asarray(labels, dtype=np.int64)
    clusters: List[Dict[str, object]] = []

    for lab in np.unique(label_arr):
        indices = np.flatnonzero(label_arr == lab)
        rows = matrix[indices]
        center = rows.mean(axis=0)
        diff = rows - center
        dist = np.einsum("ij,ij->i", diff, diff)
        # 中心に近い順、同値なら index 昇順。
        ordered = indices[np.lexsort((indices, dist))]
        clusters.append(
            {
                "size": int(ordered.size),
                "representative": int(ordered[0]),
                "members": [int(i) for i in ordered],
            }
        )

    clusters.sort(key=lambda c: (-int(c["size"]), int(c["representative"])))
    for cid, cluster in enumerate(clusters):
        cluster["cluster_id"] = cid
    return clusters
