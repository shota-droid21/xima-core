"""埋め込みベクトルの類似クラスタリング（torch / numpy 非依存の純 Python 実装）。

#102 で作った CLIP 埋め込み（`cache/embeddings`）を用いて似た画像をまとめ、
UI 側で「このクラスタに同じラベルを一括付与」できるようにするための中核ロジック。
ラベル 0 枚から効く Core 機能で、手ラベリングの工数を大きく減らす。

設計上のポイント:

- **重い依存を増やさない**。本モジュールの実装は純 Python で、numpy が無い環境でも
  そのまま動き、単体テストできる（torch 教訓の踏襲）。
- ただし純 Python の距離計算は n × k × dim で効き、実データ規模（3,489 枚 / 768 次元）
  では 24.7 秒かかって app のタイムアウトに達した。そのため numpy が在る実行環境では
  `clustering_fast` に同じ手順を写した高速路を用意し、`run_clustering` がそちらへ回す
  （#272）。**アルゴリズムと乱数の引き方は同じで、結果も揃う。**
- 埋め込みは **L2 正規化済み** 前提（embed_images がそう保存する）。よって
  ユークリッド距離順 ≒ コサイン類似順になる。
- k-means++ 初期化 + 反復。`seed` 固定で再現可能。空クラスタは最遠点で再初期化する。
- 行 ↔ item のズレが最も事故りやすいため、返すのは items への **index** に統一し、
  file_id / path への写像は呼び出し側に委ねる。
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

# 自動 k の上限。
#
# 以前は 12 だった。実データ（3,489 枚）だと 1 クラスタ平均 290 枚になり、
# 「まとまりごとに同じラベルを一括付与」には粗すぎた。上限は「多すぎると一括付与の
# 利点を損なう」ために置いたものだが、本番の量では逆に働いていた（#272）。
# sqrt(n/2) 自体が保守的なので、上限は病的に大きい n を抑えるだけの役に留める。
MAX_AUTO_K = 48
DEFAULT_MAX_ITERS = 25


def auto_k(n: int) -> int:
    """画像枚数 n から妥当なクラスタ数を推定する。

    ざっくり sqrt(n/2) を基準にし、[2, min(n, MAX_AUTO_K)] に丸める。
    n が 1 以下なら 1（クラスタリング不能）。
    """
    if n <= 1:
        return max(n, 0)
    if n == 2:
        return 2
    est = round(math.sqrt(n / 2.0))
    return max(2, min(est, n, MAX_AUTO_K))


def _distance_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _centroid(vectors: List[List[float]], indices: List[int], dim: int) -> List[float]:
    if not indices:
        return [0.0] * dim
    acc = [0.0] * dim
    for idx in indices:
        vec = vectors[idx]
        for d in range(dim):
            acc[d] += vec[d]
    inv = 1.0 / len(indices)
    return [v * inv for v in acc]


def _kmeans_pp_init(
    vectors: List[List[float]], k: int, rng: random.Random
) -> List[List[float]]:
    """k-means++ による初期中心の選択（seed で再現可能）。"""
    n = len(vectors)
    first = rng.randrange(n)
    centers: List[List[float]] = [list(vectors[first])]
    # 各点から最寄り中心までの距離^2。中心を選ぶたびに更新する。
    dist_sq = [_distance_sq(vectors[i], centers[0]) for i in range(n)]

    while len(centers) < k:
        total = sum(dist_sq)
        if total <= 0.0:
            # すべて既存中心と一致（重複データ）。残りは任意点で埋める。
            centers.append(list(vectors[rng.randrange(n)]))
            continue
        # 距離^2 に比例した確率で次の中心を引く。
        threshold = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i in range(n):
            acc += dist_sq[i]
            if acc >= threshold:
                chosen = i
                break
        centers.append(list(vectors[chosen]))
        for i in range(n):
            d = _distance_sq(vectors[i], centers[-1])
            if d < dist_sq[i]:
                dist_sq[i] = d
    return centers


def kmeans(
    vectors: List[List[float]],
    k: int,
    *,
    seed: int = 0,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> List[int]:
    """各ベクトルのクラスタ番号（0..k-1）を返す。

    - 決定的（seed 固定）。
    - 空クラスタは最も中心から遠い点を種にして再初期化する。
    - 収束（割り当て不変）で早期終了する。
    """
    n = len(vectors)
    if n == 0:
        return []
    if k <= 1:
        return [0] * n
    if k >= n:
        # 1 点 1 クラスタ。安定のため index 順に割り当てる。
        return list(range(n))

    dim = len(vectors[0])
    rng = random.Random(seed)
    centers = _kmeans_pp_init(vectors, k, rng)
    labels = [0] * n

    for _ in range(max_iters):
        changed = False
        for i in range(n):
            best_c = 0
            best_d = _distance_sq(vectors[i], centers[0])
            for c in range(1, k):
                d = _distance_sq(vectors[i], centers[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True

        members: List[List[int]] = [[] for _ in range(k)]
        for i in range(n):
            members[labels[i]].append(i)

        for c in range(k):
            if members[c]:
                centers[c] = _centroid(vectors, members[c], dim)
            else:
                # 空クラスタ: 現割り当てで最も「浮いている」点を種に据える。
                far = _farthest_point(vectors, centers, labels)
                centers[c] = list(vectors[far])
                labels[far] = c
                changed = True

        if not changed:
            break

    return labels


def _farthest_point(
    vectors: List[List[float]], centers: List[List[float]], labels: List[int]
) -> int:
    """自クラスタ中心から最も遠い点の index（空クラスタ救済用）。"""
    worst_idx = 0
    worst_d = -1.0
    for i in range(len(vectors)):
        d = _distance_sq(vectors[i], centers[labels[i]])
        if d > worst_d:
            worst_d = d
            worst_idx = i
    return worst_idx


def assemble_clusters(
    labels: List[int], vectors: List[List[float]]
) -> List[Dict[str, object]]:
    """割り当て結果を、UI が扱いやすいクラスタ配列に組み立てる。

    - メンバは中心に近い順（＝代表画像が先頭）に並べる。
    - 代表（representative）は中心最近傍の item index。
    - クラスタはサイズ降順（同数なら代表 index 昇順）で安定ソートする。

    返す index はすべて呼び出し時の `vectors`（＝items）に対するもの。
    """
    if not labels:
        return []
    dim = len(vectors[0]) if vectors else 0

    groups: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, []).append(idx)

    clusters: List[Dict[str, object]] = []
    for _lab, indices in groups.items():
        center = _centroid(vectors, indices, dim)
        ordered = sorted(indices, key=lambda i: (_distance_sq(vectors[i], center), i))
        clusters.append(
            {
                "size": len(ordered),
                "representative": ordered[0],
                "members": ordered,
            }
        )

    clusters.sort(key=lambda c: (-int(c["size"]), int(c["representative"])))
    for cid, cluster in enumerate(clusters):
        cluster["cluster_id"] = cid
    return clusters


_FAST_UNSET = object()
_fast_module: Any = _FAST_UNSET


def _load_fast() -> Any:
    """numpy 高速路（`clustering_fast`）を 1 度だけ解決する。無ければ None。"""
    global _fast_module
    if _fast_module is _FAST_UNSET:
        try:
            import clustering_fast  # noqa: PLC0415  実行環境に numpy が在るときだけ

            _fast_module = clustering_fast
        except Exception:
            # numpy が無い環境では純 Python 経路で動き続ける。
            _fast_module = None
    return _fast_module


def run_clustering(
    vectors: Any,
    *,
    k: Optional[int] = None,
    seed: int = 0,
    max_iters: int = DEFAULT_MAX_ITERS,
    use_numpy: Optional[bool] = None,
) -> Dict[str, object]:
    """クラスタリング一式を実行し、`{k, clusters}` を返す。

    k を省略すると `auto_k` で自動決定する。vectors が空/1 件でも壊れない。

    `vectors` は `list[list[float]]` でも numpy の 2 次元配列でもよい。numpy が
    在れば `clustering_fast` の同手順・高速版を使う（#272）。`use_numpy=False` で
    純 Python 経路を明示でき、両経路の一致をテストで突き合わせられる。
    """
    n = len(vectors)
    resolved_k = auto_k(n) if k is None else max(1, min(int(k), n if n > 0 else 1))

    fast = None if use_numpy is False else _load_fast()
    if fast is not None and n > 0:
        matrix = fast.as_matrix(vectors)
        labels = fast.kmeans(matrix, resolved_k, seed=seed, max_iters=max_iters)
        clusters = fast.assemble_clusters(labels, matrix)
        return {"k": resolved_k, "clusters": clusters}

    # numpy が無い、または明示的に純 Python を選んだ場合。
    plain = [list(v) for v in vectors] if n else []
    labels = kmeans(plain, resolved_k, seed=seed, max_iters=max_iters)
    clusters = assemble_clusters(labels, plain)
    return {"k": resolved_k, "clusters": clusters}
