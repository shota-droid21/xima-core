"""類似画像クラスタリング中核ロジックの単体テスト（torch / numpy 非依存）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipeline.clustering import (  # noqa: E402
    assemble_clusters,
    auto_k,
    kmeans,
    run_clustering,
)


def _three_blobs():
    """3 つに明確に分かれた 2 次元の点群（各 3 点）。"""
    return [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],       # blob A（原点付近）
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],  # blob B
        [-10.0, 5.0], [-10.1, 5.0], [-10.0, 5.1],  # blob C
    ]


def test_auto_k_scales_with_n():
    assert auto_k(0) == 0
    assert auto_k(1) == 1
    assert auto_k(2) == 2
    assert auto_k(36) >= 2
    # 上限で頭打ちになる
    assert auto_k(100000) <= 12


def test_kmeans_recovers_well_separated_blobs():
    vectors = _three_blobs()
    labels = kmeans(vectors, 3, seed=0)

    # 同じ blob 内は同一ラベル、blob 間は別ラベル
    for start in (0, 3, 6):
        group = labels[start : start + 3]
        assert len(set(group)) == 1, f"blob starting at {start} should be one cluster"
    assert len({labels[0], labels[3], labels[6]}) == 3


def test_kmeans_is_deterministic():
    vectors = _three_blobs()
    assert kmeans(vectors, 3, seed=7) == kmeans(vectors, 3, seed=7)


def test_kmeans_handles_k_ge_n():
    vectors = [[0.0], [1.0]]
    # k >= n は 1 点 1 クラスタ
    assert kmeans(vectors, 5) == [0, 1]


def test_kmeans_empty_input():
    assert kmeans([], 3) == []


def test_assemble_orders_by_size_and_puts_representative_first():
    # ラベル: クラスタ 0 に 1 点、クラスタ 1 に 3 点
    vectors = [
        [5.0, 5.0],           # lone
        [0.0, 0.0],           # big cluster
        [0.2, 0.0],
        [-0.2, 0.0],
    ]
    labels = [0, 1, 1, 1]
    clusters = assemble_clusters(labels, vectors)

    # サイズ降順: 大きいクラスタ（3 点）が先頭
    assert clusters[0]["size"] == 3
    assert clusters[1]["size"] == 1
    # cluster_id は並べ替え後に振り直される
    assert [c["cluster_id"] for c in clusters] == [0, 1]
    # 代表は中心最近傍（この並びなら原点 = index 1）
    assert clusters[0]["representative"] == 1
    assert clusters[0]["members"][0] == clusters[0]["representative"]


def test_run_clustering_auto_k_groups_blobs():
    vectors = _three_blobs()
    result = run_clustering(vectors, k=3, seed=0)

    assert result["k"] == 3
    assert sum(c["size"] for c in result["clusters"]) == len(vectors)
    # 全 index が重複なく現れる
    seen = [i for c in result["clusters"] for i in c["members"]]
    assert sorted(seen) == list(range(len(vectors)))


def test_run_clustering_k_is_clamped_to_n():
    vectors = [[0.0], [1.0], [2.0]]
    result = run_clustering(vectors, k=99)
    assert result["k"] == 3
    assert len(result["clusters"]) == 3


def test_run_clustering_empty():
    result = run_clustering([])
    assert result["clusters"] == []
