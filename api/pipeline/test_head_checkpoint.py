"""head_checkpoint のテスト（torch 不要。テンソルはダミーで置き換える）。"""

from __future__ import annotations

from head_checkpoint import linear_shape, normalize_state_dict


class _FakeTensor:
    """shape だけ持つダミー。torch を入れずに形状分岐を検証するため。"""

    def __init__(self, *shape):
        self.shape = shape


def test_normalizes_fc_prefixed_keys():
    out = normalize_state_dict({"fc.weight": _FakeTensor(3, 512), "fc.bias": _FakeTensor(3)})
    assert out is not None and set(out) == {"weight", "bias"}


def test_passes_through_already_normalized():
    src = {"weight": _FakeTensor(3, 512), "bias": _FakeTensor(3)}
    out = normalize_state_dict(src)
    assert out is not None and out["weight"] is src["weight"]


def test_handles_missing_bias():
    out = normalize_state_dict({"fc.weight": _FakeTensor(2, 4)})
    assert out is not None and set(out) == {"weight"}


def test_returns_none_without_weight():
    assert normalize_state_dict({"something.else": _FakeTensor(1)}) is None


def test_returns_none_for_empty():
    assert normalize_state_dict({}) is None


def test_linear_shape_reads_weight():
    assert linear_shape({"weight": _FakeTensor(7, 512)}) == (7, 512)


def test_linear_shape_none_without_weight():
    assert linear_shape({"bias": _FakeTensor(7)}) is None


def test_linear_shape_none_for_1d_weight():
    assert linear_shape({"weight": _FakeTensor(7)}) is None
