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


# ------------------------------------------------------------- temperature

from head_checkpoint import DEFAULT_TEMPERATURE, read_temperature  # noqa: E402


def test_reads_stored_temperature():
    assert read_temperature({"temperature": 0.0839}) == 0.0839


def test_missing_temperature_defaults_to_one():
    # 古い checkpoint。1.0 は「割らない」＝従来と同じ予測になる。
    assert read_temperature({"classes": ["a"]}) == DEFAULT_TEMPERATURE


def test_none_temperature_defaults_to_one():
    # 校正をスキップしたときは None が保存される。
    assert read_temperature({"temperature": None}) == DEFAULT_TEMPERATURE


def test_rejects_zero_and_negative():
    assert read_temperature({"temperature": 0}) == DEFAULT_TEMPERATURE
    assert read_temperature({"temperature": -1.5}) == DEFAULT_TEMPERATURE


def test_rejects_nan_and_inf():
    assert read_temperature({"temperature": float("nan")}) == DEFAULT_TEMPERATURE
    assert read_temperature({"temperature": float("inf")}) == DEFAULT_TEMPERATURE


def test_rejects_non_numeric():
    assert read_temperature({"temperature": "hot"}) == DEFAULT_TEMPERATURE


def test_tolerates_non_mapping():
    assert read_temperature(None) == DEFAULT_TEMPERATURE
