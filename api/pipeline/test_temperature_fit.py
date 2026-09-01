"""fit_temperature の安全装置（torch が要る環境でのみ動かす）。

温度は「argmax を変えずに確率だけ校正する」ものなので、**NLL を悪化させたら失敗**である。
実データで LBFGS が line search 無しに overshoot し、T=2.3e-08 / NLL 118,819 という
checkpoint が出荷された。そのとき閾値ごとの実測一致率が全帯 0.545 に潰れ、
「0.99 以上」を選んでも半分しか当たらない表を利用者に見せていた。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

torch = pytest.importorskip("torch")

from train_epoch import (  # noqa: E402
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    MIN_VAL_FOR_TEMPERATURE,
    fit_temperature,
)


def _nll(logits, targets, head_type: str, t: float) -> float:
    loss_fn = (
        torch.nn.BCEWithLogitsLoss()
        if head_type == "multi_label"
        else torch.nn.CrossEntropyLoss()
    )
    y = targets.float() if head_type == "multi_label" else targets.long()
    with torch.no_grad():
        return float(loss_fn(logits / t, y).item())


def _underconfident_multi_class(n: int = 200, c: int = 8):
    """向きは正しいが logits が小さい（＝校正が要る）状態を作る。"""
    torch.manual_seed(0)
    targets = torch.randint(0, c, (n,))
    logits = torch.randn(n, c) * 0.02
    # 正解クラスをわずかに持ち上げる。argmax はおおむね当たるが確率は一様に近い
    logits[torch.arange(n), targets] += 0.06
    return logits, targets


def test_fit_never_returns_a_temperature_that_worsens_nll():
    logits, targets = _underconfident_multi_class()
    t = fit_temperature(logits, targets, head_type="multi_class")
    if t is None:
        return  # 校正しない判断は常に安全
    assert _nll(logits, targets, "multi_class", t) < _nll(
        logits, targets, "multi_class", 1.0
    )


def test_fit_improves_calibration_for_underconfident_head():
    logits, targets = _underconfident_multi_class()
    t = fit_temperature(logits, targets, head_type="multi_class")
    assert t is not None
    assert t < 1.0  # 自信が足りないので logits を鋭くする方向へ動く
    assert MIN_TEMPERATURE <= t <= MAX_TEMPERATURE


def test_fit_does_not_change_argmax():
    """温度はスカラー除算なので順位を変えない。正解率が動かないことの担保。"""
    logits, targets = _underconfident_multi_class()
    t = fit_temperature(logits, targets, head_type="multi_class")
    assert t is not None
    assert torch.equal(logits.argmax(1), (logits / t).argmax(1))


def test_fit_on_random_labels_does_not_saturate():
    """当てずっぽうの head でも確率を 0/1 へ振り切らせない。

    これが壊れていると、学習できていない head ほど「確信度 0.99」が並び、
    一括確定がいちばん危ない対象を最優先で確定してしまう。
    """
    torch.manual_seed(1)
    logits = torch.randn(300, 6) * 0.05
    targets = torch.randint(0, 6, (300,))
    t = fit_temperature(logits, targets, head_type="multi_class")
    if t is None:
        return
    probs = torch.softmax(logits / t, dim=-1)
    assert float(probs.max(dim=-1).values.mean().item()) < 0.95


def test_multi_label_fit_is_guarded_too():
    torch.manual_seed(2)
    logits = torch.randn(200, 5) * 0.05
    targets = (torch.rand(200, 5) < 0.3).float()
    t = fit_temperature(logits, targets, head_type="multi_label")
    if t is None:
        return
    assert _nll(logits, targets, "multi_label", t) < _nll(
        logits, targets, "multi_label", 1.0
    )


def test_too_few_val_samples_skips_calibration():
    logits = torch.randn(MIN_VAL_FOR_TEMPERATURE - 1, 4)
    targets = torch.randint(0, 4, (MIN_VAL_FOR_TEMPERATURE - 1,))
    assert fit_temperature(logits, targets, head_type="multi_class") is None
