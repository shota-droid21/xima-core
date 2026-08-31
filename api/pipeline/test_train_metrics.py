"""run_meta の metrics 整形（build_head_metrics）の単体テスト。

E2E（test_pipeline_minimal_e2e）は fake torch のため epochs=0 でしか回せず、
「学習した精度が正しく捕捉されるか」を検証できない。ここでは torch を使わず、
整形ロジックだけを対象に、値の捕捉と後方互換（学習なし）を検証する。

対象の build_head_metrics は torch 非依存の pipeline/run_meta.py にあるため、
torch 未導入の環境（CI）でも import できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipeline.run_meta import build_head_metrics


def test_build_head_metrics_captures_best_and_history() -> None:
    history = [
        {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_acc": 0.5},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.3, "val_acc": 0.75},
    ]
    # best_state.epoch は 0 始まり（epoch index 1 == 2 エポック目）
    best_state = {"epoch": 1, "val_acc": 0.75, "val_loss": 0.3}

    m = build_head_metrics("multi_class", history, best_state)

    assert m["head_type"] == "multi_class"
    assert m["epochs_trained"] == 2
    assert m["history"] == history
    # 1 始まりに変換されて保存される
    assert m["best"]["epoch"] == 2
    assert m["best"]["val_acc"] == 0.75
    assert m["best"]["val_loss"] == 0.3


def test_build_head_metrics_without_training_is_null_safe() -> None:
    # epochs=0 等で 1 度も学習しなかった場合
    m = build_head_metrics("multi_label", [], None)

    assert m["epochs_trained"] == 0
    assert m["history"] == []
    assert m["best"]["epoch"] is None
    assert m["best"]["val_acc"] is None
    assert m["best"]["val_loss"] is None


# --------------------------------------------------------- diagnose_training

import math  # noqa: E402

from run_meta import chance_loss, diagnose_training  # noqa: E402


def _history(losses):
    return [{"epoch": i, "val_loss": v, "val_acc": 0.5} for i, v in enumerate(losses)]


def test_chance_loss_is_log_num_classes():
    assert abs(chance_loss("multi_class", 30) - math.log(30)) < 1e-9


def test_chance_loss_for_multi_label_is_log_two():
    assert abs(chance_loss("multi_label", 11) - math.log(2)) < 1e-9


def test_chance_loss_none_for_degenerate():
    assert chance_loss("multi_class", 1) is None


def test_flags_loss_stuck_at_chance():
    # 実際に出荷された character head の形（ln(30)=3.401 から 3.26 までしか動かない）
    losses = [3.41 - i * 0.011 for i in range(15)]
    problems = diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=_history(losses), best_val_loss=losses[-1],
    )
    assert any("当てずっぽう" in p for p in problems)


def test_flags_still_improving_at_last_epoch():
    losses = [3.41 - i * 0.011 for i in range(15)]
    problems = diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=_history(losses), best_val_loss=losses[-1],
    )
    assert any("下がり続けています" in p for p in problems)


def test_converged_run_is_clean():
    # 十分下がってから平らになった形
    losses = [3.4, 2.0, 1.0, 0.5, 0.32, 0.31, 0.305, 0.304, 0.3039, 0.3038]
    problems = diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=_history(losses), best_val_loss=losses[-1],
    )
    assert problems == []


def test_low_loss_but_still_falling_is_flagged():
    # chance からは離れているが収束していない
    losses = [3.4, 2.4, 1.4, 0.9, 0.5]
    problems = diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=_history(losses), best_val_loss=losses[-1],
    )
    assert len(problems) == 1 and "下がり続けています" in problems[0]


def test_no_history_yields_no_problems():
    assert diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=[], best_val_loss=None,
    ) == []


def test_history_without_val_loss_is_tolerated():
    assert diagnose_training(
        head_type="multi_class", num_classes=30,
        epoch_history=[{"epoch": 0, "val_acc": 0.5}], best_val_loss=None,
    ) == []
