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
