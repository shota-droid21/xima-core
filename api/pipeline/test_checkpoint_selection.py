"""どのエポックの checkpoint を採用するか（val_loss 基準）を固定する。

なぜテストで縛るか:

    以前は val_acc で選んでいた。val_acc は階段関数で、val 192 件なら 1 枚 = 0.52%。
    ノイズで数エポック横ばいに見える間も val_loss は下がり続けており、実データでは
    patience=5 で character が 8 エポックで止まり **epoch 3** の checkpoint が
    出荷されていた（最後まで回せば val_acc 0.812 -> 0.896）。

    multi_label では更に悪い。val_acc は per-element なので、11 クラス・1 画像あたり
    平均 1.50 個が正の head では「1 つも付けない」と答えるだけで 0.864 になる。
    val_acc 基準は **epoch 1 の「何も予測しないモデル」を最良として採用**していた
    （集合の完全一致は 0.000）。

ここでは train_epoch のループと同じ判定だけを取り出して確かめる。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _select(history: List[Dict[str, float]], *, patience: int, key: str) -> Tuple[int, int]:
    """train_epoch のループと同じ手順。戻り値は (回した数, 採用した epoch)。

    key="val_loss" は小さいほど良い、key="val_acc" は大きいほど良い。
    """
    best: Optional[float] = None
    best_epoch = 0
    since = 0
    ran = 0
    for i, h in enumerate(history):
        ran = i + 1
        v = h[key]
        improved = (
            best is None
            or (v < best if key == "val_loss" else v > best)
        )
        if improved:
            best, best_epoch, since = v, i + 1, 0
        else:
            since += 1
            if patience > 0 and since >= patience:
                break
    return ran, best_epoch


# val_loss は単調に下がるが、val_acc はノイズで上下する。実データの character の形。
NOISY_ACC_HISTORY = [
    {"val_loss": 2.96, "val_acc": 0.70},
    {"val_loss": 2.60, "val_acc": 0.83},   # acc のピークがここ
    {"val_loss": 2.33, "val_acc": 0.81},
    {"val_loss": 2.13, "val_acc": 0.78},
    {"val_loss": 1.97, "val_acc": 0.74},
    {"val_loss": 1.85, "val_acc": 0.79},
    {"val_loss": 1.75, "val_acc": 0.73},
    {"val_loss": 1.66, "val_acc": 0.75},
    {"val_loss": 1.59, "val_acc": 0.78},
    {"val_loss": 1.42, "val_acc": 0.88},   # 最後まで回せばここが最良
]


def test_val_loss_keeps_training_while_loss_still_falls():
    ran, best_epoch = _select(NOISY_ACC_HISTORY, patience=5, key="val_loss")
    assert ran == len(NOISY_ACC_HISTORY)
    assert best_epoch == len(NOISY_ACC_HISTORY)


def test_val_acc_would_stop_early_and_adopt_a_worse_checkpoint():
    """現行が壊れていたことの記録。val_acc 基準だと途中で切れ、悪い checkpoint を採る。"""
    ran, best_epoch = _select(NOISY_ACC_HISTORY, patience=5, key="val_acc")
    assert ran == 7               # epoch 2 の後 5 回改善せず打ち切り
    assert best_epoch == 2
    # そのとき採用される val_loss は、最後まで回した場合より明確に悪い
    assert NOISY_ACC_HISTORY[best_epoch - 1]["val_loss"] > NOISY_ACC_HISTORY[-1]["val_loss"]


def test_multi_label_all_zero_model_is_not_selected():
    """「何も予測しない」モデルは per-element の acc が高いが loss は悪い。

    val_loss で選べば採用されない。
    """
    # 実測（hair_color / 11 クラス・1 画像あたり平均 1.50 個が正）で観測した形。
    # 何も付けない状態が per-element 0.864。予測を始めると誤検出が出て一時的に下がり、
    # 学習が進んでから追い抜く。val_acc 基準はその谷を「改善なし」と数えて打ち切る。
    history = [
        {"val_loss": 1.27, "val_acc": 0.864},  # 全クラス無しと答えた状態（完全一致 0.000）
        {"val_loss": 1.22, "val_acc": 0.851},
        {"val_loss": 1.18, "val_acc": 0.848},
        {"val_loss": 1.14, "val_acc": 0.856},
        {"val_loss": 1.11, "val_acc": 0.859},
        {"val_loss": 1.08, "val_acc": 0.862},
        {"val_loss": 1.05, "val_acc": 0.881},
        {"val_loss": 1.01, "val_acc": 0.930},
    ]
    ran_loss, by_loss = _select(history, patience=5, key="val_loss")
    ran_acc, by_acc = _select(history, patience=5, key="val_acc")
    assert (ran_loss, by_loss) == (8, 8)
    # 壊れていた側は 6 エポックで打ち切り、「何も予測しないモデル」を採る
    assert (ran_acc, by_acc) == (6, 1)


def test_early_stopping_still_fires_when_loss_stops_improving():
    """過学習に対する網としては機能し続けること。"""
    history = [{"val_loss": 2.0 - i * 0.1, "val_acc": 0.5} for i in range(5)]
    history += [{"val_loss": 1.6 + i * 0.05, "val_acc": 0.5} for i in range(10)]
    ran, best_epoch = _select(history, patience=3, key="val_loss")
    assert best_epoch == 5
    assert ran == 8          # 5 の後 3 回改善せず打ち切り
    assert ran < len(history)


def test_patience_zero_disables_early_stopping():
    history = [{"val_loss": 1.0, "val_acc": 0.5} for _ in range(20)]
    ran, best_epoch = _select(history, patience=0, key="val_loss")
    assert ran == 20
    assert best_epoch == 1   # 改善しないので最初の checkpoint のまま


def test_defaults_match_between_cli_and_app():
    """CLI と app の既定がずれていないこと。

    以前は CLI が patience=0（無効）、app が 4 で、同じ操作のつもりで
    まったく違う学習になっていた。
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent / "train_epoch.py").read_text(encoding="utf-8")
    app = (
        Path(__file__).resolve().parents[3] / "app" / "src" / "pages" / "ExperimentDashboard.tsx"
    )
    if not app.exists():
        return  # 公開 core 単体には app が無い

    app_src = app.read_text(encoding="utf-8")
    cli_epochs = int(re.search(r'"--epochs",\s*type=int,(?:\s*#[^\n]*\n)*\s*default=(\d+)', src).group(1))
    cli_patience = int(
        re.search(r'"--early-stopping-patience",\s*type=int,(?:\s*#[^\n]*\n)*\s*default=(\d+)', src).group(1)
    )
    assert cli_epochs == int(re.search(r"epochs:\s*(\d+),", app_src).group(1))
    assert cli_patience == int(re.search(r"earlyStoppingPatience:\s*(\d+),", app_src).group(1))
