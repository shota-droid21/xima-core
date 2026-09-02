"""run_meta.json（学習 run のメタ情報）の読み書き・整形ヘルパ。

train_epoch から分離して torch 非依存に保つことで、
metrics 整形ロジックを torch 無しの環境（CI 等）でも単体テストできるようにする。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# run_meta.json のスキーマ版。metrics セクションの読み手（UI）が互換を判断できるようにする。
RUN_META_VERSION = "1"


def write_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    """run_meta.json を書き出す（学習前の初期版・学習後の確定版で共用）。"""
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_head_metrics(
    head_type: str,
    epoch_history: List[Dict[str, Any]],
    best_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """1 ヘッド分の学習結果（ベスト精度 + エポック推移）を run_meta 用に整形する。

    best_state は early stopping で選ばれた checkpoint（`epoch` は 0 始まり）。
    学習が 1 度も行われなかった場合（epochs=0 等）は best を None 埋めする。
    """
    best_epoch = (int(best_state.get("epoch", 0)) + 1) if best_state else None
    best: Dict[str, Any] = {
        "epoch": best_epoch,
        "val_acc": float(best_state["val_acc"]) if best_state else None,
        "val_loss": float(best_state["val_loss"]) if best_state else None,
    }
    # multi_label のときだけ入る（multi_class では val_acc と同義なので出さない）。
    # 無いことに意味があるので、None を入れて「測ったが 0 だった」と混同させない。
    exact = (best_state or {}).get("val_exact_match")
    if exact is not None:
        best["val_exact_match"] = float(exact)
    return {
        "head_type": head_type,
        "epochs_trained": len(epoch_history),
        "best": best,
        "history": list(epoch_history),
    }


# 「学習できていない」と判定する val_loss の水準（chance loss に対する比）。
#
# multi_class の chance loss は ln(C)。予測が一様分布のままなら loss はここに張り付く。
# 0.95 は、わずかに動いただけの状態も拾うための余裕。
NEAR_CHANCE_RATIO = 0.95

# 最終エポックの改善幅が、全体の改善幅のこの割合を超えていたら「まだ下がり続けている」。
# 収束していれば終盤の改善はほぼ止まる。
STILL_IMPROVING_RATIO = 0.05


def chance_loss(head_type: str, num_classes: int) -> Optional[float]:
    """当てずっぽう（一様予測）のときの loss。比較の基準線。"""
    if num_classes <= 1:
        return None
    if head_type == "multi_label":
        # per-class の BCE。p=0.5 のとき -ln(0.5) = ln 2
        return math.log(2.0)
    return math.log(float(num_classes))


# multi_label で「per-element は高いのに集合が当たっていない」と判定する差。
#
# per-element の acc は、クラス数が多く 1 画像あたりの正が少ないほど自動的に高くなる
# （11 クラス・平均 1.50 個なら「1 つも付けない」で 0.864）。この差が開いているときは、
# 表示されている acc を実力と読んではいけない。
EXACT_MATCH_GAP = 0.2


def diagnose_training(
    *,
    head_type: str,
    num_classes: int,
    epoch_history: List[Dict[str, Any]],
    best_val_loss: Optional[float],
    best_val_acc: Optional[float] = None,
    best_exact_match: Optional[float] = None,
) -> List[str]:
    """**学習が成立したか**を loss の水準と推移から判定し、問題を文章で返す。

    なぜ要るか:
        実データで、val_acc 0.875 と表示されながら softmax の最大値が 0.042 しか出ない
        head が出荷されていた。原因は既定 lr が低すぎて 15 エポック回しても loss が
        chance（ln 30 = 3.401）付近から動いていなかったこと（3.4167 -> 3.1746）。

        **val_acc だけ見ていると気づけない。**多数派クラスを当てているだけでも
        acc は上がるためである。loss が chance に張り付いているかどうかは、
        「そもそも学習が起きたか」を acc とは独立に示す。

        表示できる指標が増えても、人が毎回それを読むとは限らない。だから
        **判定して言葉で出す**ところまでを実装側に持たせる。
    """
    problems: List[str] = []
    baseline = chance_loss(head_type, num_classes)
    losses = [
        float(e["val_loss"])
        for e in epoch_history
        if isinstance(e, dict) and e.get("val_loss") is not None
    ]
    if not losses or baseline is None:
        return problems

    final = float(best_val_loss) if best_val_loss is not None else losses[-1]

    if final >= baseline * NEAR_CHANCE_RATIO:
        problems.append(
            f"val_loss が {final:.4f} で、当てずっぽうの水準（{baseline:.4f}）から"
            "ほとんど動いていません。学習が進んでいない可能性が高く、"
            "予測の確率はほぼ一様になります（正解率が高く見えても、"
            "多数派クラスを当てているだけのことがあります）。"
            "学習率を上げるか、エポック数を増やしてください。"
        )

    if (
        head_type == "multi_label"
        and best_val_acc is not None
        and best_exact_match is not None
        and best_val_acc - best_exact_match > EXACT_MATCH_GAP
    ):
        problems.append(
            f"val_acc {best_val_acc:.3f} は per-element（クラス枠ごと）の一致率で、"
            f"**画像単位で集合が完全に一致した割合は {best_exact_match:.3f}** です。"
            "クラス数が多く 1 画像あたりの正が少ないほど per-element は自動的に高く出ます"
            "（極端な場合、1 つも付けないと答えるだけで高い値になります）。"
            "一括確定が書くのは集合そのものなので、判断は完全一致の方で行ってください。"
        )

    if len(losses) >= 3:
        total_drop = losses[0] - min(losses)
        last_drop = losses[-2] - losses[-1]
        if total_drop > 0 and last_drop > total_drop * STILL_IMPROVING_RATIO:
            problems.append(
                f"最終エポックでも val_loss が下がり続けています"
                f"（直前比 {last_drop:.4f} / 全体の改善 {total_drop:.4f}）。"
                "収束前に打ち切られています。エポック数を増やすか学習率を上げてください。"
            )

    return problems
