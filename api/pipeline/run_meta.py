"""run_meta.json（学習 run のメタ情報）の読み書き・整形ヘルパ。

train_epoch から分離して torch 非依存に保つことで、
metrics 整形ロジックを torch 無しの環境（CI 等）でも単体テストできるようにする。
"""

from __future__ import annotations

import json
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
    return {
        "head_type": head_type,
        "epochs_trained": len(epoch_history),
        "best": {
            "epoch": best_epoch,
            "val_acc": float(best_state["val_acc"]) if best_state else None,
            "val_loss": float(best_state["val_loss"]) if best_state else None,
        },
        "history": list(epoch_history),
    }
