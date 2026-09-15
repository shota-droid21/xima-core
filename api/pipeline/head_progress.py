#!/usr/bin/env python3
"""head のループの中の進捗を、**1 つの単位に固定して**書く（#405）。

`train_epoch.py` は 1,100 行を超えていて torch を import するので、この規則だけを
ここに出す。**単体で確かめられることが大事**な部分である。

## なぜ要るか

以前は呼び出しごとに単位が違った。

| 書く場所 | `current` / `total` | 1 エポックあたり |
| --- | --- | --- |
| エポックの頭 | エポック / エポック数 | 1 回 |
| バッチの途中 | **バッチ / バッチ数** | **約 20 回** |
| 検証 | **渡していない（`None` で消える）** | 1 回 |
| エポック完了 | エポック / エポック数 | 1 回 |

バッチの書き込みが 1 桁多いので、**帯は大半の時間バッチを指し、エポックが
変わるたびに 1 に戻っていた**。文言の先頭は `epoch 100/500` なのに帯は
`12 / 63` を数えている、というのが利用者から見た症状である。

`update_job_progress` は渡されなかった値を `None` で上書きするので、検証のたびに
帯そのものも消えていた。

## 決めたこと

**帯が数えるのは「この head のエポック」だけ。** バッチや検証の細かさは捨てず、
`detail` で文言に、`extra` で機械が読む側に残す。

head が複数ある run では、帯は head の境目で 1 に戻る。run 全体を分母にしても、
早期打ち切りがある以上それは上限でしかなく、`学習設定` に出ているエポック数とも
揃わなくなる。戻る理由が分かるよう、head が 2 つ以上のときは文言に `head i/N` を出す。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from job_progress import update_job_progress


def head_progress(
    *,
    phase: str,
    head: str,
    head_index: int,
    heads_total: int,
    epoch: int,
    epochs_total: int,
    detail: str = "",
    extra: Optional[Dict[str, Any]] = None,
    report: Callable[..., None] = update_job_progress,
) -> None:
    """`current` / `total` に**必ずエポックを入れて**進捗を書く。

    `report` は差し替えのためだけにある（既定は `update_job_progress`）。
    """

    where = f"head {head_index}/{heads_total} " if heads_total > 1 else ""
    message = f"{where}{head} | epoch {epoch}/{epochs_total}"
    if detail:
        message = f"{message} | {detail}"

    payload: Dict[str, Any] = {"head": head, "epoch": epoch, "epochs": epochs_total}
    if heads_total > 1:
        payload.update({"head_index": head_index, "heads_total": heads_total})
    if extra:
        payload.update(extra)

    report(
        phase=phase,
        current=epoch,
        total=epochs_total,
        message=message,
        extra=payload,
    )
