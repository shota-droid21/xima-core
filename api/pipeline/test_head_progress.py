"""head のループの中の進捗が、**1 つの単位だけを数える**こと（#405）。

直す前は、同じ `progress.current` / `total` に 3 つの単位が入っていた。
エポックの頭は `epoch/epochs`、バッチの途中は `batch/batches`、検証は
**何も渡していなかった**（`update_job_progress` は渡されなかった値を `None` で
上書きするので、そのたびに帯が消えていた）。

バッチの書き込みは 1 エポックにつき 20 回ほど、エポックの書き込みは 2 回しか
無いので、**帯は大半の時間バッチを指し、エポックが変わるたびに 1 に戻っていた**。

この検査が守るのは 2 つ。

1. **`total` は常にエポック数**。何を書いていても単位が変わらない
2. **1 つの head の中で `current` が戻らない**
"""

from __future__ import annotations

from typing import Any, Dict, List

from head_progress import head_progress


def _recorder() -> tuple[List[Dict[str, Any]], Any]:
    calls: List[Dict[str, Any]] = []

    def report(**kwargs: Any) -> None:
        calls.append(kwargs)

    return calls, report


def _simulate_head(
    report: Any,
    *,
    head: str = "shape",
    head_index: int = 1,
    heads_total: int = 1,
    epochs_total: int = 4,
    batches: int = 63,
    stop_after: int | None = None,
) -> None:
    """`train_epoch` の呼び出し順をそのままなぞる。"""
    common = dict(
        head=head, head_index=head_index, heads_total=heads_total,
        epochs_total=epochs_total, report=report,
    )
    head_progress(phase="start_head", epoch=0, detail="starting", **common)

    ran = 0
    for epoch in range(1, epochs_total + 1):
        head_progress(phase="train", epoch=epoch, extra={"stage": "train"}, **common)
        # 実物は `max(1, batches // 20)` 回に 1 回書く
        for batch in range(1, batches + 1):
            if batch % max(1, batches // 20) and batch != batches:
                continue
            head_progress(
                phase="train",
                epoch=epoch,
                detail=f"batch {batch}/{batches}",
                extra={"batch": batch, "batches": batches},
                **common,
            )
        head_progress(phase="val", epoch=epoch, detail="validating", **common)
        head_progress(phase="epoch_done", epoch=epoch, detail="val_loss=0.1000", **common)
        ran = epoch
        if stop_after is not None and epoch >= stop_after:
            break

    head_progress(phase="saved", epoch=ran, detail="saved", **common)


def test_total_is_always_epochs() -> None:
    """**何を書いても `total` はエポック数。** バッチ数が混ざらない。"""
    calls, report = _recorder()
    _simulate_head(report, epochs_total=4, batches=63)

    assert calls, "書き込みが 1 件も無い"
    assert {c["total"] for c in calls} == {4}
    # 63 はどこにも現れない（`total` としては）
    assert 63 not in {c["total"] for c in calls}


def test_current_never_goes_backwards_within_a_head() -> None:
    """**1 つの head の中で帯が戻らない。** これが利用者の見た症状そのもの。"""
    calls, report = _recorder()
    _simulate_head(report, epochs_total=4, batches=63)

    currents = [c["current"] for c in calls]
    assert currents == sorted(currents), f"戻っている: {currents}"
    assert currents[0] == 0, "始まりは 0 エポック"
    assert currents[-1] == 4


def test_batch_detail_survives_in_message_and_extra() -> None:
    """細かさは捨てない。**帯から外すだけ**で、文言と `extra` には残る。"""
    calls, report = _recorder()
    _simulate_head(report, epochs_total=1, batches=63)

    batch_calls = [c for c in calls if "batch" in c["extra"]]
    assert batch_calls, "バッチの書き込みが無い"
    last = batch_calls[-1]
    assert last["extra"]["batch"] == 63
    assert last["extra"]["batches"] == 63
    assert "batch 63/63" in last["message"]
    # それでも帯はエポック
    assert (last["current"], last["total"]) == (1, 1)


def test_progress_is_never_cleared() -> None:
    """検証のときも `current` / `total` が消えない（消えると帯ごと消える）。"""
    calls, report = _recorder()
    _simulate_head(report, epochs_total=2, batches=8)

    val_calls = [c for c in calls if c["phase"] == "val"]
    assert val_calls
    for c in val_calls:
        assert c["current"] is not None
        assert c["total"] is not None


def test_head_counter_only_when_there_are_several() -> None:
    """head が 1 つのときに `head 1/1` を出さない。2 つ以上なら出す。"""
    calls, report = _recorder()
    _simulate_head(report, heads_total=1, epochs_total=1, batches=4)
    assert all("head 1/1" not in c["message"] for c in calls)
    assert all("head_index" not in c["extra"] for c in calls)

    calls, report = _recorder()
    _simulate_head(report, head="tone", head_index=2, heads_total=3,
                   epochs_total=1, batches=4)
    assert all(c["message"].startswith("head 2/3 tone | epoch ") for c in calls)
    assert calls[0]["extra"]["heads_total"] == 3


def test_early_stopping_reports_the_epochs_actually_run() -> None:
    """早期打ち切りのとき、`saved` は**実際に回った数**を出す。`total` は上限。"""
    calls, report = _recorder()
    _simulate_head(report, epochs_total=500, batches=63, stop_after=8)

    saved = [c for c in calls if c["phase"] == "saved"]
    assert len(saved) == 1
    assert saved[0]["current"] == 8
    assert saved[0]["total"] == 500
