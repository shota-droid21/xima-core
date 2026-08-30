"""推論結果を labels.json へ書き戻すときの判断ロジック（T2-2）。

MVP の到達点は「残りの画像に**自動でラベルが付いた状態**」であり、その最後の 1 手が
ここにある（`docs/strategy/product-strategy.md` §5 / Decision 037）。手動 1 回の
書き戻しは Core（無償）に置く。

このモジュールは torch / numpy に依存しない。埋め込みの読み出しと head の適用は
`predict_unlabeled.py` 側に置き、ここには「**どの item に、何を、書いてよいか**」だけを
持たせる。重い依存の無い環境（CI）で単体テストするため（`embedding_cache.py` と同じ方針）。

守っていること:

- **人が付けたラベルを上書きしない。**値が既にある head には触れない。判定は head 単位で
  行う（item 単位ではない）。同じ item の head A が埋まっていても head B は空でありうる
- **予測であることを記録に残す。**`item["predicted"]` に head ごとの由来（クラス / スコア /
  どの run か / いつか）を置く。これが無いと、あとから「人が付けたのか機械が付けたのか」を
  区別できなくなる
- **`committed` を立てない。**書き戻しは確定ではなく、人が見る前の下書きである
- **`split` に触れない。**未ラベル item の split は `unassigned` のままなので、
  `apply_label_mapping.py` の `if deleted or split not in ("train","val"): continue` により
  **dataset には入らない**。つまり書き戻しただけでは自分の予測で再学習することはない。
  学習に使うかどうかは、人が split を選ぶという明示的な操作に委ねる
- **削除対象は飛ばす。**`delete` が立った item に予測を書いても捨てられる（#165）
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# 予測の由来を書く場所。labels.json の item 直下に置く。
PREDICTED_KEY = "predicted"

# 何も指定しなかったときに書き戻す下限スコア。
#
# 0.0 にしないのは、確信の無い予測を大量に書き戻すと「見直す量」が増えるだけで、
# 到達点（自動でラベルが付いた状態）に近づかないため。逆に高すぎると 1 件も書かれず、
# 利用者には「動かなかった」と区別が付かない。実行結果として書いた数と飛ばした数を
# 必ず出すこと。
DEFAULT_MIN_SCORE = 0.5


@dataclass(frozen=True)
class Prediction:
    """1 item・1 head 分の予測。`value` は書き戻す確定値。"""

    file_id: str
    head: str
    value: Any            # multi_class は str、multi_label は list[str]
    score: float          # multi_class は最尤クラスの確率、multi_label は最大確率
    top: List[Dict[str, Any]]  # 上位クラスと確率（記録用）


@dataclass
class WritebackReport:
    """何が起きたかの内訳。UI とログの両方がこれを読む。"""

    written: int = 0
    skipped_existing: int = 0
    skipped_low_score: int = 0
    skipped_deleted: int = 0
    skipped_no_embedding: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "written": self.written,
            "skipped_existing": self.skipped_existing,
            "skipped_low_score": self.skipped_low_score,
            "skipped_deleted": self.skipped_deleted,
            "skipped_no_embedding": self.skipped_no_embedding,
        }


def is_delete_flagged(item: Mapping[str, Any]) -> bool:
    """削除対象か。旧データの split=delete も見る（app/src/lib/deleteFlag.ts と対）。"""
    if item.get("delete") is True:
        return True
    labels = item.get("labels")
    if isinstance(labels, dict):
        for key in ("split", "Split"):
            if str(labels.get(key) or "").strip().lower() == "delete":
                return True
    return False


def has_label(item: Mapping[str, Any], head: str) -> bool:
    """その head に **人が付けた値**が既にあるか。

    空文字・空リストは「無い」と見なす。予測で埋めた値は `predicted` に記録が残るため、
    再実行しても人の値と取り違えない。
    """
    labels = item.get("labels")
    if not isinstance(labels, dict):
        return False
    value = labels.get(head)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return len([v for v in value if str(v).strip()]) > 0
    return True


def value_from_scores(
    scores: Mapping[str, float],
    *,
    head_type: str,
    min_score: float,
) -> Optional[Prediction]:
    """1 head 分のスコアから、書き戻す値を決める。閾値に届かなければ None。

    `head_type` により意味が変わる:

    - `multi_class` … softmax。**最尤クラス 1 つ**を選ぶ
    - `multi_label` … sigmoid。**閾値を超えたクラスを全部**選ぶ（0 個なら書かない）
    """
    ranked = sorted(
        ((str(k), float(v)) for k, v in scores.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        return None

    top = [{"class": k, "score": v} for k, v in ranked[:3]]

    if head_type == "multi_label":
        selected = [k for k, v in ranked if v >= min_score]
        if not selected:
            return None
        return Prediction(
            file_id="", head="", value=selected, score=ranked[0][1], top=top
        )

    best_class, best_score = ranked[0]
    if best_score < min_score:
        return None
    return Prediction(
        file_id="", head="", value=best_class, score=best_score, top=top
    )


def apply_prediction(
    item: Dict[str, Any],
    *,
    head: str,
    prediction: Prediction,
    run_name: str,
    now: Optional[str] = None,
) -> None:
    """1 item に 1 head 分の予測を書き込む。呼ぶ前に書いてよいことを確認しておくこと。

    `committed` は立てない（下書きであるため）。`split` にも触れない。
    """
    labels = item.get("labels")
    if not isinstance(labels, dict):
        labels = {}
    labels[head] = prediction.value
    item["labels"] = labels

    predicted = item.get(PREDICTED_KEY)
    if not isinstance(predicted, dict):
        predicted = {}
    predicted[head] = {
        "value": prediction.value,
        "score": round(float(prediction.score), 6),
        "top": [
            {"class": t["class"], "score": round(float(t["score"]), 6)} for t in prediction.top
        ],
        "run": run_name,
        "at": now or time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    item[PREDICTED_KEY] = predicted


def plan_writeback(
    items: Sequence[Dict[str, Any]],
    *,
    head: str,
    head_type: str,
    scores_by_file_id: Mapping[str, Mapping[str, float]],
    min_score: float,
    report: WritebackReport,
) -> List[tuple[Dict[str, Any], Prediction]]:
    """書き戻す対象を決める。**この関数は items を変更しない。**

    変更を別段に分けているのは、閾値を変えて件数だけ見たい（dry-run）場合に
    同じ判断を再利用するため。
    """
    planned: List[tuple[Dict[str, Any], Prediction]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("file_id") or "").strip()
        if not file_id:
            continue

        if is_delete_flagged(item):
            report.skipped_deleted += 1
            continue
        if has_label(item, head):
            report.skipped_existing += 1
            continue

        scores = scores_by_file_id.get(file_id)
        if not scores:
            report.skipped_no_embedding += 1
            continue

        prediction = value_from_scores(
            scores, head_type=head_type, min_score=min_score
        )
        if prediction is None:
            report.skipped_low_score += 1
            continue

        planned.append((item, Prediction(
            file_id=file_id,
            head=head,
            value=prediction.value,
            score=prediction.score,
            top=prediction.top,
        )))
    return planned


def history_snapshot_path(labels_path: Path, *, now: Optional[str] = None) -> Path:
    """書き戻し前のスナップショットを置く場所。

    命名は `core/api/app/label_input.py` の `_write_history_snapshot` と**同じ規約**に
    揃える。揃えないと、既存の LabelHistory の一覧・復元導線からこのスナップショットが
    見えず、「一括で書き換えたが戻せない」状態になる。
    """
    history_dir = labels_path.parent / "history"
    stamp = now or time.strftime("%Y%m%d_%H%M%S")
    candidate = history_dir / f"{labels_path.stem}_{stamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = history_dir / f"{labels_path.stem}_{stamp}-{suffix}.json"
        suffix += 1
    return candidate


def write_json_atomic(path: Path, data: Any) -> None:
    """一時ファイル + fsync + replace で書く。中断しても元のファイルを壊さない。

    `app/utils/atomic_io.py` と同じ理由（#151）。pipeline は app を import しないため
    ここにも置く。labels.json は実データで数 MB あり、書き込みは一瞬ではない。
    truncate してから書くと、中断時に**元の内容を失ったうえに壊れたファイルが残る**。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def commit_writeback(
    labels_path: Path,
    data: Any,
    *,
    written: int,
    now: Optional[str] = None,
) -> Optional[Path]:
    """スナップショットを取ってから labels.json を書き換える。

    返り値はスナップショットのパス。**1 件も書き戻していないときは何もしない**
    （None を返す）。書いていないのに history だけ増えると、復元候補の一覧が
    「変化のない版」で埋まり、戻したい版を選べなくなる。

    スナップショットを**先**に取るのは、書き換えに失敗しても戻せる状態を作るため。
    """
    if written <= 0:
        return None

    snapshot = history_snapshot_path(labels_path, now=now)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_json_atomic(labels_path, data)
    return snapshot
