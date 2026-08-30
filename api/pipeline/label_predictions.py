"""学習済み head の予測を labels.json に **候補として**記録する（T2-2）。

置き場所とワークフロー:

    1周目  make_label_list -> labeling -> apply_label -> train
    2周目  make_label_list -> embed_images -> predict -> labeling(候補付き) -> ...

**2 周目以降にしか使えない**（学習済みモデルが要る）。ラベリングの手前に置き、
単体画面が候補を出せる状態にしてから人が確認へ入る。

いちばん大事な約束:

    **`labels` には一切書かない。**書き込むのは `item["predicted"]` だけである。

一括分類が 100% になることはあり得ず、結局は単体画面で人が確認する。だからここで
`labels` を埋めると「**確認していないのにラベルがある**」item が生まれ、`has_label()` は
true を返し、labels.json を読む全処理から本物のラベルと区別できなくなる。
区別できるのは本モジュールが書く記録だけで、他の誰もそれを読まない——
#165（delete flag と split の非対称）や #166 と同じ「実態と表示がずれた中間状態」を
新しく作ることになる。

`labels` へ実体化するのは **人の確定操作の瞬間だけ**とする。UI は
`labels[head] ?? predicted[head].value` で表示を解決し、確定するまで `labels` は空のまま。

閾値を持たない理由:

    候補は全件・全クラス分の上位を記録する。保存時に切ると、後から「もう少し低いのも
    見たい」と思ったときに再計算が要る。閾値は「**見ないで確定してよいか**」の判断なので、
    一括確定の側に置く。

このモジュールは torch / numpy に依存しない（`embedding_cache.py` と同じ方針）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# 候補の置き場所。labels.json の item 直下。`labels` とは別レイヤーであることが要点。
PREDICTED_KEY = "predicted"

# tooltip に出す上位クラスの数。全クラス保存すると schema が大きいときに labels.json が膨らむ。
TOP_K = 5

# multi_label で「その候補に含める」と見なす確率。
#
# これは調整用の閾値ではなく **sigmoid の決定点**である。multi_label は per-class の
# sigmoid なので、0.5 を境に「そのクラスと判定した / しなかった」が決まる。
# 「見ないで確定してよいか」の閾値（一括確定側）とは別物なので、混同しないこと。
MULTI_LABEL_DECISION_POINT = 0.5


@dataclass(frozen=True)
class Prediction:
    """1 item・1 head 分の予測。閾値による足切りはしない。"""

    value: Any                  # multi_class は str、multi_label は list[str]
    score: float                # multi_class は最尤クラスの確率、multi_label は最大確率
    top: List[Dict[str, Any]]   # 上位クラスと確率（tooltip 用）


@dataclass
class PredictionReport:
    """何件記録したかの内訳。ログと UI の両方がこれを読む。"""

    recorded: int = 0
    skipped_deleted: int = 0
    skipped_no_embedding: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "recorded": self.recorded,
            "skipped_deleted": self.skipped_deleted,
            "skipped_no_embedding": self.skipped_no_embedding,
        }


def is_delete_flagged(item: Mapping[str, Any]) -> bool:
    """削除対象か。旧データの split=delete も見る（app/src/lib/deleteFlag.ts と対）。"""
    if item.get("delete") is True:
        return True
    labels = item.get("labels")
    if isinstance(labels, dict):
        if str(labels.get("split") or "").strip().lower() == "delete":
            return True
    return False


def has_label(item: Mapping[str, Any], head: str) -> bool:
    """その head に **確定した値**があるか。空文字・空リストは「無い」と見なす。

    候補（`predicted`）はここに含めない。候補は確定値ではないため。
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


def prediction_from_scores(
    scores: Mapping[str, float], *, head_type: str
) -> Optional[Prediction]:
    """1 head 分のスコアから候補を組み立てる。**閾値で落とさない。**

    - `multi_class` … softmax。最尤クラス 1 つ
    - `multi_label` … sigmoid。決定点 0.5 を超えたクラス全部（0 個なら空リスト）
    """
    ranked = sorted(
        ((str(k), float(v)) for k, v in scores.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        return None

    top = [{"class": k, "score": round(v, 6)} for k, v in ranked[:TOP_K]]

    if head_type == "multi_label":
        selected = [k for k, v in ranked if v >= MULTI_LABEL_DECISION_POINT]
        return Prediction(value=selected, score=ranked[0][1], top=top)

    best_class, best_score = ranked[0]
    return Prediction(value=best_class, score=best_score, top=top)


def record_prediction(
    item: Dict[str, Any],
    *,
    head: str,
    head_type: str,
    prediction: Prediction,
    run_name: str,
    now: Optional[str] = None,
) -> None:
    """候補を item に記録する。**`labels` には触らない。**

    `committed` にも `split` にも触らない。ここで書いたものは全て「まだ確定していない」。
    """
    predicted = item.get(PREDICTED_KEY)
    if not isinstance(predicted, dict):
        predicted = {}
    predicted[head] = {
        "head_type": head_type,
        "value": prediction.value,
        "score": round(float(prediction.score), 6),
        "top": prediction.top,
        "run": run_name,
        "at": now or time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    item[PREDICTED_KEY] = predicted


def plan_predictions(
    items: Sequence[Dict[str, Any]],
    *,
    scores_by_file_id: Mapping[str, Mapping[str, float]],
    head_type: str,
    report: PredictionReport,
) -> List[tuple[Dict[str, Any], Prediction]]:
    """候補を記録する対象を決める。**この関数は items を変更しない。**

    **既にラベルが付いている item も対象にする。**class を追加した / 再ラベルする場面で
    既ラベル画像の候補が要るため。`labels` に書かない以上、対象を広げても既存の作業を
    脅かさない。

    削除対象だけは外す。書いても使われないうえ、#165 の経緯から delete 済みの item に
    余計な情報を足さない方がよい。
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

        scores = scores_by_file_id.get(file_id)
        if not scores:
            report.skipped_no_embedding += 1
            continue

        prediction = prediction_from_scores(scores, head_type=head_type)
        if prediction is None:
            report.skipped_no_embedding += 1
            continue

        planned.append((item, prediction))
    return planned


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


def history_snapshot_path(labels_path: Path, *, now: Optional[str] = None) -> Path:
    """スナップショットを置く場所。

    命名は `core/api/app/label_input.py` の `_write_history_snapshot` と**同じ規約**に
    揃える。揃えないと既存の LabelHistory の一覧・復元導線から見えず、
    「一括で書き換えたが戻せない」状態になる。

    **候補の記録では使わない。**あれは `labels` を変えないため、履歴を残す意味が薄く、
    実行のたびに復元候補が「変化のない版」で埋まる。一括確定（`labels` を書き換える側）で使う。
    """
    history_dir = labels_path.parent / "history"
    stamp = now or time.strftime("%Y%m%d_%H%M%S")
    candidate = history_dir / f"{labels_path.stem}_{stamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = history_dir / f"{labels_path.stem}_{stamp}-{suffix}.json"
        suffix += 1
    return candidate


def take_snapshot(labels_path: Path, *, now: Optional[str] = None) -> Path:
    """labels.json を history へ複写する。書き換える**前**に呼ぶこと。"""
    snapshot = history_snapshot_path(labels_path, now=now)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot
