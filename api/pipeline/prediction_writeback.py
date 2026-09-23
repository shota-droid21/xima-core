"""推論の書き戻しが、あいだに保存された人の編集を消さないようにする（#411）。

`predict_labels` は `labels.json` を**最初に丸ごと読み**、埋め込みと推論を挟んで
**丸ごと書き戻す**。書き込み自体は atomic だが、**読んだ時点からの変化を見ていない**。
推論が数分かかるあいだに画面から保存すると、その保存は候補と一緒に上書きされる
（#382 で再現）。

## 採った案

**読み直して、候補だけを載せる。**

issue はもう一つ「変わっていたら止める」を挙げていたが、採らなかった。
**数分の推論が丸ごと無駄になる**うえ、人は保存しただけで何も悪いことをしていない。
止める側に倒す理由が無い。

候補を別の場所へ載せられるのは、`record_prediction` が **`labels` にも
`committed` にも `split` にも触れない**設計だからである（#382 以前からそう）。
書き戻すのは `predicted[head]` だけなので、人が確定した値とぶつからない。

## 残る隙間

読み直してから書くまでの隙間は残る。**閉じるには保存契約そのものが要る**ので、
それは #306 に残す。ここで縮めるのは「数分」から「ミリ秒」までである。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


def read_json_with_digest(path: Path) -> Tuple[Dict[str, Any], str]:
    """JSON と、**その読み取りそのもの**の指紋を返す。

    読んだバイト列から両方を作る。別々に読むと、あいだに書き換えられたときに
    「指紋は一致するのに中身は違う」が起こりうる。
    """
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON の中身が object ではありません: {path}")
    return data, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PendingPrediction:
    """まだ書いていない候補 1 件。`file_id` で相手を探す。"""

    file_id: str
    head: str
    head_type: str
    prediction: Any
    run_name: str


@dataclass
class WritebackReport:
    applied: int = 0
    #: 手元にはあったが、読み直したら `labels.json` から消えていた分。
    skipped_missing: int = 0
    missing_file_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, int]:
        return {"applied": self.applied, "skipped_missing": self.skipped_missing}


def apply_pending(
    items: Sequence[Dict[str, Any]],
    pending: Sequence[PendingPrediction],
    *,
    record: Callable[..., None],
) -> WritebackReport:
    """読み直した `items` の上に候補を載せる。

    `record` は `label_predictions.record_prediction` を想定する。候補の形を
    決める場所を 1 つに保つため、ここでは組み立てずに呼ぶだけにする。

    **読み直した側に無い `file_id` は飛ばす。**推論のあいだに消された画像で、
    戻してはならない。
    """
    by_file_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("file_id") or "")
        if fid:
            by_file_id[fid] = item

    report = WritebackReport()
    for p in pending:
        item = by_file_id.get(p.file_id)
        if item is None:
            report.skipped_missing += 1
            if len(report.missing_file_ids) < 10:
                report.missing_file_ids.append(p.file_id)
            continue
        record(
            item,
            head=p.head,
            head_type=p.head_type,
            prediction=p.prediction,
            run_name=p.run_name,
        )
        report.applied += 1

    return report
