"""学習済み head で **未ラベル画像**を推論し、labels.json へ書き戻す（T2-2・手動 1 回）。

なぜ別スクリプトなのか（`infer_heads.py` を拡張しないのか）:

    `infer_heads.py` の対象は `dataset/index.json` である。この index は
    `apply_label_mapping.py` が `if deleted or split not in ("train","val"): continue`
    で作るため、**中身は学習に使った item だけ**であり、未ラベル画像は 1 枚も入らない。
    つまり既存の推論は「学習データに対するスコア」を出すもので、
    MVP の到達点である「**残りの画像に**ラベルが付いた状態」には届かない。

    未ラベルを対象にするには入力源を index.json から labels.json へ変える必要があり、
    出力も JSON ではなく labels.json への書き戻しになる。入力も出力も別物なので、
    動作実績のある `infer_heads.py` に分岐を足さず、別の経路として書く。

画像を読み直さない:

    `embed_images.py` は **labels.json の全 item**（未ラベルを含む）を埋め込んで
    `cache/embeddings/` に置く。head は CLIP 埋め込みの上の Linear なので、
    キャッシュに head を掛けるだけでよい。CLIP を通し直さないため、
    1,000 枚規模でも実用的な時間で終わる。

    したがって **`embed_images` を先に流しておく必要がある**。埋め込みの無い item は
    書き戻さず、件数として報告する。

安全側の作り（詳細は `label_writeback.py`）:

    - 人が付けたラベルは上書きしない（head 単位で判定）
    - 予測は `predicted` に由来を残し、`committed` は立てない
    - `split` に触れないので、書き戻しただけでは dataset に入らない
      （＝自分の予測で再学習することはない）
    - 書き戻す前に labels.json のスナップショットを history へ取る。
      既存の LabelHistory から復元できる（一括変更は必ず戻せるようにする）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from embedding_cache import embeddings_dir, load_index  # noqa: E402
from head_checkpoint import linear_shape, normalize_state_dict  # noqa: E402
from job_progress import update_job_progress  # noqa: E402
from label_schema import canonical_head_type, get_heads, load_schema  # noqa: E402
from label_writeback import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    WritebackReport,
    apply_prediction,
    commit_writeback,
    plan_writeback,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] 不正な JSON です: {path}")
    return data


def _resolve_heads(
    run_dir: Path, schema_path: Optional[Path]
) -> List[tuple[str, str, Dict[str, Any]]]:
    """run_dir の checkpoint から (head, head_type, ckpt) を集める。

    `split` 型は対象外。あれはデータの振り分けであってラベルではない。
    """
    schema_types: Dict[str, str] = {}
    if schema_path and schema_path.exists():
        try:
            for h in get_heads(load_schema(schema_path)):
                if isinstance(h, dict) and str(h.get("id") or "").strip():
                    schema_types[str(h["id"])] = canonical_head_type(h.get("type"))
        except Exception as exc:  # スキーマが壊れていても ckpt 側の型で続行できる
            print(f"[WARN] label_schema.json を読めませんでした: {exc}")

    found: List[tuple[str, str, Dict[str, Any]]] = []
    for ckpt_path in sorted(run_dir.glob("*_linear.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[WARN] checkpoint を読めませんでした: {ckpt_path} ({exc})")
            continue
        head = str(ckpt.get("class_head") or ckpt_path.stem.removesuffix("_linear"))
        head_type = schema_types.get(head) or str(ckpt.get("head_type") or "multi_class")
        if head_type == "split":
            print(f"[INFO] head '{head}' は split 型のため対象外")
            continue
        found.append((head, head_type, ckpt))
    return found


def _load_embeddings(cache_root: Path, model_name: str) -> tuple[Dict[str, int], "np.ndarray"]:
    """file_id -> 行番号 の対応と、埋め込み行列を返す。"""
    dir_path = embeddings_dir(cache_root, model_name)
    index = load_index(dir_path)
    matrix_path = dir_path / "embeddings.npy"
    if index is None or not matrix_path.exists():
        raise SystemExit(
            f"[ERROR] 埋め込みキャッシュがありません: {dir_path}\n"
            "        先に embed_images ジョブを実行してください。"
        )
    rows = {
        str(e.get("file_id")): int(e.get("row", -1))
        for e in index.get("items", [])
        if isinstance(e, dict) and e.get("file_id") is not None
    }
    return rows, np.load(matrix_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="学習済み head で未ラベル画像を推論し labels.json へ書き戻す (agent)"
    )
    parser.add_argument("--labels", type=str, required=True, help="labels.json のパス")
    parser.add_argument("--run-dir", type=str, required=True, help="学習済み head の run ディレクトリ")
    parser.add_argument("--cache-root", type=str, default=None, help="cache/ の基準（省略時は labels.json から推定）")
    parser.add_argument("--schema", type=str, default=None, help="label_schema.json（省略時は labels.json の隣を推定）")
    parser.add_argument("--heads", type=str, default=None, help="対象 head をカンマ区切りで指定")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"書き戻す下限スコア (既定 {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="labels.json を書き換えず、何件書き戻すかだけを出す",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels).resolve()
    if not labels_path.exists():
        raise SystemExit(f"[ERROR] labels.json がありません: {labels_path}")
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"[ERROR] run ディレクトリがありません: {run_dir}")

    label_input_dir = labels_path.parent
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (
        label_input_dir.parent / "cache"
    )
    schema_path = Path(args.schema).resolve() if args.schema else (
        label_input_dir / "label_schema.json"
    )

    update_job_progress(phase="load", message="loading labels and checkpoints")

    data = _load_json(labels_path)
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"[ERROR] labels.json に items がありません: {labels_path}")

    heads = _resolve_heads(run_dir, schema_path)
    if args.heads:
        requested = {h.strip() for h in args.heads.split(",") if h.strip()}
        heads = [h for h in heads if h[0] in requested]
    if not heads:
        raise SystemExit(f"[ERROR] 対象となる head がありません: {run_dir}")

    clip_name = str(heads[0][2].get("clip_model_name") or "ViT-B/32")
    rows_by_file_id, matrix = _load_embeddings(cache_root, clip_name)
    print(f"[INFO] labels.json: {labels_path}（items={len(items)}）")
    print(f"[INFO] run dir: {run_dir}")
    print(f"[INFO] 埋め込み: {matrix.shape[0]} 行 / clip={clip_name}")
    print(f"[INFO] 対象 head: {[h for h, _, _ in heads]}")
    print(f"[INFO] 下限スコア: {args.min_score}")

    report = WritebackReport()
    per_head: Dict[str, Dict[str, int]] = {}
    total_written = 0

    for hi, (head, head_type, ckpt) in enumerate(heads):
        state = normalize_state_dict(ckpt.get("state_dict") or ckpt.get("head_state") or {})
        shape = linear_shape(state) if state else None
        if state is None or shape is None:
            print(f"[WARN] head '{head}' の重みを読めませんでした（skip）")
            continue
        num_classes, in_dim = shape
        if in_dim != int(matrix.shape[1]):
            raise SystemExit(
                f"[ERROR] head '{head}' の入力次元 {in_dim} と埋め込み {matrix.shape[1]} が一致しません。\n"
                "        学習時と別の CLIP モデルで埋め込んでいる可能性があります。"
            )
        classes = [str(c) for c in (ckpt.get("classes") or [])]
        if len(classes) != num_classes:
            print(f"[WARN] head '{head}' の classes 数がずれています（skip）")
            continue

        model = torch.nn.Linear(in_dim, num_classes)
        model.load_state_dict(state)
        model.eval()

        # 未ラベル item に限って推論する。全件に掛けても捨てるだけなので、対象を先に絞る。
        head_report = WritebackReport()
        candidates = [
            it for it in items
            if isinstance(it, dict) and str(it.get("file_id") or "") in rows_by_file_id
        ]
        scores_by_file_id: Dict[str, Dict[str, float]] = {}
        with torch.no_grad():
            for chunk_start in range(0, len(candidates), 512):
                chunk = candidates[chunk_start : chunk_start + 512]
                idx = [rows_by_file_id[str(it["file_id"])] for it in chunk]
                feats = torch.from_numpy(matrix[idx]).float()
                logits = model(feats)
                probs = (
                    torch.sigmoid(logits)
                    if head_type == "multi_label"
                    else torch.softmax(logits, dim=-1)
                )
                for row_i, it in enumerate(chunk):
                    scores_by_file_id[str(it["file_id"])] = {
                        classes[c]: float(probs[row_i][c].item()) for c in range(num_classes)
                    }

        planned = plan_writeback(
            items,
            head=head,
            head_type=head_type,
            scores_by_file_id=scores_by_file_id,
            min_score=float(args.min_score),
            report=head_report,
        )
        if not args.dry_run:
            for item, prediction in planned:
                apply_prediction(item, head=head, prediction=prediction, run_name=run_dir.name)
        head_report.written = len(planned)
        total_written += len(planned)

        per_head[head] = head_report.as_dict()
        report.written += head_report.written
        report.skipped_existing += head_report.skipped_existing
        report.skipped_low_score += head_report.skipped_low_score
        report.skipped_deleted += head_report.skipped_deleted
        report.skipped_no_embedding += head_report.skipped_no_embedding

        print(
            f"[INFO] {head}: 書き戻し {head_report.written} / "
            f"既にラベルあり {head_report.skipped_existing} / "
            f"スコア不足 {head_report.skipped_low_score} / "
            f"削除対象 {head_report.skipped_deleted} / "
            f"埋め込み無し {head_report.skipped_no_embedding}"
        )
        update_job_progress(
            phase="infer", current=hi + 1, total=len(heads), message=f"predicted {head}"
        )

    if args.dry_run:
        print(f"[INFO] dry-run のため labels.json は変更していません（書き戻し予定 {total_written} 件）")
    elif total_written == 0:
        # 何も書かないのにスナップショットだけ増やさない。
        print("[INFO] 書き戻す対象がありませんでした。labels.json は変更していません")
        if report.skipped_existing and not report.skipped_low_score:
            # 「動かなかった」と誤解されやすい場面なので、理由を名指しする。
            # 2 回目以降は前回書き戻した値が「ラベルあり」に数えられるため、
            # モデルを学習し直しても既定では上書きしない（T2-2 は手動 1 回の範囲）。
            print(
                "[INFO] 全ての item に既に値があります。前回の書き戻し結果を上書きする"
                "経路は現時点ではありません（学習し直して付け直したい場合は、"
                "LabelHistory から書き戻し前の版に戻してから再実行してください）"
            )
        elif report.skipped_low_score:
            print(
                f"[INFO] スコアが下限 {args.min_score} に届かない item が "
                f"{report.skipped_low_score} 件ありました。--min-score を下げると増えます"
            )
    else:
        snapshot = commit_writeback(labels_path, data, written=total_written)
        print(f"[INFO] 書き戻し前のスナップショット: {snapshot}")
        print(f"[INFO] 書き戻しました: {labels_path}（{total_written} 件）")

    update_job_progress(
        phase="done",
        message="predict_unlabeled completed",
        extra={"written": total_written, "per_head": per_head, "dry_run": bool(args.dry_run)},
    )


if __name__ == "__main__":
    main()
