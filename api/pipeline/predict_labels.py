"""学習済み head で予測を出し、labels.json に **候補として**記録する（T2-2）。

ワークフロー上の位置（2 周目以降）:

    make_label_list -> embed_images -> **predict_labels** -> labeling(候補付き) -> apply_label -> train

`labels` には書かない。書くのは `item["predicted"]` だけで、確定は単体画面で人が行う
（理由は `label_predictions.py` の docstring）。

なぜ `infer_heads.py` を拡張しないのか:

    `infer_heads.py` の対象は `dataset/index.json` である。この index は
    `apply_label_mapping.py` が `if deleted or split not in ("train","val"): continue`
    で作るため、**中身は学習に使った item だけ**であり、未ラベル画像は 1 枚も入らない。
    入力（index.json -> labels.json）も出力（JSON 出力 -> labels.json への記録）も別物なので、
    動作実績のある `infer_heads.py` に分岐を足さず別経路として書く。

画像を読み直さない:

    `embed_images.py` は **labels.json の全 item**（未ラベルを含む）を埋め込んで
    `cache/embeddings/` に置く。head は CLIP 埋め込みの上の Linear なので、
    キャッシュに head を掛けるだけでよい。したがって **`embed_images` を先に流しておく
    必要がある**。埋め込みは学習に使ったのと**同じ CLIP モデル**のものが要る。

対象は全 item:

    未分類が主用途だが、class を追加した / 再ラベルする場面では既ラベル画像の候補も要る。
    `labels` に書かない以上、対象を広げても既存の作業を脅かさない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from embedding_cache import embeddings_dir, load_index  # noqa: E402
from head_checkpoint import (  # noqa: E402
    linear_shape,
    normalize_state_dict,
    read_temperature,
)
from job_progress import update_job_progress  # noqa: E402
from label_schema import canonical_head_type, get_heads, load_schema  # noqa: E402
from label_predictions import (  # noqa: E402
    PredictionReport,
    plan_predictions,
    record_prediction,
    write_json_atomic,
)
from prediction_reliability import (  # noqa: E402
    agreement_points,
    build_reliability,
    collect_samples,
    val_source_paths,
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
        # **どの CLIP モデルの埋め込みが要るか**まで書く。embed_images の既定は
        # ViT-B/32 であり、学習に別のモデルを使っていると「embed_images を実行しろ」
        # だけでは同じエラーに戻ってくる。
        raise SystemExit(
            f"[ERROR] このモデルに必要な埋め込みキャッシュがありません: {dir_path}\n"
            f"        学習に使われた CLIP モデルは {model_name} です。\n"
            f"        先に embed_images を **同じモデルで** 実行してください:\n"
            f"          --clip-model {model_name}"
        )
    rows = {
        str(e.get("file_id")): int(e.get("row", -1))
        for e in index.get("items", [])
        if isinstance(e, dict) and e.get("file_id") is not None
    }
    return rows, np.load(matrix_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="学習済み head で予測を出し labels.json に候補として記録する (agent)"
    )
    parser.add_argument("--labels", type=str, required=True, help="labels.json のパス")
    parser.add_argument("--run-dir", type=str, required=True, help="学習済み head の run ディレクトリ")
    parser.add_argument("--cache-root", type=str, default=None, help="cache/ の基準（省略時は labels.json から推定）")
    parser.add_argument("--schema", type=str, default=None, help="label_schema.json（省略時は labels.json の隣を推定）")
    parser.add_argument("--heads", type=str, default=None, help="対象 head をカンマ区切りで指定")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="labels.json を書き換えず、何件に候補が付くかだけを出す",
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
    # 一致率を val だけで測るための対象。無ければ人がラベルした全件に落とす
    # （楽観側に振れるので、その旨は basis として UI に渡す）。
    val_paths = val_source_paths(label_input_dir.parent / "dataset" / "index.json")

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

    per_head: Dict[str, Dict[str, int]] = {}
    reliability_heads: Dict[str, Dict[str, Any]] = {}
    total_recorded = 0

    for hi, (head, head_type, ckpt) in enumerate(heads):
        state = normalize_state_dict(ckpt.get("state_dict") or ckpt.get("head_state") or {})
        shape = linear_shape(state) if state else None
        if state is None or shape is None:
            print(f"[WARN] head '{head}' の重みを読めませんでした（skip）")
            continue
        num_classes, in_dim = shape
        if in_dim != int(matrix.shape[1]):
            raise SystemExit(
                f"[ERROR] head '{head}' の入力次元 {in_dim} と埋め込み {matrix.shape[1]} が"
                "一致しません。\n"
                f"        学習は {clip_name} で行われています。embed_images を"
                "同じモデルで実行し直してください。"
            )
        classes = [str(c) for c in (ckpt.get("classes") or [])]
        if len(classes) != num_classes:
            print(f"[WARN] head '{head}' の classes 数がずれています（skip）")
            continue

        model = torch.nn.Linear(in_dim, num_classes)
        model.load_state_dict(state)
        model.eval()
        temperature = read_temperature(ckpt)
        if temperature != 1.0:
            print(f"[INFO] {head}: 温度 T={temperature:.4f} を適用します")

        # **全 item が対象**（削除対象を除く）。class 追加・再ラベルの場面で
        # 既ラベル画像の候補も要るため。labels に書かないので広げても安全。
        head_report = PredictionReport()
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
                # 温度で割ってから確率にする。argmax は変わらず、確率だけが校正される
                # （学習時に val で当てはめた値。無い checkpoint は 1.0）。
                logits = model(feats) / temperature
                probs = (
                    torch.sigmoid(logits)
                    if head_type == "multi_label"
                    else torch.softmax(logits, dim=-1)
                )
                for row_i, it in enumerate(chunk):
                    scores_by_file_id[str(it["file_id"])] = {
                        classes[c]: float(probs[row_i][c].item()) for c in range(num_classes)
                    }

        planned = plan_predictions(
            items,
            scores_by_file_id=scores_by_file_id,
            head_type=head_type,
            report=head_report,
        )

        # 閾値ごとの実測一致率。**候補を記録する前**に、いま計算した予測から測る。
        # 一括確定ダイアログはこの表だけを見て閾値を選ばせるので、
        # ここで測らないと利用者は根拠のない数字を選ぶことになる。
        confidence_by_file_id = {}
        predicted_by_file_id = {}
        for item, prediction in planned:
            fid = str(item.get("file_id") or "")
            confidence_by_file_id[fid] = float(prediction.confidence)
            predicted_by_file_id[fid] = prediction.value
        samples = collect_samples(
            items,
            head=head,
            head_type=head_type,
            confidence_by_file_id=confidence_by_file_id,
            predicted_by_file_id=predicted_by_file_id,
            limit_to_paths=val_paths or None,
        )
        reliability_heads[head] = {
            "head_type": head_type,
            "n": len(samples),
            "points": agreement_points(samples),
        }

        if not args.dry_run:
            for item, prediction in planned:
                record_prediction(
                    item,
                    head=head,
                    head_type=head_type,
                    prediction=prediction,
                    run_name=run_dir.name,
                )
        head_report.recorded = len(planned)
        total_recorded += len(planned)
        per_head[head] = head_report.as_dict()

        print(
            f"[INFO] {head}: 候補 {head_report.recorded} 件 / "
            f"削除対象 {head_report.skipped_deleted} / "
            f"埋め込み無し {head_report.skipped_no_embedding}"
        )
        if samples:
            basis_label = "val" if val_paths else "ラベル済み全件（train を含むため高めに出ます）"
            print(f"[INFO] {head}: 閾値ごとの実測一致率（{basis_label} {len(samples)} 件）")
            for pt in reliability_heads[head]["points"]:
                if not pt["n"]:
                    continue
                print(
                    f"         >= {pt['threshold']:.2f}  {pt['n']:4d} 件  "
                    f"一致率 {pt['agreement']:.3f}"
                )
        else:
            print(f"[INFO] {head}: 一致率を測れる item がありません（一括確定は使えません）")
        update_job_progress(
            phase="infer", current=hi + 1, total=len(heads), message=f"predicted {head}"
        )

    if args.dry_run:
        print(f"[INFO] dry-run のため labels.json は変更していません（候補 {total_recorded} 件）")
    elif total_recorded == 0:
        print("[INFO] 候補を付けられる item がありませんでした。labels.json は変更していません")
    else:
        if reliability_heads:
            # item ごとの値ではなく run 全体の性質なので meta に置く。
            # app は labels.json を読むだけで一括確定の判断材料を得られる。
            meta = data.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["prediction_reliability"] = build_reliability(
                reliability_heads,
                run_name=run_dir.name,
                basis="val" if val_paths else "labeled",
                now=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            data["meta"] = meta
        # **スナップショットは取らない。**labels（確定値）を変えていないため、
        # 履歴を残す意味が薄く、実行のたびに「変化のない版」で復元候補が埋まる。
        # 書き込み自体は tmp + fsync + replace で壊さない。
        write_json_atomic(labels_path, data)
        print(f"[INFO] 候補を記録しました: {labels_path}（{total_recorded} 件）")
        print("[INFO] labels（確定値）は変更していません。確定はラベリング画面で行います")

    update_job_progress(
        phase="done",
        message="predict_labels completed",
        extra={
            "recorded": total_recorded,
            "per_head": per_head,
            "dry_run": bool(args.dry_run),
        },
    )


if __name__ == "__main__":
    main()
