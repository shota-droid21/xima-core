#!/usr/bin/env python3
"""agent/pipeline/apply_label_mapping.py

NOTE:
- This file is a copy of 02_train/apply_label_mapping.py (copy-only migration).
- Do not edit the original in 02_train during the migration phase.

Purpose:
- Build experiments/<exp>/dataset/{train,val} + index.json from label_input/labels.json.
"""

import argparse
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from dataset_split import DEFAULT_VAL_RATIO, resolve_split
from label_schema import get_head_classes, get_heads, load_schema, normalize_label_for_head
from unusable_labels import summarize, unusable_values

from job_progress import update_job_progress


def load_labels(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_id_from_path(path: str) -> str:
    """安定ID生成（ファイルパスのsha1の上位12文字）"""
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return h[:12]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "label_input/labels.json を元に dataset(train/val + index.json) を構築する "
            "(multi-head / head非依存)"
        )
    )

    parser.add_argument(
        "--labels", "-l", required=True, help="label_input/labels.json のパス"
    )
    parser.add_argument(
        "--root", "-r", required=True, help="元画像root (source_imagesなど)"
    )
    parser.add_argument(
        "--dataset-root",
        "-d",
        default="dataset",
        help="出力 dataset ルート (デフォルト: ./dataset)",
    )
    parser.add_argument(
        "--class-head",
        default=None,
        help=(
            "meta に記録するデフォルト head 名 (フィルタには使わない)。\n"
            "train_epoch 側で --heads / --class-head / スキーマのいずれも無いときの\n"
            "最後の手掛かりとして使われる。未指定なら記録しない。"
        ),
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help=(
            "label_schema.json のパス "
            "(Job API経由では通常 <experiment>/label_schema.json が注入される。"
            "CLI直接実行で省略時は legacy/10_tools/labeling/label_schema.json を探索して読み込む)"
        ),
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=(
            "split を書いていない item のうち val へ回す割合 (既定: "
            f"{DEFAULT_VAL_RATIO})。file_id のハッシュで決めるので、"
            "何度回しても同じ item は同じ側に入る (#325)"
        ),
    )

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting apply_label_mapping")

    labels_path = Path(args.labels).resolve()
    root_dir = Path(args.root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    class_head = args.class_head
    val_ratio = min(max(float(args.val_ratio), 0.0), 1.0)

    # load schema if available
    schema = None
    schema_head_map: Dict[str, Dict[str, Any]] = {}
    schema_heads: List[str] = []
    try:
        schema = load_schema(args.schema) if args.schema else load_schema()
        for head in get_heads(schema):
            if not isinstance(head, dict):
                continue
            hid = str(head.get("id") or "").strip()
            if not hid:
                continue
            schema_head_map[hid] = head
        schema_heads = list(schema_head_map.keys())
        print(f"[INFO] schema heads: {schema_heads}")
    except Exception as exc:
        if args.schema:
            raise SystemExit(f"[ERROR] failed to load schema: {exc}")
        schema = None
        schema_head_map = {}
        schema_heads = []

    if not labels_path.exists():
        raise SystemExit(f"[ERROR] labels JSON が存在しません: {labels_path}")

    data = load_labels(labels_path)
    # labels.json 形式を前提: { "meta": ..., "items": [...] }
    items: List[Dict] = data.get("items", [])
    if not isinstance(items, list):
        raise SystemExit(
            f"[ERROR] labels JSON の形式が不正です (items が list ではありません): {labels_path}"
        )

    print(f"[INFO] labels: {labels_path}")
    print(f"[INFO] root_dir: {root_dir}")
    print(f"[INFO] dataset_root: {dataset_root}")
    print(f"[INFO] class_head(meta用): {class_head if class_head else '(未指定)'}")
    print(f"[INFO] ラベル件数: {len(items)}")

    update_job_progress(
        phase="scan_items",
        current=0,
        total=len(items),
        message="scanning items",
    )

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    index_items: List[Dict] = []
    desired_paths: Set[Path] = set()

    n_processed = 0
    n_skipped = 0
    n_missing_src = 0
    # 値はあるのに、ラベルの定義に無いもの（#333）。head ごと・値ごとに数える。
    # **落とすか残すかは変えない。** 見えていなかったことだけを直す。
    unusable_values_by_head: Dict[str, Dict[str, int]] = {}
    unusable_items_by_head: Dict[str, int] = {}
    #: dataset へ入らなかった理由の内訳（`deleted` / `excluded` / `unlabeled` /
    #: `invalid_split`）。`skipped` の総数だけでは何が起きたか読めない。
    skipped_reasons: Dict[str, int] = {}

    for it in items:
        n_seen = n_processed + n_skipped + n_missing_src
        if n_seen % 50 == 0:
            update_job_progress(
                phase="scan_items",
                current=n_seen,
                total=len(items),
                message="scanning items",
                extra={
                    "processed": n_processed,
                    "skipped": n_skipped,
                    "missing_src": n_missing_src,
                },
            )

        source_path = it.get("path")
        if not source_path:
            n_skipped += 1
            continue

        labels = it.get("labels", {}) or {}
        if not isinstance(labels, dict):
            labels = {}
        out_labels = dict(labels)
        if schema_head_map:
            for hid, head in schema_head_map.items():
                head_type = str(head.get("type") or "")
                # `split` はデータ管理用の system head で、学習するものではない
                # （Decision 011）。値の正否は `dataset_split` が決めるので、
                # ここで数えると **`exclude` が「定義に無い値」として出てしまう**
                # （既存のスキーマは choices に `unassigned` を持っている）。
                if hid == "split" or head_type == "split":
                    out_labels[hid] = normalize_label_for_head(out_labels.get(hid), head)
                    continue
                # 正規化の**前**に数える。multi_label はここで値が落ちるので、
                # 落ちたあとでは何が消えたか分からない（#333）。
                bad = unusable_values(
                    out_labels.get(hid),
                    head_type=head_type,
                    classes=get_head_classes(head),
                )
                if bad:
                    counts = unusable_values_by_head.setdefault(hid, {})
                    for value in bad:
                        counts[value] = counts.get(value, 0) + 1
                    unusable_items_by_head[hid] = unusable_items_by_head.get(hid, 0) + 1
                out_labels[hid] = normalize_label_for_head(out_labels.get(hid), head)
        deleted = bool(it.get("delete", False))

        # **ラベルがあれば入る**（#325 / Decision 045）。規則は dataset_split に
        # 1 つだけ置いてある。ここで判定を書き足さない。
        split_key = str(it.get("file_id") or source_path)
        split, split_source = resolve_split(
            out_labels,
            deleted=deleted,
            key=split_key,
            head_ids=schema_heads,
            val_ratio=val_ratio,
        )
        if split is None:
            n_skipped += 1
            skipped_reasons[split_source] = skipped_reasons.get(split_source, 0) + 1
            continue
        # 自動で決めた分は labels.json へ書き戻さない。**書くと「人が決めた」と
        # 区別できなくなる。** index.json にだけ残す。
        out_labels.pop("split", None)
        if split_source == "pinned":
            out_labels["split"] = split

        src = root_dir / source_path
        if not src.exists():
            print(f"[WARN] 元ファイルがありません: {src}")
            n_missing_src += 1
            continue

        stable_id = compute_id_from_path(source_path)
        filename = stable_id + src.suffix.lower()

        dst_dir = train_dir if split == "train" else val_dir
        dst = dst_dir / filename
        desired_paths.add(dst.resolve())

        index_items.append(
            {
                "id": stable_id,
                "dataset_path": str(dst.relative_to(dataset_root)),
                "source_path": str(source_path),
                "split": split,
                "split_source": split_source,
                "delete": bool(deleted),
                "labels": out_labels,
            }
        )

        it["_copy_task"] = (src, dst)

    update_job_progress(
        phase="remove_stale",
        message="removing stale files",
    )

    n_removed = 0
    for sub in (train_dir, val_dir):
        if not sub.exists():
            continue
        for p in sub.rglob("*"):
            if p.is_file() and p.resolve() not in desired_paths:
                print(f"[REMOVE] {p}")
                p.unlink()
                n_removed += 1

    update_job_progress(
        phase="copy_files",
        current=0,
        total=len(index_items),
        message="copying files",
    )

    to_copy = [it for it in items if "_copy_task" in it]
    for i, it in enumerate(to_copy, start=1):
        if "_copy_task" not in it:
            continue
        src, dst = it["_copy_task"]
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}")
        n_processed += 1

        if i == 1 or i % 20 == 0 or i == len(to_copy):
            update_job_progress(
                phase="copy_files",
                current=i,
                total=len(to_copy),
                message=f"copying {i}/{len(to_copy)}",
            )

    index_json = {
        "meta": {
            "source_label_path": str(labels_path),
            "class_head": class_head,
            "created_at": datetime.now().isoformat(),
            # 自動割りを再現できるようにする（#325）。item ごとの `split_source` と対。
            "val_ratio": val_ratio,
            "version": 2,
        },
        "items": index_items,
    }

    ensure_dir(dataset_root)
    index_path = dataset_root / "index.json"
    update_job_progress(phase="write_index", message="writing index.json")
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)

    unusable: Dict[str, Any] = {}
    for hid, counts in unusable_values_by_head.items():
        found = summarize(counts, unusable_items_by_head.get(hid, 0))
        if found is not None:
            unusable[hid] = found

    print("----- SUMMARY -----")
    print(f" processed:   {n_processed}")
    print(f" skipped:     {n_skipped}")
    for reason, count in sorted(skipped_reasons.items()):
        print(f"   - {reason}: {count}")
    n_auto = sum(1 for it in index_items if it.get("split_source") == "auto")
    print(f" auto split:  {n_auto} (val_ratio={val_ratio})")
    print(f" missing src: {n_missing_src}")
    print(f" removed:     {n_removed}")
    for hid, found in unusable.items():
        listed = ", ".join(f"{v}({c})" for v, c in found["values"].items())
        # skipped と混ぜない。あちらは「dataset の外にある」という正常な状態で、
        # こちらは「入れるつもりだったのに値が壊れていて入らない」である（#333）。
        print(f"[WARN] {hid}: 定義に無い値 {found['items']} 件 -> {listed}")
    print(f"[INFO] index.json written to: {index_path}")
    update_job_progress(
        phase="done",
        current=n_processed,
        total=len(items),
        message="completed apply_label_mapping",
        extra={
            "processed": n_processed,
            "skipped": n_skipped,
            "missing_src": n_missing_src,
            "removed": n_removed,
            "skipped_reasons": skipped_reasons,
            "val_ratio": val_ratio,
            "index_path": str(index_path),
            # 0 件なら {}。**黙って落とさない**ためだけの数で、処理は変えていない。
            "schema_violations": unusable,
        },
    )


if __name__ == "__main__":
    main()
