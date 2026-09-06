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

from label_schema import get_heads, load_schema, normalize_label_for_head

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

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting apply_label_mapping")

    labels_path = Path(args.labels).resolve()
    root_dir = Path(args.root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    class_head = args.class_head

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
                out_labels[hid] = normalize_label_for_head(out_labels.get(hid), head)
        split = out_labels.get("split")
        if split is not None and not isinstance(split, str):
            split = str(split)
        if isinstance(split, str):
            split = split.strip().lower()
            out_labels["split"] = split
        deleted = it.get("delete", False)

        if deleted or split not in ("train", "val"):
            n_skipped += 1
            continue

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
            "version": 2,
        },
        "items": index_items,
    }

    ensure_dir(dataset_root)
    index_path = dataset_root / "index.json"
    update_job_progress(phase="write_index", message="writing index.json")
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)

    print("----- SUMMARY -----")
    print(f" processed:   {n_processed}")
    print(f" skipped:     {n_skipped}")
    print(f" missing src: {n_missing_src}")
    print(f" removed:     {n_removed}")
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
            "index_path": str(index_path),
        },
    )


if __name__ == "__main__":
    main()
