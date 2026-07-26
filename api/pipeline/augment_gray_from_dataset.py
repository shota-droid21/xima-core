#!/usr/bin/env python3
"""agent/pipeline/augment_gray_from_dataset.py

NOTE:
- This file is a copy of 02_train/augment_gray_from_dataset.py (copy-only migration).
- Do not edit the original in 02_train during the migration phase.

Purpose:
- Read experiments/<exp>/dataset/index.json and generate grayscale variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image  # pip install pillow

from job_progress import update_job_progress


THIS_FILE = Path(__file__).resolve()
TRAIN_ROOT = THIS_FILE.parent
DEFAULT_DATASET_ROOT = TRAIN_ROOT / "dataset"


def load_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        raise SystemExit(f"[ERROR] index.json が見つかりません: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "items" not in data or not isinstance(data["items"], list):
        raise SystemExit(
            "[ERROR] index.json の形式が不正です (items が list ではありません)"
        )
    return data


def is_gray_item(item: Dict[str, Any], suffix: str) -> bool:
    meta = item.get("meta") or {}
    if isinstance(meta, dict) and meta.get("is_gray") is True:
        return True

    dataset_path = item.get("dataset_path")
    if not isinstance(dataset_path, str):
        return False

    p = Path(dataset_path)
    return p.stem.endswith(suffix)


def make_gray_path(dataset_path: str, suffix: str) -> str:
    p = Path(dataset_path)
    return str(p.with_name(p.stem + suffix + p.suffix))


def convert_to_gray(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_path)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        alpha = img.split()[-1]
        bg.paste(img.convert("RGB"), mask=alpha)
        gray = bg.convert("L")
    else:
        gray = img.convert("L")

    gray.save(dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "dataset/index.json を読み込み、対象 split の画像に対してモノクロ版を生成し "
            "index.json に gray item を追加するスクリプト"
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "dataset ディレクトリのルートパス。この直下に index.json, train/, val/ などがある前提。"
        ),
    )
    parser.add_argument(
        "--index",
        type=str,
        default="index.json",
        help="index.json のファイル名 or パス (dataset-root からの相対パス)。",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="モノクロ拡張の対象 split のカンマ区切り (例: train,val)。",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_gray",
        help="生成されるファイル名に付与するサフィックス (デフォルト: _gray)。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際にはファイル/ index.json を書き換えず、対象件数のみを表示する。",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="モノクロ生成する最大件数 (デバッグ用)。",
    )

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting augment_gray_from_dataset")

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"[ERROR] dataset-root が存在しません: {dataset_root}")

    index_path = (dataset_root / args.index).resolve()
    index_data = load_index(index_path)

    target_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if not target_splits:
        raise SystemExit("[ERROR] splits が空です")

    print(f"[INFO] dataset-root: {dataset_root}")
    print(f"[INFO] index.json:   {index_path}")
    print(f"[INFO] target splits: {sorted(target_splits)}")
    print(f"[INFO] suffix:        {args.suffix}")
    if args.dry_run:
        print("[INFO] DRY-RUN モード: ファイルや index.json は書き換えません")

    items: List[Dict[str, Any]] = index_data["items"]
    existing_ids = {it.get("id") for it in items if it.get("id") is not None}

    update_job_progress(
        phase="scan_candidates",
        current=0,
        total=len(items),
        message="scanning candidates",
    )

    candidate_indices: List[int] = []
    for idx, it in enumerate(items):
        if idx % 200 == 0:
            update_job_progress(
                phase="scan_candidates",
                current=idx,
                total=len(items),
                message="scanning candidates",
                extra={"candidates": len(candidate_indices)},
            )
        split = it.get("split")
        if split not in target_splits:
            continue
        if is_gray_item(it, args.suffix):
            continue

        dataset_path = it.get("dataset_path")
        if not isinstance(dataset_path, str):
            continue

        src_path = dataset_root / dataset_path
        if not src_path.exists():
            print(f"[WARN] 元画像が見つからないためスキップ: {src_path}")
            continue

        candidate_indices.append(idx)

    if not candidate_indices:
        print("[INFO] モノクロ変換対象となる item はありませんでした")
        update_job_progress(
            phase="done",
            current=0,
            total=0,
            message="no candidates",
        )
        return

    print(f"[INFO] モノクロ変換対象 item 数: {len(candidate_indices)}")

    if args.max_items is not None and args.max_items > 0:
        candidate_indices = candidate_indices[: args.max_items]
        print(f"[INFO] max-items により先頭 {len(candidate_indices)} 件に制限")

    new_items: List[Dict[str, Any]] = []
    num_converted = 0

    update_job_progress(
        phase="convert",
        current=0,
        total=len(candidate_indices),
        message="converting images",
    )

    for i, idx in enumerate(candidate_indices, start=1):
        it = items[idx]
        dataset_path = it.get("dataset_path")
        if not isinstance(dataset_path, str):
            continue

        src_path = dataset_root / dataset_path
        dst_rel_path = make_gray_path(dataset_path, args.suffix)
        dst_path = dataset_root / dst_rel_path

        src_id = it.get("id")
        if isinstance(src_id, str):
            new_id_base = src_id + "_gray"
        else:
            new_id_base = dataset_path.replace("/", "_") + "_gray"

        new_id = new_id_base
        counter = 1
        while new_id in existing_ids:
            counter += 1
            new_id = f"{new_id_base}_{counter}"

        if args.dry_run:
            print(
                f"[DRY] [{i}/{len(candidate_indices)}] {src_path} -> {dst_path} (id={new_id})"
            )
        else:
            try:
                convert_to_gray(src_path, dst_path)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[WARN] グレースケール変換に失敗したためスキップ: {src_path}: {e}"
                )
                continue

        labels = it.get("labels") or {}
        if not isinstance(labels, dict):
            labels = {}

        new_labels = dict(labels)
        if "concept_color" in new_labels:
            new_labels["concept_color"] = None
        if "hair_color" in new_labels:
            new_labels["hair_color"] = None

        meta = it.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        new_meta = dict(meta)
        new_meta["is_gray"] = True
        if src_id is not None:
            new_meta["from_id"] = src_id

        new_item: Dict[str, Any] = {
            "id": new_id,
            "dataset_path": dst_rel_path,
            "source_path": it.get("source_path"),
            "split": it.get("split"),
            "labels": new_labels,
            "meta": new_meta,
        }

        new_items.append(new_item)
        existing_ids.add(new_id)
        num_converted += 1

        if i == 1 or i % 20 == 0 or i == len(candidate_indices):
            update_job_progress(
                phase="convert",
                current=i,
                total=len(candidate_indices),
                message=f"converted {i}/{len(candidate_indices)}",
                extra={"num_items_added": num_converted},
            )

        if not args.dry_run:
            print(
                f"[OK] [{i}/{len(candidate_indices)}] {src_path} -> {dst_path} (id={new_id})"
            )

    if args.dry_run:
        print(
            f"[INFO] DRY-RUN: 実際には index.json は更新していません (new_items={len(new_items)})"
        )
        update_job_progress(
            phase="done",
            current=len(candidate_indices),
            total=len(candidate_indices),
            message="dry-run completed",
            extra={"num_items_added": len(new_items)},
        )
        return

    if new_items:
        index_data["items"].extend(new_items)
        meta = index_data.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("gray_augmentation", {})
        ga = meta["gray_augmentation"]
        if not isinstance(ga, dict):
            ga = {}
        ga.update(
            {
                "suffix": args.suffix,
                "splits": sorted(target_splits),
                "num_items_added": num_converted,
            }
        )
        meta["gray_augmentation"] = ga
        index_data["meta"] = meta

        update_job_progress(phase="write_index", message="writing index.json")
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 追加された gray item 数: {num_converted}")
    print(f"[INFO] index.json を更新しました: {index_path}")
    update_job_progress(
        phase="done",
        current=num_converted,
        total=len(candidate_indices),
        message="completed augment_gray_from_dataset",
        extra={"num_items_added": num_converted, "index_path": str(index_path)},
    )


if __name__ == "__main__":
    main()
