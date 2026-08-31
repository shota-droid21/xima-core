#!/usr/bin/env python3
"""agent/pipeline/embed_images.py

labels.json に取り込まれた画像の CLIP 埋め込みを生成し、
experiments/<exp>/cache/embeddings/<model_slug>/ にキャッシュする。

このキャッシュは以降の機能（一括推論の高速化 / 類似クラスタリング /
active learning）の共通基盤になる。埋め込みは `cache/` 配下の
**再生成可能な派生物**であり、消えても本ジョブの再実行で復元できる（Decision 004）。

差分計算:
- 既存キャッシュと `file_id` + コンテンツハッシュが一致するものは再計算しない
- 画像が差し替われば再計算、削除されれば index から落とす
- CLIP モデルごとにディレクトリを分けるため、モデル変更は別キャッシュになる

レイアウトと差分判定のロジックは torch/numpy 非依存の embedding_cache.py にある。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import clip

from device import _normalize_device_name, get_device
from embedding_cache import (
    MATRIX_FILENAME,
    build_index_entries,
    build_index_payload,
    build_output_plan,
    content_hash,
    embeddings_dir,
    load_index,
    plan_embeddings,
    save_index,
)
from image_io import is_decodable_image, load_image_rgb
from job_progress import update_job_progress


class _ImageDataset(torch.utils.data.Dataset):
    """埋め込み対象の画像を CLIP の前処理付きで返す。"""

    def __init__(self, paths: List[Path], *, clip_preprocess) -> None:
        self.paths = paths
        self.clip_preprocess = clip_preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image_rgb(self.paths[i])
        return self.clip_preprocess(img), i


def load_labels_items(labels_path: Path) -> List[Dict[str, Any]]:
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid labels.json: {labels_path}")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"labels.json has no items: {labels_path}")
    return [it for it in items if isinstance(it, dict)]


def build_targets(items: List[Dict[str, Any]], root: Path) -> List[Dict[str, Any]]:
    """labels.json の item から、実在する画像だけを対象として組み立てる。

    欠損画像は黙って除外する（labels.json の正本性は make_label_list 側の責務）。
    """
    targets: List[Dict[str, Any]] = []
    missing = 0
    unreadable: List[str] = []
    for item in items:
        rel_path = item.get("path")
        if not rel_path:
            missing += 1
            continue
        file_id = item.get("file_id") or Path(str(rel_path)).stem
        src = (root / str(rel_path)).resolve()
        if not src.exists() or not src.is_file():
            missing += 1
            continue
        # **読めない画像はここで外す。**残すと DataLoader の中で例外になり、
        # 1 枚のゴミでジョブ全体が落ちる（実データで 879 枚中 21 枚が HTML だった）。
        if not is_decodable_image(src):
            unreadable.append(str(rel_path))
            continue
        targets.append(
            {
                "file_id": str(file_id),
                "path": str(rel_path),
                "content_hash": content_hash(src),
                "_abs_path": src,
            }
        )
    if missing:
        print(f"[WARN] 画像ファイルが見つからない item を {missing} 件スキップしました")
    if unreadable:
        # **黙って落とさない。**枚数が減った理由が分からないと、精度が出ない原因を
        # モデル側に探しに行くことになる。何件で、どれかを名指しする。
        print(
            f"[WARN] 画像として読めないファイルを {len(unreadable)} 件スキップしました"
            "（拡張子は画像でも中身が違うものがあります）:"
        )
        for rel in unreadable[:10]:
            print(f"[WARN]   {rel}")
        if len(unreadable) > 10:
            print(f"[WARN]   ... 他 {len(unreadable) - 10} 件")
    return targets


def _save_matrix(dir_path: Path, matrix: "np.ndarray") -> None:
    """埋め込み行列を原子的に書き出す（中断時に壊れた行列を残さない）。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    final_path = dir_path / MATRIX_FILENAME
    tmp_path = dir_path / (MATRIX_FILENAME + ".tmp.npy")
    np.save(tmp_path, matrix)
    tmp_path.replace(final_path)


def _load_existing_matrix(dir_path: Path, expected_rows: int) -> Optional["np.ndarray"]:
    """既存の埋め込み行列を読む。読めない/行数が合わない場合は流用を諦める。"""
    path = dir_path / MATRIX_FILENAME
    if not path.exists():
        return None
    try:
        matrix = np.load(path)
    except Exception:  # noqa: BLE001
        return None
    if matrix.ndim != 2 or matrix.shape[0] < expected_rows:
        return None
    return matrix


def encode_paths(
    paths: List[Path],
    *,
    clip_model,
    clip_preprocess,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> "np.ndarray":
    """画像を CLIP で埋め込み、L2 正規化した float32 行列を返す。

    類似度計算をそのまま内積で行えるよう、保存時点で正規化しておく。
    """
    if not paths:
        return np.zeros((0, 0), dtype=np.float32)

    dataset = _ImageDataset(paths, clip_preprocess=clip_preprocess)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    chunks: List["np.ndarray"] = []
    total_batches = max(len(loader), 1)
    done = 0
    with torch.no_grad():
        for bi, (xb, _idx) in enumerate(loader):
            xb = xb.to(device)
            feats = clip_model.encode_image(xb).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().numpy().astype(np.float32))
            done += int(xb.shape[0])

            if (bi + 1) % 5 == 0 or (bi + 1) == total_batches:
                update_job_progress(
                    phase="embed",
                    current=done,
                    total=len(paths),
                    message=f"embedding {done}/{len(paths)}",
                    extra={"batch": bi + 1, "batches": total_batches},
                )

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="labels.json の画像を CLIP で埋め込み、cache/embeddings にキャッシュする (agent)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="labels.json のパス (experiments/<exp>/label_input/labels.json)",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="画像の基準ディレクトリ (workspaces/<ws>/source_images)",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="キャッシュ基準ディレクトリ (省略時は labels.json から experiments/<exp>/cache を推定)",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="強制デバイス指定 (cpu / cuda / metal[mps])。省略時は自動判定（cuda -> mps -> cpu）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存キャッシュを無視して全件を再計算する",
    )

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting embed_images")
    start_dt = datetime.now()
    print(f"[INFO] embed start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    labels_path = Path(args.labels).resolve()
    if not labels_path.exists():
        raise SystemExit(f"[ERROR] labels.json が見つかりません: {labels_path}")
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] 画像ルートが見つかりません: {root}")

    # label_input/labels.json -> experiments/<exp>/cache
    cache_root = (
        Path(args.cache_root).resolve()
        if args.cache_root
        else labels_path.parent.parent / "cache"
    )
    out_dir = embeddings_dir(cache_root, args.clip_model)
    print(f"[INFO] embeddings dir: {out_dir}")

    update_job_progress(phase="scan", message="scanning images")
    items = load_labels_items(labels_path)
    targets = build_targets(items, root)
    if not targets:
        raise SystemExit("[ERROR] 埋め込み対象の画像がありません（labels.json を確認してください）")

    existing_index = None if args.force else load_index(out_dir)
    plan = plan_embeddings(targets, existing_index)
    print(
        f"[INFO] targets={plan.total} to_embed={len(plan.to_embed)} "
        f"reuse={len(plan.reuse)} dropped={len(plan.dropped)}"
    )
    update_job_progress(
        phase="plan",
        total=plan.total,
        message=f"to_embed={len(plan.to_embed)} reuse={len(plan.reuse)}",
        extra={
            "to_embed": len(plan.to_embed),
            "reuse": len(plan.reuse),
            "dropped": len(plan.dropped),
        },
    )

    prev_matrix = None
    if plan.reuse:
        max_row = max(int(t["source_row"]) for t in plan.reuse) + 1
        prev_matrix = _load_existing_matrix(out_dir, max_row)
        if prev_matrix is None:
            # 行列が読めないなら流用を諦めて全件計算に切り替える。
            print("[WARN] existing embeddings matrix is unusable; re-embedding all")
            plan.to_embed = plan.to_embed + plan.reuse
            plan.reuse = []

    if not plan.to_embed and plan.reuse and not plan.dropped:
        print("[INFO] cache is up to date; nothing to embed")
        update_job_progress(
            phase="done",
            message="embeddings already up to date",
            extra={"embeddings_dir": str(out_dir), "count": plan.total},
        )
        return

    new_matrix = np.zeros((0, 0), dtype=np.float32)
    if plan.to_embed:
        device = get_device(args.device)
        print(f"[INFO] device: {device.type}")
        requested_device = _normalize_device_name(args.device)
        if requested_device in {"cuda", "mps"} and device.type != requested_device:
            print(
                f"[WARN] requested device '{requested_device}' is not available. "
                f"fallback to '{device.type}'."
            )

        num_workers = int(args.num_workers)
        if device.type == "mps" and num_workers > 0:
            print(
                "[INFO] device=mps: overriding DataLoader num_workers -> 0 "
                "(macOS DataLoader worker overhead mitigation)"
            )
            num_workers = 0

        clip_model, clip_preprocess = clip.load(
            args.clip_model, device=device, jit=False
        )
        clip_model.eval()

        new_matrix = encode_paths(
            [t["_abs_path"] for t in plan.to_embed],
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=num_workers,
        )

    update_job_progress(phase="save", message="writing embedding cache")

    # 出力順（流用 -> 新規）と各行の取得元は embedding_cache 側で確定させる。
    output_plan = build_output_plan(plan)
    rows: List["np.ndarray"] = []
    for target in output_plan:
        if target["source"] == "reuse":
            assert prev_matrix is not None  # 流用時のみここに来る
            rows.append(prev_matrix[target["source_row"]])
        else:
            rows.append(new_matrix[target["source_row"]])

    matrix = (
        np.stack(rows, axis=0).astype(np.float32)
        if rows
        else np.zeros((0, 0), dtype=np.float32)
    )
    dim = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] > 0 else 0

    entries = build_index_entries(output_plan)

    _save_matrix(out_dir, matrix)
    payload = build_index_payload(
        model_name=args.clip_model, dim=dim, entries=entries
    )
    payload["normalized"] = True
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_index(out_dir, payload)

    print(
        f"[INFO] saved embeddings: count={len(entries)} dim={dim} dir={out_dir}"
    )
    update_job_progress(
        phase="done",
        current=len(entries),
        total=len(entries),
        message="embed_images completed",
        extra={
            "embeddings_dir": str(out_dir),
            "count": len(entries),
            "dim": dim,
            "embedded": len(plan.to_embed),
            "reused": len(plan.reuse),
            "dropped": len(plan.dropped),
        },
    )


if __name__ == "__main__":
    main()
