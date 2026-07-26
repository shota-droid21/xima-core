#!/usr/bin/env python3
"""agent/pipeline/train_epoch.py

NOTE:
- Ported from the legacy training script for agent pipeline usage.
- The original under legacy/ is NOT modified.

Key differences vs 02_train:
- Default dataset/model dirs are derived from --index when provided.
  In agent workflow, --index should point to:
    <workspaces>/<ws>/experiments/<exp>/dataset/index.json
  Then:
    DATASET_DIR = <...>/dataset
    MODELS_DIR  = <...>/models
- Emits job progress updates (head/epoch/batch) for real-time UI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import clip

from device import _normalize_device_name, get_device
from image_io import load_image_rgb
from job_progress import update_job_progress
from label_schema import (
    canonical_head_type,
    get_head_classes,
    get_heads,
    load_schema,
    normalize_label_for_head,
)
from run_meta import RUN_META_VERSION, build_head_metrics, write_run_meta

# ----------------------------
# Paths (defaults; overridden at runtime when --index is provided)
# ----------------------------

THIS_FILE = Path(__file__).resolve()
TRAIN_ROOT = THIS_FILE.parent  # agent/pipeline/
PROJECT_ROOT = TRAIN_ROOT.parent  # agent/

DATASET_DIR = TRAIN_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODELS_DIR = TRAIN_ROOT / "models"


# ----------------------------
# Utils
# ----------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_index(index_path: Path) -> Dict[str, Any]:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"invalid index.json: {index_path}")
    return data


@dataclass
class ItemRec:
    id: str
    dataset_path: str
    split: str
    labels: Dict[str, Any]


class IndexDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[ItemRec],
        *,
        dataset_root: Path,
        class_head: str,
        head_type: str,
        head_schema: Optional[Dict[str, Any]],
        clip_preprocess,
        class_to_idx: Dict[str, int],
    ) -> None:
        self.items = items
        self.dataset_root = dataset_root
        self.class_head = class_head
        self.head_type = head_type
        self.head_schema = head_schema or {"type": head_type}
        self.clip_preprocess = clip_preprocess
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[i]
        path = (self.dataset_root / item.dataset_path).resolve()
        img = load_image_rgb(path)
        img_tensor = self.clip_preprocess(img)

        y_raw = (item.labels or {}).get(self.class_head)
        if self.head_type == "multi_label":
            y_norm = normalize_label_for_head(y_raw, self.head_schema)
            y = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
            if y_norm is None:
                return img_tensor, y, torch.tensor(0.0, dtype=torch.float32)

            labels = y_norm if isinstance(y_norm, list) else [str(y_norm)]
            for label in labels:
                idx = self.class_to_idx.get(str(label))
                if idx is not None:
                    y[idx] = 1.0
            return img_tensor, y, torch.tensor(1.0, dtype=torch.float32)

        y_norm = normalize_label_for_head(y_raw, self.head_schema)
        idx = self.class_to_idx.get(str(y_norm), -1) if y_norm is not None else -1
        valid = 1.0 if idx >= 0 else 0.0
        return (
            img_tensor,
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(valid, dtype=torch.float32),
        )


def _collect_items(index_data: Dict[str, Any], split: str) -> List[ItemRec]:
    out: List[ItemRec] = []
    for it in index_data.get("items", []) or []:
        if not isinstance(it, dict):
            continue
        if split != "all" and str(it.get("split")) != split:
            continue
        labels = it.get("labels") if isinstance(it.get("labels"), dict) else {}
        out.append(
            ItemRec(
                id=str(it.get("id")),
                dataset_path=str(it.get("dataset_path")),
                split=str(it.get("split")),
                labels=labels,
            )
        )
    return out


def build_dataloaders(
    *,
    index_data: Dict[str, Any],
    class_head: str,
    head_type: str,
    head_schema: Optional[Dict[str, Any]],
    clip_preprocess,
    batch_size: int,
    num_workers: int,
    dataset_root: Path,
    class_names: List[str],
    pin_memory: bool,
) -> Tuple[
    IndexDataset,
    torch.utils.data.DataLoader,
    IndexDataset,
    torch.utils.data.DataLoader,
    Dict[str, int],
]:
    train_items = _collect_items(index_data, "train")
    val_items = _collect_items(index_data, "val")

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_ds = IndexDataset(
        train_items,
        dataset_root=dataset_root,
        class_head=class_head,
        head_type=head_type,
        head_schema=head_schema,
        clip_preprocess=clip_preprocess,
        class_to_idx=class_to_idx,
    )
    val_ds = IndexDataset(
        val_items,
        dataset_root=dataset_root,
        class_head=class_head,
        head_type=head_type,
        head_schema=head_schema,
        clip_preprocess=clip_preprocess,
        class_to_idx=class_to_idx,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_ds, train_loader, val_ds, val_loader, class_to_idx


def collect_classes_from_items(
    items: List[ItemRec], class_head: str, head_type: str
) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    fallback_head = {"type": head_type}
    for it in items:
        v = normalize_label_for_head((it.labels or {}).get(class_head), fallback_head)
        if v is None:
            continue
        if head_type == "multi_label":
            values = v if isinstance(v, list) else [str(v)]
            for vv in values:
                s = str(vv).strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return sorted(out)


def compute_class_weights(
    train_items: List[ItemRec],
    class_head: str,
    class_to_idx: Dict[str, int],
    head_schema: Optional[Dict[str, Any]],
) -> torch.Tensor:
    counts = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for it in train_items:
        v = normalize_label_for_head((it.labels or {}).get(class_head), head_schema)
        if v is None:
            continue
        idx = class_to_idx.get(str(v))
        if idx is None:
            continue
        counts[idx] += 1.0

    # inverse-frequency weights
    weights = torch.ones_like(counts)
    for i in range(len(counts)):
        if counts[i] > 0:
            weights[i] = 1.0 / counts[i]

    # normalize
    if weights.sum().item() > 0:
        weights = weights * (len(weights) / weights.sum())
    return weights


def compute_multilabel_pos_weights(
    train_items: List[ItemRec],
    class_head: str,
    class_to_idx: Dict[str, int],
    head_schema: Optional[Dict[str, Any]],
) -> torch.Tensor:
    pos = torch.zeros(len(class_to_idx), dtype=torch.float32)
    valid_samples = 0
    for it in train_items:
        labels = normalize_label_for_head((it.labels or {}).get(class_head), head_schema)
        if labels is None:
            continue
        valid_samples += 1
        values = labels if isinstance(labels, list) else [str(labels)]
        for v in values:
            idx = class_to_idx.get(str(v))
            if idx is None:
                continue
            pos[idx] += 1.0

    if valid_samples <= 0:
        return torch.ones(len(class_to_idx), dtype=torch.float32)

    neg = torch.full_like(pos, float(valid_samples)) - pos
    weights = torch.ones_like(pos)
    for i in range(len(pos)):
        if pos[i] > 0 and neg[i] > 0:
            weights[i] = neg[i] / pos[i]
    return weights


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def eval_one_epoch(
    *,
    model: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    head_type: str,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    head.eval()

    total_units = 0
    correct_units = 0
    loss_sum = 0.0
    eval_steps = 0

    for xb, yb, valid_mask in loader:
        xb = xb.to(device)
        valid_mask = valid_mask.to(device) > 0.5
        feats = model.encode_image(xb).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = head(feats)

        if not bool(valid_mask.any().item()):
            continue

        if head_type == "multi_label":
            yb = yb.to(device).float()
            logits_valid = logits[valid_mask]
            yb_valid = yb[valid_mask]
            loss = loss_fn(logits_valid, yb_valid)
            probs = torch.sigmoid(logits_valid)
            pred = (probs >= 0.5).float()
            total_units += int(yb_valid.numel())
            correct_units += int((pred == yb_valid).sum().item())
        else:
            yb = yb.to(device).long()
            logits_valid = logits[valid_mask]
            yb_valid = yb[valid_mask]
            loss = loss_fn(logits_valid, yb_valid)
            pred = logits_valid.argmax(dim=-1)
            total_units += int(yb_valid.numel())
            correct_units += int((pred == yb_valid).sum().item())

        loss_sum += float(loss.item())
        eval_steps += 1

    avg_loss = loss_sum / max(eval_steps, 1)
    acc = (correct_units / total_units) if total_units > 0 else 0.0
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP + Linear Head で dataset/index.json を用いて学習 (multi-head)"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "val_acc がこのエポック数連続で改善しなければ学習を打ち切る。"
            " 0 の場合は early stopping を無効にする。"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L/14@336px",
        help=("使用する CLIP モデル名。例: 'ViT-L/14@336px', 'ViT-B/32' など。"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "強制デバイス指定 (cpu / cuda / metal[mps]). "
            "省略時は自動判定（cuda -> mps -> cpu）"
        ),
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="クラス重み付けを行わない (均等重み)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="使用する index.json のパス (agent workflow: experiments/<exp>/dataset/index.json)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="label_schema.json のパス (省略時は index.json から推定)",
    )
    parser.add_argument(
        "--class-head",
        type=str,
        default=None,
        help="学習に使用する head id (デフォルト: index.json の meta.class_head か 'character')",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help=(
            "カンマ区切りで複数 head id を指定 (例: 'character,hair_color')。"
            " 指定時は --class-head より優先され、列挙された head を順番に学習します。"
        ),
    )

    args = parser.parse_args()

    global DATASET_DIR, TRAIN_DIR, VAL_DIR, MODELS_DIR

    update_job_progress(phase="start", message="starting train_epoch")

    # Seed for reproducibility (best-effort)
    set_seed(42)

    # Resolve index path and override dataset/model dirs for agent workflow.
    index_path = (
        Path(args.index).resolve() if args.index else (DATASET_DIR / "index.json")
    )
    try:
        dataset_dir = index_path.parent
        # infer experiment dir: <exp>/dataset/index.json
        exp_dir = dataset_dir.parent
        DATASET_DIR = dataset_dir
        TRAIN_DIR = dataset_dir / "train"
        VAL_DIR = dataset_dir / "val"
        MODELS_DIR = exp_dir / "models"
    except Exception:
        pass

    start_dt = datetime.now()
    print(f"[INFO] Training start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    device = get_device(args.device)
    print(f"[INFO] device: {device.type}")
    requested_device = _normalize_device_name(args.device)
    if requested_device in {"cuda", "mps"} and device.type != requested_device:
        print(
            f"[WARN] requested device '{requested_device}' is not available. "
            f"fallback to '{device.type}'."
        )

    index_data = load_index(index_path)
    train_items_all = _collect_items(index_data, "train")
    val_items_all = _collect_items(index_data, "val")

    effective_batch_size = max(1, int(args.batch_size))
    batches_per_epoch = max(1, math.ceil(len(train_items_all) / effective_batch_size))
    print(
        "[INFO] dataset stats: "
        f"train_items={len(train_items_all)} val_items={len(val_items_all)} "
        f"batch_size={effective_batch_size} batches_per_epoch={batches_per_epoch}"
    )
    if len(train_items_all) > 0 and batches_per_epoch == 1:
        print(
            "[WARN] train split is smaller than batch_size, so each epoch has only 1 step. "
            "This is not specific to metal/mps."
        )

    effective_num_workers = max(0, int(args.num_workers))
    if device.type == "mps" and effective_num_workers != 0:
        print(
            f"[INFO] device=mps: overriding num_workers {effective_num_workers} -> 0 "
            "(macOS DataLoader worker overhead mitigation)"
        )
        effective_num_workers = 0

    meta = index_data.get("meta", {}) or {}
    meta_default_head = meta.get("class_head", "character")

    schema_path: Optional[Path] = None
    if args.schema:
        schema_path = Path(args.schema).resolve()
    else:
        inferred_schema = index_path.parent.parent / "label_input" / "label_schema.json"
        if inferred_schema.exists():
            schema_path = inferred_schema.resolve()

    schema: Optional[Dict[str, Any]] = None
    schema_head_map: Dict[str, Dict[str, Any]] = {}
    if schema_path and schema_path.exists():
        try:
            schema = load_schema(schema_path)
            for h in get_heads(schema):
                if not isinstance(h, dict):
                    continue
                hid = str(h.get("id") or "").strip()
                if not hid:
                    continue
                schema_head_map[hid] = h
            print(f"[INFO] schema: {schema_path}")
        except Exception as exc:
            if args.schema:
                raise SystemExit(f"[ERROR] failed to load schema: {exc}")
            print(f"[WARN] failed to load schema ({schema_path}): {exc}")
            schema = None
            schema_head_map = {}

    if args.heads:
        heads_to_train = [h.strip() for h in args.heads.split(",") if h.strip()]
    elif args.class_head:
        heads_to_train = [args.class_head]
    elif schema_head_map:
        heads_to_train = [
            hid
            for hid, h in schema_head_map.items()
            if canonical_head_type(h.get("type")) != "split"
        ]
        if not heads_to_train:
            heads_to_train = [meta_default_head]
    else:
        heads_to_train = [meta_default_head]

    print(f"[INFO] index.json: {index_path}")
    print(f"[INFO] heads to train: {heads_to_train}")

    clip_model, clip_preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] model run dir: {run_dir}")

    # Save minimal run meta（学習後に metrics / finished_at を追記して確定させる）
    run_meta: Dict[str, Any] = {
        "run_meta_version": RUN_META_VERSION,
        "started_at": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "index_path": str(index_path),
        "schema_path": str(schema_path) if schema_path else None,
        "clip_model_name": args.clip_model,
        "heads": heads_to_train,
    }
    write_run_meta(run_dir, run_meta)

    # head ごとの学習結果（ベスト精度とエポック推移）を蓄積する。
    heads_metrics: Dict[str, Any] = {}

    n_heads_total = max(len(heads_to_train), 1)

    for head_i, head in enumerate(heads_to_train, start=1):
        print("========================================")
        print(f"[INFO] START TRAIN HEAD: {head}")
        print("========================================")

        update_job_progress(
            phase="start_head",
            current=head_i,
            total=n_heads_total,
            message=f"starting head: {head}",
            extra={"head": head, "head_index": head_i, "heads_total": n_heads_total},
        )

        head_schema = schema_head_map.get(head)
        head_type = canonical_head_type((head_schema or {}).get("type"))
        if head_type == "split":
            raise SystemExit(f"[ERROR] split head is not trainable: {head}")

        classes = get_head_classes(head_schema)
        if head_type == "multi_label" and not classes:
            raise SystemExit(
                f"[ERROR] multi_label head requires schema.classes: {head}"
            )
        if not classes:
            classes = collect_classes_from_items(train_items_all, head, head_type)

        train_ds, train_loader, val_ds, val_loader, class_to_idx = build_dataloaders(
            index_data=index_data,
            class_head=head,
            head_type=head_type,
            head_schema=head_schema,
            clip_preprocess=clip_preprocess,
            batch_size=args.batch_size,
            num_workers=effective_num_workers,
            dataset_root=DATASET_DIR,
            class_names=classes,
            pin_memory=(device.type == "cuda"),
        )

        num_classes = len(classes)
        if num_classes <= 0:
            raise SystemExit(f"[ERROR] no classes found for head={head}")

        print(f"[INFO] TRAIN_DIR: {TRAIN_DIR}")
        print(f"[INFO] VAL_DIR: {VAL_DIR}")
        print(f"[INFO] head_type: {head_type}")
        print(f"[INFO] num classes: {num_classes}")

        # Determine embedding dim
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=device)
            try:
                feat = clip_model.encode_image(dummy).float()
                in_dim = int(feat.shape[-1])
            except Exception:
                in_dim = 768

        head_model = LinearHead(in_dim=in_dim, num_classes=num_classes).to(device)

        # Loss
        if head_type == "multi_label":
            if args.no_class_weight:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                pos_weight = compute_multilabel_pos_weights(
                    train_ds.items,
                    head,
                    class_to_idx,
                    head_schema,
                ).to(device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if args.no_class_weight:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            else:
                weights = compute_class_weights(
                    train_ds.items,
                    head,
                    class_to_idx,
                    head_schema,
                ).to(device)
                loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

        optimizer = optim.AdamW(
            head_model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        best_acc = -1.0
        best_state: Optional[Dict[str, Any]] = None
        epochs_since_improve = 0
        # このヘッドのエポック推移（UI の精度表示・推移グラフ用）。
        epoch_history: list[Dict[str, Any]] = []

        epochs_total = int(args.epochs)
        for epoch in range(epochs_total):
            clip_model.eval()
            head_model.train()

            update_job_progress(
                phase="train",
                current=epoch + 1,
                total=epochs_total,
                message=f"training {head} (epoch {epoch+1}/{epochs_total})",
                extra={"head": head, "epoch": epoch + 1, "epochs": epochs_total, "stage": "train"},
            )

            loss_sum = 0.0
            train_steps = 0
            total_batches = max(len(train_loader), 1)
            # Aim ~20 progress updates per epoch, but keep at least every batch if tiny.
            progress_every = max(1, total_batches // 20)
            last_progress_ts = 0.0

            for batch_i, (xb, yb, valid_mask) in enumerate(train_loader, start=1):
                xb = xb.to(device)
                valid_mask = valid_mask.to(device) > 0.5

                with torch.no_grad():
                    feats = clip_model.encode_image(xb).float()
                    feats = feats / feats.norm(dim=-1, keepdim=True)

                logits = head_model(feats)
                if not bool(valid_mask.any().item()):
                    continue

                if head_type == "multi_label":
                    yb = yb.to(device).float()
                    loss = loss_fn(logits[valid_mask], yb[valid_mask])
                else:
                    yb = yb.to(device).long()
                    loss = loss_fn(logits[valid_mask], yb[valid_mask])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())
                train_steps += 1

                # Real-time progress updates during long epochs.
                # Throttle by both batch count and time to avoid excessive writes.
                now = time.time()
                if (batch_i % progress_every == 0) or (now - last_progress_ts >= 2.0) or (
                    batch_i == total_batches
                ):
                    last_progress_ts = now
                    train_loss_avg = loss_sum / max(train_steps, 1)
                    update_job_progress(
                        phase="train",
                        current=batch_i,
                        total=total_batches,
                        message=(
                            f"training {head} | epoch {epoch+1}/{epochs_total} "
                            f"batch {batch_i}/{total_batches}"
                        ),
                        extra={
                            "head": head,
                            "head_type": head_type,
                            "epoch": epoch + 1,
                            "epochs": epochs_total,
                            "batch": batch_i,
                            "batches": total_batches,
                            "train_loss": float(loss.item()),
                            "train_loss_avg": float(train_loss_avg),
                        },
                    )

            if train_steps <= 0:
                raise SystemExit(
                    f"[ERROR] no valid labeled samples found for head={head}"
                )
            train_loss = loss_sum / max(train_steps, 1)

            update_job_progress(
                phase="val",
                message=f"validating {head} (epoch {epoch+1}/{epochs_total})",
                extra={"head": head, "epoch": epoch + 1, "epochs": epochs_total, "stage": "val"},
            )
            val_loss, val_acc = eval_one_epoch(
                model=clip_model,
                head=head_model,
                loader=val_loader,
                device=device,
                head_type=head_type,
                loss_fn=loss_fn,
            )

            print(
                f"[EPOCH] head={head} epoch={epoch+1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

            # basic progress update
            update_job_progress(
                phase="epoch_done",
                current=epoch + 1,
                total=epochs_total,
                message=f"epoch done: {head}",
                extra={
                    "head": head,
                    "head_type": head_type,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
            )

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                best_state = {
                    "head_state": {
                        k: v.detach().cpu() for k, v in head_model.state_dict().items()
                    },
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if int(args.early_stopping_patience) > 0 and epochs_since_improve >= int(
                args.early_stopping_patience
            ):
                print(
                    f"[INFO] early stopping triggered (patience={args.early_stopping_patience}) for head={head}"
                )
                break

        # save checkpoint
        ckpt_path = run_dir / f"{head}_linear.pt"
        state_to_save = (
            best_state["head_state"] if best_state else head_model.state_dict()
        )
        torch.save(
            {
                "class_head": head,
                "head_type": head_type,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "clip_model_name": args.clip_model,
                "index_path": str(index_path),
                "schema_path": str(schema_path) if schema_path else None,
                "state_dict": state_to_save,
            },
            ckpt_path,
        )
        print(f"[INFO] saved: {ckpt_path}")

        # ベスト（early stopping で選ばれた checkpoint）と推移を記録する。
        heads_metrics[head] = build_head_metrics(head_type, epoch_history, best_state)
        # 途中でジョブが失敗しても直近の成果が残るよう、head 完了ごとに更新する。
        run_meta["metrics"] = {"heads": heads_metrics}
        write_run_meta(run_dir, run_meta)

        update_job_progress(
            phase="saved",
            current=head_i,
            total=n_heads_total,
            message=f"saved head: {head}",
            extra={"head": head, "ckpt_path": str(ckpt_path)},
        )

    run_meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_meta["metrics"] = {"heads": heads_metrics}
    write_run_meta(run_dir, run_meta)

    update_job_progress(
        phase="done",
        message="train_epoch completed",
        extra={"models_dir": str(MODELS_DIR)},
    )


if __name__ == "__main__":
    main()
