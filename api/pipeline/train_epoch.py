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

import clip

from device import _normalize_device_name, get_device
from feature_cache import (
    FEATURE_MEMORY_WARN_BYTES,
    build_targets,
    encode_features,
    feature_loader,
    feature_memory_bytes,
    usable_items,
)
from job_progress import update_job_progress
from label_schema import (
    canonical_head_type,
    get_head_classes,
    get_heads,
    load_schema,
    normalize_label_for_head,
)
from run_meta import (
    RUN_META_VERSION,
    build_head_metrics,
    diagnose_training,
    write_run_meta,
)

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


# 温度スケーリングの下限サンプル数。
#
# T は val 集合 1 つで推定するスカラーなので、val が小さいと過剰適合して
# かえって歪む。少なすぎるときは校正しない（T=1.0 = 従来どおり）方が安全。
MIN_VAL_FOR_TEMPERATURE = 20

# 温度の許容範囲。**主な安全装置は NLL の検証**（fit_temperature 参照）で、
# ここは最適化が明らかに発散した値を早めに落とすための粗い枠でしかない。
#
# 下限を 0.05 のような「常識的」な値にはしない。CLIP の linear probe は重みが小さく、
# 実データで character が T=0.039 を必要とした（それで NLL 2.138 -> 0.538）。
# 妥当な当てはめまで切り落とすと校正そのものが効かなくなる。
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 100.0


@torch.no_grad()
def _collect_val_logits(
    *,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """val 全体の logits / 正解 / 有効マスクを集める（温度の当てはめに使う）。"""
    head.eval()
    all_logits, all_y, all_mask = [], [], []
    for xb, yb, valid_mask in loader:
        all_logits.append(head(xb.to(device)).detach().cpu())
        all_y.append(yb.detach().cpu())
        all_mask.append((valid_mask > 0.5).detach().cpu())
    if not all_logits:
        empty = torch.empty(0)
        return empty, empty, empty
    return torch.cat(all_logits), torch.cat(all_y), torch.cat(all_mask)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    head_type: str,
) -> Optional[float]:
    """val 上で NLL を最小にする温度 T を 1 つ求める（Guo et al. 2017）。

    なぜ要るか:
        CLIP 埋め込みは L2 正規化された単位ベクトルなので、logits の大きさは head の
        重みの大きさだけで決まる。linear probe は分類の**向き**は学ぶが、確率を
        意味のある値にするほど重みを大きくしない。結果、正解率が 93% あっても
        softmax の最大値が 0.12 にしかならず、**確率を閾値にした判断が成立しない**。

        T は logits を割るだけのスカラーなので **argmax を変えない**。
        つまり正解率は 1 ミリも動かさずに確率だけを校正できる。

        実測（内部データ・val 64〜88 件）。head を 2 つ挙げる:
            head A  確信度 0.116 -> 0.784   ECE 0.816 -> 0.148
            head B  確信度 0.200 -> 0.604   ECE 0.472 -> 0.090
        いずれも argmax は完全に不変だった。

    返り値が None のときは校正しない（呼び出し側は T=1.0 として扱う）。
    """
    if logits.numel() == 0 or logits.shape[0] < MIN_VAL_FOR_TEMPERATURE:
        return None
    if head_type == "multi_label":
        loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        y = targets.float()
    else:
        loss_fn = nn.CrossEntropyLoss()
        y = targets.long()

    with torch.no_grad():
        baseline_nll = float(loss_fn(logits, y).item())

    log_t = torch.zeros(1, requires_grad=True)
    # line search を付けないと LBFGS が overshoot して log_t が -inf 方向へ飛ぶ。
    # 内部データのある head で T=2.3e-08 に落ち、NLL が 0.490 から **118,819** へ悪化した。
    optimizer = optim.LBFGS(
        [log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), y)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:  # noqa: BLE001 — 校正に失敗しても学習結果は捨てない
        print(f"[WARN] 温度の当てはめに失敗しました（校正なしで続行）: {exc}")
        return None

    temperature = float(log_t.exp().item())
    if not math.isfinite(temperature) or temperature <= 0:
        return None
    temperature = min(max(temperature, MIN_TEMPERATURE), MAX_TEMPERATURE)

    # **要の検証。**温度は val の NLL を最小化して求めるものなので、
    # 出発点（T=1.0）より悪い値が返ってきたら、それは当てはめの失敗である。
    #
    # ここを見ていなかったために、確率が 0 か 1 に振り切れた checkpoint が出荷された。
    # そうなると閾値ごとの実測一致率が全帯で同じ値に潰れ、
    # 「0.99 以上」を選んでも実際は 54.5% しか当たらない状態を利用者に見せてしまう。
    # 校正できないことより、**校正したつもりで壊れている**方が危険である。
    with torch.no_grad():
        fitted_nll = float(loss_fn(logits / temperature, y).item())
    if not math.isfinite(fitted_nll) or fitted_nll >= baseline_nll:
        print(
            f"[WARN] 温度 T={temperature:.6f} は NLL を改善しませんでした"
            f"（{baseline_nll:.4f} -> {fitted_nll:.4f}）。校正なしで続行します"
        )
        return None
    return temperature


@torch.no_grad()
def eval_one_epoch(
    *,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    head_type: str,
    loss_fn: nn.Module,
) -> Tuple[float, float, Optional[float]]:
    """loader は **符号化済みの特徴**を出す（`feature_cache.feature_loader`）。

    CLIP はここには現れない。特徴は run の最初に 1 度だけ計算される。

    返り値は (loss, acc, exact_match)。

    `acc` は multi_label では **per-element** である（クラス枠の数で割る）。
    この値だけを見てはいけない。内部データのある head は 11 クラスで 1 画像あたり
    平均 1.50 個が正なので、**「1 つも付けない」と答えるだけで 0.864** になる。
    そこで multi_label では `exact_match`（集合が完全に一致した画像の割合）も返す。
    同じ状態の exact_match は 0.000 であり、こちらが実態を表す。

    一括確定が `labels` へ書くのは集合そのものなので、「見ずに確定してよいか」を
    予測するのは exact_match の方である（`prediction_reliability.is_agreement` も
    集合の完全一致で測っている）。multi_class では acc と同義なので None を返す。
    """
    head.eval()

    total_units = 0
    correct_units = 0
    total_rows = 0
    exact_rows = 0
    loss_sum = 0.0
    eval_steps = 0

    for xb, yb, valid_mask in loader:
        valid_mask = valid_mask.to(device) > 0.5
        logits = head(xb.to(device))

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
            # 画像単位。1 クラスでも外していればその画像は不一致。
            total_rows += int(yb_valid.shape[0])
            exact_rows += int((pred == yb_valid).all(dim=-1).sum().item())
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
    exact = None
    if head_type == "multi_label":
        exact = (exact_rows / total_rows) if total_rows > 0 else 0.0
    return avg_loss, acc, exact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP + Linear Head で dataset/index.json を用いて学習 (multi-head)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        # 15 から引き上げた。特徴を run 内で 1 度しか符号化しなくなり
        # （feature_cache）、1 エポックが 67 秒から 0.02 秒になったため、
        # 100 エポックでも学習部分は数秒で終わる。実データでは 60 エポック回しても
        # val_loss がまだ下がっていた（character 2.977 -> 0.554）。
        default=100,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        # 0（無効）から 10 へ。判定を val_loss に変えたので、これは
        # 「途中の停滞で切る」ためではなく **過学習し始めたら止める**ための網である。
        # val_acc 基準のときに使われていた 4〜5 は、val_loss がまだ急降下している
        # 最中に切ってしまい実害が出ていた。
        default=10,
        help=(
            "val_loss がこのエポック数連続で改善しなければ学習を打ち切る。"
            " 0 の場合は early stopping を無効にする。"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        # 既定を 1e-4 から引き上げた。CLIP 埋め込みは L2 正規化された単位ベクトルで、
        # logits の大きさは head の重みの大きさだけで決まる。1e-4 では 15 エポック回しても
        # 重みが初期値からほとんど動かず、loss が chance 水準に張り付いたまま終わる。
        #
        # 内部データ（埋め込み ViT-L/14@336px）で実測した val_acc:
        #   head A  1e-4: 0.557 -> 1e-3: 0.943    head B  1e-4: 0.456 -> 1e-3: 0.956
        # 平均最大確率も 0.04 前後から意味のある水準へ動く。**精度そのものが上がる。**
        #
        # 1e-2 はさらに速いが、train_acc が 1.0 に張り付き小規模データで過学習しやすい。
        # 既定としては 1e-3 を採り、足りなければ利用者が上げる。
        default=1e-3,
    )
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
            "カンマ区切りで複数 head id を指定 (例: 'shape,color')。"
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
        # この run が実際に使った学習設定。UI は「前回の run」の隣にこれを出す。
        # 記録が無いと、画面がフォームの現在値を前回の設定として見せてしまう。
        "train_args": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
        },
    }
    write_run_meta(run_dir, run_meta)

    # ------------------------------------------------------------------
    # CLIP の特徴は **ここで 1 度だけ** 計算する。
    #
    # backbone は凍結されていて preprocess にランダム要素も無いため、
    # 毎エポック・毎 head で符号化し直しても結果は同じである
    # （詳細は feature_cache の docstring）。head は単位ベクトルの上の Linear なので、
    # ここで作った行をそのまま全 head・全エポックで使い回せる。
    # ------------------------------------------------------------------
    update_job_progress(phase="encode", message="encoding images (once per run)")

    # 読めない画像は**符号化の前に**外す。途中で落とすと特徴の行とラベルの行がずれ、
    # 間違った対応で学習しても気づけない（feature_cache.usable_items）。
    train_items_all, dropped_train = usable_items(train_items_all, dataset_root=DATASET_DIR)
    val_items_all, dropped_val = usable_items(val_items_all, dataset_root=DATASET_DIR)
    dropped = dropped_train + dropped_val
    if dropped:
        print(f"[WARN] 読めない画像を {len(dropped)} 件除外しました:")
        for name in dropped[:10]:
            print(f"         {name}")
        if len(dropped) > 10:
            print(f"         ... 他 {len(dropped) - 10} 件")

    encode_kwargs = dict(
        dataset_root=DATASET_DIR,
        clip_preprocess=clip_preprocess,
        clip_model=clip_model,
        device=device,
        batch_size=effective_batch_size,
        num_workers=effective_num_workers,
        pin_memory=(device.type == "cuda"),
    )
    encode_started = time.time()
    train_feats = encode_features(
        train_items_all,
        on_progress=lambda done, total: update_job_progress(
            phase="encode", current=done, total=total, message="encoding train images"
        ),
        **encode_kwargs,
    )
    val_feats = encode_features(
        val_items_all,
        on_progress=lambda done, total: update_job_progress(
            phase="encode", current=done, total=total, message="encoding val images"
        ),
        **encode_kwargs,
    )
    if train_feats.numel() == 0 and val_feats.numel() == 0:
        raise SystemExit(
            "[ERROR] 符号化できる画像が 1 枚もありません。"
            "index.json の dataset_path と dataset/ の中身を確認してください"
        )
    in_dim = int((train_feats if train_feats.numel() else val_feats).shape[1])
    used_bytes = feature_memory_bytes(len(train_items_all), len(val_items_all), in_dim)
    print(
        f"[INFO] 特徴を符号化しました: train={len(train_items_all)} val={len(val_items_all)} "
        f"dim={in_dim} ({used_bytes / 1024 / 1024:.1f} MB, {time.time() - encode_started:.1f} 秒)"
    )
    print("[INFO] 以降のエポックはこの特徴を使い回すため、画像は読み直しません")
    if used_bytes > FEATURE_MEMORY_WARN_BYTES:
        print(
            f"[WARN] 特徴がメモリ上で {used_bytes / 1024 / 1024 / 1024:.1f} GB を占めます。"
            "スワップが起きる場合は dataset を分割してください"
        )

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

        # 特徴は共通なので、head ごとに変わるのは**正解と有効マスクだけ**。
        class_to_idx = {c: i for i, c in enumerate(classes)}
        target_kwargs = dict(
            class_head=head,
            head_type=head_type,
            head_schema=head_schema,
            class_to_idx=class_to_idx,
        )
        train_y, train_valid = build_targets(train_items_all, **target_kwargs)
        val_y, val_valid = build_targets(val_items_all, **target_kwargs)
        train_loader = feature_loader(
            train_feats, train_y, train_valid,
            batch_size=effective_batch_size, shuffle=True,
        )
        val_loader = feature_loader(
            val_feats, val_y, val_valid,
            batch_size=effective_batch_size, shuffle=False,
        )

        num_classes = len(classes)
        if num_classes <= 0:
            raise SystemExit(f"[ERROR] no classes found for head={head}")

        print(f"[INFO] TRAIN_DIR: {TRAIN_DIR}")
        print(f"[INFO] VAL_DIR: {VAL_DIR}")
        print(f"[INFO] head_type: {head_type}")
        print(f"[INFO] num classes: {num_classes}")

        head_model = LinearHead(in_dim=in_dim, num_classes=num_classes).to(device)

        # Loss
        if head_type == "multi_label":
            if args.no_class_weight:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                pos_weight = compute_multilabel_pos_weights(
                    train_items_all,
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
                    train_items_all,
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

        best_val_loss = math.inf
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
                valid_mask = valid_mask.to(device) > 0.5
                # xb は既に符号化済みの特徴（run の最初に 1 度だけ計算した行）。
                logits = head_model(xb.to(device))
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
            val_loss, val_acc, val_exact = eval_one_epoch(
                head=head_model,
                loader=val_loader,
                device=device,
                head_type=head_type,
                loss_fn=loss_fn,
            )

            # multi_label は per-element の acc だけ出すと実態より良く見えるので、
            # 集合の完全一致も並べる（eval_one_epoch の docstring）。
            exact_note = f" val_exact={val_exact:.4f}" if val_exact is not None else ""
            print(
                f"[EPOCH] head={head} epoch={epoch+1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}{exact_note}"
            )

            record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
            if val_exact is not None:
                record["val_exact_match"] = float(val_exact)
            epoch_history.append(record)

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
                    "val_exact_match": val_exact,
                },
            )

            # **val_loss で選ぶ（val_acc ではない）。**
            #
            # val_acc は階段関数である。val 192 件なら 1 枚 = 0.52% 刻みで、
            # ノイズで数エポック横ばいに見えるのは普通だが、その間 val_loss は
            # 下がり続けている。実データでは patience=5 で
            #   character  8 epoch で停止し epoch 3 を採用（acc 0.812 / ≥0.90 帯 96 件）
            # となり、最後まで回せば acc 0.896 / ≥0.90 帯 146 件に届いていた。
            #
            # multi_label では更に悪い。val_acc は per-element なので、
            # 11 クラス・1 画像あたり平均 1.50 個が正の head では
            # 「1 つも付けない」と答えるだけで 0.864 になる。その結果 val_acc 基準は
            # **epoch 1 の「何も予測しないモデル」を最良として採用**していた
            # （集合の完全一致は 0.000）。
            #
            # 加えて、確率の質を決めるのは loss である。温度校正も一括確定の実測表も
            # loss の世界の話なので、acc で選ぶと「一括確定に使う値」を acc で選ぶことになる。
            #
            # val_acc は指標として記録・表示を続ける（history / best に入っている）。
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = {
                    "head_state": {
                        k: v.detach().cpu() for k, v in head_model.state_dict().items()
                    },
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_exact_match": val_exact,
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if int(args.early_stopping_patience) > 0 and epochs_since_improve >= int(
                args.early_stopping_patience
            ):
                print(
                    f"[INFO] early stopping triggered (val_loss が "
                    f"{args.early_stopping_patience} エポック改善せず) for head={head}"
                )
                break

        # save checkpoint
        ckpt_path = run_dir / f"{head}_linear.pt"
        state_to_save = (
            best_state["head_state"] if best_state else head_model.state_dict()
        )

        # **温度を val で当てはめてから保存する。**argmax を変えないので正解率は
        # 動かず、確率だけが意味を持つようになる（fit_temperature の docstring 参照）。
        # ベスト checkpoint に対して測るため、保存する重みを載せ直してから行う。
        head_model.load_state_dict(state_to_save)
        temperature = None
        if best_state is None:
            # 学習が 1 エポックも回っていない（--epochs 0 など）。校正する対象が無い。
            print(f"[INFO] {head}: 学習が行われていないため温度校正はしません")
        else:
            # **ここで失敗しても学習結果は捨てない。**checkpoint は既に手元にあり、
            # 校正は「あると良い」もの。val 画像が読めない等で全部を失うのは割に合わない。
            try:
                val_logits, val_y, val_mask = _collect_val_logits(
                    head=head_model, loader=val_loader, device=device
                )
                if val_logits.numel() and bool(val_mask.any().item()):
                    temperature = fit_temperature(
                        val_logits[val_mask], val_y[val_mask], head_type=head_type
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {head}: 温度校正をスキップします（{exc}）")
                temperature = None
        if temperature is not None:
            print(f"[INFO] {head}: 温度 T={temperature:.4f} を保存します（argmax は不変）")
        elif best_state is not None:
            # 学習は回ったが校正できなかった場合だけ理由を出す。
            # 未学習（--epochs 0）のときは上で既に出している。
            print(
                f"[INFO] {head}: 温度校正なし"
                f"（val が {MIN_VAL_FOR_TEMPERATURE} 件未満、または当てはめに失敗）"
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
                # 推論時に logits をこれで割る。無い checkpoint は 1.0 として扱う。
                "temperature": temperature,
            },
            ckpt_path,
        )
        print(f"[INFO] saved: {ckpt_path}")

        # ベスト（early stopping で選ばれた checkpoint）と推移を記録する。
        heads_metrics[head] = build_head_metrics(head_type, epoch_history, best_state)

        # **学習が成立したかを判定して言葉で出す。**
        # val_acc だけでは「多数派クラスを当てているだけ」と区別できない。
        # loss が chance 水準に張り付いたまま出荷された事例があるため、
        # 表示するだけでなく判定まで実装側で行う。
        problems = diagnose_training(
            head_type=head_type,
            num_classes=len(classes),
            epoch_history=epoch_history,
            best_val_loss=(best_state or {}).get("val_loss"),
            best_val_acc=(best_state or {}).get("val_acc"),
            best_exact_match=(best_state or {}).get("val_exact_match"),
        )
        for problem in problems:
            print(f"[WARN] {head}: {problem}")
        if problems:
            heads_metrics[head]["problems"] = problems
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
