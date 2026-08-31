#!/usr/bin/env python3
"""agent/pipeline/infer_heads.py

NOTE:
- Ported from 02_train/infer_heads.py for agent pipeline usage.
- The original under 02_train/ is NOT modified.

Key differences vs 02_train:
- Default dataset/eval dirs are derived from --run-dir (agent workflow).
  Expected:
    <workspaces>/<ws>/experiments/<exp>/models/run_xxx
  Then:
    DATASET_DIR = <...>/dataset
    EVAL_DIR    = <...>/eval
- Schema is optional; absence does not fail the run.
- Emits lightweight job progress updates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

import clip

from device import _normalize_device_name, get_device
from image_io import load_image_rgb
from job_progress import update_job_progress
from label_schema import canonical_head_type, get_heads, load_schema

# ----------------------------
# Paths (defaults; overridden at runtime)
# ----------------------------

THIS_FILE = Path(__file__).resolve()
TRAIN_ROOT = THIS_FILE.parent  # agent/pipeline/
PROJECT_ROOT = TRAIN_ROOT.parent

DATASET_DIR = TRAIN_ROOT / "dataset"
MODELS_DIR = TRAIN_ROOT / "models"
EVAL_DIR = TRAIN_ROOT / "eval"


def load_index(index_path: Path) -> Dict[str, Any]:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"invalid index.json: {index_path}")
    return data


class IndexInferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: List[Path],
        indices: List[int],
        *,
        clip_preprocess,
    ) -> None:
        self.paths = paths
        self.indices = indices
        self.clip_preprocess = clip_preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        idx = self.indices[i]
        path = self.paths[i]
        img = load_image_rgb(path)
        img_tensor = self.clip_preprocess(img)
        return img_tensor, idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP + 学習済み Linear heads で dataset/index.json を multi-head 推論 (agent)"
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="学習済み head の .pt が格納されたディレクトリ (agent: experiments/<exp>/models/run_xxx)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="使用する index.json のパス (省略時は experiments/<exp>/dataset/index.json を推定)",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="推論に使用する head をカンマ区切りで指定 (省略時は run-dir 内の全 head)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="推論対象 split (train, val, all)。カンマ区切りで指定も可",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="使用する CLIP モデル名。省略時は checkpoint の clip_model_name を優先",
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
        "--output",
        type=str,
        default=None,
        help="スコアを書き出す JSON ファイルパス (省略時は experiments/<exp>/eval/scores_<run_dir名>.json)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="label_schema.json のパス (省略時は見つかれば利用。無くても続行)",
    )

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting infer_heads")

    start_dt = datetime.now()
    print(f"[INFO] Infer start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    device = get_device(args.device)
    print(f"[INFO] device: {device.type}")
    requested_device = _normalize_device_name(args.device)
    if requested_device in {"cuda", "mps"} and device.type != requested_device:
        print(
            f"[WARN] requested device '{requested_device}' is not available. "
            f"fallback to '{device.type}'."
        )

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(
            f"[ERROR] run-dir が存在しないかディレクトリではありません: {run_dir}"
        )

    # Infer experiment dirs for agent workflow.
    global DATASET_DIR, MODELS_DIR, EVAL_DIR
    try:
        models_dir = run_dir.parent
        exp_dir = models_dir.parent
        MODELS_DIR = models_dir
        DATASET_DIR = exp_dir / "dataset"
        EVAL_DIR = exp_dir / "eval"
    except Exception:
        pass

    ckpt_paths = sorted(run_dir.glob("*.pt"))
    if not ckpt_paths:
        raise SystemExit(
            f"[ERROR] run-dir 内に .pt ファイルが見つかりません: {run_dir}"
        )

    head_to_ckpt: Dict[str, Dict[str, Any]] = {}
    head_to_classes: Dict[str, List[str]] = {}
    head_to_type: Dict[str, str] = {}

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        head = ckpt.get("class_head")
        head_type = canonical_head_type(ckpt.get("head_type"))
        classes = ckpt.get("classes") if isinstance(ckpt.get("classes"), list) else []
        if not classes:
            class_to_idx = ckpt.get("class_to_idx") or {}
            inv = sorted(class_to_idx.items(), key=lambda kv: kv[1])
            classes = [name for name, _ in inv]
        if not head or not classes:
            print(
                f"[WARN] checkpoint に class_head / classes がありません (skip): {ckpt_path}"
            )
            continue
        head_to_ckpt[head] = ckpt
        head_to_classes[head] = classes
        head_to_type[head] = head_type if head_type in ("multi_class", "multi_label") else "multi_class"
        print(
            f"[INFO] found head '{head}' in {ckpt_path.name} (type={head_to_type[head]}, num_classes={len(classes)})"
        )

    if not head_to_ckpt:
        raise SystemExit(
            f"[ERROR] 有効な checkpoint (.pt) が見つかりませんでした: {run_dir}"
        )

    # schema (optional)
    schema = None
    schema_heads: List[str] = []
    schema_path: Optional[Path] = None
    if args.schema:
        schema_path = Path(args.schema).resolve()
    else:
        inferred_schema = run_dir.parent.parent / "label_input" / "label_schema.json"
        if inferred_schema.exists():
            schema_path = inferred_schema.resolve()
    if schema_path and schema_path.exists():
        try:
            schema = load_schema(schema_path)
            schema_heads = [
                str(h.get("id"))
                for h in get_heads(schema)
                if isinstance(h, dict)
                and str(h.get("id") or "").strip()
                and canonical_head_type(h.get("type")) != "split"
            ]
        except Exception as exc:
            if args.schema:
                raise SystemExit(f"[ERROR] failed to load schema: {exc}")
            schema = None
            schema_heads = []

    if args.heads:
        requested = [h.strip() for h in args.heads.split(",") if h.strip()]
        heads = [h for h in requested if h in head_to_ckpt]
        missing = [h for h in requested if h not in head_to_ckpt]
        for h in missing:
            print(
                f"[WARN] requested head '{h}' は run-dir の checkpoint に存在しません (skip)"
            )
        if not heads:
            raise SystemExit(
                "[ERROR] 指定された heads がどれも run-dir の checkpoint に存在しません。"
            )
    elif schema_heads:
        heads = [h for h in schema_heads if h in head_to_ckpt]
        if not heads:
            heads = sorted(head_to_ckpt.keys())
    else:
        heads = sorted(head_to_ckpt.keys())

    print(f"[INFO] heads to infer: {heads}")

    # index.json
    if args.index:
        index_path = Path(args.index).resolve()
    else:
        some_ckpt = next(iter(head_to_ckpt.values()))
        ckpt_index = some_ckpt.get("index_path")
        if ckpt_index:
            index_path = Path(ckpt_index).resolve()
        else:
            index_path = DATASET_DIR / "index.json"

    index_data = load_index(index_path)
    items: List[Dict[str, Any]] = index_data["items"]

    print(f"[INFO] index.json: {index_path}")
    print(f"[INFO] num items (index): {len(items)}")

    split_arg = args.splits.strip().lower()
    if split_arg == "all":
        split_filter = None
    else:
        split_filter = {s.strip() for s in split_arg.split(",") if s.strip()}

    # Determine CLIP model name
    first_ckpt = next(iter(head_to_ckpt.values()))
    ckpt_clip_name = first_ckpt.get("clip_model_name")
    clip_name = args.clip_model or ckpt_clip_name or "ViT-B/32"
    print(f"[INFO] clip model: {clip_name}")

    clip_model, clip_preprocess = clip.load(clip_name, device=device, jit=False)
    clip_model.eval()
    clip_model.float()

    # Build inference target list
    target_indices: List[int] = []
    target_paths: List[Path] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if split_filter is not None:
            sp = str(it.get("split", ""))
            if sp not in split_filter:
                continue
        dp = it.get("dataset_path")
        if not dp:
            continue
        p = (DATASET_DIR / str(dp)).resolve()
        if not p.exists():
            continue
        target_indices.append(idx)
        target_paths.append(p)

    print(f"[INFO] num targets: {len(target_indices)}")

    ds = IndexInferDataset(
        target_paths, target_indices, clip_preprocess=clip_preprocess
    )
    effective_num_workers = 2
    if device.type == "mps":
        effective_num_workers = 0
        print(
            "[INFO] device=mps: overriding infer DataLoader num_workers 2 -> 0 "
            "(macOS DataLoader worker overhead mitigation)"
        )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Precompute image features
    feats_by_index: Dict[int, torch.Tensor] = {}
    total_batches = max(len(loader), 1)
    with torch.no_grad():
        for bi, (xb, idxs) in enumerate(loader):
            xb = xb.to(device)
            feats = clip_model.encode_image(xb).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.detach().cpu()
            for j, idx in enumerate(idxs.tolist()):
                feats_by_index[int(idx)] = feats[j]

            if (bi + 1) % 10 == 0 or (bi + 1) == total_batches:
                update_job_progress(
                    phase="embed",
                    current=bi + 1,
                    total=total_batches,
                    message="encoding images",
                )

    # Run heads
    out_items: List[Dict[str, Any]] = []
    for idx in target_indices:
        out_items.append({"id": str(items[idx].get("id", idx))})

    # map id -> out item
    id_to_out: Dict[int, Dict[str, Any]] = {
        ti: out_items[i] for i, ti in enumerate(target_indices)
    }

    for hi, head in enumerate(heads):
        ckpt = head_to_ckpt[head]
        classes = head_to_classes[head]
        head_type = head_to_type.get(head, "multi_class")
        state = ckpt.get("state_dict") or ckpt.get("head_state")
        if not state:
            print(f"[WARN] no state_dict in ckpt for head={head} (skip)")
            continue

        # Compatibility: train_epoch saves LinearHead with an inner `fc` layer,
        # so keys become `fc.weight` / `fc.bias`. Here we run inference with
        # `torch.nn.Linear`, which expects `weight` / `bias`.
        if (
            isinstance(state, dict)
            and "weight" not in state
            and any(str(k).endswith("fc.weight") for k in state.keys())
        ):
            w_key = next(k for k in state.keys() if str(k).endswith("fc.weight"))
            b_key = next((k for k in state.keys() if str(k).endswith("fc.bias")), None)
            normalized = {"weight": state[w_key]}
            if b_key is not None:
                normalized["bias"] = state[b_key]
            state = normalized

        # infer dim
        # state dict keys: fc.weight shape [C, D]
        w = None
        for k, v in state.items():
            if k.endswith("weight"):
                w = v
                break
        if w is None:
            print(f"[WARN] cannot infer weight for head={head} (skip)")
            continue
        num_classes = int(w.shape[0])
        in_dim = int(w.shape[1])

        head_model = torch.nn.Linear(in_dim, num_classes)
        head_model.load_state_dict(state)
        head_model.to(device)
        head_model.eval()
        # 学習時に val で当てはめた温度。無い checkpoint は 1.0（従来どおり）。
        # argmax は変わらないため、既存の scores を読む側の解釈は壊れない。
        temperature = float(ckpt.get("temperature") or 1.0)
        if temperature <= 0:
            temperature = 1.0
        if temperature != 1.0:
            print(f"[INFO] head '{head}': 温度 T={temperature:.4f} を適用します")

        with torch.no_grad():
            for idx in target_indices:
                feat = feats_by_index.get(idx)
                if feat is None:
                    continue
                logits = (
                    head_model(feat.to(device)).detach().cpu().float() / temperature
                )
                if head_type == "multi_label":
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=-1)
                scores = {
                    classes[i]: float(probs[i].item())
                    for i in range(min(len(classes), probs.numel()))
                }
                id_to_out[idx][head] = scores

        update_job_progress(
            phase="infer",
            current=hi + 1,
            total=len(heads),
            message=f"inferred {head}",
            extra={"head_type": head_type},
        )

    # output
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EVAL_DIR / f"scores_{run_dir.name}.json"

    out = {
        "items": out_items,
        "meta": {
            "run_dir": str(run_dir),
            "index_path": str(index_path),
            "schema_path": str(schema_path) if schema_path else None,
            "clip_model_name": clip_name,
            "heads": heads,
            "head_types": {h: head_to_type.get(h, "multi_class") for h in heads},
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[INFO] wrote: {output_path}")
    update_job_progress(
        phase="done",
        message="infer_heads completed",
        extra={"output": str(output_path)},
    )


if __name__ == "__main__":
    main()
