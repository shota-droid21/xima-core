#!/usr/bin/env python3
"""agent/pipeline/make_label_list.py

NOTE:
- Based on 02_train/make_label_list.py and adapted for agent pipeline usage.
- Adds job progress reporting and thumbnail pre-generation for cache/thumbs/w{width}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

from job_progress import update_job_progress

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
DEFAULT_THUMB_WIDTH = 256


def _apply_mode_for_file(path: Path, st_mode: int) -> int:
    # Image files should never need executability; normalize to 644.
    if path.suffix.lower() in IMAGE_EXTS:
        return 0o644
    # Preserve executability if already executable by any.
    if st_mode & 0o111:
        return 0o755
    return 0o644


def _fix_perms_recursive(root: Path, *, uid: int, gid: int) -> Dict[str, int]:
    """Recursively chown/chmod under root.

    Policy:
    - dirs:  755
    - files: 644 (or 755 if executable)
    - symlinks: skipped
    """

    if not root.exists():
        return {"targets": 0, "dirs": 0, "files": 0, "skipped": 0, "errors": 0}

    errors = 0
    dirs = 0
    files = 0
    skipped = 0

    def _handle_path(p: Path) -> None:
        nonlocal errors, dirs, files, skipped
        try:
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                skipped += 1
                return

            if stat.S_ISDIR(st.st_mode):
                os.chown(p, uid, gid)
                os.chmod(p, 0o755)
                dirs += 1
                return

            if stat.S_ISREG(st.st_mode):
                os.chown(p, uid, gid)
                os.chmod(p, _apply_mode_for_file(p, st.st_mode))
                files += 1
                return

            # other (fifo/socket/device)
            skipped += 1
        except Exception as exc:
            errors += 1
            print(f"[WARN] fix-perms failed: {p} ({exc})")

    # Ensure root itself is fixed first.
    _handle_path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        _handle_path(dp)
        for d in dirnames:
            _handle_path(dp / d)
        for f in filenames:
            _handle_path(dp / f)

    return {
        "targets": 1,
        "dirs": dirs,
        "files": files,
        "skipped": skipped,
        "errors": errors,
    }


def fix_workspace_permissions(
    *, out_path: Path, uid: int, gid: int
) -> Dict[str, Dict[str, int]]:
    """Fix permissions for the two major roots:
    - <ws>/source_images
    - <ws>/experiments/<exp>
    """

    inferred = infer_ws_exp_dirs_from_out_path(out_path)
    if not inferred:
        raise RuntimeError(
            f"cannot infer workspace/experiment from output path: {out_path}"
        )

    ws_dir, exp_dir = inferred
    results: Dict[str, Dict[str, int]] = {}
    results["source_images"] = _fix_perms_recursive(
        ws_dir / "source_images", uid=uid, gid=gid
    )
    results["experiment"] = _fix_perms_recursive(exp_dir, uid=uid, gid=gid)
    return results


def infer_ws_exp_dirs_from_out_path(out_path: Path) -> tuple[Path, Path] | None:
    """Infer workspace/experiment dirs from output path.

    Expected layout:
        <workspaces_root>/<ws>/experiments/<exp>/label_input/labels.json
    Returns:
        (ws_dir, exp_dir)
    """

    try:
        # labels.json -> label_input -> <exp> -> experiments -> <ws>
        exp_dir = out_path.parent.parent
        experiments_dir = exp_dir.parent
        ws_dir = experiments_dir.parent
        if experiments_dir.name != "experiments":
            return None
        return ws_dir, exp_dir
    except Exception:
        return None


def cache_root_for(out_path: Path) -> Path:
    # label_input/labels.json -> experiments/<exp>/cache/thumbs
    return out_path.parent.parent / "cache" / "thumbs"


def cache_key(raw: str) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_path(cache_root: Path, width: int, key: str) -> Path:
    safe_key = cache_key(key)
    shard = safe_key[:2]
    return cache_root / f"w{width}" / shard / f"{safe_key}.webp"


def collect_images(input_dir: Path, rel_base_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    input_dir = input_dir.resolve()
    rel_base_dir = rel_base_dir.resolve()

    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        rel_path = os.path.relpath(p, rel_base_dir)
        file_id = Path(rel_path).stem

        created_at = None
        try:
            st = p.stat()
            bt = getattr(st, "st_birthtime", None)
            ts = bt if bt and bt > 0 else st.st_mtime
            created_at = (
                datetime.utcfromtimestamp(ts)
                .replace(tzinfo=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except Exception:
            created_at = None

        items.append(
            {
                "id": len(items),
                "file_id": file_id,
                "path": rel_path,
                "label": None,
                "split": None,
                "created_at": created_at,
            }
        )
    return items


def load_existing(out_path: Path) -> Tuple[Dict, Dict[str, Dict]]:
    if not out_path.exists():
        return {}, {}

    try:
        # Empty file (e.g., interrupted previous run) should be treated as no existing data.
        try:
            if out_path.stat().st_size == 0:
                return {}, {}
        except Exception:
            pass

        with out_path.open("r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return {}, {}

        data = json.loads(raw)
        if not isinstance(data, dict):
            print(f"[WARN] existing labels.json is not an object; ignoring: {out_path}")
            return {}, {}
    except json.JSONDecodeError as e:
        # Preserve the broken file for debugging, then continue as fresh.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad_path = out_path.with_suffix(out_path.suffix + f".invalid.{ts}")
        try:
            out_path.replace(bad_path)
            print(
                f"[WARN] existing labels.json is invalid JSON; moved to {bad_path} and regenerating. "
                f"(error: {e})"
            )
        except Exception:
            print(
                f"[WARN] existing labels.json is invalid JSON; ignoring and regenerating: {out_path}. "
                f"(error: {e})"
            )
        return {}, {}

    existing_meta = data.get("meta", {}) or {}
    existing_items = data.get("items", []) or []

    existing_by_file_id: Dict[str, Dict] = {}
    for item in existing_items:
        fid = item.get("file_id")
        if not fid:
            path = item.get("path")
            if path:
                fid = Path(path).stem
                item["file_id"] = fid
        if not fid:
            fid = f"_legacy_{item.get('id')}"
            item["file_id"] = fid
        existing_by_file_id[fid] = item

    return existing_meta, existing_by_file_id


def generate_thumbnail(src: Path, dst: Path, width: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((width, width), Image.Resampling.LANCZOS)
            im.save(tmp, format="WEBP", quality=80, optimize=True)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def generate_all_thumbs(
    items: List[Dict], *, html_dir: Path, cache_root: Path, width: int
) -> Dict[str, int]:
    total = len(items)
    done = 0
    skipped = 0
    missing = 0
    failed = 0

    for idx, item in enumerate(items):
        if idx % 50 == 0:
            update_job_progress(
                phase="thumbs",
                current=idx,
                total=total,
                message="generating thumbnails",
                extra={
                    "done": done,
                    "skipped": skipped,
                    "missing": missing,
                    "failed": failed,
                },
            )

        rel_path = item.get("path")
        if not rel_path:
            missing += 1
            continue
        fid = item.get("file_id") or Path(rel_path).stem
        src = (html_dir / rel_path).resolve()
        if not src.exists() or not src.is_file():
            missing += 1
            continue

        dst = cache_path(cache_root, width, fid)
        if dst.exists():
            skipped += 1
            continue
        try:
            generate_thumbnail(src, dst, width)
            done += 1
        except Exception:
            failed += 1
            continue

    update_job_progress(
        phase="thumbs",
        current=total,
        total=total,
        message="thumbnails generated",
        extra={
            "done": done,
            "skipped": skipped,
            "missing": missing,
            "failed": failed,
        },
    )
    return {"generated": done, "skipped": skipped, "missing": missing, "failed": failed}


def embed_thumb_paths(
    items: List[Dict], *, out_path: Path, cache_root: Path, width: int
) -> Dict[str, int]:
    """Embed nginx-served thumb URL path into each item.

    We store:
      item['thumb_path'] = '/static/<path-under-workspaces-root>'

    Nginx config should map:
      /static/<path> -> /data/workspaces/<path>
    """

    inferred = infer_ws_exp_dirs_from_out_path(out_path)
    if not inferred:
        print(
            "[WARN] cannot infer workspace/experiment from output path; "
            "thumb_path will not be embedded"
        )
        return {"embedded": 0, "missing_thumb": 0, "skipped": len(items)}

    ws_dir, _exp_dir = inferred

    embedded = 0
    missing_thumb = 0
    for item in items:
        rel_path = item.get("path")
        if not rel_path:
            continue
        fid = item.get("file_id") or Path(rel_path).stem
        if not fid:
            continue

        dst = cache_path(cache_root, width, fid)
        # Convert physical path to URL under /static/ (relative to workspaces root).
        # We don't need to know the absolute workspaces root; we prefix <ws>/...
        try:
            rel_under_ws = dst.relative_to(ws_dir)
        except Exception:
            # If the directory layout is unexpected, skip embedding.
            continue

        rel_under_workspaces = Path(ws_dir.name) / rel_under_ws
        item["thumb_path"] = "/static/" + rel_under_workspaces.as_posix()
        embedded += 1

        if not dst.exists():
            missing_thumb += 1

    if missing_thumb > 0:
        print(f"[WARN] missing thumbnail files: {missing_thumb}")
    return {"embedded": embedded, "missing_thumb": missing_thumb, "skipped": 0}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_html_dir = script_dir / "label_tool"

    parser = argparse.ArgumentParser(
        description=(
            "指定ディレクトリ配下の画像一覧を JSON に出力し、サムネイルも生成（ラベリングツール用）"
        )
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="ラベリング対象の画像ディレクトリ（例: downloads/raw_batch_001）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="label_input.json",
        help="出力する JSON パス（デフォルト: label_input.json）",
    )
    parser.add_argument(
        "--remove-labels",
        type=str,
        default=None,
        help=(
            "既存の items に含まれる labels マップから削除するキーをカンマ区切りで指定。"
            " 例: --remove-labels vibe,style"
        ),
    )
    parser.add_argument(
        "--html-dir",
        "-H",
        default=str(default_html_dir),
        help=(
            "index.html が置かれているディレクトリ。画像パスはここからの相対パスとして保存される"
            "（デフォルト: スクリプトと同階層の label_tool）"
        ),
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=DEFAULT_THUMB_WIDTH,
        help="生成するサムネイル幅 (デフォルト: 256)",
    )
    parser.add_argument(
        "--skip-thumbs",
        action="store_true",
        help="サムネイル生成をスキップする",
    )

    parser.add_argument(
        "--no-fix-perms",
        action="store_true",
        help=(
            "(非推奨) source_images と experiment 配下の chmod/chown をスキップする。"
            " デフォルトは毎回実行して配信権限を揃える。"
        ),
    )
    parser.add_argument(
        "--fix-perms-uid",
        type=int,
        default=int(os.environ.get("XIMA_PERMS_UID", "1000")),
        help="chmod/chown 時の uid（デフォルト: env XIMA_PERMS_UID または 1000）",
    )
    parser.add_argument(
        "--fix-perms-gid",
        type=int,
        default=int(os.environ.get("XIMA_PERMS_GID", "1000")),
        help="chmod/chown 時の gid（デフォルト: env XIMA_PERMS_GID または 1000）",
    )

    args = parser.parse_args()

    update_job_progress(phase="start", message="starting make_label_list")

    html_dir = Path(args.html_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    out_path = Path(args.output).resolve()
    thumb_width = max(int(args.thumb_width), 1)

    remove_labels = set()
    if args.remove_labels:
        remove_labels = {s.strip() for s in args.remove_labels.split(",") if s.strip()}
        if remove_labels:
            print(f"[INFO] remove_labels: {sorted(remove_labels)}")

    if not input_dir.exists():
        raise SystemExit(f"[ERROR] input-dir が存在しません: {input_dir}")

    new_items = collect_images(input_dir, html_dir)
    print(f"[INFO] 画像ファイル数 (今回スキャン分): {len(new_items)}")
    update_job_progress(
        phase="scan",
        current=0,
        total=len(new_items),
        message="scanning images",
    )

    existing_meta, existing_by_file_id = load_existing(out_path)
    new_by_file_id = {item["file_id"]: item for item in new_items}

    merged_by_file_id: Dict[str, Dict] = {}
    for fid, new_item in new_by_file_id.items():
        existing = existing_by_file_id.get(fid)
        if existing:
            merged = dict(existing)
            merged["path"] = new_item["path"]
            if not merged.get("created_at") and new_item.get("created_at"):
                merged["created_at"] = new_item.get("created_at")
        else:
            merged = new_item
        merged_by_file_id[fid] = merged

    merged_items: List[Dict] = []
    for idx, item in enumerate(
        sorted(merged_by_file_id.values(), key=lambda x: x.get("path", ""))
    ):
        item["id"] = idx
        if "label" in item:
            item.pop("label", None)
        if "split" in item:
            item.pop("split", None)

        if remove_labels and "labels" in item and isinstance(item["labels"], dict):
            for lk in list(remove_labels):
                if lk in item["labels"]:
                    item["labels"].pop(lk, None)

        merged_items.append(item)

    meta = existing_meta or {}
    meta.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "html_dir": str(html_dir),
            "input_dir": str(input_dir),
            "ingested_at": datetime.utcnow()
            .replace(tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )
    cache_root = cache_root_for(out_path)
    thumb_stats = {"generated": 0, "missing": 0, "failed": 0}
    if not args.skip_thumbs:
        thumb_stats = generate_all_thumbs(
            merged_items, html_dir=html_dir, cache_root=cache_root, width=thumb_width
        )
        print(
            f"[INFO] thumbnails: generated={thumb_stats['generated']} "
            f"skipped={thumb_stats.get('skipped', 0)} "
            f"missing_src={thumb_stats['missing']} failed={thumb_stats['failed']} "
            f"width={thumb_width} cache_root={cache_root}"
        )
    else:
        print("[INFO] thumbnail generation skipped (--skip-thumbs)")

    update_job_progress(phase="embed", message="embedding thumb_path")
    embed_stats = embed_thumb_paths(
        merged_items, out_path=out_path, cache_root=cache_root, width=thumb_width
    )

    data = {"meta": meta, "items": merged_items}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish write: write to tmp then replace.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(out_path)

    print(f"[INFO] JSON を出力しました: {out_path}")
    print(f"[INFO] items 件数 (マージ後): {len(merged_items)}")
    print(
        f"[INFO] thumb_path embedded: {embed_stats.get('embedded', 0)} "
        f"(missing_files={embed_stats.get('missing_thumb', 0)})"
    )

    update_job_progress(
        phase="done",
        message="make_label_list completed",
        extra={
            "items": len(merged_items),
            "thumbs_generated": thumb_stats.get("generated", 0),
            "thumbs_skipped": thumb_stats.get("skipped", 0),
            "thumbs_missing": thumb_stats.get("missing", 0),
            "thumbs_failed": thumb_stats.get("failed", 0),
            "thumb_path_embedded": embed_stats.get("embedded", 0),
            "thumb_path_missing_files": embed_stats.get("missing_thumb", 0),
        },
    )

    # Final step: normalize permissions so nginx (uid=1000) can serve /static reliably.
    if not args.no_fix_perms:
        update_job_progress(phase="perms", message="fixing permissions")
        try:
            perms = fix_workspace_permissions(
                out_path=out_path,
                uid=int(args.fix_perms_uid),
                gid=int(args.fix_perms_gid),
            )
            print(f"[INFO] fix-perms complete: {perms}")
        except Exception as exc:
            raise SystemExit(f"[ERROR] fix-perms failed: {exc}")
    else:
        print("[INFO] fix-perms skipped (--no-fix-perms)")


if __name__ == "__main__":
    main()
