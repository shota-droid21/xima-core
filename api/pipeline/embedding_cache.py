"""CLIP 埋め込みキャッシュのレイアウトと差分計画。

埋め込みは `cache/` 配下の**再生成可能な派生物**として扱う（Decision 004）。
喪失しても embed_images ジョブを再実行すれば復元できる。

このモジュールは torch / numpy に依存しない。
エンコード本体（torch）と行列 I/O（numpy）は embed_images.py 側に置き、
ここには「どこに置くか」「何を再計算すべきか」だけを持たせることで、
重い依存の無い環境（CI）でも単体テストできるようにしている。

レイアウト:

    experiments/<exp>/cache/embeddings/<model_slug>/
        embeddings.npy   # float32 の [N, D] 行列（numpy。embed_images が書く）
        index.json       # 行対応と無効化キー

無効化キーは「CLIP モデル名（ディレクトリで分離）」＋「画像のコンテンツハッシュ」。
画像が差し替わればハッシュが変わり再計算され、モデルを変えれば別ディレクトリになる。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# index.json のスキーマ版。読み手が互換を判断できるようにする。
EMBEDDING_CACHE_VERSION = "1"

INDEX_FILENAME = "index.json"
MATRIX_FILENAME = "embeddings.npy"

_UNSAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(model_name: str) -> str:
    """CLIP モデル名をディレクトリ名として安全な形に変換する（例: ViT-B/32 -> ViT-B_32）。"""
    slug = _UNSAFE_SLUG_RE.sub("_", (model_name or "").strip())
    return slug or "unknown"


def embeddings_dir(cache_root: Path, model_name: str) -> Path:
    """モデルごとの埋め込みキャッシュディレクトリ。"""
    return cache_root / "embeddings" / model_slug(model_name)


def content_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """画像ファイルのコンテンツハッシュ（sha256）。

    CLIP のエンコードに比べれば読み込みコストは十分小さいため、
    mtime/size ではなく内容そのものを鍵にして取り違えを防ぐ。
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_index(dir_path: Path) -> Optional[Dict[str, Any]]:
    """index.json を読む。存在しない/壊れている場合は None（＝キャッシュ無し扱い）。"""
    index_path = dir_path / INDEX_FILENAME
    if not index_path.exists() or not index_path.is_file():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != EMBEDDING_CACHE_VERSION:
        # 版が違うキャッシュは安全側に倒して作り直す。
        return None
    return data


def save_index(dir_path: Path, payload: Dict[str, Any]) -> None:
    """index.json を原子的に書き出す（中断時に壊れた index を残さない）。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    index_path = dir_path / INDEX_FILENAME
    tmp_path = index_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, index_path)


def build_index_payload(
    *,
    model_name: str,
    dim: int,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """index.json の中身を組み立てる。entries は row 昇順で渡す。"""
    return {
        "version": EMBEDDING_CACHE_VERSION,
        "clip_model_name": model_name,
        "dim": int(dim),
        "count": len(entries),
        "items": entries,
    }


@dataclass
class EmbedPlan:
    """再計算が必要なもの / 既存キャッシュを流用できるもの / 消えたもの。"""

    to_embed: List[Dict[str, Any]] = field(default_factory=list)
    reuse: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.to_embed) + len(self.reuse)


def plan_embeddings(
    targets: List[Dict[str, Any]],
    existing_index: Optional[Dict[str, Any]],
) -> EmbedPlan:
    """今回埋め込むべき対象を、既存キャッシュとの差分から決める。

    targets: `{"file_id", "path", "content_hash"}` の列（今回の対象画像）
    existing_index: `load_index()` の戻り（None ならキャッシュ無し）

    - file_id が既存にあり content_hash も一致 → reuse（既存 row を流用）
    - 一致しない / 存在しない → to_embed
    - 既存にあって targets に無い file_id → dropped（index から落とす）
    """
    existing_items: Dict[str, Dict[str, Any]] = {}
    if existing_index:
        for item in existing_index.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            fid = item.get("file_id")
            if isinstance(fid, str):
                existing_items[fid] = item

    plan = EmbedPlan()
    seen: set[str] = set()

    for target in targets:
        fid = target.get("file_id")
        if not isinstance(fid, str):
            continue
        seen.add(fid)

        prev = existing_items.get(fid)
        prev_row = prev.get("row") if prev else None
        if (
            prev is not None
            and prev.get("content_hash") == target.get("content_hash")
            and isinstance(prev_row, int)
            and prev_row >= 0
        ):
            # 既存の行をそのまま使えるので、元の行番号を控えておく。
            plan.reuse.append({**target, "source_row": prev_row})
        else:
            plan.to_embed.append(dict(target))

    plan.dropped = [fid for fid in existing_items if fid not in seen]
    return plan


def build_output_plan(plan: EmbedPlan) -> List[Dict[str, Any]]:
    """出力行列の並び（流用分 -> 新規計算分）と、各行の取得元を確定する。

    行列と index.json のズレが最も事故りやすいため、
    「最終 row」「どこから値を持ってくるか」をここで一意に決めておく。

    各要素は `source` が
      - "reuse": 既存行列の `source_row` 行をコピーする
      - "new"  : 新規計算した行列の `source_row` 行を使う
    を意味する。
    """
    output: List[Dict[str, Any]] = []
    for target in plan.reuse:
        output.append(
            {
                **target,
                "row": len(output),
                "source": "reuse",
                "source_row": int(target["source_row"]),
            }
        )
    for i, target in enumerate(plan.to_embed):
        output.append(
            {
                **target,
                "row": len(output),
                "source": "new",
                "source_row": i,
            }
        )
    return output


def build_index_entries(output_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """index.json に載せる最小フィールドだけを抜き出す（内部用キーは落とす）。"""
    return [
        {
            "file_id": t["file_id"],
            "path": t["path"],
            "content_hash": t["content_hash"],
            "row": t["row"],
        }
        for t in output_plan
    ]
