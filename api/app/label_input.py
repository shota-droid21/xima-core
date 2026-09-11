from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from .config import ConfigManager
from .label_input_partial import count_schema_violations, normalize_tolerating_unchanged
from .utils.atomic_io import write_json_atomic, write_text_atomic
from .utils.legacy import PATH_KEYS, normalize_to_source_rel


_BACKUP_ID_RE = re.compile(r"^\d{8}_\d{6}(?:-\d+)?$")
_CANONICAL_HEAD_TYPES = {"multi_class", "multi_label"}
_HEAD_TYPE_ALIASES = {"single_class": "multi_class"}
_HEAD_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_SPLIT_HEAD_ID = "split"
#: 人が書ける `split` の値（#325 / Decision 045）。**`unassigned` は廃止した。**
#: 判定の正本は `pipeline/dataset_split.py`。process が分かれているので写している。
_SPLIT_CHOICES = ("train", "val", "exclude")
#: 旧い値の読み替え。`None` は「キーごと落とす」＝ **自動**（ラベルがあれば学習に入る）。
#: `ignore` だけ `exclude` にするのは、語が「外す」を意味するからである。自動へ寄せると
#: **外したつもりのものが黙って学習に入る**。
# NOTE: remove this mapping when legacy values are fully dropped.
_LEGACY_SPLIT_ALIASES: dict[str, str | None] = {
    "unassigned": None,
    "ignore": "exclude",
    "delete": None,
}
_DEFAULT_THUMB_WIDTH = 256


def _orphans_after_schema_change(
    cfg, workspace: str, experiment: str, new_schema: Any, old_schema: Any
) -> Dict[str, Any]:
    """スキーマ変更で行き先を失った値の件数（#330）。

    **数えられなくても保存は成功させる。** ここで落ちると、ラベルが 1 件も無い
    experiment でスキーマを保存できなくなる。
    """
    from .schema_impact import orphans

    try:
        data = load_label_json(cfg.label_input_path_for(workspace, experiment))
        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
    except (FileNotFoundError, ValueError, OSError):
        return {"heads": [], "items": 0, "entering": 0}
    return orphans(items, new_schema, old_schema=old_schema)


def label_revision(label_path: Path) -> str:
    """labels.json の版を表す文字列（#294）。

    保存の前に「読み込んだあと core 側が進んでいないか」を見るために使う。

    **更新時刻とサイズから作る。** PUT の回数を数える方式では、PUT を通らずに
    ファイルを書くジョブ（`make_label_list` / `apply_label` /
    `purge_deleted_images`）の書き換えを取りこぼす。ファイル自体から作れば、
    誰が書いても変わる。

    内容ハッシュにしないのは、数 MB のファイルを保存のたびに読み直すことになる
    ため。ここは `stat` だけで済ませる。**同じ内容で書き直しても版は変わる**が、
    保存の応答で新しい版を返すので、自分の保存を「core が進んだ」と誤検知しない。

    ファイルが無ければ空文字。まだ 1 度も書かれていない状態を表す。
    """
    try:
        stat = label_path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def load_label_json(label_path: Path) -> Dict[str, Any]:
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    try:
        return json.loads(label_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {label_path}") from exc


def _current_items(label_path: Path) -> list[Any]:
    """いまディスクにある items。読めなければ空（＝すべて「変更あり」扱い）。

    #332 で「変更が無い item は厳格に検証しない」を決めるための材料。**読めない
    ときは何も緩めない**（空を返せば、どの item も unchanged にならない）。
    """
    try:
        data = load_label_json(label_path)
    except (FileNotFoundError, ValueError, OSError):
        return []
    items = data.get("items")
    return items if isinstance(items, list) else []


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        # Accept numeric strings too.
        return int(str(value))
    except Exception:
        return None


def sort_label_input_items(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with data['items'] sorted in a stable, numeric-friendly order.

    Why:
      - Some clients persist items in key-order where id is treated as string,
        resulting in 1,10,100,... ordering.
      - The API should not depend on file order; always return a predictable order.
    """

    items = data.get("items")
    if not isinstance(items, list):
        return data

    def sort_key(it: Any) -> tuple:
        if not isinstance(it, dict):
            return (2, "")
        raw_id = it.get("id")
        n = _coerce_int(raw_id)
        if n is not None:
            # (0, numeric id, tie-breakers)
            return (
                0,
                n,
                str(it.get("file_id") or ""),
                str(it.get("rel_path") or it.get("path") or ""),
            )
        # Non-numeric ids last (1, id as string, tie-breakers)
        return (
            1,
            str(raw_id or ""),
            str(it.get("file_id") or ""),
            str(it.get("rel_path") or it.get("path") or ""),
        )

    sorted_items = sorted(items, key=sort_key)
    out = dict(data)
    out["items"] = sorted_items
    return out


def _labels_template() -> Dict[str, Any]:
    return {
        "items": [],
        "meta": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": "mvp"},
    }


def _schema_template() -> Dict[str, Any]:
    # experiments.DEFAULT_LABEL_SCHEMA と同じ識別子を使う（#251 B-1）。
    # split は予約 head なので heads は空で始まる。
    return {
        "version": 2,
        "schema_id": "default_v1",
        "heads": [],
    }


def _normalize_class_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _canonical_head_type(raw: Any) -> str:
    t = str(raw or "").strip()
    if not t:
        return "multi_class"
    t = _HEAD_TYPE_ALIASES.get(t, t)
    if t in _CANONICAL_HEAD_TYPES:
        return t
    return t


def normalize_label_schema_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    heads = payload.get("heads")
    if not isinstance(heads, list):
        return payload

    out = dict(payload)
    out_heads: list[Any] = []
    for head in heads:
        if not isinstance(head, dict):
            out_heads.append(head)
            continue
        h = dict(head)
        if str(h.get("id") or "").strip() == "split":
            continue
        h_type = _canonical_head_type(h.get("type"))
        if h_type == "split":
            continue
        h["type"] = h_type

        if h_type in ("multi_class", "multi_label"):
            classes = _normalize_class_list(h.get("classes"))
            if not classes:
                classes = _normalize_class_list(h.get("choices"))
            if classes:
                h["classes"] = classes
        out_heads.append(h)

    out["heads"] = out_heads
    return out


def validate_label_schema_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("label schema must be an object")

    heads = payload.get("heads")
    if not isinstance(heads, list):
        raise ValueError("label schema heads must be a list")

    seen: set[str] = set()
    for idx, raw_head in enumerate(heads):
        if not isinstance(raw_head, dict):
            raise ValueError(f"heads[{idx}] must be an object")

        raw_id = str(raw_head.get("id") or "")
        head_id = raw_id.strip()
        if not head_id:
            raise ValueError(f"heads[{idx}].id is required")
        if raw_id != head_id:
            raise ValueError(f"heads[{idx}].id must not contain spaces")
        if not _HEAD_ID_RE.fullmatch(head_id):
            raise ValueError(
                f"heads[{idx}].id must match ^[A-Za-z0-9_][A-Za-z0-9_-]*$"
            )
        if head_id in seen:
            raise ValueError(f"duplicate head id: {head_id}")
        seen.add(head_id)

        head_type = _canonical_head_type(raw_head.get("type"))
        if head_type not in _CANONICAL_HEAD_TYPES:
            raise ValueError(
                f"heads[{idx}] type must be one of multi_class,multi_label"
            )
        if head_type == "multi_label":
            classes = _normalize_class_list(raw_head.get("classes"))
            if not classes:
                raise ValueError(
                    f"heads[{idx}] ({head_id}) multi_label requires non-empty classes"
                )


def _normalize_scalar_label_value(
    value: Any, *, item_index: int, head_id: str
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise ValueError(f"items[{item_index}].labels.{head_id} must be a string")


def _normalize_split_label_value(
    value: Any, *, item_index: int
) -> tuple[str | None, bool]:
    normalized = _normalize_scalar_label_value(
        value, item_index=item_index, head_id=_SPLIT_HEAD_ID
    )
    if normalized is None:
        return None, False
    normalized = normalized.lower()
    if normalized in _LEGACY_SPLIT_ALIASES:
        # `None` を返すとキーごと落ちる（呼び出し側が None を書かない）。
        # それが `unassigned` の移行そのものである（Decision 045-3）。
        return _LEGACY_SPLIT_ALIASES[normalized], normalized == "delete"
    if normalized not in _SPLIT_CHOICES:
        allowed = ",".join(_SPLIT_CHOICES)
        raise ValueError(
            f"items[{item_index}].labels.{_SPLIT_HEAD_ID} must be one of {allowed}"
        )
    return normalized, False


def _normalize_delete_flag(value: Any, *, item_index: int) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    raise ValueError(f"items[{item_index}].delete must be a boolean")


def _normalize_multi_class_label_value(
    value: Any, *, classes: list[str], item_index: int, head_id: str
) -> str | None:
    normalized = _normalize_scalar_label_value(
        value, item_index=item_index, head_id=head_id
    )
    if normalized is None:
        return None
    if classes and normalized not in set(classes):
        allowed = ",".join(classes)
        raise ValueError(
            f"items[{item_index}].labels.{head_id} must be one of {allowed}"
        )
    return normalized


def _normalize_multi_label_label_value(
    value: Any, *, classes: list[str], item_index: int, head_id: str
) -> list[str] | None:
    if value is None:
        return None

    raw_values: list[str] = []
    if isinstance(value, list):
        for elem in value:
            if elem is None:
                continue
            if not isinstance(elem, (str, int, float, bool)):
                raise ValueError(
                    f"items[{item_index}].labels.{head_id} must be a string array"
                )
            s = str(elem).strip()
            if s:
                raw_values.append(s)
    elif isinstance(value, str):
        if "," in value:
            raw_values = [part.strip() for part in value.split(",") if part.strip()]
        else:
            s = value.strip()
            raw_values = [s] if s else []
    else:
        raise ValueError(
            f"items[{item_index}].labels.{head_id} must be a string array"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for label in raw_values:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)

    if not deduped:
        return None

    class_set = set(classes)
    unknown = [label for label in deduped if label not in class_set]
    if unknown:
        labels = ",".join(unknown)
        raise ValueError(
            f"items[{item_index}].labels.{head_id} contains unknown labels: {labels}"
        )

    selected = set(deduped)
    ordered = [label for label in classes if label in selected]
    return ordered if ordered else None


def normalize_label_input_payload_with_schema(
    payload: Dict[str, Any], schema: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("label_input must be an object")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("label_input.items must be a list")
    if not isinstance(schema, dict):
        raise ValueError("label schema must be an object")

    raw_heads = schema.get("heads")
    if not isinstance(raw_heads, list):
        raise ValueError("label schema heads must be a list")

    schema_heads: dict[str, dict[str, Any]] = {}
    for raw_head in raw_heads:
        if not isinstance(raw_head, dict):
            continue
        hid = str(raw_head.get("id") or "").strip()
        if not hid:
            continue
        schema_heads[hid] = raw_head

    allowed_head_ids = set(schema_heads.keys())
    allowed_head_ids.add(_SPLIT_HEAD_ID)

    out = dict(payload)
    out_items: list[dict[str, Any]] = []
    for item_index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"items[{item_index}] must be an object")
        item = dict(raw_item)

        labels_obj = item.get("labels")
        if labels_obj is None:
            labels_obj = {}
        if not isinstance(labels_obj, dict):
            raise ValueError(f"items[{item_index}].labels must be an object")

        labels: dict[str, Any] = {}
        for raw_key, value in labels_obj.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"items[{item_index}].labels key must be a string")
            hid = raw_key.strip()
            if raw_key != hid:
                raise ValueError(
                    f"items[{item_index}].labels key '{raw_key}' must not contain spaces"
                )
            labels[hid] = value

        # Compatibility: accept legacy top-level label fields on save.
        for hid in allowed_head_ids:
            if hid in labels:
                continue
            if hid in item:
                labels[hid] = item.get(hid)

        unknown_heads = sorted(
            hid for hid in labels.keys() if hid not in allowed_head_ids
        )
        if unknown_heads:
            unknown_csv = ",".join(unknown_heads)
            raise ValueError(
                f"items[{item_index}].labels contains unknown head ids: {unknown_csv}"
            )

        normalized_labels: dict[str, Any] = {}
        split_value, legacy_delete = _normalize_split_label_value(
            labels.get(_SPLIT_HEAD_ID), item_index=item_index
        )
        if split_value is not None:
            normalized_labels[_SPLIT_HEAD_ID] = split_value

        for hid, head in schema_heads.items():
            raw_value = labels.get(hid)
            if raw_value is None:
                continue

            head_type = _canonical_head_type(head.get("type"))
            classes = _normalize_class_list(head.get("classes"))
            if not classes:
                classes = _normalize_class_list(head.get("choices"))

            if head_type == "multi_label":
                normalized_value = _normalize_multi_label_label_value(
                    raw_value,
                    classes=classes,
                    item_index=item_index,
                    head_id=hid,
                )
            elif head_type == "multi_class":
                normalized_value = _normalize_multi_class_label_value(
                    raw_value,
                    classes=classes,
                    item_index=item_index,
                    head_id=hid,
                )
            else:
                continue

            if normalized_value is not None:
                normalized_labels[hid] = normalized_value

        delete_flag = _normalize_delete_flag(item.get("delete"), item_index=item_index)
        if legacy_delete:
            delete_flag = True
        if delete_flag:
            item["delete"] = True
        else:
            item.pop("delete", None)

        item["labels"] = normalized_labels
        out_items.append(item)

    out["items"] = out_items
    return out


def _migrate_legacy_current_if_needed(label_path: Path) -> None:
    if label_path.exists():
        return
    legacy_current = label_path.parent / "current.json"
    if legacy_current.exists() and legacy_current.is_file():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_current.replace(label_path)


def ensure_label_file(cfg, workspace: str, experiment: str) -> Path:
    label_path = cfg.label_input_path_for(workspace, experiment)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_current_if_needed(label_path)
    if not label_path.exists():
        write_json_atomic(label_path, _labels_template())
    (label_path.parent / "history").mkdir(parents=True, exist_ok=True)
    return label_path


def ensure_label_schema_file(cfg, workspace: str, experiment: str) -> Path:
    schema_path = cfg.label_schema_path_for(workspace, experiment)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    if not schema_path.exists():
        write_json_atomic(schema_path, _schema_template())
    return schema_path


def _history_path_for(label_path: Path, backup_id: str) -> Path:
    if not _BACKUP_ID_RE.match(backup_id):
        raise ValueError("invalid backup_id")
    history_dir = label_path.parent / "history"
    return history_dir / f"{label_path.stem}_{backup_id}.json"


def _write_history_snapshot(label_path: Path, serialized: str) -> Path:
    history_dir = label_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_id = timestamp
    backup_path = history_dir / f"{label_path.stem}_{backup_id}.json"
    if backup_path.exists():
        suffix = 1
        while True:
            backup_id = f"{timestamp}-{suffix}"
            backup_path = history_dir / f"{label_path.stem}_{backup_id}.json"
            if not backup_path.exists():
                break
            suffix += 1

    write_text_atomic(backup_path, serialized)
    return backup_path


#: 同じパッケージの他モジュールから使う口（#330 の付け替えも履歴へ残すため）。
#: **履歴を残す作法を 2 つに分けない。**
write_history_snapshot = _write_history_snapshot


def _list_history_backups(target_path: Path) -> list[Dict[str, Any]]:
    history_dir = target_path.parent / "history"
    if not history_dir.exists():
        return []

    prefix = f"{target_path.stem}_"
    backups: list[Dict[str, Any]] = []
    for p in sorted(history_dir.glob(f"{target_path.stem}_*.json"), reverse=True):
        name = p.name
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        backup_id = name[len(prefix) : -len(".json")]
        if not _BACKUP_ID_RE.match(backup_id):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        backups.append(
            {
                "id": backup_id,
                "filename": name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        )
    return backups


def normalize_label_payload(payload: Any, cfg, *, workspace: str) -> Any:
    def normalize_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        path_val = next(
            (item.get(key) for key in PATH_KEYS if isinstance(item.get(key), str)),
            None,
        )
        if path_val:
            normalized = normalize_to_source_rel(
                path_str=path_val,
                workspace_root=cfg.workspace_path(workspace).resolve(),
                source_base=cfg.source_path_for(workspace),
            )
            for key in PATH_KEYS:
                if key in item:
                    item[key] = normalized
            # Ensure downstream scripts (02_train/apply_label_mapping.py) can read `path`.
            item["rel_path"] = normalized
            item["path"] = normalized
        return item

    if isinstance(payload, list):
        return [normalize_item(elem) for elem in payload]

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            payload["items"] = [normalize_item(elem) for elem in items]
        else:
            payload = normalize_item(payload)
        return payload

    raise ValueError("label_input must be an object or array")


def thumb_path_from_file_id(
    workspace: str,
    experiment: str,
    file_id: str,
    *,
    width: int = _DEFAULT_THUMB_WIDTH,
) -> str:
    ws = str(workspace or "").strip()
    exp = str(experiment or "").strip()
    fid = str(file_id or "").strip()
    if not ws or not exp or not fid:
        raise ValueError("workspace, experiment and file_id are required")
    if width <= 0:
        raise ValueError("width must be positive")
    sha = hashlib.sha1(fid.encode("utf-8")).hexdigest()
    shard = sha[:2]
    return f"/static/{ws}/experiments/{exp}/cache/thumbs/w{width}/{shard}/{sha}.webp"


def normalize_label_thumb_paths(
    payload: Any,
    *,
    workspace: str,
    experiment: str,
    width: int = _DEFAULT_THUMB_WIDTH,
) -> Any:
    def rewrite_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        out = dict(item)
        fid = out.get("file_id")
        if isinstance(fid, str) and fid.strip():
            out["thumb_path"] = thumb_path_from_file_id(
                workspace, experiment, fid, width=width
            )
        return out

    if isinstance(payload, list):
        return [rewrite_item(elem) for elem in payload]

    if isinstance(payload, dict):
        out = dict(payload)
        items = out.get("items")
        if isinstance(items, list):
            out["items"] = [rewrite_item(elem) for elem in items]
        else:
            out = rewrite_item(out)
        return out

    return payload


def create_label_input_router(config_manager: ConfigManager) -> APIRouter:
    router = APIRouter()

    # ---- Scoped API ----

    @router.get("/workspaces/{workspace}/experiments/{experiment}/label-input")
    def get_label_input_scoped(workspace: str, experiment: str) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            _migrate_legacy_current_if_needed(label_path)
            data = load_label_json(label_path)
            return sort_label_input_items(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"labels.json not found: {cfg.label_input_path_for(workspace, experiment)}",
            )

    @router.get("/workspaces/{workspace}/experiments/{experiment}/label-schema")
    def get_label_schema(workspace: str, experiment: str) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
            normalized = normalize_label_schema_payload(load_label_json(schema_path))
            validate_label_schema_payload(normalized)
            return normalized
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="label_schema.json not found")

    @router.put("/workspaces/{workspace}/experiments/{experiment}/label-schema")
    def put_label_schema(
        workspace: str,
        experiment: str,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            if schema_path.exists() and not schema_path.is_file():
                raise ValueError("label_schema path is not a file")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        normalized_schema = normalize_label_schema_payload(payload)
        validate_label_schema_payload(normalized_schema)

        # 書き換える**前**のスキーマ。何が消えたかを言うために要る（#330）。
        previous_schema = None
        try:
            previous_schema = normalize_label_schema_payload(
                load_label_json(schema_path)
            )
        except (FileNotFoundError, ValueError):
            previous_schema = None

        serialized = json.dumps(normalized_schema, ensure_ascii=False, indent=2)
        write_text_atomic(schema_path, serialized)
        version_path = _write_history_snapshot(schema_path, serialized)
        return {
            "status": "saved",
            "path": str(schema_path),
            "version": str(version_path),
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            # **止めないが、黙らない**（#330）。行き先を失った値があれば件数を返す。
            # 画面は保存の前に `label-schema/impact` で聞くが、script から直接
            # 叩いたときにも気づけるようにしておく。
            "orphans": _orphans_after_schema_change(
                cfg, workspace, experiment, normalized_schema, previous_schema
            ),
        }

    @router.get("/workspaces/{workspace}/experiments/{experiment}/label-schema/history")
    def list_label_schema_history_scoped(
        workspace: str, experiment: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "backups": _list_history_backups(schema_path),
        }

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/label-schema/history/{backup_id}"
    )
    def get_label_schema_history_scoped(
        workspace: str, experiment: str, backup_id: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
            backup_path = _history_path_for(schema_path, backup_id)
            normalized = normalize_label_schema_payload(load_label_json(backup_path))
            validate_label_schema_payload(normalized)
            return normalized
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="backup not found")

    @router.post(
        "/workspaces/{workspace}/experiments/{experiment}/label-schema/history/{backup_id}/restore"
    )
    def restore_label_schema_history_scoped(
        workspace: str, experiment: str, backup_id: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
            backup_path = _history_path_for(schema_path, backup_id)
            normalized = normalize_label_schema_payload(load_label_json(backup_path))
            validate_label_schema_payload(normalized)
            serialized = json.dumps(normalized, ensure_ascii=False, indent=2)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="backup not found")

        write_text_atomic(schema_path, serialized)
        version_path = _write_history_snapshot(schema_path, serialized)
        return {
            "status": "restored",
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "path": str(schema_path),
            "restored_from": str(backup_path),
            "version": str(version_path),
        }

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/label-input/revision"
    )
    def get_label_revision_scoped(workspace: str, experiment: str) -> Dict[str, Any]:
        """labels.json の版だけを返す（#294）。

        保存の前に「core 側が進んでいないか」を見るための軽い口。labels.json は
        実データで数 MB あり、確認のたびに全体を取り直すのは割に合わない。

        版は **PUT だけでなく、ジョブが直接書き換えた場合も変わる**必要がある。
        `make_label_list` / `apply_label` / `purge_deleted_images` は PUT を通らず
        ファイルを書くため、PUT の回数を数える方式では取りこぼす。ファイルの
        更新時刻とサイズから作れば、誰が書いても変わる。
        """
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "revision": label_revision(label_path),
        }

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/label-input/schema-violations"
    )
    def get_schema_violations_scoped(workspace: str, experiment: str) -> Dict[str, Any]:
        """スキーマ外の値を持つ item の件数（#332）。

        **`GET /label-input` の本文には足せない。** app は取得した文書をそのまま
        PUT へ渡すので、本文に足した鍵は `labels.json` へ書き戻されてしまう。
        数えるためだけの口を分ける。

        `heads` は head ごとの件数。`items` は違反を 1 つ以上持つ item の数。
        """
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)
            schema = normalize_label_schema_payload(load_label_json(schema_path))
            validate_label_schema_payload(schema)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        heads, items = count_schema_violations(
            _current_items(label_path),
            schema,
            normalize=normalize_label_input_payload_with_schema,
        )
        return {
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "heads": heads,
            "items": items,
        }

    @router.put("/workspaces/{workspace}/experiments/{experiment}/label-input")
    def put_label_input_scoped(
        workspace: str,
        experiment: str,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            _migrate_legacy_current_if_needed(label_path)
            schema_path = ensure_label_schema_file(cfg, workspace, experiment)

            if label_path.exists() and not label_path.is_file():
                raise ValueError("labels path is not a file")

            normalized_schema = normalize_label_schema_payload(
                load_label_json(schema_path)
            )
            validate_label_schema_payload(normalized_schema)

            normalized = normalize_label_payload(payload, cfg, workspace=workspace)
            violations: Dict[str, int] = {}
            if isinstance(normalized, dict):
                normalized = sort_label_input_items(normalized)
                # **変更のある item だけ厳格に検証する（#332）。**
                # 触っていない item の違反で、無関係な保存まで止めない。
                # 残った違反は件数で返す（黙って残さない）。
                normalized, violations = normalize_tolerating_unchanged(
                    normalized,
                    normalized_schema,
                    _current_items(label_path),
                    normalize=normalize_label_input_payload_with_schema,
                )
                normalized = normalize_label_thumb_paths(
                    normalized, workspace=workspace, experiment=experiment
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Backup semantics:
        # - On PUT, snapshot the *current* labels.json to history BEFORE overwriting.
        # - If labels.json does not exist yet (initial write), snapshot the newly written content.
        backup_path = None
        if label_path.exists():
            try:
                backup_path = _write_history_snapshot(
                    label_path,
                    label_path.read_text(encoding="utf-8"),
                )
            except Exception:  # noqa: BLE001
                backup_path = None

        serialized = json.dumps(normalized, ensure_ascii=False, indent=2)
        write_text_atomic(label_path, serialized)
        if backup_path is None:
            backup_path = _write_history_snapshot(label_path, serialized)

        return {
            "status": "saved",
            "path": str(label_path),
            "backup": str(backup_path),
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            # 保存した直後の版。app はこれを控えて、次の保存前の突き合わせに使う。
            # 返さないと、自分の保存で版が動いたことを「core が進んだ」と誤検知する。
            "revision": label_revision(label_path),
            # 触っていない item に残っているスキーマ外の値（#332）。空なら {}。
            "schema_violations": violations,
        }

    @router.get("/workspaces/{workspace}/experiments/{experiment}/label-input/history")
    def list_label_history_scoped(workspace: str, experiment: str) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            _migrate_legacy_current_if_needed(label_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "backups": _list_history_backups(label_path),
        }

    @router.post("/workspaces/{workspace}/experiments/{experiment}/label-input/history")
    def snapshot_label_history_scoped(
        workspace: str, experiment: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            _migrate_legacy_current_if_needed(label_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not label_path.exists():
            raise HTTPException(status_code=404, detail="labels.json not found")
        if not label_path.is_file():
            raise HTTPException(status_code=400, detail="labels path is not a file")

        try:
            backup_path = _write_history_snapshot(
                label_path,
                label_path.read_text(encoding="utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"failed to snapshot: {exc}")

        return {
            "status": "snapshotted",
            "path": str(label_path),
            "backup": str(backup_path),
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
        }

    @router.get(
        "/workspaces/{workspace}/experiments/{experiment}/label-input/history/{backup_id}"
    )
    def get_label_history_scoped(
        workspace: str, experiment: str, backup_id: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = cfg.label_input_path_for(workspace, experiment)
            _migrate_legacy_current_if_needed(label_path)
            backup_path = _history_path_for(label_path, backup_id)
            return load_label_json(backup_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="backup not found")

    @router.post(
        "/workspaces/{workspace}/experiments/{experiment}/label-input/history/{backup_id}/restore"
    )
    def restore_label_history_scoped(
        workspace: str, experiment: str, backup_id: str
    ) -> Dict[str, Any]:
        cfg = config_manager.get_config()
        try:
            label_path = ensure_label_file(cfg, workspace, experiment)
            backup_path = _history_path_for(label_path, backup_id)
            serialized = backup_path.read_text()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="backup not found")

        current_backup = None
        if label_path.exists() and label_path.is_file():
            try:
                current_backup = _write_history_snapshot(
                    label_path, label_path.read_text()
                )
            except Exception:
                current_backup = None

        write_text_atomic(label_path, serialized)
        return {
            "status": "restored",
            "workspace": workspace.strip(),
            "experiment": experiment.strip(),
            "path": str(label_path),
            "restored_from": str(backup_path),
            "backup_before_restore": str(current_backup) if current_backup else None,
        }

    return router
