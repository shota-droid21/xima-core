#!/usr/bin/env python3
"""agent/pipeline/label_schema.py

NOTE:
- This file is copied from 02_train/label_schema.py and adapted for agent/ migration.
- Do not edit the original in 02_train during the migration phase.

The default schema path is resolved by searching upwards for:
    legacy/10_tools/labeling/label_schema.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for base in [start, *start.parents]:
        candidate = base / "legacy" / "10_tools" / "labeling" / "label_schema.json"
        if candidate.exists():
            return base
    # Fallback: repo layout expects agent/ under repo root
    return start.parents[1]


PROJECT_ROOT = _find_project_root(THIS_DIR)
DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "legacy" / "10_tools" / "labeling" / "label_schema.json"
)

CANONICAL_HEAD_TYPES = {"split", "multi_class", "multi_label"}
LEGACY_HEAD_TYPE_ALIASES = {"single_class": "multi_class"}
HEAD_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


def canonical_head_type(raw: Any) -> str:
    t = str(raw or "").strip()
    if not t:
        return "multi_class"
    t = LEGACY_HEAD_TYPE_ALIASES.get(t, t)
    if t in CANONICAL_HEAD_TYPES:
        return t
    return t


def _normalize_class_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
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


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    heads = schema.get("heads")
    if not isinstance(heads, list):
        return schema

    out = dict(schema)
    out_heads: List[Any] = []
    for head in heads:
        if not isinstance(head, dict):
            out_heads.append(head)
            continue
        h = dict(head)
        h_type = canonical_head_type(h.get("type"))
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


def validate_schema(schema: Dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise ValueError("label schema must be an object")

    heads = schema.get("heads")
    if not isinstance(heads, list):
        raise ValueError("label schema heads must be a list")

    seen: set[str] = set()
    for idx, head in enumerate(heads):
        if not isinstance(head, dict):
            raise ValueError(f"heads[{idx}] must be an object")

        raw_id = str(head.get("id") or "")
        head_id = raw_id.strip()
        if not head_id:
            raise ValueError(f"heads[{idx}].id is required")
        if raw_id != head_id:
            raise ValueError(f"heads[{idx}].id must not contain spaces")
        if not HEAD_ID_RE.fullmatch(head_id):
            raise ValueError(
                f"heads[{idx}].id must match ^[A-Za-z0-9_][A-Za-z0-9_-]*$"
            )
        if head_id in seen:
            raise ValueError(f"duplicate head id: {head_id}")
        seen.add(head_id)

        head_type = canonical_head_type(head.get("type"))
        if head_type not in CANONICAL_HEAD_TYPES:
            raise ValueError(
                f"heads[{idx}] type must be one of split,multi_class,multi_label"
            )
        if head_type == "multi_label":
            classes = get_head_classes(head)
            if not classes:
                raise ValueError(
                    f"heads[{idx}] ({head_id}) multi_label requires non-empty classes"
                )


def load_schema(path: Optional[Path | str] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_SCHEMA_PATH
    if not p.exists():
        raise FileNotFoundError(f"label schema not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        normalized = normalize_schema(json.load(f))
    validate_schema(normalized)
    return normalized


def get_heads(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    return schema.get("heads", []) if schema else []


def get_head(schema: Dict[str, Any], head_id: str) -> Optional[Dict[str, Any]]:
    if not schema:
        return None
    for h in get_heads(schema):
        if h.get("id") == head_id:
            return h
    return None


def get_head_classes(head: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(head, dict):
        return []
    classes = _normalize_class_list(head.get("classes"))
    if classes:
        return classes
    return _normalize_class_list(head.get("choices"))


def _normalize_scalar_label_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, list):
        for elem in val:
            if elem is None:
                continue
            s = str(elem).strip()
            if s:
                return s
        return None
    return str(val)


def _normalize_multi_label_value(val: Any, classes: List[str]) -> Optional[List[str]]:
    if val is None:
        return None

    raw_values: List[str] = []
    if isinstance(val, list):
        raw_values = [str(v).strip() for v in val if v is not None and str(v).strip()]
    elif isinstance(val, tuple):
        raw_values = [str(v).strip() for v in val if v is not None and str(v).strip()]
    elif isinstance(val, str):
        if "," in val:
            raw_values = [p.strip() for p in val.split(",") if p.strip()]
        else:
            s = val.strip()
            raw_values = [s] if s else []
    else:
        s = str(val).strip()
        raw_values = [s] if s else []

    if not classes:
        return []

    selected = {v for v in raw_values if v in classes}
    return [c for c in classes if c in selected]


def normalize_label_for_head(val: Any, head: Optional[Dict[str, Any]]) -> Any:
    h_type = canonical_head_type((head or {}).get("type"))
    if h_type == "multi_label":
        return _normalize_multi_label_value(val, get_head_classes(head))
    if h_type in ("split", "multi_class"):
        return _normalize_scalar_label_value(val)
    return val


def normalize_label_value(
    val: Any, head_id: Optional[str] = None, schema: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    head = get_head(schema, head_id) if schema and head_id else None
    out = normalize_label_for_head(val, head)
    if out is None:
        return None
    if isinstance(out, list):
        return ",".join(out)
    return str(out)
