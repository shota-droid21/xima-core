"""学習する head が決まらないときは、推測せずに止める（#251 C-1 / C-2）。

以前は `meta.class_head` の既定に特定の head 名（開発者のスキーマ由来）が
入っており、その名前を持たない利用者では「存在しない head を学習しようとする」
状態になっていた。原因が分かりにくい落ち方をするので、決まらないことを
その場で理由ごと告げる形に変えた。

ここでは train_epoch の解決順そのものを写した関数を検査する。
train_epoch.py は import に torch を要するため、単体で読める形にしている。
"""

from __future__ import annotations

import pytest


def resolve_heads(
    arg_heads: str | None,
    arg_class_head: str | None,
    schema_head_ids: list[str],
    meta_class_head: str | None,
) -> list[str]:
    """train_epoch.py の解決順と同じ順で head を決める。

    1. --heads / 2. --class-head / 3. スキーマの split 以外 / 4. meta.class_head
    """
    meta_default = str(meta_class_head or "").strip() or None
    if arg_heads:
        return [h.strip() for h in arg_heads.split(",") if h.strip()]
    if arg_class_head:
        return [arg_class_head]
    if schema_head_ids:
        heads = list(schema_head_ids)
        if not heads and meta_default:
            return [meta_default]
        return heads
    if meta_default:
        return [meta_default]
    return []


def test_explicit_heads_win() -> None:
    assert resolve_heads("a, b", "c", ["d"], "e") == ["a", "b"]


def test_class_head_beats_schema() -> None:
    assert resolve_heads(None, "c", ["d"], "e") == ["c"]


def test_schema_beats_meta() -> None:
    assert resolve_heads(None, None, ["d1", "d2"], "e") == ["d1", "d2"]


def test_meta_is_the_last_resort() -> None:
    assert resolve_heads(None, None, [], "e") == ["e"]


def test_nothing_resolves_to_empty_not_a_guessed_name() -> None:
    """手掛かりが 1 つも無ければ空。head 名を作り出さない。"""
    assert resolve_heads(None, None, [], None) == []


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_meta_is_treated_as_absent(blank: str | None) -> None:
    assert resolve_heads(None, None, [], blank) == []
