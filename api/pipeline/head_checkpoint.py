"""学習済み head checkpoint の state_dict を、推論側の `nn.Linear` に載る形へ正す。

`train_epoch.py` は `LinearHead`（内側に `fc` を持つ）で保存するため、キーが
`fc.weight` / `fc.bias` になる。一方、推論側は素の `torch.nn.Linear` を組み立てるので
`weight` / `bias` を要求する。この差を吸収する。

torch に依存しない（辞書のキーを詰め替えるだけ）。テンソルは中身を見ずに素通しする。
そのため torch の無い環境（CI）でも単体テストできる。

同じ処理が `infer_heads.py:366-380` にもある。**本モジュールへの寄せは行っていない**
——あちらは動作実績のある経路であり、T2-2 の変更に巻き込むと切り分けが難しくなるため。
寄せるなら別途行う。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


def normalize_state_dict(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """`fc.weight` / `fc.bias` 形式を `weight` / `bias` へ詰め替える。

    既に `weight` を持っていればそのまま返す。`weight` に相当するキーが無ければ None。
    """
    if not isinstance(state, Mapping) or not state:
        return None
    if "weight" in state:
        return dict(state)

    w_key = next((k for k in state if str(k).endswith("fc.weight")), None)
    if w_key is None:
        return None
    b_key = next((k for k in state if str(k).endswith("fc.bias")), None)

    out: Dict[str, Any] = {"weight": state[w_key]}
    if b_key is not None:
        out["bias"] = state[b_key]
    return out


def linear_shape(state: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """`(num_classes, in_dim)` を weight の shape から読む。読めなければ None。"""
    weight = state.get("weight") if isinstance(state, Mapping) else None
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[0]), int(shape[1])
