"""`GET /workspaces` と sidebar-order の応答を固定する。

なぜ書くか（#225）:
    どちらも `app/workspace.py` の中にあり、`create_workspace_router()` の
    閉じ込みに依存している。#225 でこの関数を解体するため、その前に
    「今どう答えているか」を残しておく。

    並べ替えの helper（`app/utils/sidebar_order.py`）には
    `utils/test_sidebar_order.py` があるが、**endpoint は通っていなかった**。
    ここで埋めるのは endpoint 側の入出力と、id の正規化・破棄の規則。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigManager
from app.experiments import create_experiments_router
from app.workspace import create_workspace_router

WS_A = "wslist01"
WS_B = "wslist02"


def _make_client(tmp_path: Path) -> TestClient:
    config_manager = ConfigManager(tmp_path)
    app = FastAPI()
    app.include_router(create_workspace_router(config_manager))
    app.include_router(create_experiments_router(config_manager))
    return TestClient(app)


def _list(client: TestClient, *, order: str | None = None) -> list[dict]:
    res = client.get("/workspaces", params={"order": order} if order else None)
    assert res.status_code == 200, res.text
    return res.json()["workspaces"]


def test_empty_root(tmp_path: Path) -> None:
    assert _list(_make_client(tmp_path)) == []


def test_entry_shape(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    root = tmp_path / WS_A
    (root / "source_images").mkdir(parents=True)
    (root / "source_images" / "a.jpg").write_bytes(b"x" * 5)

    entry = _list(client)[0]
    assert set(entry) == {
        "id",
        "display_name",
        "workspace_root",
        "experiments_count",
        "source_images_count",
        "size_bytes",
        "created_at",
        "updated_at",
    }
    assert entry["id"] == WS_A
    assert entry["display_name"] == WS_A
    assert entry["workspace_root"] == str(root.resolve())
    assert entry["experiments_count"] == 0
    assert entry["source_images_count"] == 1
    assert entry["size_bytes"] >= 5


def test_counts_experiments_and_source_images(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    root = tmp_path / WS_A
    root.mkdir(parents=True)
    for exp in ("expaaaa1", "expbbbb2"):
        created = client.post(
            f"/workspaces/{WS_A}/experiments",
            json={"display_name": exp, "id": exp},
        )
        assert created.status_code == 200, created.text
    # short id でないディレクトリは experiment として数えない。
    (root / "experiments" / "NotAnExperiment").mkdir(parents=True, exist_ok=True)

    source = root / "source_images"
    (source / "nested").mkdir(parents=True)
    (source / "a.jpg").write_bytes(b"x")
    (source / "nested" / "b.jpg").write_bytes(b"x")

    entry = _list(client)[0]
    assert entry["experiments_count"] == 2
    # 数え上げは再帰的。
    assert entry["source_images_count"] == 2


def test_non_workspace_directories_are_ignored(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    (tmp_path / WS_A).mkdir(parents=True)
    for name in ("TooLongWorkspaceName", "short", ".trash", "has-dash"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")

    assert [w["id"] for w in _list(client)] == [WS_A]


def test_default_order_is_by_directory_name(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    (tmp_path / WS_B).mkdir(parents=True)
    (tmp_path / WS_A).mkdir(parents=True)

    assert [w["id"] for w in _list(client)] == [WS_A, WS_B]


def test_order_sidebar_applies_saved_order(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    (tmp_path / WS_A).mkdir(parents=True)
    (tmp_path / WS_B).mkdir(parents=True)

    saved = client.put("/workspaces/sidebar-order", json={"workspaces": [WS_B, WS_A]})
    assert saved.status_code == 200, saved.text

    # order を指定しなければ保存した並びは効かない。
    assert [w["id"] for w in _list(client)] == [WS_A, WS_B]
    assert [w["id"] for w in _list(client, order="sidebar")] == [WS_B, WS_A]


def test_order_sidebar_appends_workspaces_missing_from_the_order(
    tmp_path: Path,
) -> None:
    """並び順に載っていない workspace も必ず出る。

    並び順は別ファイルなので、workspace を新しく作ると簡単にずれる。
    ずれたときに**消えるのではなく後ろに付く**ことを保証する。
    """
    client = _make_client(tmp_path)
    (tmp_path / WS_A).mkdir(parents=True)
    (tmp_path / WS_B).mkdir(parents=True)

    saved = client.put("/workspaces/sidebar-order", json={"workspaces": [WS_B]})
    assert saved.status_code == 200, saved.text

    assert [w["id"] for w in _list(client, order="sidebar")] == [WS_B, WS_A]


def test_sidebar_order_initial_value(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    body = client.get("/workspaces/sidebar-order").json()
    assert body == {
        "version": 1,
        "workspaces": [],
        "experiments": {},
        "updated_at": None,
    }


def test_sidebar_order_normalizes_and_drops_invalid_ids(tmp_path: Path) -> None:
    """保存時に id をそろえ、使えないものは落とす。

    - 前後の空白を落として小文字にする
    - 重複は先に出たほうを残す
    - `[a-z0-9]{8}` に合わないものは黙って捨てる
    """
    client = _make_client(tmp_path)
    res = client.put(
        "/workspaces/sidebar-order",
        json={"workspaces": ["  WSLIST01 ", "wslist01", "bad", "", WS_B]},
    )
    assert res.status_code == 200, res.text
    assert res.json()["workspaces"] == [WS_A, WS_B]


def test_sidebar_order_normalizes_experiment_map(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    res = client.put(
        "/workspaces/sidebar-order",
        json={
            "experiments": {
                "  WSLIST01 ": ["EXPAAAA1", "expaaaa1", "nope"],
                "bad": ["expaaaa1"],
            }
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()["experiments"] == {WS_A: ["expaaaa1"]}


def test_sidebar_order_updates_only_the_given_side(tmp_path: Path) -> None:
    """省略した側は現状のまま残す。

    サイドバーは workspace の並びと experiment の並びを別々に保存するため、
    片方の PUT でもう片方が消えると並びが失われる。
    """
    client = _make_client(tmp_path)
    client.put(
        "/workspaces/sidebar-order",
        json={"workspaces": [WS_A], "experiments": {WS_A: ["expaaaa1"]}},
    )

    res = client.put("/workspaces/sidebar-order", json={"workspaces": [WS_B]})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["workspaces"] == [WS_B]
    assert body["experiments"] == {WS_A: ["expaaaa1"]}

    # 明示的な空配列は「空にする」を意味する（省略と区別する）。
    res = client.put("/workspaces/sidebar-order", json={"workspaces": []})
    assert res.json()["workspaces"] == []
    assert res.json()["experiments"] == {WS_A: ["expaaaa1"]}


def test_sidebar_order_rejects_wrong_types(tmp_path: Path) -> None:
    """型が違う入力は endpoint に届く前に 422 になる。

    `SidebarOrderBody` が `list[str]` / `dict[str, list[str]]` を宣言しているため。
    """
    client = _make_client(tmp_path)
    assert client.put("/workspaces/sidebar-order", json={"workspaces": [5]}).status_code == 422
    assert (
        client.put(
            "/workspaces/sidebar-order", json={"experiments": {WS_A: "notalist"}}
        ).status_code
        == 422
    )


def test_sidebar_order_is_persisted(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    client.put("/workspaces/sidebar-order", json={"workspaces": [WS_B, WS_A]})

    body = client.get("/workspaces/sidebar-order").json()
    assert body["workspaces"] == [WS_B, WS_A]
    assert body["updated_at"] is not None
    assert (tmp_path / ".xima" / "sidebar_order.json").exists()
