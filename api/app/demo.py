"""デモ workspace / experiment の作成。

初見のユーザが自分の画像を用意しなくても、起動直後に
「ラベリング → 学習 → 精度確認」を体感できるようにするための導線。

ランチャー（CLI）と UI の両方から同じエンドポイントを叩けるようにし、
生成ロジックを 1 箇所に保つ。

workspace / experiment の生成は、通常の作成 API と同じプリミティブ
（`WorkspaceMeta` / `ExperimentMeta` / `generate_short_id`）を用いる。
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from .config import ConfigManager
from .utils.meta import ExperimentMeta, WorkspaceMeta
from .utils.short_id import generate_short_id

# pipeline はスクリプト実行時と同じくトップレベル名で解決する。
_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from demo_images import (  # noqa: E402
    DEFAULT_IMAGES_PER_CLASS,
    DEMO_CLASSES,
    build_demo_label_schema,
    generate_demo_images,
)

DEMO_WORKSPACE_DISPLAY_NAME = "Demo Workspace"
DEMO_EXPERIMENT_DISPLAY_NAME = "Demo Experiment"


def _new_id(root: Path) -> str:
    """既存と衝突しない short id を採番する。"""
    for _ in range(50):
        candidate = generate_short_id()
        if candidate == ".trash":
            continue
        if not (root / candidate).exists():
            return candidate
    raise HTTPException(status_code=500, detail="failed to generate unique id")


def create_demo(
    config_manager: ConfigManager,
    *,
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
) -> Dict[str, Any]:
    """デモ workspace / experiment を作り、サンプル画像と schema を配置する。

    画像の取り込み（make_label_list）は行わない。呼び出し側が jobs 経由で実行する。
    """
    cfg = config_manager.get_config()
    ws_root = cfg.workspaces_root
    ws_root.mkdir(parents=True, exist_ok=True)

    ws_id = _new_id(ws_root)
    workspace_dir = ws_root / ws_id
    source_dir = workspace_dir / cfg.source_dir
    experiments_dir = workspace_dir / cfg.experiments_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    WorkspaceMeta.create(
        workspace_dir,
        workspace_id=ws_id,
        display_name=DEMO_WORKSPACE_DISPLAY_NAME,
    )

    exp_id = _new_id(experiments_dir)
    experiment_dir = experiments_dir / exp_id
    label_input_dir = experiment_dir / "label_input"
    (label_input_dir / "history").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config").mkdir(parents=True, exist_ok=True)
    ExperimentMeta.create(
        experiment_dir,
        experiment_id=exp_id,
        display_name=DEMO_EXPERIMENT_DISPLAY_NAME,
    )

    counts = generate_demo_images(source_dir, images_per_class=images_per_class)

    # 生成した 3 クラスをそのまま選べる schema を置き、追加設定なしで学習まで通す。
    (label_input_dir / "label_schema.json").write_text(
        json.dumps(build_demo_label_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "workspace": ws_id,
        "experiment": exp_id,
        "classes": list(DEMO_CLASSES),
        "images": sum(counts.values()),
        "images_per_class": images_per_class,
        "source_dir": str(source_dir),
    }


def _wait_for_import(
    jobs_manager: Any, job_id: str | None, *, timeout_sec: float = 60.0
) -> bool:
    """make_label_list の完了を待つ。

    worker が居ない/遅い場合に無限に待たないよう上限を設け、
    待ちきれなかった場合は False を返して呼び出し側に判断を委ねる
    （デモ自体は作成済みなので、後からジョブが終われば利用できる）。
    """
    if not job_id:
        return False

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            job = jobs_manager.get_job(job_id)
        except Exception:  # noqa: BLE001
            return False
        status = str(getattr(job, "status", "") or "").lower()
        if status == "done":
            return True
        if status in ("error", "canceled"):
            return False
        time.sleep(0.3)
    return False


def create_demo_router(config_manager: ConfigManager, jobs_manager: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/demo")
    def create_demo_scoped(images_per_class: int = DEFAULT_IMAGES_PER_CLASS) -> dict:
        # 極端な枚数指定でディスクを埋めないよう上限を設ける。
        if images_per_class < 1 or images_per_class > 200:
            raise HTTPException(
                status_code=400, detail="images_per_class must be between 1 and 200"
            )

        result = create_demo(config_manager, images_per_class=images_per_class)

        # 画像の取り込みは通常の workflow と同じく make_label_list ジョブで行う。
        job = jobs_manager.create_job(
            "make_label_list",
            {},
            workspace=result["workspace"],
            experiment=result["experiment"],
        )
        job_id = getattr(job, "id", None)
        result["job_id"] = job_id

        # 取り込み完了を待ってから返す。
        # 完了前に UI が experiment を開くと「0 件」を掴んでキャッシュしてしまい、
        # デモの目的（開いてすぐ体感できる）が崩れるため。
        result["import_completed"] = _wait_for_import(jobs_manager, job_id)
        return result

    return router
