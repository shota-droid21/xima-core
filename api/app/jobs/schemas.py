from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


JOB_TYPES = {
    "apply_label",
    "augment_gray",
    "make_label_list",
    "purge_deleted_images",
    "embed_images",
    # Backward-compatible names
    "train",
    "infer_scores",
    # Preferred explicit names
    "train_epoch",
    "infer_heads",
    # 学習済み head の予測を labels.json に候補として記録する（T2-2）
    "predict_labels",
}


class JobCreateRequest(BaseModel):
    type: str = Field(
        ...,
        pattern=(
            "^(apply_label|augment_gray|make_label_list|purge_deleted_images|embed_images|"
            "train|infer_scores|train_epoch|infer_heads|predict_labels)$"
        ),
    )
    args: Dict[str, Any] = Field(default_factory=dict)


class JobDeleteRequest(BaseModel):
    job_ids: List[str] = Field(..., min_length=1)


@dataclass
class JobRecord:
    id: str
    type: str
    args: Dict[str, Any]
    status: str
    created_at: float
    command: List[str]
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    workspace: Optional[str] = None
    experiment: Optional[str] = None
    worker_task_id: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
