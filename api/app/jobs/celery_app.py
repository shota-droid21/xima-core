from __future__ import annotations

import os

from celery import Celery

BROKER_URL = os.environ.get("XIMA_CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.environ.get("XIMA_CELERY_RESULT_BACKEND", "")
DEFAULT_QUEUE = os.environ.get("XIMA_CELERY_QUEUE", "xima_jobs")

celery_app = Celery(
    "xima_agent_jobs",
    broker=BROKER_URL,
    backend=RESULT_BACKEND or None,
    include=["app.jobs.tasks"],
)

celery_app.conf.update(
    task_default_queue=DEFAULT_QUEUE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    worker_prefetch_multiplier=1,
)
