"""In-process background job runner.

Single-server, single-process design. A module-level ThreadPoolExecutor
with max_workers=1 serializes ML jobs (one at a time, matching GPU memory
and SQLite's single-writer reality).

Limits worth knowing:
- Multiple gunicorn workers would each get their own pool — run a single
  web worker, or replace this with a real queue (Celery / RQ / Django-Q)
  if you scale out.
- Jobs are lost on process restart. Use the recover_stale_runs management
  command to mark orphaned "running" rows as "failed" after a deploy.

The work function itself is pluggable via settings.ANALYSIS_WORK_FN
(import path). The contract is in synapse_web/services/work_stub.py.
"""
from __future__ import annotations

import logging
import os
import subprocess
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone
from django.utils.module_loading import import_string

from synapse_web.models import AnalysisRun

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")


class ProgressReporter:
    """Writes progress updates to the AnalysisRun row.

    Throttles by suppressing no-op writes — call .update() as often as
    you like; only meaningful changes hit the DB.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self._last_progress: Optional[int] = None
        self._last_message: Optional[str] = None

    def update(
        self,
        progress: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        fields: dict = {}
        if progress is not None:
            clamped = max(0, min(100, int(progress)))
            if clamped != self._last_progress:
                fields["progress"] = clamped
                self._last_progress = clamped
        if message is not None and message != self._last_message:
            fields["progress_message"] = message[:255]
            self._last_message = message
        if fields:
            AnalysisRun.objects.filter(id=self.run_id).update(**fields)


def current_commit_sha() -> str:
    """Best-effort git HEAD SHA, empty string when unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(settings.BASE_DIR),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def submit_run(run_id: str) -> Optional[Future]:
    """Submit an AnalysisRun for execution.

    Honors settings.ANALYSIS_JOBS_SYNC = True for tests and management
    commands — runs in-thread and returns None instead of a Future.
    """
    if settings.ANALYSIS_JOBS_SYNC:
        _execute(str(run_id))
        return None
    return _EXECUTOR.submit(_execute, str(run_id))


def _execute(run_id: str) -> None:
    close_old_connections()
    try:
        run = AnalysisRun.objects.get(id=run_id)
        AnalysisRun.objects.filter(id=run_id).update(
            status="running",
            started_at=timezone.now(),
            progress=0,
            progress_message="Starting",
            error_message="",
            error_traceback="",
        )

        work_fn = import_string(settings.ANALYSIS_WORK_FN)
        output_dir = os.path.join(
            str(settings.MEDIA_ROOT), "results", str(run.id)
        )
        os.makedirs(output_dir, exist_ok=True)

        work_fn(
            run_id=str(run.id),
            inputs=run.input_manifest or [],
            checkpoint_path=run.checkpoint_path,
            output_dir=output_dir,
            config=run.config_snapshot or {},
            reporter=ProgressReporter(str(run.id)),
        )

        AnalysisRun.objects.filter(id=run_id).update(
            status="completed",
            finished_at=timezone.now(),
            progress=100,
            progress_message="Done",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis run %s failed", run_id)
        AnalysisRun.objects.filter(id=run_id).update(
            status="failed",
            finished_at=timezone.now(),
            error_message=str(exc)[:500] or exc.__class__.__name__,
            error_traceback=traceback.format_exc(),
        )
    finally:
        close_old_connections()
