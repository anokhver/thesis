"""Filesystem layout + cleanup for ``AnalysisRun`` artifacts.

Every run owns ``MEDIA_ROOT/runs/<run_id>/`` and three subdirs::

    runs/<run_id>/input/    user upload extracted here (patches.zip or bundle.zip)
    runs/<run_id>/bundle/   embeddings.npy + metadata.csv (post-extract,
                            or copied straight from input for cluster_only)
    runs/<run_id>/output/   run_clustering output: labels.npy, patch_labels.csv,
                            Z2.npy, summary.json, plots/*.png

When an ``AnalysisRun`` row is deleted, the corresponding directory is
removed too. The signal handler does a basic path-traversal check so a
maliciously crafted ``input_dir`` cannot delete files outside the runs
root.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from django.conf import settings
from django.db.models.signals import pre_delete
from django.dispatch import receiver

from ._paths import is_within


def runs_root() -> Path:
    return Path(settings.RUNS_DIR)


def run_root(run) -> Path:
    return runs_root() / str(run.id)


def run_subdir(run, name: str) -> Path:
    return run_root(run) / name


def ensure_run_dirs(run) -> tuple[Path, Path, Path]:
    root = run_root(run)
    input_dir = root / "input"
    bundle_dir = root / "bundle"
    output_dir = root / "output"
    for d in (input_dir, bundle_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)
    return input_dir, bundle_dir, output_dir


def delete_run_artifacts(run) -> bool:
    root = run_root(run)
    if not root.exists():
        return False
    if not is_within(root, runs_root()):
        return False
    shutil.rmtree(root, ignore_errors=True)
    return True


@receiver(pre_delete, sender="synapse_web.AnalysisRun")
def _cleanup_on_delete(sender, instance, **kwargs):
    delete_run_artifacts(instance)
