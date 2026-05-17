"""Placeholder ML pipeline.

Real inference lives in the ML package that will be merged later. The
contract that the future real `do_work` must respect:

    def do_work(
        *,
        run_id: str,
        inputs: list[dict],          # one dict per image; see below
        checkpoint_path: str,        # may be empty
        output_dir: str,             # absolute path, already created
        config: dict,                # AnalysisRun.config_snapshot
        reporter,                    # services.jobs.ProgressReporter
    ) -> None:
        ...

`inputs` items shape:
    {
        "image_id": "<uuid>",
        "original_filename": "...",
        "treatment_group": "...",
        "path": "uploads/microscopy/.../foo.vsi"   # relative to MEDIA_ROOT
    }

Must:
  * Persist outputs (MIP/mask/overlay PNGs) under `output_dir` and
    register them on the matching ImageResult.
  * Use update_or_create keyed by (analysis_run, image) so reruns/resumes
    are idempotent — there's a UniqueConstraint on the model.
  * Raise on unrecoverable errors; the runner catches and records.
  * Call reporter.update(progress=..., message=...) often enough that the
    UI feels alive; do NOT hammer (the reporter throttles regardless).
"""
from __future__ import annotations

import time

from synapse_web.models import ImageResult


def do_work(
    *,
    run_id: str,
    inputs: list[dict],
    checkpoint_path: str,
    output_dir: str,
    config: dict,
    reporter,
) -> None:
    total = max(len(inputs), 1)
    for i, item in enumerate(inputs):
        reporter.update(
            progress=int(100 * i / total),
            message=f"Processing {item['original_filename']} ({i + 1}/{total})",
        )
        time.sleep(0.05)

        ImageResult.objects.update_or_create(
            analysis_run_id=run_id,
            image_id=item["image_id"],
            defaults={
                "puncta_count": 0,
                "puncta_density": 0.0,
            },
        )

    reporter.update(progress=100, message="Finalizing")
