"""Placeholder ML pipeline.

Real extract+cluster runs land in Phase E (ml_jobs.do_work). Until then
the runner falls back to this stub so the schema, run lifecycle and
progress UI can be exercised without torch.

The contract that the real `do_work` must respect:

    def do_work(
        *,
        run_id: str,
        inputs: list[dict],          # one per source image; see below
        checkpoint_path: str,        # may be empty for cluster_only
        output_dir: str,             # absolute path, already created
        config: dict,                # AnalysisRun.config_snapshot
        reporter,                    # services.jobs.ProgressReporter
    ) -> None:
        ...

`inputs` items shape (post-Phase-B):
    {
        "source_image": "1_3_PSYHARMIN.vsi",
        "treatment_group": "PSYHARMIN",
        "n_patches": 42,
    }

Must:
  * Write artifacts under ``output_dir`` (clustering produces ``labels.npy``,
    ``patch_labels.csv``, ``Z2.npy``, ``summary.json``, ``plots/*.png``).
  * Populate ``SourceImageStats`` keyed by (analysis_run, source_image).
    Use ``update_or_create`` — the UniqueConstraint guarantees idempotency
    so reruns/resumes don't blow up.
  * Raise on unrecoverable errors; the runner catches and records.
  * Call reporter.update(progress=..., message=...) often enough that the
    UI feels alive; the reporter throttles internally.
"""
from __future__ import annotations

import time

from synapse_web.models import SourceImageStats


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
            message=f"Processing {item['source_image']} ({i + 1}/{total})",
        )
        time.sleep(0.05)

        SourceImageStats.objects.update_or_create(
            analysis_run_id=run_id,
            source_image=item["source_image"],
            defaults={
                "treatment_group": item.get("treatment_group", ""),
                "n_patches": int(item.get("n_patches", 0)),
                "cluster_counts": {},
                "dominant_cluster": None,
            },
        )

    reporter.update(progress=100, message="Finalizing")
