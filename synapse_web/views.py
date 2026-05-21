from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.cache import never_cache

from .models import AnalysisRun
from .services.naming import GROUP_TREATMENTS, KNOWN_GROUPS
from .services.training_config import (
    PREPROCESSING_DEFAULTS,
    PRETRAIN_CONFIG_PATH,
    grouped_pretrain_config,
)


def run_detail(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    source_stats = list(run.source_stats.all())
    return render(
        request,
        "synapse_web/run_detail.html",
        {
            "run": run,
            "source_stats": source_stats,
            "expected_count": run.expected_image_count,
            "thesis_pdf_url": settings.THESIS_PDF_URL,
        },
    )


@never_cache
def run_status(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    return JsonResponse(
        {
            "status": run.status,
            "progress": run.progress,
            "progress_message": run.progress_message,
            "result_count": run.source_stats.count(),
            "expected_count": run.expected_image_count,
            "finished": run.status in ("completed", "failed"),
            "error_message": run.error_message,
        }
    )


def results(request):
    latest_run = AnalysisRun.objects.filter(status="completed").first()

    run_summaries = []
    for run in AnalysisRun.objects.all()[:20]:
        run_summaries.append(
            {
                "id": run.id,
                "id_short": str(run.id)[:8],
                "created_at": run.created_at,
                "status": run.status,
                "run_kind": run.get_run_kind_display(),
                "n_source_images": run.n_source_images,
                "n_patches": run.n_patches,
                "n_clusters": run.n_clusters,
                "expected_count": run.expected_image_count,
                "checkpoint_path": run.checkpoint_path or "—",
            }
        )

    return render(
        request,
        "synapse_web/results.html",
        {
            "run_summaries": run_summaries,
            "latest_run": latest_run,
        },
    )


def overview(request):
    config_sections = grouped_pretrain_config()
    treatments = [
        {"token": tok, "treatment": GROUP_TREATMENTS[tok]}
        for tok in KNOWN_GROUPS
    ]
    return render(
        request,
        "synapse_web/overview.html",
        {
            "config_sections": config_sections,
            "config_available": bool(config_sections),
            "config_filename": PRETRAIN_CONFIG_PATH.name,
            "preprocessing": PREPROCESSING_DEFAULTS,
            "treatments": treatments,
            "thesis_pdf_url": settings.THESIS_PDF_URL,
        },
    )
