import math
import uuid

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST

from .models import AnalysisRun, ImageResult, MicroscopyImage
from .services import jobs
from .services.naming import GROUP_TREATMENTS, KNOWN_GROUPS, derive_treatment_group
from .services.training_config import (
    PREPROCESSING_DEFAULTS,
    PRETRAIN_CONFIG_PATH,
    grouped_pretrain_config,
)


def _parse_relative_paths(raw: str, files: list) -> list[str]:
    """Return a relative path for each uploaded file, preferring the
    browser-supplied webkitRelativePath (one per line) and falling back to
    the file's own name when JS didn't run or the browser doesn't support it.
    """
    candidates = [p.strip() for p in raw.splitlines()]
    candidates = [p for p in candidates if p]
    if len(candidates) == len(files):
        return [p.replace("\\", "/") for p in candidates]
    return [f.name for f in files]


def _group_stats_from_results(result_rows) -> list[dict]:
    groups: dict[str, list[float]] = {}
    for r in result_rows:
        key = r.image.treatment_group or "(none)"
        groups.setdefault(key, []).append(r.puncta_density or 0.0)

    stats = []
    for group_name, densities in sorted(groups.items()):
        n = len(densities)
        mean = sum(densities) / n if n else 0.0
        variance = (
            sum((d - mean) ** 2 for d in densities) / (n - 1) if n > 1 else 0.0
        )
        stats.append(
            {
                "treatment_group": group_name,
                "n": n,
                "mean_density": round(mean, 4),
                "std_density": round(math.sqrt(variance), 4),
            }
        )
    return stats


def upload(request):
    if request.method == "POST":
        action = request.POST.get("action")

        if action == "upload":
            files = request.FILES.getlist("files")
            rel_paths = _parse_relative_paths(
                request.POST.get("relative_paths", ""), files
            )
            fallback_group = request.POST.get("treatment_group", "").strip()
            batch_id = uuid.uuid4().hex[:8]
            base_dir = f"uploads/microscopy/{batch_id}"

            vsi_count = 0
            derived_count = 0
            for f, rel_path in zip(files, rel_paths):
                target = f"{base_dir}/{rel_path}"
                saved_path = default_storage.save(target, f)
                if not rel_path.lower().endswith(".vsi"):
                    continue
                basename = rel_path.rsplit("/", 1)[-1]
                derived = derive_treatment_group(basename)
                if derived:
                    derived_count += 1
                img = MicroscopyImage(
                    original_filename=basename,
                    treatment_group=derived or fallback_group,
                )
                img.file.name = saved_path
                img.save()
                vsi_count += 1

            if vsi_count == 0:
                messages.warning(
                    request,
                    "No .vsi files found in the upload. "
                    "Pick a folder that contains the .vsi acquisitions.",
                )
            else:
                messages.success(
                    request,
                    f"Registered {vsi_count} acquisition(s) "
                    f"from {len(files)} uploaded file(s); "
                    f"{derived_count} group(s) auto-derived from filename.",
                )
            return redirect("synapse_web:upload")

        if action == "delete":
            image_id = request.POST.get("image_id")
            try:
                img = MicroscopyImage.objects.get(id=image_id)
                img.file.delete(save=False)
                img.delete()
                messages.success(request, "Image deleted.")
            except MicroscopyImage.DoesNotExist:
                messages.error(request, "Image not found.")
            return redirect("synapse_web:upload")

    images = MicroscopyImage.objects.all().order_by("-uploaded_at")
    return render(
        request,
        "synapse_web/upload.html",
        {
            "images": images,
            "known_groups": ", ".join(KNOWN_GROUPS),
            "thesis_pdf_url": settings.THESIS_PDF_URL,
        },
    )


@require_POST
def run_inference(request):
    image_ids = request.POST.getlist("image_ids")
    if not image_ids:
        messages.warning(request, "No images selected.")
        return redirect("synapse_web:upload")

    images = list(MicroscopyImage.objects.filter(id__in=image_ids))
    found_ids = {str(img.id) for img in images}
    missing = [i for i in image_ids if i not in found_ids]
    if missing:
        messages.error(
            request,
            f"{len(missing)} selected image(s) no longer exist; refresh and try again.",
        )
        return redirect("synapse_web:upload")

    checkpoint_path = request.POST.get("checkpoint_path", "").strip()

    manifest = [
        {
            "image_id": str(img.id),
            "original_filename": img.original_filename,
            "treatment_group": img.treatment_group,
            "path": img.file.name,
        }
        for img in images
    ]
    config_snapshot = {
        "checkpoint_path": checkpoint_path,
        "image_count": len(manifest),
        "work_fn": settings.ANALYSIS_WORK_FN,
    }

    run = AnalysisRun.objects.create(
        status="pending",
        checkpoint_path=checkpoint_path,
        config_snapshot=config_snapshot,
        image_manifest=manifest,
        commit_sha=jobs.current_commit_sha(),
    )

    jobs.submit_run(str(run.id))
    messages.success(
        request,
        f"Analysis queued for {len(manifest)} image(s). "
        "This page polls for progress.",
    )
    return redirect("synapse_web:run_detail", run_id=run.id)


def run_detail(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    results = list(run.results.select_related("image").all())
    return render(
        request,
        "synapse_web/run_detail.html",
        {
            "run": run,
            "results": results,
            "group_stats": _group_stats_from_results(results),
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
            "result_count": run.results.count(),
            "expected_count": run.expected_image_count,
            "finished": run.status in ("completed", "failed"),
            "error_message": run.error_message,
        }
    )


def results(request):
    latest_run = AnalysisRun.objects.filter(status="completed").first()

    group_stats = []
    image_results = []
    if latest_run:
        image_results = list(latest_run.results.select_related("image").all())
        group_stats = _group_stats_from_results(image_results)

    run_summaries = []
    for run in AnalysisRun.objects.all()[:20]:
        run_summaries.append(
            {
                "id": run.id,
                "id_short": str(run.id)[:8],
                "created_at": run.created_at,
                "status": run.status,
                "image_count": run.results.count(),
                "expected_count": run.expected_image_count,
                "checkpoint_path": run.checkpoint_path or "—",
            }
        )

    return render(
        request,
        "synapse_web/results.html",
        {
            "group_stats": group_stats,
            "run_summaries": run_summaries,
            "latest_run": latest_run,
            "image_results": image_results,
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

