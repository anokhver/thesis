import math
import uuid

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import default_storage
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .models import AnalysisRun, ImageResult, MicroscopyImage
from .services.naming import KNOWN_GROUPS, derive_treatment_group


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

    images = MicroscopyImage.objects.filter(id__in=image_ids)
    if not images.exists():
        messages.error(request, "Selected images not found.")
        return redirect("synapse_web:upload")

    # TODO: replace with actual pipeline call
    run = AnalysisRun.objects.create(status="completed")
    for img in images:
        ImageResult.objects.create(
            analysis_run=run,
            image=img,
            puncta_count=0,
            puncta_density=0.0,
        )

    messages.success(request, f"Analysis complete for {images.count()} image(s).")
    return redirect("synapse_web:results")


def results(request):
    latest_run = (
        AnalysisRun.objects.filter(status="completed")
        .order_by("-created_at")
        .first()
    )

    group_stats = []
    if latest_run:
        result_rows = latest_run.results.select_related("image").all()
        groups = {}
        for r in result_rows:
            key = r.image.treatment_group or "(none)"
            groups.setdefault(key, []).append(r.puncta_density or 0.0)

        for group_name, densities in sorted(groups.items()):
            n = len(densities)
            mean = sum(densities) / n if n else 0.0
            variance = (
                sum((d - mean) ** 2 for d in densities) / (n - 1) if n > 1 else 0.0
            )
            std = math.sqrt(variance)
            group_stats.append(
                {
                    "treatment_group": group_name,
                    "n": n,
                    "mean_density": round(mean, 4),
                    "std_density": round(std, 4),
                }
            )

    all_runs = AnalysisRun.objects.order_by("-created_at")[:20]
    run_summaries = []
    for run in all_runs:
        run_summaries.append(
            {
                "id_short": str(run.id)[:8],
                "created_at": run.created_at,
                "status": run.status,
                "image_count": run.results.count(),
                "checkpoint_path": run.checkpoint_path or "—",
            }
        )

    image_results = []
    if latest_run:
        for r in latest_run.results.select_related("image").all():
            image_results.append(r)

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
