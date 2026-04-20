import math

from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .models import AnalysisRun, ImageResult, MicroscopyImage


def upload(request):
    if request.method == "POST":
        action = request.POST.get("action")

        if action == "upload":
            files = request.FILES.getlist("files")
            treatment_group = request.POST.get("treatment_group", "")
            for f in files:
                MicroscopyImage.objects.create(
                    original_filename=f.name,
                    file=f,
                    treatment_group=treatment_group,
                )
            messages.success(request, f"Uploaded {len(files)} image(s).")
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
    return render(request, "synapse_web/upload.html", {"images": images})


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
        # Group by treatment_group using plain Python
        groups = {}
        for r in result_rows:
            key = r.image.treatment_group or "(none)"
            groups.setdefault(key, []).append(r.puncta_density or 0.0)

        for group_name, densities in sorted(groups.items()):
            n = len(densities)
            mean = sum(densities) / n if n else 0.0
            variance = (
                sum((d - mean) ** 2 for d in densities) / n if n > 1 else 0.0
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
            }
        )

    return render(
        request,
        "synapse_web/results.html",
        {
            "group_stats": group_stats,
            "run_summaries": run_summaries,
            "latest_run": latest_run,
        },
    )
